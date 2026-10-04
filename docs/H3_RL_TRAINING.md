# Online RL for MiniMax H3 LoRAs (character swap)

This fork can fine-tune a MiniMax H3 Ref2VA LoRA with **online reinforcement learning** instead of (or after) the usual
regression loss. Each step the model generates several videos for the same input, scores them with automatic rewards
(identity, background, lighting, pose, lip sync), and learns from the comparison: it is pulled toward the samples that
scored above the group average and pushed away from the ones below. The model never regresses on a target video, so it
does not copy the dataset's pixels or artifacts. It learns what "better" looks like.

It runs on **one GPU**, inside ai-toolkit, with the int8 H3 checkpoint, aligned guide latents (video-to-video editing),
references and cached text embeddings. These are the same inputs the normal H3 training uses.

## Credits

- **Algorithm:** DiffusionNFT, from Zheng et al., *"DiffusionNFT: Online Diffusion Reinforcement with Forward Process"* (2025).
- **Code adapted from [verl-omni](https://github.com/verl-project/verl-omni)** (Apache-2.0). We took the DiffusionNFT loss
  (`DiffusionNFTLoss.compute_loss`), the advantage → reward-probability mapping, the old-policy EMA schedules and the
  hyper-parameters of their MiniMax-H3 Ref2VA recipe. verl-omni itself needs 8 GPUs, Ray, vLLM-Omni, the bf16 H3 and
  FlashAttention 3 (Hopper only).
- **New in this fork:**
  - the single-GPU port;
  - the guide-latent (video editing) setting;
  - the body-swap rewards;
  - the ground-truth-free "source mode".

## How one step works

1. Take one dataset item (guide video + reference picture + caption).
2. **Rollouts:** generate `group` videos (default 6) with the **old** policy, an EMA copy of the LoRA. Each video uses
   different noise and a plain Euler ODE with `steps` sampling steps on the H3 sigma grid.
3. **Rewards:** decode them and score each one in a separate reward process (CPU + YOLO on the GPU).
4. **Advantage:** every reward is z-normalised inside the group, then they are combined by weighted sum, clipped to
   ±`adv_clip_max` and mapped to a reward probability in [0, 1].
5. **Update (DiffusionNFT):** for a random `train_fraction` of the rollout timesteps:
   - re-noise each generated video with fresh noise;
   - run the current LoRA (with grad), the old LoRA and the base model (LoRA off);
   - take the DiffusionNFT loss plus a small KL to the base (`ref_kl_coef`).
6. Every `update_interval` steps: old ← decay·old + (1−decay)·current (`decay_schedule`).

## Rewards

| key | paired item (has a ground-truth target) | source item (no ground truth) |
|---|---|---|
| `id` | ArcFace cosine to the target's face (median over 6 frames) | ArcFace cosine to the **reference picture**, minus a penalty when the face still looks like the original person |
| `char` | DINOv2 cosine between the generated subject crop and the target's subject crop | DINOv2 cosine between the generated subject crop and the **reference picture** (background removed); works for people seen from behind, anime characters and creatures, where ArcFace finds no face |
| `bg` | PSNR outside the dilated person mask vs the target | PSNR outside the union of the original and generated person masks vs the **input clip** |
| `light` | low-frequency Lab error inside the person mask vs the target | face shading layout (16×16 low-frequency luminance, correlation) and brightness vs the **original person's face**, who was lit by the scene itself |
| `pose` | YOLOv8-pose keypoint similarity (OKS) vs the target, frame by frame | same, vs the input clip |
| `lips` | correlation of the mouth-opening curve (68-point landmarks) + amplitude; neutral when nobody speaks | same, vs the input clip |

Optional rewards are only computed when they have a weight (`char` loads DINOv2 on the GPU, ~350 MB).

**Anti-copy gate (source mode).** If the generated face is still the original person (cosine > 0.45), that sample's
`bg`, `light`, `pose` and `lips` drop to their floor. Copying the input would otherwise win three rewards out of four.

Rewards that cannot be measured (no face, no speech) are neutral (0 after normalisation). A weight of 0 switches a reward
off.

## Datasets

Use batch size 1 and **disable sampling**. Rollout videos are saved every 25 steps in `output/<name>/nft_rollouts/`,
showing the target, the best rollout and the worst rollout.

### Paired items (ground truth)

This is the usual H3 body-swap layout:
- targets: the real video of person B;
- `control_path_1`: the guide (person A in the same scene), role `guide`;
- `control_path_2`: a picture of B, role `reference`.

The rewards compare with the target. Only use pairs whose guide is **pixel-aligned** with the target: same scene, same
camera, only the person differs. Otherwise background and pose against the target mean nothing.

### Source items (no ground truth): train on any real video

```
python tools/h3_rl/build_source_dataset.py --videos /my/clips --refs /pictures/of/other/people \
    --caption tools/h3_rl/caption_swap_1ref.txt --out /data/rl_source [--seconds 2.1] [--descriptions desc.json]
python tools/h3_rl/make_person_masks.py /data/rl_source/targets
```

- `targets/` and `guides/` hold the same clip, `refs/` a random other person on grey with neutralised light.
- Add the dataset (targets / guides as `guide` / refs as `reference`) and put a fragment of its path in
  `train.bfs_nft.src_datasets` (e.g. `rl_source`).
- `--descriptions` (a JSON `{clip: "the young woman in a white T-shirt"}`, e.g. written by a VLM) fills `{desc}` in the
  caption. The prompt then names *which* person to replace, which matters when several people are on screen.
- Mix in some wide / full-body / multi-person / talking clips, so every reward has something to measure.

**All targets need person masks** (`<targets>/_person_masks/<name>.npy`, written by `make_person_masks.py`).

## Configuration (`train.bfs_nft`)

The same options are available in the UI: **New Job → (MiniMax H3) → Reinforcement Learning (H3)**.

```yaml
train:
  batch_size: 1
  lr: 1e-4
  disable_sampling: true
  bfs_nft:
    group: 6               # videos per item
    steps: 10              # rollout sampling steps
    train_fraction: 0.3    # share of rollout timesteps trained per step
    mix_beta: 0.1
    adv_clip_max: 5.0
    ref_kl_coef: 1.0e-4
    decay_schedule: delayed_linear_to_0_999
    update_interval: 2
    weights: {id: 1, char: 1, bg: 1, light: 1, pose: 1, lips: 1}
    src_datasets: [rl_source]
    keep_rollouts: 2
    reward_python: ''      # a python with insightface + ultralytics (empty = the trainer's)
    reward_models: {}      # {insightface_root, pose, seg, dino}; defaults ~/.insightface, yolov8m-pose.pt,
                           # yolov8m-seg.pt, facebook/dinov2-base
```

A full example is in `config/examples/h3_rl_diffusion_nft.yaml`. Requirements for the reward process: `insightface`,
`onnxruntime`, `ultralytics`, `opencv-python` (+ `transformers` for the `char` reward). The buffalo_l face models and the YOLO weights download on first use.

**Start from a LoRA that already does the task** (`network.pretrained_lora_path`). RL can only reinforce what the
model sometimes gets right: if none of the 6 rollouts has the right light, there is nothing to pull toward.

## Cost

One step ≈ `group × steps` forwards for the rollouts, plus `group × train_fraction × steps × 3` forwards (one with grad),
plus the reward pass. With 6 / 10 / 0.3 at 512×288 and 39 frames this is **≈ 4 min per step** and ~35 GB on a 96 GB GPU.
The reward pass adds ~1–2 min on the CPU. In our runs, 100–200 steps already made a visible difference.

## Monitoring

- `output/<name>/nft_log.jsonl`: one line per step, with every reward of every rollout, the advantages, the loss and the
  KL. Average the rewards over windows of ~25 steps to see the trend. The rollouts come from the old policy, so they
  show what the model already learned.
- `output/<name>/nft_rollouts/*.mp4`: target | best | worst, every 25 steps.
- The KL should stay small and stable (≈0.2–0.5 in our runs). If it climbs, lower the lr or raise `ref_kl_coef`.

## Lessons from our runs

- **Training from scratch, then RL:** a from-scratch regression checkpoint that only copied the guide gave RL no signal.
  We started from an existing swap LoRA (at its best strength, 0.8, baked into the weights) instead.
- **A VLM as a lighting judge did not work:** Qwen3-VL 8B scored everything 9–10, even darkened or brightened videos, in
  both pointwise and side-by-side form. The face-shading reward is computed instead, and it separates pasted studio
  light from scene light.
- **Keep backward inside the LoRA context:** with gradient checkpointing the forward is recomputed during backward. If
  you leave `with network:` first, the LoRA is switched off for that recompute and checkpointing fails.
- **Do not mix unaligned pairs:** pairs whose guide shows another person in another place give meaningless background
  and pose rewards.

## Related options in this fork

- `train.scene_loss`: for regression training, weights the loss by the person mask and adds a low-frequency (lighting)
  term inside it. Uses the same `_person_masks`.
- `network.network_kwargs.frozen_ranks` / `freeze_full`: keep the first k ranks of an existing LoRA (or its full-weight
  modules) fixed and train only extra ranks on top.

# H3 character-swap RL on RTX 5090

This guide continues a working swap LoRA with the existing single-GPU DiffusionNFT implementation in `toolkit/bfs_nft.py`. Keep the aligned latent video guide and native identity image references. Read `H3_RL_TRAINING.md` for the algorithm and rewards.

## Measured acceptance test

One isolated step passed on RTX 5090 32 GB with PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, transformers 5.5.3 and the repository-pinned diffusers commit. Six rollouts used frozen DMAD at eight Euler steps. The update retained the original ten-point H3 sigma grid, 0.3 training fraction (three timesteps per rollout), frozen v1 rewards, group normalization and DiffusionNFT loss. Backward, optimizer update and checkpoint saving completed. Step time: 433.38 seconds, loss: 19.7131, KL: 0.288789. Sampled GPU memory reached 30012 MiB; this is not an instrumented peak. No claim of quality improvement or multi-step stability follows from this one-step test.

The tested container had an 87.54 GiB RAM limit although `free -h` reported 377 GiB on the host. Always inspect cgroup limits. Its text encoder/LoRA load phase approached the container limit; unloading the text encoder after caching reduced resident memory substantially. No swap was present.

The bundle asks for 49 frames; H3's VAE accepts 17n+5 frames, and this fork trims 49 to 39. Match evaluation duration/timestamps to the actual 39 frames (24 FPS). Dataset paths, model weights and initial LoRA must come from your own bundle.

## Environment

Use a separate environment. Reusing a verified compatible system PyTorch is optional:

```bash
python -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
pip install insightface onnxruntime ultralytics
python -c 'import torch; x=torch.randn(512,512,device="cuda",dtype=torch.bfloat16); y=x@x.T; torch.cuda.synchronize(); print(torch.__version__,torch.isfinite(y).all().item())'
```

Without a compatible PyTorch, install torch and torchvision from the CUDA 12.8 wheel index before the other dependencies. Keep dependency versions from the repository and save `pip freeze` for reproducibility. Check a CUDA operation, not only `torch.cuda.is_available()`.

## Models and offloading

Download the four Comfy-Org/MiniMax-H3 model files listed in the original training guide/bundle, plus Kijai/MiniMax-H3-experimental's `loras/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors`. The tested text encoder was the original int8 ConvRot checkpoint, not the NVFP4 substitute.

The example `config/examples/h3_rl_5090_8step_turbo.yaml` uses:

```yaml
train:
  cache_text_embeddings: true
  unload_text_encoder: true
  bfs_nft:
    steps: 10
    rollout_steps: 8
    rollout_turbo: true
    reward_worker: bfs_reward_worker_v1.py
model:
  low_vram: true
  layer_offloading: true
  layer_offloading_transformer_percent: 0.5
  layer_offloading_text_encoder_percent: 1.0
  model_kwargs:
    nft_turbo_lora_path: /absolute/path/to/DMAD.safetensors
```

`rollout_steps` controls only generation. `steps` still defines the training timestep grid. DMAD is frozen, never merged into int8 base weights, and active only during rollout generation. The ordinary training assistant adapter is temporarily inactive then restored before scoring and training; current/EMA/reference update predictions do not use DMAD. Adapter states are restored on exceptions. The DMAD checkpoint has variable per-layer ranks (2..128), so the loader constructs each frozen module at its exact rank rather than silently truncating it to the first rank.

This is an experimental change to the proven recipe: rollout generation and update predictions use different frozen adapters. The one-step test establishes execution, not theoretical equivalence or quality. Preserve the original configuration as a baseline. Eight steps with a four-step turbo is also a separately evaluated sampling choice.

## Acceptance before a long run

### Optional smaller rollout groups

`train.bfs_nft.group` may be reduced from six to three to reduce per-step work without replacing DiffusionNFT. With eight rollout steps and three training timesteps per rollout, six candidates require 48 rollout predictions plus 54 update predictions and 18 backward calls; three require 24 rollout predictions plus 27 update predictions and nine backward calls. Gradient checkpointing adds recomputation. Total wall time will not necessarily halve because fixed costs remain. Smaller groups provide noisier within-group comparisons and require quality validation; one candidate cannot provide a useful within-group advantage.

In this session the six-candidate experiment reached 27 completed steps, with a saved checkpoint and optimizer at 25. The authorized three-candidate continuation was launched in a separate output directory using copies of that step-25 checkpoint and optimizer, with 1,000 total trainer steps and saving every 25. Two unsaved completed steps were not carried over. The NFT rollout EMA and its call counter are not persisted by the current fork and restart from the resumed LoRA; this is not an exact state restoration. Keep original logs and map the new NFT calls to global trainer steps (new call 1 corresponds to trainer step 26). Do not claim a measured speedup until new steps complete, and do not mix the two group sizes into one reward window.

Start from the s200 initialization, use one paired item, `train.steps: 1`, separate output name, group six and the complete forward/backward/reward cycle. Restrict the dataset with matching targets/guides/refs/masks, not by omitting required files. Do not resume the full experiment from its smoke checkpoint.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 SEED=42 python -u run.py /absolute/path/to/smoke.yaml
```

Confirm checkpoint saving, finite loss/rewards, sensible rollout videos, small KL, VRAM and the container RAM limit. If a command fails, retain its exact error and stop to inspect it. Offloading can trade more CPU transfers for lower VRAM; changing resolution, data or rewards simultaneously prevents an interpretable comparison.

After acceptance, evaluate s200 on held-out inputs in both ordinary and turbo sampling with fixed seeds. Train to the first 25-step checkpoint and evaluate before proceeding. Reusing training examples does not satisfy the held-out protocol. The smoke timing suggests about three hours for 25 steps excluding full caching and evaluation; measure actual sustained throughput.

## Optional remote VLM evaluation

`tools/h3_rl/evaluate_vlm.py` sends sampled guide/generated frames and an identity reference to a local multimodal chat API. `--baseline` compares two outputs, and `--reverse-order` swaps A/B labels. These are auxiliary evaluations, not an implemented VLM training reward. Original v1 rewards remain unchanged.

UnifiedReward requires its specialized weights, not just its prompt on a generic Qwen. In Windows LM Studio, import the GGUF and matching vision `mmproj` in the same publisher/model folder. Verify a real image, load the model, start Developer's API server on port 1234. Qwen 27B may also be tested as an independent judge. Model identifiers must come from `/v1/models`; do not assume a displayed name establishes the underlying checkpoint architecture or parameter count.

UnifiedReward-Edit's official instruction-following rubric emits `score: [editing_success, preservation]`, each 0..25 with higher better. Our adapted three-image prompt is in `tools/h3_rl/prompts/unifiedreward_swap.txt`. It adds the identity reference to the official source/edited-image setup and therefore needs independent calibration. The general A/B evaluator has its own prompt; it is not an exact reproduction of the official reward prompt.

A test on a ground-truth dataset pair returned `[23,21]`, which tests response format, not a trained rollout or checkpoint. In a real rollout comparison, the model falsely claimed a candidate was absent. Do not feed such an evaluator into gradients merely because it can return scores. Use known failures, unchanged-guide controls, identity/hair mistakes, identical pairs and A/B reversal. Inspect raw evidence and reject malformed/out-of-scale scores. Sparse frames cannot establish exact temporal synchronization or lip sync.

## Private Tailscale access

This setup also connects a RunPod training container to an evaluator on your local Windows PC: training stays on RunPod, while LM Studio processes the images locally. Both devices must be signed into the same tailnet, with policy allowing the training node to reach the Windows node on HTTPS/443. Keep the PC awake and LM Studio's API server running. A RunPod public port mapping is not required for this outbound connection.

Join training and Windows machines to the same tailnet. Do not publish the API with Funnel for this setup. On Windows, private HTTPS forwarding to LM Studio can be enabled with:

```powershell
tailscale serve --bg http://127.0.0.1:1234
tailscale serve status
```

Approve HTTPS enablement if requested by Tailscale. A local test of the Windows machine's own Tailscale IP does not prove remote ingress works. Check a remote `/v1/models` request. In this environment, direct TCP/1234 timed out despite a scoped Windows allow rule; Tailscale Serve HTTPS worked.

Containers without `/dev/net/tun` can run `tailscaled --tun=userspace-networking` with a loopback SOCKS5 server. Keep state/socket files private, authenticate through `tailscale up`, and use the proxy for API calls. Userspace mode does not automatically route normal Python HTTP traffic. If DNS is unavailable through that path, retain the HTTPS hostname/SNI but connect to the peer's verified tailnet IP:

For a RunPod container without a running Tailscale daemon, install Tailscale using its official Linux installation instructions, then run:

```bash
mkdir -p /workspace/.tailscale
chmod 700 /workspace/.tailscale
nohup tailscaled --tun=userspace-networking \
  --state=/workspace/.tailscale/tailscaled.state \
  --socket=/workspace/.tailscale/tailscaled.sock \
  --socks5-server=127.0.0.1:1055 \
  > /workspace/.tailscale/tailscaled.log 2>&1 &
tailscale --socket=/workspace/.tailscale/tailscaled.sock up --hostname=h3-rl-training
tailscale --socket=/workspace/.tailscale/tailscaled.sock status
```

Complete the login URL in your browser. Replace the placeholders below with the local PC's full Tailscale DNS name and IPv4 address from `tailscale status`; do not use the RunPod node's address. Verify connectivity before sending images:

```bash
curl --fail --show-error --max-time 30 \
  --proxy socks5h://127.0.0.1:1055 \
  --connect-to YOUR-WINDOWS-NODE.YOUR-TAILNET.ts.net:443:YOUR-WINDOWS-TAILNET-IP:443 \
  https://YOUR-WINDOWS-NODE.YOUR-TAILNET.ts.net/v1/models
```

Use the returned model ID in the evaluator command. Run from the ai-toolkit root with its environment activated:

```bash
python tools/h3_rl/evaluate_vlm.py \
  --base-url https://YOUR-WINDOWS-NODE.YOUR-TAILNET.ts.net/v1 \
  --model YOUR-LOADED-MODEL-ID \
  --proxy socks5h://127.0.0.1:1055 \
  --connect-to YOUR-WINDOWS-NODE.YOUR-TAILNET.ts.net:443:YOUR-WINDOWS-TAILNET-IP:443 \
  --guide /path/to/guide.mp4 --reference /path/to/reference.png \
  --generated /path/to/new-output.mp4 --baseline /path/to/s200-output.mp4 \
  --frames 3 --output /path/to/evaluation.json
```

Match clip start, FPS and duration before sampling frames. A log of models, prompts, input paths, checkpoint, seed, sampling mode and raw response makes evaluation reviewable. The current curl proxy branch is for the tested private endpoint without API authentication; it explicitly rejects `VLM_API_KEY` rather than putting credentials in process arguments. Standard requests transport supports that environment variable. Extend authenticated proxy transport before using it on an authenticated service.

If RunPod cannot reach the endpoint, inspect the daemon log, authentication, peer availability and tailnet policy first. Test `/v1/models` from RunPod, not only from the Windows PC. Container restarts may require restarting the daemon; preserve its private state on persistent storage. Disable Windows forwarding when finished with `tailscale serve --https=443 off`.

## Choosing an additional evaluator

The desired edit replaces face, hair and visible body appearance according to the identity reference while preserving the guide's action, pose, camera and scene. ArcFace covers facial identity only. Body appearance must be judged separately from pose: preserving keypoints does not require preserving the source person's silhouette. Mark body criteria uncertain when a face-only reference cannot establish the intended proportions. Reference-matching rewards measure compatibility, not arbitrary visual difference from the guide.

Facial expression is a separate preservation criterion. The VLM prompt now checks mouth open/closed state, smile and eye open/closed state against the temporally aligned guide, not the identity reference's expression, and requests frame-specific evidence. This prompt update has not yet been tested with a live OpenAI call. Previously saved evaluations keep their original prompts and results.

Distinguish visual mouth-motion correspondence from audio/video lip synchronization. The current experimental worker's `lips` component correlates normalized mouth-opening landmark curves with an amplitude term; it does not consume audio and is not an audio lip-sync score. It is absent from the frozen v1 continuation. Evaluate mouth curves over dense aligned frames, report face/landmark detection coverage and undefined/low-motion cases, and avoid relying only on correlation, which can ignore amplitude differences. A separate audio/video metric is needed to verify phoneme timing against the original soundtrack. Sparse VLM images cannot establish this; the evaluator marks audio lip sync unobservable. Do not change the running v1 reward recipe to add these metrics mid-run.

### OpenAI checkpoint evaluation

The evaluator has an optional `--openai` transport using the official Chat Completions endpoint, default `gpt-5.4-mini`, JSON mode and recorded token usage. It rejects empty, refused or truncated answers; JSON mode ensures parseable JSON, not a fixed scoring schema or accurate judgments. This is an auxiliary checkpoint evaluator, not yet a VLM reward in the training loop. Keep ArcFace facial identity and the original background, lighting and pose rewards local and unchanged for the controlled continuation.

One live three-frame comparison completed successfully on 2026-10-05 with `gpt-5.4-mini`, low reasoning effort and the identity reference plus guide/A/B frames. It used 2,423 input tokens and 809 output tokens (including 321 reasoning tokens). At the published standard rates of $0.75/million input tokens and $4.50/million output tokens, this call cost approximately $0.00546 USD, excluding taxes or account-specific pricing. This is measured usage for one call, not a fixed per-video price; image size/count, output length, reasoning and repeated A/B checks change the cost. About 1,000 calls at that same usage would cost $5.46. Check current [model pricing](https://developers.openai.com/api/docs/models/gpt-5.4-mini) before budgeting.

The response distinguished candidates, used ties and marked body proportions uncertain where reference evidence was insufficient. This demonstrates functioning image input and useful response behavior, not established accuracy. The reversed-order OpenAI comparison and calibration against known failures have not yet been completed. These smoke-test outputs came from a training item and do not replace held-out checkpoint evaluation. GPT-5.4 mini is the initial cost/quality candidate; only switch to a cheaper model after checking its judgments on the same controls.

Configure `OPENAI_API_KEY` privately in the training process environment, never in a committed configuration or chat message. To enter it without showing it in terminal output/history:

```bash
read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
```

Then evaluate matched held-out outputs:

```bash
python tools/h3_rl/evaluate_vlm.py --openai --model gpt-5.4-mini \
  --guide /path/to/aligned-guide.mp4 --reference /path/to/reference.png \
  --generated /path/to/checkpoint.mp4 --baseline /path/to/s200.mp4 \
  --frames 3 --output /path/to/openai-evaluation.json
```

Repeat with `--reverse-order` and a different output filename, keeping every other argument identical. Save separate comparisons for normal and turbo sampling. Inspect face, hair and body criteria individually along with scene preservation. Keep the original local ArcFace, background, lighting and pose rewards active; adding an online VLM reward is a separate experiment after calibration.

Keep the original v1 ArcFace/background/lighting/pose rewards for the controlled continuation. Optional evaluators should first run on saved outputs, without modifying the training reward.

- OpenAI vision API: recommended as a candidate checkpoint judge and a way to calibrate local judges against manually checked examples. Send the identity reference and aligned guide/s200/checkpoint frames; request separate criteria and permit ties or unobservable criteria. Measure real token usage and cost on a small pilot. No OpenAI calls or OpenAI-specific evaluation backend have been validated in this setup yet; structured JSON does not establish judgment accuracy.
- UnifiedReward via local LM Studio: no per-call API fee. Q8_0 was confirmed loaded through `/api/v0/models`. On one three-frame comparison, identical inputs with A/B reversed still produced A as winner for every criterion in both orders. This is inconsistent for that pair and does not isolate quantization as the cause. Validate the official comparison prompt before using its scores for RL.
- DINOv2 image features: a candidate inexpensive auxiliary similarity reward for masked person/head crops against the reference, including nonhuman characters. Test whether it distinguishes identity/hair from pose, clothing and background; it is not an explicit hairstyle or identity classifier. The experimental recipe already has a DINOv2 character component. Do not add it to the frozen v1 recipe while measuring the effect of continuation.
- CLIP: useful for broad image/text semantic alignment, but a global similarity score can reward scene or clothing instead of the desired identity. Treat it as an auxiliary signal rather than the sole swap evaluator.
- A generic ViT is an architecture, not a ready-made reward. It needs suitable pretrained features or a task-trained scoring head. DINOv2 and CLIP already provide specific pretrained feature choices.

For every judge, test unchanged-guide failures, correct swaps, incorrect identity/hair, damaged backgrounds, identical pairs and A/B reversal with exactly the same frames. Evaluate held-out videos in normal and turbo modes. Sparse image sampling does not replace a temporal or lip-sync metric. Add one new reward at a time only after validating its behavior.

## Saved visual comparisons

`tools/h3_rl/save_eval_grid.py` saves `grid.png`, `metrics.json` and a README from aligned `guide.mp4`, `best.mp4`, `worst.mp4`, the identity reference and an NFT log call. Best/worst are selected by local advantage. Grid labels separately identify the local winner and the VLM's overall choice, with A/B mapped back to the actual files. Metrics include each selected rollout's raw rewards, advantage, training KL, VLM criteria and token usage. These rankings may disagree. Inputs must already be aligned.

```bash
python tools/h3_rl/save_eval_grid.py \
  --clips-dir /path/to/aligned-clips --reference /path/to/reference.png \
  --reward-log /path/to/nft_log.jsonl --call 1 \
  --vlm /path/to/vlm-response.json --output-dir /path/to/evaluation-grid
```

## Sources

- Existing algorithm: DiffusionNFT loss port and `docs/H3_RL_TRAINING.md`.
- [verl-omni DiffusionNFT H3](https://github.com/verl-project/verl-omni/tree/main/examples/diffusionnft_trainer/minimax_h3).
- [UnifiedReward-Edit pointwise rubric](https://github.com/CodeGoat24/UnifiedReward/blob/main/UnifiedReward-Edit/edit_pointwise_instruction_following.py).
- [LM Studio import](https://lmstudio.ai/docs/app/advanced/import-model).
- [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve).
- [Tailscale userspace networking](https://tailscale.com/docs/concepts/userspace-networking).
- [DINOv2 pretrained features](https://github.com/facebookresearch/dinov2).
- [CLIP](https://github.com/openai/CLIP).
- [OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision).

# H3 native temporal motion transfer v1

Fresh rank-64/alpha-64 LoRA on the H3 base Ref2VA model, with the usual frozen V2 training adapter. No motion-transfer or VFX LoRA weights or optimizer are inherited. Target: 3000 steps, checkpoints every 250. Swap is paused at its saved 250 checkpoint and optimizer.

Uses all 411 existing motion-transfer training pairs, unchanged captions. Dataset temporal sampling is fixed to 107 frames at 24 fps so the native video path and target share a physical frame clock. Original pairing and reference numbering are preserved. Geometry/resize of image references remains native. Source-video spatial geometry remains native as well; this is temporal alignment, not forced pixel correspondence.

The motion video is source control channel 1, independently of its packed location (images are normally packed first). The code keeps its native Ref2VA block and moves only its temporal RoPE coordinate onto the target clock. It does not use aligned_refs/AddGuide. Target queries receive a bounded soft temporal penalty when attending to distant motion-video keys. Text, image reference, target self-attention, reference audio and all non-target queries keep their original dense access. No new learned layers are introduced. CUDA uses FlexAttention without allocating an SxS bias matrix.

Temporal policy: 0.20-second free window, 0.25-second falloff scale, strength 2.0, penalty capped at 6.0. All controls can be disabled independently for ablations. This is an experimental inductive bias, not a guarantee of correspondence or quality. The model still learns target reconstruction through the standard H3 flow-matching objective.

Implementation is isolated at /data/ai-toolkit-h3-native-temporal. Training and the training-preview inference pipeline use the same layout and attention functions. LoRA metadata exports the temporal policy. Plain native ComfyUI inference without this patch is NOT equivalent; a consumer must implement the same layout/attention policy. The upstream ComfyUI and the saved swap run were not modified.

Validation covers source channel ownership, unchanged image-reference coordinates, native spatial geometry, matching temporal grids, padded batch ownership, unchanged non-target/image-reference edges, zero-bias equivalence, and CUDA forward/backward against a small dense oracle. Full-model smoke and training status are recorded separately in state.json.

## Reproduce training and preview inference

Start from `config/examples/train_h3_native_temporal.yaml`. Replace all `/path/to/` model, dataset and preview paths. Target captions are adjacent `.txt` files. Control folders use the matching target basename; channel 1 is the motion video and channels 2 onward are identity images. Keep the caption's picture numbering consistent with the supplied images. The example is a portable single-dataset recipe; the live run combines 411 pairs across 21 dataset entries.

```bash
python run.py config/examples/train_h3_native_temporal.yaml
```

Use an environment with CUDA-enabled PyTorch supporting `torch.nn.attention.flex_attention`; the live experiment uses PyTorch 2.12.0+cu130. Existing MiniMax H3 dependencies and model files remain required. Dynamic shapes may incur compilation on first encounters. No additional learned temporal layers or temporal loss are added.

The normal training sample path calls `MiniMaxH3Pipeline`, which applies the same temporal layout and attention policy as training. Configure `sample.samples` with prompt, guide and image controls; samples are enabled every 250 steps starting at step 500 in this recipe. Checkpoints save every 250 steps. Consequently the first saved checkpoint and first scheduled visual sample are different milestones.

For downstream inference, the consumer must retain the native video reference, identify its original source channel independently of packed order, call `align_layout`, preserve its spatial grid, and pass the resulting motion mask and temporal policy into the transformer. The transformer constructs the temporal context and applies the score modifier in its main attention blocks. The pipeline in this branch is the implemented inference path; this change does not ship a ComfyUI node or a standalone inference CLI. Native policy is exported as checkpoint metadata, but consumers must explicitly load/apply that policy: metadata alone does not patch attention.

## Policy and ablations

The attention score adjustment for target queries and motion-video keys is:

```text
score -= min(2.0 * (max(abs(t_target - t_guide) - 0.20, 0) / 0.25)^2, 6.0)
```

Times are in seconds. This is a soft preference, not a hard exclusion. Image and text references, target self-attention and reference-audio edges do not receive this penalty. Only the motion-video temporal RoPE origin is shifted; spatial coordinates stay native. This cannot correct mislabeled pairs, wrong identity assignments or motion that is not actually temporally paired.

- Full experiment: `native_temporal_alignment: true`, strength 2.0.
- Time-coordinate alignment only: keep enabled, set `native_temporal_strength: 0.0`.
- Native baseline: set `native_temporal_alignment: false`.

Keep all other conditions and seeds constant for a meaningful comparison. VSA is explicitly rejected in this mode. Do not combine it with aligned reference guides or source-phase RoPE patches.

## Validation and current limitations

CPU tests verify token ownership, matching temporal grids, padding and unchanged non-motion reference coordinates. CUDA tests compare FlexAttention forward/backward with a small dense oracle and exercise a small transformer with gradient checkpointing. Production attention never allocates the dense test oracle. The full H3 run has advanced through real optimization steps; visual quality and long-video generalization remain unverified pending saved samples.

```bash
python -m pytest -q tests/test_h3_native_temporal.py
python -m pytest -q tests/test_minimax_h3_aligned_refs.py tests/test_minimax_h3_video_temporal_cache.py tests/test_minimax_h3_guide_keep_aspect.py
```

The preceding prerequisite commit includes reference aspect preservation, source-aware packing, synchronized video-reference cache sampling, LoRA loader compatibility and associated regression tests. Dataset media, generated caches, credentials, checkpoints, optimizers and machine runtime state are intentionally not part of this source branch.


## ComfyUI consumer

The corresponding BFSNodes implementation is published separately on [`feat/h3-native-temporal`](https://github.com/alisson-anjos/ComfyUI-BFSNodes/tree/feat/h3-native-temporal). It provides a standalone MODEL patch node and a native-temporal option in Shot H3 Conditioning, with the same policy defaults. See [BFS inference instructions](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/feat/h3-native-temporal/H3_NATIVE_TEMPORAL.md). This is an experimental consumer with CPU parity/routing coverage; full ComfyUI GPU generation remains pending local validation.

For the compile-free inference implementation and illustrated derivation, see [Flash Partitions](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/feat/h3-native-temporal/docs/H3_FLASH_PARTITIONS.md). It preserves the bounded temporal score policy through native FlashAttention partitions and log-sum-exp merging; this does not change the active training backend or claim improved identity retention.

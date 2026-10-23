# TARA-H3 rank-64 multiscale experiment (prepared, not started)

The proposed fresh LoRA trains for 5,000 steps with native temporal alignment, rank/alpha 64, and no task-LoRA initialization. The frozen V2 H3 training adapter remains part of the training setup. The video is presented to Qwen/VLM at the existing 2fps context cadence. The full VLM presentation is deliberately unchanged by latent spatial augmentation.

The `h3_temporal_multiscale` extension picks one of `[1.0, 0.75, 0.5]` with equal probability using a deterministic per-step seed. The selected scale stays on the batch across all teacher/student forwards and gradient-checkpoint recomputations. Only the explicitly owned motion video source is resized. Identity images, other video references, the target canvas, sampled frame indices, frame count, timestamps and RoPE temporal policy remain unchanged.

The native video loader first performs ordinary H3 reference sizing and then rounds scaled dimensions down to the existing BFS /32 grid. This matches BFS's scale convention and may change aspect slightly. H3's existing area-normalized spatial RoPE is not multiplied again. In-memory and disk-cache identities distinguish non-unit scales; full-scale disk keys remain compatible with the pre-existing cache. Different scales may still produce identical rounded dimensions on very small grids; no promise of exactly scale-squared token counts is made.

The initial local split has 397 training entries and 14 held-out entries across eight source identifiers. UUID/basename ownership keeps those known derivatives together. This is not perceptual deduplication, and future comparisons should check for content leakage and include external user videos. The previous 1000-step native-temporal experiment already used rank64 and VLM video context; it did not use this multiscale augmentation.

## Evaluation gate

The FlashVSR+ restoration slider must be visually evaluated before this motion run is started. No automatic launch of this experiment is authorized by this document. Its current state is **prepared / waiting for slider evaluation**.

For the restoration slider, compare held-out degraded guides at adapter strengths 0, 0.5, and 1 with fixed prompts, seeds, sampling schedules and output dimensions. Include clean targets, cropped detail comparisons, PSNR/SSIM when aligned, and visual checks for identity change, invented texture, temporal flicker and motion alteration. A sharper-looking frame alone is not sufficient. Evaluate guide-latent mode first and native temporal mode separately before claiming compatibility. More training or mixed-mode training is conditional on observed benefit.

For TARA, compare fixed weights/inputs under native baseline, temporal RoPE only, temporal bias only, and combined TARA. Separately test reference scales 1, 0.75 and 0.5, and several seeds. Do not treat comparison to a different 10500-step task LoRA as a causal ablation. Report motion, identity retention, prompt obedience and warm/cold end-to-end latency. A future matched guide-trained baseline would require its own controlled training run.

For the Comfy challenge, prepare a functioning app/workflow, setup instructions, open-source app code, a reproducible example and a demo explaining the implementation. The organizer's [current page](https://comfy.org/open-source-higgsfield/) is the authority for eligibility and deadlines; this training plan is not a claim that the submission already satisfies all requirements.

## Reproduce the trainer configuration

Start from `config/examples/train_h3_native_temporal.yaml`, set process type to `h3_temporal_multiscale`, and add this process-level section:

```yaml
native_multiscale:
  scales: [1.0, 0.75, 0.5]
  seed: 42
```

Keep native temporal alignment enabled and Add Guide alignment disabled. Use a unique experiment name consistently in all name/meta fields and a separate output directory. When resuming, validate the exact checkpoint and optimizer together, and allow `start_step: null` to recover its step metadata. The current toolkit's automatic checkpoint choice uses filesystem timestamps, so moving old checkpoints can change what it selects; inspect the loading log before accepting a resumed run.

CPU coverage includes deterministic scale selection, valid grid sizing, ownership, and encoding/cache separation across all three scale variants. Full-model multiscale training and video quality remain unverified until the run starts.

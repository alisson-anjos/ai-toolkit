# H3 supervised restoration slider pilot

This is a separate experiment from TARA-H3. It learns a small, positive-strength restoration correction using matched degraded-guide / clean-target videos from one FlashVSR+ archive. It is **supervised residual distillation**, not a text-only Concept Slider and not a reproduction of Akane's paired slider loss.

At a shared noisy target, timestep, caption and degraded guide, compute frozen-base velocity b and the clean flow target v = noise - clean_latent in ai-toolkit's H3 convention. Train the adapter at scale a against b + a * bounded(v-b), with a sampled in [0.25,1]. A per-item RMS cap bounds the correction. Preservation batches retain the base target at a positive adapter scale. Scale zero disables the adapter; negative scales are not trained. Intermediate-scale behavior outside training inputs is not guaranteed linear.

The outer trainer owns backward and accumulation. The selected multiplier stays active through gradient-checkpoint recomputation and is restored after the training step. The objective uses video only. It does not add temporal or identity guarantees.

## Pilot preparation

Download **only** `data/flashvsr_plus_dataset/bucket_0024.tar` from [FlashVSR+](https://huggingface.co/datasets/ruchangyao/FlashVSR_Plus_Dataset), after obtaining access. Then:

```bash
python scripts/prepare_h3_flashvsr_slider.py /path/to/bucket_0024.tar /path/to/pilot --limit 64
```

The script selects clips spread across that archive, creates at most four-second/16fps pairs, caps the target at 768 pixels on the longest side, and generates guides by half-resolution down/up-sampling, mild blur and H.264 compression. Original archive videos are not modified. Guide and target canvas, frame count and frame rate are checked. One in eight selected clips is held out. Neutral restoration instructions are used as captions; no unobserved scene descriptions are invented. This is a synthetic degradation recipe, not a claim about the archive containing native LQ/HQ pairs. A single selected clip can land entirely in validation, so use multiple clips for training.

Inspect visual quality, source cuts, artifacts, available detail, duplication and representativeness before a larger run. At 512-pixel training resolution the pilot may not teach genuinely high-resolution detail. Preparation success is not dataset quality approval.

## Training

Edit the absolute paths in `config/examples/h3_flashvsr_sharpness_slider.yaml`, then:

```bash
python run.py config/examples/h3_flashvsr_sharpness_slider.yaml
```

The pilot selects only `blocks.40` through `blocks.49` MLP projections, rank/alpha 2. For the current H3 dimensions and two projections per block, that is approximately 1,075,200 adapter parameters (about 2.15 MB of BF16 tensors before file metadata). The actual export and matched module list must be checked at model initialization. No existing task LoRA is inherited. The frozen training adapter is a separate part of the existing H3 training setup.

Training uses aligned guide latents. The config restricts timestep sampling to a pilot refinement range, but scheduler/shift semantics must be checked against the actual second-pass sampling schedule; this is not a calibrated denoise prescription. The automatic preview is disabled in this initial config. Evaluate held-out clips at adapter strengths 0, 0.25, 0.5, and 1, holding seeds, prompts and sampling settings constant.

Use in **TARA-H3 native temporal mode is a separate compatibility experiment**, not a validated capability. Compare it against guide mode and check identity, motion, natural texture, flicker and unwanted oversharpening. Improvement cannot be established from training loss alone.

CPU objective checks and a single CPU-generated pair have been exercised. Full-model training, exported size, sample quality, and temporal-mode transfer are pending. Preparing or committing this experiment does not start, stop or modify an existing training run.

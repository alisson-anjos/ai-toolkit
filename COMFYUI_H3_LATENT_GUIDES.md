# ComfyUI inference contract for downscaled H3 latent guides

This fork's H3 upscale LoRA requires the same conditioning geometry at inference
as at training. A normal LoRA loader plus the native Add Guide node is not a
validated inference path for factors greater than one.

## Audited ComfyUI implementation

The audited revision is
[`8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1`](https://github.com/Comfy-Org/ComfyUI/blob/8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1/comfy_extras/nodes_minimax_h3.py#L166).
Its `MiniMaxH3AddGuide.execute` resizes guide frames to the target width and
height before VAE encoding. Resizing an image upstream to one quarter of the
output dimensions does not change that behavior.

The matching
[`H3Layout` model consumer](https://github.com/Comfy-Org/ComfyUI/blob/8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1/comfy/ldm/minimax/model.py)
builds keyframe condition rows from the full target spatial grid. A custom
node that only encodes smaller guide latents cannot satisfy that layout.

## Required node and model changes

An integration needs both a guide node with a `downscale_factor` input and a
model-side conditioning implementation that supports the resulting coarse grid:

1. Read the LoRA metadata, especially `reference_downscale_factor`,
   `minimax_h3_guide_spatial_version`, `minimax_h3_guide_position_version`, and
   the channel roles. Reject unsupported geometry versions explicitly.
2. Fit the source to the output canvas using the same centered crop as
   `toolkit.aligned_guides.prepare_guide_image`, then reduce each spatial axis
   by the factor with Lanczos **before** VAE encoding. The target axes must be
   multiples of `32 * factor`.
3. Preserve the guide's smaller VAE latent. For an output of 1024 x 768 and a
   factor of 4, the pixel guide is 256 x 192; its VAE spatial grid is 16 x 12,
   while the target's is 64 x 48. Do not interpolate the encoded guide back to
   the target latent size.
4. Build condition rows from the guide's actual token count. Its spatial
   coordinates must use every factor-th origin of the target **patch grid**:
   `target_frame_grid[::factor, ::factor]`, after reshaping that grid to its
   two spatial axes. Multiplying coordinates from an independently normalized
   guide grid by the factor is not equivalent.
5. Align guide time to the target timeline. For a source video, resample to
   the intended FPS, keep frame correspondence, and use H3's valid lengths
   `17*n + 5` (73 frames is approximately 3 seconds at 24 fps).
6. Keep pure guides out of Qwen/VLM conditioning, matching `guide_latent_only`.
   Native references retain their separate presentation and geometry.
7. Match token ordering, timestep conditioning, video/audio positions, and
   packed condition rows to this fork's `src/packing.py`. Handle single-image
   keyframe latents separately from multi-frame video guide latents.

Expected workflow after implementing and validating that integration:

```text
H3 Ref2V model + trained LoRA
    + caption-only text conditioning
    + output latent at the intended output dimensions
    + source image/video -> aligned downscaled guide node (factor 4, frame 0)
    -> compatible H3 packing/model consumer -> sampler -> VAE decode
```

## Validation gate

Use a held-out source and the same seed, prompt, factor, canvas, and sample
steps as AI Toolkit. Compare pixel preprocessing, encoded latent dimensions,
condition token counts, position IDs, and packed row ordering first. Then
compare generated outputs with LoRA strength 0 and 1. Loading a checkpoint
successfully does not demonstrate guide compatibility or upscale quality.

This document is an implementation contract, not an installed ComfyUI custom
node or a completed GPU inference test. The AI Toolkit patch does not modify
an existing ComfyUI installation.

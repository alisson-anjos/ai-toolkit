# H3 reference-scale audit — October 23, 2026

## Confirmed previous training

The local `bodyswap_h3_r64_v4_guidesidecar_f4_scratch` output contains a step-1000 safetensors checkpoint. Its embedded metadata records `training_info.step=1000`, `reference_downscale_factor=4`, `guide_latent_only=true`, `minimax_h3_guide_position_version=target_grid_stride_v1`, and `minimax_h3_guide_spatial_version=target_crop_v2`. The adjacent training config records `align_video_refs=true`, `align_image_refs=true`, and `guide_rope_layout=sidecar`.

This confirms an actual downscaled-guide training artifact, rather than just a proposed config. Factor 4 in that training API is a **divisor**, approximately scale 0.25 per spatial axis and 1/16 the guide spatial rows when dimensions divide exactly. It is not the same conditioning contract as TARA-H3 native temporal alignment. The checkpoint's existence does not establish output quality or transfer to TARA.

The current native temporal experiment's saved config uses factor 1. The later color-binding/motion run's 10500 configuration also uses factor 1. A separate `downscale2_NOT_READY` config is a proposal and is not evidence that it trained.

## Correction: spatial normalization already exists

Both this fork's `_spatial_position_grid` and the inspected ComfyUI H3 `_axis_from_sqrt_area` construct each axis using `dim / sqrt(height * width)`. Native references therefore already occupy an aspect-normalized spatial coordinate domain.

For an exact, aspect-preserving half-scale grid, the remaining patch origins coincide with every other full-resolution patch origin. Multiplying those coordinates by 2 would apply an unjustified second scaling. The regression checks in `tests/test_h3_native_scale_audit.py` exercise this property and the distinct case of changed aspect ratio.

Consequently, no new spatial-coordinate transform is enabled merely because a reference is smaller. TARA still adds its temporal clock alignment and selective temporal penalty; the spatial normalization is inherited from H3.

## Next controlled experiment

Keep output resolution, temporal grid, prompt, references, seed and sampler fixed. Compare actual (rounded) guide dimensions at 1.0, 0.75, and 0.5 and record guide token counts and aspect ratios. Compare several seeds/clips and separate:

- Loss of source information, especially small faces, hands, mouth motion and overlapping subjects.
- Training/inference scale mismatch.
- Aspect-ratio changes from rounding to valid model grids.
- Changes in aggregate reference attention associated with fewer source keys (a hypothesis, not an established cause).

Any reference attention gain or token-count compensation must be a separate ablation, not silently introduced as a fix. A temporal-strength increase alone does not restore discarded source detail.

A training recipe with varying reference scales should keep target geometry and temporal sampling fixed while carrying the chosen scale into reference resizing, layout construction, cache signatures, and preview inference. This audit does not implement or launch that mixed-scale trainer. Do not substitute a global downscale factor for per-batch multi-scale sampling and claim equivalence.

No live run was modified. CPU coordinate tests do not establish visual equivalence or native-mode quality.

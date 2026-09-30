# Experimental H3 reference RoPE

Optional H3 adaptations of overlap/sidecar layouts and per-source rotary phase,
inspired by [LTX-2 PR #289](https://github.com/Lightricks/LTX-2/pull/289).
These are new H3 experiments, not a validated transfer of LTX identity quality.
They do not change the existing latent-guide defaults or a running process.

## Configure

In the UI, open **Experimental Reference RoPE** in the H3 options. Equivalent
settings under `config.process[0].model.model_kwargs`:

```yaml
guide_rope_layout: overlap          # overlap (default) or sidecar
reference_rope_layout: overlap      # native (default), overlap or sidecar
reference_source_phase: true        # false by default
reference_phase_scale: 1.0
reference_sidecar_margin: 0.0       # normalized H3 RoPE units, not pixels
```

The default tuple is `overlap / native / false / 1.0 / 0.0`. It preserves the
old geometry and rotary angles exactly. Ref2VA only; first/last keyframe and
D-OPSD teacher modes reject incompatible experimental options.

- **Guide overlap:** preserve the target clock and stride-factor target grid.
- **Reference native:** preserve H3's separate reference grid and cumulative clock.
- **Reference overlap:** keep its own aspect-normalized grid, but share the target
  clock without advancing it. An identity portrait is not cropped into the guide
  canvas or downscaled by the guide factor.
- **Sidecar:** place the block beyond the final target width patch origin, plus
  one target patch step and the optional margin, with vertical centers aligned.
  The reference shares the target clock and keeps its own frame spacing. This
  is an H3 adaptation in normalized coordinates; it does not copy LTX patch-bound
  units or stretch a still over the target duration.
- **Source phase:** compose `source_id * phase_scale * 10000**(-d/D)` with the
  rotary angles (`D=48` for the normal H3 half-RoPE). Both duplicated halves get
  the same added phase. Target/text/target audio remain source 0, an exact no-op.
  This introduces no new learned parameters or base-model checkpoint keys.

## Guide plus identity

Use `control_path_1` with `control_role_1: guide`, and `control_path_2` with
`control_role_2: reference`. Match each control file to its target basename as
usual. Source IDs are the original control channels (1, 2, 3), independent of
image/video packing order or dropout; a missing/dropped source is not renumbered.
Keep channel roles and caption conventions consistent across the dataset.
Tags in captions do not automatically bind themselves to channels.

For the BFS caption-only identity node use `control_latent_only: true` and
`cache_text_embeddings: true` on every dataset. The underlying model remains
`minimax_h3_ref2va`. Keep `reference_downscale_factor: 4` for a 4x guide.
An additional identity reference is useful only if similar references are
present during training. The existing upscale dataset does not automatically
provide a paired identity portrait or character sheet.

Start with overlap plus phase and compare to the default baseline on held-out
people, with identical prompts/seeds. Sidecar is another ablation, not an identity
guarantee. Additional identity tokens increase attention memory.

## Resume, validation, and inference

Training and the preview sampler use the same layout and phase helpers. Checkpoint
metadata `minimax_h3_reference_rope` records all five options plus
`phase_version: h3_source_phase_v1`. A checkpoint trained with these settings
requires the same geometry at inference. Changing geometry on an old LoRA is an
experimental initialization/fine-tune, not an equivalent resume: use a separate
job/output folder and validate before a long run.

BFSNodes 1.47.0 exposes matching optional inputs on **MiniMax-H3 Downscaled Latent
Guide (BFS)** and adds **MiniMax-H3 Identity Reference + RoPE (BFS)**. Chain both
MODEL and CONDITIONING outputs. Guide source 1 and identity source 2 correspond
to training control channels 1 and 2. Select the LoRA for metadata validation;
apply its weights through the normal LoRA loader. The identity node resizes down
only, preserving the reference aspect, on the same /32 image bucket used by the
trainer, and uses the same seeded VAE posterior recipe.

Tests cover exact default behavior, source-zero invariance, batch text padding,
all overlap/sidecar combinations against the actual native Comfy layout with
image/video/audio references, and the actual validation call path. These tests
establish geometry/rotary parity, not visual identity quality. A full pretrained
H3 training + Comfy generation experiment with phase enabled remains unvalidated.

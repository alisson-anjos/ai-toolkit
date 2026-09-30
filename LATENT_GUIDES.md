# MiniMax H3 latent guides and training controls

Built on upstream `ecee894ed2b1f3716d9d7326693061ec1a3105bb`, starting from [Alisson's aligned-video patch](https://gist.github.com/alisson-anjos/b300f2b90e65cf85846519d78b660cd3).

## UI

Create/edit a job, choose **MiniMax-H3 Ref2V**, then set **Control Conditioning**:

- **Conventional references** keeps the original reference layout and VLM presentation.
- **Latent guides (images / videos) + native references** enables explicit **Channel Role** selectors beside each dataset control folder and each sample upload. Choose **Guide latent** for spatially/temporally aligned VAE conditioning, or **Native reference** for ordinary reference coordinates and VLM presentation. Defaults: channel 1 guide, channels 2/3 native. Roles can differ between datasets; reproduce the appropriate roles in samples/inference. Image guides bypass the VLM and require cached text embeddings; the mode enables that setting on existing datasets. Native image references remain available simultaneously.
- **Latent video guides + native image references** aligns video controls and omits them from the VLM. Images retain native reference latents and VLM conditioning.
- **All controls as latents** also omits image references from the VLM; captions remain. Images still use ordinary unaligned reference latents.
- **Aligned video guides + VLM presentation** keeps video presentation in both paths.

**Guide Downscale Factor** offers 1, 2, 4, and 8. Factor 2 halves both guide dimensions, giving one quarter of the guide video tokens. Target buckets and samples snap to multiples of `32 * factor` so the VAE and DiT patch grids remain valid. This can slightly change the target canvas selected for an aspect bucket. CLI users may use other positive integer factors with compatible dimensions.

The three native control dataset channels and sample upload channels are preserved. A typical arrangement is source video, identity image, and mask video. Every video control is an aligned guide in guide mode; ordinary images remain unaligned references. The same factor applies to all video guides, including masks. Still images should use **Picture** presentation. The UI prevents unsupported combinations with D-OPSD and static-video image presentation.

## Configuration fragment

Place this fragment inside a complete training process configuration:

```yaml
model:
  arch: minimax_h3_ref2va
  model_kwargs:
    align_video_refs: true
    guide_latent_only: true
    control_latent_only: false
    reference_downscale_factor: 2
    reference_dropout: 0.0
    guide_dropout: 0.1

datasets:
  - folder_path: /path/to/target
    control_path_1: /path/to/source_video
    control_path_2: /path/to/identity_images
    control_path_3: /path/to/mask_video
    resolution: [384]
    num_frames: 73
    auto_frame_count: false
    batch_size: 1
```

Pair control files with targets by basename using the normal dataset loader. The reference loader retains native media pacing and frame-count handling. Source/target clips and mask videos should represent the same timeline. Existing temporal trimming rules are unchanged.

## Geometry and cache

The guide is encoded at `target_height / factor` and `target_width / factor`. H3's spatial coordinates are already area-normalized, so coarse guide tokens use every factor-th target patch origin. Multiplying those normalized coordinates by the factor would be incorrect. Guide clocks stay aligned, ordinary image references retain their layout, and soundtracks retain full target spatial endpoints and temporal spacing.

Aligned image controls reproduce the target bucket's exact resize, crop and flips before reduction. Video guides also reproduce that spatial transform and use the target's selected frame indices (converted by source FPS when necessary). Sampling explicitly fits guide content to the output canvas. Only blocks marked as guides align; ordinary references retain their own layout. Video cache keys include crop/flips, temporal selection and FPS, as well as target dimensions and factor. Random video windows require recorded target frame indices; use deterministic shrink-to-frames or real-time tail trim for cached targets.

Both disk and memory reference caches include the factor and target dimensions. Caption-only, native-image-only VLM, and conventional VLM embeddings have distinct cache versions. Training and sampling use the same size helper, factor validation, and packing rules.

## Mask guides

A mask video can be supplied as a second aligned video guide; its rows are separate from the source guide but share the target coordinate system. Train on paired `(source, mask, caption, edited target)` examples so the model learns the mask's meaning. For example, establish white = editable and black = preserved in the dataset, then use that same convention in inference. The backend does not impose that convention or turn the mask into a loss mask.

A mask latent is learned conditioning, not a hard pixel lock. Exact preservation outside the mask requires a separate inpainting constraint or output composition step. Downscaling the mask also reduces boundary precision; factor 1 retains the most mask detail.

## Dropout

**Native Reference Dropout** and **Image / Video Guide Dropout** are probabilities in [0, 1], applied independently to each reference within the respective group. A single selection is shared across items/passes of one batch so guidance and preservation forwards see the same controls. Sampling does not apply training dropout. Dropped references are removed before VAE encoding and packing; configured control paths are not erased.

Guide dropout applies to both aligned image and video guides. Native reference dropout applies to native image/video references. Video dropout works in either latent-only video presentation mode. Image dropout requires **All controls as latents**, because cached Qwen caption embeddings can retain image information even if the image vision-token rows are removed afterward. This restriction avoids misleading partial dropout. D-OPSD/static-video image presentation reject reference dropout. Use batch size 1 for actual variable reference counts or aspects across items; larger batches retain the existing same-reference-geometry requirements.

## Auxiliary losses

The UI has a multi-select for **Pixel L1**, **ArcFace similarity**, and **Pose heatmaps**, plus independent weights. These augment the standard training objective rather than replacing it. Losses compare a differentiably decoded clean estimate `x0 = noisy_latents - sigma * prediction` against the decoded training target. Evaluators are frozen, and target evaluation has no gradients. Logs report each unweighted and weighted term.

No pretrained evaluators are bundled or downloaded implicitly. Pixel L1 works without an evaluator. ArcFace and pose require local TorchScript exports with these contracts:

| Kind | Input | Output |
| --- | --- | --- |
| ArcFace | RGB BCHW, [-1, 1], 112x112 face crop | BxD differentiable embeddings |
| Pose | RGB BCHW, [0, 1], 256x256 | BxKxHxW differentiable heatmaps |

A regular OpenPose detector returning discrete points, an ONNX-only InsightFace model, or an evaluator that detaches its output cannot directly supply a training gradient. Export/wrap the differentiable feature network in TorchScript. Inputs must follow the above normalization; adapt the wrapper for checkpoints expecting a different convention.

ArcFace uses a configurable normalized target-frame rectangle `[left, top, right, bottom]`. No face detector/alignment/tracker is provided; the ROI should contain the target face throughout the sampled loss frames. Full-frame portraits can use `[0, 0, 1, 1]`. The identity comparator targets the edited ground-truth face, not an arbitrary control-channel image.

Defaults apply losses only at sigma <= 0.5, decode the first 2 latent frames (5 pixel frames), and sample one decoded frame. The UI can expand the decoded prefix or use the full clip and multiple sampled frames. Partial active batches are weighted by their active fraction. The VAE decode adds memory/compute; tune weights and clip length against a baseline before treating these objectives as a quality improvement.

```yaml
model_kwargs:
  auxiliary_losses:
    - type: pixel_l1
      weight: 0.05
    - type: arcface
      weight: 0.1
      model_path: /models/arcface_rgb112.pt
      crop: [0.25, 0.0, 0.75, 0.6]
    - type: pose_heatmap
      weight: 0.05
      model_path: /models/pose_rgb256_heatmaps.pt
  auxiliary_loss_max_sigma: 0.5
  auxiliary_loss_latent_frames: 2
  auxiliary_loss_sample_frames: 1
```

## LoRA metadata and external inference

The training save path writes `reference_downscale_factor`, `align_video_refs`, `guide_latent_only`, `control_latent_only`, and `minimax_h3_guide_position_version=target_grid_stride_v1` into LoRA safetensors metadata. The model hook is called for network saves, not only full-model saves. Dropout probabilities and auxiliary-loss descriptions are also recorded as training provenance; evaluator file paths are omitted. Safetensors serializes numeric/boolean/list values as strings/JSON.

AI Toolkit training previews use the same guide geometry. External ComfyUI loaders must read the factor/presentation flags and construct the strided H3 guide positions. Existing same-resolution AddGuide workflows are not automatically compatible with downscaled guides. No ComfyUI code is changed by this patch.

## Validation

```bash
python -m pytest tests/test_minimax_h3_aligned_refs.py \
  tests/test_minimax_h3_downscaled_refs.py \
  tests/test_minimax_h3_training_options.py -q
node tests/test_minimax_h3_ui.cjs
cd ui
npm run check_extensions
npx tsc --noEmit --incremental false
npm run build
```

Unit tests load actual packing/helper modules and source-extracted model methods to avoid importing all unrelated training models. They cover geometry, memory/disk caches with real generated video files, sampling resize, safetensors metadata round-trip, dropout, differentiability, and UI configuration transitions. Tiny random H3 transformer forward/backward tests run on CPU; evaluator tests use tiny TorchScript fixtures, not pretrained ArcFace/OpenPose weights. These checks do not establish training quality.

The production UI build and extension TypeScript check passed. A full TypeScript check after Next generated its route types reports 43 existing upstream route-parameter diagnostics; comparison against HEAD source showed no new diagnostics from this patch. CUDA initialization fails in this workspace despite `nvidia-smi` listing an RTX 5090; no full-model GPU training or inference has been performed.

## Related upstream ideas

The spatial-factor concept follows [LTX IC-LoRA training](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py), adapted to H3's different coordinate normalization. [Akendo's H3 trainer](https://github.com/AkaneTendo25/musubi-tuner/blob/minimax-h3/docs/minimax_h3.md) documents composable auxiliary objectives such as DOP, CREPA, and SOAR, separate weighted logs, aligned guides, and conditioning masks. This patch does not port those objectives or claim they implement ArcFace/OpenPose.

## Explicit per-channel image/video guide configuration

```yaml
model:
  arch: minimax_h3_ref2va
  model_kwargs:
    align_video_refs: true
    align_image_refs: true
    guide_latent_only: true
    control_latent_only: false
    reference_downscale_factor: 4
    image_guide_channel: 1  # fallback only; explicit dataset/sample roles take precedence

datasets:
  - folder_path: /workspace/ai-toolkit/datasets/h3upscale_images_targets
    control_path_1: /workspace/ai-toolkit/datasets/h3upscale_images_guide_sources
    control_role_1: guide
    cache_text_embeddings: true
    cache_latents_to_disk: true
    resolution: [1024]
    num_frames: 1
    batch_size: 1
  - folder_path: /workspace/ai-toolkit/datasets/h3upscale_videos_targets
    control_path_1: /workspace/ai-toolkit/datasets/h3upscale_videos_guide_sources
    control_role_1: guide
    cache_text_embeddings: true
    cache_latents_to_disk: true
    resolution: [1024]
    num_frames: 5  # H3 lengths are 1 for images, or 17n+5 for videos
    auto_frame_count: false
    shrink_video_to_frames: true
    batch_size: 1

# Put each preview in sample.samples in a full job configuration:
# ctrl_img_1: /path/to/guide.png or /path/to/guide.mp4
# ctrl_role_1: guide
# ctrl_img_2: /path/to/identity.png
# ctrl_role_2: reference
```

These are fragments, not complete jobs. Dataset media stays at its source resolution; the guide is reduced once against each selected target bucket. Resolution is a square-equivalent pixel budget, not the long edge; a 4096 setting is not synonymous with 3840x2160. Start with a small canvas and 5 video frames before testing 4K memory. This change has CPU geometry/condition-builder/sampler/cache and UI coverage, but no full pretrained H3 training run; the current environment still reports CUDA initialization failure and lacks the full Python training dependency stack.

LoRA metadata additionally records `align_image_refs`, the fallback `image_guide_channel`, `minimax_h3_dataset_control_roles` (channel-role triples), and `minimax_h3_guide_spatial_version=target_crop_v2`. These describe conditioning; external inference integrations still need to implement the equivalent roles, factor, crop, and packed layout.

## Motion-transfer dataset awaiting inspection

The proposed `animate_small.zip` Google Drive file is public but currently returns a download-quota error. Its JSON and media have not been inspected. A provisional direction is GT as motion guide and synthetic as target, with a synthetic appearance/first-frame image as a separate reference or aligned first-frame guide, conditional on actual frame correspondence. Do not invert the pair or interpret JSON reference indices as temporal alignment without checking examples.

Embedded guide-video soundtracks already enter the audio VAE as clean conditioning rows; when the video block is aligned, its audio is on the target clock. This is not yet an independent UI selector for audio-guide versus native standalone audio reference. Separate audio roles/inputs and synthetic-target audio handling remain future work, informed by the paired dataset. A synthetic target without audio must not accidentally train a speech-generation objective against silence.

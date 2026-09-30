# MiniMax H3 training with latent guides and native references

This fork supports aligned image/video latent guides, reduced-resolution guides, explicit roles for each control channel, conditioning dropout, and optional auxiliary losses.

## Select the mode in the UI

Create a job, select **MiniMax-H3 Ref2V**, then choose **Latent guides (images / videos) + native references** under **Control Conditioning**.

For each dataset, select its **Target Dataset** and **Control Dataset 1/2/3** folders. Each control has a **Channel N Role** selector:

| Role | Behavior |
| --- | --- |
| **Guide latent — aligned with target** | Accepts an image or video. The VAE encodes the guide, and its tokens share the target's spatial coordinates and time. Guide Downscale Factor applies. The guide bypasses the VLM. |
| **Native reference — identity / appearance** | Uses the native H3 reference layout and VLM presentation. It retains its own coordinates. Guide Downscale Factor does not apply. |

Channel 1 defaults to a guide, and channels 2/3 default to native references. Any channel can be changed. Roles can differ between datasets. For example, use a motion video as a guide in channel 1 and an identity image as a native reference in channel 2.

An image marked as **Guide latent** uses aligned guide conditioning; it is not a native image reference. A native appearance reference does not enforce an exact first frame. An image guide aligns with the beginning of the target, but it is still learned conditioning, not a hard pixel lock.

The same role selectors appear beside sample uploads. Reproduce the training roles in sampling and inference.

## Pair targets, controls, and captions

Target and guide files must have matching names in their respective folders. Captions belong only in target folders:

```text
datasets/h3upscale_images_targets/example.jpg
datasets/h3upscale_images_targets/example.txt
datasets/h3upscale_images_guide_sources/example.jpg

datasets/h3upscale_videos_targets/example.mp4
datasets/h3upscale_videos_targets/example.txt
datasets/h3upscale_videos_guide_sources/example.mp4
```

For a baseline upscaler, guide-source folders can contain the same originals as the target folders. The loader reproduces the target crop and reduces the guide once for each training canvas. Precomputed copies at several resolutions are unnecessary. Different source/target files need appropriate content, spatial, and temporal correspondence.

Example caption:

```text
h3upscale, increase resolution and restore fine details while preserving the original content and visual style.
```

`h3upscale` is a regular caption string, not a new tokenizer token. At inference, load the LoRA, supply the low-resolution guide through the matching conditioning path, and start with the same prompt.

## Resolution and Guide Downscale Factor

**Resolution defines the target's pixel budget.** There is no separate guide-resolution field: the guide dimensions are calculated from the selected target canvas.

| Factor | Guide dimensions | Example |
| --- | --- | --- |
| 1 | Same dimensions as the target | 1024×768 → 1024×768 |
| 2 | Half the width and height | 1024×768 → 512×384 |
| 4 | One quarter of the width and height | 1024×768 → 256×192 |
| 8 | One eighth of the width and height | 1024×768 → 128×96 |

A relative scale of `0.5` means **factor 2** in this fork. Factor 4 reduces guide spatial tokens by 16×, while target-token cost remains unchanged.

Target dimensions must be multiples of **32 × factor**. Buckets and previews snap to this grid. The UI's Resolution is a square-equivalent pixel budget, not a maximum long edge: setting 4096 does not automatically mean 3840×2160. Inspect actual bucket dimensions and memory usage before enabling a 4K stage.

Image and video guides reproduce the target resize, crop, and flips before reduction. Video guides also follow the selected target frame indices, accounting for FPS. Different generated video pairs with offsets or mismatched motion still need dataset-specific synchronization.

## Start with an upscale smoke test

Add images and videos as separate datasets so their frame settings can differ:

| Target dataset | Control Dataset 1 | Channel 1 role |
| --- | --- | --- |
| `h3upscale_images_targets` | `h3upscale_images_guide_sources` | **Guide latent** |
| `h3upscale_videos_targets` | `h3upscale_videos_guide_sources` | **Guide latent** |

Enable **Cache Text Embeddings on every dataset**. This mode requires cached embeddings to separate guides from native VLM conditioning correctly. Selecting the mode enables the setting on existing datasets; check datasets added afterward too. Setting only the training-level cache flag is insufficient for mixed dataset configurations.

Start with batch size 1, one small target resolution, and **1 frame for images**. Valid video lengths are **5, 22, 39… (17n + 5)**. Use 5 frames for an initial visual-only test, with **Auto Frame Count** off and **Shrink Video to Frames** on. Start with caption dropout, guide dropout, flips, and random crops disabled. Add larger resolutions and longer clips after measuring memory and validating conditioning.

Images and videos can share a job; batches remain separated by geometry and frame count. Keep validation examples outside the training folders. Supply a nonempty sample prompt and an actual guide file; an empty, guide-free preview does not test the upscale task.

See [the smoke-test configuration](config/examples/train_lora_minimax_h3_upscale_smoke.yaml). Set its held-out guide path before running it. It deliberately starts with 50 steps, one resolution, and no target audio loss. For audio-supervised video training, supply valid target audio and preserve its timing; a silent target must not accidentally supervise speech against silence.

The baseline applies reduction only. Noise, blur, compression, and pixelation are not automatically added.

## Dropout and optional auxiliary losses

**Image / Video Guide Dropout** independently removes guides. **Native Reference Dropout** removes native references and requires caption-only VLM conditioning to avoid reference information leaking through cached embeddings. The UI disables unsupported combinations. A batch shares one selection across its training passes; inference does not apply training dropout.

Optional losses can be combined with independent weights:

- **Pixel L1** compares reconstruction against the target, without an external evaluator.
- **ArcFace** requires a local differentiable TorchScript evaluator and a configured face crop.
- **Pose heatmaps** requires a local differentiable TorchScript evaluator; discrete OpenPose keypoints cannot supply this gradient.

Evaluators are not downloaded implicitly. These losses decode part of a clean estimate and add compute/memory cost. Start without them. Evaluator contracts and sigma/frame controls are documented in [LATENT_GUIDES.md](LATENT_GUIDES.md).

## LoRA metadata and external inference

LoRA metadata records the factor, conditioning flags, observed channel-role combinations, and geometry versions. Inference must reproduce those settings. This fork's previews support the same geometry.

External integrations, including ComfyUI, need the equivalent packed layout and metadata interpretation. Loading the LoRA into an existing same-resolution AddGuide node does not automatically support downscaled guides. This patch does not modify ComfyUI.

See [the ComfyUI inference contract](COMFYUI_H3_LATENT_GUIDES.md) for the audited native node, required node/model changes, and a compatibility validation checklist.

## Long video training and OOM recovery

Five frames is a smoke-test setting, not a three-second training clip. At 24 fps,
73 frames is approximately three seconds; 107 is approximately 4.5 seconds and
124 is approximately 5.2 seconds. These lengths substantially increase target
token and activation cost. Measure a real video backward pass on the intended
GPU before launching a long mixed image/video run. Disabling
`shrink_video_to_frames` preserves the configured FPS for sufficiently long
source clips; clips too short to supply that window are still stretched by the
loader and should be checked separately.

Keep `cache_latents_to_disk: true` for reusable target latents. Text embeddings
must be cached on each dataset in latent-guide mode. Control video latents have
their own lazy disk cache; the presence of a target cache alone does not prove
all guide preprocessing has completed.

H3 target video caches store the selected source frame indices and source FPS.
Aligned guides restore this selection, including when workers reload a target
latent from disk. Older caches without this temporal information are regenerated:
their random window cannot be recovered safely. Changing `shrink_video_to_frames`
also selects a different target cache, so guide and target keep the same clock.

OOM reports include each batch file, target canvas, and frame count. Automagic
v3 with `fused: true` updates parameters during backward. An OOM can therefore
leave partial parameter updates; clearing gradients cannot roll them back.
The trainer stops rather than silently skipping such a batch. Resume from a
saved checkpoint after reducing memory use, or configure
`optimizer_params.fused: false` for conventional optimizer steps (which uses
more gradient memory). Other optimizers can still skip recoverable OOMs and
stop after three consecutive failures.

To compare one held-out image across saved checkpoints, run:

```bash
python scripts/compare_h3_validation.py \
  --guide /path/to/held-out-guide.jpg \
  --samples output/your_job/samples \
  --output output/your_job/validation_grid.png \
  --width 1024 --height 768 --factor 4
```

The grid includes the effective reduced guide, bicubic interpolation, the
source fitted to the output canvas, the earliest available sample, and recent
sampled checkpoints. The fitted source is a reference target only for
same-source upscale pairs. The script also saves the reduced guide separately.

Embedded guide-video soundtracks can enter the audio VAE as clean conditioning rows. There is **no independent native-audio versus guide-audio selector or standalone audio upload for this mode yet**. A mask guide is learned conditioning, not a loss mask or an exact lock outside the mask.

## Prepared dataset on Hugging Face

The public dataset is [Alissonerdx/h3upscale-4k](https://huggingface.co/datasets/Alissonerdx/h3upscale-4k). The complete upload contains 5,120 media files with matching captions, a manifest, and an installer.

Download the public dataset and restore its AI Toolkit folders:

```bash
hf download Alissonerdx/h3upscale-4k --repo-type dataset --local-dir /workspace/h3upscale-download
python /workspace/h3upscale-download/install_dataset.py --dataset-root /workspace/ai-toolkit/datasets
```

The installer checks for missing media/captions, recreates target and guide-source folders, and keeps validation under `.h3upscale/validation`. Guide sources are reconstructed locally so original media only needs to be stored once on the Hub. Hard-linked targets/guides share bytes; avoid modifying guide media in place.

## Update and run the UI

Dataset/caption edits do not require a build; refresh the page. UI source changes require a rebuild and restart:

```bash
git switch minimax-h3-latent-guides
git pull --ff-only
cd ui
npm install
npm run update_db
npm run build
npx concurrently -k "node dist/cron/worker.js" "node dist/cron/fileServer.js start --port 8188"
```

Stop the previous UI instance before starting another on the same port. Install the repository's Python requirements in its `.venv` or `venv`; the UI selects these automatically when present. Verify CUDA before starting a job:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

CPU tests cover geometry, condition building, caches, isolated sampling, and UI controls. They do not replace a full training run with pretrained H3 weights and a working GPU.

"""Raw, untrained ScanPE test on MiniMax-H3 (no LoRA, no fine-tune -- just the
positional-encoding change in packing.py's build_packed_sequence).

Baseline: generate N latent frames with the model's normal STATIONARY grid
(every frame reuses the identical frame_grid -- packing.py's original
behavior) -- expect near-identical/repeating frames, per ScrollScape's own
"vanilla" ablation.

ScanPE: generate the same N frames but with an accumulated per-frame
horizontal offset on the target video block's rotary grid (frame_grid + t*stride
on the width axis), turning the temporal axis into a spatial-pan axis.
ScrollScape's own ablation says untrained ScanPE breaks the repetition but
gives chaotic textures -- this script exists to see exactly that on H3.

Both conditions are decoded frame-by-frame and raw-concatenated (NO blending
/ TAP / fusion -- that's later work) into one wide strip for visual
inspection.

Run from the ai-toolkit root:
    .venv/bin/python scanpe_experiments/scanpe_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import get_model_class
from extensions_built_in.diffusion_models.minimax_h3.src import packing

OUT_DIR = "scanpe_experiments/scanpe_out"

PROMPT = """subject_definitions:

summary:
[reference generation] The target video shows a sweeping photorealistic mountain valley landscape at golden hour, with a winding river, pine forest, and distant snow-capped peaks.

retention_analysis:

detailed_description:
The image is a sharp, photorealistic wide landscape photograph, golden hour lighting, warm directional sunlight, crisp fine detail on foliage, water, and rock texture.
[Shot 1] A wide mountain valley stretches out, a winding river catching the golden light, dense pine forest on the near slopes, snow-capped peaks in the distance under a clear warm sky.

overall_soundscape:
N/A

non_diegetic_music:
N/A
"""

NUM_FRAMES = 22  # 17*1+5 -> t_lat = 7 latent frames
SIZE = 512  # square per-frame canvas: h_lat == w_lat == 32, ratio 1.0, local
# grid spans exactly _ROPE_SPATIAL_SCALE (32) units -- so stride=32 tiles
# frames edge-to-edge with no overlap
STRIDE = float(packing._ROPE_SPATIAL_SCALE)
STEPS = 40
SHIFT = 3.0  # the Path-1 recalibrated value, isolates the ScanPE effect from
# the separate schedule-miscalibration issue


def make_strip(frames_uint8, out_path):
    imgs = [Image.fromarray(f) for f in frames_uint8]
    w, h = imgs[0].size
    strip = Image.new("RGB", (w * len(imgs), h))
    for i, im in enumerate(imgs):
        strip.paste(im, (i * w, 0))
    strip.save(out_path)
    print(f"saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model_config = ModelConfig(
        name_or_path="Comfy-Org/MiniMax-H3",
        arch="minimax_h3",
        dtype="bf16",
        quantize=True,
        quantize_te=True,
        layer_offloading=True,
        layer_offloading_transformer_percent=0.85,
        layer_offloading_text_encoder_percent=0.85,
        model_kwargs={"partition": "ref2va_pruned"},
    )
    ModelClass = get_model_class(model_config)
    model = ModelClass(device="cuda:0", model_config=model_config, dtype="bf16")
    model.load_model()
    pipeline = model.get_generation_pipeline()

    conditional_embeds = model.get_prompt_embeds(PROMPT, control_images=None)

    t_lat = packing.video_latent_num_frames(packing.align_num_frames_down(NUM_FRAMES))
    print(f"t_lat = {t_lat}")

    for label, offsets in [
        ("baseline_stationary", None),
        (
            "scanpe_linear",
            torch.stack(
                [torch.zeros(t_lat), torch.arange(t_lat, dtype=torch.float64) * STRIDE],
                dim=-1,
            ),
        ),
    ]:
        print(f"\n=== {label} ===")
        generator = torch.Generator(device="cpu").manual_seed(42)
        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=None,
            height=SIZE,
            width=SIZE,
            num_frames=NUM_FRAMES,
            num_inference_steps=STEPS,
            generator=generator,
            ref_images=None,
            with_audio=False,
            sigma_shift=SHIFT,
            scan_offsets=offsets,
        )
        video = result["video"].numpy()  # (T, H, W, C)
        for i in range(video.shape[0]):
            Image.fromarray(video[i]).save(os.path.join(OUT_DIR, f"{label}_f{i}.png"))
        make_strip(video, os.path.join(OUT_DIR, f"{label}_strip.png"))


if __name__ == "__main__":
    main()

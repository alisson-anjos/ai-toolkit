"""ScanPE as a 2D tiling grid instead of a 1D pan: use the temporal axis to
cover a big SQUARE/standard-aspect canvas at higher-than-native resolution,
snake-ordered (row L->R, next row R->L, ...) so temporally-adjacent frames
are always spatially adjacent -- matches ScrollScape's "Snake Fusion Mode".

Still no LoRA / fine-tune -- raw untrained ScanPE, just a different offset
trajectory than the linear-pan test. Tiles are pasted edge-to-edge with NO
blending (non-overlapping grid, unlike ScrollScape's overlapping-window
design) since we want to see the raw per-tile behavior first.

Run from the ai-toolkit root:
    .venv/bin/python scanpe_experiments/scanpe_grid_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import get_model_class
from extensions_built_in.diffusion_models.minimax_h3.src import packing

OUT_DIR = "scanpe_experiments/scanpe_grid_out"

PROMPT = """subject_definitions:

summary:
[reference generation] The target video shows an extremely detailed, ultra-high-resolution photorealistic aerial view of a dense ancient forest canopy with a winding river, scattered waterfalls, and distant mountains, captured in crisp golden-hour light.

retention_analysis:

detailed_description:
The image is an ultra-high-resolution photorealistic aerial landscape photograph, golden hour lighting, extremely crisp fine detail on foliage, water, and rock texture throughout the entire frame.
[Shot 1] A dense ancient forest canopy stretches across a vast aerial view, a winding river catching golden light, scattered waterfalls cascading down rocky outcrops, distant mountains on the horizon under a clear warm sky, extreme fine detail on every tree and rock.

overall_soundscape:
N/A

non_diegetic_music:
N/A
"""

ROWS, COLS = 3, 4  # 12 tiles -> t_lat=12 -> num_frames = 39 (17*2+5)
NUM_FRAMES = 39
TILE = 512
STRIDE = float(packing._ROPE_SPATIAL_SCALE)
STEPS = 40
SHIFT = 3.0


def snake_order(rows, cols):
    order = []
    for r in range(rows):
        col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in col_range:
            order.append((r, c))
    return order


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
    n_tiles = ROWS * COLS
    assert t_lat == n_tiles, f"t_lat={t_lat} != n_tiles={n_tiles}, pick a matching NUM_FRAMES"
    print(f"t_lat = {t_lat}, grid = {ROWS}x{COLS}")

    positions = snake_order(ROWS, COLS)
    offsets = torch.tensor(
        [(r * STRIDE, c * STRIDE) for r, c in positions], dtype=torch.float64
    )

    generator = torch.Generator(device="cpu").manual_seed(42)
    result = pipeline(
        conditional_embeds=conditional_embeds,
        unconditional_embeds=None,
        height=TILE,
        width=TILE,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        generator=generator,
        ref_images=None,
        with_audio=False,
        sigma_shift=SHIFT,
        scan_offsets=offsets,
    )
    video = result["video"].numpy()  # (T_pixel, H, W, C), T_pixel > t_lat (VAE temporal upsample)
    t_pixel = video.shape[0]
    print(f"decoded {t_pixel} pixel frames for {t_lat} latent tiles")

    # pick one representative pixel-frame per latent tile (evenly spaced)
    # to fill the grid -- decode upsamples t_lat -> t_pixel temporally, so
    # tile i's anchor sits near pixel-frame round(i * (t_pixel-1) / (t_lat-1))
    canvas = Image.new("RGB", (COLS * TILE, ROWS * TILE))
    for i, (r, c) in enumerate(positions):
        pf = round(i * (t_pixel - 1) / (t_lat - 1))
        tile_img = Image.fromarray(video[pf])
        canvas.paste(tile_img, (c * TILE, r * TILE))
        tile_img.save(os.path.join(OUT_DIR, f"tile_r{r}_c{c}.png"))

    canvas.save(os.path.join(OUT_DIR, "stitched_grid.png"))
    print(f"saved {os.path.join(OUT_DIR, 'stitched_grid.png')}")


if __name__ == "__main__":
    main()

"""Cut each WildPPS wide street-panorama photo (2048x400) into 7 square tiles
for ScanPE continuity training. Each tile is squashed to a fixed square size
so the offset math stays simple (stride = _ROPE_SPATIAL_SCALE exactly tiles
edge-to-edge, same as the validated inference tests).

Run from the ai-toolkit root:
    .venv/bin/python scanpe_experiments/prep_wildpps_tiles.py
"""
import os
import glob

from PIL import Image

SRC_DIR = "scanpe_experiments/WildPPS/leftImg8bit/val"
OUT_DIR = "scanpe_experiments/wildpps_tiles"
N_TILES = 7
TILE_SIZE = 192  # /32 divisible, matches CANVAS_MULTIPLE; kept small to fit
# training activations on a single 5090 (no grad checkpointing -- see train script)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(SRC_DIR, "*", "*.png")))
    print(f"found {len(paths)} source photos")

    n_ok = 0
    for path in paths:
        city = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        tile_w = w // N_TILES
        if tile_w < 16:
            continue
        for i in range(N_TILES):
            left = i * tile_w
            right = w if i == N_TILES - 1 else (i + 1) * tile_w
            tile = img.crop((left, 0, right, h)).resize(
                (TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS
            )
            tile.save(os.path.join(OUT_DIR, f"{city}_t{i}.png"))
        n_ok += 1

    print(f"prepared {n_ok} sequences x {N_TILES} tiles -> {OUT_DIR}")


if __name__ == "__main__":
    main()

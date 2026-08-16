import os
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = "scanpe_experiments/scanpe_out"
OUT_PATH = "scanpe_experiments/scanpe_grid.png"

FRAME_IDXS = [0, 3, 7, 11, 15, 18, 21]
ROWS = [("baseline_stationary", "stationary (baseline)"), ("scanpe_linear", "ScanPE linear (untrained)")]

THUMB = 220
LABEL_H = 30
PAD = 4

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    font = ImageFont.load_default()

grid_w = len(FRAME_IDXS) * (THUMB + PAD) + PAD
grid_h = len(ROWS) * (THUMB + LABEL_H + PAD) + PAD
grid = Image.new("RGB", (grid_w, grid_h), "white")
draw = ImageDraw.Draw(grid)

for r, (prefix, row_label) in enumerate(ROWS):
    for c, idx in enumerate(FRAME_IDXS):
        path = os.path.join(SRC_DIR, f"{prefix}_f{idx}.png")
        x = PAD + c * (THUMB + PAD)
        y = PAD + r * (THUMB + LABEL_H + PAD)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize((THUMB, THUMB))
            grid.paste(img, (x, y))
        draw.text((x + 2, y + THUMB + 2), f"f{idx}", fill="black", font=font)
    draw.text((PAD, PAD + r * (THUMB + LABEL_H + PAD) - 2), row_label, fill="red", font=font)

grid.save(OUT_PATH)
print(f"saved {OUT_PATH}")

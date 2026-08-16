import os
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = "scanpe_experiments/shift_sweep_out2"
OUT_PATH = "scanpe_experiments/shift_sweep_grid2.png"

ROWS = [("nf1", "stylized, t_lat=1")]
COLS = ["12.0", "3.0", "1.5"]

THUMB = 384
LABEL_H = 36
PAD = 6

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()

grid_w = len(COLS) * (THUMB + PAD) + PAD
grid_h = len(ROWS) * (THUMB + LABEL_H + PAD) + PAD
grid = Image.new("RGB", (grid_w, grid_h), "white")
draw = ImageDraw.Draw(grid)

for r, (prefix, row_label) in enumerate(ROWS):
    for c, shift in enumerate(COLS):
        path = os.path.join(SRC_DIR, f"{prefix}_shift{shift}.png")
        x = PAD + c * (THUMB + PAD)
        y = PAD + r * (THUMB + LABEL_H + PAD)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize((THUMB, THUMB))
            grid.paste(img, (x, y))
        label = f"{row_label}  shift={shift}"
        draw.text((x + 4, y + THUMB + 4), label, fill="black", font=font)

grid.save(OUT_PATH)
print(f"saved {OUT_PATH}")

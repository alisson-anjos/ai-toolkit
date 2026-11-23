"""Contour seeds of every guide video, for attention-routed LoRA training (the same contour reading as the ComfyUI node
BFS H3 Regional Reference Attention, guide_seeds): per color, per frame, the filled area of a thin closed line of that
color, stored at 1/8 of the frame size as <group>/seeds/<stem>.npz (beside motion/, so no loader takes it for a control)
    colors: int array of the color indexes present (0 = red = Picture 1, 1 = blue = Picture 2, ...)
    fills:  uint8 [len(colors), frames, H/8, W/8] (0..255 coverage)
usage: python extensions_built_in/diffusion_models/minimax_h3/tools/precompute_guide_seeds.py MOTION_DIR [MOTION_DIR ...]
MOTION_DIR is the folder of marked guide videos (<group>/motion); targets are <group>/targets/<stem>.mp4."""
import sys
from pathlib import Path

import cv2
import numpy as np

COLORS = np.array([(235, 40, 40), (35, 110, 235), (60, 205, 60), (240, 205, 30), (220, 50, 220), (30, 205, 225)], np.float32) / 255.0


def frame_fills(rgb):
    """rgb float [H, W, 3] in 0..1 -> {color: filled mask float [H, W]}"""
    out = {}
    H, W = rgb.shape[:2]
    for c in range(len(COLORS)):
        line = (np.abs(rgb - COLORS[c]).max(-1) < 0.12).astype(np.uint8)
        if line.sum() < 40:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(line, np.ones((3, 3), np.uint8)), connectivity=8)
        best = None
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 40:
                continue
            ys, xs = np.nonzero(lab == i)
            hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
            area = cv2.contourArea(hull)
            if area < 400 or stats[i, cv2.CC_STAT_AREA] / area > 0.55:
                continue
            if best is None or area > best[0]:
                best = (area, hull)
        if best is not None:
            m = np.zeros((H, W), np.float32); cv2.fillPoly(m, [best[1]], 1.0)
            out[c] = m
    return out


def seeds_of(video):
    cap = cv2.VideoCapture(str(video)); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        return None
    H, W = frames[0].shape[:2]; h8, w8 = max(1, H // 8), max(1, W // 8)
    per = [frame_fills(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0) for f in frames]
    colors = sorted({c for p in per for c in p})
    fills = np.zeros((len(colors), len(frames), h8, w8), np.uint8)
    for fi, p in enumerate(per):
        for ci, c in enumerate(colors):
            if c in p:
                fills[ci, fi] = (cv2.resize(p[c], (w8, h8), interpolation=cv2.INTER_AREA) * 255).round().astype(np.uint8)
    return np.array(colors, np.int64), fills


def main():
    for d in sys.argv[1:]:
        vids = sorted(Path(d).glob("*.mp4")); done = 0
        for v in vids:
            out = v.parent.parent / "seeds" / f"{v.stem}.npz"; out.parent.mkdir(exist_ok=True)
            if out.exists():
                continue
            r = seeds_of(v)
            if r is None:
                print("unreadable", v, flush=True); continue
            np.savez_compressed(out, colors=r[0], fills=r[1]); done += 1
        print(d, len(vids), "videos,", done, "new", flush=True)


if __name__ == "__main__":
    main()

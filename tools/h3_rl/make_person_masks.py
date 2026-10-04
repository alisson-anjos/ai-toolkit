"""Person masks for H3 RL / scene-loss datasets: <targets>/_person_masks/<name>.npy, uint8 [F,H,W] (255 = person),
long side 128, one per frame. Used by the background / lighting rewards and by train.scene_loss.

    python tools/h3_rl/make_person_masks.py /path/to/dataset/targets [more folders] [--model yolov8m-seg.pt]
"""
import argparse, glob, os
import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("folders", nargs="+")
ap.add_argument("--model", default="yolov8m-seg.pt", help="YOLO person segmentation weights (downloaded if a bare name)")
ap.add_argument("--device", default=None)
a = ap.parse_args()
seg = YOLO(a.model)
for folder in a.folders:
    os.makedirs(f"{folder}/_person_masks", exist_ok=True)
    n = 0
    for v in sorted(glob.glob(f"{folder}/*.mp4")):
        dst = f"{folder}/_person_masks/{os.path.splitext(os.path.basename(v))[0]}.npy"
        if os.path.exists(dst):
            continue
        cap, out = cv2.VideoCapture(v), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            H, W = f.shape[:2]
            s = 128 / max(H, W)
            r = seg(f, verbose=False, conf=0.3, classes=[0], device=a.device)[0]
            m = np.zeros((H, W), np.uint8)
            if r.masks is not None:
                for mm in r.masks.data.cpu().numpy():
                    m |= cv2.resize((mm > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            out.append(cv2.resize(m * 255, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA))
        cap.release()
        if out:
            np.save(dst, np.stack(out))
            n += 1
    print(folder, "masks", n, flush=True)

"""FROZEN copy of the reward worker that trained RL v1 steps 0-200 (paired / ground-truth mode only: id, bg, light, pose).
Use it with train.bfs_nft.reward_worker: bfs_reward_worker_v1.py to reproduce that run exactly. Model paths are local defaults:
edit INSIGHTFACE_ROOT / POSE / SEG below for another machine.

Reward worker for BFS GRPO (runs in a python that has insightface + cv2, e.g. the ComfyUI venv; CPU only).

Protocol: one JSON request per stdin line {"npz": path}, one JSON reply per stdout line.
The npz holds gen uint8 [G,F,H,W,3] (the group's videos), tgt uint8 [F,H,W,3] (the real target video) and,
optionally, mask uint8 [F,H,W] (the target person, 1 = person).

Rewards per generated video (higher is better):
  id     ArcFace cosine between the generated faces and the target's faces (identity transfer)
  bg     PSNR/40 outside the dilated person mask against the target (background preservation)
  light  minus the low-frequency Lab error inside the person mask against the target (lighting / colour)
  pose   keypoint similarity (OKS, YOLOv8-pose) between the generated person and the target person (motion transfer)
"""
import json
import os
import sys

import cv2
import numpy as np


def _arcface():
    import insightface
    fa = insightface.app.FaceAnalysis(name="buffalo_l", root=os.environ.get("BFS_INSIGHTFACE_ROOT") or os.path.expanduser("~/.insightface"),
                                      allowed_modules=["detection", "recognition"],
                                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    return fa


FA = None
POSE = None


def keypoints(frame_rgb):
    """(17,2) xy, (17,) conf and the box scale of the biggest person, or None."""
    r = POSE(frame_rgb[..., ::-1].copy(), verbose=False, device="cpu", conf=0.25)[0]
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return None
    b = r.boxes.xyxy.cpu().numpy()
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    xy = r.keypoints.xy[i].cpu().numpy()
    cf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
    area = max(1.0, float((b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])))
    return xy, cf, area


def oks(a, b, k=0.04):
    """Object keypoint similarity of b against reference a, over the keypoints both see."""
    if a is None or b is None:
        return None
    (xa, ca, area), (xb, cb, _) = a, b
    vis = (ca > 0.3) & (cb > 0.3)
    if vis.sum() < 3:
        return None
    d2 = ((xa[vis] - xb[vis]) ** 2).sum(1)
    return float(np.exp(-d2 / (2 * area * k ** 2)).mean())


def face_embed(frame_rgb):
    fs = FA.get(frame_rgb[..., ::-1].copy())
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    e = f.normed_embedding
    return e / (np.linalg.norm(e) + 1e-8)


def rewards(gen, tgt, mask, n_frames=6):
    G, F = gen.shape[:2]
    pick = np.unique(np.linspace(0, F - 1, min(n_frames, F)).round().astype(int))
    t_emb = [e for e in (face_embed(tgt[i]) for i in pick) if e is not None]
    t_emb = np.mean(t_emb, 0) if t_emb else None
    H, W = tgt.shape[1:3]
    k = max(3, int(min(H, W) * 0.04) | 1)
    if mask is not None:
        m = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in mask]) > 0
        grown = np.stack([cv2.dilate(x.astype(np.uint8), np.ones((k, k), np.uint8)) for x in m]) > 0
    blur = max(3, int(min(H, W) / 12) | 1)
    t_lab = np.stack([cv2.cvtColor(cv2.GaussianBlur(tgt[i], (blur, blur), 0), cv2.COLOR_RGB2LAB) for i in pick]).astype(np.float32)
    t_kp = [keypoints(tgt[i]) for i in pick]
    out = {"id": [], "bg": [], "light": [], "pose": []}
    for g in range(G):
        e = [x for x in (face_embed(gen[g, i]) for i in pick) if x is not None]
        out["id"].append(float(np.median([x @ t_emb for x in e])) if (e and t_emb is not None) else (0.0 if t_emb is not None else None))
        if mask is not None:
            bgm = ~grown
            diff = (gen[g].astype(np.float32) - tgt.astype(np.float32)) ** 2
            mse = float(diff[bgm].mean()) if bgm.any() else None
            out["bg"].append(None if mse is None else min(50.0, 10 * np.log10(255.0 ** 2 / max(mse, 1e-6))) / 40.0)
            sel = m[pick]
        else:
            out["bg"].append(None)
            sel = np.ones((len(pick), H, W), bool)
        g_lab = np.stack([cv2.cvtColor(cv2.GaussianBlur(gen[g, i], (blur, blur), 0), cv2.COLOR_RGB2LAB) for i in pick]).astype(np.float32)
        sims = [x for x in (oks(t_kp[j], keypoints(gen[g, i])) for j, i in enumerate(pick)) if x is not None]
        out["pose"].append(float(np.mean(sims)) if sims else (0.0 if any(k is not None for k in t_kp) else None))
        out["light"].append(-float(np.abs(g_lab - t_lab)[sel].mean()) / 20.0 if sel.any() else None)
    return out


def main():
    global FA, POSE
    FA = _arcface()
    from ultralytics import YOLO
    POSE = YOLO(os.environ.get("BFS_POSE_MODEL") or "yolov8m-pose.pt")
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            d = np.load(req["npz"])
            res = rewards(d["gen"], d["tgt"], d["mask"] if "mask" in d.files else None)
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

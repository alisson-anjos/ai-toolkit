"""Reward worker for BFS GRPO (runs in a python that has insightface + cv2, e.g. the ComfyUI venv; CPU only).

Protocol: one JSON request per stdin line {"npz": path}, one JSON reply per stdout line.
The npz holds gen uint8 [G,F,H,W,3] (the group's videos), tgt uint8 [F,H,W,3] (the real target video) and,
optionally, mask uint8 [F,H,W] (the target person, 1 = person).

Rewards per generated video (higher is better):
  id     ArcFace cosine between the generated faces and the target's faces (identity transfer)
  bg     PSNR/40 outside the dilated person mask against the target (background preservation)
  light  minus the low-frequency Lab error inside the person mask against the target (lighting / colour)
  pose   keypoint similarity (OKS, YOLOv8-pose) between the generated person and the target person (motion transfer)

Source mode (no ground truth: the npz also holds `ref` uint8 [H,W,3], the picture of the NEW person, and `tgt` is the
ORIGINAL input video whose person is being replaced):
  id     ArcFace median cosine to the reference picture, minus a penalty when the face still looks like the original
         person (copying the input is the obvious hack)
  bg     PSNR/40 against the input outside the dilated union of the original and the generated person masks
  pose   OKS against the original person
  light  face shading layout (16x16 low-frequency luminance, correlation) and brightness against the original
         person's face, who was lit by the scene itself

Both modes:
  char   character identity for ANY subject (people from behind, anime, creatures): DINOv2 cosine between the
         generated subject crop and the reference picture (source mode) or the target's subject crop (paired mode);
         computed only when the run gives it a weight
  lips   lip sync: correlation over time of the mouth opening (inner-lip height / width from the 68-point landmarks)
         between the generated and the original face, plus an amplitude match; None when the original does not speak
"""
import json
import os
import sys

import cv2
import numpy as np

# model locations (BFS_REWARD_MODELS, a JSON dict set by the trainer from train.bfs_nft.reward_models); unset entries
# fall back to the defaults below, and a bare YOLO name ("yolov8m-pose.pt") is downloaded by ultralytics
MODELS = {"insightface_root": os.path.expanduser("~/.insightface"), "pose": "yolov8m-pose.pt", "seg": "yolov8m-seg.pt",
          "dino": "facebook/dinov2-base"}
MODELS.update({k: v for k, v in json.loads(os.environ.get("BFS_REWARD_MODELS") or "{}").items() if v})
# rewards with a weight in the run (BFS_REWARD_KEYS); optional ones (char) are only computed when asked for
KEYS = set(json.loads(os.environ.get("BFS_REWARD_KEYS") or "[]"))


def _arcface():
    import insightface
    fa = insightface.app.FaceAnalysis(name="buffalo_l", root=MODELS["insightface_root"],
                                      allowed_modules=["detection", "recognition", "landmark_3d_68"],
                                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    return fa


def _landmarks():
    import insightface
    fl = insightface.app.FaceAnalysis(name="buffalo_l", root=MODELS["insightface_root"],
                                      allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"])
    fl.prepare(ctx_id=-1, det_size=(384, 384))
    return fl


DEV = "cpu"   # YOLO (torch) goes to the GPU when one is visible; InsightFace (onnx) stays on the CPU
FA = None
POSE = None
SEG = None
FL = None   # face detection + 68-point 3D landmarks (lip sync)


_FACES = {}


def faces(frame_rgb, model=None):
    """InsightFace results for a frame, cached per request (the same frame serves identity, light and lips)."""
    base = (frame_rgb.__array_interface__["data"][0], frame_rgb.shape, frame_rgb.strides)
    if model is FL and ("full",) + base in _FACES:       # a full analysis of this frame already has the landmarks
        return _FACES[("full",) + base]
    k = ("lite" if model is FL else "full",) + base
    if k not in _FACES:
        _FACES[k] = (model or FA).get(frame_rgb[..., ::-1].copy())
    return _FACES[k]


def mouth_open(frame_rgb):
    fs = faces(frame_rgb, FL)
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    p = getattr(f, "landmark_3d_68", None)
    if p is None or (f.bbox[3] - f.bbox[1]) < 24:
        return None
    p = p[:, :2]
    width = np.linalg.norm(p[60] - p[64]) + 1e-6
    return float((np.linalg.norm(p[61] - p[67]) + np.linalg.norm(p[62] - p[66]) + np.linalg.norm(p[63] - p[65])) / 3 / width)


def lip_series(frames, step=2):
    return np.array([np.nan if (m := mouth_open(f)) is None else m for f in frames[::step]])


def lips_reward(src_series, gen_frames):
    """Correlation of the mouth-opening curves (+ amplitude); None when the original barely moves its mouth."""
    a = src_series
    ok_a = ~np.isnan(a)
    if ok_a.sum() < 5 or np.nanstd(a) < 0.03:
        return None
    b = lip_series(gen_frames)
    ok = ok_a & ~np.isnan(b)
    if ok.sum() < 5:
        return -0.5                                   # the face (or its mouth) is lost
    x, y = a[ok], b[ok]
    corr = float(np.corrcoef(x, y)[0, 1]) if y.std() > 1e-4 else 0.0
    amp = abs(np.log((y.std() + 1e-3) / (x.std() + 1e-3)))
    return corr - 0.3 * min(amp, 2.0)


DINO = None


def _dino():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    m = AutoModel.from_pretrained(MODELS["dino"]).to(DEV).eval()
    return m, AutoImageProcessor.from_pretrained(MODELS["dino"])


def dino_embed(images):
    """L2-normalised DINOv2 CLS embeddings of RGB uint8 crops."""
    import torch
    global DINO
    if DINO is None:
        DINO = _dino()
    m, proc = DINO
    with torch.no_grad():
        x = proc(images=[np.ascontiguousarray(i) for i in images], return_tensors="pt")["pixel_values"].to(DEV)
        e = m(pixel_values=x).last_hidden_state[:, 0].float()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def subject_crop(frame, mask=None, pad=0.08):
    """Crop around the subject: the given mask, else YOLO people, else the whole frame."""
    m = mask if mask is not None and mask.any() else person_mask(frame)
    if not m.any():
        return frame
    ys, xs = np.where(m)
    H, W = frame.shape[:2]
    py, px = int((ys.max() - ys.min()) * pad) + 2, int((xs.max() - xs.min()) * pad) + 2
    return frame[max(0, ys.min() - py):min(H, ys.max() + py), max(0, xs.min() - px):min(W, xs.max() + px)]


def ref_crop(ref):
    """The reference without its plain (grey / white) background."""
    d = np.abs(ref.astype(np.int16) - np.median(ref.reshape(-1, 3), 0).astype(np.int16)).sum(-1) > 30
    return subject_crop(ref, d, pad=0.03) if d.any() else ref


def char_reward(gen_frames, gen_masks, anchor_emb):
    e = dino_embed([subject_crop(f, m) for f, m in zip(gen_frames, gen_masks)])
    return float(np.median(e @ anchor_emb))


def person_mask(frame_rgb):
    r = SEG(frame_rgb[..., ::-1].copy(), verbose=False, device=DEV, conf=0.3, classes=[0])[0]
    H, W = frame_rgb.shape[:2]
    m = np.zeros((H, W), bool)
    if r.masks is not None:
        for mm in r.masks.data.cpu().numpy():
            m |= cv2.resize(mm, (W, H)) > 0.5
    return m


def face_light(frame):
    """(16x16 normalised shading pattern, mean luminance) of the biggest face, or None."""
    fs = faces(frame)
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, int(f.bbox[0])), max(0, int(f.bbox[1])), min(W, int(f.bbox[2])), min(H, int(f.bbox[3]))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    L = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    L = cv2.resize(cv2.GaussianBlur(L, (0, 0), max(1.0, (x1 - x0) / 12)), (16, 16))
    return (L - L.mean()) / (L.std() + 1e-3), float(L.mean())


def light_vs_source(gen_frames, src_frames, margin=0.3):
    sc = []
    for g, s in zip(gen_frames, src_frames):
        a, b = face_light(s), face_light(g)
        if a is None or b is None:
            continue
        ratio = abs(np.log((b[1] + 1) / (a[1] + 1)))     # brightness; the margin leaves room for skin tone
        sc.append(float((a[0] * b[0]).mean()) - max(0.0, ratio - margin) * 2.0)
    return float(np.mean(sc)) if sc else None


def rewards_source(gen, src, mask, ref, n_frames=6):
    G, F = gen.shape[:2]
    pick = np.unique(np.linspace(0, F - 1, min(n_frames, F)).round().astype(int))
    r_emb = face_embed(ref)
    s_emb = [e for e in (face_embed(src[i]) for i in pick) if e is not None]
    s_emb = np.mean(s_emb, 0) / (np.linalg.norm(np.mean(s_emb, 0)) + 1e-8) if s_emb else None
    H, W = src.shape[1:3]
    k = max(3, int(min(H, W) * 0.05) | 1)
    src_m = (np.stack([cv2.resize(mask[i], (W, H), interpolation=cv2.INTER_NEAREST) for i in pick]) > 0) if mask is not None \
        else np.stack([person_mask(src[i]) for i in pick])
    s_kp = [keypoints(src[i]) for i in pick]
    s_lips = lip_series(src)
    r_char = dino_embed([ref_crop(ref)])[0] if "char" in KEYS else None
    out = {"id": [], "bg": [], "light": [], "pose": [], "lips": [], "copy": [], "char": []}
    for g in range(G):
        e = [x for x in (face_embed(gen[g, i]) for i in pick) if x is not None]
        if r_emb is None:
            out["id"].append(None); out["copy"].append(None)
        elif not e:
            out["id"].append(0.0); out["copy"].append(None)
        else:
            to_ref = float(np.median([x @ r_emb for x in e]))
            to_src = float(np.median([x @ s_emb for x in e])) if s_emb is not None else 0.0
            out["id"].append(to_ref - 0.5 * max(0.0, to_src - 0.25))
            out["copy"].append(to_src)
        gm = np.stack([person_mask(gen[g, i]) for i in pick])
        out["char"].append(char_reward(gen[g, pick], [a | b for a, b in zip(src_m, gm)], r_char) if r_char is not None else None)
        union = np.stack([cv2.dilate((a | b).astype(np.uint8), np.ones((k, k), np.uint8)) for a, b in zip(src_m, gm)]) > 0
        bgm = ~union
        diff = (gen[g, pick].astype(np.float32) - src[pick].astype(np.float32)) ** 2
        mse = float(diff[bgm].mean()) if bgm.any() else None
        out["bg"].append(None if mse is None else min(50.0, 10 * np.log10(255.0 ** 2 / max(mse, 1e-6))) / 40.0)
        sims = [x for x in (oks(s_kp[j], keypoints(gen[g, i])) for j, i in enumerate(pick)) if x is not None]
        out["pose"].append(float(np.mean(sims)) if sims else (0.0 if any(x is not None for x in s_kp) else None))
        out["light"].append(light_vs_source(gen[g, pick], src[pick]))
        out["lips"].append(lips_reward(s_lips, gen[g]))
        if out["copy"][-1] is not None and out["copy"][-1] > 0.45:
            # still the original person: a copy of the input scores perfectly on scene, light and pose, so it gets
            # the floor there instead (otherwise three rewards out of four would reward the hack)
            out["bg"][-1], out["light"][-1], out["pose"][-1] = 0.0, -1.0, 0.0
            out["lips"][-1] = None if out["lips"][-1] is None else -1.0
            out["char"][-1] = None if out["char"][-1] is None else -1.0
    return out


def keypoints(frame_rgb):
    """(17,2) xy, (17,) conf and the box scale of the biggest person, or None."""
    r = POSE(frame_rgb[..., ::-1].copy(), verbose=False, device=DEV, conf=0.25)[0]
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
    fs = faces(frame_rgb)
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
    t_lips = lip_series(tgt)
    t_char = None
    if "char" in KEYS:
        tm = m[pick] if mask is not None else [None] * len(pick)
        t_char = dino_embed([subject_crop(tgt[i], x) for i, x in zip(pick, tm)])
        t_char = t_char.mean(0) / (np.linalg.norm(t_char.mean(0)) + 1e-8)
    out = {"id": [], "bg": [], "light": [], "pose": [], "lips": [], "char": []}
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
        out["lips"].append(lips_reward(t_lips, gen[g]))
        out["char"].append(char_reward(gen[g, pick], list(m[pick]) if mask is not None else [None] * len(pick), t_char)
                           if t_char is not None else None)
    return out


def main():
    global FA, POSE, SEG, FL, DEV
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    FA = _arcface()
    FL = _landmarks()
    from ultralytics import YOLO
    POSE = YOLO(MODELS["pose"])
    SEG = YOLO(MODELS["seg"])
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            _FACES.clear()
            d = np.load(req["npz"])
            mask = d["mask"] if "mask" in d.files else None
            res = (rewards_source(d["gen"], d["tgt"], mask, d["ref"]) if "ref" in d.files
                   else rewards(d["gen"], d["tgt"], mask))
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

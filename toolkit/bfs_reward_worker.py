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
  gpt_id, gpt_q  a vision LLM judge (OpenAI, default gpt-5-mini): ONE call per group with one picture = the reference X and
         the candidates A..F side by side (2 frames each); identity (face, hair, apparent gender / age, build vs X) and
         quality (natural scene lighting, not pasted, no artefacts), 0-10 -> 0-1. Needs `judge_ref` in the npz and a key
         (OPENAI_API_KEY or the file in BFS_OPENAI_KEY_FILE). Only computed when weighted; None when the call fails.
  char   character identity for ANY subject (people from behind, anime, creatures): DINOv2 cosine between the
         generated subject crop and the reference picture (source mode) or the target's subject crop (paired mode);
         computed only when the run gives it a weight
  wpose  whole-body pose (ViTPose-L whole-body, 133 keypoints: body + feet, both hands with fingers, face): OKS per region
         against the target / input frame by frame, each region normalised by its own size, body 0.5 / hands 0.3 /
         face 0.2 (regions missing in either side are left out). Model: reward_models.vitpose (ONNX, GPU via the CUDA
         execution provider). Only computed when weighted.
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
VIT = None


def _vitpose():
    import onnxruntime as ort
    path = MODELS.get("vitpose") or "vitpose-l-wholebody.onnx"
    return ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def _person_box(frame_rgb):
    r = POSE(frame_rgb[..., ::-1].copy(), verbose=False, device=DEV, conf=0.25)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    b = r.boxes.xyxy.cpu().numpy()
    return b[int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))]


def wholebody(frames):
    """[(133,2) xy, (133,) conf] or None per frame (biggest person), one batched ViTPose pass."""
    global VIT
    if VIT is None:
        VIT = _vitpose()
    crops, metas = [], []
    for f in frames:
        b = _person_box(f)
        if b is None:
            metas.append(None); continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        w, h = (b[2] - b[0]) * 1.25, (b[3] - b[1]) * 1.25
        if w / h > 0.75: h = w / 0.75
        else: w = h * 0.75
        src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]])
        M = cv2.getAffineTransform(src, np.float32([[0, 0], [192, 0], [0, 256]]))
        c = cv2.warpAffine(f, M, (192, 256), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        c = (c - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
        crops.append(c.transpose(2, 0, 1)); metas.append((cx - w / 2, cy - h / 2, w, h))
    if not crops:
        return [None] * len(frames)
    hm = VIT.run(None, {VIT.get_inputs()[0].name: np.stack(crops).astype(np.float32)})[0]
    out, k = [], 0
    for m in metas:
        if m is None:
            out.append(None); continue
        H = hm[k]; k += 1
        J, hh, ww = H.shape
        idx = H.reshape(J, -1).argmax(1)
        conf = H.reshape(J, -1).max(1)
        ys, xs = idx // ww, idx % ww
        x0, y0, w, h = m
        xy = np.stack([x0 + (xs + 0.5) / ww * w, y0 + (ys + 0.5) / hh * h], 1)
        out.append((xy, conf))
    return out


WB_REGIONS = {"body": (list(range(0, 23)), 0.5), "hands": (list(range(91, 133)), 0.3), "face": (list(range(23, 91)), 0.2)}


def _region_oks(a, b, idx, k=0.06, thr=0.3):
    (xa, ca), (xb, cb) = a, b
    idx = np.array(idx)
    vis = (ca[idx] > thr) & (cb[idx] > thr)
    if vis.sum() < 3:
        return None
    pa, pb = xa[idx][vis], xb[idx][vis]
    span = np.ptp(xa[idx][ca[idx] > thr], axis=0) if (ca[idx] > thr).sum() > 1 else np.array([1.0, 1.0])
    area = max(1.0, float(span[0] * span[1]))
    return float(np.exp(-((pa - pb) ** 2).sum(1) / (2 * area * k ** 2)).mean())


def wpose_score(ref_kp, gen_kp):
    """Mean over frames of the region-weighted OKS (body / hands / face), None when nothing comparable."""
    vals = []
    for a, b in zip(ref_kp, gen_kp):
        if a is None or b is None:
            continue
        parts = [(w, _region_oks(a, b, idx)) for idx, w in WB_REGIONS.values()]
        parts = [(w, v) for w, v in parts if v is not None]
        if parts:
            vals.append(sum(w * v for w, v in parts) / sum(w for w, _ in parts))
    return float(np.mean(vals)) if vals else None

JUDGE_MODEL = os.environ.get("BFS_JUDGE_MODEL") or "gpt-5-mini"
JUDGE_Q = ("You judge AI character swaps. Column X is the REFERENCE character who must appear in the video. Columns {labels} "
           "are candidate videos of the same scene (three frames each: start, middle, end, top to bottom). Judge every frame: a "
           "candidate that drifts, misaligns or deforms in any frame scores lower. For EACH candidate give "
           "integer scores 0-10: identity = how clearly it is the SAME character as X (face, hair length / style / colour, "
           "apparent gender and age, body build; a candidate that keeps another person's face or hair scores low); quality = "
           "natural result (lit by the scene, not pasted or flat, no artefacts, no distortion). Compare the candidates with "
           "each other and use the full range so that the differences are visible. Reply with JSON only: "
           "{{\"A\": {{\"identity\": n, \"quality\": n}}, ...}}")


def _judge_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    kf = os.environ.get("BFS_OPENAI_KEY_FILE") or os.path.expanduser("~/.config/bfs_openai_key")
    if not key and os.path.exists(kf):
        key = open(kf).read().strip()
    return OpenAI(api_key=key) if key else None


def judge(ref, gen, frames=(0.05, 0.5, 0.95), h=200, calls=2):
    """[(identity, quality) in 0-1 or (None, None)] per candidate: `calls` vision-LLM calls, each with the candidates in a
    different random order (un-shuffled and averaged), so the judge's position bias cancels instead of becoming signal."""
    import random
    G = gen.shape[0]
    acc = [[[], []] for _ in range(G)]
    for c in range(calls):
        order = list(range(G)); random.shuffle(order)
        r = _judge_once(ref, gen[order], frames, h)
        for slot, g in enumerate(order):
            a, b = r[slot]
            if a is not None: acc[g][0].append(a); acc[g][1].append(b)
    return [(float(np.mean(a)), float(np.mean(b))) if a else (None, None) for a, b in acc]


def _judge_once(ref, gen, frames, h):
    import base64
    G, F = gen.shape[:2]
    try:
        client = _judge_client()
        if client is None:
            return [(None, None)] * G
        cols, labels = [], [chr(65 + g) for g in range(G)]
        rs = cv2.resize(ref, (int(ref.shape[1] * len(frames) * h / ref.shape[0]), len(frames) * h))
        cols.append(rs)
        for g in range(G):
            fr = [gen[g, int((F - 1) * t)] for t in frames]
            fr = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in fr]
            cols.append(np.vstack(fr))
        heads = []
        for lab, c in zip(["X"] + labels, cols):
            band = np.full((36, c.shape[1], 3), 255, np.uint8)
            cv2.putText(band, lab, (c.shape[1] // 2 - 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            heads.append(np.vstack([band, c, np.full((c.shape[0] * 0 + 0, c.shape[1], 3), 255, np.uint8)]))
        H = max(x.shape[0] for x in heads)
        sheet = np.hstack([np.vstack([x, np.full((H - x.shape[0], x.shape[1], 3), 255, np.uint8)]) for x in heads])
        sheet = np.hstack([np.hstack([c, np.full((H, 8, 3), 255, np.uint8)]) for c in [sheet]])
        url = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
        r = client.responses.create(model=JUDGE_MODEL, reasoning={"effort": "low"}, input=[{"role": "user", "content": [
            {"type": "input_text", "text": JUDGE_Q.format(labels=", ".join(labels))},
            {"type": "input_image", "image_url": url, "detail": "high"}]}])
        t = r.output_text
        d = json.loads(t[t.find("{"):t.rfind("}") + 1])
        return [(float(d[l]["identity"]) / 10.0, float(d[l]["quality"]) / 10.0) if l in d else (None, None) for l in labels]
    except Exception as exc:  # noqa: BLE001 - no judge this step
        print(f"judge failed: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr, flush=True)
        return [(None, None)] * G


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
    s_wb = wholebody([src[i] for i in pick]) if "wpose" in KEYS else None
    s_lips = lip_series(src) if (not KEYS or "lips" in KEYS) else None
    r_char = dino_embed([ref_crop(ref)])[0] if "char" in KEYS else None
    out = {"id": [], "bg": [], "light": [], "pose": [], "lips": [], "copy": [], "char": [], "wpose": []}
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
        out["lips"].append(lips_reward(s_lips, gen[g]) if s_lips is not None else None)
        out["wpose"].append(wpose_score(s_wb, wholebody([gen[g, i] for i in pick])) if s_wb is not None else None)
        if out["copy"][-1] is not None and out["copy"][-1] > 0.45:
            # still the original person: a copy of the input scores perfectly on scene, light and pose, so it gets
            # the floor there instead (otherwise three rewards out of four would reward the hack)
            out["bg"][-1], out["light"][-1], out["pose"][-1] = 0.0, -1.0, 0.0
            out["lips"][-1] = None if out["lips"][-1] is None else -1.0
            out["char"][-1] = None if out["char"][-1] is None else -1.0
            out["wpose"][-1] = None if out["wpose"][-1] is None else 0.0
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


def rewards(gen, tgt, mask, n_frames=10):
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
    t_wb = wholebody([tgt[i] for i in pick]) if "wpose" in KEYS else None
    t_lips = lip_series(tgt) if (not KEYS or "lips" in KEYS) else None
    t_char = None
    if "char" in KEYS:
        tm = m[pick] if mask is not None else [None] * len(pick)
        t_char = dino_embed([subject_crop(tgt[i], x) for i, x in zip(pick, tm)])
        t_char = t_char.mean(0) / (np.linalg.norm(t_char.mean(0)) + 1e-8)
    out = {"id": [], "bg": [], "light": [], "pose": [], "lips": [], "char": [], "wpose": []}
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
        out["lips"].append(lips_reward(t_lips, gen[g]) if t_lips is not None else None)
        out["wpose"].append(wpose_score(t_wb, wholebody([gen[g, i] for i in pick])) if t_wb is not None else None)
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
            gen = d["gen"]
            res = (rewards_source(gen, d["tgt"], mask, d["ref"]) if "ref" in d.files
                   else rewards(gen, d["tgt"], mask))
            if KEYS & {"gpt_id", "gpt_q"}:
                jref = d["judge_ref"] if "judge_ref" in d.files else d["tgt"][len(d["tgt"]) // 2]
                jr = judge(jref, gen)
                res["gpt_id"] = [a for a, _ in jr]
                res["gpt_q"] = [b for _, b in jr]
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

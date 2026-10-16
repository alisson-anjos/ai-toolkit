"""DiffusionNFT (online, forward-process RL) for the H3 body-swap LoRA.

Adapted from verl-omni (https://github.com/verl-project/verl-omni, Apache License 2.0): the DiffusionNFT loss
(verl_omni/trainer/diffusion/diffusion_algos.py, DiffusionNFTLoss.compute_loss), the advantage -> reward_prob mapping
(DiffusionNFTLoss._advantage_to_reward_prob) and the old-policy EMA decay schedules
(verl_omni/trainer/diffusion/diffusion_trainer_utils.py). DiffusionNFT itself: Zheng et al., "DiffusionNFT: Online
Diffusion Reinforcement with Forward Process" (2025). Defaults follow verl-omni's MiniMax-H3 Ref2VA recipe.

Ported from verl-omni's DiffusionNFTLoss so
it runs on one GPU inside ai-toolkit, with our guide latents / references / cached text embeddings untouched.

Per optimizer step (one dataset item = one prompt group):
  1. rollout: `group` videos from the OLD policy (EMA copy of the LoRA), plain Euler ODE on the H3 sigma grid;
  2. reward: decode, score in a separate CPU worker (identity / background / lighting against the real target),
     z-normalise each reward inside the group, weighted sum -> advantage -> clip -> reward_prob in [0, 1];
  3. update: for a random subset of the rollout timesteps, noise each generated x0 with fresh noise and take
     verl-omni's DiffusionNFT loss with the current (grad), old and reference (LoRA off) predictions;
  4. after the optimizer step (next call): old <- decay * old + (1 - decay) * current.

Config (train.bfs_nft):
  group 6, steps 10, train_fraction 0.3, mix_beta 0.1, adv_clip_max 5, ref_kl_coef 1e-4,
  adaptive_weight_min 1e-5, decay_schedule delayed_linear_to_0_999, update_interval 2,
  weights {id: 1, bg: 1, light: 1, pose: 1, lips: 1}, src_datasets [], keep_rollouts 2,
  reward_python (a python with insightface + ultralytics; default: this one),
  reward_models {insightface_root, pose, seg, dino} (defaults: ~/.insightface, yolov8m-pose.pt, yolov8m-seg.pt,
  facebook/dinov2-base); weights may add char (DINOv2 subject identity)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
from contextlib import contextmanager

import numpy as np
import torch

DECAY_SCHEDULES = {"copy": (0, 0.0, 0.0), "linear_to_0_5": (0, 0.001, 0.5), "delayed_linear_to_0_999": (75, 0.0075, 0.999)}
DEFAULTS = {"group": 6, "steps": 10, "train_fraction": 0.3, "mix_beta": 0.1, "adv_clip_max": 5.0, "ref_kl_coef": 1e-4,
            "adaptive_weight_min": 1e-5, "decay_schedule": "delayed_linear_to_0_999", "update_interval": 2,
            "weights": {"id": 1.0, "bg": 1.0, "light": 1.0, "pose": 1.0, "lips": 1.0}, "src_datasets": [], "reward_python": "", "reward_models": {},
            "keep_rollouts": 2}


def old_policy_decay(step: int, schedule: str) -> float:
    warm, rate, top = DECAY_SCHEDULES[schedule]
    return 0.0 if step < warm else min((step - warm) * rate, top)


def nft_loss(forward_pred, old_pred, ref_pred, x0, xt, t, reward_prob, beta, adv_clip_max, ref_kl_coef, weight_min):
    """verl-omni DiffusionNFTLoss.compute_loss (predictions are noise - x0, x0 = xt - t * v)."""
    old_pred, ref_pred = old_pred.detach(), ref_pred.detach()
    dims = tuple(range(1, x0.ndim))
    pos = beta * forward_pred + (1.0 - beta) * old_pred
    neg = (1.0 + beta) * old_pred - beta * forward_pred
    x0_pos = xt - t * pos
    x0_neg = xt - t * neg
    with torch.no_grad():
        w_pos = (x0_pos.double() - x0.double()).abs().mean(dim=dims, keepdim=True).clip(min=weight_min).to(x0_pos.dtype)
        w_neg = (x0_neg.double() - x0.double()).abs().mean(dim=dims, keepdim=True).clip(min=weight_min).to(x0_neg.dtype)
    pos_loss = ((x0_pos - x0) ** 2 / w_pos).mean(dim=dims)
    neg_loss = ((x0_neg - x0) ** 2 / w_neg).mean(dim=dims)
    policy = ((reward_prob * pos_loss / beta) + ((1.0 - reward_prob) * neg_loss / beta)) * adv_clip_max
    kl = ((forward_pred - ref_pred) ** 2).mean(dim=dims)
    return (policy + ref_kl_coef * kl).mean(), {"policy": float(policy.detach().mean()), "kl": float(kl.detach().mean())}


# below these spreads a reward's differences inside a group are measurement noise (ArcFace on tiny faces, ...):
# z-normalising them would turn noise into full-size advantages, so the std is floored at this value
STD_FLOOR = {"expr": 0.05, "id": 0.03, "bg": 0.02, "light": 0.05, "pose": 0.03, "lips": 0.05, "char": 0.02, "gpt_id": 0.05, "gpt_q": 0.05,
             "wpose": 0.02, "layout": 0.02, "cam": 0.05, "cloth": 0.02}


# a rollout the judge says is not the reference character (a copy of the guide person, a half swap) scores almost
# perfectly on everything that follows the guide: those rewards get their floor, whether or not ArcFace found a face
JUDGE_FLOOR = {"bg": 0.0, "light": -1.0, "pose": 0.0, "lips": -1.0, "wpose": 0.0, "expr": -1.0, "cam": 0.0, "layout": 0.0}


def judge_gate(rewards: dict, gate: float | None) -> tuple[dict, int]:
    """Floors the guide-following rewards of rollouts whose gpt_id is under `gate`; returns (rewards, how many)."""
    ids = rewards.get("gpt_id")
    if gate is None or not ids:
        return rewards, 0
    out = {k: list(v) for k, v in rewards.items()}
    n = 0
    for g, v in enumerate(ids):
        if v is not None and float(v) < float(gate):
            n += 1
            for k, fl in JUDGE_FLOOR.items():
                if out.get(k) and out[k][g] is not None:
                    out[k][g] = fl
    return out, n


def advantages(rewards: dict, weights: dict, floors: dict | None = None) -> torch.Tensor:
    """Each reward normalised inside the group by max(std, floor) (missing values -> 0), weighted sum."""
    floors = {**STD_FLOOR, **(floors or {})}
    total = None
    for k, w in weights.items():
        vals = rewards.get(k)
        if not vals or all(v is None for v in vals):
            continue
        x = torch.tensor([np.nan if v is None else float(v) for v in vals], dtype=torch.float64)
        ok = ~torch.isnan(x)
        z = torch.zeros_like(x)
        if ok.sum() > 1:
            sd = max(float(x[ok].std()), float(floors.get(k, 1e-6)))
            z[ok] = (x[ok] - x[ok].mean()) / sd
        total = z * w if total is None else total + z * w
    return total if total is not None else torch.zeros(len(next(iter(rewards.values()))), dtype=torch.float64)


class RewardClient:
    def __init__(self, python: str, models: dict | None = None, weights: dict | None = None):
        import sys
        worker = os.path.join(os.path.dirname(__file__), os.environ.get("BFS_REWARD_WORKER", "bfs_reward_worker.py"))
        # onnxruntime's CUDA provider (ViTPose) needs cuDNN 9 / cuBLAS: use the copies that ship with that python's torch
        import glob as _glob
        venv = os.path.dirname(os.path.dirname(python or sys.executable))
        nv = _glob.glob(os.path.join(venv, "lib", "python*", "site-packages", "nvidia", "*", "lib"))
        ld = ":".join(nv + [os.environ.get("LD_LIBRARY_PATH", "")])
        self.p = subprocess.Popen([python or sys.executable, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1,
                                  env=dict(os.environ, OMP_NUM_THREADS="8", LD_LIBRARY_PATH=ld, BFS_REWARD_MODELS=json.dumps(models or {}),
                                           BFS_INSIGHTFACE_ROOT=(models or {}).get("insightface_root", ""),
                                           BFS_POSE_MODEL=(models or {}).get("pose", ""),
                                           BFS_REWARD_KEYS=json.dumps(sorted(k for k, w in (weights or {}).items() if w))))
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("reward worker died at start")
            if line.strip().startswith("{") and json.loads(line).get("ready"):
                break

    def __call__(self, gen: np.ndarray, tgt: np.ndarray, mask: np.ndarray | None, ref: np.ndarray | None = None,
                 judge_ref: np.ndarray | None = None) -> dict:
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            kw = {"gen": gen, "tgt": tgt}
            if mask is not None:
                kw["mask"] = mask
            if ref is not None:
                kw["ref"] = ref
            if judge_ref is not None:
                kw["judge_ref"] = judge_ref
            np.savez(path, **kw)
            self.p.stdin.write(json.dumps({"npz": path}) + "\n")
            self.p.stdin.flush()
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("reward worker died")
                if line.strip().startswith("{"):
                    out = json.loads(line)
                    break
        finally:
            os.remove(path)
        if "error" in out:
            raise RuntimeError("reward worker: " + out["error"])
        return out


class NFTState:
    def __init__(self, trainer, cfg: dict):
        os.environ["BFS_COPY_FLOOR_LAYOUT"] = "1" if cfg.get("copy_floor_layout") else "0"   # read by the reward worker
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cfg["weights"] = {**DEFAULTS["weights"], **(cfg or {}).get("weights", {})}
        self.params = [p for p in trainer.network.parameters() if p.requires_grad]
        self.old = [p.detach().clone() for p in self.params]
        # kl_anchor "init": the KL reference is the LoRA the run started from (kept on disk so a resume keeps it), not the
        # base model with the LoRA off, so RL refines that LoRA instead of drifting / growing away from it
        self.anchor = None
        if self.cfg.get("kl_anchor") == "init":
            ap = os.path.join(trainer.save_root, "nft_anchor.pt")
            if os.path.exists(ap):
                saved = torch.load(ap, map_location="cpu")
                self.anchor = [t.to(p.device, p.dtype) for t, p in zip(saved, self.params)]
            else:
                self.anchor = [p.detach().clone() for p in self.params]
                os.makedirs(trainer.save_root, exist_ok=True)
                torch.save([t.cpu() for t in self.anchor], ap)
        if self.cfg.get("reward_worker"):          # e.g. a frozen older worker, to reproduce a run exactly
            os.environ["BFS_REWARD_WORKER"] = self.cfg["reward_worker"]
        self.rewards = RewardClient(self.cfg["reward_python"], self.cfg.get("reward_models"), self.cfg["weights"])
        self.calls = 0
        self.log_path = os.path.join(trainer.save_root, "nft_log.jsonl")
        self.pool = None
        if self.cfg.get("ref_pool"):          # a new reference character every step, made while the run trains
            from toolkit.bfs_ref_pool import RefPool
            self.pool = RefPool(self.cfg["ref_pool"])
        self.pool_meta = None
        self.rollout_dir = os.path.join(trainer.save_root, "nft_rollouts")

    def swap_anchor(self):
        for p, o in zip(self.params, self.anchor):
            p.data, o.data = o.data, p.data

    def swap_old(self):
        for p, o in zip(self.params, self.old):
            p.data, o.data = o.data, p.data

    @torch.no_grad()
    def ema_update(self, step: int):
        d = old_policy_decay(step, self.cfg["decay_schedule"])
        for p, o in zip(self.params, self.old):
            o.mul_(d).add_(p.detach(), alpha=1.0 - d)


def _sigmas(trainer, steps: int) -> torch.Tensor:
    from extensions_built_in.diffusion_models.minimax_h3.src.packing import build_sigma_schedule
    shift = getattr(trainer.sd, "video_sigma_shift", None)
    return build_sigma_schedule(steps, shift) if shift is not None else build_sigma_schedule(steps)


def _to_uint8(video: torch.Tensor) -> np.ndarray:
    """(1,3,F,H,W) in [-1,1] -> (F,H,W,3) uint8."""
    v = ((video[0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return v.permute(1, 2, 3, 0).cpu().numpy()


def _pixel_mask(item_path: str, frames: int, size: tuple[int, int]) -> np.ndarray | None:
    import cv2
    d, n = os.path.split(item_path)
    f = os.path.join(d, "_person_masks", os.path.splitext(n)[0] + ".npy")
    if not os.path.exists(f):
        return None
    mk = np.load(f)
    idx = np.linspace(0, mk.shape[0] - 1, frames).round().astype(int)
    return np.stack([cv2.resize(mk[i], (size[1], size[0]), interpolation=cv2.INTER_NEAREST) for i in idx])


def _any_ref(item_path: str) -> np.ndarray | None:
    """<dataset>/refs/<name>.* for any item (the judge's reference picture), or None."""
    import cv2
    root = os.path.dirname(os.path.dirname(os.path.abspath(item_path)))
    stem = os.path.splitext(os.path.basename(item_path))[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(root, "refs", stem + ext)
        if os.path.exists(p):
            img = cv2.imread(p)
            return None if img is None else img[..., ::-1].copy()
    return None


def _source_ref(item_path: str, cfg: dict) -> np.ndarray | None:
    """Source-mode items (a dataset listed in `src_datasets`: the target IS the input video, no ground truth) get their
    reference picture, <dataset>/refs/<name>.(png|jpg), for the reference-based rewards; other items get None."""
    import cv2
    if not any(k in item_path for k in cfg.get("src_datasets", [])):
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(item_path)))
    stem = os.path.splitext(os.path.basename(item_path))[0]
    # refs_reward/ (optional) is the picture the rewards compare against when the model is shown something else in
    # refs/ (a character sheet, a close-up): one plain full-body picture of the same person
    for sub in ("refs_reward", "refs"):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = os.path.join(root, sub, stem + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                return None if img is None else img[..., ::-1].copy()
    return None


def _pool_group(item_path: str, cfg: dict) -> str | None:
    """The ref_pool group of an item (its dataset path contains a key of ref_pool.datasets), or None."""
    rp = cfg.get("ref_pool") or {}
    for frag, group in (rp.get("datasets") or {}).items():
        if frag in item_path:
            return group
    return None


# caption words that assume a human, made generic when the pool character is not a person
GENERIC_CAPTION = [("is the person shown in", "is the character shown in"),
                   ("define the person in the target video", "define the character in the target video"),
                   ("face, hair, skin,", "face, hair or fur, skin,"), ("facial structure, hairstyle, skin,", "facial structure, hair or fur, skin,")]


_META_CACHE: dict[str, dict] = {}


def _clip_desc(item_path: str) -> str:
    """The description of the person in a clip, from <dataset>/meta.json {"<name>": {"desc": ...}} (rl_v2 style)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(item_path)))
    if root not in _META_CACHE:
        f = os.path.join(root, "meta.json")
        try:
            _META_CACHE[root] = json.load(open(f)) if os.path.exists(f) else {}
        except Exception:  # noqa: BLE001
            _META_CACHE[root] = {}
    m = _META_CACHE[root].get(os.path.splitext(os.path.basename(item_path))[0]) or {}
    return str(m.get("desc") or "") if isinstance(m, dict) else ""


def _pool_reference(trainer, st, batch, group: str, dev, dtype):
    """Swap the batch's reference picture for a fresh pool character and re-encode the prompt with it (the cached
    embeds hold the dataset's own picture). Returns (embeds, picture RGB uint8)."""
    rp = st.cfg["ref_pool"]
    img, meta = st.pool.take(group)
    st.pool_meta = {k: meta.get(k) for k in ("category", "who", "file", "reused", "progress") if k in meta}
    fi = batch.file_items[0]
    ref = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
    flags = list(getattr(fi, "aligned_image_guide_flags", []) or [])
    ctl = batch.control_tensor_list
    if ctl is not None and len(ctl) and len(ctl[0]):
        # the reference picture(s) = the image controls that are not aligned guides; the first one becomes the pool
        # character (a second reference, if any, is kept as it is)
        refs_idx = [i for i in range(len(ctl[0])) if not (i < len(flags) and flags[i])]
        if not refs_idx:
            raise ValueError("ref_pool: the item has no reference control to replace")
        ctl[0][refs_idx[0]] = ref.to(ctl[0][refs_idx[0]].device, ctl[0][refs_idx[0]].dtype)
    elif getattr(batch, "control_tensor", None) is not None:
        batch.control_tensor = torch.nn.functional.interpolate(
            ref[None], size=batch.control_tensor.shape[-2:], mode="bilinear", align_corners=False
        ).to(batch.control_tensor.device, batch.control_tensor.dtype)
    else:
        raise ValueError("ref_pool: the item has no control images (needs a reference control path)")
    caption = fi.caption or ""
    who = str(meta.get("who") or "").strip()
    if who and rp.get("caption_template", True):
        # a caption built for this character: who comes in (the pool description) and who leaves (the clip's own
        # description from <dataset>/meta.json "desc", e.g. "the young woman in a white blouse")
        from toolkit.bfs_ref_pool import character_caption
        target = _clip_desc(fi.path) or "the person"
        st.pool_meta["caption"] = caption = character_caption(meta.get("category", "creature"), who, target)
    elif who and rp.get("caption_who", True):
        # name the character itself, as a prompt would: "<Subject 1> is the pale elf with pointed ears shown in ..."
        for art in ("a ", "an "):
            if who.lower().startswith(art):
                who = who[len(art):]
                break
        caption = caption.replace("is the person shown in", f"is the {who} shown in", 1)
    if meta.get("category") != "person" and not rp.get("caption_template", True):
        for a, b in rp.get("caption_replace", GENERIC_CAPTION):
            caption = caption.replace(a, b)
        if meta.get("category") == "anime":
            caption = caption.replace(" The result is a real, photographic video.", "")
    target_size = (fi.crop_width, fi.crop_height) if getattr(fi, "crop_width", None) and getattr(fi, "crop_height", None) else None
    sd = trainer.sd
    with torch.no_grad():
        pe = sd.encode_prompt(caption, control_images=[ref[None].to(dev, dtype)], target_size=target_size)
    if not rp.get("te_on_gpu", True):
        sd.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    return pe.to(dev, dtype=dtype), img


# How the source video reaches the model this step (bfs_nft.guide_modes = {mode: weight}):
#   guide  - aligned guide latent, hidden from the text encoder (how the LoRAs are used with Add Guide)
#   native - the same clip as a native reference video <Video 1> (own clock, seen by the text encoder), no guide
#   both   - the aligned guide AND the clip as <Video 1>
GUIDE_MODES = ("guide", "native", "both")


def native_caption(caption: str) -> str:
    """The caption with the source clip named as <Video 1> (images come first, so it is the first video)."""
    c = caption.replace("The guide video provides the scene and the performance:",
                        "<Video 1> is the guide video: it provides the scene and the performance:")
    return c.replace(" in the guide video", " in <Video 1>").replace(" of the guide video", " of <Video 1>")


def _pick_mode(cfg: dict) -> str:
    w = {k: float(v) for k, v in (cfg.get("guide_modes") or {"guide": 1.0}).items() if k in GUIDE_MODES and float(v) > 0}
    if not w:
        return "guide"
    return random.choices(list(w), weights=list(w.values()))[0]


@contextmanager
def _source_mode(trainer, batch, mode: str, dev, dtype):
    """Rewires this step's source clip for `mode` and yields the prompt embeds to use (None = the cached ones).
    The file item's flags are restored afterwards (items are reused every epoch)."""
    vids = getattr(batch, "control_video_paths_list", None)
    fi = batch.file_items[0]
    flags = list(getattr(fi, "aligned_video_guide_flags", []) or [])
    if mode == "guide" or not vids or not vids[0] or not any(flags):
        yield None
        return
    sd = trainer.sd
    old_vids, old_flags = [list(v) for v in vids], flags
    old_native = set(getattr(sd, "_native_video_ref_paths", set()) or set())
    gpath = vids[0][flags.index(True)]
    paths, new_flags = ([gpath], [False]) if mode == "native" else ([gpath, gpath], [True, False])
    try:
        batch.control_video_paths_list = [paths]
        fi.aligned_video_guide_flags = new_flags
        sd._native_video_ref_paths = {gpath}
        sd._ref_video_dataset_config = batch.dataset_config
        img_flags = list(getattr(fi, "aligned_image_guide_flags", []) or [])
        ctl = batch.control_tensor_list[0] if batch.control_tensor_list else []
        refs = [ctl[i][None] if ctl[i].ndim == 3 else ctl[i] for i in range(len(ctl))
                if not (i < len(img_flags) and img_flags[i])]
        target = (fi.crop_width, fi.crop_height) if getattr(fi, "crop_width", None) and getattr(fi, "crop_height", None) else None
        with torch.no_grad():
            pe = sd.encode_prompt(native_caption(fi.caption or ""),
                                  control_images=[[r.to(dev, dtype) for r in refs] + [gpath]], target_size=target)
        yield pe.to(dev, dtype=dtype)
    finally:
        batch.control_video_paths_list = old_vids
        fi.aligned_video_guide_flags = old_flags
        sd._native_video_ref_paths = old_native


@contextmanager
def _rollout_adapter(trainer, cfg):
    """Enable frozen DMAD only for generation, restoring the training adapter."""
    if not cfg.get("rollout_turbo", False):
        yield
        return
    turbo = getattr(trainer.sd, "nft_turbo_lora", None)
    if turbo is None:
        raise ValueError("bfs_nft.rollout_turbo requires model_kwargs.nft_turbo_lora_path")
    assistant = getattr(trainer.sd, "assistant_lora", None)
    turbo_was = turbo.is_active
    assistant_was = assistant.is_active if assistant is not None else None
    try:
        turbo.is_active = True
        if assistant is not None:
            assistant.is_active = False
        yield
    finally:
        turbo.is_active = turbo_was
        if assistant is not None:
            assistant.is_active = assistant_was


def nft_step(trainer, batch, accum_scale: float = 1.0) -> torch.Tensor:
    from toolkit.train_tools import get_torch_dtype
    st: NFTState = getattr(trainer, "_bfs_nft", None)
    if st is None:
        st = trainer._bfs_nft = NFTState(trainer, trainer.train_config.bfs_nft)
    c = st.cfg
    if st.calls > 0 and st.calls % int(c["update_interval"]) == 0:
        st.ema_update(st.calls)
    st.calls += 1
    dtype = get_torch_dtype(trainer.train_config.dtype)
    dev = trainer.device_torch
    net = trainer.network
    net.multiplier = batch.get_network_weight_list()

    with torch.no_grad():
        batch = trainer.preprocess_batch(batch)
        trainer.process_general_training_batch(batch)
        embeds = batch.prompt_embeds.clone().detach().to(dev, dtype=dtype)
        pool_img, st.pool_meta = None, None
        if st.pool is not None:      # the creature curriculum follows the run's progress
            total = max(1, int(getattr(trainer.train_config, "steps", 1) or 1))
            st.pool.progress = min(1.0, float(getattr(trainer, "step_num", st.calls)) / total)
        group = _pool_group(batch.file_items[0].path, c) if st.pool is not None else None
        if group:
            embeds, pool_img = _pool_reference(trainer, st, batch, group, dev, dtype)
        mode = _pick_mode(c)
        st.mode_ctx = _source_mode(trainer, batch, mode, dev, dtype)
        pe = st.mode_ctx.__enter__()
        if pe is not None:
            embeds = pe
        st.mode = mode if pe is not None else "guide"
        x_target = batch.latents
        x_target = (x_target.tensor if hasattr(x_target, "tensor") else x_target).to(dev, torch.float32)
    shape = tuple(x_target.shape)
    sig = _sigmas(trainer, int(c["steps"])).to(dev, torch.float32)
    rollout_sig = _sigmas(trainer, int(c.get("rollout_steps", c["steps"]))).to(dev, torch.float32)
    G = int(c["group"])
    seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(G)]

    def pred(x, s, seed, grad=False):
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(seed)
            with torch.set_grad_enabled(grad):
                return trainer.predict_noise(noisy_latents=x.to(dev, dtype=dtype),
                                             timesteps=(s * 1000.0).reshape(1).to(dev),
                                             conditional_embeds=embeds, batch=batch).float()

    # 1. rollout with the old policy (plain ODE: DiffusionNFT needs no likelihoods)
    x0s = []
    t_start = time.time()
    trainer.sd.unet.eval() if hasattr(trainer.sd.unet, "eval") else None
    st.swap_old()
    try:
        with _rollout_adapter(trainer, c), torch.no_grad(), net:
            for g in range(G):
                gen = torch.Generator(device="cpu").manual_seed(seeds[g])
                x = torch.randn(shape, generator=gen).to(dev)
                for i in range(len(rollout_sig) - 1):
                    v = pred(x, rollout_sig[i], seeds[g] + i)
                    x = x + (rollout_sig[i + 1] - rollout_sig[i]) * v
                x0s.append(x)
    finally:
        st.swap_old()

    # 2. rewards against the real target
    t_rollout = time.time()
    with torch.no_grad():
        tgt = _to_uint8(trainer.sd.decode_latents(x_target.to(dtype)))
        gens = np.stack([_to_uint8(trainer.sd.decode_latents(x.to(dtype))) for x in x0s])
    item = batch.file_items[0].path
    mask = _pixel_mask(item, tgt.shape[0], tgt.shape[1:3])
    if pool_img is not None:          # the pool character is this step's reference for every reward
        jref = pool_img if any(k in c["weights"] for k in ("gpt_id", "gpt_q")) else None
        r = st.rewards(gens, tgt, mask, pool_img, jref)
    else:
        jref = _any_ref(item) if any(k in c["weights"] for k in ("gpt_id", "gpt_q")) else None
        r = st.rewards(gens, tgt, mask, _source_ref(item, c), jref)
    gate = c.get("judge_gate")
    r, n_gated = judge_gate(r, gate)
    ids = [v for v in (r.get("gpt_id") or []) if v is not None]
    # every rollout failed the swap: no example to move toward, and the group would only rank copies against each other
    skip_group = bool(c.get("skip_failed_groups")) and gate is not None and len(ids) == G and max(ids) < float(gate)
    adv = advantages(r, c["weights"], c.get("std_floor")).clamp(-c["adv_clip_max"], c["adv_clip_max"])
    if skip_group:
        adv = torch.zeros_like(adv)
    reward_prob = (adv / c["adv_clip_max"] / 2.0 + 0.5).float().to(dev)
    if int(c.get("keep_rollouts", 0)) and st.calls % 25 == 1:
        _save_rollouts(st, trainer, gens, tgt, r, adv, batch.file_items[0].path, int(c["keep_rollouts"]), pool_img)

    # 3. DiffusionNFT update on a subset of the rollout timesteps
    t_reward = time.time()
    trainer.sd.unet.train() if hasattr(trainer.sd.unet, "train") else None
    grid = sig[:-1]
    n_t = max(1, int(round(len(grid) * float(c["train_fraction"]))))
    total, logs = 0.0, []
    for g in range(G):
        if skip_group or (adv[g] == 0 and not c["ref_kl_coef"]):
            continue
        for k in random.sample(range(len(grid)), n_t):
            s = grid[k]
            noise = torch.randn_like(x0s[g])
            xt = (1.0 - s) * x0s[g] + s * noise
            seed = seeds[g] * 7 + k
            with torch.no_grad():
                st.swap_old()
                try:
                    with net:
                        old_p = pred(xt, s, seed)
                finally:
                    st.swap_old()
                if st.anchor is not None:
                    st.swap_anchor()
                    try:
                        with net:
                            ref_p = pred(xt, s, seed)
                    finally:
                        st.swap_anchor()
                else:
                    was = net.is_active
                    net.is_active = False
                    try:
                        ref_p = pred(xt, s, seed)
                    finally:
                        net.is_active = was
            # backward INSIDE the network context: with gradient checkpointing the forward is recomputed during
            # backward, and leaving the context first switches the LoRA off for that recompute
            with net:
                cur = pred(xt, s, seed, grad=True)
                loss, m = nft_loss(cur, old_p, ref_p, x0s[g], xt, s.view(1, 1, 1, 1, 1), reward_prob[g:g + 1],
                                   float(c["mix_beta"]), float(c["adv_clip_max"]), float(c["ref_kl_coef"]),
                                   float(c["adaptive_weight_min"]))
                scale = accum_scale / (G * n_t)
                trainer.accelerator.backward(loss * scale)
            total += float(loss) / (G * n_t)
            logs.append(m)

    with open(st.log_path, "a") as f:
        f.write(json.dumps({"call": st.calls, "sampling_steps": int(c.get("rollout_steps", c["steps"])),
                            "training_grid_steps": int(c["steps"]),
                            "rollout_turbo": bool(c.get("rollout_turbo", False)), "item": os.path.basename(batch.file_items[0].path), "rewards": r,
                            "adv": [round(float(a), 3) for a in adv], "loss": total,
                            "policy": float(np.mean([x["policy"] for x in logs])) if logs else 0.0,
                            "kl": float(np.mean([x["kl"] for x in logs])) if logs else 0.0,
                            "judge_gated": n_gated, "skipped": skip_group, "mode": st.mode, "group": G, "seeds": seeds,
                            "frames": int(shape[2]) if len(shape) == 5 else None,
                            "secs": {"rollout": round(t_rollout - t_start, 1), "reward": round(t_reward - t_rollout, 1),
                                     "update": round(time.time() - t_reward, 1)},
                            **({"pool_ref": st.pool_meta} if st.pool_meta else {})}) + "\n")
    st.mode_ctx.__exit__(None, None, None)
    return torch.tensor(total, device=dev)


def _save_rollouts(st, trainer, gens, tgt, r, adv, item, keep, ref=None):
    """Best and worst rollout of the group next to the target (and the pool character), for eyeballing what the
    reward prefers."""
    import cv2
    os.makedirs(st.rollout_dir, exist_ok=True)
    order = np.argsort(np.asarray(adv))
    picks = [order[-1], order[0]][:keep]
    rows = [tgt] + [gens[i] for i in picks]
    if ref is not None:             # the step's pool character, held as a still column
        h, w = tgt.shape[1:3]
        s_ = min(h / ref.shape[0], w / ref.shape[1])
        im = cv2.resize(ref, (max(1, int(ref.shape[1] * s_)), max(1, int(ref.shape[0] * s_))))
        col = np.full((h, w, 3), 128, np.uint8)
        y0, x0 = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
        col[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        rows = [np.repeat(col[None], tgt.shape[0], 0)] + rows
    F = min(x.shape[0] for x in rows)
    path = os.path.join(st.rollout_dir, f"{st.calls:05d}_{os.path.splitext(os.path.basename(item))[0]}.mp4")
    H, W = rows[0].shape[1:3]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (W * len(rows), H))
    for i in range(F):
        vw.write(np.concatenate([x[i] for x in rows], 1)[..., ::-1].copy())
    vw.release()

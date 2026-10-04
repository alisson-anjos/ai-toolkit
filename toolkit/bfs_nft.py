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
STD_FLOOR = {"id": 0.03, "bg": 0.02, "light": 0.05, "pose": 0.03, "lips": 0.05, "char": 0.02}


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
        worker = os.path.join(os.path.dirname(__file__), "bfs_reward_worker.py")
        self.p = subprocess.Popen([python or sys.executable, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1,
                                  env=dict(os.environ, OMP_NUM_THREADS="8", BFS_REWARD_MODELS=json.dumps(models or {}),
                                           BFS_REWARD_KEYS=json.dumps(sorted(k for k, w in (weights or {}).items() if w))))
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("reward worker died at start")
            if line.strip().startswith("{") and json.loads(line).get("ready"):
                break

    def __call__(self, gen: np.ndarray, tgt: np.ndarray, mask: np.ndarray | None, ref: np.ndarray | None = None) -> dict:
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            kw = {"gen": gen, "tgt": tgt}
            if mask is not None:
                kw["mask"] = mask
            if ref is not None:
                kw["ref"] = ref
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
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cfg["weights"] = {**DEFAULTS["weights"], **(cfg or {}).get("weights", {})}
        self.params = [p for p in trainer.network.parameters() if p.requires_grad]
        self.old = [p.detach().clone() for p in self.params]
        self.rewards = RewardClient(self.cfg["reward_python"], self.cfg.get("reward_models"), self.cfg["weights"])
        self.calls = 0
        self.log_path = os.path.join(trainer.save_root, "nft_log.jsonl")
        self.rollout_dir = os.path.join(trainer.save_root, "nft_rollouts")

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


def _source_ref(item_path: str, cfg: dict) -> np.ndarray | None:
    """Source-mode items (a dataset listed in `src_datasets`: the target IS the input video, no ground truth) get their
    reference picture, <dataset>/refs/<name>.(png|jpg), for the reference-based rewards; other items get None."""
    import cv2
    if not any(k in item_path for k in cfg.get("src_datasets", [])):
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(item_path)))
    stem = os.path.splitext(os.path.basename(item_path))[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(root, "refs", stem + ext)
        if os.path.exists(p):
            img = cv2.imread(p)
            return None if img is None else img[..., ::-1].copy()
    return None


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
        x_target = batch.latents
        x_target = (x_target.tensor if hasattr(x_target, "tensor") else x_target).to(dev, torch.float32)
    shape = tuple(x_target.shape)
    sig = _sigmas(trainer, int(c["steps"])).to(dev, torch.float32)
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
    trainer.sd.unet.eval() if hasattr(trainer.sd.unet, "eval") else None
    st.swap_old()
    try:
        with torch.no_grad(), net:
            for g in range(G):
                gen = torch.Generator(device="cpu").manual_seed(seeds[g])
                x = torch.randn(shape, generator=gen).to(dev)
                for i in range(len(sig) - 1):
                    v = pred(x, sig[i], seeds[g] + i)
                    x = x + (sig[i + 1] - sig[i]) * v
                x0s.append(x)
    finally:
        st.swap_old()

    # 2. rewards against the real target
    with torch.no_grad():
        tgt = _to_uint8(trainer.sd.decode_latents(x_target.to(dtype)))
        gens = np.stack([_to_uint8(trainer.sd.decode_latents(x.to(dtype))) for x in x0s])
    item = batch.file_items[0].path
    mask = _pixel_mask(item, tgt.shape[0], tgt.shape[1:3])
    r = st.rewards(gens, tgt, mask, _source_ref(item, c))
    adv = advantages(r, c["weights"], c.get("std_floor")).clamp(-c["adv_clip_max"], c["adv_clip_max"])
    reward_prob = (adv / c["adv_clip_max"] / 2.0 + 0.5).float().to(dev)
    if int(c.get("keep_rollouts", 0)) and st.calls % 25 == 1:
        _save_rollouts(st, trainer, gens, tgt, r, adv, batch.file_items[0].path, int(c["keep_rollouts"]))

    # 3. DiffusionNFT update on a subset of the rollout timesteps
    trainer.sd.unet.train() if hasattr(trainer.sd.unet, "train") else None
    grid = sig[:-1]
    n_t = max(1, int(round(len(grid) * float(c["train_fraction"]))))
    total, logs = 0.0, []
    for g in range(G):
        if adv[g] == 0 and not c["ref_kl_coef"]:
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
        f.write(json.dumps({"call": st.calls, "item": os.path.basename(batch.file_items[0].path), "rewards": r,
                            "adv": [round(float(a), 3) for a in adv], "loss": total,
                            "policy": float(np.mean([x["policy"] for x in logs])) if logs else 0.0,
                            "kl": float(np.mean([x["kl"] for x in logs])) if logs else 0.0}) + "\n")
    return torch.tensor(total, device=dev)


def _save_rollouts(st, trainer, gens, tgt, r, adv, item, keep):
    """Best and worst rollout of the group next to the target, for eyeballing what the reward prefers."""
    import cv2
    os.makedirs(st.rollout_dir, exist_ok=True)
    order = np.argsort(np.asarray(adv))
    picks = [order[-1], order[0]][:keep]
    rows = [tgt] + [gens[i] for i in picks]
    F = min(x.shape[0] for x in rows)
    path = os.path.join(st.rollout_dir, f"{st.calls:05d}_{os.path.splitext(os.path.basename(item))[0]}.mp4")
    H, W = rows[0].shape[1:3]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (W * len(rows), H))
    for i in range(F):
        vw.write(np.concatenate([x[i] for x in rows], 1)[..., ::-1].copy())
    vw.release()

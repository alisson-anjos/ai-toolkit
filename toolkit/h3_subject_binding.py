"""Subject binding for MiniMax H3 (next to TARA): who in the video becomes which Picture, from Molmo2.

For every subject k the data gives Molmo2's own output for the person to replace: its track (t seconds, x, y in
[0, 1], one point per Molmo2 frame) and a vector (the mean hidden state of the tokens Molmo2 generated for that person,
4096-d for Molmo2-8B). Two things enter H3, both added to the video token embeddings right after `video_patch_proj`:

- target video tokens: sum_k M_k(t, y, x) * (slot_k + proj(vector_k)), with M_k a Gaussian around subject k's point
  at that moment (on the latent token grid, interpolated in time);
- the tokens of <Picture k>: slot_k.

So the region where subject k is and the picture it must become carry the same learned tag. `slot` and `proj` start
at zero: a fresh module changes nothing until it is trained. The text prompt never says where anyone is.
"""
from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

VERSION = "subject_binding_v1"


def options(model_kwargs: dict) -> dict:
    result = dict(
        enabled=bool(model_kwargs.get("subject_binding", False)),
        max_subjects=int(model_kwargs.get("subject_binding_max_subjects", 4)),
        sigma=float(model_kwargs.get("subject_binding_sigma_tokens", 1.5)),
        vector_dim=int(model_kwargs.get("subject_binding_vector_dim", 4096)),
    )
    if result["max_subjects"] < 1 or result["sigma"] <= 0 or result["vector_dim"] < 1:
        raise ValueError(f"Invalid subject binding options: {result}")
    return result


@dataclass
class Subject:
    """One subject: its Molmo2 track (n, 3) = (t seconds, x, y) and vector (d,), and which Picture it becomes (1-based,
    in prompt order)."""
    track: torch.Tensor
    vector: Optional[torch.Tensor]
    picture: int


def latent_times(num_latent_frames: int, num_frames: int, fps: float = 24.0) -> torch.Tensor:
    """The moment (seconds) each latent frame stands for: frames spread evenly over the clip."""
    if num_latent_frames <= 1: return torch.zeros(max(1, num_latent_frames))
    return torch.linspace(0, (num_frames - 1) / fps, num_latent_frames)


def point_at(track: torch.Tensor, t: torch.Tensor):
    """(x, y, visible) of a track at times t: linear between Molmo2 points, held at the ends; invisible with no points."""
    if track.numel() == 0:
        z = torch.zeros_like(t); return z, z, torch.zeros_like(t, dtype=torch.bool)
    order = torch.argsort(track[:, 0]); tr = track[order].float()
    ts, xs, ys = tr[:, 0], tr[:, 1], tr[:, 2]
    if ts.numel() == 1:
        return xs.expand_as(t), ys.expand_as(t), torch.ones_like(t, dtype=torch.bool)
    idx = torch.clamp(torch.searchsorted(ts, t), 1, ts.numel() - 1)
    t0, t1 = ts[idx - 1], ts[idx]; w = torch.clamp((t - t0) / (t1 - t0).clamp_min(1e-6), 0, 1)
    return xs[idx - 1] * (1 - w) + xs[idx] * w, ys[idx - 1] * (1 - w) + ys[idx] * w, torch.ones_like(t, dtype=torch.bool)


def maps(subjects: Sequence[Subject], num_latent_frames: int, grid_h: int, grid_w: int, num_frames: int, sigma: float,
         fps: float = 24.0) -> torch.Tensor:
    """(K, T, H, W) Gaussian maps on the target token grid, peak 1 at each subject's point."""
    t = latent_times(num_latent_frames, num_frames, fps)
    yy = (torch.arange(grid_h).float() + .5)[None, :, None]; xx = (torch.arange(grid_w).float() + .5)[None, None, :]
    out = torch.zeros(len(subjects), num_latent_frames, grid_h, grid_w)
    for k, s in enumerate(subjects):
        x, y, vis = point_at(s.track, t)
        cx, cy = (x * grid_w)[:, None, None], (y * grid_h)[:, None, None]
        g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        out[k] = g * vis[:, None, None]
    return out


class SubjectBinding(nn.Module):
    def __init__(self, hidden: int, vector_dim: int = 4096, max_subjects: int = 4):
        super().__init__()
        self.max_subjects = max_subjects
        self.slot = nn.Parameter(torch.zeros(max_subjects, hidden))
        self.proj = nn.Linear(vector_dim, hidden, bias=False)
        nn.init.zeros_(self.proj.weight)

    def video_bias(self, subjects: Sequence[Subject], picture_rows: List[Optional[slice]], num_cond_rows: int,
                   target_grid, num_frames: int, sigma: float, dtype=None, device=None) -> torch.Tensor:
        """(num_cond_rows + T*H*W, hidden) to add to the video embeddings of one sample. picture_rows[p] is the slice of
        condition rows of Picture p+1 (None for a picture that is not in this sample)."""
        T, Hg, Wg = target_grid; hidden = self.slot.shape[1]
        device = device or self.slot.device; dtype = dtype or self.slot.dtype
        bias = torch.zeros(num_cond_rows + T * Hg * Wg, hidden, device=device, dtype=torch.float32)
        if not subjects: return bias.to(dtype)
        if len(subjects) > self.max_subjects: raise ValueError(f"{len(subjects)} subjects, at most {self.max_subjects}")
        m = maps(subjects, T, Hg, Wg, num_frames, sigma).to(device).reshape(len(subjects), -1)       # (K, T*H*W)
        tags = []
        for k, s in enumerate(subjects):
            tag = self.slot[k].float()
            if s.vector is not None: tag = tag + self.proj(s.vector.to(device, self.proj.weight.dtype)).float()
            tags.append(tag)
            rows = picture_rows[s.picture - 1] if 0 < s.picture <= len(picture_rows) else None
            if rows is not None: bias[rows] = bias[rows] + self.slot[k].float()
        bias[num_cond_rows:] = bias[num_cond_rows:] + m.T @ torch.stack(tags)
        return bias.to(dtype)


def save_metadata(cfg: dict) -> dict:
    return {"subject_binding": f"{VERSION}:max_subjects={cfg['max_subjects']}:sigma={cfg['sigma']}:vector_dim={cfg['vector_dim']}"}

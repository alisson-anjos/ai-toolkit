"""Attention-routed training for the face swap LoRA: the same soft attention bias as the ComfyUI node
BFS H3 Regional Reference Attention, applied in the training forward so the LoRA learns to swap under it.

No masks: each person is found by the model's own attention. The guide contours (precomputed per clip by
tools/precompute_guide_seeds.py into <group>/seeds/<stem>.npz) seed every marked person; at layer `route_layer` the value
vectors of the guide rows are tracked frame to frame (label propagation); from that layer on every attention call gets
an exact rank-2P logit bias, appended as extra head dims (q' = [q, qe], k' = [k, ke]):
    target / guide token of person j -> Picture j: +strength, other pictures: -strength
    token of no tracked person (background, unmarked people) -> every picture: -background
    Picture i -> Picture j: -strength; people columns: a person reads little of another person.

Enable with model_kwargs.regional_attention: {strength: 4, background: 4, route_layer: 12, sharpness: 0.05, prob: 1.0}.
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

CTX = None   # the RegionalContext of the current train step (batch size 1), read by MiniMaxH3Attention
STATS = {"steps": 0, "routed": 0, "built": 0, "why": 0}


def _why(msg):
    if STATS["why"] < 8:
        STATS["why"] += 1
        print(f"[H3 regional attention] skipped: {msg}", flush=True)
    return None


def latent_spans(n_lat):
    return [1] + [4] * (n_lat - 1)


def load_seeds(target_path, n_lat, gh, gw, frames):
    """<group>/seeds/<stem>.npz -> (colors, bool seeds [P, n_lat * gh * gw]) on the guide token grid, or None."""
    group = os.path.dirname(os.path.dirname(target_path))
    path = os.path.join(group, "seeds", os.path.splitext(os.path.basename(target_path))[0] + ".npz")
    if not os.path.exists(path):
        return None
    d = np.load(path); colors = d["colors"].tolist(); fills = d["fills"]
    if not colors:
        return None
    fl = torch.from_numpy(fills[:, :frames].astype(np.float32) / 255.0)                 # [P, F, h8, w8]
    P, F_ = fl.shape[:2]
    grid = F.interpolate(fl.reshape(P * F_, 1, *fl.shape[2:]), size=(gh, gw), mode="area").reshape(P, F_, gh, gw)
    out, f = [], 0
    for span in latent_spans(n_lat):
        idx = [min(i, F_ - 1) for i in range(f, f + span)]; f += span
        out.append(grid[:, idx].amax(1))
    return colors, (torch.stack(out, 1) > 0.3).reshape(P, -1)


def track_owners(v, guide_rows, seeds, n_lat, gh, gw, sharpness=0.05, radius=3, topk=8, min_sim=0.3):
    """Label propagation on the model's value vectors (content, no rope), the same as the ComfyUI node.
    Returns soft affinities [P, n_lat * gh * gw]."""
    H, N, D = v.shape[1], len(guide_rows), v.shape[-1]
    f = F.normalize(v[0, :, guide_rows].float(), dim=-1).permute(1, 0, 2).reshape(N, H * D) / H ** 0.5
    f = f.reshape(n_lat, gh, gw, H * D)
    P = seeds.shape[0]; dev = v.device
    sd = seeds.reshape(P, n_lat, gh, gw).to(dev).float()
    anyseed = sd.amax(0)
    near = F.max_pool2d(anyseed[:, None], 5, stride=1, padding=2)[:, 0]
    marked = anyseed.flatten(1).amax(1) > 0
    fixed = torch.zeros(P + 1, n_lat, gh, gw, device=dev)
    fixed[:P] = sd; fixed[P] = (near == 0).float() * marked[:, None, None].float()
    certain = fixed.sum(0) > 0
    k = 2 * radius + 1

    def step(lab_prev, f_prev, f_cur):
        nb_f = F.unfold(f_prev.permute(2, 0, 1)[None], k, padding=radius)[0].reshape(f.shape[-1], k * k, gh * gw)
        nb_l = F.unfold(lab_prev[None], k, padding=radius)[0].reshape(P + 1, k * k, gh * gw)
        sim = torch.einsum("ckn,nc->kn", nb_f, f_cur.reshape(gh * gw, -1))
        sim = sim.masked_fill(~(nb_l.sum(0) > 0), -1.0)
        top, idx = sim.topk(min(topk, k * k), dim=0)
        w = torch.softmax(top / sharpness, 0) * (top > min_sim)
        return (nb_l.gather(1, idx[None].expand(P + 1, -1, -1)) * w[None]).sum(1).reshape(P + 1, gh, gw)

    passes = []
    for order in (range(n_lat), range(n_lat - 1, -1, -1)):
        lab = torch.zeros(P + 1, n_lat, gh, gw, device=dev); prev = None
        for t in order:
            cur = fixed[:, t].clone() if prev is None else step(lab[:, prev], f[prev], f[t])
            lab[:, t] = torch.where(certain[t][None], fixed[:, t], cur); prev = t
        passes.append(lab)
    lab = torch.maximum(passes[0], passes[1])
    lab = lab / lab.sum(0, keepdim=True).clamp(min=1.0)
    return lab[:P].reshape(P, -1)


class RegionalContext:
    def __init__(self, seq_len, guide_pos, grid, seeds, pic_pos, text_pos, target_pos, cfg):
        self.S = seq_len; self.guide_pos = guide_pos; self.grid = grid; self.seeds = seeds
        self.pic_pos = pic_pos; self.text_pos = text_pos; self.target_pos = target_pos
        self.route_layer = int(cfg.get("route_layer", 12)); self.cfg = cfg
        self.bias = None; self.missing = []

    @torch.no_grad()
    def build(self, v):
        n_lat, gh, gw = self.grid
        aff = track_owners(v.detach(), self.guide_pos, self.seeds, n_lat, gh, gw, float(self.cfg.get("sharpness", 0.05)))
        self.bias = soft_bias(self, aff, v.shape[-1], v.device, v.dtype)
        self.shares = [round(float((aff[j] > 0.5).float().mean()) * 100, 1) for j in range(aff.shape[0])]
        STATS["built"] += 1
        if STATS["built"] % 25 == 1:
            print(f"[H3 regional attention] {STATS['routed']}/{STATS['steps']} steps routed; this one: {aff.shape[0]} people, "
                  f"% of guide tokens {self.shares}", flush=True)


def soft_bias(ctx, aff, head_dim, device, dtype):
    """Index version of the node's soft_bias (isolation 'people')."""
    P = aff.shape[0]; s = float(ctx.cfg.get("strength", 4.0)) * head_dim ** 0.5
    bgs = float(ctx.cfg.get("background", 4.0)) * head_dim ** 0.5
    E = 2 * P
    qe = torch.zeros(ctx.S, E, device=device); ke = torch.zeros(ctx.S, E, device=device)
    for j, pos in enumerate(ctx.pic_pos):
        ke[pos, j] = 1.0; qe[pos, :P] = -s; qe[pos, j] = 0.0
    for j, pos in enumerate(ctx.text_pos[:P]):
        ke[pos, j] = 1.0
    tp, gp = ctx.target_pos, ctx.guide_pos
    n = min(len(tp), aff.shape[1]); a_v = aff[:, :n].T
    tot = a_v.sum(1, keepdim=True).clamp(max=1.0)
    qe[tp[:n], :P] = s * (2 * a_v - tot) - bgs * (1 - tot)
    qe[tp[n:], :P] = -bgs
    a_g = aff[:, :len(gp)].T; tg = a_g.sum(1, keepdim=True).clamp(max=1.0)
    qe[gp[:a_g.shape[0]], :P] = s * (2 * a_g - tg) - bgs * (1 - tg)
    qe[tp[:n], P:] = -s * (1 - a_v); ke[tp[:n], P:] = a_v
    qe[gp[:a_g.shape[0]], P:] = -s * (1 - a_g)
    for j in getattr(ctx, "missing", []):       # never found in the guide: no push towards or away from Picture j
        qe[tp, j] = 0.0; qe[gp, j] = 0.0; qe[:, P + j] = 0.0; ke[:, P + j] = 0.0
    pad = (-(head_dim + E)) % 8
    return F.pad(qe, (0, pad)).to(dtype), F.pad(ke, (0, pad)).to(dtype)


def biased_attention(q, k, v, bias):
    """q, k, v [B, H, S, D] -> [B, H, S, D] with the logit bias qe . ke (scale of the real head dim)."""
    qe, ke = bias; B, H, S, D = q.shape
    q2 = torch.cat([q, qe.to(q.dtype).expand(B, H, S, -1)], -1)
    k2 = torch.cat([k, ke.to(k.dtype).expand(B, H, S, -1)], -1)
    v2 = F.pad(v, (0, qe.shape[-1]))
    return F.scaled_dot_product_attention(q2, k2, v2, scale=D ** -0.5)[..., :D]


def maybe_attend(module, q, k, v, attn_mask):
    """Called by MiniMaxH3Attention with q, k, v [B, H, S, D]; returns the biased output or None (plain attention)."""
    ctx = CTX
    if ctx is None or attn_mask is not None or q.shape[0] != 1 or q.shape[2] != ctx.S:
        if ctx is not None and not ctx.cfg.get("_warned"):
            ctx.cfg["_warned"] = True
            print(f"[H3 regional attention] attention call does not match: mask {attn_mask is not None}, q {tuple(q.shape)}, S {ctx.S}", flush=True)
        return None
    layer = getattr(module, "_ra_layer", -1)
    if layer < 0 and not ctx.cfg.get("_untagged"):
        ctx.cfg["_untagged"] = True
        print("[H3 regional attention] attention module without a layer index (tag_layers missed it)", flush=True)
    if layer < ctx.route_layer:
        return None
    if ctx.bias is None:
        if layer != ctx.route_layer:
            return None
        ctx.build(v)
    return biased_attention(q, k, v, ctx.bias)


def tag_layers(transformer):
    for i, blk in enumerate(transformer.blocks):
        blk.attn._ra_layer = i


def make_context(cfg, layout, ref_blocks, text_tags, file_path, num_frames, video_tag=0):
    """Positions of the guide / picture / text-vision / target rows in the packed sequence of one batch item.
    Cond rows are packed in the ref_blocks order (blocks with no video rows, like the audio guide, take none);
    the guide is the aligned multi-frame block, the pictures are the other blocks in order (Picture 1, 2, ...)."""
    STATS["steps"] += 1
    if random.random() >= float(cfg.get("prob", 1.0)):
        return None
    vid = layout.video_indices; num_cond = layout.num_condition_video_rows
    rows, at = [], 0
    for blk in ref_blocks:
        t, h, w = int(blk[0]), int(blk[1]), int(blk[2]); n = t * (h // 2) * (w // 2)
        rows.append((blk, at, at + n)); at += n
    if at != num_cond:
        return _why(f"cond rows {at} != {num_cond}; blocks {[tuple(b[:5]) for b in ref_blocks]}")
    guide = [r for r in rows if bool(r[0][4]) and int(r[0][0]) > 1]
    if len(guide) != 1:
        return _why(f"{len(guide)} guide blocks; blocks {[tuple(b[:5]) for b in ref_blocks]}")
    gblk, ga, gb = guide[0]
    pics = [vid[a:b] for blk, a, b in rows if (blk, a, b) != guide[0] and b > a]
    n_lat, gh, gw = int(gblk[0]), int(gblk[1]) // 2, int(gblk[2]) // 2
    sd = load_seeds(file_path, n_lat, gh, gw, num_frames)
    if sd is None:
        return _why(f"no seeds for {file_path}")
    colors, found = sd
    if not colors or max(colors) >= len(pics):   # color j = Picture j; a color with no picture: not this data
        return _why(f"colors {colors} for {len(pics)} pictures")
    seeds = torch.zeros(len(pics), found.shape[1], dtype=torch.bool)
    seeds[colors] = found                        # a picture whose color the guide reading missed: unrouted (neutral)
    tags = text_tags.reshape(-1).tolist(); spans, start = [], None
    for i, tv in enumerate(tags + [1]):
        if tv == video_tag and start is None:
            start = i
        elif tv != video_tag and start is not None:
            spans.append((start, i)); start = None
    tidx = layout.text_indices
    text_pos = [tidx[a:b] for a, b in spans]
    STATS["routed"] += 1
    ctx = RegionalContext(int(layout.sequence_length), vid[ga:gb], (n_lat, gh, gw), seeds, pics, text_pos,
                          vid[num_cond:], cfg)
    ctx.missing = [j for j in range(len(pics)) if j not in colors]
    return ctx

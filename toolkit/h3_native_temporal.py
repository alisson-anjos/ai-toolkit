"""TARA-H3: Temporally Aligned Reference Attention for MiniMax H3 Ref2VA.

The motion video stays a *native* H3 reference (its own VAE latent rows, its
own spatial grid). Two things change, and no learned layer is added:

1. Shared clock: the motion video's temporal RoPE coordinate is shifted so its
   first latent frame sits on the generated video's first latent frame.
   Picture references keep their native coordinates.
2. Temporal bias: only when a generated-video query attends to a motion-video
   key, the attention logit is lowered by a bounded penalty that grows with
   their distance in seconds:

       p(dt) = min(cap, strength * (max(0, dt - window) / falloff) ** 2)

Text, picture references, audio and the generated video's self-attention get
no added penalty. On CUDA the bias is evaluated inside FlexAttention from
O(sequence) metadata; no S x S bias tensor is materialized.

Technical report: https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.2/research/tara-h3
"""
from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F

VERSION = "native_temporal_v1"
# H3 packs video and audio on one rotary clock of 40 units per second.
ROPE_UNITS_PER_SECOND = 40.0

_FLEX = None


def options(model_kwargs: dict) -> dict:
    """Validated TARA options from ``model.model_kwargs``."""
    enabled = model_kwargs.get("native_temporal_alignment", False)
    if not isinstance(enabled, bool):
        raise ValueError("native_temporal_alignment must be true or false")
    result = dict(
        enabled=enabled,
        source_channel=model_kwargs.get("native_motion_source_channel", 1),
        strength=model_kwargs.get("native_temporal_strength", 2.0),
        window=model_kwargs.get("native_temporal_window_seconds", 0.2),
        falloff=model_kwargs.get("native_temporal_falloff_seconds", 0.25),
        cap=model_kwargs.get("native_temporal_max_penalty", 6.0),
    )
    channel = result["source_channel"]
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 1:
        raise ValueError("native_motion_source_channel must be a control channel number >= 1")
    for name in ("strength", "window", "falloff", "cap"):
        value = result[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (name == "falloff" and value == 0)):
            raise ValueError(f"Invalid native temporal {name}: {value!r}")
    return result


def save_metadata(cfg: dict) -> dict:
    """LoRA metadata, so consumers can reproduce the training-time policy."""
    return {"native_temporal_alignment": {"version": VERSION, **cfg}}


def reference_order(items, is_video):
    """Native H3 packs pictures before videos; keep the order within each kind."""
    return sorted(items, key=lambda item: bool(is_video(item)))


def align_layout(layout, ref_blocks, source_ids, cfg) -> torch.Tensor:
    """Put the motion video on the target clock; return its packed-row mask.

    ``layout`` is a ``packing.PackedLayout`` (its ``position_ids`` are replaced
    by an edited copy). ``ref_blocks`` are ``(t, h, w[, audio_latents])`` in
    packing order and ``source_ids`` the control channel of each block.
    Exactly one block must come from ``cfg['source_channel']`` and be a video.
    """
    mask = torch.zeros(layout.sequence_length, dtype=torch.bool)
    if not cfg["enabled"] or not ref_blocks:
        return mask
    if len(source_ids) != len(ref_blocks):
        raise ValueError("TARA needs one control source channel per reference block")
    target_idx = layout.video_indices[layout.num_condition_video_rows:]
    target_origin = layout.position_ids[int(target_idx[0]), 0].clone()
    cursor = int(layout.text_indices.numel())
    found = 0
    for block, source in zip(ref_blocks, source_ids):
        t, h, w = (int(x) for x in block[:3])
        audio_latents = int(block[3]) if len(block) > 3 else 0
        cursor += audio_latents * 2  # a reference soundtrack packs before its video rows
        rows = t * (h // 2) * (w // 2)
        if source == cfg["source_channel"]:
            if t <= 1:
                raise ValueError("The TARA motion channel must hold a video, not a picture")
            found += 1
            mask[cursor:cursor + rows] = True
            position_ids = layout.position_ids.clone()
            position_ids[cursor:cursor + rows, 0] += target_origin - position_ids[cursor, 0]
            layout.position_ids = position_ids
            # never silently stretch one performance onto a different duration
            src_times = torch.unique_consecutive(position_ids[cursor:cursor + rows, 0])
            dst_times = torch.unique_consecutive(position_ids[target_idx, 0])
            if src_times.shape != dst_times.shape or not torch.allclose(src_times, dst_times, atol=1e-9, rtol=0):
                raise ValueError("TARA motion video and target must share the same temporal grid "
                                 f"({src_times.numel()} vs {dst_times.numel()} latent frames); for samples, "
                                 "use a num_frames no longer than the motion video")
        cursor += rows
    if found != 1:
        raise ValueError(f"TARA expects exactly one motion video in control channel "
                         f"{cfg['source_channel']}, found {found}")
    return mask


def pad_motion_masks(layouts, masks) -> torch.Tensor:
    """Batch the per-item masks the way ``pad_layouts_to_batch`` pads text."""
    max_text = max(int(layout.text_indices.numel()) for layout in layouts)
    total = max_text + layouts[0].sequence_length - int(layouts[0].text_indices.numel())
    result = torch.zeros(len(layouts), total, dtype=torch.bool)
    for i, (layout, mask) in enumerate(zip(layouts, masks)):
        n = int(layout.text_indices.numel())
        result[i, :n] = mask[:n]
        result[i, max_text:] = mask[n:]
    return result


@dataclass(frozen=True)
class TemporalContext:
    times: torch.Tensor   # (B, S) seconds on the shared clock
    source: torch.Tensor  # (B, S) motion-video key rows
    target: torch.Tensor  # (B, S) generated-video query rows
    live: torch.Tensor    # (B, S) non-padding rows
    policy: torch.Tensor  # [strength, window, falloff, cap]


def context(position_ids, motion_mask, token_tags, video_indices, target_grid, cfg) -> Optional[TemporalContext]:
    if not cfg or not cfg.get("enabled") or motion_mask is None or not bool(motion_mask.any()):
        return None
    if motion_mask.shape != token_tags.shape:
        raise ValueError("TARA motion mask must match the packed token rows")
    target = torch.zeros_like(motion_mask)
    target[:, video_indices[-math.prod(target_grid):]] = True
    policy = torch.tensor([cfg["strength"], cfg["window"], cfg["falloff"], cfg["cap"]],
                          device=position_ids.device, dtype=torch.float32)
    return TemporalContext(position_ids[:, :, 0].float() / ROPE_UNITS_PER_SECOND,
                           motion_mask, target, token_tags >= 0, policy)


def penalty(dt: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
    return torch.minimum(policy[0] * (torch.relu(dt - policy[1]) / policy[2]).square(), policy[3])


def score_modifier(ctx: TemporalContext):
    times, source, target, live, policy = ctx.times, ctx.source, ctx.target, ctx.live, ctx.policy
    last = times.shape[1] - 1

    def modify(score, b, h, q, k):
        qi = q.clamp(max=last)
        ki = k.clamp(max=last)
        edited = torch.where(target[b, qi] & source[b, ki],
                             score - penalty((times[b, qi] - times[b, ki]).abs(), policy), score)
        return torch.where(live[b, ki], edited, float("-inf"))

    return modify


def dense_reference(q, k, v, ctx: TemporalContext):
    """Exact dense oracle for tests and CPU; refuses production lengths."""
    if q.shape[-2] > 4096:
        raise ValueError("The dense TARA reference is only meant for short test sequences")
    delta = (ctx.times[:, :, None] - ctx.times[:, None, :]).abs()
    bias = torch.where(ctx.target[:, :, None] & ctx.source[:, None, :], -penalty(delta, ctx.policy.to(delta.device)), 0.0)
    bias = bias.masked_fill(~ctx.live[:, None, :], float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias[:, None].to(q.dtype))


def attention(q, k, v, ctx: TemporalContext):
    """q, k, v: (B, heads, S, head_dim)."""
    global _FLEX
    if q.device.type != "cuda":
        return dense_reference(q, k, v, ctx)
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=True, fullgraph=True)
    return _FLEX(q, k, v, score_mod=score_modifier(ctx))

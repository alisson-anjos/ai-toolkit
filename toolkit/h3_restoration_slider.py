"""Positive restoration slider objective in ai-toolkit H3 flow convention.

This is supervised residual distillation, not a text-only Concept Slider.
Zero preserves the frozen model; positive scales approach the clean flow target.
Negative scales are deliberately not trained or advertised as degradation.
"""
import math
import torch


def residual_target(base, clean_flow, scale, residual_cap=1.0):
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('slider scale must be finite and in [0, 1]')
    if not math.isfinite(residual_cap) or residual_cap <= 0:
        raise ValueError('residual_cap must be finite and positive')
    base, clean_flow = base.detach().float(), clean_flow.detach().float()
    if base.shape != clean_flow.shape:
        raise ValueError('base and clean flow target must have equal shapes')
    delta = clean_flow - base
    # Bound the correction per item, preserving direction. This is a guard
    # against huge teacher errors, not a guarantee of perceptual preservation.
    dims = tuple(range(1, delta.ndim))
    rms = delta.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    delta = delta * (residual_cap / rms).clamp(max=1)
    return base + scale * delta

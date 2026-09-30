"""Experimental H3 reference geometry and source phase; shared with BFSNodes.

Coordinates use H3's normalized RoPE units, not LTX pixel patch bounds.
Source phase v1 composes theta**(-d/half_dim) with the original rotary angles.
"""
import math
import torch

PHASE_VERSION = 'h3_source_phase_v1'


def validate_options(guide_layout='overlap', reference_layout='native', source_phase=False, phase_scale=1.0, sidecar_margin=0.0):
    if guide_layout not in ('overlap', 'sidecar'):
        raise ValueError('guide_rope_layout must be overlap or sidecar')
    if reference_layout not in ('native', 'overlap', 'sidecar'):
        raise ValueError('reference_rope_layout must be native, overlap or sidecar')
    if not isinstance(source_phase, bool):
        raise ValueError('reference_source_phase must be boolean')
    for name, value in [('reference_phase_scale', phase_scale), ('reference_sidecar_margin', sidecar_margin)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    return dict(guide_rope_layout=guide_layout, reference_rope_layout=reference_layout,
                reference_source_phase=source_phase, reference_phase_scale=float(phase_scale),
                reference_sidecar_margin=float(sidecar_margin))


def options_from_kwargs(kwargs):
    return validate_options(kwargs.get('guide_rope_layout', 'overlap'), kwargs.get('reference_rope_layout', 'native'),
                            kwargs.get('reference_source_phase', False), kwargs.get('reference_phase_scale', 1.0),
                            kwargs.get('reference_sidecar_margin', 0.0))


def shift_reference(positions, target, layout, margin=0.0):
    if layout in ('native', 'overlap'):
        return positions
    if layout != 'sidecar' or not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid sidecar layout or margin')
    result=positions.clone()
    # Place beyond the final target patch origin, separated by one patch step.
    widths=torch.unique(target[:,2],sorted=True)
    step=float(widths[1]-widths[0]) if widths.numel()>1 else 1.0
    result[:,2]+=target[:,2].max()+step+margin-positions[:,2].min()
    result[:,1]+=(target[:,1].min()+target[:,1].max()-positions[:,1].min()-positions[:,1].max())/2
    return result


def add_source_phase(angles, source_values, theta=10000.0):
    """Angles [..., S, 2*D], duplicated halves; zero sources stay bit-identical."""
    if source_values is None:
        return angles
    if tuple(source_values.shape)!=tuple(angles.shape[:-1]):
        raise ValueError('Source phase rows must match rotary rows')
    if not torch.isfinite(source_values).all() or (source_values<0).any():
        raise ValueError('Source phase values must be finite and nonnegative')
    if not torch.count_nonzero(source_values):
        return angles
    half=angles.shape[-1]//2
    rates=theta**(-torch.arange(half,device=angles.device,dtype=torch.float32)/half)
    phase=source_values.to(device=angles.device,dtype=torch.float32)[...,None]*rates
    return angles+torch.cat((phase,phase),dim=-1)

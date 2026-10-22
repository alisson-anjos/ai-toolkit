"""Native-reference temporal alignment v1. No new learned model weights.

Only source channel 1's VIDEO latent rows are moved onto the target clock.
Image-reference spatial/temporal positions remain native. Target-to-motion
attention receives a bounded, soft temporal penalty; all other edges stay dense.
The CUDA implementation uses FlexAttention, never a quadratic dense bias tensor.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

VERSION = 'native_temporal_v1'
_FLEX = None

def options(kwargs):
    enabled = kwargs.get('native_temporal_alignment', False)
    if not isinstance(enabled, bool):raise ValueError('native_temporal_alignment must be boolean')
    result = dict(enabled=enabled, source_channel=kwargs.get('native_motion_source_channel',1),
                  strength=kwargs.get('native_temporal_strength',2.0),
                  window=kwargs.get('native_temporal_window_seconds',0.2),
                  falloff=kwargs.get('native_temporal_falloff_seconds',0.25),
                  cap=kwargs.get('native_temporal_max_penalty',6.0))
    if isinstance(result['source_channel'],bool) or not isinstance(result['source_channel'],int) or result['source_channel']<1:raise ValueError('Invalid native motion source channel')
    for name in ['strength','window','falloff','cap']:
        x=result[name]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 or (name=='falloff' and x==0):raise ValueError('Invalid native temporal '+name)
    if enabled and (kwargs.get('align_video_refs') or kwargs.get('align_image_refs') or kwargs.get('reference_source_phase') or kwargs.get('reference_rope_layout','native')!='native'):
        raise ValueError('Native temporal mode requires native references, no AddGuide alignment or source-phase patch')
    return result

def align_layout(layout, ref_blocks, source_ids, cfg):
    """Return a sequence-row mask, aligning only the explicit motion video's clock.

    Native video latents remain in ref_blocks and keep their own spatial grid.
    Reference audio is not temporally biased by this video-only experiment.
    """
    mask=torch.zeros(layout.sequence_length,dtype=torch.bool)
    if not cfg['enabled'] or not ref_blocks:return mask
    if len(source_ids)!=len(ref_blocks):raise ValueError('Native temporal mode requires exact reference source IDs')
    cursor=int(layout.text_indices.numel());found=0
    # target rows are always the final packed rows; condition video indices precede them.
    target_start=int(layout.audio_indices[-1])+1 if layout.audio_indices.numel() else layout.sequence_length
    # For zero audio use the count of conditioning video rows to locate the target.
    target_idx=layout.video_indices[layout.num_condition_video_rows:]
    target_start=int(target_idx[0]);target_origin=layout.position_ids[target_start,0].clone()
    for block, source in zip(ref_blocks,source_ids):
        t,h,w=block[:3];a=block[3] if len(block)>3 else 0
        cursor+=int(a)*2
        rows=int(t)*(int(h)//2)*(int(w)//2)
        if source==cfg['source_channel']:
            if t<=1:raise ValueError('Motion channel must be a video, not an image')
            if len(block)>4 and block[4]:raise ValueError('Native motion video was accidentally converted to aligned guide')
            found+=1;mask[cursor:cursor+rows]=True
            layout.position_ids=layout.position_ids.clone()
            layout.position_ids[cursor:cursor+rows,0]+=target_origin-layout.position_ids[cursor,0].clone()
            # A common temporal extent is required; never silently stretch motion to fit.
            src_times=torch.unique_consecutive(layout.position_ids[cursor:cursor+rows,0])
            dst_times=torch.unique_consecutive(layout.position_ids[target_idx,0])
            if src_times.shape!=dst_times.shape or not torch.allclose(src_times,dst_times,atol=1e-9,rtol=0):raise ValueError('Native motion and target must use identical temporal grids; resample both to matching frame windows')
        cursor+=rows
    if found!=1:raise ValueError(f'Expected one motion video in channel {cfg["source_channel"]}, found {found}')
    assert not mask[target_idx].any()
    return mask

def pad_motion_masks(layouts, masks):
    max_text=max(int(x.text_indices.numel()) for x in layouts)
    total=max_text+layouts[0].sequence_length-int(layouts[0].text_indices.numel())
    result=torch.zeros(len(layouts),total,dtype=torch.bool)
    for i,(layout,mask) in enumerate(zip(layouts,masks)):
        n=int(layout.text_indices.numel());result[i,:n]=mask[:n];result[i,max_text:]=mask[n:]
    return result

@dataclass(frozen=True)
class TemporalContext:
    times: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    live: torch.Tensor
    strength: float
    window: float
    falloff: float
    cap: float
    policy: torch.Tensor | None = None

def context(position_ids, source_mask, tags, video_indices, target_grid, cfg):
    if not cfg['enabled'] or source_mask is None or not bool(source_mask.any()):return None
    if source_mask.shape!=tags.shape:raise ValueError('Motion mask must match packed token rows')
    target=torch.zeros_like(source_mask);n=math.prod(target_grid)
    target[:,video_indices[-n:]]=True
    return TemporalContext(position_ids[:,:,0].float()/40.0,source_mask,target,tags>=0,
                           float(cfg['strength']),float(cfg['window']),float(cfg['falloff']),float(cfg['cap']),
                           torch.tensor([cfg['strength'],cfg['window'],cfg['falloff'],cfg['cap']],device=position_ids.device,dtype=torch.float32))

def score_modifier(ctx):
    # Captured tensors are O(sequence length). No SxS matrix is materialized on CUDA.
    times,source,target,live=ctx.times,ctx.source,ctx.target,ctx.live
    policy=ctx.policy if ctx.policy is not None else torch.tensor([ctx.strength,ctx.window,ctx.falloff,ctx.cap],device=times.device,dtype=torch.float32)
    def modify(score,b,h,q,k):
        qi=q.clamp(max=times.shape[1]-1);ki=k.clamp(max=times.shape[1]-1)
        dt=(times[b,qi]-times[b,ki]).abs()
        penalty=torch.minimum(policy[0]*(torch.relu(dt-policy[1])/policy[2]).square(),policy[3])
        edited=torch.where(target[b,qi]&source[b,ki],score-penalty,score)
        return torch.where(live[b,ki],edited,float('-inf'))
    return modify

def dense_reference(q,k,v,ctx):
    """Tiny test oracle only; forbidden at production sequence lengths."""
    if q.shape[-2]>256:raise ValueError('Dense temporal reference is only for tiny tests')
    delta=(ctx.times[:,:,None]-ctx.times[:,None,:]).abs()
    penalty=(ctx.strength*(F.relu(delta-ctx.window)/ctx.falloff).square()).clamp(max=ctx.cap)
    bias=torch.where(ctx.target[:,:,None]&ctx.source[:,None,:],-penalty,0.0)
    bias=bias.masked_fill(~ctx.live[:,None,:],float('-inf'))
    return F.scaled_dot_product_attention(q,k,v,attn_mask=bias[:,None].to(q.dtype))

def attention(q,k,v,ctx):
    global _FLEX
    if q.device.type!='cuda':return dense_reference(q,k,v,ctx)
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX=torch.compile(flex_attention,dynamic=True,fullgraph=True)
    return _FLEX(q,k,v,score_mod=score_modifier(ctx))

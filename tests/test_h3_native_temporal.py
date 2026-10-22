import pytest,torch
from toolkit import h3_native_temporal as nt
from minimax_h3_test_utils import packing

def layout(text=4,frames=7,blocks=((1,4,8,0,False),(7,8,4,0,False)),ids=(2,1)):
    cfg=nt.options({'native_temporal_alignment':True})
    l=packing.build_packed_sequence(torch.ones(text,dtype=torch.long),frames,8,8,0,ref_blocks=blocks,ref_source_ids=ids)
    before=l.position_ids.clone();mask=nt.align_layout(l,blocks,ids,cfg)
    return l,mask,before

def test_only_motion_time_changes_and_identity_positions_stay_native():
    l,m,before=layout();assert m.sum()==7*4*2
    assert torch.equal(l.position_ids[~m],before[~m])
    assert torch.equal(l.position_ids[m,1:],before[m,1:])
    src=torch.unique_consecutive(l.position_ids[m,0]);dst=torch.unique_consecutive(l.position_ids[l.video_indices[l.num_condition_video_rows:],0])
    torch.testing.assert_close(src,dst,rtol=0,atol=1e-9)

def test_motion_source_id_is_not_packed_image_order():
    l,m,b=layout(blocks=((7,8,4,0,False),(1,4,8,0,False)),ids=(1,2))
    assert m[4] and not m[-1]

def test_mismatched_temporal_lengths_are_rejected():
    with pytest.raises(ValueError,match='temporal grids'):layout(blocks=((6,8,8,0,False),),ids=(1,))

def test_missing_motion_channel_is_rejected():
    with pytest.raises(ValueError,match='Expected one motion'):layout(ids=(2,3))

def test_image_is_not_accepted_as_motion():
    with pytest.raises(ValueError,match='video'):layout(blocks=((1,8,8,0,False),),ids=(1,))

def test_padding_preserves_exact_row_ownership():
    a,ma,_=layout(text=4);b,mb,_=layout(text=8)
    masks=nt.pad_motion_masks([a,b],[ma,mb]);assert not masks[0,4:8].any()
    assert torch.equal(masks[0,8:],ma[4:]) and torch.equal(masks[1],mb)

def ctx(device='cpu',strength=2):
    times=torch.arange(64,device=device).float()[None]/24
    src=torch.zeros(1,64,dtype=torch.bool,device=device);src[:,8:32]=True
    tgt=torch.zeros_like(src);tgt[:,32:]=True
    live=torch.ones_like(src);live[:,-1]=False
    return nt.TemporalContext(times,src,tgt,live,strength,.2,.25,6.)

def test_bias_leaves_identity_and_non_target_queries_unmodified():
    c=ctx();mod=nt.score_modifier(c);z=torch.tensor(0.);i=lambda x:torch.tensor(x)
    assert mod(z,i(0),i(0),i(33),i(2))==0
    assert mod(z,i(0),i(0),i(10),i(30))==0
    assert mod(z,i(0),i(0),i(33),i(30))==0
    assert mod(z,i(0),i(0),i(50),i(8))==-6
    assert torch.isneginf(mod(z,i(0),i(0),i(33),i(63)))

def test_zero_bias_is_the_unmodified_padded_attention():
    torch.manual_seed(1);q,k,v=[torch.randn(1,2,64,32) for _ in range(3)];c=ctx(strength=0)
    actual=nt.dense_reference(q,k,v,c)
    expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=c.live[:,None,None])
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

def test_refuses_quadratic_production_fallback():
    q=torch.zeros(1,1,257,32)
    with pytest.raises(ValueError,match='tiny'):nt.dense_reference(q,q,q,ctx())

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_flex_forward_and_backward_matches_dense_oracle():
    torch.manual_seed(11);c=ctx('cuda')
    q,k,v=[torch.randn(1,2,64,32,device='cuda',requires_grad=True) for _ in range(3)]
    actual=nt.attention(q,k,v,c);expected=nt.dense_reference(q,k,v,c)
    torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-4)
    ga=torch.autograd.grad(actual.square().sum(),(q,k,v))
    ge=torch.autograd.grad(expected.square().sum(),(q,k,v))
    for x,y in zip(ga,ge):torch.testing.assert_close(x,y,rtol=5e-3,atol=5e-4)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_real_transformer_forward_backward_with_checkpointing():
    import importlib.util,sys
    from pathlib import Path
    path=Path(__file__).parents[1]/'extensions_built_in/diffusion_models/minimax_h3/src/transformer.py'
    spec=importlib.util.spec_from_file_location('_h3_native_tiny_transformer',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    p=mod.MiniMaxH3TransformerParams(hidden_size=64,num_layers=2,token_refiner_num_layers=1,
        num_attention_heads=2,attention_head_dim=32,ffn_hidden_size=128,latents_dim=4,audio_latents_dim=4,
        text_dim=32,timestep_input_dim=32,time_embed_hidden_size=64,time_embed_dim=64,rope_inv_freq_len=4)
    model=mod.MiniMaxH3Transformer(p).cuda();model.gradient_checkpointing=True
    l,mask,_=layout();tags=l.token_tags[None].cuda();nv=l.video_indices.numel();na=l.audio_indices.numel()
    result=model(hidden_states=torch.randn(1,nv,16,device='cuda'),
        audio_hidden_states=torch.randn(1,na,4,device='cuda'),encoder_hidden_states=torch.randn(1,4,32,device='cuda'),
        row_timesteps=packing.build_row_timesteps(l,.5,.5)[None].cuda(),token_tags=tags,
        position_ids=l.position_ids[None].cuda(),video_indices=l.video_indices.cuda(),
        audio_indices=l.audio_indices.cuda(),text_indices=l.text_indices.cuda(),vsa_video_grid=(7,4,4),
        native_motion_mask=mask[None].cuda(),native_temporal_options=nt.options({'native_temporal_alignment':True}))
    loss=sum(x.square().mean() for x in result if x.numel());loss.backward()
    grad=model.blocks[0].attn.qkv_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0

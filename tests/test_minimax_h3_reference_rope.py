import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from toolkit.h3_reference_rope import add_source_phase, options_from_kwargs
from minimax_h3_test_utils import ROOT, packing

BFS=ROOT.parent/'ComfyUI-BFSNodes'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('guide_layout',['overlap','sidecar'])
@pytest.mark.parametrize('ref_layout',['native','overlap','sidecar'])
@pytest.mark.parametrize('phase',[False,True])
def test_training_and_comfy_mixed_image_video_audio_parity(guide_layout,ref_layout,phase):
    if not BFS.exists():pytest.skip('Cross-repository parity test requires BFS checkout')
    sys.path.insert(0,str(BFS))
    node=load(BFS/'minimax_h3_downscaled_guide.py','bfs_rope_test')
    native=load(BFS/'tests/fixtures/comfy_h3_layout_8cfe5e1.py','comfy_native_rope_test')
    options=options_from_kwargs(dict(guide_rope_layout=guide_layout,reference_rope_layout=ref_layout,reference_source_phase=phase,reference_phase_scale=0.7,reference_sidecar_margin=2.5))
    kf={'resolved_frame_index':0,'latent':torch.zeros(1,24,7,2,2),'audio_latent':torch.zeros(1,32,2,4),node.MARKER:{'downscale_factor':4,**node.rope_marker(options,1,'guide')}}
    ref={'kind':'image','latent_h':6,'latent_w':4,'latent':torch.zeros(1,24,1,6,4),node.MARKER:node.rope_marker(options,2,'reference')}
    core=native.PackedLayout(3,7,8,8,4,keyframes=[kf],refs=[ref])
    comfy=node.downscaled_layout(core,[kf],[ref])
    trained=packing.build_packed_sequence(torch.ones(3,dtype=torch.long),7,8,8,4,
        ref_blocks=((1,6,4,0),(7,2,2,4)),aligned_refs=(False,True),reference_downscale_factor=4,
        ref_source_ids=(2,1),**options)
    # Compare by source/modality, since native Comfy packs keyframes before refs.
    segments={kind:(a,b) for a,b,kind in comfy.segments}
    guide_indices=trained.video_indices[6:13];identity_indices=trained.video_indices[:6]
    groups=[('cond',guide_indices),('ref_img',identity_indices),('cond_audio',trained.audio_indices[:8]),
            ('video',trained.video_indices[13:]),('audio',trained.audio_indices[8:])]
    for kind,ids in groups:
        a,b=segments[kind]
        torch.testing.assert_close(comfy.position_ids[a:b],trained.position_ids[ids],rtol=0,atol=1e-12)
        if phase:torch.testing.assert_close(comfy.bfs_source_phase_values[a:b],trained.source_phase_values[ids])
    assert torch.equal(core.position_ids[:3],comfy.position_ids[:3])
    assert (ROOT/'toolkit/h3_reference_rope.py').read_bytes()==(BFS/'minimax_h3_reference_rope.py').read_bytes()


def test_source_zero_is_exact_noop_and_phase_changes_only_tagged_rows():
    angles=torch.randn(2,10,96);zeros=torch.zeros(2,10)
    assert add_source_phase(angles,None) is angles
    assert add_source_phase(angles,zeros) is angles
    zeros[:,2:4]=2
    output=add_source_phase(angles,zeros)
    assert torch.equal(output[:,:2],angles[:,:2])
    assert torch.equal(output[:,4:],angles[:,4:])
    assert not torch.equal(output[:,2:4],angles[:,2:4])
    torch.testing.assert_close(output.cos().square()+output.sin().square(),torch.ones_like(output))


def test_batch_padding_and_stable_channel_ids():
    layouts=[packing.build_packed_sequence(torch.ones(t,dtype=torch.long),2,8,8,2,
        ref_blocks=((2,2,2),),aligned_refs=(True,),reference_downscale_factor=4,
        reference_source_phase=True,ref_source_ids=(3,)) for t in (3,5)]
    values=packing.pad_source_phases(layouts)
    assert values.shape[0]==2 and torch.equal(values[0],values[1])
    assert torch.equal(values[:,:5],torch.zeros(2,5))
    assert torch.equal(values[:,5:7],torch.full((2,2),3.0))
    assert not values[:,7:].any()
    # Dropping an earlier source must not change channel 3's identity.
    assert layouts[0].source_phase_values[3].item()==3


def test_actual_h3_rope_composes_phase_and_default_state_keys_stay_same():
    path=ROOT/'extensions_built_in/diffusion_models/minimax_h3/src/transformer.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3Rope')
    ns=dict(torch=torch,nn=torch.nn,add_source_phase=add_source_phase)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
    rope=ns['MiniMaxH3Rope']();positions=torch.randn(1,7,3);values=torch.tensor([[0,1,1,2,0,0,0]],dtype=torch.float32)
    c,s=rope(positions);c0,s0=rope(positions,torch.zeros_like(values));assert torch.equal(c,c0) and torch.equal(s,s0)
    expected=positions.float().unsqueeze(-1)*rope.inv_freq.view(1,1,1,-1)
    half=expected.flatten(2,3);a=add_source_phase(torch.cat((half,half),-1),values)
    cp,sp=rope(positions,values);torch.testing.assert_close(cp,a.cos());torch.testing.assert_close(sp,a.sin())
    assert list(rope.state_dict())==['inv_freq']


@pytest.mark.parametrize('kwargs',[{'guide_rope_layout':'unknown'},{'reference_phase_scale':float('nan')},{'reference_source_phase':1},{'reference_sidecar_margin':-1}])
def test_bad_options_fail_before_model_loading(kwargs):
    with pytest.raises(ValueError):options_from_kwargs(kwargs)


def test_actual_validation_pipeline_forwards_geometry_and_source_channels():
    import numpy as np
    from PIL import Image
    from typing import Optional
    from diffusers.utils.torch_utils import randn_tensor
    from minimax_h3_test_utils import source_function
    kw=dict(guide_rope_layout='sidecar',reference_rope_layout='overlap',reference_source_phase=True,reference_phase_scale=0.7)
    calls=[]
    def transformer(**args):
        calls.append(args)
        return torch.zeros_like(args['hidden_states']),torch.zeros_like(args['audio_hidden_states'])
    model=SimpleNamespace(device_torch='cpu',torch_dtype=torch.float32,transformer=transformer,
        max_text_length=512,model_config=SimpleNamespace(model_kwargs=kw),
        _aligned_ref_flags=lambda blocks:tuple(bool(b[4]) for b in blocks),
        _reference_downscale_factor=lambda:4,
        decode_latents=lambda z:torch.zeros(1,3,1,128,128))
    ns=dict(torch=torch,np=np,Image=Image,Optional=Optional,packing=packing,
        randn_tensor=randn_tensor,options_from_kwargs=options_from_kwargs,
        trim_caption_tokens=lambda e,t,cap:(e,t))
    ns.update({name:getattr(packing,name) for name in ('FPS','AUDIO_CHANNELS','VIDEO_SIGMA_SHIFT','AUDIO_SIGMA_SHIFT',
        'KEYFRAME_NOISE_AUG_T','build_packed_sequence','build_row_timesteps','build_sigma_schedule','remap_sigma',
        'pack_audio_latents','patchify_video_latents','unpack_audio_tokens','unpatchify_video_tokens')})
    call=source_function(ROOT/'extensions_built_in/diffusion_models/minimax_h3/src/pipeline.py','__call__',ns,'MiniMaxH3Pipeline')
    embeds=SimpleNamespace(text_embeds=[torch.zeros(3,2)],text_token_tags=[torch.ones(3,dtype=torch.long)])
    refs=[{'latent':torch.zeros(24,1,2,2),'aligned':True},{'latent':torch.zeros(24,1,6,4),'aligned':False}]
    output=call(SimpleNamespace(model=model),embeds,height=128,width=128,num_frames=1,num_inference_steps=1,
                ref_images=refs,ref_source_ids=[1,3],with_audio=False,generator=torch.Generator().manual_seed(42))
    assert len(output)==1 and len(calls)==1
    expected=packing.build_packed_sequence(torch.ones(3,dtype=torch.long),1,8,8,2,
        ref_blocks=((1,2,2,0),(1,6,4,0)),aligned_refs=(True,False),reference_downscale_factor=4,
        ref_source_ids=(1,3),**options_from_kwargs(kw))
    torch.testing.assert_close(calls[0]['position_ids'][0],expected.position_ids)
    torch.testing.assert_close(calls[0]['source_phase_values'][0],expected.source_phase_values)


def test_explicit_channels_are_not_renumbered_when_first_path_is_absent():
    from toolkit.h3_reference_rope import control_source_channels
    assert control_source_channels(SimpleNamespace(control_path_2='identity',control_path_3='guide'),2)==[2,3]
    assert control_source_channels(SimpleNamespace(),2)==[1,2]
    with pytest.raises(ValueError):control_source_channels(SimpleNamespace(control_path_3='guide'),2)

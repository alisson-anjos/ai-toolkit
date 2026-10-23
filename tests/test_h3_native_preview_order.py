from types import SimpleNamespace as NS
import os
import torch
from PIL import Image
from toolkit import h3_native_temporal as nt
from minimax_h3_test_utils import packing, model_method


def test_order_keeps_picture_numbering_and_video_order_stable():
    entries=[(1,'guide.mp4'),(2,'identity.webp'),(3,'outfit.png'),(4,'second.mp4')]
    ordered=nt.reference_order(entries,lambda x:x[1].endswith('.mp4'))
    assert [x[0] for x in ordered]==[2,3,1,4]


def test_preview_images_get_identical_rope_positions_to_training_after_ordering():
    opts=nt.options({'native_temporal_alignment':True})
    image=(1,8,8,0,False);video=(32,24,24,0,False)
    train=[(2,image),(3,image),(1,video)]
    original_preview=[(1,video),(2,image),(3,image)]
    def positions(items):
        blocks=tuple(x[1] for x in items);ids=tuple(x[0] for x in items)
        l=packing.build_packed_sequence(torch.ones(4,dtype=torch.long),32,24,24,0,ref_blocks=blocks,ref_source_ids=ids)
        nt.align_layout(l,blocks,ids,opts)
        return l.position_ids
    # Regression: the previous code had different image clocks, not merely a row permutation.
    before=positions(original_preview);after=positions(nt.reference_order(original_preview,lambda x:x[1][0]>1))
    assert not torch.equal(before,positions(train))
    assert torch.equal(after,positions(train))


def test_actual_preview_method_passes_pictures_before_video_and_preserves_source_ids(tmp_path):
    p1=tmp_path/'p1.png';p2=tmp_path/'p2.png';Image.new('RGB',(32,32),'red').save(p1);Image.new('RGB',(32,32),'blue').save(p2)
    refvideo={'latent':torch.zeros(24,7,2,2),'aligned':False}
    class Model:
        model=NS(device=torch.device('meta'))
        device_torch=torch.device('cpu')
        native_temporal_options={'enabled':True}
        model_config=NS(model_kwargs={'sample_audio':False})
        get_bucket_divisibility=lambda self:32
        _image_ref_video_frames=lambda self:0
        _encode_ref_video_for_sampling=lambda self,*a:refvideo
        sample_control_is_guide=lambda self,*a:False
    gen=NS(width=32,height=32,num_frames=22,fps=24,ctrl_img=None,ctrl_img_1='guide.mp4',ctrl_img_2=str(p1),ctrl_img_3=str(p2),num_inference_steps=1,guidance_scale=1,latents=None)
    seen={}
    def pipeline(**kw):seen.update(kw);return {'video':'result'}
    sample=model_method('generate_single_image','MinimaxH3Ref2VAModel',torch=torch,os=os,MiniMaxH3Pipeline=object,GenerateImageConfig=object,AdvancedPromptEmbeds=object,packing_video_exts=['.mp4'],partial=lambda *a:None,new_save_image_function=object(),blank_log_image_function=object())
    sample(Model(),pipeline,gen,None,None,None,{})
    assert seen['ref_source_ids']==[2,3,1]
    assert isinstance(seen['ref_images'][0],Image.Image) and isinstance(seen['ref_images'][1],Image.Image)
    assert seen['ref_images'][2] is refvideo


def test_actual_prompt_encoder_uses_the_same_picture_first_presentation():
    class Video:pass
    video=Video();a=Image.new('RGB',(32,32),'red');b=Image.new('RGB',(32,32),'blue');seen=[]
    def encode(*args,**kw):seen.extend(kw['keyframes']);return torch.zeros(2,3),torch.ones(2,dtype=torch.long)
    class Model:
        text_encoder=NS(device=torch.device('meta'))
        device_torch=torch.device('cpu');torch_dtype=torch.float32;tokenizer=object();processor=object();max_text_length=128
        native_temporal_options={'enabled':True};_sample_ref_max_frames=22
        _text_control_images=lambda self,x:x
    method=model_method('get_prompt_embeds',torch=torch,VideoRef=Video,load_video_ref_for_te=lambda *a,**k:video,encode_minimax_h3_prompt=encode,AdvancedPromptEmbeds=NS)
    method(Model(),'motion_transfer: example',control_images=['guide.mp4',a,b])
    assert seen[0] is a and seen[1] is b and seen[2] is video

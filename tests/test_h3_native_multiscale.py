from types import SimpleNamespace as NS
import pytest
from toolkit.h3_native_multiscale import validate_scales,scale_for_step,scaled_native_size,motion_scale


def test_scales_are_reproducible_and_varied():
    a=[scale_for_step([1,.75,.5],42,i) for i in range(30)]
    assert set(a)=={1,.75,.5}
    assert a==[scale_for_step([1,.75,.5],42,i) for i in range(30)]


def test_scale_matches_bfs_grid_rounding():
    assert scaled_native_size(416,736,.5)==(192,352)
    assert scaled_native_size(416,736,1)==(416,736)


def test_only_selected_motion_source_is_scaled():
    b=NS(_h3_native_reference_scale=.5);item=NS(control_video_source_ids=[3,1])
    cfg={'enabled':True,'source_channel':1}
    assert motion_scale(b,item,0,cfg)==1
    assert motion_scale(b,item,1,cfg)==.5
    assert motion_scale(b,item,1,{'enabled':False})==1


@pytest.mark.parametrize('scales',[[],[True],[0],[1.5],[float('nan')]])
def test_bad_scales(scales):
    with pytest.raises(ValueError):validate_scales(scales)


def test_real_video_cache_keeps_scale_variants_separate(tmp_path):
    import cv2
    import numpy as np
    import torch
    from test_minimax_h3_downscaled_refs import cache_helpers
    path=tmp_path/'guide.avi'
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),24,(96,64))
    assert w.isOpened()
    for i in range(22):w.write(np.full((64,96,3),i*8,dtype=np.uint8))
    w.release()
    seen=[]
    def encode(images):
        h,w=images[0].shape[-2:];seen.append((h,w))
        return [torch.zeros(24,7,h//8,w//8)]
    model=NS(encode_images=encode,latent_space_version='scale-test',native_temporal_options={'enabled':True})
    cfg=NS(auto_frame_count=False,num_frames=22,fps=24,trim_auto_frame_count_tail=True)
    load=cache_helpers()['load_ref_video_latent']
    entries=[load(model,str(path),cfg,128,192,native_scale=s) for s in [1,.75,.5,1,.5]]
    assert len(seen)==3
    assert len(set(seen))==3
    assert len(list((tmp_path/'_latent_cache').glob('*.safetensors')))==3
    assert entries[0] is entries[3] and entries[2] is entries[4]

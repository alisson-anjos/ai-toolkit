import os
from collections import OrderedDict
from types import SimpleNamespace as NS
import pytest
import torch
from safetensors.torch import load_file, save_file
from minimax_h3_test_utils import ROOT, source_function

PATH = ROOT / 'toolkit/dataloader_mixins.py'
class DTO:
    pass

def method(name, cls):
    return source_function(PATH, name, dict(torch=torch, os=os, OrderedDict=OrderedDict,
        DTO=DTO, load_file=load_file, save_file=save_file,
        _dto_extras_from_state_dict=lambda state: {}, get_meta_for_safetensors=lambda info: {},
        UnusableFileError=RuntimeError), cls)

@pytest.mark.parametrize('in_memory', [False, True])
def test_temporal_window_survives_disk_and_memory_cache(tmp_path, in_memory):
    cache = method('_cache_one_latent', 'LatentCachingMixin')
    sd = NS(torch_dtype=torch.float32, device_torch='cpu', encode_images=lambda x:x)
    ds = NS(sd=sd,dataset_config=NS(cache_tensors_to_disk=False,do_i2v=False))
    indices=tuple(range(11,84))
    item=NS(is_video=True,latent_space_version='minimax_h3_v1',is_audio_model=False,
        tensor=torch.randn(73,3,2,2),audio_data=None,num_frames=73,
        video_frame_indices=indices,video_source_fps=29.97,
        get_latent_info_dict=lambda:{},cleanup=lambda:None)
    path=str(tmp_path/'cache.safetensors')
    cache(ds,item,path,None,True,True,in_memory)
    state=load_file(path)
    assert tuple(state['video_frame_indices'].tolist())==indices
    assert state['video_source_fps'].item()==29.97
    fresh=NS(is_video=True,latent_space_version='minimax_h3_v1')
    cache(ds,fresh,path,state if in_memory else None,False,True,in_memory)
    assert fresh.video_frame_indices==indices
    assert fresh.video_source_fps==29.97
    # A worker loading directly from disk must restore the same window too.
    worker=NS(is_latent_cached=True,_encoded_latent=None,get_latent_path=lambda:path,
        _cached_tensor_uint8=None,_cached_waveform_int16=None)
    latent=method('get_latent','LatentCachingFileItemDTOMixin')(worker)
    assert latent.shape[0]==73
    assert worker.video_frame_indices==indices
    assert worker.video_source_fps==29.97

def test_h3_cache_key_invalidates_unrecoverable_old_windows():
    info=method('get_latent_info_dict','LatentCachingFileItemDTOMixin')
    ds=NS(auto_frame_count=False,num_frames=73,fps=24,do_i2v=False,do_audio=False,
        cache_tensors_to_disk=False,shrink_video_to_frames=False)
    item=NS(path='clip.mp4',scale_to_width=1280,scale_to_height=768,crop_x=0,crop_y=0,
        crop_width=1280,crop_height=768,latent_space_version='minimax_h3_v1',latent_version=1,
        flip_x=False,flip_y=False,is_video=True,dataset_config=ds,is_audio_model=False,load_rgba=False)
    a=info(item)
    assert a['video_temporal_cache_version']==2
    ds.shrink_video_to_frames=True
    assert info(item)!=a
    item.latent_space_version='another_model'
    assert 'video_temporal_cache_version' not in info(item)

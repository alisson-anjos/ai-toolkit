import base64
import copy
import hashlib
import json
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace, ModuleType

import cv2
import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from minimax_h3_test_utils import ROOT, SRC, packing, model_method, source_function


def layout(factor=1, refs=(), flags=(), audio=0):
    return packing.build_packed_sequence(
        torch.ones(4, dtype=torch.long), 7, 16, 24, audio,
        ref_blocks=refs, aligned_refs=flags, reference_downscale_factor=factor,
    )


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_guide_positions_are_strided_target_positions(factor):
    result = layout(factor, ((7, 16 // factor, 24 // factor),), (True,))
    target = result.position_ids[-7 * 8 * 12:].reshape(7, 8, 12, 3)
    count = 7 * (8 // factor) * (12 // factor)
    guide = result.position_ids[result.video_indices[:count]]
    assert torch.equal(guide, target[:, ::factor, ::factor].reshape(-1, 3))
    assert result.num_condition_video_rows == 7 * 8 * 12 // factor**2
    assert torch.equal(target.reshape(-1, 3), layout().position_ids[-7 * 8 * 12:])


def test_identity_and_soundtrack_keep_their_positions():
    blocks = ((1, 8, 12, 0), (7, 8, 12, 5))
    small = layout(2, blocks, (False, True), audio=5)
    full = layout(1, ((1, 8, 12, 0), (7, 16, 24, 5)), (False, True), audio=5)
    assert torch.equal(small.position_ids[small.video_indices[:24]], full.position_ids[full.video_indices[:24]])
    assert torch.equal(small.position_ids[small.audio_indices], full.position_ids[full.audio_indices])
    assert torch.equal(small.position_ids[-7 * 8 * 12:], full.position_ids[-7 * 8 * 12:])


@pytest.mark.parametrize('factor', [0, -1, 1.5, True, '2'])
def test_invalid_factors(factor):
    with pytest.raises(ValueError, match='integer >= 1'):
        layout(factor)


@pytest.mark.parametrize('dims', [(8, 24), (16, 24), (7, 12)])
def test_wrong_guide_grid_rejected(dims):
    with pytest.raises(ValueError, match='aligned reference'):
        layout(2, ((7, *dims),), (True,))


@pytest.mark.parametrize('dims,factor', [((288, 512), 2), ((256, 480), 2), ((64, 96), 4)])
def test_pixel_dimensions_must_preserve_exact_ratio(dims, factor):
    with pytest.raises(ValueError, match='positive multiples'):
        packing.aligned_reference_pixel_size(*dims, factor)


def test_valid_pixel_dimensions():
    assert packing.aligned_reference_pixel_size(576, 1024, 2) == (288, 512)
    assert packing.aligned_reference_pixel_size(576, 1024, 1) == (576, 1024)


@pytest.mark.parametrize('kw', [
    {'reference_downscale_factor': 2},
    {'reference_downscale_factor': 2, 'align_video_refs': True, 'dopsd': True},
    {'reference_downscale_factor': 2, 'align_video_refs': True, 'image_refs_as_video': True},
])
def test_incompatible_model_settings(kw):
    with pytest.raises(ValueError):
        model_method('_reference_downscale_factor')(SimpleNamespace(model_config=SimpleNamespace(model_kwargs=kw)))


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / 'guide.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 24, (96, 64))
    assert writer.isOpened()
    for i in range(22):
        writer.write(np.full((64, 96, 3), i * 8, dtype=np.uint8))
    writer.release()
    return str(path)


def cache_helpers():
    ns = dict(torch=torch, cv2=cv2, np=np, os=os, json=json, base64=base64, hashlib=hashlib,
              load_file=load_file, save_file=save_file,
              get_quick_signature_string=lambda path: str(os.stat(path).st_size),
              reference_video_pixel_size=packing.reference_video_pixel_size,
              aligned_reference_pixel_size=packing.aligned_reference_pixel_size,
              validate_reference_downscale_factor=packing.validate_reference_downscale_factor)
    for name in ['ref_frame_indices', 'read_frames_at', '_cache_path', 'load_ref_video_latent']:
        source_function(SRC / 'ref_video_cache.py', name, ns)
    return ns


def test_video_encoding_and_cache_separate_resolutions(clip):
    ns = cache_helpers()
    sizes = []
    def encode(images):
        pixels = images[0]
        sizes.append(tuple(pixels.shape[-2:]))
        return [torch.zeros(24, 7, pixels.shape[-2] // 16, pixels.shape[-1] // 16)]
    model = SimpleNamespace(encode_images=encode, latent_space_version='test')
    config = SimpleNamespace(auto_frame_count=False, num_frames=22, fps=24, trim_auto_frame_count_tail=True)
    def get(factor, h=128, w=192):
        return ns['load_ref_video_latent'](model, clip, config, h, w, align=True, reference_downscale_factor=factor)
    small = get(2)
    full = get(1)
    assert small['latent'].shape[-2:] == (4, 6)
    assert full['latent'].shape[-2:] == (8, 12)
    assert get(2) is small
    assert sizes == [(64, 96), (128, 192)]
    model._ref_video_cache.clear()
    assert torch.equal(get(2)['latent'], small['latent'])  # disk hit, no encode
    assert sizes == [(64, 96), (128, 192)]
    assert get(2, 192, 128)['latent'].shape[-2:] == (6, 4)  # same area, different grid
    assert sizes[-1] == (96, 64)


def test_sampling_uses_same_guide_size_as_training(clip, monkeypatch):
    ns = cache_helpers()
    package = 'minimax_h3_test'
    module = ModuleType(package + '.src.ref_video_cache')
    module.read_frames_at = ns['read_frames_at']
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sizes = []
    def encode(pixels, **kwargs):
        sizes.append(tuple(pixels.shape[-2:]))
        return torch.zeros(1, 24, 7, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
    class FakeModel:
        model_config = SimpleNamespace(model_kwargs={'align_video_refs': True, 'reference_downscale_factor': 2})
        _reference_downscale_factor = model_method('_reference_downscale_factor')
        video_vae = SimpleNamespace(encode=encode, dtype=torch.float32)
        vae = SimpleNamespace(device='cpu')
    method = model_method('_encode_ref_video_for_sampling', 'MinimaxH3Ref2VAModel',
                          torch=torch, __package__=package, KEYFRAME_ENCODE_SEED=packing.KEYFRAME_ENCODE_SEED,
                          ref_frame_indices=ns['ref_frame_indices'])
    result = method(FakeModel(), clip, SimpleNamespace(height=128, width=192, num_frames=22))
    assert sizes == [(64, 96)]
    assert result['latent'].shape[-2:] == (4, 6)


def test_metadata_round_trip_and_lora_save_hook(tmp_path):
    class Model:
        model_config = SimpleNamespace(model_kwargs={'align_video_refs': True, 'reference_downscale_factor': 2})
        control_latent_only = True
        guide_latent_only = False
        _reference_downscale_factor = model_method('_reference_downscale_factor')
        get_additional_save_metadata = model_method('get_additional_save_metadata', 'MinimaxH3Ref2VAModel')
    model = Model()
    # Exercise the actual pre-serialization section shared by LoRA/full-model saves.
    import ast
    tree = ast.parse((ROOT / 'jobs/process/BaseSDTrainProcess.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseSDTrainProcess')
    save = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'save')
    start = next(i for i,n in enumerate(save.body) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='save_meta' for t in n.targets))
    end = next(i for i in range(start+1,len(save.body)) if isinstance(save.body[i],ast.Assign) and any(isinstance(t,ast.Name) and t.id=='save_meta' for t in save.body[i].targets))
    ns = {'self': SimpleNamespace(sd=model, meta={'existing': 'kept'}, adapter=None), 'copy': copy, 'CustomAdapter': type('CustomAdapter', (), {})}
    exec(compile(ast.Module(body=save.body[start:end],type_ignores=[]), '<actual save metadata hook>', 'exec'), ns)
    serialize = source_function(ROOT / 'toolkit/metadata.py', 'get_meta_for_safetensors', {'json': json, 'OrderedDict': OrderedDict})
    meta = serialize(ns['save_meta'], add_software_info=False)
    path = tmp_path / 'lora.safetensors'
    save_file({'lora_dummy.weight': torch.ones(2, 2)}, str(path), metadata=meta)
    with safe_open(str(path), framework='pt') as f:
        saved = f.metadata()
    assert saved['reference_downscale_factor'] == '2'
    assert saved['align_video_refs'] == 'true'
    assert saved['control_latent_only'] == 'true'
    assert saved['minimax_h3_guide_position_version'] == 'target_grid_stride_v1'
    assert saved['existing'] == 'kept'


def test_source_and_mask_are_separate_aligned_video_blocks():
    result = layout(2, ((7, 8, 12), (7, 8, 12), (1, 8, 12)), (True, True, False))
    count = 7 * 4 * 6
    video_positions = result.position_ids[result.video_indices]
    assert torch.equal(video_positions[:count], video_positions[count:2 * count])
    assert result.num_condition_video_rows == count * 2 + 24
    expected = result.position_ids[-7 * 8 * 12:].reshape(7, 8, 12, 3)[:, ::2, ::2].reshape(-1, 3)
    assert torch.equal(video_positions[:count], expected)


def text_filter_model(guide_only, all_only):
    class Model:
        guide_latent_only = guide_only
        control_latent_only = all_only
        _text_control_images = model_method('_text_control_images', os=os,
                                            packing_video_exts=['.mp4', '.avi', '.mov', '.mkv'])
    return Model()


def test_native_images_retained_while_source_and_mask_bypass_vlm():
    face = torch.ones(3, 8, 8)
    controls = [[face, 'source.mp4', 'mask.MP4'], ['face.png', 'source2.avi']]
    filtered = text_filter_model(True, False)._text_control_images(controls)
    assert len(filtered) == 2
    assert len(filtered[0]) == 1 and filtered[0][0] is face
    assert filtered[1] == ['face.png']
    assert text_filter_model(False, False)._text_control_images(controls) is controls
    assert text_filter_model(True, True)._text_control_images(controls) is None
    assert text_filter_model(True, False)._text_control_images([['source.mp4'], ['mask.mp4']]) == [[], []]


def test_presentation_modes_have_separate_text_cache_versions():
    cache_version = model_method('text_embedding_space_version', 'MinimaxH3Ref2VAModel')
    model = SimpleNamespace(arch='minimax_h3_ref2va', _image_ref_video_frames=lambda: 0,
                            guide_latent_only=False, control_latent_only=False)
    normal = cache_version.fget(model)
    model.guide_latent_only = True
    guide = cache_version.fget(model)
    model.control_latent_only = True
    all_only = cache_version.fget(model)
    assert len({normal, guide, all_only}) == 3


@pytest.mark.parametrize('factor', [1, 2])
def test_small_transformer_forward_backward_with_source_mask_and_identity(factor):
    """Real attention/RoPE/backprop, random tiny weights; excludes model-loading mixin."""
    import ast
    import types
    tree = ast.parse((SRC / 'transformer.py').read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == 'toolkit.models.v2._mixin'
    )]
    module = types.ModuleType('_minimax_h3_test_transformer')
    module.OstrisModelMixin = type('OstrisModelMixin', (), {})
    sys.modules[module.__name__] = module
    exec(compile(tree, str(SRC / 'transformer.py'), 'exec'), module.__dict__)
    params = module.MiniMaxH3TransformerParams(
        hidden_size=64, num_layers=1, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=32, ffn_hidden_size=128,
        text_dim=16, timestep_input_dim=16, time_embed_hidden_size=64,
        time_embed_dim=32, rope_inv_freq_len=4,
    )
    model = module.MiniMaxH3Transformer(params)
    packed = packing.build_packed_sequence(
        torch.ones(4, dtype=torch.long), 2, 8, 8, 0,
        ref_blocks=((2, 8 // factor, 8 // factor), (2, 8 // factor, 8 // factor), (1, 4, 4)),
        aligned_refs=(True, True, False), reference_downscale_factor=factor,
    )
    video = torch.randn(1, len(packed.video_indices), 96, requires_grad=True)
    pred, audio = model(
        hidden_states=video,
        audio_hidden_states=torch.empty(1, 0, 32),
        encoder_hidden_states=torch.randn(1, 4, 16),
        row_timesteps=packing.build_row_timesteps(packed, 0.5, 0.5)[None],
        token_tags=packed.token_tags[None], position_ids=packed.position_ids[None],
        video_indices=packed.video_indices, audio_indices=packed.audio_indices,
        text_indices=packed.text_indices,
    )
    assert pred.shape == video.shape and audio.shape == (1, 0, 32)
    target_prediction = pred[:, packed.num_condition_video_rows:]
    assert target_prediction.shape == (1, 32, 96)
    loss = target_prediction.square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(video.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('factor,expected', [(1, 32), (2, 64), (4, 128)])
def test_training_and_sampling_bucket_grid(factor, expected):
    class Model:
        model_config = SimpleNamespace(model_kwargs={'align_video_refs': True, 'reference_downscale_factor': factor})
        _reference_downscale_factor = model_method('_reference_downscale_factor')
    assert model_method('get_bucket_divisibility', 'MinimaxH3Ref2VAModel')(Model()) == expected

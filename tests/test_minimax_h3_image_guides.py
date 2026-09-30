from types import SimpleNamespace as NS
import os

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from minimax_h3_test_utils import (
    ROOT, packing, model_method, source_function, prepare_guide_image,
    is_image_guide, image_guide_channel,
)
from test_minimax_h3_training_options import controls
from test_minimax_h3_downscaled_refs import cache_helpers


def item(**kwargs):
    return NS(scale_to_width=384, scale_to_height=256, crop_x=64, crop_y=32,
              crop_width=256, crop_height=128, flip_x=True, flip_y=False, **kwargs)


def image():
    y, x = np.mgrid[:256, :384]
    return Image.fromarray(np.stack((x % 256, y, (x + y) % 256), -1).astype('uint8'))


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_image_guide_reproduces_target_crop_and_flip(factor):
    src = image()
    target = ImageOps.mirror(src).resize((384, 256), Image.Resampling.BICUBIC)
    target = target.crop((64, 32, 320, 160))
    expected = target.resize((256 // factor, 128 // factor), Image.Resampling.LANCZOS)
    actual = prepare_guide_image(src, 128, 256, factor, item())
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_guide_role_uses_control_channel_not_first_available_image():
    config = NS(control_path_1='/source', control_path_2='/identity', control_path_3=None)
    kw = {'align_image_refs': True, 'image_guide_channel': 1}
    assert is_image_guide('/source/frame.png', config, kw)
    assert not is_image_guide('/identity/face.png', config, kw)
    config.control_path_1 = None
    assert not is_image_guide('/identity/face.png', config, kw)
    for bad in [True, 0, 4, '1']:
        with pytest.raises(ValueError):
            image_guide_channel({'image_guide_channel': bad})


def test_actual_training_builder_image_guide_and_native_identity():
    encoded = []
    class Model:
        model_config = NS(model_kwargs={'align_image_refs': True, 'align_video_refs': True,
                                       'guide_latent_only': True, 'reference_downscale_factor': 4})
        _reference_dropout_probabilities = lambda self: (0, 0)
        _reference_downscale_factor = lambda self: 4
        _image_ref_video_frames = lambda self: 0
        _append_video_ref_blocks = lambda self, *args: None
        def encode_keyframe_latents(self, px):
            encoded.append(tuple(px.shape))
            return torch.zeros(px.shape[0], 24, 1, px.shape[-2] // 16, px.shape[-1] // 16)
    model = Model()
    batch = NS(control_tensor_list=[[torch.ones(3, 32, 64), torch.ones(3, 128, 128)]],
               control_video_paths_list=[], file_items=[NS(aligned_image_guide_flags=[True, False])])
    build = model_method('_build_condition', 'MinimaxH3Ref2VAModel', torch=torch,
                         reference_keep_masks=controls.reference_keep_masks,
                         patchify_video_latents=packing.patchify_video_latents,
                         KEYFRAME_NOISE_AUG_T=packing.KEYFRAME_NOISE_AUG_T)
    rows, _, _, blocks = build(model, batch, (1, 8, 16), 'cpu', torch.float32)
    assert encoded[0] == (1, 3, 1, 32, 64)
    assert encoded[1][-2] > 32  # Native reference retains ordinary full-area sizing.
    flags = model_method('_aligned_ref_flags')(model, blocks)
    assert flags == (True, False)
    layout = packing.build_packed_sequence(torch.zeros(3, dtype=torch.long), 1, 8, 16, 0,
        ref_blocks=blocks, aligned_refs=flags, reference_downscale_factor=4)
    target = layout.position_ids[-32:].reshape(4, 8, 3)
    assert torch.equal(layout.position_ids[3:5], target[::4, ::4].reshape(-1, 3))
    assert rows.shape[1] == layout.num_condition_video_rows


def test_image_guides_use_guide_dropout_and_native_images_use_reference_dropout():
    batch = NS()
    keep = controls.reference_keep_masks(batch, 2, 1, 0, 1, (True, False))
    assert keep == ((False, True), (False,))
    assert controls.reference_keep_masks(batch, 2, 1, 0, 1, (True, False)) == keep


def test_actual_control_loader_reduces_only_aligned_image(tmp_path):
    src = tmp_path / 'source.jpg'; ref = tmp_path / 'identity.jpg'
    image().save(src); image().save(ref)
    class TensorTransform:
        def __call__(self, img):
            return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    transforms = NS(Compose=lambda _: TensorTransform(), ToTensor=TensorTransform)
    # The helper is already loaded; avoid importing unrelated training dependencies
    # in this isolated AST loader's one local helper import.
    import ast
    tree = ast.parse((ROOT / 'toolkit/dataloader_mixins.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ControlFileItemDTOMixin')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'load_control_image')
    for node in ast.walk(fn):
        if hasattr(node, 'body') and isinstance(node.body, list):
            node.body[:] = [n for n in node.body if not (isinstance(n, ast.ImportFrom) and n.module == 'toolkit.aligned_guides')]
    ns = dict(Image=Image, exif_transpose=ImageOps.exif_transpose, transforms=transforms,
              prepare_guide_image=prepare_guide_image, print_acc=print, torch=torch)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-control-loader>', 'exec'), ns)
    fi = item(has_control_image=True, get_new_control_paths=lambda: [str(src), str(ref)],
              load_rgba=False, dataset_config=NS(control_transparent_color=[255]*3),
              full_size_control_images=True, use_raw_control_images=True,
              aligned_image_guide_flags=[True, False], _aligned_image_guide_factor=4,
              aug_replay_spatial_transforms=None, control_tensor=None, control_tensor_list=None)
    ns['load_control_image'](fi)
    assert [tuple(t.shape) for t in fi.control_tensor_list] == [(3, 32, 64), (3, 256, 384)]


def test_video_cache_tracks_crop_and_target_frame_selection(tmp_path):
    import cv2
    path = str(tmp_path / 'guide.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (384, 256))
    assert writer.isOpened()
    for index in range(8):
        frame = np.asarray(image()).copy()
        frame[..., 2] = index * 25
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    decoded = []
    def encode(images):
        decoded.append(images[0].clone())
        return [torch.zeros(24, 2, 2, 4)]
    model = NS(encode_images=encode, latent_space_version='test')
    config = NS(auto_frame_count=False, num_frames=5, fps=24, trim_auto_frame_count_tail=True)
    fi = item(video_frame_indices=(1, 2, 3, 4, 5), video_source_fps=24)
    get = cache_helpers()['load_ref_video_latent']
    kwargs = dict(align=True, reference_downscale_factor=4, file_item=fi)
    get(model, path, config, 128, 256, **kwargs)
    assert len(decoded) == 1 and decoded[0].shape == (5, 3, 32, 64)
    assert decoded[0][0, 2].mean().item() == pytest.approx(25/127.5 - 1, abs=0.05)
    get(model, path, config, 128, 256, **kwargs)
    assert len(decoded) == 1
    fi.crop_x = 0
    get(model, path, config, 128, 256, **kwargs)
    assert len(decoded) == 2
    assert not torch.equal(decoded[0][:, :2], decoded[1][:, :2])
    fi.video_frame_indices = (2, 3, 4, 5, 6)
    get(model, path, config, 128, 256, **kwargs)
    assert len(decoded) == 3


def test_sampling_encodes_image_guide_and_preserves_native_reference(tmp_path):
    guide = tmp_path / 'guide.png'; ref = tmp_path / 'ref.png'
    image().save(guide); image().save(ref)
    received = []
    class Model:
        model = NS(device='cuda')
        guide_latent_only = True
        control_latent_only = False
        model_config = NS(model_kwargs={'align_image_refs': True, 'image_guide_channel': 1,
                                       'reference_downscale_factor': 4})
        get_bucket_divisibility = lambda self: 128
        _reference_downscale_factor = lambda self: 4
        _image_ref_video_frames = lambda self: 0
        sample_control_is_guide = model_method('sample_control_is_guide', 'MinimaxH3Ref2VAModel', os=os, packing_video_exts=['.mp4'])
        def encode_keyframe_latents(self, pixels):
            assert pixels.shape == (1, 3, 1, 32, 64)
            return torch.zeros(1, 24, 1, 2, 4)
    gen = NS(width=256, height=128, num_frames=1, ctrl_img=None, ctrl_img_1=str(guide),
             ctrl_img_2=str(ref), ctrl_img_3=None, num_inference_steps=1,
             guidance_scale=1, latents=None)
    def pipeline(**kwargs):
        received.append(kwargs)
        return ['image-result']
    sample = model_method('generate_single_image', 'MinimaxH3Ref2VAModel', torch=torch,
        os=os, MiniMaxH3Pipeline=object, GenerateImageConfig=object, AdvancedPromptEmbeds=object,
        packing_video_exts=['.mp4'])
    assert sample(Model(), pipeline, gen, None, None, None, {}) == 'image-result'
    refs = received[0]['ref_images']
    assert refs[0]['aligned'] is True and isinstance(refs[1], Image.Image)
    assert model_method('sample_text_control_skip_slots', 'MinimaxH3Ref2VAModel', os=os, packing_video_exts=['.mp4'])(Model(), gen) == {'ctrl_img_1'}


def test_dataset_role_overrides_and_sample_roles_are_independent():
    cfg = NS(control_path_1='/video', control_path_2='/frame', control_path_3='/identity',
             control_role_1='guide', control_role_2='guide', control_role_3='reference')
    kw = {'align_image_refs': True, 'image_guide_channel': 1}
    assert is_image_guide('/frame/first.png', cfg, kw)
    assert not is_image_guide('/identity/face.png', cfg, kw)
    cfg.control_role_1 = 'reference'
    assert not is_image_guide('/video/clip.mp4', cfg, kw)
    model = NS(model_config=NS(model_kwargs=kw))
    role = model_method('sample_control_is_guide', 'MinimaxH3Ref2VAModel', os=os, packing_video_exts=['.mp4'])
    sample = NS(ctrl_role_1='reference', ctrl_role_2='guide')
    assert not role(model, sample, 1, '/video/clip.mp4')
    assert role(model, sample, 2, '/frame/first.png')
    assert controls.reference_keep_masks(NS(), 0, 2, 1, 0, (), (False, True)) == ((), (False, True))

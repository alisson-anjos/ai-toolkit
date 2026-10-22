"""guide_keep_aspect: an aligned guide on its own aspect grid, on the target's clock."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from minimax_h3_test_utils import ROOT, packing, keep_aspect_guide_size, prepare_guide_image
from toolkit.h3_reference_rope import options_from_kwargs

BFS = Path(os.environ.get('BFS_NODES_DIR', ROOT.parent / 'ComfyUI-BFSNodes'))
TEXT = torch.ones(3, dtype=torch.long)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack(t, h, w, blocks, aligned, factor=1, keep=False, **kw):
    return packing.build_packed_sequence(TEXT, t, h, w, 4, ref_blocks=blocks, aligned_refs=aligned,
                                         reference_downscale_factor=factor, guide_keep_aspect=keep, **kw)


@pytest.mark.parametrize('factor', [1, 2])
def test_same_aspect_is_exactly_the_stride_grid(factor):
    guide = (7, 8 // factor, 12 // factor, 0)
    old = pack(7, 8, 12, (guide,), (True,), factor)
    new = pack(7, 8, 12, (guide,), (True,), factor, keep=True)
    torch.testing.assert_close(new.position_ids, old.position_ids, rtol=0, atol=1e-12)


def test_landscape_guide_on_portrait_target_is_centred_and_frame_locked():
    # portrait target 12x8 latent, landscape guide 8x12 latent
    layout = pack(7, 12, 8, ((7, 8, 12, 0),), (True,), keep=True)
    guide = layout.position_ids[layout.video_indices[:7 * 4 * 6]].reshape(7, 4 * 6, 3)
    target = layout.position_ids[layout.video_indices[7 * 4 * 6:]].reshape(7, 6 * 4, 3)
    torch.testing.assert_close(guide[:, 0, 0], target[:, 0, 0])  # same clock, frame by frame
    for axis in (1, 2):  # both frames centred on the same point
        torch.testing.assert_close(guide[0, :, axis].mean(), target[0, :, axis].mean())
    # landscape: wider in w and shorter in h than the portrait target
    assert guide[0, :, 2].max() > target[0, :, 2].max() and guide[0, :, 1].max() < target[0, :, 1].max()


def test_keep_aspect_guide_does_not_move_the_target_clock():
    plain = pack(7, 12, 8, ((1, 6, 4, 0),), (False,))
    keep = pack(7, 12, 8, ((1, 6, 4, 0), (7, 8, 12, 0)), (False, True), keep=True)
    n_plain, n_keep = len(plain.video_indices), len(keep.video_indices)
    rows = 7 * 6 * 4
    torch.testing.assert_close(keep.position_ids[keep.video_indices[n_keep - rows:]],
                               plain.position_ids[plain.video_indices[n_plain - rows:]])


def test_keep_aspect_rejects_off_grid_guides_and_keyframes():
    with pytest.raises(ValueError):
        pack(7, 12, 8, ((7, 7, 12, 0),), (True,), keep=True)
    with pytest.raises(ValueError):
        packing.build_packed_sequence(TEXT, 7, 12, 8, 4, keyframe_anchors=('first',), guide_keep_aspect=True)


def test_guide_size_and_preparation_keep_the_whole_frame():
    # 1280x720 landscape guide for a 432x768 portrait target: same pixel area, own aspect, /32
    assert keep_aspect_guide_size(1280, 720, 768, 432, 1) == (448, 768)
    assert keep_aspect_guide_size(720, 1280, 768, 432, 1) == (768, 448)  # same aspect -> target size
    assert keep_aspect_guide_size(720, 1280, 768, 432, 2) == (384, 224)  # and target / factor
    image = Image.new('RGB', (1280, 720))
    image.paste((255, 0, 0), (0, 0, 40, 720))  # red strip on the left edge
    flipped = type('Item', (), {'flip_x': True, 'flip_y': False})()
    out = prepare_guide_image(image, 768, 432, 1, flipped, keep_aspect=True)
    assert out.size == (768, 448)
    assert out.getpixel((766, 200))[0] > 200 and out.getpixel((1, 200))[0] < 50  # mirrored, nothing cropped


@pytest.mark.parametrize('guide_layout', ['overlap', 'sidecar'])
@pytest.mark.parametrize('order', ['identity_first', 'guide_first'])
def test_training_and_comfy_keep_aspect_parity(guide_layout, order):
    if not BFS.exists():
        pytest.skip('Cross-repository parity test requires a BFS checkout (BFS_NODES_DIR)')
    sys.path.insert(0, str(BFS))
    node = load(BFS / 'minimax_h3_downscaled_guide.py', 'bfs_keep_aspect_test')
    native = load(BFS / 'tests/fixtures/comfy_h3_layout_8cfe5e1.py', 'comfy_native_keep_aspect_test')
    options = options_from_kwargs(dict(guide_rope_layout=guide_layout, guide_keep_aspect=True,
                                       reference_sidecar_margin=2.5))
    guide = {'kind': 'video', 'latent_t': 7, 'latent_h': 8, 'latent_w': 12, 'ref_audio_t': 0,
             'latent': torch.zeros(1, 24, 7, 8, 12),
             node.MARKER: {'downscale_factor': 1, 'keep_aspect': True, **node.rope_marker(options, 1, 'guide')}}
    identity = {'kind': 'image', 'latent_h': 6, 'latent_w': 4, 'latent': torch.zeros(1, 24, 1, 6, 4),
                node.MARKER: node.rope_marker(options, 2, 'reference')}
    refs = [identity, guide] if order == 'identity_first' else [guide, identity]
    core = native.PackedLayout(3, 7, 12, 8, 4, keyframes=[], refs=refs)
    comfy = node.downscaled_layout(core, [], refs)
    trained = pack(7, 12, 8, ((1, 6, 4, 0), (7, 8, 12, 0)), (False, True), keep=True,
                   **{k: v for k, v in options.items() if k != 'guide_keep_aspect'})
    ref_segments = [(a, b) for a, b, kind in comfy.segments if kind == 'ref_img']
    comfy_identity, comfy_guide = ref_segments if order == 'identity_first' else ref_segments[::-1]
    video = next((a, b) for a, b, kind in comfy.segments if kind == 'video')
    audio = next((a, b) for a, b, kind in comfy.segments if kind == 'audio')
    for (a, b), ids in ((comfy_identity, trained.video_indices[:6 * 4 // 4]),
                        (comfy_guide, trained.video_indices[6:6 + 7 * 4 * 6]),
                        (video, trained.video_indices[6 + 7 * 4 * 6:]),
                        (audio, trained.audio_indices)):
        torch.testing.assert_close(comfy.position_ids[a:b], trained.position_ids[ids], rtol=0, atol=1e-12)

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minimax_h3_test_utils import SRC, model_method


def load(name):
    spec = importlib.util.spec_from_file_location('_h3_test_' + name, SRC / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

controls = load('training_controls')
auxiliary = load('auxiliary_losses')


@pytest.mark.parametrize('value', [-1, 2, float('nan'), '0.5', True])
def test_bad_dropout_probability(value):
    with pytest.raises(ValueError):
        controls.dropout_probability(value, 'reference_dropout')


def test_dropout_has_independent_references_and_reuses_selection(monkeypatch):
    calls = []
    def rand(n):
        calls.append(n)
        return torch.tensor([0.1, 0.9, 0.3][:n])
    monkeypatch.setattr(torch, 'rand', rand)
    batch = SimpleNamespace()
    masks = controls.reference_keep_masks(batch, 3, 2, 0.5, 0.5)
    assert masks == ((False, True, False), (False, True))
    assert controls.reference_keep_masks(batch, 3, 2, 0.5, 0.5) == masks
    assert calls == [3, 2]
    with pytest.raises(ValueError, match='counts changed'):
        controls.reference_keep_masks(batch, 2, 2, 0.5, 0.5)


def test_drop_all_and_keep_all_and_caption_only():
    assert controls.reference_keep_masks(SimpleNamespace(), 3, 2, 0, 1) == ((True,)*3, (False,)*2)


@pytest.mark.parametrize('kw', [
    {'reference_dropout': 0.1},
    {'guide_dropout': 0.2, 'guide_latent_only': True, 'dopsd': True},
    {'reference_dropout': 0.5, 'guide_latent_only': True, 'image_refs_as_video': True},
])
def test_dropout_validates_presentation(kw):
    method = model_method('_reference_dropout_probabilities', 'MinimaxH3Ref2VAModel', dropout_probability=controls.dropout_probability)
    with pytest.raises(ValueError):
        method(SimpleNamespace(model_config=SimpleNamespace(model_kwargs=kw)))


class FaceEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3))


class PoseEvaluator(nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, 4)


class DetachedEvaluator(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3)).detach()


@pytest.fixture
def evaluator_paths(tmp_path):
    paths = {}
    for name, model in [('arcface', FaceEvaluator()), ('pose_heatmap', PoseEvaluator()), ('detached', DetachedEvaluator())]:
        path = tmp_path / (name + '.pt')
        torch.jit.script(model).save(str(path))
        paths[name] = str(path)
    return paths


def inputs():
    prediction = torch.randn(2, 3, 7, 8, 8, requires_grad=True)
    noisy = torch.randn_like(prediction)
    clean = torch.randn_like(prediction)
    sigma = torch.tensor([0.2, 0.8]).view(2, 1, 1, 1, 1)
    return prediction, noisy, clean, sigma


def test_multiple_custom_losses_sum_and_backprop_without_training_evaluators(evaluator_paths):
    specs = [{'type': 'pixel_l1', 'weight': 0.2},
             {'type': 'arcface', 'weight': 0.3, 'model_path': evaluator_paths['arcface'], 'crop': [0.25, 0.25, 0.75, 0.75]},
             {'type': 'pose_heatmap', 'weight': 0.4, 'model_path': evaluator_paths['pose_heatmap']}]
    losses = auxiliary.VideoAuxiliaryLosses(specs, latent_frames=2, sample_frames=2)
    args = inputs()
    decoded = []
    def decode(z):
        decoded.append((z.shape[2], z.requires_grad))
        return z.tanh()
    result, logs = losses(*args, decode)
    assert decoded == [(2, True), (2, False)]
    expected = sum(logs['loss/aux_' + spec['type'] + '_weighted'] for spec in specs) * 0.5
    assert result.item() == pytest.approx(expected)
    result.backward()
    assert args[0].grad[0].abs().sum() > 0
    assert args[0].grad[1].abs().sum() == 0  # high-sigma item excluded
    assert torch.isfinite(args[0].grad).all()
    assert all(not p.requires_grad and p.grad is None for m in losses.models.values() for p in m.parameters())


def test_high_sigma_skips_decode_and_disabled_losses_do_nothing():
    args = list(inputs()); args[-1] = torch.ones_like(args[-1])
    def forbidden(_):
        raise AssertionError('decode must not run')
    result, logs = auxiliary.VideoAuxiliaryLosses([{'type': 'pixel_l1'}])(*args, forbidden)
    assert result.item() == 0 and logs['loss/aux_active'] == 0
    result.backward()
    assert auxiliary.VideoAuxiliaryLosses([])(*args, forbidden) == (None, {})


def test_detached_evaluator_is_rejected(evaluator_paths):
    losses = auxiliary.VideoAuxiliaryLosses([{'type': 'arcface', 'model_path': evaluator_paths['detached']}])
    with pytest.raises(RuntimeError, match='detached'):
        losses(*inputs(), lambda z: z.tanh())


@pytest.mark.parametrize('spec', [
    [{'type': 'openpose'}],
    [{'type': 'arcface', 'model_path': '/missing'}],
    [{'type': 'pixel_l1', 'weight': -1}],
    [{'type': 'pixel_l1', 'weight': float('nan')}],
    [{'type': 'pixel_l1'}, {'type': 'pixel_l1'}],
    [{'type': 'pixel_l1', 'crop': [1, 0, 0, 1]}],
])
def test_invalid_auxiliary_configuration(spec):
    with pytest.raises(ValueError):
        auxiliary.VideoAuxiliaryLosses(spec)


def test_actual_condition_builder_drops_references_before_encoding():
    from minimax_h3_test_utils import packing
    class Model:
        model_config = SimpleNamespace(model_kwargs={'reference_dropout': 1.0, 'guide_dropout': 0, 'control_latent_only': True})
        _reference_dropout_probabilities = model_method('_reference_dropout_probabilities', 'MinimaxH3Ref2VAModel', dropout_probability=controls.dropout_probability)
        _image_ref_video_frames = lambda self: 0
        _append_video_ref_blocks = lambda self, *args: None
        def encode_keyframe_latents(self, pixels):
            raise AssertionError('Dropped references must not be encoded')
    model = Model()
    batch = SimpleNamespace(control_tensor_list=[[torch.ones(3, 64, 64) for _ in range(3)]], control_video_paths_list=[], file_items=[object()])
    build = model_method('_build_condition', 'MinimaxH3Ref2VAModel', torch=torch,
                         reference_keep_masks=controls.reference_keep_masks,
                         patchify_video_latents=packing.patchify_video_latents,
                         KEYFRAME_NOISE_AUG_T=packing.KEYFRAME_NOISE_AUG_T)
    assert build(model, batch, (7, 8, 8), 'cpu', torch.float32) == (None, None, (), ())
    assert batch._h3_reference_keep[1] == (False, False, False)


def test_auxiliary_loss_uses_actual_flow_clean_estimate():
    loss = auxiliary.VideoAuxiliaryLosses([{'type': 'pixel_l1', 'weight': 1}], max_sigma=1, latent_frames=0)
    clean = torch.rand(1, 3, 2, 4, 4)
    noise = torch.rand_like(clean)
    sigma = torch.tensor([0.4])
    noisy = (1 - sigma) * clean + sigma * noise
    prediction = (noise - clean).requires_grad_()
    result, _ = loss(prediction, noisy, clean, sigma, lambda z: z)
    assert result.item() < 1e-7
    result.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()

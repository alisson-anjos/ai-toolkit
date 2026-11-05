"""CPU tests for toolkit/h3_subject_binding.py."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolkit import h3_subject_binding as sb  # noqa: E402


def track(*pts):
    return torch.tensor(pts, dtype=torch.float32)


def test_map_peaks_on_the_point_and_follows_it():
    s = sb.Subject(track((0.0, .25, .5), (1.0, .75, .5)), None, 1)
    m = sb.maps([s], num_latent_frames=3, grid_h=8, grid_w=8, num_frames=25, sigma=1.0)
    assert m.shape == (1, 3, 8, 8)
    first = torch.nonzero(m[0, 0] == m[0, 0].max())[0].tolist(); last = torch.nonzero(m[0, 2] == m[0, 2].max())[0].tolist()
    assert first[1] in (1, 2) and last[1] in (5, 6)          # moved from x=.25 to x=.75
    assert .7 < float(m.max()) <= 1     # a point between cell centres peaks a little under 1


def test_no_points_means_no_map():
    m = sb.maps([sb.Subject(torch.zeros(0, 3), None, 1)], 2, 4, 4, 17, 1.0)
    assert float(m.abs().sum()) == 0


def test_fresh_module_changes_nothing():
    mod = sb.SubjectBinding(hidden=16, vector_dim=8, max_subjects=2)
    s = [sb.Subject(track((0, .5, .5)), torch.randn(8), 1)]
    b = mod.video_bias(s, [slice(0, 4)], num_cond_rows=4, target_grid=(2, 3, 3), num_frames=9, sigma=1.)
    assert b.shape == (4 + 18, 16) and float(b.abs().sum()) == 0


def test_picture_rows_and_subject_region_share_the_tag():
    mod = sb.SubjectBinding(hidden=4, vector_dim=3, max_subjects=2)
    with torch.no_grad(): mod.slot.copy_(torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0]]))
    subjects = [sb.Subject(track((0, .1, .1)), None, 2), sb.Subject(track((0, .9, .9)), None, 1)]
    b = mod.video_bias(subjects, [slice(0, 2), slice(2, 4)], num_cond_rows=4, target_grid=(1, 4, 4), num_frames=5, sigma=.6)
    assert torch.allclose(b[2], torch.tensor([1., 0, 0, 0])) and torch.allclose(b[0], torch.tensor([0, 1., 0, 0]))
    target = b[4:].reshape(4, 4, 4)
    assert target[0, 0, 0] > target[0, 0, 1] and target[3, 3, 1] > target[3, 3, 0]


def test_gradients_reach_slot_and_projection():
    mod = sb.SubjectBinding(hidden=6, vector_dim=5, max_subjects=1)
    b = mod.video_bias([sb.Subject(track((0, .5, .5)), torch.randn(5), 1)], [slice(0, 1)], 1, (1, 2, 2), 5, 1.)
    (b * torch.randn_like(b)).sum().backward()
    assert mod.slot.grad.abs().sum() > 0 and mod.proj.weight.grad.abs().sum() > 0


def test_model_builds_the_batch_bias_from_molmo_files(tmp_path):
    """MinimaxH3Ref2VAModel._subject_video_bias: reads molmo/<stem>.npz beside the targets folder, tags the Picture rows
    found in ref_blocks (pictures are the 1-frame blocks) and the target rows."""
    import importlib.util, types, numpy as np
    spec = importlib.util.spec_from_file_location("mh3", Path(__file__).resolve().parents[1] / "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py",
                                                  submodule_search_locations=[])
    try:
        from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import MinimaxH3Ref2VAModel as M
    except Exception as e:   # heavy deps missing on this machine
        import pytest; pytest.skip(f"model module not importable here: {e}")
    (tmp_path / "targets").mkdir(); (tmp_path / "molmo").mkdir()
    np.savez(tmp_path / "molmo" / "clip1.npz", tracks_1=np.array([[0, .5, .5]], np.float32), vector_1=np.ones(8, np.float32))
    mod = sb.SubjectBinding(hidden=4, vector_dim=8, max_subjects=2)
    with torch.no_grad(): mod.slot[0] = torch.tensor([1., 2, 3, 4])
    fake = types.SimpleNamespace(subject_binding=mod, subject_binding_options={"sigma": 1.0}, load_subjects=M.load_subjects)
    batch = types.SimpleNamespace(file_items=[types.SimpleNamespace(path=str(tmp_path / "targets" / "clip1.mp4"))])
    # one picture block (1 x 4 x 4 latents -> 4 rows) then a motion video block (2 x 4 x 4 -> 8 rows)
    bias = M._subject_video_bias(fake, batch, ((1, 4, 4, 0), (2, 4, 4, 0)), 12, (1, 2, 2), 5, "cpu", torch.float32)
    assert bias.shape == (1, 12 + 4, 4)
    assert torch.allclose(bias[0, 0], torch.tensor([1., 2, 3, 4])) and float(bias[0, 4:12].abs().sum()) == 0
    assert float(bias[0, 12:].abs().sum()) > 0

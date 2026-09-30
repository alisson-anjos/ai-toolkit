"""An aligned reference must land on the target's coordinates exactly.

That is the whole point of the flag: a guide row and the target row it drives share a rotary
position, so attention between them costs nothing positionally. If the clock or the spatial grid
drifts by even one unit the guide stops being frame-accurate, and the failure is invisible —
training still runs, the model just never learns tight control.
"""

import pytest
import torch

from minimax_h3_test_utils import packing, model_method

build_packed_sequence = packing.build_packed_sequence

TEXT = torch.ones(4, dtype=torch.long)
FRAMES, H, W = 7, 8, 8


def _layout(ref_blocks=(), aligned_refs=()):
    return build_packed_sequence(
        text_token_tags=TEXT,
        num_latent_frames=FRAMES,
        latent_height=H,
        latent_width=W,
        num_audio_latents=0,
        ref_blocks=ref_blocks,
        aligned_refs=aligned_refs,
    )


def _rows(layout, start, count):
    return layout.position_ids[start : start + count]


def test_aligned_reference_matches_the_target_coordinates_row_for_row() -> None:
    layout = _layout(ref_blocks=((FRAMES, H, W),), aligned_refs=(True,))

    rows_per_ref = FRAMES * (H // 2) * (W // 2)
    guide = _rows(layout, layout.video_indices[0].item(), rows_per_ref)
    target = layout.position_ids[-rows_per_ref:]

    assert torch.equal(guide, target)


def test_aligned_reference_does_not_shift_the_target_clock() -> None:
    """The target must sit where it would with no reference at all."""
    bare = _layout()
    aligned = _layout(ref_blocks=((FRAMES, H, W),), aligned_refs=(True,))

    rows = FRAMES * (H // 2) * (W // 2)
    assert torch.equal(bare.position_ids[-rows:], aligned.position_ids[-rows:])


def test_unaligned_reference_still_sits_beside_the_target() -> None:
    """The default path is untouched: an ordinary reference keeps its own clock."""
    layout = _layout(ref_blocks=((1, H, W),), aligned_refs=(False,))

    rows_per_ref = (H // 2) * (W // 2)
    ref = _rows(layout, layout.video_indices[0].item(), rows_per_ref)
    target = layout.position_ids[-FRAMES * rows_per_ref :]

    assert ref[:, 0].max() < target[:, 0].min()


def test_omitting_the_flag_keeps_upstream_behaviour() -> None:
    explicit = _layout(ref_blocks=((1, H, W),), aligned_refs=(False,))
    implicit = _layout(ref_blocks=((1, H, W),))

    assert torch.equal(explicit.position_ids, implicit.position_ids)


def test_head_swap_pack_aligns_only_the_guide() -> None:
    """Guide aligned to the target, identity reference parked beside it."""
    layout = _layout(ref_blocks=((FRAMES, H, W), (1, H, W)), aligned_refs=(True, False))

    rows_per_frame = (H // 2) * (W // 2)
    guide_rows = FRAMES * rows_per_frame
    start = layout.video_indices[0].item()

    guide = _rows(layout, start, guide_rows)
    identity = _rows(layout, start + guide_rows, rows_per_frame)
    target = layout.position_ids[-guide_rows:]

    assert torch.equal(guide, target)
    assert identity[:, 0].max() < target[:, 0].min()


def test_shorter_aligned_guide_starts_at_the_targets_first_frame() -> None:
    short = 3
    layout = _layout(ref_blocks=((short, H, W),), aligned_refs=(True,))

    rows_per_frame = (H // 2) * (W // 2)
    guide = _rows(layout, layout.video_indices[0].item(), short * rows_per_frame)
    target = layout.position_ids[-FRAMES * rows_per_frame :]

    assert torch.equal(guide, target[: short * rows_per_frame])


def test_aligned_reference_must_match_the_target_resolution() -> None:
    """A different grid cannot be aligned, and silently misaligning is the worst outcome."""
    with pytest.raises(ValueError, match="aligned reference"):
        _layout(ref_blocks=((FRAMES, H // 2, W),), aligned_refs=(True,))


class _FakeModel:
    """Just enough of the model to exercise the flag's plumbing."""

    def __init__(self, enabled):
        self.model_config = type("C", (), {"model_kwargs": {"align_video_refs": enabled}})()

    _aligned_ref_flags = None  # bound below


def _flags(enabled, ref_blocks):
    return model_method("_aligned_ref_flags")(_FakeModel(enabled), ref_blocks)


def test_flag_off_aligns_nothing() -> None:
    assert _flags(False, ((7, 8, 8, 0), (1, 8, 8, 0))) == ()


def test_flag_on_aligns_video_blocks_only() -> None:
    """Video reference becomes the guide; the image reference stays a reference."""
    assert _flags(True, ((7, 8, 8, 0), (1, 8, 8, 0))) == (True, False)


def test_no_references_needs_no_flags() -> None:
    assert _flags(True, ()) == ()

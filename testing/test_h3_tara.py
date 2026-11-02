"""Unit tests for MiniMax H3 TARA (toolkit/h3_native_temporal.py).

Run: python -m pytest testing/test_h3_tara.py
CPU-only tests use the dense oracle; the FlexAttention test needs CUDA.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toolkit import h3_native_temporal as nt  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_h3_tara_test_packing", ROOT / "extensions_built_in/diffusion_models/minimax_h3/src/packing.py")
packing = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = packing
_spec.loader.exec_module(packing)

ON = nt.options({"native_temporal_alignment": True})
PICTURE = (1, 4, 8)
MOTION = (7, 8, 4)


def build(text=4, frames=7, blocks=(PICTURE, MOTION), ids=(2, 1)):
    layout = packing.build_packed_sequence(torch.ones(text, dtype=torch.long), frames, 8, 8, 0, ref_blocks=blocks)
    before = layout.position_ids.clone()
    mask = nt.align_layout(layout, blocks, ids, ON)
    return layout, mask, before


def test_options_default_off_and_validated():
    assert nt.options({})["enabled"] is False
    assert ON["strength"] == 2.0 and ON["window"] == 0.2 and ON["falloff"] == 0.25 and ON["cap"] == 6.0
    with pytest.raises(ValueError):
        nt.options({"native_temporal_alignment": "yes"})
    with pytest.raises(ValueError):
        nt.options({"native_temporal_falloff_seconds": 0})
    with pytest.raises(ValueError):
        nt.options({"native_motion_source_channel": 0})


def test_only_the_motion_clock_moves():
    layout, mask, before = build()
    assert int(mask.sum()) == 7 * 4 * 2
    # picture, text and target rows keep every coordinate; motion keeps its spatial grid
    assert torch.equal(layout.position_ids[~mask], before[~mask])
    assert torch.equal(layout.position_ids[mask, 1:], before[mask, 1:])
    target = layout.video_indices[layout.num_condition_video_rows:]
    torch.testing.assert_close(torch.unique_consecutive(layout.position_ids[mask, 0]),
                               torch.unique_consecutive(layout.position_ids[target, 0]), rtol=0, atol=1e-9)


def test_motion_is_found_by_channel_not_position():
    layout, mask, _ = build(blocks=(MOTION, PICTURE), ids=(1, 2))
    assert mask[4] and not mask[-1]


def test_mismatched_temporal_grids_are_rejected():
    with pytest.raises(ValueError, match="temporal grid"):
        build(blocks=((6, 8, 8),), ids=(1,))


def test_missing_motion_channel_is_rejected():
    with pytest.raises(ValueError, match="exactly one motion video"):
        build(ids=(2, 3))


def test_a_picture_is_not_a_motion_video():
    with pytest.raises(ValueError, match="video"):
        build(blocks=((1, 8, 8),), ids=(1,))


def test_off_returns_an_empty_mask_and_keeps_positions():
    layout = packing.build_packed_sequence(torch.ones(4, dtype=torch.long), 7, 8, 8, 0, ref_blocks=(PICTURE, MOTION))
    before = layout.position_ids.clone()
    mask = nt.align_layout(layout, (PICTURE, MOTION), (), nt.options({}))
    assert not mask.any() and torch.equal(layout.position_ids, before)


def test_batch_padding_keeps_row_ownership():
    a, ma, _ = build(text=4)
    b, mb, _ = build(text=8)
    masks = nt.pad_motion_masks([a, b], [ma, mb])
    assert not masks[0, 4:8].any()
    assert torch.equal(masks[0, 8:], ma[4:]) and torch.equal(masks[1], mb)


def make_context(strength=2.0, device="cpu"):
    times = torch.arange(64, device=device).float()[None] / 24
    source = torch.zeros(1, 64, dtype=torch.bool, device=device)
    source[:, 8:32] = True
    target = torch.zeros_like(source)
    target[:, 32:] = True
    live = torch.ones_like(source)
    live[:, -1] = False
    policy = torch.tensor([strength, 0.2, 0.25, 6.0], device=device)
    return nt.TemporalContext(times, source, target, live, policy)


def test_bias_only_on_target_to_motion_edges():
    modify = nt.score_modifier(make_context())
    zero, i = torch.tensor(0.0), torch.tensor
    assert modify(zero, i(0), i(0), i(33), i(2)) == 0    # target -> text/picture
    assert modify(zero, i(0), i(0), i(10), i(30)) == 0   # motion -> motion
    assert modify(zero, i(0), i(0), i(33), i(30)) == 0   # inside the free window
    assert modify(zero, i(0), i(0), i(50), i(8)) == -6   # far apart: capped
    assert torch.isneginf(modify(zero, i(0), i(0), i(33), i(63)))  # padding


def test_zero_strength_is_plain_attention():
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 64, 32) for _ in range(3))
    ctx = make_context(strength=0.0)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=ctx.live[:, None, None])
    torch.testing.assert_close(nt.dense_reference(q, k, v, ctx), expected, rtol=1e-5, atol=1e-6)


def test_dense_reference_refuses_production_lengths():
    q = torch.zeros(1, 1, 4097, 8)
    with pytest.raises(ValueError, match="short test"):
        nt.dense_reference(q, q, q, make_context())


def test_control_source_channels_keep_ui_numbers():
    from toolkit.dataloader_mixins import control_source_channels
    assert control_source_channels(SimpleNamespace(), 2) == [1, 2]
    assert control_source_channels(SimpleNamespace(control_path_1=None, control_path_2="a", control_path_3="b"), 2) == [2, 3]
    with pytest.raises(ValueError):
        control_source_channels(SimpleNamespace(control_path_1="a"), 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flex_attention_matches_dense_forward_and_backward():
    torch.manual_seed(11)
    ctx = make_context(device="cuda")
    q, k, v = (torch.randn(1, 2, 64, 32, device="cuda", requires_grad=True) for _ in range(3))
    actual = nt.attention(q, k, v, ctx)
    expected = nt.dense_reference(q, k, v, ctx)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
    for a, e in zip(torch.autograd.grad(actual.square().sum(), (q, k, v)),
                    torch.autograd.grad(expected.square().sum(), (q, k, v))):
        torch.testing.assert_close(a, e, rtol=5e-3, atol=5e-4)


def test_tiny_transformer_forward_backward_with_tara():
    """End to end through MiniMaxH3Transformer (CPU dense oracle path)."""
    spec = importlib.util.spec_from_file_location(
        "_h3_tara_test_transformer", ROOT / "extensions_built_in/diffusion_models/minimax_h3/src/transformer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    params = mod.MiniMaxH3TransformerParams(
        hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
        ffn_hidden_size=128, latents_dim=4, audio_latents_dim=4, text_dim=32, timestep_input_dim=32,
        time_embed_hidden_size=64, time_embed_dim=64, rope_inv_freq_len=4)
    torch.manual_seed(0)
    model = mod.MiniMaxH3Transformer(params)
    model.gradient_checkpointing = True
    layout, mask, _ = build()

    def run(options, motion_mask):
        return model(
            hidden_states=torch.randn(1, layout.video_indices.numel(), 16, generator=torch.Generator().manual_seed(1)),
            audio_hidden_states=torch.zeros(1, layout.audio_indices.numel(), 4),
            encoder_hidden_states=torch.randn(1, 4, 32, generator=torch.Generator().manual_seed(2)),
            row_timesteps=torch.full((1, layout.sequence_length), 0.5), token_tags=layout.token_tags[None],
            position_ids=layout.position_ids[None], video_indices=layout.video_indices,
            audio_indices=layout.audio_indices, text_indices=layout.text_indices, vsa_video_grid=(7, 4, 4),
            native_motion_mask=motion_mask, native_temporal_options=options)

    with_tara = run(ON, mask[None])
    loss = sum(x.square().mean() for x in with_tara if x.numel())
    loss.backward()
    grad = model.blocks[0].attn.qkv_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        # zero strength + zero cap must equal the plain path on the same positions
        off = run(nt.options({"native_temporal_alignment": True, "native_temporal_strength": 0.0,
                              "native_temporal_max_penalty": 0.0}), mask[None])
        plain = run(None, None)
    torch.testing.assert_close(off[0], plain[0], rtol=1e-4, atol=1e-5)

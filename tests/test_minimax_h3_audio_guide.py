"""The audio guide packs exactly like ComfyUI's "Add Guide for MiniMax H3" with an audio at frame 0.

An NKD Event Curve guide is anchored that way at inference, so training must put the guide's
rows at the same rotary positions, in the same order, or the LoRA learns a layout the sampler
never shows it. The ComfyUI side needs a ComfyUI checkout: set COMFYUI_PATH to run it.
"""
import os
import subprocess
import sys

import pytest
import torch

from extensions_built_in.diffusion_models.minimax_h3.src.packing import (
    AUDIO_CHANNELS,
    build_packed_sequence,
)

TEXT, T_LAT, H_LAT, W_LAT = 37, 9, 22, 40          # 141 frames at 352x640
A_LAT = 235                                          # 5.875 s at 40 latents/s
REFS = [(30, 30), (36, 20)]                          # two image references (latent h, w)


def trainer_layout(guide_latents=A_LAT, refs=REFS):
    blocks = [(0, H_LAT, W_LAT, guide_latents)] + [(1, h, w, 0) for h, w in refs]
    aligned = [True] + [False] * len(refs)
    return build_packed_sequence(
        text_token_tags=torch.ones(TEXT, dtype=torch.long),
        num_latent_frames=T_LAT, latent_height=H_LAT, latent_width=W_LAT,
        num_audio_latents=A_LAT, ref_blocks=tuple(blocks), aligned_refs=tuple(aligned),
    )


def test_guide_rows_sit_on_the_target_clock():
    lay = trainer_layout()
    plain = build_packed_sequence(
        text_token_tags=torch.ones(TEXT, dtype=torch.long),
        num_latent_frames=T_LAT, latent_height=H_LAT, latent_width=W_LAT,
        num_audio_latents=A_LAT, ref_blocks=tuple((1, h, w, 0) for h, w in REFS),
    )
    n_guide = A_LAT * AUDIO_CHANNELS
    assert lay.num_condition_audio_rows == n_guide
    assert lay.num_condition_video_rows == plain.num_condition_video_rows
    # the guide does not move the media clock: the targets keep their positions
    assert torch.equal(lay.position_ids[-(plain.sequence_length - TEXT - plain.num_condition_video_rows):],
                       plain.position_ids[TEXT + plain.num_condition_video_rows:])
    guide = lay.position_ids[lay.audio_indices[:n_guide]]
    target = lay.position_ids[lay.audio_indices[n_guide:]]
    assert torch.equal(guide, target)                # same time and width as the target soundtrack
    # and the guide rows come first after the text, like the ComfyUI keyframe segment
    assert int(lay.audio_indices[0]) == TEXT


def test_shorter_guide_starts_at_frame_zero():
    lay = trainer_layout(guide_latents=100)
    n = 100 * AUDIO_CHANNELS
    guide = lay.position_ids[lay.audio_indices[:n]]
    target = lay.position_ids[lay.audio_indices[n:]]
    assert torch.equal(guide[:100], target[:100])
    assert torch.equal(guide[100:], target[A_LAT:A_LAT + 100])


_COMFY_SCRIPT = r"""
import sys, torch
args = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]          # ComfyUI reads its CLI flags on import
sys.path.insert(0, args[0])
import comfy.options
comfy.options.enable_args_parsing()
from comfy.ldm.minimax.model import PackedLayout
text, t, h, w, a, g = map(int, args[1:7])
out = args[7]
refs = [tuple(map(int, r.split("x"))) for r in args[8:]]
lay = PackedLayout(text, t, h, w, a,
                   keyframes=[{"resolved_frame_index": 0, "audio_latent": torch.zeros(1, 32, 2, g)}],
                   refs=[{"kind": "image", "latent_h": rh, "latent_w": rw} for rh, rw in refs])
torch.save({"seq_len": lay.seq_len, "position_ids": lay.position_ids, "audio_pos": lay.audio_pos,
            "audio_update": lay.audio_update, "img_pos": lay.img_pos, "img_update": lay.img_update}, out)
"""


def _comfy_layout(tmp_path, guide_latents=A_LAT, refs=REFS):
    """ComfyUI's own layout, built by ComfyUI's Python (its deps differ from the trainer's).
    Set COMFYUI_PATH, and COMFYUI_PYTHON when ComfyUI runs from another environment."""
    root = os.environ.get("COMFYUI_PATH")
    if not root:
        pytest.skip("set COMFYUI_PATH to compare with ComfyUI's packing")
    python = os.environ.get("COMFYUI_PYTHON", sys.executable)
    out = tmp_path / "comfy_layout.pt"
    args = [python, "-c", _COMFY_SCRIPT, root, *map(str, (TEXT, T_LAT, H_LAT, W_LAT, A_LAT, guide_latents)),
            str(out), *[f"{h}x{w}" for h, w in refs]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(args, capture_output=True, text=True, env=env, cwd=root)
    if proc.returncode != 0:
        pytest.fail("ComfyUI layout script failed:\n" + proc.stderr[-2000:])
    return torch.load(out)


@pytest.mark.parametrize("guide_latents", [A_LAT, 100])
def test_matches_comfyui_add_guide(tmp_path, guide_latents):
    comfy = _comfy_layout(tmp_path, guide_latents)
    ours = trainer_layout(guide_latents)
    assert comfy["seq_len"] == ours.sequence_length
    assert torch.allclose(comfy["position_ids"], ours.position_ids, atol=1e-9)
    # same rows are audio, and the same audio rows are conditioning (clean, not denoised)
    assert torch.equal(comfy["audio_pos"], ours.audio_indices)
    n_cond = int((~comfy["audio_update"]).sum())
    assert n_cond == ours.num_condition_audio_rows
    assert torch.equal(comfy["img_pos"][~comfy["img_update"]], ours.video_indices[:ours.num_condition_video_rows])

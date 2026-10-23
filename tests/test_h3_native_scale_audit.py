"""Prevent double-scaling H3's already area-normalized spatial RoPE grid."""
import math
import torch
from minimax_h3_test_utils import packing


def test_native_half_grid_already_matches_every_other_full_grid_origin():
    h,w=32,64
    for dim in (h,w):
        full=packing._spatial_position_grid(dim,2,math.sqrt(h*w))
        half=packing._spatial_position_grid(dim//2,2,math.sqrt((h//2)*(w//2)))
        assert torch.allclose(half,full[::2],atol=1e-12,rtol=0)


def test_changed_aspect_changes_normalized_support():
    # Grid rounding can change aspect: do not treat it as uniform downscale.
    full=packing._spatial_position_grid(26,2,math.sqrt(26*46))
    rounded=packing._spatial_position_grid(12,2,math.sqrt(12*22))
    assert not torch.isclose(full[0],rounded[0],atol=1e-10,rtol=0)

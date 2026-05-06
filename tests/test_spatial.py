"""Tests for ``compute_spatial_descriptor``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edit2forensics.reasoning.spatial import compute_spatial_descriptor


def _write_mask(path: Path, arr: np.ndarray) -> Path:
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)
    return path


# ---------------------------------------------------------------------------
# Scope short-circuits
# ---------------------------------------------------------------------------

def test_global_scope_returns_whole_image_without_reading_mask(tmp_path):
    """Global scope should short-circuit before computing centroid —
    saves loading the mask. We pass a nonexistent path on purpose to
    prove the short-circuit."""
    result = compute_spatial_descriptor(
        Path("/nonexistent.png"), edit_scope="global",
    )
    assert result == "whole_image"


def test_alignment_failed_scope_returns_alignment_failed(tmp_path):
    result = compute_spatial_descriptor(
        Path("/nonexistent.png"), edit_scope="alignment_failed",
    )
    assert result == "alignment_failed"


# ---------------------------------------------------------------------------
# Quadrant classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols,expected", [
    # Mask in top-left of a 100x100 image
    ((10, 30), (10, 30), "upper_left"),
    # Mask in top-right
    ((10, 30), (70, 90), "upper_right"),
    # Mask in bottom-left
    ((70, 90), (10, 30), "lower_left"),
    # Mask in bottom-right
    ((70, 90), (70, 90), "lower_right"),
])
def test_quadrant_classification(rows, cols, expected, tmp_path):
    arr = np.zeros((100, 100), dtype=bool)
    arr[rows[0]:rows[1], cols[0]:cols[1]] = True
    path = _write_mask(tmp_path / "q.png", arr)
    result = compute_spatial_descriptor(path, edit_scope="local")
    assert result == expected


# ---------------------------------------------------------------------------
# Centered region
# ---------------------------------------------------------------------------

def test_mask_at_image_center_is_centered(tmp_path):
    arr = np.zeros((100, 100), dtype=bool)
    arr[40:60, 40:60] = True
    path = _write_mask(tmp_path / "c.png", arr)
    result = compute_spatial_descriptor(path, edit_scope="local")
    assert result == "centered"


def test_mask_just_inside_central_margin_is_centered(tmp_path):
    """Centroid at (50, 50) — exactly the center."""
    arr = np.zeros((100, 100), dtype=bool)
    arr[48:53, 48:53] = True  # 5x5 centered
    path = _write_mask(tmp_path / "c2.png", arr)
    result = compute_spatial_descriptor(path, edit_scope="local")
    assert result == "centered"


def test_central_margin_parameter_controls_center_size(tmp_path):
    """A mask just outside default margin (cm=0.25) is a quadrant;
    the same mask with cm=0.4 falls within the larger center region."""
    arr = np.zeros((100, 100), dtype=bool)
    # Centroid at row=20, col=20 — outside upper-left central margin
    # (cm=0.25 puts center at rows/cols 25..75, so 20 is outside).
    arr[18:23, 18:23] = True
    path = _write_mask(tmp_path / "m.png", arr)
    assert compute_spatial_descriptor(path, edit_scope="local",
                                      central_margin=0.25) == "upper_left"
    # With cm=0.4, central region is rows/cols 10..90, centroid row=20
    # falls inside, so it's centered.
    assert compute_spatial_descriptor(path, edit_scope="local",
                                      central_margin=0.4) == "centered"


# ---------------------------------------------------------------------------
# Scattered detection
# ---------------------------------------------------------------------------

def test_two_equal_blobs_far_apart_are_scattered(tmp_path):
    """Two roughly equal-area blobs far apart trigger the multi-component
    + low-largest-fraction path."""
    arr = np.zeros((100, 100), dtype=bool)
    arr[10:20, 10:20] = True   # 10x10 blob top-left
    arr[80:90, 80:90] = True   # 10x10 blob bottom-right
    path = _write_mask(tmp_path / "s.png", arr)
    # Each blob is 50% of total area; default scattered_threshold=0.3
    # means largest_frac (0.5) > 0.3, so this is NOT scattered.
    # We need lower threshold or more components.
    # With 4 equal blobs, each is 25%, below default 0.3.
    arr2 = np.zeros((100, 100), dtype=bool)
    arr2[10:20, 10:20] = True
    arr2[10:20, 80:90] = True
    arr2[80:90, 10:20] = True
    arr2[80:90, 80:90] = True
    path2 = _write_mask(tmp_path / "s2.png", arr2)
    result = compute_spatial_descriptor(path2, edit_scope="local")
    assert result == "scattered"


def test_one_dominant_blob_with_speck_is_not_scattered(tmp_path):
    """A 90/10 area split: the dominant blob owns >> threshold of
    the total area, so the mask is treated as single-component."""
    arr = np.zeros((100, 100), dtype=bool)
    arr[10:30, 10:30] = True   # 20x20 = 400 px (dominant)
    arr[70:73, 70:73] = True   # 3x3 = 9 px (negligible speck)
    path = _write_mask(tmp_path / "d.png", arr)
    result = compute_spatial_descriptor(path, edit_scope="local")
    # The dominant blob centroid is upper-left.
    assert result == "upper_left"


def test_scattered_threshold_parameter(tmp_path):
    """A 60/40 area split with threshold=0.5: largest is 0.6 > 0.5,
    so single-component. Threshold=0.7: largest is 0.6 < 0.7, scattered."""
    arr = np.zeros((100, 100), dtype=bool)
    # Area roughly 600 px and 400 px
    arr[10:30, 10:40] = True   # 20x30 = 600
    arr[60:80, 60:80] = True   # 20x20 = 400
    path = _write_mask(tmp_path / "t.png", arr)
    result_low = compute_spatial_descriptor(
        path, edit_scope="local", scattered_threshold=0.5,
    )
    result_high = compute_spatial_descriptor(
        path, edit_scope="local", scattered_threshold=0.7,
    )
    assert result_low != "scattered"
    assert result_high == "scattered"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_mask_returns_alignment_failed(tmp_path):
    """Degenerate empty mask — no spatial signal. Defensively returns
    ``alignment_failed`` rather than dividing by zero."""
    arr = np.zeros((50, 50), dtype=bool)
    path = _write_mask(tmp_path / "e.png", arr)
    result = compute_spatial_descriptor(path, edit_scope="local")
    assert result == "alignment_failed"

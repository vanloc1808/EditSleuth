"""Tests for ``scripts/add_mask_compactness.py`` core logic.

The compactness function combines two geometric features:
  area_to_bbox     = mask_area / (bbox_h * bbox_w)
  largest_frac     = area_of_largest_component / total_mask_area
  compactness      = sqrt(area_to_bbox * largest_frac)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Script lives in scripts/, not the package — direct import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from add_mask_compactness import _compute_compactness  # noqa: E402


def _write_mask(path: Path, arr: np.ndarray) -> Path:
    """Write a boolean array as a single-channel mask PNG."""
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)
    return path


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------

def test_empty_mask_returns_zero(tmp_path):
    """A mask with no foreground pixels is degenerate; we return 0.0
    by convention. The downstream scorer reads this as 'maximally
    diffuse' (compactness_score = 1.0) which is the right default
    for a degenerate input."""
    arr = np.zeros((32, 32), dtype=bool)
    path = _write_mask(tmp_path / "empty.png", arr)
    assert _compute_compactness(path) == 0.0


def test_full_mask_returns_one(tmp_path):
    """An all-ones mask (global scope from Stage B) is maximally
    compact: it perfectly fills its bounding box AND has only one
    connected component."""
    arr = np.ones((32, 32), dtype=bool)
    path = _write_mask(tmp_path / "full.png", arr)
    assert _compute_compactness(path) == pytest.approx(1.0, abs=1e-6)


def test_single_solid_blob_is_one(tmp_path):
    """A single solid square should have compactness = 1.0:
    perfectly fills its bbox, single component."""
    arr = np.zeros((32, 32), dtype=bool)
    arr[5:15, 5:15] = True
    path = _write_mask(tmp_path / "blob.png", arr)
    assert _compute_compactness(path) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Shape sensitivity (area_to_bbox term)
# ---------------------------------------------------------------------------

def test_l_shape_has_lower_compactness_than_solid_square(tmp_path):
    """An L-shape doesn't fill its bounding box (the inner corner is
    background), so area_to_bbox < 1, so compactness < 1."""
    arr = np.zeros((32, 32), dtype=bool)
    # L-shape: full row plus a partial column
    arr[10:20, 5:15] = True   # 10x10 square = 100 px
    arr[10:30, 5:10] = True   # 20x5 column = 100 px (overlaps top half of square)
    # Actual filled cells: 10x10 (top square) + 10x5 (bottom half of column) = 150
    # bbox: rows 10..29 (20), cols 5..14 (10) => bbox area = 200
    # area_to_bbox = 150/200 = 0.75
    # single connected component: largest_frac = 1.0
    # compactness = sqrt(0.75 * 1.0) = sqrt(0.75) ≈ 0.866
    path = _write_mask(tmp_path / "l.png", arr)
    val = _compute_compactness(path)
    assert val == pytest.approx(np.sqrt(0.75), abs=1e-3)


def test_thin_strip_has_compactness_one(tmp_path):
    """A perfectly straight 1-pixel-wide strip *does* fill its
    bounding box (the bbox is also 1 pixel wide). The geometric
    'compactness' captured here is bbox-relative, not perimeter-
    relative — a perfect line is bbox-compact even though it is
    not perimetrically compact. This test pins that contract.
    """
    arr = np.zeros((32, 32), dtype=bool)
    arr[15, 5:25] = True
    path = _write_mask(tmp_path / "strip.png", arr)
    val = _compute_compactness(path)
    # bbox is 1x20 = 20, area is 20; area_to_bbox = 1.0
    # single component; compactness = 1.0
    assert val == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Component sensitivity (largest_frac term)
# ---------------------------------------------------------------------------

def test_two_equal_blobs_have_lower_compactness(tmp_path):
    """Two equal-sized blobs share the area equally, so largest_frac
    = 0.5. The bbox spans both blobs, inflating bbox_area, so
    area_to_bbox is also low. Compactness should be substantially
    less than 1."""
    arr = np.zeros((32, 32), dtype=bool)
    arr[5:10, 5:10] = True   # 5x5 = 25 px
    arr[20:25, 20:25] = True  # 5x5 = 25 px
    # Total area = 50
    # bbox: rows 5..24 (20), cols 5..24 (20), bbox area = 400
    # area_to_bbox = 50/400 = 0.125
    # largest_frac = 25/50 = 0.5
    # compactness = sqrt(0.125 * 0.5) = sqrt(0.0625) = 0.25
    path = _write_mask(tmp_path / "two.png", arr)
    val = _compute_compactness(path)
    assert val == pytest.approx(0.25, abs=1e-3)


def test_scattered_small_blobs_have_low_compactness(tmp_path):
    """A scatter of many small blobs spread across the image has
    compactness ≪ 1 because both terms drop: area_to_bbox shrinks
    (large bbox vs small area) and largest_frac shrinks (no single
    blob owns most of the area)."""
    arr = np.zeros((32, 32), dtype=bool)
    # Four 2x2 blobs at the corners-ish.
    arr[2:4, 2:4] = True
    arr[2:4, 28:30] = True
    arr[28:30, 2:4] = True
    arr[28:30, 28:30] = True
    # Each blob is 4 px; total area = 16
    # bbox: rows 2..29 (28), cols 2..29 (28), bbox area = 784
    # area_to_bbox = 16/784 ≈ 0.0204
    # largest_frac = 4/16 = 0.25
    # compactness ≈ sqrt(0.0204 * 0.25) ≈ sqrt(0.0051) ≈ 0.0714
    path = _write_mask(tmp_path / "scatter.png", arr)
    val = _compute_compactness(path)
    assert val < 0.10
    assert val > 0.05  # but not zero — there is *some* concentration


def test_one_big_blob_one_small_blob(tmp_path):
    """A 90/10 area split: largest_frac = 0.9. bbox now spans both
    blobs but most of the area is concentrated in one. Compactness
    is determined mainly by area_to_bbox in this case."""
    arr = np.zeros((32, 32), dtype=bool)
    arr[5:15, 5:15] = True    # 10x10 = 100 px (big blob)
    arr[28:30, 28:30] = True  # 2x2 = 4 px (small blob)
    # Total = 104; bbox spans rows 5..29 (25), cols 5..29 (25); bbox area = 625
    # area_to_bbox = 104/625 ≈ 0.166
    # largest_frac = 100/104 ≈ 0.962
    # compactness = sqrt(0.166 * 0.962) ≈ 0.4
    path = _write_mask(tmp_path / "big_small.png", arr)
    val = _compute_compactness(path)
    assert val == pytest.approx(np.sqrt(0.166 * 0.962), abs=2e-2)


# ---------------------------------------------------------------------------
# Range and dtype guarantees
# ---------------------------------------------------------------------------

def test_compactness_always_in_unit_interval(tmp_path):
    """For random binary masks of various densities, compactness must
    stay in [0, 1]."""
    rng = np.random.default_rng(0)
    for i, density in enumerate([0.05, 0.2, 0.5, 0.8]):
        arr = rng.uniform(0, 1, size=(32, 32)) < density
        path = tmp_path / f"r{i}.png"
        _write_mask(path, arr)
        val = _compute_compactness(path)
        assert 0.0 <= val <= 1.0, f"out of range at density {density}: {val}"


def test_returns_float_not_numpy_scalar(tmp_path):
    """The artifact's parquet roundtrip needs builtin floats, not
    numpy scalars (asdict-style serialization preserves the latter
    in unhelpful ways)."""
    arr = np.zeros((32, 32), dtype=bool)
    arr[5:15, 5:15] = True
    path = _write_mask(tmp_path / "f.png", arr)
    val = _compute_compactness(path)
    assert isinstance(val, float)

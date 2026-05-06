"""Tests for the pure helper functions in ``scripts/compare_mask_runs.py``.

End-to-end Hydra wiring is exercised by the script's smoke run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# The script lives in scripts/, not the package — import directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_mask_runs import _load_mask_binary, _pairwise_iou  # noqa: E402


def _write_mask(path: Path, arr: np.ndarray) -> None:
    """Write a boolean array as a single-channel PNG."""
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)


# ---------------------------------------------------------------------------
# _load_mask_binary
# ---------------------------------------------------------------------------

def test_load_mask_binary_threshold(tmp_path):
    """Pixel values >127 are foreground; <=127 are background."""
    path = tmp_path / "m.png"
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[1, 1] = 255
    arr[2, 2] = 100  # below threshold
    Image.fromarray(arr, mode="L").save(path)
    out = _load_mask_binary(path)
    assert out.shape == (4, 4)
    assert out[1, 1] == True  # noqa: E712
    assert out[2, 2] == False  # noqa: E712
    assert out[0, 0] == False  # noqa: E712


def test_load_mask_binary_handles_rgb_input(tmp_path):
    """RGB white masks (MagicBrush convention) should still load as
    binary, treating any non-black pixel as foreground after grayscale
    conversion."""
    path = tmp_path / "rgb.png"
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[1, 1] = [255, 255, 255]
    Image.fromarray(arr, mode="RGB").save(path)
    out = _load_mask_binary(path)
    assert out.shape == (4, 4)
    assert out[1, 1] == True  # noqa: E712


# ---------------------------------------------------------------------------
# _pairwise_iou
# ---------------------------------------------------------------------------

def test_pairwise_iou_identical_masks(tmp_path):
    arr = np.zeros((16, 16), dtype=bool)
    arr[2:8, 2:8] = True
    p_a = tmp_path / "a.png"
    p_b = tmp_path / "b.png"
    _write_mask(p_a, arr)
    _write_mask(p_b, arr)
    iou = _pairwise_iou(p_a, p_b)
    assert iou == 1.0


def test_pairwise_iou_disjoint_masks(tmp_path):
    a = np.zeros((16, 16), dtype=bool)
    a[:8, :8] = True
    b = np.zeros((16, 16), dtype=bool)
    b[8:, 8:] = True
    p_a = tmp_path / "a.png"
    p_b = tmp_path / "b.png"
    _write_mask(p_a, a)
    _write_mask(p_b, b)
    iou = _pairwise_iou(p_a, p_b)
    assert iou == 0.0


def test_pairwise_iou_partial_overlap(tmp_path):
    """Two 8x8 squares overlapping in a 4x4 corner: intersection = 16,
    union = 64 + 64 - 16 = 112, IoU = 16/112 ≈ 0.1429."""
    a = np.zeros((16, 16), dtype=bool)
    a[:8, :8] = True
    b = np.zeros((16, 16), dtype=bool)
    b[4:12, 4:12] = True
    p_a = tmp_path / "a.png"
    p_b = tmp_path / "b.png"
    _write_mask(p_a, a)
    _write_mask(p_b, b)
    iou = _pairwise_iou(p_a, p_b)
    assert iou == pytest.approx(16 / 112, abs=1e-6)


def test_pairwise_iou_both_empty(tmp_path):
    """Two empty masks: union = 0. The function returns 1.0 (perfect
    agreement on "nothing edited") rather than NaN/error."""
    arr = np.zeros((16, 16), dtype=bool)
    p_a = tmp_path / "a.png"
    p_b = tmp_path / "b.png"
    _write_mask(p_a, arr)
    _write_mask(p_b, arr)
    iou = _pairwise_iou(p_a, p_b)
    assert iou == 1.0


def test_pairwise_iou_one_empty(tmp_path):
    """One empty mask, one populated: intersection = 0, union > 0.
    IoU = 0 (complete disagreement)."""
    a = np.zeros((16, 16), dtype=bool)
    b = np.zeros((16, 16), dtype=bool)
    b[2:8, 2:8] = True
    p_a = tmp_path / "a.png"
    p_b = tmp_path / "b.png"
    _write_mask(p_a, a)
    _write_mask(p_b, b)
    iou = _pairwise_iou(p_a, p_b)
    assert iou == 0.0


def test_pairwise_iou_resizes_when_shapes_differ(tmp_path):
    """Two masks at different resolutions: function nearest-neighbor-
    resizes one to the other and computes IoU. A 16x16 mask compared
    to its own nearest-resized 32x32 version should produce IoU = 1.0
    (modulo nearest-resize quantization, which is exact for powers of 2).
    """
    a = np.zeros((16, 16), dtype=bool)
    a[2:8, 2:8] = True
    # b is the same mask at 32x32.
    b = np.zeros((32, 32), dtype=bool)
    b[4:16, 4:16] = True  # 4x scaled equivalent of [2:8, 2:8]
    p_a = tmp_path / "a.png"
    p_b = tmp_path / "b.png"
    _write_mask(p_a, a)
    _write_mask(p_b, b)
    iou = _pairwise_iou(p_a, p_b)
    # Nearest-resizing 32x32 down to 16x16 picks every other pixel,
    # and the source content is at exactly 4x scale, so the resulting
    # mask should match `a` perfectly.
    assert iou == 1.0


def test_pairwise_iou_returns_none_when_file_missing(tmp_path):
    p_a = tmp_path / "exists.png"
    arr = np.zeros((4, 4), dtype=bool)
    arr[1, 1] = True
    _write_mask(p_a, arr)
    p_missing = tmp_path / "nope.png"
    assert _pairwise_iou(p_a, p_missing) is None
    assert _pairwise_iou(p_missing, p_a) is None

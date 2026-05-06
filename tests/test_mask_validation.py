"""Tests for the mask validation utilities (IoU, summary)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edit2forensics.mask.validation import (
    MaskComparison,
    compare_masks,
    summarize_comparisons,
)


def _write_mask(path: Path, arr: np.ndarray) -> None:
    """Write a boolean array as a white-on-black PNG."""
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)


def test_perfect_overlap(tmp_path):
    mask = np.zeros((32, 32), dtype=bool)
    mask[:16, :16] = True
    auto = tmp_path / "auto.png"
    gt = tmp_path / "gt.png"
    _write_mask(auto, mask)
    _write_mask(gt, mask)

    cmp = compare_masks(auto, gt, "t_perfect")
    assert cmp.iou == pytest.approx(1.0)
    assert cmp.dice == pytest.approx(1.0)
    assert cmp.gt_area_frac == pytest.approx(0.25)
    assert cmp.auto_area_frac == pytest.approx(0.25)


def test_disjoint_masks(tmp_path):
    a = np.zeros((32, 32), dtype=bool)
    a[:16, :16] = True
    b = np.zeros((32, 32), dtype=bool)
    b[16:, 16:] = True
    auto = tmp_path / "auto.png"
    gt = tmp_path / "gt.png"
    _write_mask(auto, a)
    _write_mask(gt, b)

    cmp = compare_masks(auto, gt, "t_disjoint")
    assert cmp.iou == 0.0
    assert cmp.dice == 0.0


def test_both_empty_is_perfect_agreement(tmp_path):
    empty = np.zeros((16, 16), dtype=bool)
    auto = tmp_path / "auto.png"
    gt = tmp_path / "gt.png"
    _write_mask(auto, empty)
    _write_mask(gt, empty)

    cmp = compare_masks(auto, gt, "t_empty")
    # Both empty -> IoU defined as 1.0 (see validation.py docstring).
    assert cmp.iou == 1.0
    assert cmp.dice == 1.0


def test_one_empty_one_full_is_zero(tmp_path):
    empty = np.zeros((16, 16), dtype=bool)
    full = np.ones((16, 16), dtype=bool)
    auto = tmp_path / "auto.png"
    gt = tmp_path / "gt.png"
    _write_mask(auto, empty)
    _write_mask(gt, full)

    cmp = compare_masks(auto, gt, "t_one_empty")
    assert cmp.iou == 0.0
    assert cmp.dice == 0.0


def test_resize_on_shape_mismatch(tmp_path):
    """GT at a different resolution must be resized nearest-neighbor, not error."""
    # Auto: 32x32 with top-left quadrant filled.
    a = np.zeros((32, 32), dtype=bool)
    a[:16, :16] = True
    # GT: 16x16 with top-left quadrant filled (same region, smaller resolution).
    g = np.zeros((16, 16), dtype=bool)
    g[:8, :8] = True
    auto = tmp_path / "auto.png"
    gt = tmp_path / "gt.png"
    _write_mask(auto, a)
    _write_mask(gt, g)

    cmp = compare_masks(auto, gt, "t_resize")
    # Should be high overlap (nearest-neighbor resize of the 16x16 to 32x32
    # recovers the same top-left quadrant).
    assert cmp.iou > 0.9


def test_summarize_empty_list():
    assert summarize_comparisons([]) == {"n": 0}


def test_summarize_produces_expected_fields():
    cmps = [
        MaskComparison("a", iou=0.9, dice=0.9, gt_area_frac=0.3, auto_area_frac=0.3),
        MaskComparison("b", iou=0.7, dice=0.75, gt_area_frac=0.2, auto_area_frac=0.25),
        MaskComparison("c", iou=0.5, dice=0.55, gt_area_frac=0.1, auto_area_frac=0.1),
        MaskComparison("d", iou=0.1, dice=0.15, gt_area_frac=0.4, auto_area_frac=0.5),
    ]
    summary = summarize_comparisons(cmps)
    assert summary["n"] == 4
    assert summary["mean_iou"] == pytest.approx((0.9 + 0.7 + 0.5 + 0.1) / 4)
    assert "iou_quartiles" in summary
    assert "iou_bins" in summary
    assert sum(summary["iou_bins"].values()) == 4
    assert summary["frac_iou_above_0_6"] == pytest.approx(0.5)


def test_summary_histogram_covers_all_bins():
    cmps = [
        MaskComparison(f"x{i}", iou=v, dice=v, gt_area_frac=0.1, auto_area_frac=0.1)
        for i, v in enumerate([0.05, 0.25, 0.45, 0.65, 0.85, 1.0])
    ]
    summary = summarize_comparisons(cmps)
    assert summary["iou_bins"]["[0.0, 0.2)"] == 1
    assert summary["iou_bins"]["[0.2, 0.4)"] == 1
    assert summary["iou_bins"]["[0.4, 0.6)"] == 1
    assert summary["iou_bins"]["[0.6, 0.8)"] == 1
    assert summary["iou_bins"]["[0.8, 1.0]"] == 2  # 0.85 and 1.0

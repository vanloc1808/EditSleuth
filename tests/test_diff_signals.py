"""Tests for diff signals."""
from __future__ import annotations

import numpy as np
import pytest

from edit2forensics.mask.signals import LabPixelDiff


def test_identical_images_give_zero_diff():
    """An image compared to itself must produce an all-zeros map."""
    img = np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    sig = LabPixelDiff()
    diff = sig.compute(img, img)
    assert diff.shape == (32, 32)
    assert diff.dtype == np.float32
    assert np.allclose(diff, 0.0)


def test_diff_is_bounded():
    """Output must lie in [0, 1]."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    diff = LabPixelDiff().compute(a, b)
    assert diff.min() >= 0.0
    assert diff.max() <= 1.0


def test_localized_edit_produces_localized_diff():
    """A pair differing only in one region should have its diff concentrated there."""
    h, w = 48, 48
    real = np.full((h, w, 3), 128, dtype=np.uint8)
    edited = real.copy()
    # Strong color change in the top-left quadrant.
    edited[:24, :24] = [220, 30, 30]

    diff = LabPixelDiff().compute(real, edited)
    assert diff[:24, :24].mean() > 0.5
    assert diff[24:, 24:].max() < 1e-3


def test_shape_mismatch_raises():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        LabPixelDiff().compute(a, b)


def test_non_rgb_raises():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected HxWx3"):
        LabPixelDiff().compute(a, b)


def test_percentile_validation():
    with pytest.raises(ValueError):
        LabPixelDiff(normalize_percentile=40)

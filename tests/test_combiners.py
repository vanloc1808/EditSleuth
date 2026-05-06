"""Tests for `SignalCombiner` strategies."""
from __future__ import annotations

import numpy as np
import pytest

from edit2forensics.mask.combiners import MaxCombiner, MeanCombiner


# ---------------------------------------------------------------------------
# MaxCombiner
# ---------------------------------------------------------------------------

def test_max_combiner_returns_pointwise_maximum():
    a = np.array([[0.1, 0.8], [0.9, 0.2]], dtype=np.float32)
    b = np.array([[0.4, 0.3], [0.7, 0.6]], dtype=np.float32)
    out = MaxCombiner().combine({"a": a, "b": b})
    assert np.allclose(out, [[0.4, 0.8], [0.9, 0.6]])


def test_max_combiner_single_signal_passthrough():
    m = np.array([[0.1, 0.9], [0.5, 0.3]], dtype=np.float32)
    out = MaxCombiner().combine({"only": m})
    assert np.allclose(out, m)


def test_max_combiner_preserves_value_range():
    """Max of inputs in [0, 1] stays in [0, 1]."""
    rng = np.random.default_rng(0)
    sigs = {
        f"s{i}": rng.uniform(0, 1, size=(16, 16)).astype(np.float32)
        for i in range(4)
    }
    out = MaxCombiner().combine(sigs)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_max_combiner_output_shape_matches_input_maps():
    a = np.zeros((8, 12), dtype=np.float32)
    b = np.ones((8, 12), dtype=np.float32)
    out = MaxCombiner().combine({"a": a, "b": b})
    assert out.shape == (8, 12)
    assert out.dtype in (np.float32, np.float64)  # numpy may upcast


def test_max_combiner_name():
    assert MaxCombiner().name == "max"


# ---------------------------------------------------------------------------
# MeanCombiner
# ---------------------------------------------------------------------------

def test_mean_combiner_returns_arithmetic_mean():
    a = np.array([[0.1, 0.8], [0.9, 0.2]], dtype=np.float32)
    b = np.array([[0.5, 0.4], [0.3, 0.6]], dtype=np.float32)
    out = MeanCombiner().combine({"a": a, "b": b})
    expected = (a + b) / 2.0
    assert np.allclose(out, expected)


def test_mean_combiner_with_three_signals():
    a = np.full((4, 4), 0.3, dtype=np.float32)
    b = np.full((4, 4), 0.6, dtype=np.float32)
    c = np.full((4, 4), 0.9, dtype=np.float32)
    out = MeanCombiner().combine({"a": a, "b": b, "c": c})
    assert np.allclose(out, 0.6)


def test_mean_combiner_compresses_dynamic_range():
    """Key property: a single saturating signal does NOT saturate the
    mean when other signals are low. This is the motivating observation
    for MeanCombiner as a noise-suppressing alternative to MaxCombiner.
    """
    saturating = np.ones((8, 8), dtype=np.float32)      # 1.0
    quiet_a = np.full((8, 8), 0.1, dtype=np.float32)
    quiet_b = np.full((8, 8), 0.2, dtype=np.float32)
    out = MeanCombiner().combine({
        "saturating": saturating, "a": quiet_a, "b": quiet_b
    })
    # (1.0 + 0.1 + 0.2) / 3 ~= 0.433 — well below 1.0 across the map.
    assert np.allclose(out, (1.0 + 0.1 + 0.2) / 3)
    assert out.max() < 0.5


def test_mean_combiner_preserves_value_range():
    rng = np.random.default_rng(1)
    sigs = {
        f"s{i}": rng.uniform(0, 1, size=(16, 16)).astype(np.float32)
        for i in range(3)
    }
    out = MeanCombiner().combine(sigs)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_mean_combiner_single_signal_passthrough():
    m = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = MeanCombiner().combine({"only": m})
    assert np.allclose(out, m)


def test_mean_combiner_name():
    assert MeanCombiner().name == "mean"

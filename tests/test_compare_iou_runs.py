"""Tests for the paired-bootstrap inner function used by
``scripts/compare_iou_runs.py``.

We test the mathematical properties (paired-bootstrap behavior on
constructed inputs) here. End-to-end Hydra wiring is exercised by
the script's smoke run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The script lives in scripts/, not the package — import directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_iou_runs import _paired_bootstrap  # noqa: E402


# ---------------------------------------------------------------------------
# Boundary / shape behavior
# ---------------------------------------------------------------------------

def test_empty_input_returns_nans():
    rng = np.random.default_rng(0)
    out = _paired_bootstrap(np.array([]), np.array([]), n_iter=100, rng=rng)
    assert out["n"] == 0
    for key in ("mean_a", "mean_b", "delta", "ci_low", "ci_high", "p_value_one_sided"):
        assert np.isnan(out[key])


def test_length_mismatch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="length mismatch"):
        _paired_bootstrap(np.zeros(5), np.zeros(7), n_iter=10, rng=rng)


def test_returns_correct_n_and_means():
    rng = np.random.default_rng(0)
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.2, 0.6, 0.8])
    out = _paired_bootstrap(a, b, n_iter=100, rng=rng)
    assert out["n"] == 3
    assert out["mean_a"] == pytest.approx(0.5)
    assert out["mean_b"] == pytest.approx(0.5333333, abs=1e-6)
    assert out["delta"] == pytest.approx(0.5333333 - 0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Mathematical properties of the bootstrap
# ---------------------------------------------------------------------------

def test_identical_runs_produce_zero_delta_and_centered_pvalue():
    """When a == b, every bootstrap resample produces delta = 0
    exactly. The p-value (fraction of deltas <= 0) is therefore 1.0
    by definition of the comparison — useful as a sanity check that
    the test is not asymmetric."""
    rng = np.random.default_rng(42)
    x = np.array([0.1, 0.5, 0.7, 0.9, 0.3, 0.8, 0.2])
    out = _paired_bootstrap(x, x, n_iter=2000, rng=rng)
    assert out["delta"] == 0.0
    assert out["ci_low"] == 0.0
    assert out["ci_high"] == 0.0
    # Every bootstrap iteration produced delta == 0, which is "<= 0",
    # so p_value_one_sided == 1.0. This is the correct, conservative
    # behavior — there's no evidence b > a when they're identical.
    assert out["p_value_one_sided"] == 1.0


def test_b_clearly_above_a_yields_small_pvalue_and_positive_ci():
    """When B is uniformly higher than A by a large margin relative
    to per-triplet variance, the bootstrap should detect this with a
    small p-value and a CI that excludes zero."""
    rng = np.random.default_rng(7)
    n = 200
    base_difficulty = np.linspace(0.1, 0.9, n)
    a = base_difficulty
    b = base_difficulty + 0.05  # uniform +0.05 — strong, consistent shift

    out = _paired_bootstrap(a, b, n_iter=5000, rng=rng)
    assert out["delta"] == pytest.approx(0.05, abs=1e-6)
    assert out["ci_low"] > 0.0
    assert out["ci_high"] > 0.0
    # With this large a uniform shift on 200 paired samples, basically
    # zero bootstrap iterations should have delta <= 0.
    assert out["p_value_one_sided"] < 0.001


def test_b_clearly_below_a_yields_large_pvalue():
    """Mirror of the above: when B is uniformly lower than A, the
    one-sided 'B > A' p-value should be near 1."""
    rng = np.random.default_rng(8)
    n = 200
    base = np.linspace(0.1, 0.9, n)
    a = base
    b = base - 0.05

    out = _paired_bootstrap(a, b, n_iter=5000, rng=rng)
    assert out["delta"] == pytest.approx(-0.05, abs=1e-6)
    # Both endpoints of the CI should be negative.
    assert out["ci_high"] < 0.0
    # All bootstrap deltas <= 0, so p_value_one_sided ≈ 1.
    assert out["p_value_one_sided"] > 0.99


def test_paired_test_separates_pairing_from_intrinsic_variance():
    """The motivating reason to use paired bootstrap: when per-triplet
    difficulty varies wildly but the run-to-run difference is small
    and consistent, paired bootstrap should still detect the shift.

    A two-sample bootstrap on the same data would have CI ~10x wider
    because intrinsic difficulty variance dominates. We don't test
    that comparison here (no two-sample function to test against),
    but we DO verify that the paired test correctly detects a tiny
    consistent shift on noisy data.
    """
    rng = np.random.default_rng(11)
    n = 500
    # Per-triplet IoU spans the full range [0, 1] — high intrinsic
    # variance. But the shift between runs is tiny (+0.005) and
    # consistent.
    intrinsic = rng.uniform(0.0, 1.0, size=n)
    a = intrinsic
    b = np.clip(intrinsic + 0.005, 0.0, 1.0)

    out = _paired_bootstrap(a, b, n_iter=5000, rng=rng)
    # Delta is tiny but the test should still reject "B == A" because
    # the paired structure removes the intrinsic-variance noise.
    # delta is approximately 0.005 (slightly less because clipping at 1
    # absorbs some of the shift on already-near-1 samples).
    assert 0.003 < out["delta"] < 0.006
    # Even though delta is small, the paired CI should exclude zero.
    assert out["ci_low"] > 0.0


def test_random_noise_with_zero_mean_difference_centers_pvalue():
    """When the per-triplet differences are pure mean-zero noise, the
    one-sided p-value should be near the middle of [0, 1] — neither
    tiny (which would indicate spurious "B > A" detection) nor near 1
    (spurious "B < A").

    For a single fixed seed with small finite n, the observed delta
    is drawn from N(0, sigma/sqrt(n)) and the p-value can land anywhere
    in [~0.05, ~0.95] without indicating a bug. We use [0.05, 0.95]
    as a generous bound that catches genuine asymmetry without
    being seed-fragile.
    """
    rng = np.random.default_rng(13)
    n = 300
    base = rng.uniform(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 0.05, size=n)
    a = np.clip(base, 0.0, 1.0)
    b = np.clip(base + noise, 0.0, 1.0)

    out = _paired_bootstrap(a, b, n_iter=5000, rng=rng)
    # delta is small (close to zero); p-value should be neither tiny
    # nor near 1.
    assert 0.05 < out["p_value_one_sided"] < 0.95


def test_reproducible_with_fixed_seed():
    """Same RNG state + inputs => byte-identical CI bounds."""
    a = np.array([0.1, 0.4, 0.6, 0.9])
    b = np.array([0.2, 0.5, 0.5, 0.95])
    out1 = _paired_bootstrap(a, b, n_iter=500, rng=np.random.default_rng(123))
    out2 = _paired_bootstrap(a, b, n_iter=500, rng=np.random.default_rng(123))
    assert out1 == out2

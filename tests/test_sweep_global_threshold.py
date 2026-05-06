"""Tests for the pure helper functions in
``scripts/sweep_global_threshold.py``.

End-to-end Hydra wiring is exercised by the script's smoke run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The script lives in scripts/, not the package — import directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from sweep_global_threshold import (  # noqa: E402
    _find_candidate_thresholds,
    _sweep_thresholds,
)


# ---------------------------------------------------------------------------
# _sweep_thresholds
# ---------------------------------------------------------------------------

def test_sweep_path1_rate_decreases_monotonically_with_threshold():
    """As the threshold rises, fewer values exceed it. Path-1 rate
    should be monotonically non-increasing in threshold."""
    rng = np.random.default_rng(0)
    cdm = rng.uniform(0, 1, size=1000)
    thresholds = [round(0.1 * i, 2) for i in range(11)]  # 0.0..1.0
    sweep = _sweep_thresholds(cdm, thresholds)
    rates = [row["path1_global_rate"] for row in sweep]
    for a, b in zip(rates, rates[1:]):
        assert a >= b, f"non-monotone at threshold {b}: {a} -> {b}"


def test_sweep_at_threshold_zero_classifies_everything_as_global():
    """combined_diff_mean >= 0 is true for all non-negative values."""
    cdm = np.array([0.1, 0.5, 0.9, 0.0, 0.7])
    sweep = _sweep_thresholds(cdm, [0.0])
    assert sweep[0]["path1_global_rate"] == 1.0
    assert sweep[0]["n_path1_global"] == 5


def test_sweep_at_threshold_above_max_classifies_nothing():
    cdm = np.array([0.1, 0.5, 0.9])
    sweep = _sweep_thresholds(cdm, [1.0])
    assert sweep[0]["path1_global_rate"] == 0.0
    assert sweep[0]["n_path1_global"] == 0


def test_sweep_handles_empty_input():
    sweep = _sweep_thresholds(np.array([]), [0.0, 0.5])
    assert all(r["path1_global_rate"] == 0.0 for r in sweep)
    assert all(r["n_path1_global"] == 0 for r in sweep)


def test_sweep_threshold_inclusive_at_lower_bound():
    """`combined.mean() >= threshold` is the production condition;
    a value exactly at the threshold should count as global."""
    cdm = np.array([0.4, 0.4, 0.4])
    sweep = _sweep_thresholds(cdm, [0.4])
    assert sweep[0]["path1_global_rate"] == 1.0
    assert sweep[0]["n_path1_global"] == 3


def test_sweep_returns_one_row_per_threshold():
    cdm = np.array([0.5])
    sweep = _sweep_thresholds(cdm, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert len(sweep) == 5
    # And in the order requested.
    assert [r["threshold"] for r in sweep] == [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# _find_candidate_thresholds
# ---------------------------------------------------------------------------

def test_find_candidates_picks_smallest_threshold_meeting_target():
    """The function picks the smallest swept threshold whose Path-1
    rate is <= each target rate."""
    sweep = [
        {"threshold": 0.0, "path1_global_rate": 1.0, "n_path1_global": 100},
        {"threshold": 0.2, "path1_global_rate": 0.8, "n_path1_global": 80},
        {"threshold": 0.4, "path1_global_rate": 0.6, "n_path1_global": 60},
        {"threshold": 0.6, "path1_global_rate": 0.3, "n_path1_global": 30},
        {"threshold": 0.8, "path1_global_rate": 0.1, "n_path1_global": 10},
    ]
    candidates = _find_candidate_thresholds(sweep, [0.5, 0.3, 0.1])
    # For target 0.50 -> smallest threshold with rate <= 0.5 is 0.6 (rate 0.3).
    assert candidates["0.50"] == 0.6
    # For target 0.30 -> smallest threshold with rate <= 0.3 is 0.6 (rate 0.3).
    assert candidates["0.30"] == 0.6
    # For target 0.10 -> smallest threshold with rate <= 0.1 is 0.8 (rate 0.1).
    assert candidates["0.10"] == 0.8


def test_find_candidates_returns_none_when_no_threshold_meets_target():
    """If even the highest swept threshold can't get below the target
    rate, return None — signal that the user should sweep more
    aggressive thresholds or accept a higher rate."""
    sweep = [
        {"threshold": 0.0, "path1_global_rate": 1.0, "n_path1_global": 100},
        {"threshold": 0.5, "path1_global_rate": 0.8, "n_path1_global": 80},
        {"threshold": 1.0, "path1_global_rate": 0.6, "n_path1_global": 60},
    ]
    candidates = _find_candidate_thresholds(sweep, [0.5, 0.1])
    assert candidates["0.50"] is None
    assert candidates["0.10"] is None


def test_find_candidates_picks_threshold_zero_when_target_already_met():
    """If the lowest threshold (likely 0.0) already meets the target,
    it should be picked. This represents a dataset that is naturally
    below the target — no thresholding intervention needed."""
    sweep = [
        {"threshold": 0.0, "path1_global_rate": 0.05, "n_path1_global": 5},
        {"threshold": 0.5, "path1_global_rate": 0.02, "n_path1_global": 2},
    ]
    candidates = _find_candidate_thresholds(sweep, [0.10])
    # The first row already satisfies; returns 0.0.
    assert candidates["0.10"] == 0.0


def test_find_candidates_handles_empty_sweep():
    """Defensive: if no thresholds were swept, every target is None."""
    candidates = _find_candidate_thresholds([], [0.25, 0.50])
    assert candidates["0.25"] is None
    assert candidates["0.50"] is None

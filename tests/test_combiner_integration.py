"""Integration tests: `MaskGenerator` + pluggable combiners.

These tests exercise the end-to-end wiring rather than combiner math
(which is covered in test_combiners.py). We check:

* ``MaskGenerator`` uses the injected combiner (not the default) when
  one is provided.
* ``MaskGenerator`` falls back to ``MaxCombiner`` when no combiner is
  passed — backward compatibility with pre-existing tests.
* A scenario where the choice of combiner visibly changes the mask
  (one noisy signal that max-pool lets through but mean-pool dilutes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edit2forensics.data.triplet import EditTriplet
from edit2forensics.mask.combiners import MaxCombiner, MeanCombiner, SignalCombiner
from edit2forensics.mask.generator import MaskGenerator, MaskGeneratorConfig
from edit2forensics.mask.signals import DiffSignal


# ---------------------------------------------------------------------------
# Test fixtures: deterministic signals and combiners
# ---------------------------------------------------------------------------

class ConstantSignal(DiffSignal):
    """Returns a fixed (H, W) map regardless of input."""

    def __init__(self, name: str, value_map: np.ndarray) -> None:
        self.name = name
        self._map = value_map.astype(np.float32)

    def compute(self, real, edited):
        h, w = real.shape[:2]
        assert self._map.shape == (h, w)
        return self._map


class RecordingCombiner(SignalCombiner):
    """Records the per_signal dict it receives for inspection."""

    name = "recording"

    def __init__(self):
        self.seen: list[dict[str, np.ndarray]] = []

    def combine(self, per_signal):
        self.seen.append({k: v.copy() for k, v in per_signal.items()})
        # Return arbitrary-but-valid combined map (zero-filled).
        any_map = next(iter(per_signal.values()))
        return np.zeros_like(any_map)


def _write_triplet(tmp_path: Path, real: np.ndarray, edited: np.ndarray) -> EditTriplet:
    real_path = tmp_path / "real.png"
    edited_path = tmp_path / "edited.png"
    Image.fromarray(real).save(real_path)
    Image.fromarray(edited).save(edited_path)
    return EditTriplet(
        triplet_id="cmb_0001",
        source_dataset="test",
        real_path=real_path,
        edited_path=edited_path,
        instruction="fixture",
    )


def _dummy_image_pair(size: tuple[int, int]):
    h, w = size
    return (
        np.zeros((h, w, 3), dtype=np.uint8),
        np.zeros((h, w, 3), dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_default_combiner_is_max_when_not_specified(tmp_path):
    """Pre-existing tests construct MaskGenerator(signals=...) without
    passing combiner. That must keep working and use MaxCombiner."""
    real, edited = _dummy_image_pair((64, 64))
    triplet = _write_triplet(tmp_path, real, edited)

    # Set up two signals with clearly distinct maps. Max-pool result
    # should be the element-wise maximum.
    low = np.zeros((64, 64), dtype=np.float32)
    high = np.full((64, 64), 0.9, dtype=np.float32)
    gen = MaskGenerator(
        signals=[ConstantSignal("low", low), ConstantSignal("high", high)]
    )

    # Capture the combined map by injecting a RecordingCombiner via
    # attribute replacement — confirms the default was MaxCombiner.
    assert isinstance(gen.combiner, MaxCombiner)


# ---------------------------------------------------------------------------
# The combiner is actually called
# ---------------------------------------------------------------------------

def test_generator_delegates_combination_to_injected_combiner(tmp_path):
    real, edited = _dummy_image_pair((64, 64))
    triplet = _write_triplet(tmp_path, real, edited)

    rec = RecordingCombiner()
    s1 = ConstantSignal("s1", np.full((64, 64), 0.2, dtype=np.float32))
    s2 = ConstantSignal("s2", np.full((64, 64), 0.5, dtype=np.float32))

    gen = MaskGenerator(signals=[s1, s2], combiner=rec)
    gen.generate(triplet, tmp_path / "m.png")

    assert len(rec.seen) == 1
    seen = rec.seen[0]
    # The combiner received both signals by name, with the expected maps.
    assert set(seen.keys()) == {"s1", "s2"}
    assert np.allclose(seen["s1"], 0.2)
    assert np.allclose(seen["s2"], 0.5)


# ---------------------------------------------------------------------------
# Max vs. Mean produces observably different masks on a constructed case
# ---------------------------------------------------------------------------

def test_max_and_mean_give_different_masks_on_noisy_signal_scenario(tmp_path):
    """Scenario: signal A is clean (high only on the true edit region);
    signal B is noisy (fires at low-but-nonzero levels everywhere,
    including outside the edit).

    Max-pool lets B's noise saturate far outside the true region after
    percentile normalization. Mean-pool averages B's noise with A's
    clean zero background, keeping the off-region signal low. We expect
    the masks to differ in favor of mean-pool here.
    """
    real, edited = _dummy_image_pair((64, 64))
    triplet = _write_triplet(tmp_path, real, edited)

    # Signal A: sharp peak in top-left 16x16 quadrant, zero elsewhere.
    a_map = np.zeros((64, 64), dtype=np.float32)
    a_map[:16, :16] = 0.95
    # Signal B: uniform low-level noise everywhere (including off-region).
    b_map = np.full((64, 64), 0.5, dtype=np.float32)
    # (Signals are max-pooled or mean-pooled in the combiner; then the
    # combined map is percentile-normalized implicitly by Otsu's data
    # dependence. What matters is the relative ordering of regions.)

    signals = [ConstantSignal("clean", a_map), ConstantSignal("noisy", b_map)]

    gen_max = MaskGenerator(signals=signals, combiner=MaxCombiner())
    gen_mean = MaskGenerator(signals=signals, combiner=MeanCombiner())

    art_max = gen_max.generate(triplet, tmp_path / "max.png")
    art_mean = gen_mean.generate(triplet, tmp_path / "mean.png")

    mask_max = np.asarray(Image.open(art_max.mask_path)) > 127
    mask_mean = np.asarray(Image.open(art_mean.mask_path)) > 127

    # With max-pool, the noisy signal elevates the whole image's combined
    # values — after Otsu (or the global-detection branch) the mask
    # likely covers a much larger area than the true 16x16 region.
    # With mean-pool, the off-region average (0 + 0.5)/2 = 0.25 is
    # lower than the on-region average (0.95 + 0.5)/2 = 0.725, so Otsu
    # can separate them cleanly.
    # We don't assert exact areas (they depend on scope routing), only
    # that the two masks actually differ — the choice of combiner
    # affects the mask.
    assert not np.array_equal(mask_max, mask_mean), (
        "max-pool and mean-pool produced identical masks on a scenario "
        "constructed to distinguish them; did combiner wiring regress?"
    )


# ---------------------------------------------------------------------------
# Reporting: diff_strongest_signal is independent of the combiner
# ---------------------------------------------------------------------------

def test_strongest_signal_is_independent_of_combiner(tmp_path):
    """The `diff_strongest_signal` field on MaskArtifact should reflect
    each signal's raw contribution, not the post-combination map. This
    matters because Stage E (reasoning annotator) uses it to pick which
    artifact family to reference — that should stay consistent across
    combiner choices.
    """
    real, edited = _dummy_image_pair((64, 64))
    triplet = _write_triplet(tmp_path, real, edited)

    low = ConstantSignal("low", np.full((64, 64), 0.1, dtype=np.float32))
    high = ConstantSignal("high", np.full((64, 64), 0.8, dtype=np.float32))

    art_max = MaskGenerator(
        signals=[low, high], combiner=MaxCombiner()
    ).generate(triplet, tmp_path / "a.png")
    art_mean = MaskGenerator(
        signals=[low, high], combiner=MeanCombiner()
    ).generate(triplet, tmp_path / "b.png")

    assert art_max.diff_strongest_signal == "high"
    assert art_mean.diff_strongest_signal == "high"

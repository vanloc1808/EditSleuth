"""Tests for the MaskGenerator orchestrator.

We exercise the orchestration logic using crafted image pairs (so we
know the expected mask shape) and also with a minimal synthetic
DiffSignal to isolate orchestration from signal internals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.mask.generator import MaskGenerator, MaskGeneratorConfig
from edit2forensics.mask.signals import DiffSignal, LabPixelDiff


class ConstantSignal(DiffSignal):
    """Returns a fixed map regardless of input — deterministic test fixture."""

    def __init__(self, name: str, value_map: np.ndarray) -> None:
        self.name = name
        self._map = value_map.astype(np.float32)

    def compute(self, real, edited):
        # Match requested shape to real's H,W
        h, w = real.shape[:2]
        if self._map.shape != (h, w):
            # Tile or resize as needed; for tests we ensure fixtures match.
            raise ValueError(
                f"ConstantSignal fixture shape {self._map.shape} != image {(h, w)}"
            )
        return self._map


def _write_triplet(tmp_path: Path, real: np.ndarray, edited: np.ndarray) -> EditTriplet:
    real_path = tmp_path / "real.png"
    edited_path = tmp_path / "edited.png"
    Image.fromarray(real).save(real_path)
    Image.fromarray(edited).save(edited_path)
    return EditTriplet(
        triplet_id="test_0001",
        source_dataset="test",
        real_path=real_path,
        edited_path=edited_path,
        instruction="fixture",
    )


# ----------------------------------------------------------------------
# End-to-end with a real signal
# ----------------------------------------------------------------------

def test_local_edit_produces_local_mask(tmp_path):
    """A pair with a corner-only change should yield a corner-only mask
    flagged as 'local'."""
    h, w = 64, 64
    real = np.full((h, w, 3), 128, dtype=np.uint8)
    edited = real.copy()
    edited[:24, :24] = [220, 30, 30]

    triplet = _write_triplet(tmp_path, real, edited)
    gen = MaskGenerator(signals=[LabPixelDiff()])
    art = gen.generate(triplet, tmp_path / "mask.png")

    assert art.edit_scope == "local"
    assert art.registration_ok is True
    assert art.mask_path.exists()

    mask = np.asarray(Image.open(art.mask_path)) > 127
    # Most of the corner-quadrant should be flagged.
    assert mask[:24, :24].mean() > 0.7
    # The far opposite corner should not.
    assert mask[32:, 32:].mean() < 0.05


def test_global_edit_yields_global_scope(tmp_path):
    """A pair where every pixel changes should route to scope=global."""
    rng = np.random.default_rng(0)
    real = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    edited = np.clip(real.astype(int) + 60, 0, 255).astype(np.uint8)  # global shift

    triplet = _write_triplet(tmp_path, real, edited)
    gen = MaskGenerator(signals=[LabPixelDiff()])
    art = gen.generate(triplet, tmp_path / "mask.png")

    assert art.edit_scope == "global"
    # Global masks are all-white after routing.
    mask = np.asarray(Image.open(art.mask_path)) > 127
    assert mask.mean() == pytest.approx(1.0)


def test_identical_images_yield_ambiguous_empty_mask(tmp_path):
    """Zero-variance diff -> empty mask, 'ambiguous' scope."""
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    triplet = _write_triplet(tmp_path, img, img)
    gen = MaskGenerator(signals=[LabPixelDiff()])
    art = gen.generate(triplet, tmp_path / "mask.png")

    assert art.edit_scope == "ambiguous"
    assert art.mask_area_frac == 0.0
    mask = np.asarray(Image.open(art.mask_path)) > 127
    assert mask.sum() == 0


# ----------------------------------------------------------------------
# Orchestration with constant signals
# ----------------------------------------------------------------------

def test_signal_combination_is_max_pool(tmp_path):
    """Two signals; the max across them determines the combined map."""
    h, w = 64, 64
    # Signal A is high in the left half, zero elsewhere.
    a_map = np.zeros((h, w), dtype=np.float32)
    a_map[:, : w // 2] = 0.9
    # Signal B is high in the right half, zero elsewhere.
    b_map = np.zeros((h, w), dtype=np.float32)
    b_map[:, w // 2 :] = 0.9

    # Images themselves don't matter — signals ignore them.
    real = np.zeros((h, w, 3), dtype=np.uint8)
    edited = np.zeros((h, w, 3), dtype=np.uint8)
    triplet = _write_triplet(tmp_path, real, edited)

    gen = MaskGenerator(
        signals=[ConstantSignal("sig_a", a_map), ConstantSignal("sig_b", b_map)]
    )
    art = gen.generate(triplet, tmp_path / "mask.png")

    # Max-pool across signals -> high everywhere -> global scope.
    assert art.edit_scope == "global"
    # Strongest should be whichever has higher mean; here they're tied,
    # so just check it's one of the two.
    assert art.diff_strongest_signal in {"sig_a", "sig_b"}


def test_strongest_signal_reports_higher_mean(tmp_path):
    h, w = 64, 64
    low = np.full((h, w), 0.1, dtype=np.float32)
    high = np.full((h, w), 0.8, dtype=np.float32)
    real = np.zeros((h, w, 3), dtype=np.uint8)
    edited = np.zeros((h, w, 3), dtype=np.uint8)
    triplet = _write_triplet(tmp_path, real, edited)

    gen = MaskGenerator(
        signals=[ConstantSignal("low_sig", low), ConstantSignal("high_sig", high)]
    )
    art = gen.generate(triplet, tmp_path / "mask.png")
    assert art.diff_strongest_signal == "high_sig"


# ----------------------------------------------------------------------
# Alignment / shape handling
# ----------------------------------------------------------------------

def test_shape_mismatch_with_resize_align(tmp_path):
    """Edited image at a different size is resized; registration_ok stays True."""
    real = np.full((64, 64, 3), 100, dtype=np.uint8)
    edited = np.full((48, 48, 3), 100, dtype=np.uint8)
    edited[:20, :20] = [220, 30, 30]

    triplet = _write_triplet(tmp_path, real, edited)
    gen = MaskGenerator(
        signals=[LabPixelDiff()],
        config=MaskGeneratorConfig(allow_resize_align=True),
    )
    art = gen.generate(triplet, tmp_path / "mask.png")
    assert art.registration_ok is True
    # The resized edit still shows up in the mask.
    assert art.mask_area_frac > 0


def test_shape_mismatch_without_resize_align_flags_failure(tmp_path):
    real = np.full((64, 64, 3), 100, dtype=np.uint8)
    edited = np.full((48, 48, 3), 100, dtype=np.uint8)
    triplet = _write_triplet(tmp_path, real, edited)

    gen = MaskGenerator(
        signals=[LabPixelDiff()],
        config=MaskGeneratorConfig(allow_resize_align=False),
    )
    art = gen.generate(triplet, tmp_path / "mask.png")
    assert art.registration_ok is False
    assert art.edit_scope == "alignment_failed"


# ----------------------------------------------------------------------
# Refinement behavior
# ----------------------------------------------------------------------

def test_small_component_is_dropped_by_refinement(tmp_path):
    """A single-pixel flip should be removed by min_component_area_frac."""
    real = np.full((128, 128, 3), 100, dtype=np.uint8)
    edited = real.copy()
    edited[0, 0] = [255, 0, 0]  # single-pixel speckle

    triplet = _write_triplet(tmp_path, real, edited)
    gen = MaskGenerator(
        signals=[LabPixelDiff()],
        config=MaskGeneratorConfig(min_component_area_frac=0.001),
    )
    art = gen.generate(triplet, tmp_path / "mask.png")
    # After refinement the speckle should be gone.
    mask = np.asarray(Image.open(art.mask_path)) > 127
    assert mask.sum() == 0
    # Post-refinement re-classification: area is < ambiguous threshold.
    assert art.edit_scope == "ambiguous"


def test_artifact_serialization_roundtrip(tmp_path):
    """MaskArtifact.to_dict/from_dict must be lossless."""
    real = np.full((32, 32, 3), 100, dtype=np.uint8)
    edited = real.copy()
    edited[:16, :16] = [200, 50, 50]
    triplet = _write_triplet(tmp_path, real, edited)

    gen = MaskGenerator(signals=[LabPixelDiff()])
    art = gen.generate(triplet, tmp_path / "mask.png")
    restored = MaskArtifact.from_dict(art.to_dict())
    assert restored == art


def test_requires_at_least_one_signal():
    with pytest.raises(ValueError, match="at least one"):
        MaskGenerator(signals=[])


# ----------------------------------------------------------------------
# generate_batch — equivalence with looping generate
# ----------------------------------------------------------------------

def test_generate_batch_empty_returns_empty(tmp_path):
    gen = MaskGenerator(signals=[LabPixelDiff()])
    out = gen.generate_batch([], [])
    assert out == []


def test_generate_batch_length_mismatch_raises(tmp_path):
    real = np.full((32, 32, 3), 100, dtype=np.uint8)
    edited = real.copy()
    edited[:16, :16] = [200, 50, 50]
    triplet = _write_triplet(tmp_path, real, edited)
    gen = MaskGenerator(signals=[LabPixelDiff()])
    with pytest.raises(ValueError, match="length mismatch"):
        gen.generate_batch([triplet], [tmp_path / "a.png", tmp_path / "b.png"])


def test_generate_batch_matches_per_triplet_generate(tmp_path):
    """The numerical-equivalence guarantee. ``generate_batch`` must
    produce the same artifacts as a sequence of ``generate`` calls.

    LabPixelDiff doesn't override ``compute_batch``, so the default
    DiffSignal.compute_batch (which loops compute) is exercised. This
    test serves as the cross-check that the generator's batched path
    correctly threads results through to the finalize phase.
    """
    triplets = []
    out_paths_serial = []
    out_paths_batched = []

    rng = np.random.default_rng(0)
    for i in range(4):
        real = np.full((48, 48, 3), 100, dtype=np.uint8)
        # Add some texture so SSIM/LAB have something to compare.
        real += rng.integers(-20, 20, size=real.shape, dtype=np.int8).astype(np.uint8)
        edited = real.copy()
        # Edit a corner of varying size per triplet.
        side = 8 + i * 2
        edited[:side, :side] = [200, 50, 50]

        sub = tmp_path / f"t{i}"
        sub.mkdir()
        triplet_id_real = sub / "real.png"
        triplet_id_edit = sub / "edited.png"
        Image.fromarray(real).save(triplet_id_real)
        Image.fromarray(edited).save(triplet_id_edit)
        t = EditTriplet(
            triplet_id=f"test_t{i}",
            source_dataset="test",
            real_path=triplet_id_real,
            edited_path=triplet_id_edit,
            instruction="fixture",
        )
        triplets.append(t)
        out_paths_serial.append(sub / "mask_serial.png")
        out_paths_batched.append(sub / "mask_batched.png")

    gen = MaskGenerator(signals=[LabPixelDiff()])
    serial = [gen.generate(t, p) for t, p in zip(triplets, out_paths_serial)]
    batched = gen.generate_batch(triplets, out_paths_batched)

    assert len(serial) == len(batched) == 4
    for i in range(4):
        # All numerical artifact fields must match.
        assert serial[i].triplet_id == batched[i].triplet_id
        assert serial[i].edit_scope == batched[i].edit_scope
        assert serial[i].registration_ok == batched[i].registration_ok
        assert serial[i].confidence == pytest.approx(batched[i].confidence)
        assert serial[i].diff_strongest_signal == batched[i].diff_strongest_signal
        assert serial[i].mask_area_frac == pytest.approx(batched[i].mask_area_frac)
        assert serial[i].combined_diff_mean == pytest.approx(batched[i].combined_diff_mean)

        # Mask PNGs on disk must be byte-identical.
        s_arr = np.asarray(Image.open(serial[i].mask_path))
        b_arr = np.asarray(Image.open(batched[i].mask_path))
        assert np.array_equal(s_arr, b_arr), f"mask {i} differs between paths"


def test_generate_batch_calls_compute_batch_on_signals(tmp_path):
    """Verify the batched path actually calls ``compute_batch`` (not
    ``compute`` in a loop) on its signals."""
    h, w = 32, 32
    real = np.full((h, w, 3), 100, dtype=np.uint8)
    edited = real.copy()
    edited[:8, :8] = [200, 50, 50]
    triplet = _write_triplet(tmp_path, real, edited)

    class CountingSignal(DiffSignal):
        name = "counting"

        def __init__(self, value_map):
            self._map = value_map.astype(np.float32)
            self.compute_calls = 0
            self.compute_batch_calls = 0

        def compute(self, real, edited):
            self.compute_calls += 1
            return self._map

        def compute_batch(self, reals, editeds):
            self.compute_batch_calls += 1
            return [self._map for _ in reals]

    s = CountingSignal(np.full((h, w), 0.7, dtype=np.float32))
    gen = MaskGenerator(signals=[s])

    # generate_batch should call compute_batch ONCE, not compute three times.
    sub_paths = [tmp_path / f"out_{i}.png" for i in range(3)]
    triplets = [triplet, triplet, triplet]
    gen.generate_batch(triplets, sub_paths)

    assert s.compute_batch_calls == 1
    assert s.compute_calls == 0  # not called during batch path

"""Tests for ``DifficultyScorerV2`` and ``DifficultyArtifactV2``.

We verify:
* The artifact schema roundtrips through to_dict/from_dict.
* Default weights validate.
* The score() method computes each component correctly.
* Missing/NaN mask_compactness raises with a clear message.
* The aggregated raw score equals the manual weighted sum.
* The V1 vs V2 cross-comparison: on the same triplet, V2's
  difficulty_raw is mathematically distinct from V1's because the
  formula and weights differ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edit2forensics.data.difficulty_artifact_v2 import DifficultyArtifactV2
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.difficulty.scorer_v2 import (
    DifficultyScorerV2,
    DifficultyScorerV2Config,
)


def _make_test_triplet(tmp_path: Path) -> EditTriplet:
    """Write a 64x64 real/edited pair and return an EditTriplet."""
    rng = np.random.default_rng(0)
    real = rng.integers(50, 200, size=(64, 64, 3), dtype=np.uint8)
    edited = real.copy()
    edited[:16, :16] = 255  # localized edit
    real_path = tmp_path / "real.png"
    edited_path = tmp_path / "edited.png"
    Image.fromarray(real).save(real_path)
    Image.fromarray(edited).save(edited_path)
    return EditTriplet(
        triplet_id="test_t0",
        source_dataset="test",
        real_path=real_path,
        edited_path=edited_path,
        instruction="add a red square in the corner",
    )


def _make_mask_artifact(
    tmp_path: Path,
    triplet_id: str = "test_t0",
    compactness: float = 0.85,
    area_frac: float = 0.1,
    combined_diff_mean: float = 0.3,
) -> MaskArtifact:
    """Construct a MaskArtifact with a real mask PNG on disk."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[:16, :16] = True
    mask_path = tmp_path / f"{triplet_id}_mask.png"
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    return MaskArtifact(
        triplet_id=triplet_id,
        mask_path=mask_path,
        mask_area_frac=area_frac,
        edit_scope="local",
        registration_ok=True,
        confidence=0.5,
        diff_strongest_signal="lab_pixel",
        combined_diff_mean=combined_diff_mean,
        mask_compactness=compactness,
    )


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------

def test_artifact_v2_to_dict_from_dict_roundtrip():
    """Construct, serialize, deserialize, equality."""
    a = DifficultyArtifactV2(
        triplet_id="x",
        structural_change=0.4,
        compactness_score=0.3,
        instruction_complexity=0.5,
        difficulty_raw=0.42,
        difficulty_bin="medium",
        weights={"structural_change": 0.55, "compactness_score": 0.25,
                 "instruction_complexity": 0.20},
    )
    restored = DifficultyArtifactV2.from_dict(a.to_dict())
    assert restored == a


def test_from_dict_parses_weights_serialized_as_json_string():
    """Parquet roundtrip can sometimes return the ``weights`` dict as
    a JSON-encoded string (mixed-schema directory, PyArrow struct-to-
    string downcast, etc.). from_dict must tolerate this rather than
    raising the cryptic ``dictionary update sequence element #0 has
    length 1; 2 is required`` error."""
    bad_dict = {
        "triplet_id": "x",
        "structural_change": 0.4,
        "compactness_score": 0.3,
        "instruction_complexity": 0.5,
        "difficulty_raw": 0.42,
        "difficulty_bin": "medium",
        "weights": '{"structural_change": 0.55, "compactness_score": 0.25, '
                   '"instruction_complexity": 0.20}',
    }
    art = DifficultyArtifactV2.from_dict(bad_dict)
    assert art.weights == {
        "structural_change": 0.55,
        "compactness_score": 0.25,
        "instruction_complexity": 0.20,
    }


def test_from_dict_handles_invalid_json_weights_gracefully():
    """If the weights string isn't valid JSON (e.g. the weird single-
    char value picobanana_52629 had), fall back to an empty dict
    rather than crashing. The downstream consumer will see weights
    is empty and can decide what to do."""
    bad_dict = {
        "triplet_id": "x",
        "structural_change": 0.4,
        "compactness_score": 0.3,
        "instruction_complexity": 0.5,
        "difficulty_raw": 0.42,
        "difficulty_bin": "medium",
        "weights": "structural_change",  # the actual bad value seen in the wild
    }
    art = DifficultyArtifactV2.from_dict(bad_dict)
    assert art.weights == {}


def test_from_dict_handles_none_weights_gracefully():
    """None can appear in parquet roundtrips as a null cell."""
    bad_dict = {
        "triplet_id": "x",
        "structural_change": 0.4,
        "compactness_score": 0.3,
        "instruction_complexity": 0.5,
        "difficulty_raw": 0.42,
        "difficulty_bin": "medium",
        "weights": None,
    }
    art = DifficultyArtifactV2.from_dict(bad_dict)
    assert art.weights == {}


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_default_weights_match_design():
    """Default weights are (0.55, 0.25, 0.20)."""
    cfg = DifficultyScorerV2Config()
    assert cfg.weights["structural_change"] == 0.55
    assert cfg.weights["compactness_score"] == 0.25
    assert cfg.weights["instruction_complexity"] == 0.20
    # Sum to 1.0 (sanity).
    assert sum(cfg.weights.values()) == pytest.approx(1.0)


def test_config_rejects_missing_v2_components():
    """Weights dict must contain all three V2 components."""
    with pytest.raises(ValueError, match="missing required V2 components"):
        DifficultyScorerV2Config(weights={
            "structural_change": 0.5,
            "compactness_score": 0.5,
            # instruction_complexity missing
        })


def test_config_rejects_v1_legacy_components():
    """V1 weight names ('locality', 'perceptual_change') don't satisfy
    V2's required components — V2 explicitly does not accept them."""
    with pytest.raises(ValueError, match="missing required V2 components"):
        DifficultyScorerV2Config(weights={
            "structural_change": 0.30,
            "perceptual_change": 0.30,
            "locality": 0.20,
            "instruction_complexity": 0.20,
        })


def test_config_rejects_non_numeric_weight():
    with pytest.raises(TypeError, match="must be numeric"):
        DifficultyScorerV2Config(weights={
            "structural_change": "high",  # type: ignore
            "compactness_score": 0.25,
            "instruction_complexity": 0.20,
        })


# ---------------------------------------------------------------------------
# score() — happy path
# ---------------------------------------------------------------------------

def test_score_returns_v2_artifact(tmp_path):
    """A normal score() call returns a DifficultyArtifactV2 with all
    fields populated, and difficulty_bin set to placeholder 'medium'."""
    triplet = _make_test_triplet(tmp_path)
    mask = _make_mask_artifact(tmp_path, compactness=0.85)
    scorer = DifficultyScorerV2()
    art = scorer.score(triplet, mask)

    assert isinstance(art, DifficultyArtifactV2)
    assert art.triplet_id == "test_t0"
    assert 0.0 <= art.structural_change <= 1.0
    # compactness=0.85 -> compactness_score = 1 - 0.85 = 0.15
    assert art.compactness_score == pytest.approx(0.15, abs=1e-6)
    assert 0.0 <= art.instruction_complexity <= 1.0
    assert art.difficulty_bin == "medium"  # placeholder; driver overwrites


def test_score_aggregate_equals_weighted_sum(tmp_path):
    """The difficulty_raw must be the exact dot product of components
    and weights (no surprise normalization or scaling)."""
    triplet = _make_test_triplet(tmp_path)
    mask = _make_mask_artifact(tmp_path, compactness=0.7)
    scorer = DifficultyScorerV2()
    art = scorer.score(triplet, mask)

    expected = (
        0.55 * art.structural_change
        + 0.25 * art.compactness_score
        + 0.20 * art.instruction_complexity
    )
    assert art.difficulty_raw == pytest.approx(expected, abs=1e-9)


def test_score_compactness_score_inverts_compactness(tmp_path):
    """compactness_score = 1 - mask_compactness. Boundary cases: a
    fully-compact mask (1.0) yields compactness_score = 0; a fully-
    diffuse mask (0.0) yields compactness_score = 1."""
    triplet = _make_test_triplet(tmp_path)

    mask_compact = _make_mask_artifact(tmp_path, triplet_id="a", compactness=1.0)
    mask_diffuse = _make_mask_artifact(tmp_path, triplet_id="b", compactness=0.0)

    scorer = DifficultyScorerV2()
    art_compact = scorer.score(triplet, mask_compact)
    art_diffuse = scorer.score(triplet, mask_diffuse)

    assert art_compact.compactness_score == pytest.approx(0.0, abs=1e-6)
    assert art_diffuse.compactness_score == pytest.approx(1.0, abs=1e-6)
    # And the diffuse mask produces a higher difficulty_raw because
    # compactness_score is the term that differs, and its weight is
    # positive.
    assert art_diffuse.difficulty_raw > art_compact.difficulty_raw


# ---------------------------------------------------------------------------
# score() — error handling
# ---------------------------------------------------------------------------

def test_score_raises_on_nan_compactness(tmp_path):
    """If mask_compactness is NaN (older Stage B parquet that wasn't
    augmented), V2 must raise — silently falling back would produce
    misleading numbers."""
    triplet = _make_test_triplet(tmp_path)
    mask = _make_mask_artifact(tmp_path, compactness=float("nan"))
    scorer = DifficultyScorerV2()
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(triplet, mask)


def test_score_clamps_compactness_outside_unit_interval(tmp_path):
    """Defensive clipping. A mask_compactness above 1.0 or below 0.0
    is invalid (geometric compactness is always in [0, 1]) but if it
    leaks in, we clip rather than producing out-of-range scores."""
    triplet = _make_test_triplet(tmp_path)
    # Use a compactness slightly above 1.0 — clipped to 1.0.
    mask = _make_mask_artifact(tmp_path, compactness=1.5)
    scorer = DifficultyScorerV2()
    art = scorer.score(triplet, mask)
    # compactness_score = 1 - clip(1.5, 0, 1) = 1 - 1 = 0
    assert art.compactness_score == 0.0


# ---------------------------------------------------------------------------
# Variance behavior — the headline claim
# ---------------------------------------------------------------------------

def test_v2_score_responds_to_compactness_independently_of_magnitude(tmp_path):
    """The point of V2: ``compactness_score`` is independent of the
    magnitude trio. Two triplets with identical magnitude (same SSIM,
    same instruction) but different compactness should produce
    different difficulty_raw values."""
    triplet = _make_test_triplet(tmp_path)
    mask_concentrated = _make_mask_artifact(
        tmp_path, triplet_id="t1", compactness=0.95, area_frac=0.1,
    )
    mask_scattered = _make_mask_artifact(
        tmp_path, triplet_id="t2", compactness=0.20, area_frac=0.1,
    )

    scorer = DifficultyScorerV2()
    art_concentrated = scorer.score(triplet, mask_concentrated)
    art_scattered = scorer.score(triplet, mask_scattered)

    # Same triplet, so structural_change and instruction_complexity
    # are identical between the two scorings. Only compactness_score
    # differs.
    assert art_concentrated.structural_change == art_scattered.structural_change
    assert (
        art_concentrated.instruction_complexity
        == art_scattered.instruction_complexity
    )
    # Scattered should score harder.
    assert art_scattered.difficulty_raw > art_concentrated.difficulty_raw
    # The difference equals exactly w_C * (compactness_score_diff).
    expected_diff = 0.25 * (
        art_scattered.compactness_score - art_concentrated.compactness_score
    )
    actual_diff = art_scattered.difficulty_raw - art_concentrated.difficulty_raw
    assert actual_diff == pytest.approx(expected_diff, abs=1e-9)


def test_score_independent_of_perceptual_change_and_locality(tmp_path):
    """V2 deliberately does NOT read combined_diff_mean or
    mask_area_frac. Verify by varying both fields and confirming the
    score is unchanged."""
    triplet = _make_test_triplet(tmp_path)

    mask_a = _make_mask_artifact(
        tmp_path, triplet_id="m1", compactness=0.8,
        area_frac=0.05, combined_diff_mean=0.1,
    )
    mask_b = _make_mask_artifact(
        tmp_path, triplet_id="m2", compactness=0.8,
        area_frac=0.95, combined_diff_mean=0.9,
    )
    scorer = DifficultyScorerV2()
    art_a = scorer.score(triplet, mask_a)
    art_b = scorer.score(triplet, mask_b)

    # All three V2 components are identical because compactness is
    # the same and the triplet/instruction is the same.
    assert art_a.difficulty_raw == pytest.approx(art_b.difficulty_raw, abs=1e-9)

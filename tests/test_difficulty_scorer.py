"""Tests for `DifficultyScorer` and tertile binning."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edit2forensics.data.difficulty_artifact import DifficultyArtifact
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.difficulty.instruction import InstructionComplexityScorer
from edit2forensics.difficulty.scorer import (
    DifficultyScorer,
    DifficultyScorerConfig,
    assign_tertile_bins,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FixedComplexity(InstructionComplexityScorer):
    """Returns a fixed score regardless of instruction — for deterministic tests."""

    name = "fixed"

    def __init__(self, value: float):
        self.value = value

    def score(self, instruction: str) -> float:
        return self.value


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (96, 96), color=color).save(path)


def _make_triplet(tmp_path: Path, real_color, edited_color) -> EditTriplet:
    real_p = tmp_path / "real.png"
    edited_p = tmp_path / "edited.png"
    _write_image(real_p, real_color)
    _write_image(edited_p, edited_color)
    return EditTriplet(
        triplet_id="t_001",
        source_dataset="test",
        real_path=real_p,
        edited_path=edited_p,
        instruction="Edit the image somehow.",
    )


def _make_mask_artifact(
    tmp_path: Path,
    triplet_id: str,
    *,
    mask_area_frac: float,
    combined_diff_mean: float,
    edit_scope: str = "local",
) -> MaskArtifact:
    mask_p = tmp_path / f"{triplet_id}_mask.png"
    Image.new("L", (96, 96), 0).save(mask_p)
    return MaskArtifact(
        triplet_id=triplet_id,
        mask_path=mask_p,
        mask_area_frac=mask_area_frac,
        edit_scope=edit_scope,
        registration_ok=True,
        confidence=0.5,
        diff_strongest_signal="lab_pixel",
        combined_diff_mean=combined_diff_mean,
    )


# ---------------------------------------------------------------------------
# Component plumbing
# ---------------------------------------------------------------------------

def test_components_are_in_unit_interval(tmp_path):
    """Every component in the artifact must be in [0, 1]."""
    triplet = _make_triplet(tmp_path, (100, 100, 100), (160, 110, 110))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.1, combined_diff_mean=0.4
    )
    scorer = DifficultyScorer(instruction_scorer=FixedComplexity(0.5))
    art = scorer.score(triplet, mask)

    for v in (
        art.structural_change, art.perceptual_change,
        art.locality, art.instruction_complexity,
    ):
        assert 0.0 <= v <= 1.0


def test_perceptual_change_passes_through_from_mask_artifact(tmp_path):
    triplet = _make_triplet(tmp_path, (100, 100, 100), (100, 100, 100))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.0, combined_diff_mean=0.42
    )
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.0)
    ).score(triplet, mask)
    assert art.perceptual_change == pytest.approx(0.42)


def test_locality_is_one_minus_mask_area(tmp_path):
    triplet = _make_triplet(tmp_path, (100, 100, 100), (100, 100, 100))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.2, combined_diff_mean=0.0
    )
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.0)
    ).score(triplet, mask)
    assert art.locality == pytest.approx(0.8)


def test_identical_images_have_zero_structural_change(tmp_path):
    """SSIM(x, x) == 1, so 1 - SSIM == 0."""
    triplet = _make_triplet(tmp_path, (100, 150, 200), (100, 150, 200))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.0, combined_diff_mean=0.0
    )
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.0)
    ).score(triplet, mask)
    assert art.structural_change == pytest.approx(0.0, abs=1e-3)


def test_different_images_have_positive_structural_change(tmp_path):
    triplet = _make_triplet(tmp_path, (50, 50, 50), (200, 200, 200))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.0, combined_diff_mean=0.0
    )
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.0)
    ).score(triplet, mask)
    assert art.structural_change > 0.1


def test_perceptual_change_is_clipped_to_unit_interval(tmp_path):
    """A corrupted/stale artifact with combined_diff_mean > 1 must not
    propagate that — the scorer clips."""
    triplet = _make_triplet(tmp_path, (100, 100, 100), (100, 100, 100))
    bad_mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.5, combined_diff_mean=1.7
    )
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.5)
    ).score(triplet, bad_mask)
    assert art.perceptual_change == 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_difficulty_raw_is_weighted_sum(tmp_path):
    """With identical images, structural_change ≈ 0 and the other three
    components are easy to control. The aggregate must equal the weighted
    sum, not some other formula."""
    triplet = _make_triplet(tmp_path, (100, 100, 100), (100, 100, 100))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.4, combined_diff_mean=0.6
    )
    scorer = DifficultyScorer(instruction_scorer=FixedComplexity(0.8))
    art = scorer.score(triplet, mask)

    expected = (
        0.30 * art.structural_change
        + 0.30 * 0.6                    # perceptual_change
        + 0.20 * (1.0 - 0.4)            # locality
        + 0.20 * 0.8                    # instruction_complexity
    )
    assert art.difficulty_raw == pytest.approx(expected, abs=1e-6)


def test_custom_weights_are_used(tmp_path):
    triplet = _make_triplet(tmp_path, (100, 100, 100), (100, 100, 100))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.0, combined_diff_mean=1.0
    )
    cfg = DifficultyScorerConfig(weights={
        "structural_change": 0.0,
        "perceptual_change": 1.0,
        "locality": 0.0,
        "instruction_complexity": 0.0,
    })
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.0),
        config=cfg,
    ).score(triplet, mask)
    # Only perceptual_change contributes, and it is 1.0.
    assert art.difficulty_raw == pytest.approx(1.0)


def test_weights_persisted_in_artifact(tmp_path):
    """Per-row weight persistence is what makes mixed-calibration runs
    analyzable post-hoc."""
    triplet = _make_triplet(tmp_path, (100, 100, 100), (100, 100, 100))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.0, combined_diff_mean=0.0
    )
    custom = {
        "structural_change": 0.4, "perceptual_change": 0.4,
        "locality": 0.1, "instruction_complexity": 0.1,
    }
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.0),
        config=DifficultyScorerConfig(weights=custom),
    ).score(triplet, mask)
    assert art.weights == custom


def test_missing_weight_component_raises(tmp_path):
    with pytest.raises(ValueError, match="missing required components"):
        DifficultyScorerConfig(weights={
            "structural_change": 0.5, "perceptual_change": 0.5,
            # missing locality and instruction_complexity
        })


# ---------------------------------------------------------------------------
# Tertile binning
# ---------------------------------------------------------------------------

def test_tertile_bins_split_into_thirds():
    """With a uniform distribution, each bin should hold roughly N/3."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=300)
    bins = assign_tertile_bins(scores)
    counts = {b: int((bins == b).sum()) for b in ("easy", "medium", "hard")}
    # Allow ±5% of N for ties / quantile boundary effects.
    for c in counts.values():
        assert 85 <= c <= 115, f"counts {counts} unbalanced for uniform input"


def test_tertile_bins_preserve_ordering():
    """Hard scores should be >= medium scores >= easy scores."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bins = assign_tertile_bins(scores)
    easy_max = scores[bins == "easy"].max()
    medium_min = scores[bins == "medium"].min()
    medium_max = scores[bins == "medium"].max()
    hard_min = scores[bins == "hard"].min()
    assert easy_max <= medium_min
    assert medium_max <= hard_min


def test_tertile_bins_handle_empty_array():
    out = assign_tertile_bins(np.array([], dtype=np.float32))
    assert out.size == 0


def test_tertile_bins_reject_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        assign_tertile_bins(np.zeros((5, 5)))


def test_tertile_bins_with_constant_scores():
    """All-equal scores: the implementation must still produce some
    valid labeling without crashing. The exact distribution between
    bins is undefined for constant input, but every label must be
    one of the three valid values."""
    scores = np.full(30, 0.5)
    bins = assign_tertile_bins(scores)
    assert set(bins.tolist()) <= {"easy", "medium", "hard"}


# ---------------------------------------------------------------------------
# Artifact roundtripping
# ---------------------------------------------------------------------------

def test_difficulty_artifact_serialization_roundtrip(tmp_path):
    triplet = _make_triplet(tmp_path, (100, 100, 100), (160, 110, 110))
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.15, combined_diff_mean=0.35
    )
    art = DifficultyScorer(
        instruction_scorer=FixedComplexity(0.5)
    ).score(triplet, mask)

    restored = DifficultyArtifact.from_dict(art.to_dict())
    assert restored == art


def test_mask_artifact_roundtrip_preserves_combined_diff_mean(tmp_path):
    """Stage B's new field must round-trip through to_dict/from_dict."""
    mask = _make_mask_artifact(
        tmp_path, "t_001", mask_area_frac=0.1, combined_diff_mean=0.42
    )
    restored = MaskArtifact.from_dict(mask.to_dict())
    assert restored.combined_diff_mean == pytest.approx(0.42)


def test_old_mask_artifact_dict_without_combined_diff_mean(tmp_path):
    """A dict from a pre-this-field Stage B run must still load, with
    combined_diff_mean defaulting to 0.0."""
    mask_p = tmp_path / "old.png"
    Image.new("L", (16, 16), 0).save(mask_p)
    old_dict = {
        "triplet_id": "old_001",
        "mask_path": str(mask_p),
        "mask_area_frac": 0.1,
        "edit_scope": "local",
        "registration_ok": True,
        "confidence": 0.5,
        "diff_strongest_signal": "lab_pixel",
        # combined_diff_mean intentionally absent
    }
    restored = MaskArtifact.from_dict(old_dict)
    assert restored.combined_diff_mean == 0.0

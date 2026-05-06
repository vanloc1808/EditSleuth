"""Tests for ``ReasoningAnnotator`` and the chain composition logic."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edit2forensics.data.category_artifact import CategoryArtifact, EDIT_CATEGORIES
from edit2forensics.data.difficulty_artifact_v2 import DifficultyArtifactV2
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.reasoning_artifact import ReasoningArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.reasoning.annotator import (
    ReasoningAnnotator,
    ReasoningAnnotatorConfig,
)
from edit2forensics.reasoning.templates import (
    CATEGORY_FORENSIC_PRIORS,
    TEMPLATE_VERSION,
    get_forensic_prior,
)


def _write_local_mask(path: Path, h: int = 100, w: int = 100) -> Path:
    """Write a small upper-left local mask covering ~10% of a 100x100 image."""
    arr = np.zeros((h, w), dtype=bool)
    arr[10:30, 10:40] = True  # 20x30 = 600 px → 6% of 10000
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)
    return path


def _write_global_mask(path: Path, h: int = 100, w: int = 100) -> Path:
    arr = np.ones((h, w), dtype=bool)
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)
    return path


def _make_artifacts(
    tmp_path: Path,
    *,
    instruction: str = "add a hat to the dog",
    category: str = "object_addition",
    category_source: str = "dataset_label",
    category_confidence: float = 1.0,
    edit_scope: str = "local",
    mask_area_frac: float = 0.06,
    mask_compactness: float = 0.85,
    structural_change: float = 0.45,
    compactness_score: float = 0.15,
    instruction_complexity: float = 0.5,
    difficulty_raw: float = 0.4,
    difficulty_bin: str = "medium",
    triplet_id: str = "t_001",
):
    """Build a quartet of artifacts for one synthetic triplet."""
    if edit_scope == "global":
        mask_path = _write_global_mask(tmp_path / f"{triplet_id}_mask.png")
    else:
        mask_path = _write_local_mask(tmp_path / f"{triplet_id}_mask.png")

    triplet = EditTriplet(
        triplet_id=triplet_id,
        source_dataset="test",
        real_path=tmp_path / "real.png",
        edited_path=tmp_path / "edited.png",
        instruction=instruction,
    )
    mask = MaskArtifact(
        triplet_id=triplet_id,
        mask_path=mask_path,
        mask_area_frac=mask_area_frac,
        edit_scope=edit_scope,
        registration_ok=True,
        confidence=0.5,
        diff_strongest_signal="lab_pixel",
        combined_diff_mean=0.3,
        mask_compactness=mask_compactness,
    )
    difficulty = DifficultyArtifactV2(
        triplet_id=triplet_id,
        structural_change=structural_change,
        compactness_score=compactness_score,
        instruction_complexity=instruction_complexity,
        difficulty_raw=difficulty_raw,
        difficulty_bin=difficulty_bin,
        weights={"structural_change": 0.55, "compactness_score": 0.25,
                 "instruction_complexity": 0.20},
    )
    cat = CategoryArtifact(
        triplet_id=triplet_id,
        category=category,
        confidence=category_confidence,
        source=category_source,
        rationale=f"label:{category}",
    )
    return triplet, mask, difficulty, cat


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_artifact_to_dict_from_dict_roundtrip(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(tmp_path)
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    restored = ReasoningArtifact.from_dict(art.to_dict())
    assert restored == art


def test_template_version_is_recorded(tmp_path):
    """Each chain records the template version used to generate it.
    Pinning this lets us mix chains from different versions safely."""
    triplet, mask, diff, cat = _make_artifacts(tmp_path)
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert art.template_version == TEMPLATE_VERSION


# ---------------------------------------------------------------------------
# Header structure
# ---------------------------------------------------------------------------

def test_header_contains_all_four_fields(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path,
        category="object_removal",
        edit_scope="local",
        difficulty_bin="hard",
        category_source="rule_based",
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert "category=object_removal" in art.header
    assert "scope=local" in art.header
    assert "difficulty=hard" in art.header
    assert "source=rule_based" in art.header


# ---------------------------------------------------------------------------
# Mirrored fields
# ---------------------------------------------------------------------------

def test_mirrored_fields_match_inputs(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path,
        category="style_transfer",
        difficulty_bin="easy",
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert art.category == "style_transfer"
    assert art.difficulty_bin == "easy"


# ---------------------------------------------------------------------------
# Spatial descriptor in chain
# ---------------------------------------------------------------------------

def test_global_scope_chain_uses_whole_image_phrase(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path,
        edit_scope="global",
        mask_area_frac=1.0,
        category="style_transfer",
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert art.spatial_descriptor == "whole_image"
    assert "spans the entire image" in art.chain


def test_local_scope_chain_uses_quadrant_phrase(tmp_path):
    """The default _write_local_mask puts the mask in upper-left."""
    triplet, mask, diff, cat = _make_artifacts(tmp_path, edit_scope="local")
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert art.spatial_descriptor == "upper_left"
    assert "upper-left region" in art.chain


# ---------------------------------------------------------------------------
# Per-category forensic-signature inclusion
# ---------------------------------------------------------------------------

def test_chain_contains_category_forensic_prior(tmp_path):
    """The forensic-signature step (Step 5) must include the
    hand-curated prior for the classified category."""
    triplet, mask, diff, cat = _make_artifacts(tmp_path, category="object_addition")
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    prior = get_forensic_prior("object_addition")
    # Take a distinctive substring of the prior — contains
    # "boundary discontinuities at the edge"
    assert "boundary discontinuities" in art.chain


def test_different_categories_get_different_priors(tmp_path):
    """Sanity: priors are category-specific, not cookie-cutter."""
    triplet1, mask1, diff1, cat1 = _make_artifacts(
        tmp_path, category="photometric", triplet_id="p_001",
    )
    triplet2, mask2, diff2, cat2 = _make_artifacts(
        tmp_path, category="object_addition", triplet_id="p_002",
    )
    annotator = ReasoningAnnotator()
    art1 = annotator.annotate(triplet1, mask1, diff1, cat1)
    art2 = annotator.annotate(triplet2, mask2, diff2, cat2)
    # Photometric prior mentions "histogram shift"; addition mentions
    # "boundary discontinuities". They should be present in their
    # respective chains and absent from the other.
    assert "histogram shift" in art1.chain
    assert "histogram shift" not in art2.chain
    assert "boundary discontinuities" in art2.chain
    assert "boundary discontinuities" not in art1.chain


def test_all_canonical_categories_have_priors():
    """Every category in EDIT_CATEGORIES must have a non-empty prior
    in the templates dict — guards against typos or missing entries."""
    for category in EDIT_CATEGORIES:
        prior = CATEGORY_FORENSIC_PRIORS.get(category)
        assert prior is not None, f"missing prior for {category}"
        assert len(prior) > 20, f"prior for {category} is suspiciously short"


# ---------------------------------------------------------------------------
# Magnitude phrasing thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("structural,expected_phrase", [
    (0.10, "minor"),
    (0.40, "moderate"),
    (0.70, "substantial"),
])
def test_magnitude_phrase_thresholds(structural, expected_phrase, tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, structural_change=structural,
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert expected_phrase in art.chain


@pytest.mark.parametrize("compact_score,expected_phrase", [
    (0.10, "well-concentrated"),
    (0.45, "moderately concentrated"),
    (0.80, "diffuse"),
])
def test_compactness_phrase_thresholds(compact_score, expected_phrase, tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, compactness_score=compact_score,
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert expected_phrase in art.chain


# ---------------------------------------------------------------------------
# Difficulty phrasing per bin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bin_label,expected_phrase", [
    ("easy", "easier than average"),
    ("medium", "moderate detection difficulty"),
    ("hard", "harder than average"),
])
def test_difficulty_phrase_per_bin(bin_label, expected_phrase, tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, difficulty_bin=bin_label,
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert expected_phrase in art.chain


# ---------------------------------------------------------------------------
# Category source phrasing
# ---------------------------------------------------------------------------

def test_dataset_label_source_phrasing(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, category_source="dataset_label", category_confidence=1.0,
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert "curated edit-type label" in art.chain


def test_rule_based_source_phrasing(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, category_source="rule_based", category_confidence=0.85,
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert "rule-based" in art.chain
    assert "0.85" in art.chain  # confidence appears in the chain


def test_fallback_source_phrasing(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, category_source="fallback", category_confidence=0.2,
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert "could not be determined confidently" in art.chain


# ---------------------------------------------------------------------------
# Instruction handling
# ---------------------------------------------------------------------------

def test_chain_quotes_the_instruction(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, instruction="put a santa hat on the cat",
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert '"put a santa hat on the cat"' in art.chain


def test_long_instruction_is_truncated(tmp_path):
    """Chains over ~200 words risk teaching the VLM to generate
    padding. We truncate very long instructions to keep chain length
    bounded."""
    long_instr = "x" * 500
    triplet, mask, diff, cat = _make_artifacts(tmp_path, instruction=long_instr)
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    # Truncated form ends in "..."
    assert "..." in art.chain
    # The chain shouldn't contain the full 500 chars
    assert "x" * 300 not in art.chain


def test_empty_instruction_uses_inferred_phrasing(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(tmp_path, instruction="")
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert "inferred from visual evidence" in art.chain


# ---------------------------------------------------------------------------
# Chain shape (numbered steps, all six present)
# ---------------------------------------------------------------------------

def test_chain_contains_six_numbered_steps(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(tmp_path)
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    # Each step starts with "N. " on its own line.
    for n in range(1, 7):
        assert f"{n}. " in art.chain


def test_chain_is_not_too_long(tmp_path):
    """Sanity check: chains should be bounded in length so VLM
    training doesn't learn to generate padding."""
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path, instruction="add a hat",
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    word_count = len(art.chain.split())
    assert 50 <= word_count <= 250, (
        f"chain word count {word_count} outside expected range"
    )


# ---------------------------------------------------------------------------
# Alignment-failed and edge cases
# ---------------------------------------------------------------------------

def test_alignment_failed_chain_states_failure(tmp_path):
    triplet, mask, diff, cat = _make_artifacts(
        tmp_path,
        edit_scope="alignment_failed",
        mask_area_frac=0.0,
    )
    annotator = ReasoningAnnotator()
    art = annotator.annotate(triplet, mask, diff, cat)
    assert art.spatial_descriptor == "alignment_failed"
    assert "alignment failure" in art.chain.lower()

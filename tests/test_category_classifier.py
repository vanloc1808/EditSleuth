"""Tests for ``CategoryClassifier`` and the canonical schema."""
from __future__ import annotations

from pathlib import Path

import pytest

from edit2forensics.category.classifier import (
    CategoryClassifier,
    PICO_BANANA_LABEL_MAP,
    category_distribution,
)
from edit2forensics.data.category_artifact import (
    CategoryArtifact,
    EDIT_CATEGORIES,
)
from edit2forensics.data.triplet import EditTriplet


def _triplet(
    *,
    triplet_id: str = "t0",
    instruction: str = "",
    source_dataset: str = "test",
    metadata: dict | None = None,
) -> EditTriplet:
    """Fixture builder. Uses fake paths since the classifier never
    opens images."""
    return EditTriplet(
        triplet_id=triplet_id,
        source_dataset=source_dataset,
        real_path=Path("/nonexistent/real.png"),
        edited_path=Path("/nonexistent/edited.png"),
        instruction=instruction,
        metadata=metadata or {},
    )


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

def test_artifact_roundtrip_preserves_all_fields():
    a = CategoryArtifact(
        triplet_id="x",
        category="object_addition",
        confidence=0.85,
        source="rule_based",
        rationale="rule:add",
    )
    restored = CategoryArtifact.from_dict(a.to_dict())
    assert restored == a


def test_from_dict_raises_diagnostic_on_missing_field():
    """If an upstream parquet is missing a required field, from_dict
    should raise a ValueError that includes the available keys, not
    a bare KeyError. This was added after a Stage E error of the
    form ``annotate failed for X: 'category'`` left users guessing
    which parquet was malformed."""
    incomplete = {
        "triplet_id": "magicbrush_dev_515743_t02",
        "confidence": 0.5,
        "source": "rule_based",
        "rationale": "rule:something",
        # 'category' missing!
    }
    with pytest.raises(ValueError) as exc_info:
        CategoryArtifact.from_dict(incomplete)
    msg = str(exc_info.value)
    assert "missing required fields ['category']" in msg
    assert "magicbrush_dev_515743_t02" in msg
    # Available keys should appear in the message.
    assert "available keys" in msg
    assert "confidence" in msg


def test_from_dict_lists_all_missing_fields():
    """When multiple required fields are missing, the error message
    lists them all so the diagnosis is complete in one shot."""
    very_incomplete = {"triplet_id": "x"}
    with pytest.raises(ValueError) as exc_info:
        CategoryArtifact.from_dict(very_incomplete)
    msg = str(exc_info.value)
    for field in ("category", "confidence", "source", "rationale"):
        assert field in msg


def test_edit_categories_constant_matches_literal():
    """The EDIT_CATEGORIES tuple must list exactly the canonical
    categories. If a new category is added to the Literal, the
    tuple must be updated too — this test pins that contract."""
    assert len(EDIT_CATEGORIES) == 12
    assert "other" in EDIT_CATEGORIES
    assert "photometric" in EDIT_CATEGORIES   # added April 2026
    assert "human_centric" in EDIT_CATEGORIES  # added April 2026
    assert len(set(EDIT_CATEGORIES)) == len(EDIT_CATEGORIES)  # no duplicates


def test_category_distribution_includes_zero_buckets():
    """The helper always returns counts for every category, including
    those with zero artifacts. Downstream tooling shouldn't have to
    handle missing keys."""
    arts = [
        CategoryArtifact(triplet_id="a", category="object_addition",
                         confidence=1.0, source="dataset_label", rationale=""),
        CategoryArtifact(triplet_id="b", category="object_addition",
                         confidence=1.0, source="dataset_label", rationale=""),
        CategoryArtifact(triplet_id="c", category="style_transfer",
                         confidence=1.0, source="dataset_label", rationale=""),
    ]
    dist = category_distribution(arts)
    assert set(dist.keys()) == set(EDIT_CATEGORIES)
    assert dist["object_addition"] == 2
    assert dist["style_transfer"] == 1
    assert dist["object_removal"] == 0  # zero bucket


# ----------------------------------------------------------------------
# Path 1: dataset label mapping
# ----------------------------------------------------------------------

def test_pico_banana_label_maps_to_canonical_category():
    """A Pico-Banana triplet with a recognized source_edit_type
    should be routed via the label-mapping path with confidence 1.0."""
    clf = CategoryClassifier()
    t = _triplet(
        source_dataset="pico_banana",
        metadata={"source_edit_type": "Add a new object to the scene"},
        instruction="(any instruction text — should be ignored when label is present)",
    )
    art = clf.classify(t)
    assert art.category == "object_addition"
    assert art.confidence == 1.0
    assert art.source == "dataset_label"
    assert "Add a new object to the scene" in art.rationale


def test_pico_banana_source_dataset_string_with_hyphen():
    """The Pico-Banana adapter writes ``source_dataset='pico-banana'``
    (with hyphen). The classifier must dispatch the same as for
    ``'pico_banana'`` (underscore) — historical bug had the dispatch
    fail silently and route everything through the rule path."""
    clf = CategoryClassifier()
    t = _triplet(
        source_dataset="pico-banana",  # actual adapter spelling
        metadata={"source_edit_type": "Add a new object to the scene"},
        instruction="(should be ignored when label is present)",
    )
    art = clf.classify(t)
    assert art.category == "object_addition"
    assert art.source == "dataset_label"
    assert art.confidence == 1.0


def test_source_dataset_dispatch_is_case_insensitive():
    clf = CategoryClassifier()
    for variant in ("Pico-Banana", "PICO-BANANA", "pico_banana", "Pico_Banana"):
        t = _triplet(
            source_dataset=variant,
            metadata={"source_edit_type": "Add a new object to the scene"},
        )
        art = clf.classify(t)
        assert art.category == "object_addition", (
            f"failed for source_dataset={variant!r}, got {art.category}"
        )
        assert art.source == "dataset_label"



    clf = CategoryClassifier()
    for variant in ("ADD A NEW OBJECT TO THE SCENE", "Add A New Object To The Scene", "  add a new object to the scene  "):
        t = _triplet(
            source_dataset="pico_banana",
            metadata={"source_edit_type": variant},
        )
        art = clf.classify(t)
        assert art.category == "object_addition", f"failed on variant {variant!r}"


def test_unmapped_source_label_falls_to_other_with_unknown_confidence():
    """A label exists but isn't in the map. We don't fail loudly —
    we route to 'other' with the unknown-label confidence (default
    0.5) so the unmapped label is visible in the parquet for
    auditing without breaking the pipeline."""
    clf = CategoryClassifier()
    t = _triplet(
        source_dataset="pico_banana",
        metadata={"source_edit_type": "Some Brand New Category That Doesn't Exist"},
    )
    art = clf.classify(t)
    assert art.category == "other"
    assert art.source == "dataset_label"
    assert art.confidence == 0.5
    assert "unmapped:" in art.rationale


def test_label_map_dispatches_by_source_dataset():
    """A non-pico_banana triplet with an unrelated label dict shouldn't
    pull from PICO_BANANA_LABEL_MAP — it should fall to the rule path."""
    clf = CategoryClassifier()
    t = _triplet(
        source_dataset="magicbrush",
        metadata={"source_edit_type": "add a new object to the scene"},
        instruction="put a hat on the dog",
    )
    art = clf.classify(t)
    # Should NOT use the metadata (wrong dataset). Should fall to rules,
    # which match "put" -> object_addition.
    assert art.source == "rule_based"
    assert art.category == "object_addition"


# ----------------------------------------------------------------------
# Path 2: rule-based classification
# ----------------------------------------------------------------------

@pytest.mark.parametrize("instruction,expected_category,expected_rationale_substring", [
    ("Add a red hat to the dog", "object_addition", "add"),
    ("Insert a small bird in the sky", "object_addition", "add"),
    ("Place a coffee cup on the table", "object_addition", "add"),
    ("Remove the person from the photo", "object_removal", "remove"),
    ("Delete the watermark", "object_removal", "remove"),
    ("Erase the background text", "object_removal", "remove"),
    ("Get rid of the trash bin", "object_removal", "remove"),
    ("Replace the apple with an orange", "object_replacement", "replace"),
    ("Swap the chair for a sofa", "object_replacement", "swap"),
    ("Turn the cat into a dog", "object_replacement", "turn-into"),
    ("Make the car red", "attribute_change", "color"),
    ("Change the color of the dress to blue", "attribute_change", "color"),
    ("Make the chair wooden", "attribute_change", "material"),
    ("Convert the photo to a Van Gogh style painting", "style_transfer", "style-of"),
    ("Convert this image to a cartoon", "style_transfer", "style-keyword"),
    ("Change to a watercolor", "style_transfer", "style-keyword"),
    ("Make it night time", "scene_transformation", "time-of-day"),
    ("Make it look like a foggy day", "scene_transformation", "weather"),
    ("Add some snow to the scene", "scene_transformation", "weather"),
    ("Change the season to winter", "scene_transformation", "season"),
    ("Adjust the lighting to be brighter", "scene_transformation", "lighting"),
    ("Replace the background with a forest", "background_change", "background"),  # 'background' beats 'replace' — semantic precedence
    ("Change the background to a forest", "background_change", "background"),
    ("Remove the text from the sign", "object_removal", "remove"),  # 'remove' beats 'text'
    ("Add the word HELLO to the sign", "object_addition", "add"),  # 'add' beats 'text'
    ("Modify the text on the label", "text_edit", "text-keyword"),
    ("Crop the image to a square", "geometric", "geometric-op"),
    ("Rotate the image 90 degrees", "geometric", "geometric-op"),
    ("Flip horizontally", "geometric", "geometric-op"),
    # ===================================================================
    # MagicBrush conversational patterns
    # ===================================================================
    # Object addition via "give X Y" / "let X have Y" / "wearing/holding"
    ("Give him a hat", "object_addition", "give"),
    ("Give the dog a frisbee", "object_addition", "give"),
    ("Have the man wearing a coat", "object_addition", "wear-hold"),
    ("She is holding a flower", "object_addition", "wear-hold"),
    # Attribute change via conversational frames + color
    ("Let it be red", "attribute_change", "frame-color"),
    ("Have the cat be orange", "attribute_change", "frame-color"),
    ("Make him purple", "attribute_change", "color"),
    # Scene transformation via inflected weather/lighting words
    ("Let it be raining", "scene_transformation", "weather"),
    ("Make it snowing outside", "scene_transformation", "weather"),
    ("Make the room darker", "scene_transformation", "lighting"),
    ("Let the scene be brighter", "scene_transformation", "lighting"),
    # 'is now X' attribute frame (descriptive form, not imperative)
    ("The cat is now red", "attribute_change", "frame-color"),
    ("The dog is now blue", "attribute_change", "frame-color"),
    ("The flowers are now purple", "attribute_change", "frame-color"),
    # Size comparatives
    ("The dog is bigger", "attribute_change", "size"),
    ("Make the cat smaller", "attribute_change", "size"),
    ("Have the man taller", "attribute_change", "size"),
    # Holiday / seasonal scene
    ("Change the scene to christmas", "scene_transformation", "holiday"),
    ("Halloween decorations everywhere", "scene_transformation", "holiday"),
    ("Easter theme", "scene_transformation", "holiday"),
])
def test_rule_based_classification(instruction, expected_category, expected_rationale_substring):
    """End-to-end check on rule patterns. The expected categories
    reflect intentional rule precedence (e.g. 'replace' wins over
    'background'). When precedence intuition disagrees, the test
    will surface it."""
    clf = CategoryClassifier()
    t = _triplet(instruction=instruction)
    art = clf.classify(t)
    assert art.category == expected_category, (
        f"instruction={instruction!r} expected={expected_category} "
        f"got={art.category} rationale={art.rationale}"
    )
    assert art.source == "rule_based"
    assert expected_rationale_substring in art.rationale


def test_no_rule_match_yields_fallback_other():
    """An instruction with no rule keywords falls to the 'other'
    bucket with low confidence and a fallback source tag."""
    clf = CategoryClassifier()
    t = _triplet(instruction="kfjlds qpwo asdfghjkl")
    art = clf.classify(t)
    assert art.category == "other"
    assert art.source == "fallback"
    assert art.confidence < 0.3
    assert "no-rule-matched" in art.rationale


def test_empty_instruction_yields_fallback():
    clf = CategoryClassifier()
    t = _triplet(instruction="")
    art = clf.classify(t)
    assert art.category == "other"
    assert art.source == "fallback"


def test_pico_banana_label_takes_precedence_over_instruction():
    """When both a recognized source label AND a strong rule-matching
    instruction are present, the label path wins (it's higher-
    confidence per the design)."""
    clf = CategoryClassifier()
    t = _triplet(
        source_dataset="pico_banana",
        metadata={"source_edit_type": "Remove an existing object"},
        # Instruction text that would otherwise match the 'add' rule:
        instruction="add a cup to the table",
    )
    art = clf.classify(t)
    assert art.category == "object_removal"  # from label, not from rule
    assert art.source == "dataset_label"


# ----------------------------------------------------------------------
# Map coverage sanity
# ----------------------------------------------------------------------

def test_pico_banana_label_map_covers_real_dataset_labels():
    """Spot-check using one example per canonical category. The label
    strings here are taken from the actual Pico-Banana release manifest
    (as scanned by scripts/scan_source_labels.py); if any of these
    don't map, the corresponding entry was removed from
    PICO_BANANA_LABEL_MAP and downstream Stage D would silently
    misclassify thousands of triplets to ``other``.

    The mappings here reflect the April 2026 taxonomy revision that
    adopted the Pico-Banana paper's own 8-category grouping for
    photometric and human-centric edits.
    """
    real_label_to_canonical = [
        ("Add a new object to the scene", "object_addition"),
        ("Remove an existing object", "object_removal"),
        ("Replace one object category with another", "object_replacement"),
        ("Change an object's attribute (e.g., color/material)", "attribute_change"),
        ("Strong artistic style transfer (e.g., Van Gogh/anime/etc.)", "style_transfer"),
        # photometric: paper's "Pixel & Photometric" category
        ("Add film grain or vintage filter", "photometric"),
        ("Change overall color tone (warm ↔ cool)", "photometric"),
        # scene_transformation: includes scene context/background per paper grouping
        ("Adjust global lighting (golden hour/fluorescent)", "scene_transformation"),
        ("Add new scene context/background", "scene_transformation"),
        ("Add new (handwritten/printed/etc) text", "text_edit"),
        ("Zoom in", "geometric"),
        # human_centric: paper's "Human-Centric" category
        ("Pose tweak (minor plausible change)", "human_centric"),
        ("Funko-Pop–style toy figure of the person", "human_centric"),
        ("Convert person to 2D anime/manga style (identity-preserving)", "human_centric"),
        ("Clothing edit (change color/outfit)", "human_centric"),
    ]
    clf = CategoryClassifier()
    for label, expected_canonical in real_label_to_canonical:
        t = _triplet(
            source_dataset="pico-banana",
            metadata={"source_edit_type": label},
        )
        art = clf.classify(t)
        assert art.category == expected_canonical, (
            f"label={label!r} expected={expected_canonical!r} "
            f"got={art.category!r} rationale={art.rationale!r}"
        )
        assert art.source == "dataset_label"
        assert art.confidence == 1.0



    """Every value in the Pico-Banana label map must be one of the
    canonical categories — guards against typos."""
    for source_label, canonical in PICO_BANANA_LABEL_MAP.items():
        assert canonical in EDIT_CATEGORIES, (
            f"label {source_label!r} maps to non-canonical {canonical!r}"
        )


def test_pico_banana_label_map_keys_are_lowercase_and_stripped():
    """Map keys must be lowercase and stripped, since lookup
    normalizes the same way. This test catches accidental whitespace
    or capitalization in the map definition."""
    for key in PICO_BANANA_LABEL_MAP:
        assert key == key.strip().lower(), (
            f"map key {key!r} is not normalized"
        )

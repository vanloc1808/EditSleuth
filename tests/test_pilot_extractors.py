"""Tests for the pilot evaluation field extractor.

A bug in the original extractor reported 0% difficulty-bin accuracy
even when the chain-target model produced correct chains: the
extractor only looked for the bin in a structured header that the
chain dataset never trains the model to produce, while the actual
bin information lived in step 6 of the prose. The current extractor
reads from prose, and these tests pin that behavior so it doesn't
regress.
"""
from __future__ import annotations

import sys
from pathlib import Path

# pilot_evaluate.py lives in scripts/ alongside other pilot tools.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _import_extractors():
    """Import lazily so tests don't fail to even collect when pilot
    deps (transformers, peft) aren't installed. The extractors
    themselves only need the standard library."""
    # Stub out heavy imports that pilot_evaluate.py performs at the
    # module level but doesn't actually need for the extractor logic.
    import types
    sys.modules.setdefault("hydra", types.ModuleType("hydra"))
    omegaconf_mod = types.ModuleType("omegaconf")
    omegaconf_mod.DictConfig = object
    omegaconf_mod.OmegaConf = type("OmegaConf", (), {"to_yaml": staticmethod(lambda *a, **k: "")})
    sys.modules.setdefault("omegaconf", omegaconf_mod)
    if not hasattr(sys.modules["hydra"], "main"):
        sys.modules["hydra"].main = lambda **kwargs: (lambda f: f)
    from pilot_evaluate import _extract_chain_fields, _extract_label_only_fields
    return _extract_chain_fields, _extract_label_only_fields


# ---------------------------------------------------------------------------
# Chain extractor: difficulty bin (the field that originally scored 0%)
# ---------------------------------------------------------------------------

def test_chain_extractor_finds_easy_bin_from_step6():
    extract, _ = _import_extractors()
    chain = (
        "1. The edit instruction states: \"add a sticker\". "
        "2. The mask is centered in the image. "
        "6. Overall, this triplet is easier than average to detect."
    )
    result = extract(chain)
    assert result["difficulty_bin"] == "easy"


def test_chain_extractor_finds_medium_bin_from_step6():
    extract, _ = _import_extractors()
    chain = "6. Overall, this triplet is of moderate detection difficulty."
    assert extract(chain)["difficulty_bin"] == "medium"


def test_chain_extractor_finds_hard_bin_from_step6():
    extract, _ = _import_extractors()
    chain = "6. Overall, this triplet is harder than average to detect."
    assert extract(chain)["difficulty_bin"] == "hard"


def test_chain_extractor_returns_none_when_no_difficulty_phrase():
    extract, _ = _import_extractors()
    chain = "Some text without any difficulty hint."
    assert extract(chain)["difficulty_bin"] is None


# ---------------------------------------------------------------------------
# Chain extractor: spatial descriptor (was missing several variants)
# ---------------------------------------------------------------------------

def test_chain_extractor_finds_whole_image_spatial():
    extract, _ = _import_extractors()
    chain = "2. The mask spans the entire image."
    assert extract(chain)["spatial_descriptor"] == "whole_image"


def test_chain_extractor_finds_centered_spatial():
    extract, _ = _import_extractors()
    chain = "2. The mask is centered in the image."
    assert extract(chain)["spatial_descriptor"] == "centered"


def test_chain_extractor_finds_quadrant_spatial():
    extract, _ = _import_extractors()
    chain = "2. The mask is concentrated in the upper-left region."
    assert extract(chain)["spatial_descriptor"] == "upper_left"


def test_chain_extractor_handles_quadrant_without_hyphen():
    """Some model generations may produce 'upper left' without hyphen."""
    extract, _ = _import_extractors()
    chain = "2. The mask is concentrated in the upper left region."
    assert extract(chain)["spatial_descriptor"] == "upper_left"


def test_chain_extractor_finds_scattered_spatial():
    extract, _ = _import_extractors()
    chain = "2. The mask is scattered across multiple regions."
    assert extract(chain)["spatial_descriptor"] == "scattered"


# ---------------------------------------------------------------------------
# Chain extractor: category from prose (with underscore and space variants)
# ---------------------------------------------------------------------------

def test_chain_extractor_finds_category_with_underscore():
    extract, _ = _import_extractors()
    chain = "4. The edit is classified as object_addition, based on the label."
    assert extract(chain)["category"] == "object_addition"


def test_chain_extractor_finds_category_with_space():
    """Models often drift between underscore and space forms."""
    extract, _ = _import_extractors()
    chain = "4. The edit is classified as object addition, based on the label."
    assert extract(chain)["category"] == "object_addition"


def test_chain_extractor_finds_two_word_categories():
    extract, _ = _import_extractors()
    for cat, prose_form in (
        ("style_transfer", "style transfer"),
        ("scene_transformation", "scene transformation"),
        ("text_edit", "text edit"),
        ("human_centric", "human centric"),
    ):
        chain = f"4. The edit is classified as {prose_form}, based on the label."
        assert extract(chain)["category"] == cat, (
            f"failed to extract {cat} from prose form '{prose_form}'"
        )


def test_chain_extractor_returns_none_for_unknown_category():
    extract, _ = _import_extractors()
    chain = "4. The edit is classified as foobar, based on the label."
    assert extract(chain)["category"] is None


# ---------------------------------------------------------------------------
# Full chain (regression test: end-to-end extraction)
# ---------------------------------------------------------------------------

def test_chain_extractor_full_chain_produces_all_fields():
    """Regression: a complete six-step chain should yield non-None
    values for all three fields. This was the actual production
    failure that motivated the extractor rewrite."""
    extract, _ = _import_extractors()
    chain = (
        "1. The edit instruction states: \"add a hat to the dog\".\n"
        "2. The mask of changed pixels covers roughly 6% of the image "
        "and is concentrated in the upper-left region.\n"
        "3. Structural change relative to the original is moderate.\n"
        "4. The edit is classified as object_addition, based on the "
        "dataset's curated edit-type label.\n"
        "5. Edits of this type typically exhibit boundary discontinuities.\n"
        "6. Overall, this triplet is of moderate detection difficulty."
    )
    result = extract(chain)
    assert result["category"] == "object_addition"
    assert result["spatial_descriptor"] == "upper_left"
    assert result["difficulty_bin"] == "medium"


def test_chain_extractor_handles_global_edit_chain():
    extract, _ = _import_extractors()
    chain = (
        "1. The edit instruction states: \"convert to van gogh style\".\n"
        "2. The mask of changed pixels spans the entire image.\n"
        "3. Structural change is substantial.\n"
        "4. The edit is classified as style_transfer, based on the dataset label.\n"
        "5. Edits of this type typically exhibit global texture patterns.\n"
        "6. Overall, this triplet is harder than average to detect."
    )
    result = extract(chain)
    assert result["category"] == "style_transfer"
    assert result["spatial_descriptor"] == "whole_image"
    assert result["difficulty_bin"] == "hard"


# ---------------------------------------------------------------------------
# Header path (still supported for future training runs that include it)
# ---------------------------------------------------------------------------

def test_chain_extractor_uses_header_when_present():
    extract, _ = _import_extractors()
    text = (
        "[category=object_removal, scope=local, difficulty=easy, source=dataset_label]\n"
        "1. The edit instruction states: \"remove the cup\"."
    )
    result = extract(text)
    assert result["category"] == "object_removal"
    assert result["difficulty_bin"] == "easy"


# ---------------------------------------------------------------------------
# Label-only extractor (unchanged but worth pinning)
# ---------------------------------------------------------------------------

def test_label_only_extractor_parses_clean_json():
    _, extract = _import_extractors()
    text = '{"category": "object_addition", "spatial_descriptor": "centered", "difficulty_bin": "medium"}'
    result = extract(text)
    assert result["category"] == "object_addition"
    assert result["spatial_descriptor"] == "centered"
    assert result["difficulty_bin"] == "medium"


def test_label_only_extractor_returns_none_on_invalid_json():
    _, extract = _import_extractors()
    text = "this is not json"
    result = extract(text)
    assert result == {"category": None, "spatial_descriptor": None, "difficulty_bin": None}


def test_label_only_extractor_validates_canonical_values():
    _, extract = _import_extractors()
    text = '{"category": "fake_category", "spatial_descriptor": "centered", "difficulty_bin": "medium"}'
    result = extract(text)
    assert result["category"] is None  # invalid category rejected
    assert result["spatial_descriptor"] == "centered"
    assert result["difficulty_bin"] == "medium"

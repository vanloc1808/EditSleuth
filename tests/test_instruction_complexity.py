"""Tests for `RuleBasedInstructionComplexity`."""
from __future__ import annotations

import pytest

from edit2forensics.difficulty.instruction import RuleBasedInstructionComplexity


def test_empty_instruction_scores_zero():
    s = RuleBasedInstructionComplexity()
    assert s.score("") == 0.0


def test_punctuation_only_instruction_scores_zero():
    """Tokenization should leave nothing useful; result is 0 not error."""
    s = RuleBasedInstructionComplexity()
    assert s.score("!!! ??? ...") == 0.0


def test_short_simple_instruction_is_low():
    """A short, single-verb, non-spatial instruction should score low."""
    s = RuleBasedInstructionComplexity()
    score = s.score("Brighten the image.")
    assert 0.0 < score < 0.3


def test_long_instruction_increases_length_subscore():
    s = RuleBasedInstructionComplexity()
    short = s.score("Brighten the image.")
    long = s.score(
        "Brighten the image while preserving sharpness and avoiding "
        "any color shift in the shadows or highlights, retaining the "
        "original tonal balance overall, and keeping all details intact."
    )
    assert long > short


def test_multi_clause_instruction_increases_complexity():
    """Conjunction tokens raise the conjunction sub-score."""
    s = RuleBasedInstructionComplexity()
    single = s.score("Remove the watch.")
    multi = s.score("Remove the watch and replace the strap.")
    assert multi > single


def test_spatial_qualifiers_increase_complexity():
    """Instructions specifying location are typically subtler edits."""
    s = RuleBasedInstructionComplexity()
    no_loc = s.score("Add a flower.")
    with_loc = s.score("Add a flower in the upper-left corner.")
    assert with_loc > no_loc


def test_multiple_edit_verbs_increase_complexity():
    s = RuleBasedInstructionComplexity()
    one_op = s.score("Remove the dog.")
    multi_op = s.score("Remove the dog, replace it with a cat, and recolor the floor.")
    assert multi_op > one_op


def test_score_in_unit_interval():
    """Output must always be in [0, 1] regardless of how 'complex' input is."""
    s = RuleBasedInstructionComplexity()
    extreme = (
        "Remove and replace and add and modify the upper left corner and "
        "lower right edge near the center, then recolor and brighten it "
        "while shifting and rotating the side. " * 5
    )
    score = s.score(extreme)
    assert 0.0 <= score <= 1.0


def test_case_insensitivity():
    """Tokens are matched case-insensitively."""
    s = RuleBasedInstructionComplexity()
    lower = s.score("remove the dog from the left.")
    upper = s.score("REMOVE the DOG from the LEFT.")
    assert lower == upper


def test_pico_banana_style_instruction_is_nontrivial():
    """A real Pico-Banana instruction should score in the meaningful
    middle range — not 0, not 1."""
    s = RuleBasedInstructionComplexity()
    real = (
        "Remove the red flag and its white pole from the upper right "
        "of the image, seamlessly extending the clear blue sky, the "
        "sandy dune with its subtle texture, and the wooden fence to "
        "fill the void."
    )
    score = s.score(real)
    assert 0.2 < score < 1.0


def test_feature_weights_must_be_nonneg_and_positive_sum():
    with pytest.raises(ValueError, match="non-negative"):
        RuleBasedInstructionComplexity(feature_weights=(0.5, -0.1, 0.5, 0.5))
    with pytest.raises(ValueError, match="sum to"):
        RuleBasedInstructionComplexity(feature_weights=(0.0, 0.0, 0.0, 0.0))


def test_weight_change_affects_output():
    """If we zero out the length weight, two instructions identical in
    other features but differing in length should score the same.

    We choose instructions carefully: both contain exactly one edit
    verb ("Add") and no conjunctions/spatial qualifiers/other edit-verb
    nouns. Then the only feature differing between them is length.
    """
    only_length_matters = RuleBasedInstructionComplexity(
        feature_weights=(1.0, 0.0, 0.0, 0.0)
    )
    no_length = RuleBasedInstructionComplexity(
        feature_weights=(0.0, 0.5, 0.5, 0.0)
    )
    short = "Add a hat."
    long = "Add a wide-brimmed straw hat with a thin leather band."

    # With only_length_matters, longer should score higher.
    assert only_length_matters.score(long) > only_length_matters.score(short)
    # With length disabled, both lacking other features should score
    # similarly low.
    assert abs(no_length.score(long) - no_length.score(short)) < 1e-6

"""Strategies for scoring natural-language instruction complexity.

There are three options (rule-based, LLM-based,
embedding-based) and recommend starting with rule-based and validating
against LLM ratings on a subset. We ship the rule-based scorer here.
LLM-based variants slot into the same interface when needed — adding
one is a config change.

Why pluggable
-------------
Instruction complexity is the most subjective component of the
difficulty formula. Reviewers will want to see how robust the final
difficulty distribution is to the choice of complexity scorer. Making
the strategy a config knob turns "swap rule-based for LLM-based" into
a one-line override, which is the right level of friction for an
ablation.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod


class InstructionComplexityScorer(ABC):
    """Abstract scorer producing a complexity in ``[0, 1]``."""

    name: str = ""

    @abstractmethod
    def score(self, instruction: str) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ----------------------------------------------------------------------
# Rule-based scorer
# ----------------------------------------------------------------------

# Words that suggest precise spatial localization is required, which
# typically corresponds to a more subtle edit (you need to find the
# *upper-left corner* — the edit is local).
_SPATIAL_QUALIFIERS = frozenset({
    "left", "right", "top", "bottom", "upper", "lower", "center", "centre",
    "middle", "above", "below", "behind", "front", "near", "next",
    "corner", "edge", "side",
})

# Tokens that mark composition / multi-step structure. The presence of
# any of these suggests the instruction is not a single primitive op.
_CONJUNCTION_TOKENS = frozenset({"and", "while", "then", "also", "but"})

# Verbs that look like edit operations (rough; case-insensitive). Used
# to count how many distinct operations the instruction implies.
_EDIT_VERBS = frozenset({
    "add", "remove", "replace", "change", "modify", "transform", "convert",
    "swap", "shift", "make", "turn", "set", "put", "move", "rotate",
    "delete", "erase", "paint", "color", "recolor", "blur", "sharpen",
    "lighten", "darken", "brighten", "tint",
})

_TOKEN_RE = re.compile(r"[A-Za-z']+")


class RuleBasedInstructionComplexity(InstructionComplexityScorer):
    """Cheap, transparent complexity score based on linguistic features.

    Each feature contributes a sub-score in ``[0, 1]``, scaled by a
    small per-feature weight, and the sum is clipped to ``[0, 1]``.

    Features
    --------
    * Length (tokens). Longer instructions tend to specify more
      constraints, hence more complex edits. Saturates at 25 tokens —
      pico-banana median is ~15 tokens, longest instructions are
      usually verbose specifications of textures/lighting.
    * Edit-verb count. More verbs = more distinct operations.
      Saturates at 3.
    * Conjunction count. Multi-clause instructions are harder.
      Saturates at 2.
    * Spatial qualifier count. Instructions that bother specifying
      spatial location are usually about local edits. Saturates at 2.

    Why these specific feature weights
    -----------------------------------
    They are deliberately even-handed (0.25 each). Calibration via the
    calibrate_difficulty_weights script can downweight any feature that
    doesn't correlate with manual labels, and the per-feature
    saturation keeps any single noisy feature from dominating.

    Known limitations
    -----------------
    The verb count is bag-of-words: "color" matches the edit-verb list
    even when used as a noun ("the same color as..."). This is the kind
    of noise calibration is designed to absorb — if verb counts don't
    correlate with manual labels on your dataset, calibration will
    downweight that feature. A POS-tagged or LLM-based scorer would
    handle this cleanly; we keep the rule-based variant as the cheap
    transparent default and the LLM-based variant as a future swap-in.
    """

    name = "rule_based"

    def __init__(
        self,
        length_saturation: int = 25,
        verb_saturation: int = 3,
        conjunction_saturation: int = 2,
        spatial_saturation: int = 2,
        # Even weights by default; calibration may shift these.
        feature_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        if any(w < 0 for w in feature_weights):
            raise ValueError("feature_weights must be non-negative")
        if sum(feature_weights) <= 0:
            raise ValueError("feature_weights must sum to > 0")
        self.length_saturation = length_saturation
        self.verb_saturation = verb_saturation
        self.conjunction_saturation = conjunction_saturation
        self.spatial_saturation = spatial_saturation
        self.feature_weights = feature_weights

    def score(self, instruction: str) -> float:
        if not instruction:
            return 0.0

        tokens = [t.lower() for t in _TOKEN_RE.findall(instruction)]
        if not tokens:
            return 0.0

        # --- per-feature sub-scores in [0, 1] -----------------------------
        length_score = min(1.0, len(tokens) / self.length_saturation)

        n_verbs = sum(1 for t in tokens if t in _EDIT_VERBS)
        verb_score = min(1.0, n_verbs / self.verb_saturation)

        n_conj = sum(1 for t in tokens if t in _CONJUNCTION_TOKENS)
        conj_score = min(1.0, n_conj / self.conjunction_saturation)

        n_spatial = sum(1 for t in tokens if t in _SPATIAL_QUALIFIERS)
        spatial_score = min(1.0, n_spatial / self.spatial_saturation)

        wL, wV, wC, wS = self.feature_weights
        total_w = wL + wV + wC + wS
        score = (
            wL * length_score
            + wV * verb_score
            + wC * conj_score
            + wS * spatial_score
        ) / total_w

        return float(max(0.0, min(1.0, score)))

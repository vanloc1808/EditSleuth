"""V2 difficulty scoring: three-component formula with mask compactness.

Design rationale
----------------
The V1 four-component formula

    difficulty = w_S * (1 - SSIM)
               + w_P * combined_diff_mean
               + w_L * (1 - mask_area_frac)
               + w_I * instruction_complexity

was found to suffer from rank-2 redundancy on both MagicBrush and
Pico-Banana: ``structural_change``, ``perceptual_change``, and
``locality`` form a single magnitude axis with strong pairwise
correlations (|r| from 0.68 to 0.91). The three components partially
cancel under weighted summation when their underlying anticorrelation
is mediated by ``locality = 1 - mask_area_frac``, suppressing the
combined score's variance to roughly 1/4 of any single component's.

V2 collapses the magnitude trio to its single canonical
representative — ``structural_change`` — and adds ``compactness_score``
as a genuinely independent spatial-concentration signal. Mask
compactness measures how concentrated the edited region is in space
(a single tight blob has high compactness; a scattered or elongated
mask has low compactness), and is mathematically uncorrelated with
edit magnitude.

V2 formula
----------

    difficulty = w_S * structural_change
               + w_C * compactness_score
               + w_I * instruction_complexity

with default weights ``(0.55, 0.25, 0.20)`` reflecting:

* structural_change carries the dominant variance in the dataset
  (edit magnitude); it gets the heaviest weight.
* compactness_score is novel and hasn't been validated empirically
  yet — modest weight pending calibration.
* instruction_complexity has the smallest natural variance but is
  genuinely independent; modest weight to reflect that.

Stage B compactness requirement
-------------------------------
V2 requires that ``MaskArtifact`` carries a ``mask_compactness`` field.
Existing Stage B output is augmented retrospectively via
``scripts/add_mask_compactness.py``; future Stage B runs will compute
this natively. If V2 sees a mask artifact without ``mask_compactness``
(e.g., from an old Stage B run that hasn't been augmented), it raises.

V1 vs V2 coexistence
--------------------
V1 (``DifficultyScorer``) and V2 (this class) coexist intentionally.
V1 is the "naive baseline" referenced in the paper's ablation; V2 is
the contribution. Both can be run on the same Stage B output; they
produce parquets with distinct schemas (``DifficultyArtifact`` vs
``DifficultyArtifactV2``) so downstream tooling dispatches by file.

Empirical results
-----------------
V2 was evaluated on both datasets used in the paper, comparing against
V1 on identical mask sets:

* **MagicBrush dev** (n=528): V1 σ=0.034 → V2 σ=0.066 (+94%).
* **Pico-Banana** (n=257,725, after threshold calibration to
  ``global_mean_threshold=0.62`` for ~30% global routing rate):
  V1 σ=0.070 → V2 σ=0.109 (+55%).

V2 widens the score distribution on both datasets. The relative
improvement is larger on MagicBrush because V1's rank-2 collapse
was more severe there (component correlations 0.78–0.91 vs 0.68–0.85
on Pico-Banana). The absolute σ on MagicBrush remains lower than on
Pico-Banana because MagicBrush's intrinsic edit distribution is
narrower (small, subtle local edits), not because of a formula
limitation.

V2 is the recommended production scorer for both datasets.

Known limitation: residual structural ↔ compactness correlation
---------------------------------------------------------------
The diagnostic at ``scripts/diagnose_v2_local_subset.py`` revealed
that ``structural_change`` and ``compactness_score`` retain a moderate
negative Pearson correlation even on the local-edit subset (r=-0.44
on Pico-Banana locals, n=179,437). Mechanism: Stage B's
threshold+morphology pipeline produces *cleaner* binary masks for
high-magnitude edits (strong diff signal → coherent foreground) and
*noisier* masks for subtle edits (weak diff signal → scattered
small components surviving the filter). Compactness therefore tracks
mask quality, which itself tracks edit magnitude — partial coupling
back to ``structural_change``.

This is an artifact of computing the third component *downstream of
Stage B's diff signal*; any mask-derived feature inherits this
property to some degree. A future formula version (V3) would add a
component computed *directly from the (real, edited) image pair*
without going through the mask — e.g., a frozen-encoder image-pair
embedding distance — to break this coupling. Tracked as future work
rather than blocking V2 ship because V2 still produces a substantively
better score distribution than V1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from edit2forensics.data.difficulty_artifact_v2 import DifficultyArtifactV2
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.difficulty.instruction import (
    InstructionComplexityScorer,
    RuleBasedInstructionComplexity,
)
from edit2forensics.difficulty.structural import compute_structural_change

log = logging.getLogger(__name__)


# Default weights — see module docstring for rationale.
_DEFAULT_WEIGHTS_V2: dict[str, float] = {
    "structural_change": 0.55,
    "compactness_score": 0.25,
    "instruction_complexity": 0.20,
}

# Component names in canonical order for deterministic serialization.
_COMPONENT_NAMES_V2 = (
    "structural_change",
    "compactness_score",
    "instruction_complexity",
)


@dataclass
class DifficultyScorerV2Config:
    """Tuning knobs for V2 scoring."""

    weights: Mapping[str, float] = None  # type: ignore[assignment]
    """Per-component weights. Must contain all three V2 components.
    Defaults to ``(0.55, 0.25, 0.20)``."""

    ssim_image_size: int = 256
    """Side length to which images are bilinear-resized before SSIM.
    Identical default to V1 — the structural_change component is
    computationally identical between scorers."""

    ssim_data_range: float = 1.0

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = dict(_DEFAULT_WEIGHTS_V2)
        else:
            self.weights = dict(self.weights)
        missing = [n for n in _COMPONENT_NAMES_V2 if n not in self.weights]
        if missing:
            raise ValueError(
                f"weights missing required V2 components: {missing}; "
                f"need all of {list(_COMPONENT_NAMES_V2)}"
            )
        for n, v in self.weights.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"weight for {n!r} must be numeric, got {type(v)}")


class DifficultyScorerV2:
    """Compute one ``DifficultyArtifactV2`` per (triplet, mask_artifact) pair.

    Stateless — instantiate once per process. The output's
    ``difficulty_bin`` is a placeholder; the driver fills it in during
    a second pass over the full dataset.
    """

    def __init__(
        self,
        instruction_scorer: InstructionComplexityScorer | None = None,
        config: DifficultyScorerV2Config | None = None,
    ) -> None:
        self.instruction_scorer = instruction_scorer or RuleBasedInstructionComplexity()
        self.config = config or DifficultyScorerV2Config()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(
        self,
        triplet: EditTriplet,
        mask_artifact: MaskArtifact,
    ) -> DifficultyArtifactV2:
        """Return a `DifficultyArtifactV2` with raw components and aggregate.

        Requires that ``mask_artifact`` carries a ``mask_compactness``
        field. Raises ``ValueError`` if the field is missing or NaN —
        V2 cannot fall back to a default because compactness is the
        whole point of V2 over V1.
        """
        compactness_value = self._extract_compactness(mask_artifact)

        # --- structural change (SSIM) -------------------------------------
        structural_change = compute_structural_change(
            triplet,
            ssim_image_size=self.config.ssim_image_size,
            ssim_data_range=self.config.ssim_data_range,
        )

        # --- compactness_score = 1 - mask_compactness ---------------------
        # Higher compactness_score = more diffuse edit = higher difficulty.
        # The (1 - x) flip matches the convention of the other components
        # (higher = harder).
        compactness_score = float(np.clip(1.0 - compactness_value, 0.0, 1.0))

        # --- instruction complexity --------------------------------------
        instruction_complexity = float(
            np.clip(self.instruction_scorer.score(triplet.instruction), 0.0, 1.0)
        )

        components = {
            "structural_change": structural_change,
            "compactness_score": compactness_score,
            "instruction_complexity": instruction_complexity,
        }

        # --- aggregate ---------------------------------------------------
        weights = self.config.weights
        difficulty_raw = float(
            sum(weights[n] * components[n] for n in _COMPONENT_NAMES_V2)
        )

        return DifficultyArtifactV2(
            triplet_id=triplet.triplet_id,
            structural_change=structural_change,
            compactness_score=compactness_score,
            instruction_complexity=instruction_complexity,
            difficulty_raw=difficulty_raw,
            difficulty_bin="medium",  # placeholder; driver fills in.
            weights=dict(weights),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_compactness(mask_artifact: MaskArtifact) -> float:
        """Pull mask_compactness off the artifact, raising if absent.

        V2 is structurally dependent on this field; running V2 against
        an unaugmented Stage B parquet is a usage error worth surfacing
        loudly rather than silently falling back to V1-equivalent
        behavior.

        We accept compactness either as an attribute (proper
        MaskArtifact instances after Stage B persists it natively) or
        from the artifact's ``__dict__`` / a duck-typed access pattern
        compatible with the parquet-roundtripped form.
        """
        # Direct attribute access first (works once Stage B is updated
        # to include the field natively, AND for the retrospective
        # script's MaskArtifact roundtrip if from_dict is updated).
        c = getattr(mask_artifact, "mask_compactness", None)
        if c is None:
            raise ValueError(
                f"mask_artifact for {mask_artifact.triplet_id!r} has no "
                f"mask_compactness; run scripts/add_mask_compactness.py to "
                f"augment the Stage B parquet, or update Stage B to compute "
                f"this field natively."
            )
        c_float = float(c)
        if not np.isfinite(c_float):
            raise ValueError(
                f"mask_compactness for {mask_artifact.triplet_id!r} is "
                f"non-finite ({c}); this indicates a bug in the upstream "
                f"compactness computation."
            )
        return float(np.clip(c_float, 0.0, 1.0))

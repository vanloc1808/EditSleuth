"""V2 artifact schema for the revised three-component difficulty score.

The V1 schema (``DifficultyArtifact``) used four components — three of
which (structural_change, perceptual_change, locality) collapsed onto
a single magnitude axis with strong negative correlation, suppressing
the variance of the weighted sum. The V2 formula collapses that trio
into a single canonical magnitude term and adds mask compactness as a
genuinely independent spatial-concentration signal. See
``DifficultyScorerV2`` for the design rationale.

V1 and V2 coexist intentionally: V1 is the "naive baseline" against
which the paper's ablation reports the improvement. They are not
mutually exclusive — the same Stage B output supports both, since V2
reads ``mask_artifact.mask_compactness`` (added by the retrospective
``add_mask_compactness.py`` script or natively by future Stage B
runs) while V1 ignores it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

DifficultyBin = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class DifficultyArtifactV2:
    """Per-triplet output of ``DifficultyScorerV2``.

    Schema-distinct from V1's ``DifficultyArtifact`` so downstream
    tools can dispatch by parquet schema. The components share their
    canonical names (in particular ``structural_change`` and
    ``instruction_complexity`` are identical to V1) — only ``locality``
    and ``perceptual_change`` are dropped, replaced by
    ``compactness_score``.
    """

    triplet_id: str
    """Join key into the triplets and mask_artifacts tables."""

    # ---- raw component scores, each in [0, 1] -------------------------- #

    structural_change: float
    """``1 - SSIM(real, edited)``. The single canonical magnitude term
    in V2. Replaces the V1 ``(structural_change, perceptual_change,
    locality)`` trio, which the rank-2 correlation analysis showed
    were redundant. Identical computation to V1's structural_change."""

    compactness_score: float
    """``1 - mask_compactness``. Captures spatial concentration of the
    edit: a single tight blob has compactness ~1, so compactness_score
    ~0 (concentrated edits are easier to localize). A scattered or
    elongated mask has low compactness and high compactness_score
    (diffuse edits are harder). Genuinely independent of magnitude.
    Sourced from ``MaskArtifact.mask_compactness``."""

    instruction_complexity: float
    """Heuristic complexity of the natural-language edit instruction
    in ``[0, 1]``. Identical computation to V1. Independent of the
    image-derived components by construction."""

    # ---- aggregated score + bin --------------------------------------- #

    difficulty_raw: float
    """Weighted sum of the three components. In ``[0, 1]`` for default
    (non-negative, sum-to-one) weights."""

    difficulty_bin: DifficultyBin
    """Empirical tertile of the dataset's `difficulty_raw` distribution.
    Computed in a second pass over the full Stage C output."""

    # ---- audit ------------------------------------------------------- #

    weights: dict[str, float]
    """The weight vector that produced ``difficulty_raw``. Persisted
    per-row so post-hoc analysis on mixed runs from different
    calibrations works correctly."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DifficultyArtifactV2":
        # Tolerate a parquet roundtrip artifact: in some cases (mixed-
        # schema directories from multiple Stage C runs, or PyArrow
        # struct-vs-string downcast under specific column statistics)
        # the ``weights`` field comes back as a JSON-encoded string
        # instead of a dict. Parse it back into a dict; if parsing
        # fails or the value is something unexpected, fall back to
        # an empty dict and let the consumer decide.
        raw_weights = d.get("weights", {})
        if isinstance(raw_weights, str):
            try:
                weights = json.loads(raw_weights)
            except json.JSONDecodeError:
                weights = {}
        elif isinstance(raw_weights, dict):
            weights = raw_weights
        else:
            # Unknown type (e.g. None from a parquet null cell).
            weights = {}

        return cls(
            triplet_id=d["triplet_id"],
            structural_change=float(d["structural_change"]),
            compactness_score=float(d["compactness_score"]),
            instruction_complexity=float(d["instruction_complexity"]),
            difficulty_raw=float(d["difficulty_raw"]),
            difficulty_bin=d["difficulty_bin"],
            weights=dict(weights),
        )

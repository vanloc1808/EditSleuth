"""Artifact schema produced by Stage C (DifficultyScorer).

Per-triplet record carrying:

* The four raw difficulty components (SSIM-based, perceptual, locality,
  instruction complexity) preserved separately so that weight
  recalibration is a parquet-only operation — never requires re-reading
  images.
* The aggregated raw score and the empirical-tertile bin.

The component-preservation property matters. Re-fitting weights against
manual labels (the calibration script) reads ONLY this parquet plus a
small JSON of labels — no GPU, no image I/O. That keeps the experimental
loop tight when iterating on the scoring formula.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

DifficultyBin = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class DifficultyArtifact:
    """Per-triplet output of the DifficultyScorer."""

    triplet_id: str
    """Join key into the triplets and mask_artifacts tables."""

    # ---- raw component scores, each in [0, 1] -------------------------- #

    structural_change: float
    """``1 - SSIM(real, edited)``. Captures geometric/structural change.
    Robust to global luminance shifts that fool pixel-distance metrics."""

    perceptual_change: float
    """Mean of the (combiner-output, pre-threshold) diff map from Stage B.
    Captures the per-signal-aggregated perceptual change. Sourced from
    ``MaskArtifact.combined_diff_mean`` so this stage requires no GPU."""

    locality: float
    """``1 - mask_area_frac``. Smaller edits are harder to detect, so
    they get higher difficulty contribution. Sourced from
    ``MaskArtifact.mask_area_frac``."""

    instruction_complexity: float
    """Heuristic complexity of the natural-language edit instruction in
    ``[0, 1]``. Multi-step or spatially-qualified edits tend to be
    subtler at the pixel level."""

    # ---- aggregated score + bin --------------------------------------- #

    difficulty_raw: float
    """Weighted sum of the four components. In ``[0, 1]`` for default
    (non-negative, sum-to-one) weights; range may exceed ``[0, 1]`` if
    custom calibration produces non-convex weights."""

    difficulty_bin: DifficultyBin
    """Empirical tertile of the dataset's `difficulty_raw` distribution.
    Computed in a second pass over the full Stage C output, so a single
    triplet's bin depends on the rest of the dataset (intentional —
    "hard" means hard *relative to this dataset*)."""

    # ---- audit ------------------------------------------------------- #

    weights: dict[str, float]
    """The weight vector that produced ``difficulty_raw``. Persisted
    per-row (rather than once per-run) so post-hoc analysis on mixed
    runs from different calibrations works correctly."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DifficultyArtifact":
        # Tolerate parquet-roundtrip artifacts where ``weights`` comes
        # back as a JSON string instead of a dict (mixed-schema directory,
        # PyArrow downcast, etc.). Same pattern as DifficultyArtifactV2.
        raw_weights = d.get("weights", {})
        if isinstance(raw_weights, str):
            try:
                weights = json.loads(raw_weights)
            except json.JSONDecodeError:
                weights = {}
        elif isinstance(raw_weights, dict):
            weights = raw_weights
        else:
            weights = {}

        return cls(
            triplet_id=d["triplet_id"],
            structural_change=float(d["structural_change"]),
            perceptual_change=float(d["perceptual_change"]),
            locality=float(d["locality"]),
            instruction_complexity=float(d["instruction_complexity"]),
            difficulty_raw=float(d["difficulty_raw"]),
            difficulty_bin=d["difficulty_bin"],
            weights=dict(weights),
        )

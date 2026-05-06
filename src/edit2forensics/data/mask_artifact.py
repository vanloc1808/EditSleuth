"""Artifact schema produced by Stage B (MaskGenerator).

A `MaskArtifact` is the per-triplet record linking an EditTriplet to its
auto-generated manipulation mask plus quality metadata. Downstream stages
(DifficultyScorer, ReasoningAnnotator, training) consume this alongside
the EditTriplet parquet.

Separation of concerns
----------------------
The actual mask image lives on disk as a PNG at ``mask_path``. This
artifact record carries only scalar metadata — it's safe to load the
whole table into RAM even for millions of triplets, then lazily open
masks when needed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

EditScope = Literal["local", "global", "ambiguous", "alignment_failed"]


@dataclass(frozen=True)
class MaskArtifact:
    """Per-triplet output of the MaskGenerator."""

    triplet_id: str
    """Matches EditTriplet.triplet_id — the join key into the triplets table."""

    mask_path: Path
    """Absolute path to the auto-generated binary mask PNG (single-channel)."""

    mask_area_frac: float
    """Fraction of image pixels marked as edited. In [0, 1]."""

    edit_scope: EditScope
    """Categorical classification of the edit:

    - ``local``: a meaningful sub-region mask (0.5% < area < 90%).
    - ``global``: mask covers >=90% of the image; treat as whole-image edit.
    - ``ambiguous``: mask covers <0.5% (after refinement). Edit exists but
      is too small for the spatial mask to be reliable.
    - ``alignment_failed``: registration step could not align the pair;
      mask is best-effort or empty. Downstream code should skip or
      deprioritize these samples.
    """

    registration_ok: bool
    """True if size/layout alignment succeeded (or was unnecessary)."""

    confidence: float
    """Heuristic confidence in [0, 1] derived from diff-map contrast.
    Higher = clearer separation between edited and unedited regions."""

    diff_strongest_signal: str
    """Name of the DiffSignal that contributed most to the final mask
    (useful for Stage E reasoning annotations to reference the right
    evidence type)."""

    combined_diff_mean: float = 0.0
    """Mean value of the (post-combine, pre-threshold) diff map over the
    whole image, in ``[0, 1]``.

    Persisted so that Stage C (DifficultyScorer) can use it as the
    "perceptual change" component of the difficulty formula without
    needing to re-run any neural signals. Default ``0.0`` so older
    parquet files from pre-this-field Stage B runs continue to load —
    Stage C should treat such rows as if perceptual change is unknown.
    """

    mask_compactness: float = float("nan")
    """Geometric compactness of the binary mask in ``[0, 1]``: the
    geometric mean of (mask_area / bbox_area) and (largest_component
    / total_area). Captures spatial concentration of the edit
    independently of its magnitude.

    Default NaN means "not computed yet" — used by ``DifficultyScorerV2``
    to detect Stage B runs that pre-date the field. Older parquets
    that don't have this column will load with NaN here, and V2 will
    raise a clear error pointing the user at
    ``scripts/add_mask_compactness.py``.

    Stage B will eventually compute and persist this natively in the
    finalize phase; until then, the retrospective augmentation script
    populates it post-hoc.
    """

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mask_path"] = str(self.mask_path)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MaskArtifact":
        return cls(
            triplet_id=d["triplet_id"],
            mask_path=Path(d["mask_path"]),
            mask_area_frac=float(d["mask_area_frac"]),
            edit_scope=d["edit_scope"],
            registration_ok=bool(d["registration_ok"]),
            confidence=float(d["confidence"]),
            diff_strongest_signal=d["diff_strongest_signal"],
            # Tolerate older artifacts that lack these fields.
            combined_diff_mean=float(d.get("combined_diff_mean", 0.0)),
            mask_compactness=float(d.get("mask_compactness", float("nan"))),
        )

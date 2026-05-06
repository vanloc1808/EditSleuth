"""Artifact schema produced by Stage D (CategoryClassifier).

Per-triplet record mapping each EditTriplet to a canonical edit
category, plus enough metadata to audit the classification decision.

The category schema is deliberately coarse (~10 labels) — fine enough
to support per-category accuracy reporting in the paper's results
section, coarse enough that each bucket has thousands of samples on
datasets like Pico-Banana.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

# Canonical edit categories. The taxonomy is curated to:
# 1. Cover the principal axes of forensic distinction (additive vs.
#    subtractive vs. replacement; local vs. global; structural vs.
#    photometric).
# 2. Mappable from Pico-Banana's 35-category source taxonomy and from
#    MagicBrush's free-text instructions.
# 3. Include an explicit "other" bucket so the classifier never has
#    to make a forced choice on edge cases.
#
# Taxonomy revision (April 2026, after inspecting the Pico-Banana
# release manifest's 8-category grouping):
# * Added ``photometric`` for global tonal/filter effects (film grain,
#   color tone) — the Pico-Banana paper's "Pixel & Photometric"
#   category. Forensically distinct from artistic style transfer.
# * Added ``human_centric`` for person-specific edits (pose,
#   expression, age, clothing, accessories, plus the 9 person-
#   stylization variants like Funko-Pop / LEGO / Simpsonize). The
#   Pico-Banana paper groups these under one "Human-Centric"
#   category; we follow that lead because they share a forensic
#   signature (subject-localized, identity-preserving stylization
#   on a human figure).
EditCategory = Literal[
    "object_addition",       # add a new object/element
    "object_removal",        # remove an existing object
    "object_replacement",    # swap one object for another
    "attribute_change",      # change color/material/size/texture of a non-human object
    "style_transfer",        # whole-image artistic style change (Van Gogh, cartoon, anime)
    "photometric",           # global photometric overlay (film grain, vintage filter, color tone)
    "scene_transformation",  # whole-scene environmental change (lighting, time, weather, season, scene context)
    "background_change",     # background-only swap (foreground preserved)
    "text_edit",             # add/remove/modify in-image text
    "geometric",             # spatial transformation (crop, zoom, rotate, outpaint, relocate)
    "human_centric",         # person-specific edit (pose, clothing, accessories, person stylization)
    "other",                 # residual / cannot confidently classify
]

# All values of the EditCategory literal in canonical order. Used for
# deterministic iteration (e.g. building per-category statistics).
EDIT_CATEGORIES: tuple[EditCategory, ...] = (
    "object_addition",
    "object_removal",
    "object_replacement",
    "attribute_change",
    "style_transfer",
    "photometric",
    "scene_transformation",
    "background_change",
    "text_edit",
    "geometric",
    "human_centric",
    "other",
)


@dataclass(frozen=True)
class CategoryArtifact:
    """Per-triplet output of the CategoryClassifier."""

    triplet_id: str
    """Join key into triplets / mask_artifacts / difficulty tables."""

    category: EditCategory
    """Canonical category label."""

    confidence: float
    """Heuristic confidence in ``[0, 1]``. For label-driven
    classification (mapping a known source label to a canonical one)
    this is 1.0 by construction. For rule-based classification on
    free-text instructions, it reflects how strongly the rules fired
    (e.g., a single keyword match → 0.5; multiple corroborating
    signals → 0.9). Downstream code can filter low-confidence
    classifications if needed."""

    source: Literal["dataset_label", "rule_based", "fallback"]
    """How the category was determined:

    * ``dataset_label``: mapped from a per-triplet label provided by
      the source dataset (Pico-Banana ``source_edit_type``).
    * ``rule_based``: derived from the instruction text and/or mask
      geometry by the rule-based classifier.
    * ``fallback``: no rule fired with sufficient confidence; the
      ``other`` category was assigned. ``confidence`` will be low.
    """

    rationale: str
    """Short human-readable explanation of why this category was
    picked. For ``dataset_label``: the source label that was mapped.
    For ``rule_based``: the matched keyword or rule. For ``fallback``:
    a brief note. Length-bounded (~120 chars) so the parquet stays
    cheap to scan.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CategoryArtifact":
        # Raise a diagnostic error rather than the cryptic bare KeyError
        # if a required field is missing. This commonly indicates that
        # an upstream stage produced a parquet with a different schema
        # than expected — surfacing the actual keys present makes the
        # diagnosis fast.
        required = ("triplet_id", "category", "confidence", "source", "rationale")
        missing = [k for k in required if k not in d]
        if missing:
            tid = d.get("triplet_id", "<unknown>")
            raise ValueError(
                f"CategoryArtifact row for triplet_id={tid!r} is missing "
                f"required fields {missing}; available keys: {sorted(d.keys())}"
            )
        return cls(
            triplet_id=d["triplet_id"],
            category=d["category"],
            confidence=float(d["confidence"]),
            source=d["source"],
            rationale=d["rationale"],
        )

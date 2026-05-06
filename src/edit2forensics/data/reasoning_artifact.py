"""Stage E artifact: per-triplet reasoning chain.

The reasoning chain is the supervised target for downstream VLM
training: the model learns to generate a faithful, step-by-step
forensic explanation when shown a (real, edited, instruction) triplet.

Schema design notes
-------------------
* The ``chain`` field is the actual training target — numbered prose
  matching the chain-of-thought conventions in the VLM literature.
* The ``header`` is a one-line structured summary used by downstream
  tooling for filtering, not by the model.
* A handful of raw fields (``category``, ``difficulty_bin``,
  ``spatial_descriptor``) are mirrored from the chain's content so
  downstream filtering and per-category statistics don't require
  parsing the prose. Source of truth is still the chain text; these
  fields are conveniences.
* ``template_version`` lets us evolve templates without invalidating
  chains generated under earlier versions. Each chain records which
  template iteration produced it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from edit2forensics.data.category_artifact import EditCategory

DifficultyBin = Literal["easy", "medium", "hard"]
SpatialDescriptor = Literal[
    "whole_image",      # global scope (all-ones mask)
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "centered",
    "scattered",        # multiple disconnected regions
    "alignment_failed", # registration failed in Stage B; no spatial info
]


@dataclass(frozen=True)
class ReasoningArtifact:
    """Per-triplet reasoning chain output by Stage E."""

    triplet_id: str
    """Join key into all upstream tables."""

    header: str
    """One-line structured summary, not part of the model target.
    Format: ``[category=X, scope=Y, difficulty=Z, source=W]``."""

    chain: str
    """Numbered-prose reasoning chain. ~80-150 words, 6 steps. The
    supervised target for VLM training."""

    template_version: str
    """Identifier for the template version used to generate this chain.
    Format: ``v<major>.<minor>`` (e.g. ``v1.0``). Bump on template edits
    that materially change chain content; downstream training can filter
    by this if mixing chains from different versions becomes a concern.
    """

    # ---- mirrored fields for downstream filtering ------------------------ #

    category: EditCategory
    """Edit category from Stage D. Mirrored for filtering convenience."""

    difficulty_bin: DifficultyBin
    """Difficulty tertile from Stage C. Mirrored."""

    spatial_descriptor: SpatialDescriptor
    """Coarse spatial locale of the edit, computed from the mask in
    Stage E. Not present in any upstream artifact — Stage E is where
    the spatial-locale categorization happens."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasoningArtifact":
        return cls(
            triplet_id=d["triplet_id"],
            header=d["header"],
            chain=d["chain"],
            template_version=d["template_version"],
            category=d["category"],
            difficulty_bin=d["difficulty_bin"],
            spatial_descriptor=d["spatial_descriptor"],
        )

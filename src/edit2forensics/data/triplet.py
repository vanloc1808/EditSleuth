"""Canonical data schema for Edit2Forensics.

The `EditTriplet` is the single unit all downstream pipeline stages consume.
Every adapter (pico-banana, magicbrush, instructpix2pix, ultraedit) must
produce `EditTriplet` instances with identical semantics, regardless of the
source dataset's native format.

Design notes
------------
* Paths are stored as absolute `pathlib.Path` objects. Adapters are
  responsible for resolving any relative paths from the source dataset
  to absolute paths.
* `provided_mask_path` is `None` when the source dataset does not ship
  ground-truth masks. It is populated (e.g., by MagicBrush) when GT masks
  are available. Downstream code MUST check for `None` before using it.
* `metadata` is a free-form dict preserved from the source record. It is
  never consumed by pipeline logic — it exists for debugging, audit, and
  potential secondary signals (e.g., Pico-Banana's own `edit_type` label
  can seed the category classifier).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EditTriplet:
    """A single (real, edited, instruction) record in canonical form."""

    triplet_id: str
    """Globally unique, stable across runs. Format: `{source}_{zero_padded_index}`."""

    source_dataset: str
    """One of: 'pico-banana', 'magicbrush', 'instructpix2pix', 'ultraedit'."""

    real_path: Path
    """Absolute path to the original (unedited) image on local disk."""

    edited_path: Path
    """Absolute path to the edited image on local disk."""

    instruction: str
    """Natural-language edit instruction. The primary (unabridged) form."""

    provided_mask_path: Path | None = None
    """Ground-truth mask path when the source dataset provides one, else None."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Source-specific fields preserved verbatim for audit and debugging."""

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a dict suitable for parquet/jsonl serialization.

        `Path` objects are stringified; `None` is preserved.
        """
        d = asdict(self)
        d["real_path"] = str(self.real_path)
        d["edited_path"] = str(self.edited_path)
        d["provided_mask_path"] = (
            str(self.provided_mask_path) if self.provided_mask_path is not None else None
        )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditTriplet":
        """Inverse of `to_dict`. Accepts stringified paths.

        Also tolerates pandas/parquet roundtrip artifacts:
        * When a parquet column has any NaNs, pandas reads non-null
          string values fine but represents missing values as NaN
          (float). We treat any non-string ``provided_mask_path`` as
          "no mask".
        * The ``metadata`` field is a free-form dict that adapters
          serialize as a JSON string when writing the triplets parquet.
          On read-back, parquet typically gives us the string. We
          parse it here so all downstream consumers see a dict.
        """
        raw_mask = d.get("provided_mask_path")
        mask_path = Path(raw_mask) if isinstance(raw_mask, str) and raw_mask else None

        raw_metadata = d.get("metadata", {})
        if isinstance(raw_metadata, str):
            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError:
                metadata = {}
        elif isinstance(raw_metadata, dict):
            metadata = raw_metadata
        else:
            metadata = {}

        return cls(
            triplet_id=d["triplet_id"],
            source_dataset=d["source_dataset"],
            real_path=Path(d["real_path"]),
            edited_path=Path(d["edited_path"]),
            instruction=d["instruction"],
            provided_mask_path=mask_path,
            metadata=metadata,
        )

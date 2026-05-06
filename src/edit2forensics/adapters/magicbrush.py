"""Adapter for the MagicBrush dataset (HuggingFace `osunlp/MagicBrush`).

Source record shape (one row per turn)::

    {
      "img_id":      "000000391895",     # COCO image id as string
      "turn_index":  1,                   # 1-based; multi-turn editing
      "source_img":  <PIL.Image>,         # input to this turn's edit
      "mask_img":    <PIL.Image>,         # GT manipulation mask (white=edited)
      "instruction": "...",
      "target_img":  <PIL.Image>,         # output of this turn's edit
    }

Mapping to `EditTriplet`
------------------------
* ``triplet_id``: ``magicbrush_{split}_{img_id}_t{turn_index:02d}``.
  Encodes split + source image + turn. The split tag guarantees train
  and dev never collide even if someone later merges ingested parquets.
* ``real_path``: materialized ``source_img``.
* ``edited_path``: materialized ``target_img``.
* ``provided_mask_path``: materialized ``mask_img``. This is the
  distinguishing feature of MagicBrush — it is the GT-mask source of
  truth for validating our `MaskGenerator`.
* ``instruction``: raw ``instruction`` string.
* ``metadata``: preserves ``img_id``, ``turn_index``, ``split``, and a
  ``source_is_authentic`` flag that is True only for ``turn_index==1``
  (for turns >= 2 the "source" is itself a previously-edited image).

Multi-turn handling
-------------------
MagicBrush is a multi-turn editing dataset. For turns >= 2 the
``source_img`` is the *edited output from the previous turn*, not an
original photograph. We still emit these as EditTriplets because each
turn is a legitimate edit operation with a known mask. Downstream code
that requires a pristine real image (e.g., hardcore real/fake training
pairs) should filter on ``metadata["source_is_authentic"] == True``.
The adapter exposes a convenience flag ``single_turn_only`` that filters
at ingest time.

Mask encoding (IMPORTANT)
-------------------------
MagicBrush masks are NOT plain "white pixel = edited" binary images.
The ``mask_img`` field is an RGB (or occasionally RGBA) image where the
edited region is painted pure black ``[0, 0, 0]`` over the target
image's color content. A naive ``.convert("L")`` + ``>127`` threshold
flips the semantics and mislabels every naturally dark pixel as edited.

Extraction is handled by ``magicbrush_mask_to_binary`` in
``utils/image_io``; the on-disk masks this adapter produces are clean
single-channel PNGs with 255==edited, 0==unedited — the same convention
our own Stage B auto-masks use, so downstream validation IoU is
meaningful.

Caching
-------
Images are materialized to a per-split cache directory. The same
``triplet_id`` always maps to the same filenames, so reruns are
idempotent and cheap. HuggingFace `datasets` keeps its own compressed
parquet cache; this layer converts that into per-file PNGs for the
pipeline's disk-path contract.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from edit2forensics.adapters.base import BaseAdapter
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.utils.image_io import magicbrush_mask_to_binary, materialize_image

log = logging.getLogger(__name__)

_VALID_SPLITS = ("train", "dev")


class MagicBrushAdapter(BaseAdapter):
    """Adapter for the `osunlp/MagicBrush` HuggingFace dataset."""

    source_dataset = "magicbrush"

    def __init__(
        self,
        cache_root: Path | str,
        split: str = "train",
        hf_repo: str = "osunlp/MagicBrush",
        hf_cache_dir: Path | str | None = None,
        single_turn_only: bool = False,
        max_records: int | None = None,
        skip_if_exists: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        cache_root
            Directory where per-triplet PNG files are written. One
            subdirectory per split is created inside.
        split
            'train' or 'dev'. MagicBrush's test split is not on HF; use
            the official OSU-NLP zip separately if needed.
        hf_repo
            HuggingFace repository id. Overridable for forks / mirrors.
        hf_cache_dir
            Optional directory for the HuggingFace `datasets` download
            cache. If None, the library's default location is used.
        single_turn_only
            If True, only emit ``turn_index == 1`` records. Turns >= 2
            have previously-edited images as their source.
        max_records
            Optional cap on number of yielded triplets.
        skip_if_exists
            Passed to ``materialize_image``. Set False to force rewrite.
        """
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {_VALID_SPLITS}, got {split!r}"
            )
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.split = split
        self.hf_repo = hf_repo
        self.hf_cache_dir = (
            Path(hf_cache_dir).expanduser().resolve() if hf_cache_dir else None
        )
        self.single_turn_only = single_turn_only
        self.max_records = max_records
        self.skip_if_exists = skip_if_exists

        self.split_cache = self.cache_root / "magicbrush" / self.split
        self.split_cache.mkdir(parents=True, exist_ok=True)

        self.__post_init_check__()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest(self) -> Iterator[EditTriplet]:
        """Yield `EditTriplet`s, materializing images to disk as we go."""
        # Import here so the library-wide dependency is lazy: adapters
        # unrelated to HuggingFace can be used without `datasets` installed
        # (relevant for lightweight test environments).
        from datasets import load_dataset

        log.info(
            "loading %s[%s] (hf_cache_dir=%s)",
            self.hf_repo,
            self.split,
            self.hf_cache_dir,
        )
        ds = load_dataset(
            self.hf_repo,
            split=self.split,
            cache_dir=str(self.hf_cache_dir) if self.hf_cache_dir else None,
        )

        yielded = 0
        skipped = 0
        for idx, record in enumerate(ds):
            triplet = self._record_to_triplet(record, row_index=idx)
            if triplet is None:
                skipped += 1
                continue
            yielded += 1
            yield triplet
            if self.max_records is not None and yielded >= self.max_records:
                break

        log.info(
            "magicbrush[%s] finished: %d yielded, %d skipped",
            self.split,
            yielded,
            skipped,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _record_to_triplet(
        self, record: dict[str, Any], row_index: int
    ) -> EditTriplet | None:
        """Convert one HuggingFace row into an `EditTriplet`."""
        img_id = record.get("img_id")
        turn_index = record.get("turn_index")
        instruction = record.get("instruction")
        source_img = record.get("source_img")
        target_img = record.get("target_img")
        mask_img = record.get("mask_img")

        # --- validate required fields --------------------------------------
        if not img_id or turn_index is None or not instruction:
            log.warning(
                "magicbrush row %d: missing required field "
                "(img_id/turn_index/instruction)",
                row_index,
            )
            return None
        if source_img is None or target_img is None:
            log.warning(
                "magicbrush row %d (%s t%s): missing source or target image",
                row_index,
                img_id,
                turn_index,
            )
            return None

        # --- optional turn filter -----------------------------------------
        if self.single_turn_only and int(turn_index) != 1:
            return None

        # --- build stable identifier --------------------------------------
        triplet_id = (
            f"magicbrush_{self.split}_{img_id}_t{int(turn_index):02d}"
        )

        # --- materialize images to disk -----------------------------------
        real_path = materialize_image(
            source_img,
            self.split_cache / f"{triplet_id}__real.png",
            skip_if_exists=self.skip_if_exists,
        )
        edited_path = materialize_image(
            target_img,
            self.split_cache / f"{triplet_id}__edited.png",
            skip_if_exists=self.skip_if_exists,
        )

        # Mask is typically present but defensively tolerate its absence.
        # MagicBrush masks are NOT plain grayscale: the edited region is
        # encoded as pure-black RGB (or alpha=0 for RGBA variants) on top
        # of the target image's color content. Naive ``.convert("L")``
        # inverts the semantics and mislabels naturally dark pixels.
        # `magicbrush_mask_to_binary` handles all known encodings and
        # returns a clean binary L-mode image where 255==edited.
        mask_path: Path | None = None
        if mask_img is not None:
            mask_binary = magicbrush_mask_to_binary(mask_img)
            mask_path = materialize_image(
                mask_binary,
                self.split_cache / f"{triplet_id}__mask.png",
                skip_if_exists=self.skip_if_exists,
            )

        # --- metadata -----------------------------------------------------
        metadata = {
            "img_id": img_id,
            "turn_index": int(turn_index),
            "split": self.split,
            # Only turn 1's source_img is a pristine real photograph.
            # Downstream stages that need "never-edited" reals should
            # filter on this flag.
            "source_is_authentic": int(turn_index) == 1,
            "row_index": row_index,
        }

        return EditTriplet(
            triplet_id=triplet_id,
            source_dataset=self.source_dataset,
            real_path=real_path,
            edited_path=edited_path,
            instruction=instruction,
            provided_mask_path=mask_path,
            metadata=metadata,
        )

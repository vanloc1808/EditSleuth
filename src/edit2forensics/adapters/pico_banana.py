"""Adapter for the Pico-Banana-400K editing dataset.

Source record shape (from the dataset's JSONL manifest)::

    {
      "open_image_input_url": "https://...jpg",
      "text": "Remove the red flag ...",
      "output_image": "images/positive-edit/1.png",
      "edit_type": "Remove an existing object",
      "summarized_text": "Flag removed; ...",
      "local_input_image": "openimages/images/train_0/0000e2205e460318.jpg"
    }

Mapping to `EditTriplet`
------------------------
* ``triplet_id``: ``picobanana_{stem}`` where ``stem`` is the sanitized
  filename stem from ``output_image`` (e.g.
  ``images/positive-edit/1.png`` -> ``picobanana_1``,
  ``images/positive-edit/kewsee_retry1.png`` -> ``picobanana_kewsee_retry1``).
  Characters outside ``[A-Za-z0-9_-]`` are replaced with ``_``.

  An earlier version of this adapter parsed a numeric index from the
  filename, but that produced ~22% duplicate IDs on the full Pico-Banana
  release because not all filenames are numeric (``kewsee_retry1.png``,
  etc.) and the regex captured only the trailing integer, colliding
  with files like ``1.png``. The current behavior preserves the full
  stem and is collision-free as long as the source dataset uses
  unique output filenames.
* ``real_path``: resolved from ``local_input_image`` under ``dataset_root``.
  If that file is missing and ``download_missing=True``, we fall back to
  downloading ``open_image_input_url`` into a cache directory.
* ``edited_path``: resolved from ``output_image`` under ``dataset_root``.
  If missing, the record is skipped (the edited image is mandatory).
* ``instruction``: the long-form ``text``. The short ``summarized_text`` is
  preserved in metadata and can be used by downstream stages for
  lightweight analysis.
* ``provided_mask_path``: always ``None``. Pico-Banana ships no masks;
  `MaskGenerator` will synthesize them in Stage B.
* ``metadata``: preserves ``edit_type``, ``summarized_text``,
  ``open_image_input_url``, the original manifest line number, and the
  raw record's ``output_image`` path for audit.

Note on `edit_type`
-------------------
Pico-Banana's own ``edit_type`` (e.g. "Remove an existing object") is NOT
the same as our Level-3 taxonomy. We stash it in metadata so the
`CategoryClassifier` can use it as a prior signal. Treating it as
authoritative would make our taxonomy dependent on an upstream schema
we don't control.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

from edit2forensics.adapters.base import BaseAdapter
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.utils.download import download_image

log = logging.getLogger(__name__)

# Characters that are unsafe in a filesystem path (and therefore unsafe
# in a triplet_id, since IDs flow into output filenames like
# `masks/{triplet_id}.png`). Anything outside this set is replaced with
# an underscore by `_sanitize_id_stem`.
_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]+")


class PicoBananaAdapter(BaseAdapter):
    """Adapter for Pico-Banana-400K JSONL-style manifests."""

    source_dataset = "pico-banana"

    def __init__(
        self,
        manifest_path: Path | str,
        dataset_root: Path | str,
        download_missing: bool = False,
        download_cache: Path | str | None = None,
        max_records: int | None = None,
        on_output_collision: str = "drop_all",
    ) -> None:
        """
        Parameters
        ----------
        manifest_path
            Path to the JSONL manifest file. Each line is one record.
        dataset_root
            Root directory under which ``local_input_image`` and
            ``output_image`` paths resolve. For a typical Pico-Banana
            download this is the top-level extracted folder.
        download_missing
            If True, when ``local_input_image`` is absent on disk we
            attempt to download ``open_image_input_url`` into
            ``download_cache``. If False (default), such records are
            skipped with a warning — safer for reproducibility.
        download_cache
            Directory for downloaded real images. Required if
            ``download_missing=True``.
        max_records
            Optional cap for smoke-testing / dev runs.
        on_output_collision
            Policy for records that share an ``output_image`` path with
            another record in the manifest. Pico-Banana's manifest
            occasionally contains such collisions, and on inspection
            the colliding records have been corrupted (mismatched
            instructions, output images that don't reflect the input).
            Two records cannot both correspond to a single edited file
            on disk, so we treat ``output_image`` collisions as a data-
            quality signal:

            - ``"drop_all"`` (default): drop every record involved in a
              collision. Conservative; matches the empirical observation
              that colliding records in Pico-Banana are corrupted.
            - ``"keep_first"``: keep the first manifest occurrence,
              skip subsequent ones. Use only if you have evidence the
              first occurrence is reliable.

            Detection of mismatched-instruction-vs-image is a separate,
            VLM-scale problem and is intentionally NOT done by this
            adapter — adapters are I/O-only by design.
        """
        if on_output_collision not in ("drop_all", "keep_first"):
            raise ValueError(
                f"on_output_collision must be 'drop_all' or 'keep_first', "
                f"got {on_output_collision!r}"
            )

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.download_missing = download_missing
        self.download_cache = (
            Path(download_cache).expanduser().resolve() if download_cache else None
        )
        self.max_records = max_records
        self.on_output_collision = on_output_collision

        if self.download_missing and self.download_cache is None:
            raise ValueError(
                "download_missing=True requires download_cache to be set"
            )
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root not found: {self.dataset_root}")

        self.__post_init_check__()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest(self) -> Iterator[EditTriplet]:
        """Yield validated `EditTriplet` instances from the manifest.

        We do a cheap first pass over the manifest to discover
        ``output_image`` collisions (records that would otherwise produce
        the same triplet_id), then a normal second pass that yields
        records, skipping the colliding ones according to
        ``self.on_output_collision``.

        The first pass parses JSON and computes the id stem from
        ``output_image`` only — no path resolution, no image I/O —
        so it's cheap relative to the second pass.
        """
        ids_to_drop = self._compute_collision_drop_set()

        yielded = 0
        skipped_collision = 0
        skipped_other = 0

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as e:
                    log.warning(
                        "pico-banana: malformed JSON at line %d: %s", line_no, e
                    )
                    skipped_other += 1
                    continue

                # Drop pre-flagged collisions WITHOUT doing any further
                # work (path resolution, file existence checks). We
                # don't want to penalize the run for missing files in
                # records we're about to drop anyway.
                output_image = record.get("output_image", "")
                stem = self._derive_id_stem(output_image)
                key: tuple[str, int] | str
                if self.on_output_collision == "drop_all":
                    key = stem if stem is not None else ""
                    is_dropped = key in ids_to_drop
                else:
                    # keep_first: drop only after the first occurrence
                    key = (stem if stem is not None else "", line_no)
                    is_dropped = key in ids_to_drop
                if is_dropped:
                    skipped_collision += 1
                    continue

                triplet = self._record_to_triplet(record, line_no)
                if triplet is None:
                    skipped_other += 1
                    continue

                yielded += 1
                yield triplet

                if self.max_records is not None and yielded >= self.max_records:
                    break

        log.info(
            "pico-banana adapter finished: %d yielded, %d dropped (collision), "
            "%d skipped (other reasons)",
            yielded, skipped_collision, skipped_other,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _compute_collision_drop_set(self) -> set:
        """First-pass scan: identify records to drop due to id collisions.

        Returns a set whose membership semantics depend on
        ``self.on_output_collision``:

        - ``"drop_all"``: set of id stems that appear 2+ times.
          Membership-by-stem drops every record sharing that stem.
        - ``"keep_first"``: set of ``(stem, line_no)`` pairs identifying
          the SECOND-and-later occurrences. The first occurrence is
          not in the set and will be yielded normally.

        We also log the colliding lines so the operator can investigate
        the source manifest.
        """
        # First sub-pass: count occurrences per stem and remember line
        # numbers of each occurrence.
        stem_to_lines: dict[str, list[int]] = {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    # Malformed lines are handled in the second pass;
                    # don't double-warn here.
                    continue
                output_image = record.get("output_image")
                if not output_image:
                    continue
                stem = self._derive_id_stem(output_image)
                if stem is None:
                    continue
                stem_to_lines.setdefault(stem, []).append(line_no)

        colliding = {s: lines for s, lines in stem_to_lines.items() if len(lines) > 1}
        if not colliding:
            return set()

        # Surface the collisions clearly so the user can audit.
        n_collisions = len(colliding)
        n_records = sum(len(v) for v in colliding.values())
        log.warning(
            "pico-banana: detected %d output_image collisions involving %d "
            "manifest records. Policy: %s.",
            n_collisions, n_records, self.on_output_collision,
        )
        # Log a few examples for debuggability without flooding the log.
        for stem, lines in list(colliding.items())[:5]:
            log.warning(
                "  collision on id stem %r at manifest lines %s",
                stem, lines,
            )
        if n_collisions > 5:
            log.warning("  ... and %d more (suppressed)", n_collisions - 5)

        if self.on_output_collision == "drop_all":
            # Drop every record at any colliding stem.
            return set(colliding.keys())
        else:
            # keep_first: drop only the SECOND and later occurrences.
            drop = set()
            for stem, lines in colliding.items():
                for ln in lines[1:]:
                    drop.add((stem, ln))
            return drop

    def _record_to_triplet(
        self, record: dict, line_no: int
    ) -> EditTriplet | None:
        """Convert one manifest record into an `EditTriplet`, or None if invalid."""
        # --- required fields ------------------------------------------------
        output_image = record.get("output_image")
        instruction = record.get("text")
        if not output_image or not instruction:
            log.warning(
                "pico-banana L%d: missing required field (output_image/text)",
                line_no,
            )
            return None

        # --- triplet_id from output_image filename stem ---------------------
        # Earlier versions of this adapter used a numeric-only ID derived
        # from the filename digits. That collided badly: filenames like
        # `kewsee_retry1.png` mapped to the same ID as `1.png`, producing
        # ~22% duplicate IDs on Pico-Banana. We now use the full
        # filename stem (sanitized to filesystem-safe characters).
        stem = self._derive_id_stem(output_image)
        if stem is None:
            log.warning(
                "pico-banana L%d: cannot derive id stem from output_image=%r",
                line_no,
                output_image,
            )
            return None
        triplet_id = f"picobanana_{stem}"

        # --- edited image must exist ---------------------------------------
        edited_path = (self.dataset_root / output_image).resolve()
        if not edited_path.exists():
            log.warning(
                "pico-banana L%d (%s): edited image missing at %s",
                line_no,
                triplet_id,
                edited_path,
            )
            return None

        # --- real image: local first, download fallback ---------------------
        real_path = self._resolve_real_image(record, triplet_id, line_no)
        if real_path is None:
            return None

        # --- metadata (preserve everything for audit) -----------------------
        metadata = {
            "source_edit_type": record.get("edit_type"),
            "summarized_text": record.get("summarized_text"),
            "open_image_input_url": record.get("open_image_input_url"),
            "manifest_line_no": line_no,
            "source_output_image": output_image,
            "source_local_input_image": record.get("local_input_image"),
        }

        return EditTriplet(
            triplet_id=triplet_id,
            source_dataset=self.source_dataset,
            real_path=real_path,
            edited_path=edited_path,
            instruction=instruction,
            provided_mask_path=None,  # Pico-Banana ships no masks
            metadata=metadata,
        )

    def _resolve_real_image(
        self, record: dict, triplet_id: str, line_no: int
    ) -> Path | None:
        """Return an absolute path to the real image, or None if unavailable."""
        local_rel = record.get("local_input_image")
        if local_rel:
            local_abs = (self.dataset_root / local_rel).resolve()
            if local_abs.exists():
                return local_abs

        url = record.get("open_image_input_url")
        if self.download_missing and url and self.download_cache is not None:
            downloaded = download_image(url, self.download_cache)
            if downloaded is not None:
                return downloaded
            log.warning(
                "pico-banana L%d (%s): download failed for %s",
                line_no,
                triplet_id,
                url,
            )
        else:
            log.warning(
                "pico-banana L%d (%s): real image missing and download_missing=False",
                line_no,
                triplet_id,
            )
        return None

    @staticmethod
    def _derive_id_stem(output_image: str) -> str | None:
        """Build a stable, filesystem-safe id stem from an output_image path.

        Examples
        --------
        >>> PicoBananaAdapter._derive_id_stem("images/positive-edit/1.png")
        '1'
        >>> PicoBananaAdapter._derive_id_stem("images/positive-edit/00042.png")
        '00042'
        >>> PicoBananaAdapter._derive_id_stem("images/positive-edit/kewsee_retry1.png")
        'kewsee_retry1'
        >>> PicoBananaAdapter._derive_id_stem("a/b/foo bar!.jpg")
        'foo_bar_'

        Returns ``None`` only for empty/whitespace inputs (we DON'T require
        the stem to be parseable as an integer — that was the old behavior
        and it produced collisions).
        """
        if not output_image or not output_image.strip():
            return None
        # Take only the filename portion, then drop the extension.
        stem = Path(output_image).stem
        if not stem:
            return None
        # Replace any character outside [A-Za-z0-9_-] with underscore.
        # Collapsing runs of unsafe chars to a single underscore keeps
        # the stem readable (`foo bar!.jpg` -> `foo_bar_`, not
        # `foo_bar__`).
        sanitized = _ID_SAFE_RE.sub("_", stem)
        # Defensive: if everything got sanitized away (extremely unlikely
        # given Pico-Banana's filenames), surface that as a parse failure.
        return sanitized if sanitized else None

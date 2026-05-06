"""Stage E entry point: generate per-triplet reasoning chains.

Consumes four parquet directories — triplets (Stage A), mask artifacts
(Stage B), difficulty artifacts V2 (Stage C), and category artifacts
(Stage D) — joins on ``triplet_id``, and emits one
``ReasoningArtifact`` per triplet.

Why a single-pass driver
------------------------
Unlike Stage C (which had a Pass-1 / Pass-2 split for tertile binning)
or Stage B (which had heavy GPU work), Stage E's per-triplet work is
template-string composition plus a small ``scipy.ndimage`` call for
the spatial descriptor. CPU-bound, no global statistics needed, no
GPU. Single pass over shards is sufficient.

Resumability and parallelism
----------------------------
Same patterns as the other drivers:

* Output shards mirror input triplet shard names — workers writing
  different input shards never collide.
* ``shard_indices`` restricts processing to a chosen subset.
* Existing output shards are skipped unless
  ``overwrite_existing_shards=true``.

Usage::

    uv run python scripts/annotate_reasoning.py \\
        triplets_path=artifacts/triplets/pico_banana.parquet \\
        mask_artifacts_path=artifacts/mask_artifacts/pico_banana_lab_lpips_ssim_calibrated_with_compactness.parquet \\
        difficulty_path=artifacts/difficulty/pico_banana_v2.parquet \\
        category_path=artifacts/categories/pico_banana.parquet \\
        run_tag=pico_banana \\
        paths.output_root=artifacts
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from edit2forensics.data.category_artifact import (
    CategoryArtifact,
    EDIT_CATEGORIES,
)
from edit2forensics.data.difficulty_artifact_v2 import DifficultyArtifactV2
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet

log = logging.getLogger(__name__)


def _load_parquet_dataset_dedup(path: Path, label: str) -> pd.DataFrame:
    """Load a parquet directory, dedup on triplet_id (keeping last
    occurrence)."""
    shards = sorted(path.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {path}")
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates(subset="triplet_id", keep="last")
    if len(df) < n_raw:
        log.warning(
            "%s: dropped %d duplicate triplet_ids (%d -> %d unique)",
            label, n_raw - len(df), n_raw, len(df),
        )
    return df


def _select_shards(triplets_path: Path, shard_indices) -> list[Path]:
    """Pick which input triplet shards to process."""
    all_shards = sorted(triplets_path.glob("part-*.parquet"))
    if not all_shards:
        raise FileNotFoundError(
            f"no parquet shards under {triplets_path}; did you run ingest.py?"
        )
    if shard_indices is None:
        return all_shards
    indices = list(shard_indices)
    n = len(all_shards)
    bad = [i for i in indices if i < 0 or i >= n]
    if bad:
        raise IndexError(
            f"shard_indices={indices} contains out-of-range entries {bad} "
            f"(only {n} shards available, indices 0..{n - 1})"
        )
    return [all_shards[i] for i in indices]


@hydra.main(version_base=None, config_path="../configs", config_name="annotate_reasoning")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    annotator = instantiate(cfg.reasoning_annotator)

    triplets_path = Path(cfg.triplets_path)
    mask_path = Path(cfg.mask_artifacts_path)
    difficulty_path = Path(cfg.difficulty_path)
    category_path = Path(cfg.category_path)
    out_dir = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load lookups from the three non-triplet stages ---------------
    log.info("loading mask, difficulty, and category artifacts")
    mask_df = _load_parquet_dataset_dedup(mask_path, "mask_artifacts")
    diff_df = _load_parquet_dataset_dedup(difficulty_path, "difficulty")
    cat_df = _load_parquet_dataset_dedup(category_path, "category")

    log.info(
        "joined cardinalities: mask=%d, difficulty=%d, category=%d",
        len(mask_df), len(diff_df), len(cat_df),
    )

    # Index by triplet_id for O(1) lookup during the per-row loop.
    mask_by_id = {row["triplet_id"]: row for _, row in mask_df.iterrows()}
    diff_by_id = {row["triplet_id"]: row for _, row in diff_df.iterrows()}
    cat_by_id = {row["triplet_id"]: row for _, row in cat_df.iterrows()}

    # ---- iterate triplet shards ---------------------------------------
    shard_indices = cfg.get("shard_indices")
    if shard_indices is not None:
        shard_indices = [int(i) for i in shard_indices]
    selected = _select_shards(triplets_path, shard_indices)
    log.info("processing %d triplet shards", len(selected))

    overwrite = bool(cfg.get("overwrite_existing_shards", False))
    total = 0
    skipped_done = 0
    skipped_missing = 0
    n_failed = 0
    category_counts: Counter = Counter()
    spatial_counts: Counter = Counter()

    for shard in selected:
        out_path = out_dir / shard.name
        if out_path.exists() and not overwrite:
            log.info("output shard %s already exists; skipping", shard.name)
            skipped_done += 1
            continue

        triplets_df = pd.read_parquet(shard)
        rows: list[dict] = []
        for _, t_row in tqdm(
            triplets_df.iterrows(),
            total=len(triplets_df),
            desc=f"reason[{shard.name}]",
            unit="triplet",
        ):
            tid = t_row["triplet_id"]
            mask_row = mask_by_id.get(tid)
            diff_row = diff_by_id.get(tid)
            cat_row = cat_by_id.get(tid)
            if mask_row is None or diff_row is None or cat_row is None:
                # One of the upstream stages didn't process this triplet.
                # Skip rather than crash — Stage E is downstream of three
                # other stages; it should be tolerant of upstream gaps.
                skipped_missing += 1
                continue

            try:
                triplet_dict = t_row.to_dict()
                if isinstance(triplet_dict.get("metadata"), str):
                    triplet_dict["metadata"] = json.loads(triplet_dict["metadata"])
                triplet = EditTriplet.from_dict(triplet_dict)
                mask = MaskArtifact.from_dict(mask_row.to_dict())
                difficulty = DifficultyArtifactV2.from_dict(diff_row.to_dict())
                category = CategoryArtifact.from_dict(cat_row.to_dict())
                artifact = annotator.annotate(triplet, mask, difficulty, category)
            except Exception as e:
                log.warning("annotate failed for %s: %s", tid, e)
                n_failed += 1
                continue

            rows.append(artifact.to_dict())
            category_counts[artifact.category] += 1
            spatial_counts[artifact.spatial_descriptor] += 1

        if rows:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            log.info("wrote %d reasoning artifacts to %s", len(rows), shard.name)
        total += len(rows)

    # ---- summary -------------------------------------------------------
    summary = {
        "n_total": int(total),
        "n_failed": int(n_failed),
        "n_skipped_due_to_done_output": int(skipped_done),
        "n_skipped_due_to_missing_upstream": int(skipped_missing),
        "category_counts": {
            c: int(category_counts.get(c, 0)) for c in EDIT_CATEGORIES
        },
        "spatial_counts": {k: int(v) for k, v in spatial_counts.items()},
    }
    report_path = Path(cfg.summary_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))
    log.info("\n===== reasoning summary =====\n%s", json.dumps(summary, indent=2))
    log.info("wrote summary to %s", report_path)


if __name__ == "__main__":
    main()

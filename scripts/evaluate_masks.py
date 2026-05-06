"""Validate auto-generated masks against ground-truth masks.

For sources that ship GT masks (MagicBrush), this script joins the
triplets table against the mask_artifacts table, compares each
auto-mask with its ``provided_mask_path``, and prints/persists an
IoU distribution summary.

Usage::

    uv run python scripts/evaluate_masks.py \\
        triplets_path=/path/to/artifacts/triplets/magicbrush_dev.parquet \\
        mask_artifacts_path=/path/to/artifacts/mask_artifacts/magicbrush_dev.parquet \\
        output_report=/path/to/artifacts/reports/mask_validation_magicbrush_dev.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from edit2forensics.mask.validation import compare_masks, summarize_comparisons

log = logging.getLogger(__name__)


def _load_parquet_dataset(path: Path) -> pd.DataFrame:
    shards = sorted(path.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {path}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_masks")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    triplets = _load_parquet_dataset(Path(cfg.triplets_path))
    artifacts = _load_parquet_dataset(Path(cfg.mask_artifacts_path))

    # Restrict to triplets that have a GT mask path (i.e. MagicBrush).
    gt_available = triplets[triplets["provided_mask_path"].notna()].copy()
    log.info(
        "triplets: %d total, %d with GT masks", len(triplets), len(gt_available)
    )
    if len(gt_available) == 0:
        log.error(
            "no triplets in this dataset have GT masks — validation is only "
            "meaningful on sources like MagicBrush that ship provided_mask_path"
        )
        return

    # Join on triplet_id. The triplets table has `provided_mask_path`
    # (the GT); the artifacts table has `mask_path` (our auto mask).
    # Different column names so no suffixes are needed.
    merged = gt_available.merge(
        artifacts[["triplet_id", "mask_path", "edit_scope"]],
        on="triplet_id",
        how="inner",
    )
    log.info("joined rows: %d", len(merged))

    # --- per-sample comparisons -------------------------------------------
    cmps = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="compare"):
        cmp = compare_masks(
            auto_path=row["mask_path"],
            gt_path=row["provided_mask_path"],
            triplet_id=row["triplet_id"],
        )
        cmps.append(cmp)

    summary = summarize_comparisons(cmps)

    # Stratify by edit_scope too — one common failure mode is that global
    # and ambiguous edits drag down the overall IoU. Per-scope breakdown
    # is useful for the paper.
    by_scope: dict[str, dict] = {}
    scope_series = merged["edit_scope"]
    for scope in sorted(scope_series.unique()):
        idxs = [i for i, s in enumerate(scope_series) if s == scope]
        by_scope[scope] = summarize_comparisons([cmps[i] for i in idxs])
    summary["by_scope"] = by_scope

    # --- persist + display -------------------------------------------------
    out_path = Path(cfg.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    # Persist per-triplet IoU/Dice records too. The JSON summary contains
    # only aggregate statistics (mean, std, histogram, by-scope means), so
    # downstream analyses that need per-sample data — paired bootstrap
    # comparisons across runs, error-mode investigation, calibration —
    # would otherwise have to recompute from scratch. The per-triplet
    # parquet is small (~50 bytes per row) and removes that friction.
    per_triplet_path = out_path.with_suffix("").with_suffix(".per_triplet.parquet")
    pd.DataFrame([
        {
            "triplet_id": c.triplet_id,
            "iou": c.iou,
            "dice": c.dice,
            "gt_area_frac": c.gt_area_frac,
            "auto_area_frac": c.auto_area_frac,
            "edit_scope": scope_series.iloc[i],
        }
        for i, c in enumerate(cmps)
    ]).to_parquet(per_triplet_path, index=False)
    log.info("wrote per-triplet records to %s", per_triplet_path)

    log.info("\n===== mask validation summary =====\n%s", json.dumps(summary, indent=2))
    log.info("wrote full report to %s", out_path)


if __name__ == "__main__":
    main()

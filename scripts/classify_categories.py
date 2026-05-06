"""Stage D entry point: classify each triplet's edit category.

Consumes a parquet directory produced by ``scripts/ingest.py`` and
emits a parquet of `CategoryArtifact`s. One artifact per triplet.

Two-path classification
-----------------------
For each triplet, the classifier either:

1. Maps a per-row source label (e.g., Pico-Banana
   ``source_edit_type``) to a canonical category, or
2. Applies rule-based classification on the instruction text.

Both paths produce uniform-shape `CategoryArtifact` records; downstream
tooling treats them identically. The ``source`` field on each artifact
indicates which path was taken, for audit.

Resumability and parallelism
----------------------------
Same patterns as Stage B and Stage C drivers:

* Output shards mirror input shard names. An existing output shard is
  skipped unless ``overwrite_existing_shards=true``.
* ``shard_indices`` restricts processing to a chosen subset of input
  shards for parallel-worker invocation.

Stage D is CPU-cheap (rule matching + dict lookup) so parallelism
isn't usually needed; the knobs exist for symmetry with the other
stages.

Usage::

    uv run python scripts/classify_categories.py \\
        triplets_path=/path/to/artifacts/triplets/pico_banana.parquet \\
        paths.output_root=/path/to/artifacts \\
        run_tag=pico_banana
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from edit2forensics.category.classifier import category_distribution
from edit2forensics.data.triplet import EditTriplet

log = logging.getLogger(__name__)


def _flush_batch(rows: list[dict], out_dir: Path, shard_name: str) -> None:
    """Write a batch of category artifact dicts to a parquet shard.

    Output filename mirrors the input shard name — same convention as
    Stage B and Stage C drivers, ensuring parallel workers writing
    different input shards never collide.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path = out_dir / shard_name
    df.to_parquet(out_path, index=False)
    log.info("wrote %d category artifacts to %s", len(df), shard_name)


def _select_shards(triplets_path: Path, shard_indices) -> list[Path]:
    """Resolve which shards to process. Same pattern as generate_masks."""
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


def _iter_shard_triplets(shard: Path):
    """Yield EditTriplet instances from one shard."""
    df = pd.read_parquet(shard)
    for _, row in df.iterrows():
        d = row.to_dict()
        if isinstance(d.get("metadata"), str):
            d["metadata"] = json.loads(d["metadata"])
        yield EditTriplet.from_dict(d)


@hydra.main(version_base=None, config_path="../configs", config_name="classify_categories")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    classifier = instantiate(cfg.category_classifier)

    triplets_path = Path(cfg.triplets_path)
    out_dir = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_indices = cfg.get("shard_indices")
    if shard_indices is not None:
        shard_indices = [int(i) for i in shard_indices]
    selected = _select_shards(triplets_path, shard_indices)
    log.info(
        "processing %d/%d input shards%s",
        len(selected),
        len(sorted(triplets_path.glob("part-*.parquet"))),
        f" (indices={shard_indices})" if shard_indices is not None else "",
    )

    overwrite = bool(cfg.get("overwrite_existing_shards", False))
    total = 0
    skipped_done = 0
    all_artifacts = []  # accumulated across shards for final summary

    for shard in selected:
        out_path = out_dir / shard.name
        if out_path.exists() and not overwrite:
            log.info("output shard %s already exists; skipping", shard.name)
            # Still load the existing artifacts to include in the summary,
            # so the per-category distribution is comprehensive.
            existing = pd.read_parquet(out_path)
            for _, row in existing.iterrows():
                all_artifacts.append({
                    "category": row["category"],
                    "source": row["source"],
                })
            skipped_done += 1
            continue

        batch: list[dict] = []
        for triplet in tqdm(
            _iter_shard_triplets(shard),
            desc=f"category[{shard.name}]",
            unit="triplet",
        ):
            try:
                artifact = classifier.classify(triplet)
            except Exception as e:
                log.warning("classification failed for %s: %s", triplet.triplet_id, e)
                continue
            d = artifact.to_dict()
            batch.append(d)
            all_artifacts.append({"category": d["category"], "source": d["source"]})

        _flush_batch(batch, out_dir, shard_name=shard.name)
        total += len(batch)

    # ---- summary -------------------------------------------------------
    # Per-category distribution and per-source breakdown.
    if all_artifacts:
        df_summary = pd.DataFrame(all_artifacts)
        category_counts = df_summary["category"].value_counts().to_dict()
        source_counts = df_summary["source"].value_counts().to_dict()
        # Ensure every canonical category appears in the summary, even
        # if its count is zero.
        from edit2forensics.data.category_artifact import EDIT_CATEGORIES
        category_counts_full = {
            c: int(category_counts.get(c, 0)) for c in EDIT_CATEGORIES
        }

        summary = {
            "n_total": int(len(df_summary)),
            "category_counts": category_counts_full,
            "source_counts": {k: int(v) for k, v in source_counts.items()},
            "fraction_other": float(
                category_counts_full.get("other", 0) / len(df_summary)
            ),
        }

        report_path = Path(cfg.summary_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2))
        log.info("\n===== category summary =====\n%s", json.dumps(summary, indent=2))
        log.info("wrote summary to %s", report_path)

    log.info(
        "category classification complete: %d artifacts written, %d shards skipped",
        total, skipped_done,
    )


if __name__ == "__main__":
    main()

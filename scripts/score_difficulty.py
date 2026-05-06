"""Stage C entry point: difficulty-score every triplet.

Joins the Stage A triplets parquet with the Stage B mask_artifacts
parquet, runs the `DifficultyScorer` on each row, and emits a parquet
of `DifficultyArtifact`s with empirical-tertile bin labels.

Two-pass design
---------------
Pass 1 (per-row, parallelizable): compute the four raw components and
the weighted-sum `difficulty_raw`. The bin field is set to a
placeholder.

Pass 2 (whole-dataset, fast): read all `difficulty_raw` values, compute
the 33% and 66% empirical quantiles, assign each row its bin, and
overwrite the parquet shards with the populated bins.

The two passes are stitched together in a single invocation by default
(``run_pass1=true, run_pass2=true``), but can be split for parallelism.

Resumability and parallelism
----------------------------
Output shards mirror their input shards' filenames. As with Stage B,
this gives us:

* **Resumability**: an output shard already on disk is skipped unless
  ``overwrite_existing_shards=true``. Interrupted runs resume cleanly.

* **Process-level parallelism**: ``shard_indices`` restricts processing
  to a chosen input-shard subset. Different processes can handle
  disjoint shards concurrently. CRITICAL: parallel workers must run
  Pass 1 only (``run_pass2=false``) because Pass 2 mutates every output
  shard with bin labels and would race. After all parallel Pass-1
  workers complete, run ONCE more with ``run_pass1=false, run_pass2=true``
  to compute and assign the bins over the merged output.

Example for 8-way parallelism::

    N=$(ls artifacts/triplets/pico_banana.parquet/part-*.parquet | wc -l)

    # Launch 8 Pass-1 workers (no Pass 2)
    seq 0 $((N-1)) | xargs -n1 -P8 -I{} \\
        uv run python scripts/score_difficulty.py \\
            triplets_path=artifacts/triplets/pico_banana.parquet \\
            mask_artifacts_path=artifacts/mask_artifacts/pico_banana.parquet \\
            run_tag=pico_banana paths.output_root=artifacts \\
            run_pass2=false "shard_indices=[{}]"

    # Then run Pass 2 once over the merged output
    uv run python scripts/score_difficulty.py \\
        triplets_path=artifacts/triplets/pico_banana.parquet \\
        mask_artifacts_path=artifacts/mask_artifacts/pico_banana.parquet \\
        run_tag=pico_banana paths.output_root=artifacts \\
        run_pass1=false

Usage (sequential, default)::

    uv run python scripts/score_difficulty.py \\
        triplets_path=/path/to/artifacts/triplets/magicbrush_dev.parquet \\
        mask_artifacts_path=/path/to/artifacts/mask_artifacts/magicbrush_dev.parquet \\
        run_tag=magicbrush_dev \\
        paths.output_root=/path/to/artifacts
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.difficulty.scorer import assign_tertile_bins

log = logging.getLogger(__name__)


def _load_parquet_dataset(path: Path) -> pd.DataFrame:
    shards = sorted(path.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {path}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def _flush_batch(rows: list[dict], out_dir: Path, shard_name: str) -> None:
    """Write a batch of artifact dicts to ``out_dir / shard_name``.

    The output filename is derived from the *input* shard name passed
    in, not from a counter. This makes parallel workers safe — two
    concurrent workers writing different input shards never pick the
    same output filename.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    # `weights` is a dict per row; serialize to JSON for parquet.
    df["weights"] = df["weights"].apply(json.dumps)
    df.to_parquet(out_dir / shard_name, index=False)
    log.info("wrote %d difficulty artifacts to %s", len(df), shard_name)


def _pass1_score(
    cfg: DictConfig,
    scorer,
    out_dir: Path,
) -> None:
    """Pass 1: per-row scoring with placeholder bins.

    Memory strategy
    ---------------
    Pico-Banana has ~400K triplets. Loading both the triplets and
    mask-artifacts parquets fully into memory, then doing a
    pandas merge, was triggering the OOM killer on full-dataset runs
    (the script vanished without a traceback because the kernel killed
    the parent shell). MagicBrush dev (528 rows) was fine — the bug was
    purely a function of dataset size.

    Fix: stream the triplets parquet shard by shard, and use a small
    in-memory lookup for the mask artifacts (which have only scalar
    columns and stay under ~100 MB even at 400K rows). Per-shard
    processing keeps memory flat regardless of dataset size.

    We also restrict the triplet read to only the columns the scorer
    actually needs — the `metadata` column on Pico-Banana carries
    JSON-encoded URLs and source paths and is the heaviest column by
    far, but Stage C never reads it.
    """
    triplets_path = Path(cfg.triplets_path)
    masks_path = Path(cfg.mask_artifacts_path)

    all_triplet_shards = sorted(triplets_path.glob("part-*.parquet"))
    if not all_triplet_shards:
        raise FileNotFoundError(f"no triplet shards under {triplets_path}")

    # Optional shard-level parallelism — see module docstring for usage.
    shard_indices = cfg.get("shard_indices")
    if shard_indices is not None:
        shard_indices = [int(i) for i in shard_indices]
        n = len(all_triplet_shards)
        bad = [i for i in shard_indices if i < 0 or i >= n]
        if bad:
            raise IndexError(
                f"shard_indices={shard_indices} contains out-of-range entries "
                f"{bad} (only {n} shards available, indices 0..{n - 1})"
            )
        triplet_shards = [all_triplet_shards[i] for i in shard_indices]
        log.info(
            "processing %d/%d input shards (indices=%s)",
            len(triplet_shards), n, shard_indices,
        )
    else:
        triplet_shards = all_triplet_shards

    # --- mask lookup -------------------------------------------------------
    # Read all mask artifacts once. Only scalar columns; safe at 400K+ rows.
    mask_columns = [
        "triplet_id", "mask_path", "mask_area_frac", "edit_scope",
        "registration_ok", "confidence", "diff_strongest_signal",
    ]
    # combined_diff_mean and mask_compactness are both optional — present
    # in newer artifacts (the latter only after add_mask_compactness.py
    # has been run, or from future Stage B runs that compute it natively).
    # Read defensively.
    masks_df = _load_parquet_dataset(masks_path)
    if "combined_diff_mean" in masks_df.columns:
        mask_columns.append("combined_diff_mean")
    if "mask_compactness" in masks_df.columns:
        mask_columns.append("mask_compactness")
    masks_df = masks_df[mask_columns]

    # Deduplicate on triplet_id BEFORE indexing.
    #
    # Why: when Stage B is re-run without clearing its output directory,
    # the driver appends new shards rather than overwriting them. Every
    # such re-run leaves another full copy of the artifacts on disk,
    # all keyed by the same triplet_ids. The downstream join then sees
    # N copies of every key, producing an N-fold Cartesian explosion
    # against the triplets — observed in the wild as a 50K-triplet
    # join exploding to ~58M rows.
    #
    # We keep the LAST occurrence per triplet_id (most-recently-written
    # shard wins). The choice of "last" rather than "first" matches the
    # operator intuition that re-running a stage updates results.
    n_raw = len(masks_df)
    masks_df = masks_df.drop_duplicates(subset="triplet_id", keep="last")
    n_unique = len(masks_df)
    if n_unique < n_raw:
        log.warning(
            "mask artifacts parquet contains %d duplicate triplet_ids "
            "(%d total rows -> %d unique). Most likely cause: Stage B was "
            "re-run without clearing its output directory. Keeping the "
            "most-recent record per triplet_id; consider clearing %s and "
            "re-running Stage B for a clean parquet.",
            n_raw - n_unique, n_raw, n_unique, masks_path,
        )

    masks_indexed = masks_df.set_index("triplet_id", drop=False)
    # Defense in depth: if our dedup somehow failed (unexpected), the
    # later .join would explode again. Assert the index is unique here
    # so we fail fast with a clear message rather than silently OOM.
    if not masks_indexed.index.is_unique:
        raise RuntimeError(
            "mask_artifacts index is not unique after deduplication; "
            "this should not happen — please file a bug"
        )
    log.info("loaded %d mask artifacts as in-memory lookup", len(masks_indexed))

    # --- columns we need from triplets ------------------------------------
    # Avoids loading the heavy `metadata` column.
    triplet_columns = [
        "triplet_id", "source_dataset", "real_path", "edited_path",
        "instruction", "provided_mask_path",
    ]

    total_scored = 0
    total_failed = 0
    total_unmatched = 0
    total_skipped_done = 0
    # Triplet-side dedup. Same root cause as the mask-side dedup above:
    # re-running Stage A without clearing its output directory leaves
    # duplicate rows. We can't dedup per-shard because duplicates may
    # span shards, so we keep a running seen-set of triplet_ids
    # (~50 bytes per ID; cheap even at millions of triplets).
    seen_triplet_ids: set[str] = set()
    total_triplet_dupes = 0
    overwrite = bool(cfg.get("overwrite_existing_shards", False))

    for shard_idx, shard in enumerate(triplet_shards):
        out_path = out_dir / shard.name
        if out_path.exists() and not overwrite:
            log.info(
                "shard %d/%d (%s): output already exists; skipping",
                shard_idx + 1, len(triplet_shards), shard.name,
            )
            # Still record this shard's triplet_ids in seen_triplet_ids
            # so the dedup logic stays correct across resumed runs.
            seen_in_done = pd.read_parquet(shard, columns=["triplet_id"])
            seen_triplet_ids.update(seen_in_done["triplet_id"].tolist())
            total_skipped_done += 1
            continue

        triplets_shard = pd.read_parquet(shard, columns=triplet_columns)
        log.info(
            "shard %d/%d (%s): %d triplets",
            shard_idx + 1, len(triplet_shards), shard.name, len(triplets_shard),
        )

        # Drop rows whose triplet_id was already processed in an earlier
        # shard or earlier in this shard.
        is_dupe = triplets_shard["triplet_id"].isin(seen_triplet_ids) | \
                  triplets_shard["triplet_id"].duplicated(keep="first")
        if is_dupe.any():
            n_dupe = int(is_dupe.sum())
            total_triplet_dupes += n_dupe
            triplets_shard = triplets_shard[~is_dupe]
        seen_triplet_ids.update(triplets_shard["triplet_id"].tolist())

        # Per-shard inner join against the mask lookup. With masks_indexed
        # already keyed by triplet_id, this is a hash lookup, not a
        # quadratic scan.
        joined = triplets_shard.join(
            masks_indexed.drop(columns=["triplet_id"]),
            on="triplet_id",
            how="inner",
        )

        unmatched = len(triplets_shard) - len(joined)
        total_unmatched += unmatched

        # Per-shard batch — flushed at the end of the shard with the
        # mirrored input shard name. We deliberately don't honor
        # write_batch_size as a mid-shard flush boundary anymore;
        # mid-shard flushing was useful only when output names were
        # counter-derived and is incompatible with parallel safety.
        shard_batch: list[dict] = []
        for _, row in tqdm(
            joined.iterrows(),
            total=len(joined),
            desc=f"shard {shard_idx + 1}/{len(triplet_shards)}",
        ):
            triplet_dict = {
                "triplet_id": row["triplet_id"],
                "source_dataset": row["source_dataset"],
                "real_path": row["real_path"],
                "edited_path": row["edited_path"],
                "instruction": row["instruction"],
                "provided_mask_path": row.get("provided_mask_path"),
                # metadata isn't loaded; downstream scorer doesn't need it.
                "metadata": {},
            }

            mask_dict = {
                "triplet_id": row["triplet_id"],
                "mask_path": row["mask_path"],
                "mask_area_frac": row["mask_area_frac"],
                "edit_scope": row["edit_scope"],
                "registration_ok": row["registration_ok"],
                "confidence": row["confidence"],
                "diff_strongest_signal": row["diff_strongest_signal"],
                "combined_diff_mean": (
                    row["combined_diff_mean"]
                    if "combined_diff_mean" in joined.columns
                    else 0.0
                ),
                "mask_compactness": (
                    row["mask_compactness"]
                    if "mask_compactness" in joined.columns
                    else float("nan")
                ),
            }

            try:
                triplet = EditTriplet.from_dict(triplet_dict)
                mask = MaskArtifact.from_dict(mask_dict)
                artifact = scorer.score(triplet, mask)
            except Exception as e:
                log.warning("scoring failed for %s: %s", row["triplet_id"], e)
                total_failed += 1
                continue

            shard_batch.append(artifact.to_dict())
            total_scored += 1

        _flush_batch(shard_batch, out_dir, shard_name=shard.name)

        # Free shard memory before reading the next one.
        del triplets_shard
        del joined
        del shard_batch

    log.info(
        "pass 1 complete: %d scored, %d failed, %d unmatched (no mask artifact), %d shards skipped",
        total_scored, total_failed, total_unmatched, total_skipped_done,
    )
    if total_triplet_dupes > 0:
        log.warning(
            "skipped %d duplicate triplet_ids on the triplets side. Most "
            "likely cause: Stage A was re-run without clearing %s. "
            "Consider clearing it and re-ingesting for a clean parquet.",
            total_triplet_dupes, triplets_path,
        )


def _pass2_assign_bins(out_dir: Path) -> dict:
    """Pass 2: compute empirical tertiles and rewrite shards in place.

    Returns a small summary dict suitable for logging.
    """
    shards = sorted(out_dir.glob("part-*.parquet"))
    if not shards:
        return {"n": 0}

    # Read all raw scores (one float per row — fits in memory easily for
    # any plausible dataset size).
    score_arrays: list[np.ndarray] = []
    for s in shards:
        score_arrays.append(pd.read_parquet(s, columns=["difficulty_raw"])["difficulty_raw"].to_numpy())
    all_scores = np.concatenate(score_arrays)

    if all_scores.size == 0:
        return {"n": 0}

    bins_all = assign_tertile_bins(all_scores)

    # Rewrite each shard with its slice of the bin assignments. We keep
    # the iteration order identical to read-order above.
    cursor = 0
    for s in shards:
        df = pd.read_parquet(s)
        n = len(df)
        df["difficulty_bin"] = bins_all[cursor : cursor + n]
        cursor += n
        df.to_parquet(s, index=False)

    # Quantile cut points + bin counts for the run report.
    q33 = float(np.quantile(all_scores, 1.0 / 3.0))
    q66 = float(np.quantile(all_scores, 2.0 / 3.0))
    counts: dict[str, int] = {}
    for b in ("easy", "medium", "hard"):
        counts[b] = int((bins_all == b).sum())

    return {
        "n": int(all_scores.size),
        "raw_mean": float(all_scores.mean()),
        "raw_std": float(all_scores.std()),
        "q33": q33,
        "q66": q66,
        "bin_counts": counts,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="score_difficulty")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    out_dir = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_pass1 = bool(cfg.get("run_pass1", True))
    run_pass2 = bool(cfg.get("run_pass2", True))

    if run_pass1:
        scorer = instantiate(cfg.difficulty_scorer)
        log.info("Instruction scorer: %s", type(scorer.instruction_scorer).__name__)
        log.info("Weights: %s", scorer.config.weights)
        _pass1_score(cfg, scorer, out_dir)
    else:
        log.info("run_pass1=false; skipping per-row scoring")

    if run_pass2:
        summary = _pass2_assign_bins(out_dir)
        # Persist the summary alongside the parquet for run audit.
        if summary.get("n", 0) > 0:
            report_path = Path(cfg.summary_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(summary, indent=2))
            log.info("\n===== difficulty summary =====\n%s", json.dumps(summary, indent=2))
            log.info("wrote summary to %s", report_path)
    else:
        log.info("run_pass2=false; skipping bin assignment")


if __name__ == "__main__":
    main()

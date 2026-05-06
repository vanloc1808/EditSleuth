"""Paired-bootstrap significance test for two mask-validation runs.

Given two per-triplet IoU parquets produced by ``evaluate_masks.py``,
this script asks: **is run B's IoU genuinely better than run A's, or
is the observed difference within sampling noise?**

Why paired
----------
Both runs score the same triplets. Some triplets are intrinsically
hard (low IoU under any signal); others are easy. A two-sample test
that ignores this pairing would mistake intrinsic-difficulty variance
for run-to-run variance and produce wide, useless confidence
intervals. The paired test resamples *triplets* — i.e. resamples
``(iou_a_i, iou_b_i)`` pairs together — and looks at the distribution
of mean ``iou_b - iou_a`` over those resamples. This isolates the
signal-difference effect from the per-triplet difficulty effect.

What the script reports
-----------------------
For the overall set, and stratified by ``edit_scope``:

* Mean per-triplet IoU for each run.
* The observed mean difference ``Δ = mean(iou_b - iou_a)``.
* A 95% percentile bootstrap CI for Δ.
* A one-sided percentile p-value for "B > A" (= fraction of bootstrap
  Δs at or below zero). A small p-value (e.g. < 0.05) is evidence
  that B beats A on this dataset.

The stratification is the more useful number in practice: a third
signal might help on local edits while being neutral on globals, and
the stratified table makes that visible where the aggregate would
dilute it.

Usage::

    uv run python scripts/compare_iou_runs.py \\
        run_a.label=lab_lpips \\
        run_a.report_path=/path/to/artifacts/reports/mask_val_lab_lpips.per_triplet.parquet \\
        run_b.label=lab_lpips_ssim \\
        run_b.report_path=/path/to/artifacts/reports/mask_val_lab_lpips_ssim.per_triplet.parquet \\
        output_report=/path/to/artifacts/reports/iou_bootstrap.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired bootstrap of mean(b - a) over resampled triplet indices.

    Resamples row indices with replacement. The same indices select
    rows from both ``a`` and ``b``, preserving the per-triplet pairing.
    """
    n = len(a)
    if n != len(b):
        raise ValueError(f"length mismatch: a={n}, b={len(b)}")
    if n == 0:
        return {
            "n": 0, "mean_a": float("nan"), "mean_b": float("nan"),
            "delta": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "p_value_one_sided": float("nan"),
        }

    deltas = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(b[idx].mean() - a[idx].mean())

    return {
        "n": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": float(b.mean() - a.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        # One-sided test for "B > A": fraction of bootstrap deltas
        # at or below zero. A small value is evidence that B is
        # consistently above A across resamples.
        "p_value_one_sided": float((deltas <= 0).mean()),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="compare_iou_runs")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    a_path = Path(cfg.run_a.report_path)
    b_path = Path(cfg.run_b.report_path)
    label_a = cfg.run_a.label
    label_b = cfg.run_b.label

    df_a = pd.read_parquet(a_path)[["triplet_id", "iou", "edit_scope"]]
    df_b = pd.read_parquet(b_path)[["triplet_id", "iou", "edit_scope"]]
    log.info(
        "%s: %d rows, %s: %d rows",
        label_a, len(df_a), label_b, len(df_b),
    )

    # Inner-join on triplet_id. Edit scope may differ slightly between
    # runs (a config that produces different masks may classify scopes
    # differently); when stratifying we use run A's scope as the
    # ground-truth grouping variable, so the partition is the same
    # subset on both sides.
    merged = df_a.merge(
        df_b, on="triplet_id", suffixes=(f"_{label_a}", f"_{label_b}"),
    )
    log.info("triplets in both runs: %d", len(merged))
    if len(merged) == 0:
        log.error("no triplets in common between the two runs; aborting")
        return

    n_iter = int(cfg.n_iter)
    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)

    iou_a = merged[f"iou_{label_a}"].to_numpy(dtype=np.float64)
    iou_b = merged[f"iou_{label_b}"].to_numpy(dtype=np.float64)

    # --- aggregate bootstrap ----------------------------------------------
    overall = _paired_bootstrap(iou_a, iou_b, n_iter, rng)

    # --- stratified by edit_scope ------------------------------------------
    by_scope: dict[str, dict] = {}
    scope_col = f"edit_scope_{label_a}"
    for scope in sorted(merged[scope_col].unique()):
        mask = merged[scope_col] == scope
        sub_a = merged.loc[mask, f"iou_{label_a}"].to_numpy(dtype=np.float64)
        sub_b = merged.loc[mask, f"iou_{label_b}"].to_numpy(dtype=np.float64)
        # Use a fresh sub-generator so cross-scope stats are independent
        # of the iteration order of unique scopes.
        sub_rng = np.random.default_rng(seed + hash(scope) % 10_000_000)
        by_scope[str(scope)] = _paired_bootstrap(sub_a, sub_b, n_iter, sub_rng)

    summary: dict = {
        "run_a": label_a,
        "run_b": label_b,
        "n_iter": n_iter,
        "seed": seed,
        "overall": overall,
        "by_scope": by_scope,
    }

    out_path = Path(cfg.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "\n===== paired-bootstrap IoU comparison =====\n%s",
        json.dumps(summary, indent=2),
    )
    log.info("wrote full report to %s", out_path)


if __name__ == "__main__":
    main()

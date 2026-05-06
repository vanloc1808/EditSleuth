"""Sweep ``global_mean_threshold`` retrospectively over existing mask
artifacts, reporting how the global classification rate would change.

Why
---
Stage B's ``_threshold`` routes a triplet to ``global`` scope under two
paths:

1. **Pre-routing**: ``combined.mean() >= global_mean_threshold``.
2. **Post-Otsu**: Otsu's threshold produces a mask covering ≥
   ``global_area_threshold`` of the image; promote to global.

Path 1 is what we can simulate retrospectively from saved artifacts:
``MaskArtifact.combined_diff_mean`` is the value the comparison was
done against. Path 2 we can't simulate without re-running Stage B.

This script gives a calibrated answer to **"what `global_mean_threshold`
should I have used?"** without re-generating any masks. It computes,
for each run × candidate threshold:

* The Path-1-only global rate at that threshold.
* The total observed global rate at the run's original threshold
  (a constant per-run; included for context).

The Path-1 rate is a lower bound on the resulting actual global rate
under the new threshold (Path 2 may add more). If Path-1 alone is
already too high at a candidate threshold, the threshold is too low —
no amount of tuning will help. If Path-1 is well-calibrated but the
total is much higher, the issue is Path 2 (Otsu promoting masks that
ended up covering ≥90% of the image after thresholding).

What you do with the output
---------------------------
The summary JSON reports "candidate thresholds": the smallest
``global_mean_threshold`` for which Path-1 global rate is ≤ each of
several target rates (25%, 30%, 35%, 40%, 50%). Pick one based on the
target dataset's expected genuine-global fraction. For Pico-Banana,
edit_type analysis suggests 25-40% of triplets are plausibly
global (style transfers, photo→cartoon, time-of-day, etc.); a
candidate threshold from that band is the right target.

Once you pick a threshold, re-run Stage B with
``mask_generator.config.global_mean_threshold=<value>`` to produce
the production masks under the new calibration.

Usage
-----
::

    uv run python scripts/sweep_global_threshold.py \\
        +runs.lab_ssim=/path/to/artifacts/mask_artifacts/pico_banana_lab_ssim.parquet \\
        +runs.lab_lpips=/path/to/artifacts/mask_artifacts/pico_banana_lab_lpips.parquet \\
        +runs.lab_lpips_ssim=/path/to/artifacts/mask_artifacts/pico_banana_lab_lpips_ssim.parquet \\
        output_report=/path/to/artifacts/reports/threshold_sweep.json

The ``+`` prefix is Hydra's "add new key" idiom; the ``runs:`` mapping
in the YAML is left open so any number of runs can be passed.
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


def _load_artifacts(path: Path, label: str) -> pd.DataFrame:
    """Load mask-artifacts parquet directory; deduplicate on triplet_id.

    Same deduplication as ``compare_mask_runs.py`` — Stage B re-runs
    that don't clear output leave duplicate rows.
    """
    shards = sorted(path.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {path}")
    df = pd.concat(
        [pd.read_parquet(s, columns=["triplet_id", "edit_scope",
                                      "mask_area_frac", "combined_diff_mean"])
         for s in shards],
        ignore_index=True,
    )
    n_raw = len(df)
    df = df.drop_duplicates(subset="triplet_id", keep="last")
    if len(df) < n_raw:
        log.warning(
            "%s: dropped %d duplicate triplet_ids (%d -> %d unique)",
            label, n_raw - len(df), n_raw, len(df),
        )
    return df


def _sweep_thresholds(
    cdm: np.ndarray, thresholds: list[float]
) -> list[dict]:
    """For each threshold, compute the fraction of rows that Path 1
    would have classified as global (``combined_diff_mean >= t``)."""
    n = len(cdm)
    results = []
    for t in thresholds:
        path1_rate = float((cdm >= t).mean()) if n > 0 else 0.0
        results.append({
            "threshold": float(t),
            "path1_global_rate": path1_rate,
            "n_path1_global": int((cdm >= t).sum()),
        })
    return results


def _find_candidate_thresholds(
    sweep: list[dict],
    target_rates: list[float],
) -> dict[str, float | None]:
    """For each target rate, find the smallest threshold whose Path-1
    rate is at or below it.

    Returns a dict ``{target_rate_str: chosen_threshold}``. If no swept
    threshold achieves the target, returns ``None`` for that target.
    The user can then re-sweep at finer resolution if desired.
    """
    out: dict[str, float | None] = {}
    # Sweep is in threshold-ascending order; find the first row whose
    # rate is <= target_rate.
    for tr in target_rates:
        chosen: float | None = None
        for row in sweep:
            if row["path1_global_rate"] <= tr:
                chosen = row["threshold"]
                break
        out[f"{tr:.2f}"] = chosen
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="sweep_global_threshold")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Build the threshold list. Default is 0.0..1.0 in steps of 0.05;
    # configurable via cfg.thresholds (a list) or cfg.threshold_step (a
    # scalar that builds a uniform sweep).
    if "thresholds" in cfg and cfg.thresholds is not None:
        thresholds = sorted(float(t) for t in cfg.thresholds)
    else:
        step = float(cfg.threshold_step)
        # Use a slightly-extended range so the sweep brackets all
        # plausible operating points.
        n_steps = int(round(1.0 / step)) + 1
        thresholds = [round(i * step, 6) for i in range(n_steps)]
    log.info("sweeping %d thresholds from %g to %g (step %g)",
             len(thresholds),
             thresholds[0], thresholds[-1],
             thresholds[1] - thresholds[0] if len(thresholds) > 1 else 0)

    target_rates = [float(r) for r in cfg.target_rates]
    log.info("target rates for candidate selection: %s", target_rates)

    runs_cfg = cfg.runs
    if not runs_cfg or len(runs_cfg) == 0:
        raise ValueError("at least one run must be specified under `runs:`")

    summary: dict = {
        "thresholds": thresholds,
        "target_rates": target_rates,
        "runs": {},
    }

    for label, artifact_path in runs_cfg.items():
        df = _load_artifacts(Path(artifact_path), label=label)
        cdm = df["combined_diff_mean"].to_numpy(dtype=np.float64)

        # The actual observed global rate at the run's original
        # threshold — fixed per-run, doesn't depend on the swept
        # threshold; included for sanity-checking and to surface the
        # Path-2 contribution.
        observed_global_rate = float((df["edit_scope"] == "global").mean())
        observed_global_count = int((df["edit_scope"] == "global").sum())

        sweep = _sweep_thresholds(cdm, thresholds)
        candidates = _find_candidate_thresholds(sweep, target_rates)

        log.info(
            "[%s] n=%d, cdm mean=%.3f, std=%.3f, observed global rate=%.3f",
            label, len(df), cdm.mean(), cdm.std(), observed_global_rate,
        )
        log.info(
            "[%s] candidate thresholds for target rates: %s",
            label, candidates,
        )

        summary["runs"][str(label)] = {
            "n_triplets": int(len(df)),
            "combined_diff_mean": {
                "mean": float(cdm.mean()),
                "std": float(cdm.std()),
                "p25": float(np.quantile(cdm, 0.25)),
                "p50": float(np.quantile(cdm, 0.50)),
                "p75": float(np.quantile(cdm, 0.75)),
            },
            "observed_global_rate": observed_global_rate,
            "observed_global_count": observed_global_count,
            "sweep": sweep,
            "candidate_thresholds": candidates,
        }

    out_path = Path(cfg.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("wrote sweep report to %s", out_path)

    # Also print a compact cross-run table for at-a-glance reading.
    log.info("\n===== Path-1 global rate by run × threshold =====")
    header = "threshold | " + " | ".join(f"{lbl:>14}" for lbl in runs_cfg.keys())
    log.info(header)
    log.info("-" * len(header))
    n_runs = len(runs_cfg)
    rate_table = {
        lbl: summary["runs"][lbl]["sweep"] for lbl in runs_cfg.keys()
    }
    for i, t in enumerate(thresholds):
        cells = [f"{t:>9.2f}"]
        for lbl in runs_cfg.keys():
            cells.append(f"{rate_table[lbl][i]['path1_global_rate']:>14.3f}")
        log.info(" | ".join(cells))


if __name__ == "__main__":
    main()

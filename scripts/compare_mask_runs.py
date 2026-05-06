"""Compare two mask-artifact parquets from different signal configurations.

Used to diagnose whether differences between signal stacks (e.g. LAB-only
vs. LAB+LPIPS, LAB+LPIPS vs. LAB+LPIPS+SSIM) come from genuine signal
complementarity or from one configuration spuriously upgrading local
edits to global. Useful when ground-truth masks are unavailable
(Pico-Banana, GenImage cross-generator sets) and head-to-head IoU
vs GT isn't possible.

What it reports
---------------
For each triplet present in BOTH parquets, we cross-tabulate the
``edit_scope`` field (local/global/ambiguous/alignment_failed) and
report:

* **Confirmed globals**: both runs say ``global``. High confidence the
  edit is genuinely global.
* **Disputed globals**: one run says ``global``, the other says
  ``local``. Diagnostic of signal-induced routing changes — small
  numbers mean the runs basically agree; large numbers mean the new
  signal is shifting routing decisions in ways worth examining.
* **Both local**: triplets where both runs produced local masks.
  Pairwise IoU on this subset quantifies localization disagreement
  when both runs agree on scope.
* **IoU on disputed**: pairwise IoU on the scope-disputed subset.
  Tells us whether disputed cases are small mask-area edits (the runs
  mostly agree on the localized region but disagree on whether it's
  big enough to be global) or fundamentally different localizations.
* **Disputed examples**: a small list of disputed triplet_ids with
  their per-run scope and area, for spot-checking.

The script does NOT need GT masks — it operates entirely on the
relationship between two adapter-produced runs.

Usage::

    uv run python scripts/compare_mask_runs.py \\
        run_a.label=lab_lpips \\
        run_a.mask_artifacts_path=/path/to/artifacts/mask_artifacts/pico_banana_lab_lpips.parquet \\
        run_b.label=lab_lpips_ssim \\
        run_b.mask_artifacts_path=/path/to/artifacts/mask_artifacts/pico_banana_lab_lpips_ssim.parquet \\
        output_report=/path/to/artifacts/reports/mask_comparison_lpips_vs_lpips_ssim.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)


def _load_parquet_dataset(path: Path, *, label: str) -> pd.DataFrame:
    """Load a mask-artifacts parquet directory.

    Deduplicates on ``triplet_id`` (keeping the most-recent record).
    Without this, re-runs of Stage B that didn't clear the output dir
    leave duplicate rows, and the downstream merge explodes
    Cartesian-style — observed in the Stage C OOM bug.
    """
    shards = sorted(path.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {path}")
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates(subset="triplet_id", keep="last")
    if len(df) < n_raw:
        log.warning(
            "%s: dropped %d duplicate triplet_ids (%d total -> %d unique). "
            "Most likely cause: Stage B was re-run without clearing %s.",
            label, n_raw - len(df), n_raw, len(df), path,
        )
    return df


def _load_mask_binary(path: str | Path) -> np.ndarray:
    """Load a mask PNG as a boolean (H, W) array."""
    return np.asarray(Image.open(path).convert("L")) > 127


def _pairwise_iou(mask_a_path, mask_b_path) -> float | None:
    """IoU between two on-disk masks. ``None`` if either file is missing
    or the shapes don't agree (we don't try to align cross-run masks
    here — they should have been generated from the same triplet)."""
    try:
        a = _load_mask_binary(mask_a_path)
        b = _load_mask_binary(mask_b_path)
    except FileNotFoundError:
        return None
    if a.shape != b.shape:
        # Resize b to a's resolution; nearest preserves binary structure.
        b_img = Image.open(mask_b_path).convert("L").resize(
            (a.shape[1], a.shape[0]), Image.NEAREST
        )
        b = np.asarray(b_img) > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty => perfect agreement
    return float(inter) / float(union)


@hydra.main(version_base=None, config_path="../configs", config_name="compare_mask_runs")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    df_a = _load_parquet_dataset(Path(cfg.run_a.mask_artifacts_path), label=cfg.run_a.label)
    df_b = _load_parquet_dataset(Path(cfg.run_b.mask_artifacts_path), label=cfg.run_b.label)
    label_a = cfg.run_a.label
    label_b = cfg.run_b.label

    log.info("%s rows: %d, %s rows: %d", label_a, len(df_a), label_b, len(df_b))

    # Inner-join on triplet_id so we only compare triplets present in both.
    joined = df_a.merge(
        df_b, on="triplet_id", suffixes=(f"_{label_a}", f"_{label_b}"),
    )
    log.info("triplets in both runs: %d", len(joined))
    if len(joined) == 0:
        log.error("no triplets in common; aborting")
        return

    scope_a = joined[f"edit_scope_{label_a}"]
    scope_b = joined[f"edit_scope_{label_b}"]
    area_a = joined[f"mask_area_frac_{label_a}"]
    area_b = joined[f"mask_area_frac_{label_b}"]
    cdm_a = joined.get(f"combined_diff_mean_{label_a}", pd.Series([0.0] * len(joined)))
    cdm_b = joined.get(f"combined_diff_mean_{label_b}", pd.Series([0.0] * len(joined)))

    # --- scope cross-tab --------------------------------------------------
    crosstab = pd.crosstab(scope_a, scope_b, dropna=False)
    log.info("\nscope cross-tab (%s rows x %s cols):\n%s",
             label_a, label_b, crosstab.to_string())

    # --- key segments -----------------------------------------------------
    confirmed_global = (scope_a == "global") & (scope_b == "global")
    a_only_global = (scope_a == "global") & (scope_b == "local")
    b_only_global = (scope_a == "local") & (scope_b == "global")
    both_local = (scope_a == "local") & (scope_b == "local")

    log.info("confirmed global (both runs): %d", int(confirmed_global.sum()))
    log.info("%s-only global (vs local in %s): %d",
             label_a, label_b, int(a_only_global.sum()))
    log.info("%s-only global (vs local in %s): %d",
             label_b, label_a, int(b_only_global.sum()))
    log.info("both local: %d", int(both_local.sum()))

    # --- IoU between mask sets, on chosen subsets ------------------------
    # We compute IoU on TWO subsets:
    #   1. `both_local`: triplets where both runs produced a non-trivial
    #      local mask. Tells us how much the configurations disagree on
    #      localization when they agree on scope.
    #   2. `disputed_local`: union of the two "X-only-global" segments
    #      restricted to the run that DIDN'T promote to global. Tells us
    #      whether disputed cases are small mask-area edits (the runs
    #      mostly agree on the localized region but disagree on whether
    #      it's "big enough" to be global) or fundamentally different
    #      localizations.
    #
    # iou_sample_size=-1 means "use all shared rows in the subset" (full
    # pass). Otherwise we sample for speed — the I/O cost is two PNG
    # loads per sampled triplet.
    def _iou_on_subset(rows: pd.DataFrame, desc: str) -> dict:
        if len(rows) == 0:
            return {"n": 0}
        n_sample_cap = int(cfg.iou_sample_size)
        if n_sample_cap == -1 or n_sample_cap >= len(rows):
            sub = rows
        else:
            sub = rows.sample(n=n_sample_cap, random_state=int(cfg.seed))
        ious: list[float] = []
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=desc):
            v = _pairwise_iou(
                row[f"mask_path_{label_a}"], row[f"mask_path_{label_b}"]
            )
            if v is not None:
                ious.append(v)
        if not ious:
            return {"n": 0}
        arr = np.array(ious)
        return {
            "n": int(arr.size),
            "n_sampled_from": int(len(rows)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        }

    iou_both_local = _iou_on_subset(joined[both_local], desc="iou:both_local")
    iou_disputed = _iou_on_subset(
        joined[a_only_global | b_only_global], desc="iou:disputed",
    )

    # --- list of disputed triplets for actionable follow-up ---------------
    # These are the triplets where the runs disagree on scope. A small
    # number means the runs basically agree; a large number means SSIM
    # (or whichever new signal) is shifting the routing decision in
    # ways worth examining manually. Save up to `disputed_list_cap`
    # triplet_ids to the report so the operator can grep for them in
    # the source manifest.
    disputed_rows = joined[a_only_global | b_only_global]
    cap = int(cfg.get("disputed_list_cap", 50))
    disputed_records = []
    for _, r in disputed_rows.head(cap).iterrows():
        disputed_records.append({
            "triplet_id": r["triplet_id"],
            f"scope_{label_a}": r[f"edit_scope_{label_a}"],
            f"scope_{label_b}": r[f"edit_scope_{label_b}"],
            f"area_{label_a}": float(r[f"mask_area_frac_{label_a}"]),
            f"area_{label_b}": float(r[f"mask_area_frac_{label_b}"]),
        })

    # --- assemble report --------------------------------------------------
    summary: dict = {
        "n_in_both": int(len(joined)),
        "scope_crosstab": {
            f"{a}__{b}": int(crosstab.loc[a, b])
            for a in crosstab.index for b in crosstab.columns
        },
        "confirmed_global": int(confirmed_global.sum()),
        f"{label_a}_only_global": int(a_only_global.sum()),
        f"{label_b}_only_global": int(b_only_global.sum()),
        "both_local": int(both_local.sum()),
        "combined_diff_mean": {
            label_a: {
                "mean": float(cdm_a.mean()),
                "global_subset_mean": float(cdm_a[scope_a == "global"].mean())
                if (scope_a == "global").any() else None,
            },
            label_b: {
                "mean": float(cdm_b.mean()),
                "global_subset_mean": float(cdm_b[scope_b == "global"].mean())
                if (scope_b == "global").any() else None,
            },
        },
        "area_frac": {
            label_a: {
                "mean": float(area_a.mean()),
                "frac_eq_1": float((area_a >= 0.999).mean()),
            },
            label_b: {
                "mean": float(area_b.mean()),
                "frac_eq_1": float((area_b >= 0.999).mean()),
            },
        },
        "iou_on_shared_local_subset": iou_both_local,
        "iou_on_disputed_subset": iou_disputed,
        "disputed_examples": disputed_records,
    }

    out_path = Path(cfg.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "\n===== mask comparison summary =====\n%s",
        json.dumps(summary, indent=2),
    )
    log.info("wrote full report to %s", out_path)


if __name__ == "__main__":
    main()

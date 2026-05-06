"""Diagnostic: V2 difficulty score behavior on the local-scope subset.

Answers the question: **on local-scope triplets alone, is
compactness_score genuinely independent of structural_change?**

Background
----------
On the full Pico-Banana dataset, V2 raw_std is 0.109 — a 55% improvement
over V1's 0.070, but smaller than predicted. The correlation matrix
showed structural_change ↔ compactness_score = -0.61, suggesting
compactness was not the independent signal we'd hoped.

Hypothesis: the anticorrelation is driven by global-scope triplets,
where Stage B's routing forces an all-ones mask, which deterministically
makes compactness=1 (compactness_score=0). Globals with their high
structural_change but zero compactness_score create an artifact that
isn't a property of compactness — it's a property of the routing.

If the hypothesis is right, the local-scope subset should show:
* Much weaker correlation between structural_change and compactness_score.
* Higher per-component compactness_score variance (no clamped-at-zero mass).
* A predicted V2 raw_std (computed analytically from the local-subset
  covariance) substantially larger than the 0.109 observed on the full set.

What this script reports
------------------------
1. Subset sizes (full dataset, local subset, global subset, ambiguous, alignment_failed).
2. Per-component statistics on each subset (mean, std, quantiles).
3. Correlation matrices on the full set, local-only, and global-only.
4. Analytic predicted V2 raw_std on each subset, computed from the
   weighted sum's variance formula.

Usage
-----
::

    uv run python scripts/diagnose_v2_local_subset.py \\
        --difficulty-parquet artifacts/difficulty/pico_banana_v2.parquet \\
        --mask-parquet artifacts/mask_artifacts/pico_banana_lab_lpips_ssim_calibrated_with_compactness.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# V2 default weights — used to compute predicted raw_std analytically.
V2_WEIGHTS = {
    "structural_change": 0.55,
    "compactness_score": 0.25,
    "instruction_complexity": 0.20,
}
V2_COMPONENTS = list(V2_WEIGHTS.keys())


def _load_and_dedup(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    """Load a parquet directory, dedup on triplet_id, return only
    the requested columns. Same defensive pattern used elsewhere —
    re-runs of upstream stages can leave duplicate rows."""
    shards = sorted(path.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {path}")
    df = pd.concat(
        [pd.read_parquet(s, columns=columns) for s in shards],
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


def _component_stats(df: pd.DataFrame, label: str) -> dict:
    """Per-component summary stats on the V2 components."""
    out: dict = {"n": int(len(df))}
    if len(df) == 0:
        return out
    for comp in V2_COMPONENTS:
        col = df[comp].to_numpy(dtype=np.float64)
        out[comp] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "q25": float(np.quantile(col, 0.25)),
            "q50": float(np.quantile(col, 0.50)),
            "q75": float(np.quantile(col, 0.75)),
            "max": float(col.max()),
            # Mass of compactness_score at exactly 0 — diagnostic for
            # the global-routing degeneracy.
            "frac_eq_0": float((col == 0.0).mean()) if comp == "compactness_score" else None,
        }
    return out


def _correlation_matrix(df: pd.DataFrame) -> dict:
    """Pairwise Pearson correlations among V2 components."""
    if len(df) < 2:
        return {}
    M = df[V2_COMPONENTS].to_numpy(dtype=np.float64)
    # np.corrcoef wants variables in rows.
    C = np.corrcoef(M.T)
    out: dict = {}
    for i, a in enumerate(V2_COMPONENTS):
        out[a] = {b: float(C[i, j]) for j, b in enumerate(V2_COMPONENTS)}
    return out


def _predicted_raw_std(df: pd.DataFrame) -> dict:
    """Compute predicted V2 raw_std on this subset using the closed
    form for variance of a weighted sum:

        var(w · X) = sum_i w_i² var(X_i)
                   + 2 sum_{i<j} w_i w_j cov(X_i, X_j)

    The point is to predict what raw_std *would be* on this subset
    without actually re-running the scorer — useful because we want
    to know whether restricting to locals would solve the variance
    problem.
    """
    if len(df) < 2:
        return {"std": float("nan"), "n": int(len(df))}

    M = df[V2_COMPONENTS].to_numpy(dtype=np.float64)
    cov = np.cov(M.T, ddof=0)  # population cov to match np.std default
    w = np.array([V2_WEIGHTS[c] for c in V2_COMPONENTS])
    var = float(w @ cov @ w)
    if var < 0:
        # Floating-point can produce tiny negative values for near-rank-
        # deficient covariances. Clamp to zero rather than return NaN.
        var = 0.0
    raw_mean = float(M @ w).mean() if False else float((M @ w).mean())
    return {
        "predicted_std": float(np.sqrt(var)),
        "predicted_mean": raw_mean,
        "n": int(len(df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--difficulty-parquet", type=Path, required=True,
        help="V2 difficulty parquet directory",
    )
    parser.add_argument(
        "--mask-parquet", type=Path, required=True,
        help="Stage B mask_artifacts parquet directory (provides edit_scope)",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Optional path to write the full report as JSON.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    diff_df = _load_and_dedup(
        args.difficulty_parquet,
        columns=["triplet_id", "structural_change", "compactness_score",
                 "instruction_complexity", "difficulty_raw"],
        label="difficulty",
    )
    mask_df = _load_and_dedup(
        args.mask_parquet,
        columns=["triplet_id", "edit_scope"],
        label="mask_artifacts",
    )
    joined = diff_df.merge(mask_df, on="triplet_id", how="inner")
    log.info(
        "diff: %d, mask: %d, joined: %d",
        len(diff_df), len(mask_df), len(joined),
    )

    scope_counts = joined["edit_scope"].value_counts().to_dict()
    log.info("scope distribution: %s", scope_counts)

    # Build per-subset slices.
    subsets = {
        "all": joined,
        "local": joined[joined["edit_scope"] == "local"],
        "global": joined[joined["edit_scope"] == "global"],
        "ambiguous": joined[joined["edit_scope"] == "ambiguous"],
        "alignment_failed": joined[joined["edit_scope"] == "alignment_failed"],
    }

    report: dict = {
        "scope_counts": {k: int(v) for k, v in scope_counts.items()},
        "v2_weights": V2_WEIGHTS,
        "subsets": {},
    }
    for name, sub in subsets.items():
        if len(sub) == 0:
            continue
        report["subsets"][name] = {
            "stats": _component_stats(sub, name),
            "correlations": _correlation_matrix(sub),
            "predicted_raw_std": _predicted_raw_std(sub),
        }

    # --- pretty-print to stdout for human inspection ----------------------
    print("\n===== V2 local-subset diagnostic =====\n")
    print(f"Scope distribution: {report['scope_counts']}")
    print(f"V2 weights: {V2_WEIGHTS}")

    for name in ("all", "local", "global"):
        sub_data = report["subsets"].get(name)
        if sub_data is None:
            continue
        n = sub_data["stats"]["n"]
        print(f"\n--- subset: {name} (n={n}) ---")

        # Stats table
        print("\nComponent stats:")
        print(f"{'component':<25} {'mean':>8} {'std':>8} {'q25':>8} {'q50':>8} {'q75':>8}")
        for comp in V2_COMPONENTS:
            s = sub_data["stats"][comp]
            print(
                f"{comp:<25} {s['mean']:>8.4f} {s['std']:>8.4f} "
                f"{s['q25']:>8.4f} {s['q50']:>8.4f} {s['q75']:>8.4f}"
            )
            if comp == "compactness_score" and s.get("frac_eq_0") is not None:
                print(f"{'  └ frac_eq_0':<25} {s['frac_eq_0']:>8.4f}")

        # Correlation matrix
        print("\nCorrelation matrix:")
        print(f"{'':25} " + " ".join(f"{c[:18]:>20}" for c in V2_COMPONENTS))
        for a in V2_COMPONENTS:
            row = sub_data["correlations"][a]
            print(
                f"{a:<25} "
                + " ".join(f"{row[b]:>20.4f}" for b in V2_COMPONENTS)
            )

        # Predicted raw_std on this subset
        pred = sub_data["predicted_raw_std"]
        print(f"\nPredicted V2 raw_std on this subset: {pred['predicted_std']:.4f}")
        print(f"  (analytic; from weighted-sum variance over the subset's covariance)")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.output_json}")


if __name__ == "__main__":
    main()

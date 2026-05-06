"""Calibrate the four difficulty-component weights against manual labels.

This script implements the weight-fitting protocol.
Given:

* A parquet of `DifficultyArtifact`s (Stage C output, with raw component
  scores per triplet).
* A small JSON of manual easy/medium/hard labels keyed by ``triplet_id``.

we fit non-negative weights summing to 1 such that the resulting
``difficulty_raw`` ranks the labeled triplets in agreement with the
manual labels (Spearman rank correlation).

The fit is a small numerical optimization on the simplex (4 parameters,
sum-to-one, non-negative). We use scipy's ``minimize`` with SLSQP,
which handles the constraints cleanly. SLSQP is overkill for 4
parameters, but the convenience wins out over hand-rolling a projected
gradient.

Output: a small JSON with the fitted weights and the achieved Spearman
correlation. Drop the weights into ``configs/difficulty_scorer/...``
and re-run ``score_difficulty.py`` to reflect the new calibration.

Usage::

    uv run python scripts/calibrate_difficulty_weights.py \\
        difficulty_path=/path/to/artifacts/difficulty/magicbrush_dev.parquet \\
        labels_path=/path/to/manual_labels.json \\
        output_path=/path/to/artifacts/reports/calibrated_weights.json

The labels JSON has the form::

    {"picobanana_00000001": "easy",
     "picobanana_00000042": "hard",
     ...}

Only labels present in BOTH the parquet and the JSON are used.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

_COMPONENTS = (
    "structural_change",
    "perceptual_change",
    "locality",
    "instruction_complexity",
)

# Map manual labels to a numeric "ground-truth difficulty" ordering.
_LABEL_TO_RANK = {"easy": 0.0, "medium": 1.0, "hard": 2.0}


def _objective(weights: np.ndarray, components: np.ndarray, gt: np.ndarray) -> float:
    """Negative Spearman ρ — we minimize this to maximize ρ."""
    raw = components @ weights  # (N,)
    rho, _ = spearmanr(raw, gt)
    if np.isnan(rho):
        # Happens for degenerate weights (e.g., all-zero); penalize.
        return 1.0
    return -float(rho)


def _fit_weights(components: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit non-negative simplex weights via SLSQP.

    `components` is shape (N, 4), `gt` is shape (N,).
    Returns (weights, achieved_rho).
    """
    # Start from the default values so a single optimizer step
    # near them tells us whether they're already a local optimum.
    x0 = np.array([0.30, 0.30, 0.20, 0.20], dtype=np.float64)

    constraints = [
        {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
    ]
    bounds = [(0.0, 1.0)] * 4

    result = minimize(
        fun=_objective,
        x0=x0,
        args=(components, gt),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 200, "disp": False},
    )
    return result.x, -float(result.fun)


@hydra.main(version_base=None, config_path="../configs", config_name="calibrate_difficulty")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- load Stage C parquet ---------------------------------------------
    diff_path = Path(cfg.difficulty_path)
    shards = sorted(diff_path.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {diff_path}")
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    log.info("loaded %d difficulty rows", len(df))

    # --- load manual labels -----------------------------------------------
    labels = json.loads(Path(cfg.labels_path).read_text())
    if not isinstance(labels, dict):
        raise ValueError("labels JSON must be a {triplet_id: label} object")
    bad = [v for v in labels.values() if v not in _LABEL_TO_RANK]
    if bad:
        raise ValueError(
            f"labels contain unrecognized values {set(bad)}; "
            f"expected one of {set(_LABEL_TO_RANK)}"
        )

    # --- inner-join on triplet_id ----------------------------------------
    df_labeled = df[df["triplet_id"].isin(labels.keys())].copy()
    if df_labeled.empty:
        raise ValueError("no overlap between difficulty parquet and labels")

    df_labeled["__gt"] = df_labeled["triplet_id"].map(
        lambda t: _LABEL_TO_RANK[labels[t]]
    )
    log.info("labeled overlap: %d rows", len(df_labeled))
    if len(df_labeled) < 50:
        log.warning(
            "only %d labeled rows — calibration may be noisy. "
            "Require ~500 manual labels.",
            len(df_labeled),
        )

    # --- assemble components matrix --------------------------------------
    components = np.stack(
        [df_labeled[c].to_numpy(dtype=np.float64) for c in _COMPONENTS],
        axis=1,
    )
    gt = df_labeled["__gt"].to_numpy(dtype=np.float64)

    # --- baseline: default weights -----------------------------
    default_w = np.array([0.30, 0.30, 0.20, 0.20], dtype=np.float64)
    baseline_raw = components @ default_w
    baseline_rho, _ = spearmanr(baseline_raw, gt)

    # --- fit weights via SLSQP -------------------------------------------
    fitted_w, fitted_rho = _fit_weights(components, gt)

    log.info("baseline Spearman rho: %.4f", float(baseline_rho))
    log.info("fitted Spearman rho:               %.4f", fitted_rho)
    log.info("fitted weights: %s",
             dict(zip(_COMPONENTS, fitted_w.round(4).tolist())))

    out = {
        "n_labeled": int(len(df_labeled)),
        "baseline": {
            "weights": dict(zip(_COMPONENTS, default_w.tolist())),
            "spearman_rho": float(baseline_rho) if not np.isnan(baseline_rho) else None,
        },
        "fitted": {
            "weights": dict(zip(_COMPONENTS, fitted_w.tolist())),
            "spearman_rho": float(fitted_rho),
        },
    }

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote calibration report to %s", out_path)


if __name__ == "__main__":
    main()

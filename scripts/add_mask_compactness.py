"""Augment an existing mask_artifacts parquet with mask compactness.

Walks a Stage B output directory, reads each mask PNG, computes a
single ``compactness`` score per row, and writes a new parquet with
all original columns plus ``mask_compactness``. Does not modify the
input parquet — the augmented version is a separate output.

Why retrospective
-----------------
Stage B will eventually compute and persist this natively (one-line
addition to ``MaskArtifact`` plus a few lines in the generator's
finalize phase), but doing so requires re-running Stage B on the full
dataset. This script avoids that cost when an existing mask set is
already on disk: compactness is a pure function of the binary mask
PNG, so we can compute it post-hoc without touching the original Stage
B logic.

What gets computed
------------------
``mask_compactness = sqrt(area_to_bbox * largest_component_frac)`` where:

* **area_to_bbox** = ``mask_area / (bbox_height * bbox_width)``.
  Captures "blobby vs elongated" — a tall thin strip is less compact
  than a square of the same area.
* **largest_component_frac** = ``area_of_largest_connected_component /
  total_mask_area``. Captures "concentrated vs scattered" — a single
  blob has frac = 1.0, an N-blob scatter has frac < 1.0.

The geometric mean treats both terms symmetrically; a low value in
either drives the combined value low. Both terms are in ``[0, 1]``,
so the combined is too.

Conventional values for boundary cases:
* All-ones mask (global scope): area_to_bbox=1.0, largest_frac=1.0,
  compactness=1.0.
* All-zeros mask (degenerate; should not occur in normal Stage B
  output): compactness=0.0 by convention. The downstream scorer reads
  this as "no concentration" which is the right default for a
  degenerate input.

Usage
-----
::

    uv run python scripts/add_mask_compactness.py \\
        mask_artifacts_path=artifacts/mask_artifacts/pico_banana_lab_lpips_ssim_calibrated.parquet \\
        output_path=artifacts/mask_artifacts/pico_banana_lab_lpips_ssim_calibrated_with_compactness.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

log = logging.getLogger(__name__)


def _compute_compactness(mask_path: Path) -> float:
    """Compute compactness ∈ [0, 1] for a binary mask PNG.

    Uses the geometric mean of (area / bbox_area) and (largest_component
    / total_area). See module docstring for rationale.

    Returns 0.0 for empty masks (defensive; should not occur in normal
    Stage B output but is possible after morphological refinement that
    eats a small mask entirely).
    """
    arr = np.asarray(Image.open(mask_path).convert("L"))
    binary = arr > 127
    total_area = int(binary.sum())
    if total_area == 0:
        return 0.0

    # bounding-box area
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    bbox_area = int(bbox_h * bbox_w)
    area_to_bbox = total_area / bbox_area  # always in (0, 1]

    # largest connected component
    # 4-connectivity matches scipy.ndimage default; we don't bother
    # with 8-connectivity since edge effects are below noise for
    # mask_compactness's level of precision.
    labeled, n_components = ndi.label(binary)
    if n_components == 1:
        largest_frac = 1.0
    else:
        # bincount[0] is background (label 0); skip it.
        component_areas = np.bincount(labeled.ravel())[1:]
        largest_frac = float(component_areas.max() / total_area)

    return float(np.sqrt(area_to_bbox * largest_frac))


def _augment_shard(
    in_path: Path,
    out_path: Path,
) -> tuple[int, int]:
    """Read one mask_artifacts shard, compute compactness, write
    augmented shard. Returns (n_rows, n_failed)."""
    df = pd.read_parquet(in_path)
    compactness_values: list[float] = []
    failed = 0
    for mask_path_str in tqdm(
        df["mask_path"].tolist(),
        desc=f"compactness[{in_path.name}]",
        unit="mask",
    ):
        mask_path = Path(mask_path_str)
        try:
            c = _compute_compactness(mask_path)
        except Exception as e:
            log.warning("compactness failed for %s: %s", mask_path, e)
            c = float("nan")
            failed += 1
        compactness_values.append(c)
    df["mask_compactness"] = compactness_values
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df), failed


@hydra.main(version_base=None, config_path="../configs", config_name="add_mask_compactness")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    in_root = Path(cfg.mask_artifacts_path)
    out_root = Path(cfg.output_path)

    shards = sorted(in_root.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {in_root}")
    log.info("processing %d shards from %s", len(shards), in_root)

    total = 0
    total_failed = 0
    for shard in shards:
        out_path = out_root / shard.name
        if out_path.exists() and not cfg.get("overwrite_existing_shards", False):
            log.info("output shard %s already exists; skipping", shard.name)
            continue
        n, failed = _augment_shard(shard, out_path)
        total += n
        total_failed += failed
        log.info(
            "%s: %d rows, %d failed (cumulative: %d / %d)",
            shard.name, n, failed, total_failed, total,
        )

    log.info("compactness augmentation complete: %d rows, %d failed", total, total_failed)


if __name__ == "__main__":
    main()

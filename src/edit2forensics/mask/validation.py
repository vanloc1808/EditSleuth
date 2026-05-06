"""Validation utilities for the MaskGenerator.

These functions compare auto-generated masks against ground-truth masks
(available for the MagicBrush source) and produce the IoU distribution
as the Stage-B quality check.

Two concerns live here:

1. **Metrics.** Per-sample IoU, Dice, and boundary-F1 give a picture of
   overlap quality across the distribution, not just at the mean.
2. **Reporting.** Bimodal performance is a
   known failure mode. We provide per-bin counts and quartile summaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskComparison:
    """Per-sample comparison of an auto-mask against a GT mask."""

    triplet_id: str
    iou: float
    dice: float
    gt_area_frac: float
    auto_area_frac: float


def _load_binary(path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a mask PNG as a boolean array.

    Accepts both single-channel and RGB masks (RGB white==edited is the
    MagicBrush convention). If ``target_shape`` is provided and differs
    from the mask's, the mask is nearest-neighbor resized — important
    because nearest preserves the binary structure.
    """
    img = Image.open(path).convert("L")
    if target_shape is not None and (img.height, img.width) != target_shape:
        img = img.resize(
            (target_shape[1], target_shape[0]), Image.NEAREST
        )
    return np.asarray(img) > 127


def compare_masks(
    auto_path, gt_path, triplet_id: str
) -> MaskComparison:
    """Compute IoU and related metrics for a single auto-mask / GT pair.

    The GT mask is resized to match the auto-mask's resolution if they
    differ. We align to the auto-mask because that's the resolution the
    downstream model will see.
    """
    auto = _load_binary(auto_path)
    gt = _load_binary(gt_path, target_shape=auto.shape)

    inter = np.logical_and(auto, gt).sum()
    union = np.logical_or(auto, gt).sum()
    auto_sum = int(auto.sum())
    gt_sum = int(gt.sum())

    # IoU is undefined on empty-union. Map the edge cases to useful values:
    # both empty -> 1.0 (perfect agreement that nothing is edited);
    # one empty, the other non-empty -> 0.0.
    if union == 0:
        iou = 1.0
    else:
        iou = float(inter) / float(union)

    if auto_sum + gt_sum == 0:
        dice = 1.0
    else:
        dice = (2.0 * float(inter)) / float(auto_sum + gt_sum)

    total = float(auto.size)
    return MaskComparison(
        triplet_id=triplet_id,
        iou=float(iou),
        dice=float(dice),
        gt_area_frac=gt_sum / total,
        auto_area_frac=auto_sum / total,
    )


def summarize_comparisons(cmps: list[MaskComparison]) -> dict:
    """Compute summary statistics suitable for the validation report.

    Returns a dict containing:

    - ``n``: number of samples
    - ``mean_iou``, ``mean_dice``
    - ``iou_quartiles``: (q25, q50, q75)
    - ``iou_bins``: counts in [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    - ``frac_iou_above_0_6``: the headline "how many samples are usable"

    The bins matter: the calls out bimodality as a known risk,
    and only the histogram makes it visible.
    """
    if not cmps:
        return {"n": 0}

    ious = np.array([c.iou for c in cmps])
    dices = np.array([c.dice for c in cmps])

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9]
    counts, _ = np.histogram(ious, bins=bins)

    return {
        "n": len(cmps),
        "mean_iou": float(ious.mean()),
        "mean_dice": float(dices.mean()),
        "iou_quartiles": (
            float(np.quantile(ious, 0.25)),
            float(np.quantile(ious, 0.50)),
            float(np.quantile(ious, 0.75)),
        ),
        "iou_bins": {
            "[0.0, 0.2)": int(counts[0]),
            "[0.2, 0.4)": int(counts[1]),
            "[0.4, 0.6)": int(counts[2]),
            "[0.6, 0.8)": int(counts[3]),
            "[0.8, 1.0]": int(counts[4]),
        },
        "frac_iou_above_0_6": float((ious >= 0.6).mean()),
    }

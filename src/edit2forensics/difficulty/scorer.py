"""Per-triplet difficulty scoring (Stage C).

Given an `EditTriplet` and its `MaskArtifact` from Stage B, compute the
four raw difficulty components and aggregate via a weighted sum. Bin
assignment is deferred to a second pass over the full dataset because
it requires the empirical distribution.

Inputs we DO need image I/O for
-------------------------------
* SSIM — needs both images. Computed here per-triplet.

Inputs we DON'T need image I/O for
----------------------------------
* Perceptual change — read directly from ``MaskArtifact.combined_diff_mean``.
  Stage B already computed this; persisting it there means Stage C runs
  CPU-only, no neural signals, no GPU.
* Locality — derived from ``MaskArtifact.mask_area_frac``.
* Instruction complexity — text-only.

This split is deliberate. Image I/O at Stage C is dominated by SSIM,
which is fast on CPU. The expensive parts (LPIPS, DINOv2) stay in
Stage B where the GPU lives. Stage C is then a CPU-bound batch job
that re-runs cheaply when weights are recalibrated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from edit2forensics.data.difficulty_artifact import DifficultyArtifact
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.difficulty.instruction import (
    InstructionComplexityScorer,
    RuleBasedInstructionComplexity,
)

log = logging.getLogger(__name__)


# Default weights, calibration can override.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "structural_change": 0.30,
    "perceptual_change": 0.30,
    "locality": 0.20,
    "instruction_complexity": 0.20,
}

# Component names in canonical order (used everywhere we serialize a
# weight vector — keeps output deterministic).
_COMPONENT_NAMES = (
    "structural_change",
    "perceptual_change",
    "locality",
    "instruction_complexity",
)


@dataclass
class DifficultyScorerConfig:
    """Tuning knobs for Stage C scoring."""

    weights: Mapping[str, float] = None  # type: ignore[assignment]
    """Per-component weights. Must contain all four canonical component
    names. Defaults to the (0.30, 0.30, 0.20, 0.20).
    Calibration produces alternative weight vectors via the
    `calibrate_difficulty_weights` script."""

    ssim_image_size: int = 256
    """Side length to which images are bilinear-resized before SSIM.
    SSIM is O(H*W); on full-resolution edit images the cost dominates
    the stage. 256 captures structural change at the granularity that
    matters for a difficulty signal without being expensive."""

    ssim_data_range: float = 1.0
    """Passed to ``skimage.metrics.structural_similarity``. Images are
    normalized to ``[0, 1]`` floats, so 1.0 is correct."""

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = dict(_DEFAULT_WEIGHTS)
        else:
            self.weights = dict(self.weights)
        missing = [n for n in _COMPONENT_NAMES if n not in self.weights]
        if missing:
            raise ValueError(
                f"weights missing required components: {missing}; "
                f"need all of {list(_COMPONENT_NAMES)}"
            )
        for n, v in self.weights.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"weight for {n!r} must be numeric, got {type(v)}")


class DifficultyScorer:
    """Compute one ``DifficultyArtifact`` per (triplet, mask_artifact) pair.

    Stateless — instantiate once per process. The ``DifficultyArtifact``
    output does NOT yet have its ``difficulty_bin`` populated; binning
    requires the full dataset's distribution and is filled in by the
    driver script after all rows are scored.
    """

    def __init__(
        self,
        instruction_scorer: InstructionComplexityScorer | None = None,
        config: DifficultyScorerConfig | None = None,
    ) -> None:
        self.instruction_scorer = instruction_scorer or RuleBasedInstructionComplexity()
        self.config = config or DifficultyScorerConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(
        self,
        triplet: EditTriplet,
        mask_artifact: MaskArtifact,
    ) -> DifficultyArtifact:
        """Return a `DifficultyArtifact` with raw components and aggregate.

        ``difficulty_bin`` is set to ``"medium"`` as a placeholder; the
        driver overwrites it during the second pass once the empirical
        distribution is known.
        """
        # --- structural change (SSIM) -------------------------------------
        structural_change = self._compute_structural_change(triplet)

        # --- perceptual change (from Stage B) -----------------------------
        # Bound to [0, 1] defensively; Stage B's combiner output is in
        # [0, 1], but a stale or hand-edited artifact could violate that.
        perceptual_change = float(
            np.clip(mask_artifact.combined_diff_mean, 0.0, 1.0)
        )

        # --- locality (smaller mask area = harder) ------------------------
        locality = float(np.clip(1.0 - mask_artifact.mask_area_frac, 0.0, 1.0))

        # --- instruction complexity ---------------------------------------
        instruction_complexity = float(
            np.clip(self.instruction_scorer.score(triplet.instruction), 0.0, 1.0)
        )

        components = {
            "structural_change": structural_change,
            "perceptual_change": perceptual_change,
            "locality": locality,
            "instruction_complexity": instruction_complexity,
        }

        # --- aggregate ---------------------------------------------------
        weights = self.config.weights
        difficulty_raw = float(sum(weights[n] * components[n] for n in _COMPONENT_NAMES))

        return DifficultyArtifact(
            triplet_id=triplet.triplet_id,
            structural_change=structural_change,
            perceptual_change=perceptual_change,
            locality=locality,
            instruction_complexity=instruction_complexity,
            difficulty_raw=difficulty_raw,
            difficulty_bin="medium",  # placeholder; driver fills in.
            weights=dict(weights),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _compute_structural_change(self, triplet: EditTriplet) -> float:
        """Compute ``1 - SSIM(real, edited)`` on resized images.

        Both images are resized to ``ssim_image_size x ssim_image_size``
        and converted to grayscale. SSIM is run with the default 7x7
        window; resizing caps cost regardless of input resolution.
        """
        size = (self.config.ssim_image_size, self.config.ssim_image_size)
        real_img = Image.open(triplet.real_path).convert("L").resize(size, Image.BILINEAR)
        edited_img = (
            Image.open(triplet.edited_path).convert("L").resize(size, Image.BILINEAR)
        )

        real = np.asarray(real_img, dtype=np.float32) / 255.0
        edited = np.asarray(edited_img, dtype=np.float32) / 255.0

        sim = float(ssim(real, edited, data_range=self.config.ssim_data_range))
        # SSIM ranges in [-1, 1] but on natural images it's nearly always
        # in [0, 1]; clip defensively. Higher SSIM = more similar = LOWER
        # structural change.
        return float(np.clip(1.0 - sim, 0.0, 1.0))


# ---------------------------------------------------------------------- #
# Helpers exposed for the driver and the calibration script
# ---------------------------------------------------------------------- #

def assign_tertile_bins(scores: np.ndarray) -> np.ndarray:
    """Assign empirical tertile bin labels to a 1-D array of raw scores.

    Returns an array of strings (``"easy"`` / ``"medium"`` / ``"hard"``)
    of the same length. Cut points are the 33.3% and 66.7% empirical
    quantiles, so each bin receives ~1/3 of the dataset by construction
    (modulo ties).

    Why empirical, not fixed
    ------------------------
    Explicitly calls for tertiles of the empirical
    distribution rather than fixed thresholds at 0.33/0.66. This makes
    the bin populations independent of the scoring formula's particular
    weights — change the weights and the same triplets will tend to
    occupy the same bins, just with different absolute scores.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if scores.size == 0:
        return np.array([], dtype=object)

    q33 = float(np.quantile(scores, 1.0 / 3.0))
    q66 = float(np.quantile(scores, 2.0 / 3.0))

    bins = np.full(scores.shape, "medium", dtype=object)
    bins[scores <= q33] = "easy"
    bins[scores > q66] = "hard"
    return bins

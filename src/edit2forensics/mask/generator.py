"""Mask generation orchestrator for Stage B.

Given a pair of images (real, edited), produce:

1. A binary manipulation mask (written to disk as PNG).
2. A `MaskArtifact` record with quality metadata for downstream stages.

Pipeline
--------
::

    (real, edited)
        │
        ▼  _align_pair             — size/layout reconciliation
    aligned pair
        │
        ▼  signals[i].compute       — one diff map per signal
    {lab_pixel: H×W, ...}
        │
        ▼  combiner.combine          — collapse signals (max/mean/...)
    combined : H×W in [0,1]
        │
        ▼  threshold_otsu + scope-routing
    binary mask
        │
        ▼  morphological refinement — opening, closing, min-component
    final mask
        │
        ▼  write PNG + emit MaskArtifact

Every stage reports enough state into `MaskArtifact` that a reviewer can
reconstruct why a given mask looks the way it does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.morphology import closing, disk, opening

from edit2forensics.data.mask_artifact import EditScope, MaskArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.mask.combiners import MaxCombiner, SignalCombiner
from edit2forensics.mask.signals import DiffSignal
from edit2forensics.utils.image_io import materialize_image

log = logging.getLogger(__name__)


@dataclass
class MaskGeneratorConfig:
    """Tunable knobs for the mask pipeline.

    Adjust via Hydra at runtime.
    """

    #: Images smaller than this side length are upscaled before processing
    #: (morphology kernels and percentiles misbehave on tiny inputs).
    min_side_for_processing: int = 64

    #: Allow resizing `edited` to match `real` if they differ in shape.
    #: Rare in well-prepared datasets but happens with some sources.
    allow_resize_align: bool = True

    #: Area fraction above which the mask is classified as a global edit.
    global_area_threshold: float = 0.90

    #: Mean diff-map value above which the edit is pre-classified as global
    #: (skips Otsu, which would spuriously subdivide a uniformly-high map).
    #: The LAB pixel signal typically produces mean ~0.5 for clean global
    #: color grades and <0.15 for localized edits, so 0.4 gives a safe margin.
    global_mean_threshold: float = 0.4

    #: Area fraction below which the mask is classified as ambiguous
    #: (too small to be a reliable spatial mask).
    ambiguous_area_threshold: float = 0.005

    #: Structuring-element radius (pixels) for opening/closing. Scaled
    #: lightly with image size at runtime.
    morph_disk_radius: int = 2

    #: Drop connected components smaller than this fraction of image area.
    min_component_area_frac: float = 0.001

    #: When True, skip opening/closing if the initial mask is already
    #: marked as global (whole-image edit — morphology is pointless).
    skip_refinement_on_global: bool = True


class MaskGenerator:
    """Runs the Stage B pipeline on one triplet at a time.

    The generator is intentionally stateless between calls — the same
    instance can process an entire dataset, and Hydra can instantiate it
    once per process.
    """

    def __init__(
        self,
        signals: list[DiffSignal],
        config: MaskGeneratorConfig | None = None,
        combiner: SignalCombiner | None = None,
    ) -> None:
        """
        Parameters
        ----------
        signals
            One or more `DiffSignal` instances. Each produces a dense
            ``(H, W)`` map in ``[0, 1]``.
        config
            Tuning knobs for thresholding, refinement, and scope routing.
        combiner
            Strategy for collapsing per-signal maps into the single
            combined map fed to thresholding. Defaults to ``MaxCombiner()``
            — an edit detectable in any signal is preserved. Use
            ``MeanCombiner()`` to require agreement across signals (see
            combiners.py for the caveat about downstream threshold
            retuning).
        """
        if not signals:
            raise ValueError("at least one DiffSignal must be provided")
        self.signals = signals
        self.config = config or MaskGeneratorConfig()
        self.combiner = combiner or MaxCombiner()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        triplet: EditTriplet,
        mask_out_path: Path,
    ) -> MaskArtifact:
        """Compute mask + artifact for one triplet, persisting the mask PNG.

        Per-triplet path: useful for ad-hoc inspection, tests, and
        configurations where batching wouldn't help. The driver uses
        ``generate_batch`` for the production GPU-batched path.
        """
        real, edited, registration_ok = self._load_and_align(triplet)
        per_signal = {s.name: s.compute(real, edited) for s in self.signals}
        return self._finalize(
            triplet=triplet,
            real=real,
            per_signal=per_signal,
            registration_ok=registration_ok,
            mask_out_path=mask_out_path,
        )

    def generate_batch(
        self,
        triplets: list[EditTriplet],
        mask_out_paths: list[Path],
    ) -> list[MaskArtifact]:
        """Compute masks for a batch of triplets with batched signal calls.

        Numerical equivalence with looping ``generate``: identical, since
        ``DiffSignal.compute_batch`` is required to match looping
        ``compute`` per pair, and the per-triplet finalize phase
        (thresholding, refinement, persistence) is unchanged.

        Implementation
        --------------
        1. Per-triplet load + align (CPU-bound, fast).
        2. ONE call per signal across the whole batch: ``compute_batch``
           lets each signal use whatever batching strategy it wants
           internally (LPIPS does GPU batching by shape group; LAB and
           SSIM fall back to the default loop).
        3. Per-triplet threshold + refine + persist.

        We keep step 1 and step 3 sequential because:
        * Step 1 is I/O-bound, dominated by PIL decode time, not by
          per-call overhead — batching doesn't help.
        * Step 3 produces side effects (writes mask PNGs) that need
          stable per-triplet ordering for parallel-driver correctness.
        """
        if len(triplets) != len(mask_out_paths):
            raise ValueError(
                f"length mismatch: triplets={len(triplets)} "
                f"paths={len(mask_out_paths)}"
            )
        if not triplets:
            return []

        # --- step 1: per-triplet load + align ------------------------------
        loaded: list[tuple[np.ndarray, np.ndarray, bool]] = [
            self._load_and_align(t) for t in triplets
        ]
        reals = [r for r, _, _ in loaded]
        editeds = [e for _, e, _ in loaded]

        # --- step 2: one batched call per signal ---------------------------
        per_signal_batched: dict[str, list[np.ndarray]] = {}
        for sig in self.signals:
            per_signal_batched[sig.name] = sig.compute_batch(reals, editeds)

        # --- step 3: per-triplet finalize ----------------------------------
        artifacts: list[MaskArtifact] = []
        for i, triplet in enumerate(triplets):
            per_signal_i = {
                name: per_signal_batched[name][i]
                for name in per_signal_batched
            }
            artifact = self._finalize(
                triplet=triplet,
                real=reals[i],
                per_signal=per_signal_i,
                registration_ok=loaded[i][2],
                mask_out_path=mask_out_paths[i],
            )
            artifacts.append(artifact)
        return artifacts

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _finalize(
        self,
        triplet: EditTriplet,
        real: np.ndarray,
        per_signal: dict[str, np.ndarray],
        registration_ok: bool,
        mask_out_path: Path,
    ) -> MaskArtifact:
        """Combine + threshold + refine + persist for one triplet.

        Shared between ``generate`` and ``generate_batch``. The signal
        maps in ``per_signal`` have already been computed (via either
        per-pair ``compute`` or batched ``compute_batch``); this method
        is what turns them into an artifact + on-disk mask.
        """
        combined, strongest = self._combine_signals(per_signal)

        # --- threshold + scope routing -------------------------------------
        binary, scope, confidence = self._threshold(combined)

        # --- morphological refinement --------------------------------------
        if not (self.config.skip_refinement_on_global and scope == "global"):
            binary = self._refine(binary, image_shape=real.shape[:2])

        # Recompute area after refinement — it may have shrunk.
        final_area_frac = float(binary.mean())

        # Refinement can push a "local" mask below the ambiguous threshold.
        # Re-classify once after refinement so downstream scope is honest.
        if scope == "local" and final_area_frac < self.config.ambiguous_area_threshold:
            scope = "ambiguous"

        # --- persist mask ---------------------------------------------------
        mask_img = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
        written = materialize_image(mask_img, mask_out_path)

        # Override registration flag into the scope if alignment failed
        # outright so downstream filters can be coarse.
        final_scope: EditScope = (
            "alignment_failed" if not registration_ok else scope
        )

        return MaskArtifact(
            triplet_id=triplet.triplet_id,
            mask_path=written,
            mask_area_frac=final_area_frac,
            edit_scope=final_scope,
            registration_ok=registration_ok,
            confidence=confidence,
            diff_strongest_signal=strongest,
            combined_diff_mean=float(combined.mean()),
        )

    def _load_and_align(
        self, triplet: EditTriplet
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Load both images as uint8 RGB, resolving any shape mismatch.

        Returns ``(real, edited, registration_ok)``. For editing datasets
        the two images almost always share shape; when they don't, we
        bilinear-resize ``edited`` to ``real``'s shape (gated by config).
        Anything more exotic (rotation, crop) is flagged as failed
        registration so callers can skip or deprioritize the triplet.
        """
        real_img = Image.open(triplet.real_path).convert("RGB")
        edited_img = Image.open(triplet.edited_path).convert("RGB")

        # Upscale tiny thumbnails (rare but breaks morphology kernels).
        min_side = self.config.min_side_for_processing
        if min(real_img.size) < min_side:
            scale = min_side / min(real_img.size)
            new_size = (int(real_img.width * scale), int(real_img.height * scale))
            real_img = real_img.resize(new_size, Image.BILINEAR)

        # Shape reconciliation.
        if real_img.size != edited_img.size:
            if not self.config.allow_resize_align:
                log.warning(
                    "mask: shape mismatch for %s (real=%s edited=%s); "
                    "allow_resize_align is False",
                    triplet.triplet_id, real_img.size, edited_img.size,
                )
                real_arr = np.asarray(real_img)
                edited_arr = np.asarray(edited_img.resize(real_img.size, Image.BILINEAR))
                return real_arr, edited_arr, False

            edited_img = edited_img.resize(real_img.size, Image.BILINEAR)

        return np.asarray(real_img), np.asarray(edited_img), True

    def _combine_signals(
        self,
        per_signal: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, str]:
        """Combine per-signal maps via the configured strategy; also
        return the name of the most-contributing signal.

        The combination itself is delegated to ``self.combiner`` (a
        pluggable `SignalCombiner`). The "strongest signal" is reported
        independently — it's computed from each signal's mean value
        across the image, NOT from the combined map, so it remains
        meaningful regardless of which combiner is in use.

        The ``diff_strongest_signal`` field on the resulting
        `MaskArtifact` is what Stage E (the reasoning annotator) uses
        to decide which artifact family to reference in generated
        rationales (e.g., "color inconsistency" if LAB dominates,
        "texture artifact" if LPIPS dominates, "semantic substitution"
        if DINOv2 dominates).
        """
        names = list(per_signal.keys())
        stack = np.stack([per_signal[n] for n in names], axis=0)  # (S, H, W)

        combined = self.combiner.combine(per_signal)

        # "Strongest" signal = the one whose mean contribution to the
        # per-signal stack is largest. Deliberately computed on the
        # pre-combination stack so it stays comparable across combiners.
        mean_contribs = stack.mean(axis=(1, 2))
        strongest = names[int(np.argmax(mean_contribs))]
        return combined, strongest

    def _threshold(
        self, combined: np.ndarray
    ) -> tuple[np.ndarray, EditScope, float]:
        """Otsu threshold the combined map; route degenerate cases.

        Returns ``(binary_mask, scope, confidence)``.

        Scope at this stage is one of ``local`` / ``global`` / ``ambiguous``.
        ``alignment_failed`` is set later by the caller.

        Global-edit detection runs *before* Otsu. This matters because on
        a uniformly-shifted image (style transfer, color grade, global
        tone map), every pixel has high diff but Otsu will still find a
        threshold *within* the high-diff distribution, splitting the
        image into "high" and "higher" bins and producing a meaningless
        mask covering ~20% of the image. Instead, if the mean diff is
        itself near-global, we treat the whole image as edited.
        """
        flat = combined.ravel()
        if float(flat.std()) < 1e-4:
            # Combined map is essentially uniform. Two cases:
            # - uniformly low -> no evidence of any edit (empty/ambiguous).
            # - uniformly high -> every pixel changed by about the same
            #   amount (a clean global edit; unusual but valid).
            mean_val = float(flat.mean())
            if mean_val >= 0.5:
                confidence = float(np.clip(mean_val, 0.0, 1.0))
                return np.ones_like(combined, dtype=bool), "global", confidence
            return np.zeros_like(combined, dtype=bool), "ambiguous", 0.0

        # --- early global detection ---------------------------------------
        # If the image's mean diff is itself high, the change is spread
        # broadly enough that Otsu subdivision would be misleading.
        # Using mean rather than "fraction above 0.5" is robust to
        # saturating edits (clipping at 0/255 reduces per-pixel diff for
        # some pixels even when the edit is clearly global).
        if float(combined.mean()) >= self.config.global_mean_threshold:
            confidence = float(np.clip(combined.mean(), 0.0, 1.0))
            return np.ones_like(combined, dtype=bool), "global", confidence

        try:
            thr = float(threshold_otsu(combined))
        except ValueError:
            # skimage raises when the histogram has <2 bins populated.
            return np.zeros_like(combined, dtype=bool), "ambiguous", 0.0

        binary = combined > thr
        area = float(binary.mean())

        # Confidence = contrast between above- and below-threshold regions.
        # Normalized to [0, 1] by the theoretical max (the combined map
        # is already in [0, 1]).
        above = combined[binary]
        below = combined[~binary]
        if above.size == 0 or below.size == 0:
            confidence = 0.0
        else:
            confidence = float(np.clip(above.mean() - below.mean(), 0.0, 1.0))

        # Otsu can still land >= global_area_threshold on edits that are
        # genuinely localized-but-huge (say, 95% of the image replaced).
        # Still route to global in that case — the spatial mask adds no
        # information when it covers everything.
        if area >= self.config.global_area_threshold:
            return np.ones_like(combined, dtype=bool), "global", confidence

        if area <= self.config.ambiguous_area_threshold:
            return binary, "ambiguous", confidence

        return binary, "local", confidence

    def _refine(self, binary: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        """Morphological cleanup: opening -> closing -> drop small CCs.

        Opening removes speckle; closing fills small holes. The
        connected-component filter then enforces a minimum region size
        so isolated pixels from JPEG noise don't leak into training.
        """
        r = self.config.morph_disk_radius
        selem = disk(r)

        cleaned = opening(binary, selem)
        cleaned = closing(cleaned, selem)

        min_area = max(
            1,
            int(self.config.min_component_area_frac * image_shape[0] * image_shape[1]),
        )

        # Keep components whose area is above `min_area`.
        labeled = cc_label(cleaned, connectivity=2)
        if labeled.max() == 0:
            return cleaned

        keep_mask = np.zeros_like(cleaned, dtype=bool)
        for region in regionprops(labeled):
            if region.area >= min_area:
                keep_mask[labeled == region.label] = True
        return keep_mask

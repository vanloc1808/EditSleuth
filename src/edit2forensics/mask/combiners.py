"""Pluggable strategies for combining multiple `DiffSignal` outputs.

The `MaskGenerator` computes one ``(H, W)`` difference map per signal,
then collapses them into a single combined map before thresholding. The
choice of collapse function — max-pool, mean-pool, or something fancier
— has measurable effects on the final mask quality and is an important
ablation axis.

We expose this choice as a pluggable `SignalCombiner` strategy so that:

* Ablations (max vs. mean; later weighted-mean, geometric-mean, learned
  gating) are config-level changes, not code changes.
* The Hydra run directory records exactly which combiner was used via
  the resolved config — reproducibility is automatic.
* Tests can swap in trivial combiners to isolate orchestrator logic
  from combiner semantics.

Contract
--------
A combiner takes a dict ``{signal_name: (H, W) float32 array in [0, 1]}``
and returns a single ``(H, W) float32 array in [0, 1]``. Output must
remain in the same value range as the inputs — downstream thresholding
(Otsu, mean-based global detection) assumes this.

Interaction with scope routing
------------------------------
MaskGenerator's scope-routing thresholds (``global_mean_threshold=0.4``,
``global_area_threshold=0.90``) were calibrated for max-pool outputs.
Mean-pool compresses the dynamic range — a single-signal global edit
reaching 1.0 becomes mean ≈ 1/S across S signals — so those thresholds
may need retuning when swapping combiners. See docstring on
``MeanCombiner`` for details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SignalCombiner(ABC):
    """Abstract strategy for collapsing per-signal diff maps into one."""

    #: Short identifier used in logs and artifact metadata.
    name: str = ""

    @abstractmethod
    def combine(self, per_signal: dict[str, np.ndarray]) -> np.ndarray:
        """Return the combined ``(H, W)`` map.

        Parameters
        ----------
        per_signal
            Mapping from signal name to its ``(H, W)`` float32 map in
            ``[0, 1]``. At least one entry is guaranteed by the caller.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class MaxCombiner(SignalCombiner):
    """Per-pixel max across signal maps.

    An edit detectable in *any* signal is preserved in the combined map.
    This is the default and the combiner the scope-routing thresholds in
    ``MaskGeneratorConfig`` were calibrated against.

    Behavior
    --------
    * Single-signal agreement produces a clean, high-contrast combined
      map — good for Otsu.
    * Any one signal firing falsely at a location raises that pixel's
      combined value, potentially driving a false positive. If one
      signal is much noisier than the others, max-pool amplifies its
      noise (this matters for ablating noisier learned signals).
    """

    name = "max"

    def combine(self, per_signal: dict[str, np.ndarray]) -> np.ndarray:
        stack = np.stack(list(per_signal.values()), axis=0)  # (S, H, W)
        return stack.max(axis=0)


class MeanCombiner(SignalCombiner):
    """Per-pixel arithmetic mean across signal maps.

    A vote-style combiner: a pixel's combined value is high only when
    multiple signals agree that something has changed there. False
    positives from any single noisy signal are diluted by the others.

    Caveat (important for reproducibility)
    --------------------------------------
    Mean-pool compresses the dynamic range. If one signal saturates at
    1.0 on an edited region but the others report 0.2 each, the max is
    1.0 while the mean is ~0.47 (for S=3). Two downstream knobs become
    sensitive to this:

    * ``global_mean_threshold`` (default 0.4) — the bar at which we
      pre-classify an edit as global. A genuine global edit on one
      signal won't cross 0.4 mean across signals.
    * Otsu's threshold — still data-driven, so less sensitive, but the
      separation between above/below classes narrows.

    In practice, swapping to ``MeanCombiner`` is most useful as an
    *ablation* — for production, retune both thresholds if you keep
    mean-pool.
    """

    name = "mean"

    def combine(self, per_signal: dict[str, np.ndarray]) -> np.ndarray:
        stack = np.stack(list(per_signal.values()), axis=0)  # (S, H, W)
        return stack.mean(axis=0)

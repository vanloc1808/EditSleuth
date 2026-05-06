"""Compute a coarse spatial descriptor for a binary mask.

Returns one of seven categorical labels describing where the edit
is located: ``whole_image``, ``upper_left``, ``upper_right``,
``lower_left``, ``lower_right``, ``centered``, ``scattered``, or
``alignment_failed``. These appear in reasoning chains so the VLM
can learn to reason about edit localization without overfitting to
pixel-precise coordinates.

Algorithm
---------
* If the mask scope is ``"global"``, return ``whole_image``
  (skipping centroid computation since it would be at the image
  center by construction).
* If the mask scope is ``"alignment_failed"``, return
  ``alignment_failed`` (no spatial info available).
* Otherwise, compute the mask's connected components. If there are
  multiple components covering substantial area each, return
  ``scattered``.
* For a single-component mask, compute its center of mass and
  classify into one of five regions: four quadrants or center.
  Quadrants are determined by a margin around the image center;
  if the centroid falls within the central margin (default 25%
  of image width/height), the region is ``centered``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Type alias matches the Literal in reasoning_artifact.py. We keep it
# as a plain string here to avoid a circular import.
SpatialDescriptor = str


def compute_spatial_descriptor(
    mask_path: Path,
    edit_scope: str,
    *,
    central_margin: float = 0.25,
    scattered_threshold: float = 0.3,
) -> SpatialDescriptor:
    """Categorize where an edit is in the image.

    Parameters
    ----------
    mask_path
        Path to the binary mask PNG produced by Stage B.
    edit_scope
        Stage B's scope label: one of ``"global"``, ``"local"``,
        ``"ambiguous"``, ``"alignment_failed"``.
    central_margin
        Fraction of image width/height around the center that counts
        as "centered" rather than a quadrant. 0.25 means the central
        50%×50% box is the center region.
    scattered_threshold
        For multi-component masks, the largest component must hold
        at least this fraction of total mask area to be considered
        "single-component"; below this, the mask is ``scattered``.
        0.3 means a single component dominates if it's ≥30% of the
        total area.

    Returns
    -------
    One of the SpatialDescriptor literals.
    """
    if edit_scope == "global":
        return "whole_image"
    if edit_scope == "alignment_failed":
        return "alignment_failed"

    arr = np.asarray(Image.open(mask_path).convert("L"))
    binary = arr > 127
    total_area = int(binary.sum())
    if total_area == 0:
        # Degenerate mask — shouldn't occur from Stage B normally.
        # Treat as alignment_failed since we have no spatial signal.
        return "alignment_failed"

    # Connected components: detect "scattered" before computing centroid.
    labeled, n_components = ndi.label(binary)
    if n_components > 1:
        component_areas = np.bincount(labeled.ravel())[1:]
        largest_frac = component_areas.max() / total_area
        if largest_frac < scattered_threshold:
            return "scattered"

    # Single-component (or one-dominant-component) mask: use centroid.
    centroid_row, centroid_col = ndi.center_of_mass(binary)
    h, w = binary.shape
    cy_norm = centroid_row / h    # 0 = top, 1 = bottom
    cx_norm = centroid_col / w    # 0 = left, 1 = right

    # Center region: a (2 * central_margin)-wide strip around 0.5.
    cm = central_margin
    in_center_h = (0.5 - cm) <= cy_norm <= (0.5 + cm)
    in_center_w = (0.5 - cm) <= cx_norm <= (0.5 + cm)
    if in_center_h and in_center_w:
        return "centered"

    # Quadrant classification.
    upper = cy_norm < 0.5
    left = cx_norm < 0.5
    if upper and left:
        return "upper_left"
    if upper and not left:
        return "upper_right"
    if not upper and left:
        return "lower_left"
    return "lower_right"


# Human-readable phrasings for use in chain prose.
SPATIAL_DESCRIPTOR_PROSE: dict[str, str] = {
    "whole_image": "spans the entire image",
    "upper_left": "is concentrated in the upper-left region",
    "upper_right": "is concentrated in the upper-right region",
    "lower_left": "is concentrated in the lower-left region",
    "lower_right": "is concentrated in the lower-right region",
    "centered": "is centered in the image",
    "scattered": "is scattered across multiple regions of the image",
    "alignment_failed": "could not be localized due to image alignment failure",
}

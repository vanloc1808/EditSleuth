"""Utilities for writing in-memory PIL images to disk deterministically.

HuggingFace-backed adapters receive images as PIL objects. Our pipeline
contract uses `pathlib.Path`, so every such adapter materializes images
once to a cache directory. Keeping that logic here (rather than in each
adapter) avoids duplication and ensures consistent behavior.

Design
------
* Filenames are constructed from the caller's stable key (triplet_id +
  role, e.g. ``magicbrush_dev_000000001_t01__real.png``). The caller owns
  the naming; this module just writes bytes.
* Writes are atomic: we write to a ``.partial`` sibling and rename on
  success. A crashed run cannot leave a half-written PNG in place.
* If the target already exists and ``skip_if_exists=True``, no work is
  done — reruns are cheap.
* Format is always PNG. JPEG would be lossy and the storage savings are
  not worth mask-edge degradation or re-compression artifacts that would
  contaminate later forensic analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def materialize_image(
    img: Image.Image,
    dest: Path,
    skip_if_exists: bool = True,
) -> Path:
    """Write a PIL image to `dest` as PNG, atomically.

    Parameters
    ----------
    img
        PIL Image. May be in any mode; mask images should be converted to
        ``L`` by the caller before invoking this function so the file on
        disk is a clean single-channel mask.
    dest
        Absolute path where the PNG will be written. Parent directories
        are created as needed.
    skip_if_exists
        If True and `dest` already exists with non-zero size, return it
        without rewriting.

    Returns
    -------
    The resolved absolute path of the written file.
    """
    dest = dest.resolve()
    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        img.save(tmp, format="PNG")
        tmp.rename(dest)
    except Exception:
        # Clean up partial file on any failure so retries work.
        tmp.unlink(missing_ok=True)
        raise
    return dest


def magicbrush_mask_to_binary(mask_img: Image.Image) -> Image.Image:
    """Convert a MagicBrush mask image to a clean binary (L-mode) PNG.

    MagicBrush masks do NOT follow the "white pixel = edited region"
    convention one might assume. Inspection of a sample shows the mask
    is an RGB image in which the *edited region is painted pure black*
    ``[0, 0, 0]`` over the original image content (which is retained in
    the unedited region). A naive ``.convert("L")`` followed by
    ``> 127`` thresholding therefore inverts the semantics AND mislabels
    any naturally dark pixels (shadows, dark objects) as edited.

    We also defensively handle two other encodings that MagicBrush
    preview materials hint at:

    - **RGBA**: transparent (alpha==0) marks the edited region.
    - **L**: a plain binary mask, white==edited.

    The returned image is always mode ``"L"`` with two values: 0
    (unedited) and 255 (edited). This is the convention the Stage B
    validation code expects (``pixel > 127`` means edited).

    Parameters
    ----------
    mask_img
        The raw PIL image from the dataset.

    Returns
    -------
    A new PIL ``"L"`` mode image of the same resolution, binary-valued.
    """
    arr = np.asarray(mask_img)

    if mask_img.mode == "RGBA":
        # Alpha=0 (transparent) encodes the edited region.
        alpha = arr[..., 3]
        edited = alpha == 0
    elif mask_img.mode in ("RGB", "P"):
        # Pure-black pixels encode the edited region. Use ALL channels
        # because single-channel ==0 would misfire on colored pixels
        # that happen to have one zero channel (saturated reds, etc.).
        if mask_img.mode == "P":
            # Palette images — expand to RGB before the check.
            arr = np.asarray(mask_img.convert("RGB"))
        edited = np.all(arr == 0, axis=-1)
    elif mask_img.mode in ("L", "1"):
        # Already a single-channel mask. Assume white==edited (the
        # convention used by our own generated masks).
        edited = arr > 127
    else:
        # Unknown mode — fall back to RGB interpretation rather than
        # silently producing something wrong. The warning will surface
        # in ingestion logs if this ever fires.
        log.warning(
            "magicbrush_mask_to_binary: unexpected mask mode %r; "
            "falling back to RGB all-zero extraction",
            mask_img.mode,
        )
        arr = np.asarray(mask_img.convert("RGB"))
        edited = np.all(arr == 0, axis=-1)

    out = (edited.astype(np.uint8) * 255)
    return Image.fromarray(out, mode="L")

"""Shared structural-change computation for V1 and V2 difficulty scorers.

Both scorers compute ``1 - SSIM(real, edited)`` identically; this module
factors out the math so V1 and V2 stay byte-identical on this component
even if one diverges from the other in the future.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from edit2forensics.data.triplet import EditTriplet


def compute_structural_change(
    triplet: EditTriplet,
    ssim_image_size: int = 256,
    ssim_data_range: float = 1.0,
) -> float:
    """Compute ``1 - SSIM(real, edited)`` on resized grayscale images.

    Returns a value in ``[0, 1]``. Higher = more structural change =
    higher difficulty contribution.
    """
    size = (ssim_image_size, ssim_image_size)
    real_img = Image.open(triplet.real_path).convert("L").resize(size, Image.BILINEAR)
    edited_img = (
        Image.open(triplet.edited_path).convert("L").resize(size, Image.BILINEAR)
    )

    real = np.asarray(real_img, dtype=np.float32) / 255.0
    edited = np.asarray(edited_img, dtype=np.float32) / 255.0

    sim = float(ssim(real, edited, data_range=ssim_data_range))
    return float(np.clip(1.0 - sim, 0.0, 1.0))

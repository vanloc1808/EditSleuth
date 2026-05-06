"""Tests for `magicbrush_mask_to_binary`.

This utility exists because MagicBrush's mask encoding is counter-intuitive
(edited = pure-black RGB over the target image's content, not a plain
binary image). Getting it wrong silently poisons every downstream IoU
against GT. These tests pin the correct behavior for each encoding
variant and document the bug that motivated them.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from edit2forensics.utils.image_io import magicbrush_mask_to_binary


def _assert_binary_matches(out: Image.Image, expected_edited: np.ndarray) -> None:
    """Assert `out` is an L-mode binary image with 255 on `expected_edited`."""
    assert out.mode == "L"
    arr = np.asarray(out)
    assert set(np.unique(arr).tolist()) <= {0, 255}
    assert np.array_equal(arr > 127, expected_edited)


# ----------------------------------------------------------------------
# RGB case (the one from the real MagicBrush dataset)
# ----------------------------------------------------------------------

def test_rgb_mask_extracts_all_zero_pixels():
    """Pure-black RGB pixels mark the edited region; everything else is
    treated as unedited regardless of its brightness."""
    # Background with varied (but non-zero) colors to exercise the rule:
    # "all three channels == 0" rather than per-channel.
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, :, 0] = 50   # R
    arr[:, :, 1] = 30   # G
    arr[:, :, 2] = 60   # B
    # Paint a 4x4 edited region as pure black.
    arr[4:8, 4:8] = [0, 0, 0]

    expected = np.zeros((16, 16), dtype=bool)
    expected[4:8, 4:8] = True

    out = magicbrush_mask_to_binary(Image.fromarray(arr, mode="RGB"))
    _assert_binary_matches(out, expected)


def test_rgb_mask_does_not_confuse_single_zero_channel():
    """A pixel with channel-0 = 0 but other channels non-zero is NOT
    edited — this is the critical distinction that a per-channel test
    would get wrong."""
    arr = np.full((8, 8, 3), fill_value=[0, 150, 200], dtype=np.uint8)  # cyan-ish, R=0
    # Only the true edited region has all three channels zero.
    arr[0:2, 0:2] = [0, 0, 0]

    expected = np.zeros((8, 8), dtype=bool)
    expected[0:2, 0:2] = True

    out = magicbrush_mask_to_binary(Image.fromarray(arr, mode="RGB"))
    _assert_binary_matches(out, expected)


def test_rgb_mask_does_not_mislabel_dark_pixels_as_edited():
    """Regression test for the original bug. Dark-but-not-zero pixels
    (e.g. shadows, ``[10, 5, 8]``) must not be treated as edited."""
    arr = np.full((8, 8, 3), fill_value=[10, 5, 8], dtype=np.uint8)  # very dark
    # No truly edited pixels.
    out = magicbrush_mask_to_binary(Image.fromarray(arr, mode="RGB"))
    arr_out = np.asarray(out)
    assert arr_out.sum() == 0, (
        "dark but non-zero pixels should NOT be classified as edited"
    )


# ----------------------------------------------------------------------
# RGBA case (defensive — some MagicBrush previews encode via alpha)
# ----------------------------------------------------------------------

def test_rgba_mask_uses_alpha_channel():
    """Alpha==0 (transparent) means edited."""
    arr = np.zeros((12, 12, 4), dtype=np.uint8)
    arr[..., :3] = [100, 150, 200]  # RGB filled
    arr[..., 3] = 255               # fully opaque by default
    # A transparent patch marks the edited region.
    arr[3:7, 3:7, 3] = 0

    expected = np.zeros((12, 12), dtype=bool)
    expected[3:7, 3:7] = True

    out = magicbrush_mask_to_binary(Image.fromarray(arr, mode="RGBA"))
    _assert_binary_matches(out, expected)


def test_rgba_ignores_rgb_when_alpha_present():
    """If alpha is present it's authoritative — pure-black RGB pixels
    with alpha==255 should NOT be flagged."""
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[..., 3] = 255  # fully opaque
    # Place black RGB pixels that are NOT transparent — these are just
    # ordinary dark content, not an edit mark.
    arr[1:3, 1:3, :3] = 0

    out = magicbrush_mask_to_binary(Image.fromarray(arr, mode="RGBA"))
    assert np.asarray(out).sum() == 0


# ----------------------------------------------------------------------
# L case (our own auto-masks round-trip through here in principle)
# ----------------------------------------------------------------------

def test_l_mask_uses_white_as_edited():
    """Plain binary mask — the convention our own Stage B masks use."""
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[2:5, 2:5] = 255

    expected = arr > 127

    out = magicbrush_mask_to_binary(Image.fromarray(arr, mode="L"))
    _assert_binary_matches(out, expected)


# ----------------------------------------------------------------------
# Adapter-level integration: materialized on-disk mask has correct semantics
# ----------------------------------------------------------------------

def test_adapter_materializes_semantic_mask(tmp_path, monkeypatch):
    """End-to-end: feed an RGB-encoded magicbrush mask through the
    adapter and verify the on-disk PNG is binary with edited==255."""
    import datasets as hf_datasets
    from edit2forensics.adapters.magicbrush import MagicBrushAdapter

    # Construct a MagicBrush-style mask: pink-ish background with a
    # pure-black edited patch.
    mask_arr = np.full((16, 16, 3), fill_value=[200, 100, 120], dtype=np.uint8)
    mask_arr[2:6, 2:6] = [0, 0, 0]
    raw_mask = Image.fromarray(mask_arr, mode="RGB")

    src = Image.new("RGB", (16, 16), (10, 10, 10))
    tgt = Image.new("RGB", (16, 16), (20, 20, 20))

    records = [{
        "img_id": "test", "turn_index": 1,
        "source_img": src, "mask_img": raw_mask,
        "instruction": "fixture", "target_img": tgt,
    }]
    monkeypatch.setattr(
        hf_datasets, "load_dataset",
        lambda repo, split=None, cache_dir=None: records,
    )

    adapter = MagicBrushAdapter(cache_root=tmp_path, split="dev")
    triplet = list(adapter.ingest())[0]
    assert triplet.provided_mask_path is not None

    # The written-to-disk mask must be binary, with the edited patch == 255.
    on_disk = np.asarray(Image.open(triplet.provided_mask_path))
    assert on_disk.ndim == 2, "mask on disk should be single-channel"
    assert set(np.unique(on_disk).tolist()) <= {0, 255}

    # Compare to expected binary.
    expected = np.zeros((16, 16), dtype=bool)
    expected[2:6, 2:6] = True
    assert np.array_equal(on_disk > 127, expected)


def test_adapter_magicbrush_mask_is_comparable_with_generated(tmp_path, monkeypatch):
    """The GT mask extracted by the adapter must be directly comparable
    to masks our own Stage B generates (same convention: L-mode,
    255==edited, 0==unedited). If we feed a manufactured pair where the
    edited region is known, a fresh auto-mask and the adapter's GT mask
    should both identify the same region — giving a high IoU."""
    import datasets as hf_datasets
    from edit2forensics.adapters.magicbrush import MagicBrushAdapter
    from edit2forensics.mask.generator import MaskGenerator
    from edit2forensics.mask.signals import LabPixelDiff
    from edit2forensics.mask.validation import compare_masks

    # Source: smooth gradient; target: same with a recolored patch.
    h = w = 96
    src_arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        src_arr[y] = [80 + y, 100, 150 - y // 2]
    tgt_arr = src_arr.copy()
    tgt_arr[20:50, 20:50] = [220, 30, 30]  # the "edit"

    # Build a MagicBrush-style mask: black over the edited area on a
    # tinted background.
    mask_arr = np.full((h, w, 3), fill_value=[200, 100, 120], dtype=np.uint8)
    mask_arr[20:50, 20:50] = [0, 0, 0]

    records = [{
        "img_id": "semantic_check", "turn_index": 1,
        "source_img": Image.fromarray(src_arr, mode="RGB"),
        "mask_img": Image.fromarray(mask_arr, mode="RGB"),
        "instruction": "Red square in top-left region.",
        "target_img": Image.fromarray(tgt_arr, mode="RGB"),
    }]
    monkeypatch.setattr(
        hf_datasets, "load_dataset",
        lambda repo, split=None, cache_dir=None: records,
    )

    adapter = MagicBrushAdapter(cache_root=tmp_path, split="dev")
    triplet = list(adapter.ingest())[0]

    # Generate our own auto-mask and compare to the adapter-extracted GT.
    gen = MaskGenerator(signals=[LabPixelDiff()])
    auto_mask_path = tmp_path / "auto.png"
    gen.generate(triplet, auto_mask_path)

    cmp = compare_masks(auto_mask_path, triplet.provided_mask_path, triplet.triplet_id)
    # Semantics agree -> high IoU. The old buggy code produced IoU ~0.0 here.
    assert cmp.iou > 0.7, (
        f"GT and auto masks should broadly agree on a clean synthetic edit; "
        f"got IoU={cmp.iou:.3f}"
    )

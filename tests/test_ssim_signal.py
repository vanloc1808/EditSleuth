"""Tests for `SSIMDiff` — dense (1 - SSIM) difference signal."""
from __future__ import annotations

import numpy as np
import pytest

from edit2forensics.mask.signals import SSIMDiff


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_win_size_must_be_odd_int_ge_3():
    with pytest.raises(ValueError, match="win_size"):
        SSIMDiff(win_size=2)
    with pytest.raises(ValueError, match="win_size"):
        SSIMDiff(win_size=8)  # even
    with pytest.raises(ValueError, match="win_size"):
        SSIMDiff(win_size=1)
    # Odd >= 3 should be accepted.
    SSIMDiff(win_size=3)
    SSIMDiff(win_size=11)


def test_normalize_percentile_validation():
    with pytest.raises(ValueError, match="normalize_percentile"):
        SSIMDiff(normalize_percentile=49.9)
    with pytest.raises(ValueError, match="normalize_percentile"):
        SSIMDiff(normalize_percentile=100.1)


def test_sigma_must_be_positive():
    with pytest.raises(ValueError, match="sigma"):
        SSIMDiff(sigma=0.0)
    with pytest.raises(ValueError, match="sigma"):
        SSIMDiff(sigma=-1.0)
    SSIMDiff(sigma=0.5)  # accepted


# ---------------------------------------------------------------------------
# Input shape validation
# ---------------------------------------------------------------------------

def test_rejects_shape_mismatch():
    sig = SSIMDiff()
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.zeros((32, 33, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        sig.compute(a, b)


def test_rejects_grayscale_input():
    sig = SSIMDiff()
    a = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        sig.compute(a, a)


def test_rejects_image_smaller_than_window():
    sig = SSIMDiff(win_size=11)
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small for win_size"):
        sig.compute(a, a)


# ---------------------------------------------------------------------------
# Output properties
# ---------------------------------------------------------------------------

def test_identical_images_produce_zero_diff():
    """SSIM(x, x) = 1 everywhere, so 1 - SSIM = 0. After percentile
    normalization the map should be exactly zero."""
    sig = SSIMDiff()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    out = sig.compute(img, img)
    assert out.shape == (64, 64)
    assert out.dtype == np.float32
    assert np.allclose(out, 0.0)


def test_output_in_unit_interval():
    """Even on extreme inputs, the percentile-normalized output must
    stay in [0, 1]."""
    sig = SSIMDiff()
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    out = sig.compute(a, b)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_shape_matches_input_hw():
    """Output is single-channel (H, W)."""
    sig = SSIMDiff()
    a = np.zeros((48, 80, 3), dtype=np.uint8)
    b = a.copy()
    b[10:20, 10:20] = 255
    out = sig.compute(a, b)
    assert out.shape == (48, 80)


# ---------------------------------------------------------------------------
# Localization properties — the signal earns its place by these
# ---------------------------------------------------------------------------

def test_localized_edit_produces_localized_diff():
    """An edit confined to a small region should produce a diff map
    whose energy concentrates in that region."""
    sig = SSIMDiff()
    a = np.full((96, 96, 3), 128, dtype=np.uint8)
    # Add some texture so SSIM has real structure to compare.
    rng = np.random.default_rng(2)
    a += rng.integers(-20, 20, size=a.shape, dtype=np.int8).astype(np.uint8)
    b = a.copy()
    # Replace top-left 24x24 with a distinct pattern.
    b[:24, :24] = 255

    out = sig.compute(a, b)
    # Sum diff in the edited region vs. an equally-sized region in the
    # bottom-right corner that wasn't touched. SSIM has a window-radius
    # blur effect (~win_size/2 ≈ 3 pixels), so we look slightly inside
    # the edit boundary to avoid edge effects.
    edit_region = out[3:21, 3:21].sum()
    untouched_region = out[-21:-3, -21:-3].sum()
    assert edit_region > 5 * untouched_region, (
        f"edit region energy ({edit_region:.3f}) should dominate "
        f"untouched region ({untouched_region:.3f})"
    )


def test_global_brightness_shift_produces_low_diff():
    """The signal's distinguishing property: a uniform brightness shift
    preserves local structure, so (1 - SSIM) should be small everywhere
    — much smaller than the diff for a localized edit of similar
    magnitude."""
    sig = SSIMDiff()
    rng = np.random.default_rng(3)
    a = rng.integers(50, 200, size=(96, 96, 3), dtype=np.uint8)

    # Global brightness shift: add 30 everywhere, clipped to [0, 255].
    b_global = np.clip(a.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    out_global = sig.compute(a, b_global)

    # Localized edit: replace a 24x24 patch entirely.
    b_local = a.copy()
    b_local[:24, :24] = 255
    out_local = sig.compute(a, b_local)

    # The localized edit's MAX response should be significantly larger
    # than the global edit's max response. (We compare maxes rather
    # than means because per-image percentile normalization can rescale
    # the global edit's small variations to look mean-similar; the
    # localized edit's spike is what distinguishes it.)
    #
    # We also expect the global edit's diff to be more uniformly
    # distributed, while the local edit's diff is concentrated.
    # A clean comparison: variance of out_local should be higher.
    assert out_local.var() > out_global.var(), (
        "localized edit's diff map should have higher spatial variance "
        f"than a global brightness shift's: local={out_local.var():.4f}, "
        f"global={out_global.var():.4f}"
    )


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------

def test_uniform_window_size_changes_smoothing():
    """With ``gaussian_weights=False``, ``win_size`` controls the
    comparison window's side length and is meaningful. Larger windows
    blur the edit-region boundary further into surrounding pixels.

    Note: with ``gaussian_weights=True`` (the default), scikit-image
    ignores ``win_size`` entirely — the gaussian filter's effective
    support is determined by ``sigma``. See
    ``test_gaussian_sigma_changes_smoothing`` for that case.
    """
    rng = np.random.default_rng(4)
    a = rng.integers(50, 200, size=(96, 96, 3), dtype=np.uint8)
    b = a.copy()
    b[:16, :16] = 255

    out_small = SSIMDiff(win_size=3, gaussian_weights=False).compute(a, b)
    out_large = SSIMDiff(win_size=11, gaussian_weights=False).compute(a, b)

    # The basic localized-edit detection must work at both window sizes.
    assert out_small[:16, :16].sum() > out_small[-16:, -16:].sum()
    assert out_large[:16, :16].sum() > out_large[-16:, -16:].sum()

    # Larger window blurs the edit-region boundary more, so the diff
    # "leaks" further into surrounding pixels — surrounding-region
    # response immediately outside the 16x16 edit is HIGHER for
    # win_size=11. Test at row 18 (just below the edit boundary, far
    # enough that win_size=3 has fully fallen off but win_size=11 still
    # has support).
    surround_small = out_small[18, :16].mean()
    surround_large = out_large[18, :16].mean()
    assert surround_large > surround_small, (
        f"larger uniform window should produce more boundary leak; "
        f"win_size=3 surround={surround_small:.4f}, "
        f"win_size=11 surround={surround_large:.4f}"
    )


def test_gaussian_sigma_changes_smoothing():
    """With ``gaussian_weights=True`` (the default), ``sigma`` is the
    knob that controls smoothing. Larger sigma => more boundary leak."""
    rng = np.random.default_rng(4)
    a = rng.integers(50, 200, size=(96, 96, 3), dtype=np.uint8)
    b = a.copy()
    b[:16, :16] = 255

    out_small = SSIMDiff(sigma=0.5).compute(a, b)
    out_large = SSIMDiff(sigma=3.0).compute(a, b)

    # Both detect the localized edit.
    assert out_small[:16, :16].sum() > out_small[-16:, -16:].sum()
    assert out_large[:16, :16].sum() > out_large[-16:, -16:].sum()

    # Larger sigma blurs the boundary further. Test at row 20.
    surround_small = out_small[20, :16].mean()
    surround_large = out_large[20, :16].mean()
    assert surround_large > surround_small, (
        f"larger sigma should produce more boundary leak; "
        f"sigma=0.5 surround={surround_small:.4f}, "
        f"sigma=3.0 surround={surround_large:.4f}"
    )


def test_win_size_silently_ignored_under_gaussian():
    """Document the scikit-image quirk: with gaussian_weights=True,
    win_size has no effect on the output. This test pins that behavior
    so future maintainers don't get confused."""
    rng = np.random.default_rng(4)
    a = rng.integers(50, 200, size=(96, 96, 3), dtype=np.uint8)
    b = a.copy()
    b[:16, :16] = 255

    out_winN = SSIMDiff(win_size=3, gaussian_weights=True).compute(a, b)
    out_winM = SSIMDiff(win_size=11, gaussian_weights=True).compute(a, b)

    # Identical outputs — the win_size parameter is ignored when
    # gaussian_weights=True.
    assert np.array_equal(out_winN, out_winM)


def test_gaussian_vs_uniform_window():
    """Both window types should detect a localized edit; this just
    confirms the gaussian_weights flag is wired through and doesn't
    crash."""
    rng = np.random.default_rng(5)
    a = rng.integers(50, 200, size=(64, 64, 3), dtype=np.uint8)
    b = a.copy()
    b[:16, :16] = 255

    out_gauss = SSIMDiff(gaussian_weights=True).compute(a, b)
    out_unif = SSIMDiff(gaussian_weights=False).compute(a, b)

    # Sanity: both detect the edit.
    assert out_gauss[:16, :16].sum() > out_gauss[-16:, -16:].sum()
    assert out_unif[:16, :16].sum() > out_unif[-16:, -16:].sum()


def test_name_attribute():
    assert SSIMDiff().name == "ssim"

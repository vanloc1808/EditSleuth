"""Tests for `DINOv2Diff`.

These tests deliberately do NOT require torch, torchvision, DINOv2 weights,
or a GPU. The user will exercise the real code path on GPU separately.

Strategy
--------
We stub the lazy-imported `torch` and `torchvision` modules via
`sys.modules` injection, and stub `torch.hub.load` to return a fake DINOv2
model that:

* Accepts an NCHW tensor of shape (1, 3, image_size, image_size).
* Records the input shapes it saw (so we can pin preprocessing).
* Returns a dict with `x_norm_patchtokens` of shape (1, N, D) where
  N = (image_size // 14)^2 and D is a fixed feature dim.

We verify:

* Output shape matches the input (H, W) — NOT image_size — because the
  compute path upsamples the patch grid back to native resolution.
* Output dtype is float32 and values lie in [0, 1].
* Preprocessing produces (1, 3, image_size, image_size) regardless of
  the caller's input shape.
* Config validation (bad backbone, non-multiple-of-14 image_size, etc.).
* Lazy-import failure raises a clean ImportError when torch is absent.

We do NOT exercise real feature geometry. That's the user's GPU run.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lazy-import failure path
# ---------------------------------------------------------------------------

def test_dinov2_missing_deps_raises_clean_import_error(monkeypatch):
    """Missing torch/torchvision must produce a targeted ImportError
    rather than an obscure failure deep in the forward call."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "torchvision", raising=False)

    class _BlockingFinder:
        def find_module(self, name, path=None):
            if name in ("torch", "torchvision"):
                raise ImportError(f"simulated: {name} not available")
            return None

        def find_spec(self, name, path=None, target=None):
            if name in ("torch", "torchvision"):
                raise ImportError(f"simulated: {name} not available")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockingFinder()] + sys.meta_path)
    monkeypatch.delitem(sys.modules, "edit2forensics.mask.signals", raising=False)

    from edit2forensics.mask.signals import DINOv2Diff

    with pytest.raises(ImportError, match="torch.*torchvision|torchvision.*torch"):
        DINOv2Diff()


# ---------------------------------------------------------------------------
# Fake-backend plumbing
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Array wrapper implementing the torch.Tensor subset used by DINOv2Diff."""

    def __init__(self, arr: np.ndarray):
        self._a = np.ascontiguousarray(arr)

    # ---- shape / views -------------------------------------------------
    @property
    def shape(self):
        return self._a.shape

    def permute(self, *dims):
        return _FakeTensor(np.transpose(self._a, dims))

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self._a, axis=dim))

    def squeeze(self, dim):
        return _FakeTensor(np.squeeze(self._a, axis=dim))

    def contiguous(self):
        return _FakeTensor(np.ascontiguousarray(self._a))

    def view(self, *shape):
        return _FakeTensor(self._a.reshape(shape))

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    # ---- elementwise ---------------------------------------------------
    def __truediv__(self, other):
        other_a = other._a if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._a / other_a)

    def __sub__(self, other):
        other_a = other._a if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._a - other_a)

    def __mul__(self, other):
        other_a = other._a if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._a * other_a)

    def __rsub__(self, other):
        return _FakeTensor(other - self._a)

    # ---- reductions ----------------------------------------------------
    def sum(self, dim):
        # Match torch signature: sum(dim)
        return _FakeTensor(self._a.sum(axis=dim))


def _install_stubs(monkeypatch, feature_dim: int = 384):
    """Install fake torch + torchvision so DINOv2Diff can construct and run."""
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "float32"
    torch_mod.Tensor = _FakeTensor

    def _from_numpy(arr):
        return _FakeTensor(arr.astype(np.float32))

    def _tensor(seq, device=None):
        return _FakeTensor(np.array(seq, dtype=np.float32))

    torch_mod.from_numpy = _from_numpy
    torch_mod.tensor = _tensor

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch_mod.no_grad = _NoGrad

    # torch.nn.functional.interpolate
    def _interpolate(t, size=None, mode="bilinear", align_corners=False):
        arr = t._a
        # Preserve leading dims, resize last two to `size` with a naive fill.
        lead = arr.shape[:-2]
        new_h, new_w = size
        out = np.full(lead + (new_h, new_w), float(arr.mean()), dtype=arr.dtype)
        return _FakeTensor(out)

    # torch.nn.functional.normalize: L2-normalize along `dim`.
    def _normalize(t, p=2, dim=-1, eps=1e-8):
        arr = t._a
        # Only p=2 is used by the signal; no need to implement others.
        assert p == 2, f"fake F.normalize only supports p=2, got p={p}"
        norm = np.linalg.norm(arr, ord=2, axis=dim, keepdims=True)
        return _FakeTensor(arr / np.maximum(norm, eps))

    nn_mod = types.ModuleType("torch.nn")
    func_mod = types.ModuleType("torch.nn.functional")
    func_mod.interpolate = _interpolate
    func_mod.normalize = _normalize
    nn_mod.functional = func_mod
    torch_mod.nn = nn_mod

    # torch.hub.load returns a fake DINOv2 model.
    class _FakeDinov2:
        def __init__(self, feature_dim: int):
            self.feature_dim = feature_dim
            self.forward_calls = []

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def forward_features(self, x):
            # Record the input shape so tests can pin preprocessing.
            self.forward_calls.append(x.shape)
            n, c, h, w = x._a.shape
            patch = 14
            assert h % patch == 0 and w % patch == 0, (
                f"fake DINOv2 expects input divisible by 14, got {(h, w)}"
            )
            g_h, g_w = h // patch, w // patch
            n_tokens = g_h * g_w
            # Return L2-normalized patch tokens so cosine sim = dot product.
            # Use deterministic values so tests are reproducible.
            raw = np.ones((n, n_tokens, self.feature_dim), dtype=np.float32)
            norm = np.linalg.norm(raw, axis=-1, keepdims=True)
            patchtokens = raw / np.maximum(norm, 1e-8)
            return {
                "x_norm_patchtokens": _FakeTensor(patchtokens),
                "x_norm_clstoken": _FakeTensor(np.zeros((n, self.feature_dim), np.float32)),
            }

    hub_mod = types.ModuleType("torch.hub")

    def _hub_load(repo, model_name, verbose=False):
        # Capture the args so tests can assert on them.
        _hub_load.last_call = (repo, model_name)
        return _FakeDinov2(feature_dim=feature_dim)

    _hub_load.last_call = None
    hub_mod.load = _hub_load
    torch_mod.hub = hub_mod

    # torchvision only needs to import successfully.
    torchvision_mod = types.ModuleType("torchvision")

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", func_mod)
    monkeypatch.setitem(sys.modules, "torch.hub", hub_mod)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision_mod)
    monkeypatch.delitem(sys.modules, "edit2forensics.mask.signals", raising=False)

    return hub_mod


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_dinov2_rejects_bad_backbone(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    with pytest.raises(ValueError, match="backbone must be"):
        DINOv2Diff(backbone="clip_vit_b_32")


def test_dinov2_rejects_non_multiple_of_14_image_size(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    with pytest.raises(ValueError, match="multiple of 14"):
        DINOv2Diff(image_size=256)  # 256 / 14 is not integer


def test_dinov2_rejects_tiny_image_size(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    # 28 is a multiple of 14 but too small — produces a 2x2 grid.
    with pytest.raises(ValueError, match="image_size must be >="):
        DINOv2Diff(image_size=28)


def test_dinov2_rejects_bad_percentile(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    with pytest.raises(ValueError, match="normalize_percentile"):
        DINOv2Diff(normalize_percentile=40)


# ---------------------------------------------------------------------------
# Shape / dtype / contract checks
# ---------------------------------------------------------------------------

def test_output_shape_matches_original_input(monkeypatch):
    """compute((H, W, 3)) must return (H, W), independent of image_size."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff

    sig = DINOv2Diff(image_size=224)
    real = np.full((96, 120, 3), 100, dtype=np.uint8)
    edited = np.full((96, 120, 3), 150, dtype=np.uint8)

    out = sig.compute(real, edited)
    assert out.shape == (96, 120)
    assert out.dtype == np.float32


def test_output_shape_on_square_input(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    sig = DINOv2Diff(image_size=224)
    real = np.zeros((256, 256, 3), dtype=np.uint8)
    edited = np.zeros((256, 256, 3), dtype=np.uint8)
    out = sig.compute(real, edited)
    assert out.shape == (256, 256)


def test_output_values_in_zero_to_one(monkeypatch):
    """Normalization must clip into [0, 1]."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    sig = DINOv2Diff(image_size=224)
    real = np.zeros((64, 64, 3), dtype=np.uint8)
    edited = np.full((64, 64, 3), 200, dtype=np.uint8)
    out = sig.compute(real, edited)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_shape_mismatch_raises(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    sig = DINOv2Diff(image_size=224)
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.zeros((16, 24, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        sig.compute(a, b)


def test_non_rgb_input_raises(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    sig = DINOv2Diff(image_size=224)
    bad = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        sig.compute(bad, bad)


def test_signal_name_is_dinov2(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    sig = DINOv2Diff(image_size=224)
    assert sig.name == "dinov2"


# ---------------------------------------------------------------------------
# Internal plumbing: preprocessing, hub load args, patch grid geometry
# ---------------------------------------------------------------------------

def test_backbone_receives_square_image_size_input(monkeypatch):
    """Regardless of the caller's aspect ratio, the ViT sees exactly
    (1, 3, image_size, image_size) — DINOv2 requires that."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff

    sig = DINOv2Diff(image_size=224)
    real = np.zeros((96, 160, 3), dtype=np.uint8)  # non-square
    edited = np.zeros((96, 160, 3), dtype=np.uint8)
    sig.compute(real, edited)

    # Two forward calls (real + edited), both same shape.
    shapes = sig._model.forward_calls
    assert len(shapes) == 2
    for s in shapes:
        assert s == (1, 3, 224, 224)


def test_backbone_receives_custom_image_size(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff

    # 336 = 14 * 24, producing a 24x24 patch grid.
    sig = DINOv2Diff(image_size=336)
    real = np.zeros((80, 80, 3), dtype=np.uint8)
    edited = np.zeros((80, 80, 3), dtype=np.uint8)
    sig.compute(real, edited)

    for s in sig._model.forward_calls:
        assert s == (1, 3, 336, 336)


def test_hub_load_called_with_configured_repo_and_backbone(monkeypatch):
    hub_mod = _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff

    DINOv2Diff(backbone="dinov2_vitb14", hub_repo="my-mirror/dinov2")
    assert hub_mod.load.last_call == ("my-mirror/dinov2", "dinov2_vitb14")


def test_grid_side_matches_image_size_over_patch(monkeypatch):
    """Internal grid_side = image_size // 14 — pinned so a later change
    to DINOv2's patch size is caught."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    sig = DINOv2Diff(image_size=224)
    assert sig._grid_side == 16
    sig2 = DINOv2Diff(image_size=336)
    assert sig2._grid_side == 24


def test_default_backbone_is_vits14(monkeypatch):
    """Fastest variant is default — matches the LPIPSDiff `alex` default."""
    hub_mod = _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import DINOv2Diff
    DINOv2Diff()
    assert hub_mod.load.last_call[1] == "dinov2_vits14"


# ---------------------------------------------------------------------------
# Regression: cosine similarity must L2-normalize tokens explicitly.
# ---------------------------------------------------------------------------

def test_patch_tokens_are_l2_normalized_before_dot_product(monkeypatch):
    """DINOv2's `x_norm_patchtokens` are LayerNorm-normalized, not unit-norm
    L2-normalized. Computing ``1 - (a * b).sum(-1)`` as 'cosine distance'
    on raw tokens produces garbage whose scale is dominated by token norms
    (observed values around -400 for typical patch magnitudes). The fix
    is to apply F.normalize before the dot product.

    This test pins the fix by feeding identical patch tokens with large
    norms through the signal: cosine similarity of a vector with itself
    is 1.0 regardless of its norm, so the distance should be 0 and the
    resulting map — after percentile normalization with a near-zero
    denominator — should itself be all zeros.
    """
    # Install stubs with a fake DINOv2 that returns tokens with LARGE,
    # non-unit norms (mimicking the real model's layer-normed output
    # with high-norm artifact tokens).
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "float32"
    torch_mod.Tensor = _FakeTensor

    def _from_numpy(arr):
        return _FakeTensor(arr.astype(np.float32))

    def _tensor(seq, device=None):
        return _FakeTensor(np.array(seq, dtype=np.float32))

    torch_mod.from_numpy = _from_numpy
    torch_mod.tensor = _tensor

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch_mod.no_grad = _NoGrad

    def _interpolate(t, size=None, mode="bilinear", align_corners=False):
        arr = t._a
        lead = arr.shape[:-2]
        new_h, new_w = size
        out = np.full(lead + (new_h, new_w), float(arr.mean()), dtype=arr.dtype)
        return _FakeTensor(out)

    def _normalize(t, p=2, dim=-1, eps=1e-8):
        arr = t._a
        norm = np.linalg.norm(arr, ord=2, axis=dim, keepdims=True)
        return _FakeTensor(arr / np.maximum(norm, eps))

    nn_mod = types.ModuleType("torch.nn")
    func_mod = types.ModuleType("torch.nn.functional")
    func_mod.interpolate = _interpolate
    func_mod.normalize = _normalize
    nn_mod.functional = func_mod
    torch_mod.nn = nn_mod

    class _FakeDinov2LargeNorm:
        """Returns patch tokens with large, non-unit norms."""

        def __init__(self):
            self.forward_calls = []

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward_features(self, x):
            self.forward_calls.append(x.shape)
            n, c, h, w = x._a.shape
            g = h // 14
            n_tokens = g * g
            # Fixed, identical non-unit tokens for real and edited. Large
            # magnitude is critical to reproducing the bug: when tokens
            # have norms ~20, the raw dot product is ~400, and
            # 1 - 400 = -399, which the old code would then process.
            rng = np.random.default_rng(42)
            raw = rng.normal(scale=20.0, size=(n, n_tokens, 384)).astype(np.float32)
            return {
                "x_norm_patchtokens": _FakeTensor(raw),
                "x_norm_clstoken": _FakeTensor(np.zeros((n, 384), np.float32)),
            }

    hub_mod = types.ModuleType("torch.hub")
    shared_model = _FakeDinov2LargeNorm()

    def _hub_load(repo, model_name, verbose=False):
        return shared_model

    hub_mod.load = _hub_load
    torch_mod.hub = hub_mod

    torchvision_mod = types.ModuleType("torchvision")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", func_mod)
    monkeypatch.setitem(sys.modules, "torch.hub", hub_mod)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision_mod)
    monkeypatch.delitem(sys.modules, "edit2forensics.mask.signals", raising=False)

    from edit2forensics.mask.signals import DINOv2Diff

    sig = DINOv2Diff(image_size=224)
    real = np.zeros((32, 32, 3), dtype=np.uint8)
    edited = np.zeros((32, 32, 3), dtype=np.uint8)
    out = sig.compute(real, edited)

    # Tokens are identical (same fake model, same seed via shared_model),
    # so after L2 normalization cos_sim = 1 exactly, distance = 0, and the
    # output map is all zeros regardless of the tokens' original norms.
    # With the OLD buggy code (raw dot product), the output would be
    # dominated by token-norm magnitudes and NOT be all-zero.
    assert np.allclose(out, 0.0, atol=1e-5), (
        f"Identical tokens should yield zero cosine distance; "
        f"got range [{out.min():.3f}, {out.max():.3f}] "
        f"(did the fix that L2-normalizes tokens regress?)"
    )

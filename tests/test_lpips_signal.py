"""Tests for `LPIPSDiff`.

These tests deliberately do NOT require torch, LPIPS weights, or a GPU.
The user will exercise the real code path on GPU separately.

Strategy
--------
We stub the lazy-imported `torch` and `lpips` modules via `sys.modules`
injection + monkeypatching `importlib`. The stubs are just enough for
`LPIPSDiff.__init__` and `.compute()` to run — they let us verify:

* Shape / dtype of the output map matches the input (H, W, 3) -> (H, W).
* Output dtype is float32 and values lie in [0, 1].
* Preprocessing produces the expected 1x3xHxW tensor.
* Config validation (bad `net`, bad percentile, bad `min_side`).
* Lazy-import failure raises a clean ImportError when torch is absent.

We do NOT exercise any actual numeric behavior of LPIPS (that's a
backbone-weight-dependent property). The real forward path will be
exercised on the user's GPU run.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lazy-import failure path (no stubs installed)
# ---------------------------------------------------------------------------

def test_lpips_missing_deps_raises_clean_import_error(monkeypatch):
    """If torch or lpips is not installed, construction must fail fast
    with a clear ImportError pointing at the optional extra, not with
    some obscure attribute error later."""
    # Wipe any pre-injected stubs from other tests so the inner `import`
    # in LPIPSDiff.__init__ goes through sys.path properly.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "lpips", raising=False)

    # Force the import to fail by shadowing the names with a finder that
    # raises. We can't just delete from sys.modules because torch might
    # actually be installed on some CI machines; simulate "not installed".
    class _BlockingFinder:
        def find_module(self, name, path=None):
            if name in ("torch", "lpips"):
                raise ImportError(f"simulated: {name} not available")
            return None

        def find_spec(self, name, path=None, target=None):
            if name in ("torch", "lpips"):
                raise ImportError(f"simulated: {name} not available")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockingFinder()] + sys.meta_path)

    # Ensure our signal module is re-imported fresh so it re-tries the
    # inner imports.
    monkeypatch.delitem(sys.modules, "edit2forensics.mask.signals", raising=False)
    from edit2forensics.mask.signals import LPIPSDiff

    with pytest.raises(ImportError, match="torch.*lpips|lpips.*torch"):
        LPIPSDiff()


# ---------------------------------------------------------------------------
# Stubbed-dependency tests (most of the suite)
# ---------------------------------------------------------------------------

def _install_stubs(monkeypatch):
    """Install just-enough fake ``torch`` and ``lpips`` modules in sys.modules
    so that `LPIPSDiff` can construct and run ``.compute()`` end-to-end.

    The stubs record the tensor shapes they see (for shape assertions) and
    return shape-consistent fake outputs.
    """

    # --- fake torch -------------------------------------------------------
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "float32"  # just a sentinel

    class _FakeTensor:
        """A thin wrapper around a numpy array that supports the subset of
        the torch.Tensor API `LPIPSDiff.compute` uses."""

        def __init__(self, array: np.ndarray):
            self._a = array

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

        def to(self, *args, **kwargs):
            # Device/dtype moves are no-ops in the fake.
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._a

        def __truediv__(self, other):
            return _FakeTensor(self._a / other)

        def __sub__(self, other):
            return _FakeTensor(self._a - other)

        def __getitem__(self, idx):
            # Support slicing along dim 0 (used by compute_batch to scatter
            # batched outputs back to per-pair).
            return _FakeTensor(self._a[idx])

    def _from_numpy(arr):
        return _FakeTensor(arr.astype(np.float32))

    torch_mod.from_numpy = _from_numpy
    torch_mod.Tensor = _FakeTensor

    # torch.cat: concatenate along a dim (only dim=0 used by compute_batch).
    def _cat(tensors, dim=0):
        arrs = [t._a for t in tensors]
        return _FakeTensor(np.concatenate(arrs, axis=dim))

    torch_mod.cat = _cat

    # no_grad context manager
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch_mod.no_grad = _NoGrad

    # torch.nn.functional.interpolate
    def _interpolate(t, size=None, mode="bilinear", align_corners=False):
        # t has shape (N, C, H, W); resize to (N, C, *size) via naive nearest.
        # We don't care about fidelity — only that the shape is right.
        arr = t._a
        n, c = arr.shape[:2]
        new_h, new_w = size
        out = np.zeros((n, c, new_h, new_w), dtype=arr.dtype)
        # Fill with the mean so the percentile-normalization path is stable.
        out[...] = arr.mean()
        return _FakeTensor(out)

    nn_mod = types.ModuleType("torch.nn")
    func_mod = types.ModuleType("torch.nn.functional")
    func_mod.interpolate = _interpolate
    nn_mod.functional = func_mod
    torch_mod.nn = nn_mod

    # --- fake lpips -------------------------------------------------------
    lpips_mod = types.ModuleType("lpips")

    class _FakeLPIPS:
        def __init__(self, net="alex", spatial=True, verbose=False):
            self.net = net
            self.spatial = spatial
            self.forward_calls = []
            # When input_dependent_output=True (set by tests after
            # construction), the forward returns per-pair outputs that
            # depend on the input data — letting us verify that
            # compute_batch correctly scatters per-pair results back to
            # the right slot.
            self.input_dependent_output = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, a, b):
            # Record for shape inspection
            self.forward_calls.append((a.shape, b.shape))
            n, c, h, w = a._a.shape
            if self.input_dependent_output:
                # Use the mean of (a - b) per-pair as a per-pair scalar
                # filling the dense map. Discriminating: different
                # pairs produce different outputs.
                diff = a._a - b._a
                per_pair_means = diff.reshape(n, -1).mean(axis=1)  # (N,)
                out = np.broadcast_to(
                    per_pair_means.reshape(n, 1, 1, 1),
                    (n, 1, h, w),
                ).astype(np.float32).copy()
            else:
                # Spatial LPIPS output: (N, 1, H, W). Fill with a scalar
                # so the postprocess percentile is well-defined.
                out = np.full((n, 1, h, w), 0.5, dtype=np.float32)
            return _FakeTensor(out)

    lpips_mod.LPIPS = _FakeLPIPS

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", func_mod)
    monkeypatch.setitem(sys.modules, "lpips", lpips_mod)

    # Force re-import of signals module so the lazy import picks up stubs.
    monkeypatch.delitem(sys.modules, "edit2forensics.mask.signals", raising=False)


# ---------------------------------------------------------------------------
# Config validation — doesn't need the stubs (fires before lazy import)
# ---------------------------------------------------------------------------

def test_lpips_rejects_bad_net(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    with pytest.raises(ValueError, match="net must be"):
        LPIPSDiff(net="resnet")


def test_lpips_rejects_bad_percentile(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    with pytest.raises(ValueError, match="normalize_percentile"):
        LPIPSDiff(normalize_percentile=10.0)


def test_lpips_rejects_tiny_min_side(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    with pytest.raises(ValueError, match="min_side"):
        LPIPSDiff(min_side=16)


# ---------------------------------------------------------------------------
# Shape / dtype / contract checks with stubbed backend
# ---------------------------------------------------------------------------

def test_output_shape_matches_input_spatial(monkeypatch):
    """compute((H, W, 3)) must return (H, W), regardless of internal
    upsampling for small inputs."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=224)
    real = np.full((64, 80, 3), 100, dtype=np.uint8)
    edited = np.full((64, 80, 3), 150, dtype=np.uint8)

    out = sig.compute(real, edited)
    assert out.shape == (64, 80)
    assert out.dtype == np.float32


def test_output_shape_when_no_upsampling(monkeypatch):
    """Images already at/above min_side skip the upsample path, but shape
    of the output should still match the input (H, W)."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=64)
    real = np.full((128, 96, 3), 100, dtype=np.uint8)
    edited = np.full((128, 96, 3), 200, dtype=np.uint8)

    out = sig.compute(real, edited)
    assert out.shape == (128, 96)


def test_output_values_in_zero_to_one(monkeypatch):
    """Normalization must clip to [0, 1]."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=64)
    real = np.zeros((64, 64, 3), dtype=np.uint8)
    edited = np.ones((64, 64, 3), dtype=np.uint8) * 200

    out = sig.compute(real, edited)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_shape_mismatch_raises(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    sig = LPIPSDiff(device="cpu", min_side=64)
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.zeros((16, 24, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        sig.compute(a, b)


def test_non_rgb_input_raises(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    sig = LPIPSDiff(device="cpu", min_side=64)
    bad = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        sig.compute(bad, bad)


# ---------------------------------------------------------------------------
# Internal plumbing: preprocessing and upsample routing
# ---------------------------------------------------------------------------

def test_backbone_receives_nchw_tensor(monkeypatch):
    """The LPIPS model must be called with tensors of shape (1, 3, H, W)
    after preprocessing — this pins the permute/unsqueeze/contiguous
    sequence in the compute path."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=64)
    real = np.zeros((96, 120, 3), dtype=np.uint8)
    edited = np.zeros((96, 120, 3), dtype=np.uint8)
    sig.compute(real, edited)

    assert len(sig._model.forward_calls) == 1
    a_shape, b_shape = sig._model.forward_calls[0]
    assert a_shape == (1, 3, 96, 120)
    assert b_shape == (1, 3, 96, 120)


def test_small_inputs_are_upsampled_to_min_side(monkeypatch):
    """A 48x64 input with min_side=224 must be upsampled before LPIPS
    (width scales so the shorter side becomes 224)."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=224)
    real = np.zeros((48, 64, 3), dtype=np.uint8)
    edited = np.zeros((48, 64, 3), dtype=np.uint8)
    out = sig.compute(real, edited)

    # Backbone should have seen a larger spatial size than the raw input.
    a_shape, _ = sig._model.forward_calls[0]
    _, _, up_h, up_w = a_shape
    assert up_h >= 224 or up_w >= 224
    # But the *output* map is downsampled back to the original shape.
    assert out.shape == (48, 64)


def test_large_inputs_skip_upsampling(monkeypatch):
    """If both sides are already >= min_side, LPIPS sees the raw shape."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=224)
    real = np.zeros((256, 300, 3), dtype=np.uint8)
    edited = np.zeros((256, 300, 3), dtype=np.uint8)
    sig.compute(real, edited)

    a_shape, _ = sig._model.forward_calls[0]
    assert a_shape == (1, 3, 256, 300)


def test_signal_name_is_lpips(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    sig = LPIPSDiff(device="cpu", min_side=64)
    assert sig.name == "lpips"


# ---------------------------------------------------------------------------
# compute_batch — GPU-batched path
# ---------------------------------------------------------------------------

def test_batch_size_validation():
    """batch_size=0 or negative is rejected at construction."""
    import pytest

    # Don't need stubs — validation fires before lazy import.
    with pytest.raises(ValueError, match="batch_size"):
        # Constructor will raise on validation before the lazy import of
        # torch/lpips, so this works without stubs being installed.
        from edit2forensics.mask.signals import LPIPSDiff
        LPIPSDiff(device="cpu", batch_size=0)


def test_compute_batch_empty_list_returns_empty(monkeypatch):
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    sig = LPIPSDiff(device="cpu", min_side=64)
    out = sig.compute_batch([], [])
    assert out == []
    # No forward call was made.
    assert sig._model.forward_calls == []


def test_compute_batch_length_mismatch_raises(monkeypatch):
    import pytest
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    sig = LPIPSDiff(device="cpu", min_side=64)
    a = [np.zeros((64, 64, 3), dtype=np.uint8)]
    b = [np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="length mismatch"):
        sig.compute_batch(a, b)


def test_compute_batch_groups_same_shape_into_one_forward_call(monkeypatch):
    """Three pairs of the same shape should produce ONE batched forward
    call of batch size 3 — not three separate calls."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=64, batch_size=8)
    pair_shape = (96, 96, 3)
    reals = [np.zeros(pair_shape, dtype=np.uint8) for _ in range(3)]
    editeds = [np.zeros(pair_shape, dtype=np.uint8) for _ in range(3)]

    sig.compute_batch(reals, editeds)

    assert len(sig._model.forward_calls) == 1
    a_shape, b_shape = sig._model.forward_calls[0]
    # Batch dimension is 3.
    assert a_shape[0] == 3
    assert b_shape[0] == 3


def test_compute_batch_splits_different_shapes_into_separate_forwards(monkeypatch):
    """Pairs of different shapes can't share a batch — they should be
    grouped and run as separate forward calls, one per shape group."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=64, batch_size=8)
    reals = [
        np.zeros((96, 96, 3), dtype=np.uint8),
        np.zeros((128, 128, 3), dtype=np.uint8),
        np.zeros((96, 96, 3), dtype=np.uint8),
    ]
    editeds = [r.copy() for r in reals]

    sig.compute_batch(reals, editeds)

    # Two shape groups -> two forward calls.
    assert len(sig._model.forward_calls) == 2
    batch_sizes = sorted(call[0][0] for call in sig._model.forward_calls)
    # One group has 2 pairs (the 96x96), the other has 1 (the 128x128).
    assert batch_sizes == [1, 2]


def test_compute_batch_chunks_at_batch_size_limit(monkeypatch):
    """If a shape group has more pairs than batch_size, it splits into
    chunks of <= batch_size."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=64, batch_size=4)
    pair_shape = (96, 96, 3)
    n = 10  # > batch_size
    reals = [np.zeros(pair_shape, dtype=np.uint8) for _ in range(n)]
    editeds = [r.copy() for r in reals]

    sig.compute_batch(reals, editeds)

    # 10 pairs / 4 per batch = 3 forward calls (4 + 4 + 2).
    assert len(sig._model.forward_calls) == 3
    batch_sizes = sorted(call[0][0] for call in sig._model.forward_calls)
    assert batch_sizes == [2, 4, 4]


def test_compute_batch_results_match_per_pair_compute(monkeypatch):
    """The numerical equivalence guarantee. compute_batch([a, b, c])
    must produce the same outputs as [compute(a), compute(b), compute(c)],
    triplet by triplet."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig_serial = LPIPSDiff(device="cpu", min_side=64, batch_size=4)
    sig_batched = LPIPSDiff(device="cpu", min_side=64, batch_size=4)
    # Switch the fake into input-dependent mode so different pairs
    # produce different outputs — without this, every output is 0.5
    # and the equivalence check passes trivially.
    sig_serial._model.input_dependent_output = True
    sig_batched._model.input_dependent_output = True

    rng = np.random.default_rng(0)
    n = 5
    pair_shape = (96, 96, 3)
    reals = [rng.integers(0, 255, pair_shape, dtype=np.uint8) for _ in range(n)]
    editeds = [rng.integers(0, 255, pair_shape, dtype=np.uint8) for _ in range(n)]

    serial_results = [sig_serial.compute(r, e) for r, e in zip(reals, editeds)]
    batched_results = sig_batched.compute_batch(reals, editeds)

    assert len(serial_results) == len(batched_results) == n
    for i in range(n):
        # Identical results — the only difference between the paths is
        # forward-pass batch size, which must not affect the answer.
        assert np.array_equal(serial_results[i], batched_results[i]), (
            f"pair {i}: serial and batched outputs differ"
        )


def test_compute_batch_preserves_input_order_with_mixed_shapes(monkeypatch):
    """When input pairs are interleaved by shape, compute_batch must
    return outputs in the SAME order as the inputs — not in
    shape-grouped order."""
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff

    sig = LPIPSDiff(device="cpu", min_side=64, batch_size=4)
    sig._model.input_dependent_output = True

    # Build three pairs with shape A, B, A — interleaved.
    rng = np.random.default_rng(1)
    pairs = [
        (rng.integers(0, 255, (96, 96, 3), dtype=np.uint8),
         rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)),
        (rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
         rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)),
        (rng.integers(0, 255, (96, 96, 3), dtype=np.uint8),
         rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)),
    ]
    reals = [p[0] for p in pairs]
    editeds = [p[1] for p in pairs]

    batched = sig.compute_batch(reals, editeds)

    # Outputs at positions 0 and 2 are 96x96; position 1 is 128x128.
    # Verify shape — this would fail if the scatter logic put position 1
    # at position 0 or 2.
    assert batched[0].shape == (96, 96)
    assert batched[1].shape == (128, 128)
    assert batched[2].shape == (96, 96)

    # Also verify against per-pair compute for each position.
    serial = [sig.compute(r, e) for r, e in zip(reals, editeds)]
    for i in range(3):
        assert np.array_equal(serial[i], batched[i]), f"order mismatch at {i}"


def test_compute_batch_per_pair_shape_validation(monkeypatch):
    """Per-pair validation (shape mismatch, non-RGB) fires inside
    compute_batch the same way it does for compute."""
    import pytest
    _install_stubs(monkeypatch)
    from edit2forensics.mask.signals import LPIPSDiff
    sig = LPIPSDiff(device="cpu", min_side=64)

    # Pair 0: ok. Pair 1: shape mismatch within the pair.
    reals = [
        np.zeros((96, 96, 3), dtype=np.uint8),
        np.zeros((96, 96, 3), dtype=np.uint8),
    ]
    editeds = [
        np.zeros((96, 96, 3), dtype=np.uint8),
        np.zeros((100, 96, 3), dtype=np.uint8),  # different shape from pair-mate
    ]
    with pytest.raises(ValueError, match="pair 1 shape mismatch"):
        sig.compute_batch(reals, editeds)


"""Difference-signal abstractions for the MaskGenerator.

A `DiffSignal` computes a dense ``[H, W]`` difference map in ``[0, 1]``
from a pair of aligned RGB images. The MaskGenerator combines one or
more signals (max-pool across signals) before thresholding.

Design rationale
----------------
There are three candidate signals — pixel diff
(LAB), LPIPS, and DINOv2 — with the expectation that they are
combined. Rather than hardcoding the combination, we define a small
interface so additional signals can be added without touching the
orchestrator.

Keeping this interface explicit pays off in three places:

* **Testing.** We can instantiate tiny deterministic signals (e.g. a
  constant-map) in unit tests, isolating the orchestration logic from
  the specifics of any one signal.
* **Compute tiering.** LPIPS and DINOv2 require a GPU. Pixel-only runs
  must still produce sensible masks. Making the signal set configurable
  supports this.
* **Ablations.** The paper's ablation table requires swapping signals on
  and off. That's a config change if we expose signals as instantiable
  components.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from skimage.color import rgb2lab


class DiffSignal(ABC):
    """Abstract difference-signal producer.

    Implementations receive two RGB uint8 arrays of identical shape and
    return a float32 ``[H, W]`` array in ``[0, 1]`` where larger values
    indicate stronger disagreement between the images.
    """

    #: Short identifier used in artifact metadata and logging.
    name: str = ""

    @abstractmethod
    def compute(self, real: np.ndarray, edited: np.ndarray) -> np.ndarray:
        """Return a normalized diff map of shape ``(H, W)`` in ``[0, 1]``.

        Parameters
        ----------
        real, edited
            uint8 RGB arrays, same shape ``(H, W, 3)``.
        """
        raise NotImplementedError

    def compute_batch(
        self,
        reals: list[np.ndarray],
        editeds: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Return diff maps for a list of image pairs.

        Default implementation calls ``compute`` once per pair — i.e.
        no actual batching. Signals where batched computation is faster
        (notably ``LPIPSDiff`` and ``DINOv2Diff`` on GPU) override this
        to issue a single batched forward pass.

        Implementations MUST produce results numerically identical (up
        to floating-point reduction-order differences) to looping
        ``compute`` over the same inputs. The driver relies on this
        equivalence when it elects to batch — batching is purely a
        performance optimization, never a behavioral change.

        Parameters
        ----------
        reals, editeds
            Lists of equal length. Each pair ``(reals[i], editeds[i])``
            must have matching shape, but pair-to-pair shapes can vary.
            The implementation is responsible for grouping by shape if
            batching requires uniform spatial dimensions.
        """
        if len(reals) != len(editeds):
            raise ValueError(
                f"length mismatch: reals={len(reals)} editeds={len(editeds)}"
            )
        return [self.compute(r, e) for r, e in zip(reals, editeds)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class LabPixelDiff(DiffSignal):
    """Per-pixel difference in CIELAB color space.

    Why LAB rather than RGB? LAB is perceptually uniform, so equal
    Euclidean distances correspond to roughly equal perceived color
    differences. This makes a single global threshold meaningful across
    edits that shift luminance vs. chroma.

    The raw distance is normalized by the image's own 99th percentile
    rather than a fixed constant. Per-image normalization handles the
    wide range of edit intensities seen across datasets (a subtle
    recoloring produces small LAB distances everywhere; a style
    transfer produces large distances everywhere) — without it, a
    fixed threshold either catches everything in the first case or
    nothing in the second.
    """

    name = "lab_pixel"

    def __init__(self, normalize_percentile: float = 99.0) -> None:
        if not 50.0 <= normalize_percentile <= 100.0:
            raise ValueError("normalize_percentile must be in [50, 100]")
        self.normalize_percentile = normalize_percentile

    def compute(self, real: np.ndarray, edited: np.ndarray) -> np.ndarray:
        if real.shape != edited.shape:
            raise ValueError(
                f"shape mismatch: real={real.shape} edited={edited.shape}"
            )
        if real.ndim != 3 or real.shape[2] != 3:
            raise ValueError(f"expected HxWx3, got {real.shape}")

        # skimage's rgb2lab expects floats in [0, 1] (for uint8 input it
        # does the conversion itself, but being explicit is faster and
        # avoids version-dependent behavior).
        real_lab = rgb2lab(real.astype(np.float32) / 255.0)
        edited_lab = rgb2lab(edited.astype(np.float32) / 255.0)

        # Euclidean distance in LAB == Delta E_76. Good enough for masking.
        dist = np.sqrt(np.sum((real_lab - edited_lab) ** 2, axis=2))

        # Per-image normalization. Using percentile (not max) is robust
        # to single-pixel outliers from JPEG noise.
        denom = float(np.percentile(dist, self.normalize_percentile))
        if denom <= 1e-6:
            # Images are essentially identical. Return all-zeros rather
            # than NaN or 1.0 (which would suggest confident "all edited").
            return np.zeros(dist.shape, dtype=np.float32)

        return np.clip(dist / denom, 0.0, 1.0).astype(np.float32)


class LPIPSDiff(DiffSignal):
    """Dense perceptual-difference signal based on LPIPS.

    LPIPS [Zhang et al., CVPR 2018] measures perceptual distance by
    comparing activations of a pretrained backbone (AlexNet or VGG)
    between the two input images. In its standard usage it returns a
    scalar per image pair. We use ``spatial=True`` mode, which skips
    the final spatial pooling and returns a dense per-pixel map
    instead — the form the MaskGenerator needs.

    Why this signal
    ---------------
    LAB pixel diff catches *color/luminance* changes. LPIPS catches
    perceptual/textural changes that LAB misses: a repainted object
    with matched color histogram but different texture, a swap of
    one fur pattern for another, an inpainted region whose color
    blends in but whose high-frequency statistics don't. Combining
    LAB and LPIPS via the MaskGenerator's max-pool covers both
    failure modes of either alone.

    Compute profile
    ---------------
    A single forward pass through AlexNet/VGG per image pair. Fast
    on GPU (<10ms for 224x224), usable but slow on CPU (~100ms).
    Construction downloads the pretrained weights on first use via
    the ``lpips`` package; subsequent runs are cached.

    Dependencies
    ------------
    Requires ``torch`` and ``lpips``. These are optional — install via
    the ``lpips`` extra in ``pyproject.toml``. The imports are
    performed lazily inside ``__init__`` so code that never constructs
    an ``LPIPSDiff`` (CI, light-weight iteration) doesn't need them.
    """

    name = "lpips"

    def __init__(
        self,
        net: str = "alex",
        device: str = "cpu",
        min_side: int = 224,
        normalize_percentile: float = 99.0,
        batch_size: int = 16,
    ) -> None:
        """
        Parameters
        ----------
        net
            Backbone to use: ``"alex"`` (fastest, default), ``"vgg"``
            (slower, sometimes slightly sharper), or ``"squeeze"``.
        device
            Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``,
            ``"mps"``). Default CPU so tests and light iteration work
            without a GPU; override when running the full pipeline.
        min_side
            Images smaller than this on either side are bilinear-upsampled
            to ``min_side`` before LPIPS (and the output map is downsampled
            back to the original resolution). LPIPS is trained on natural
            image statistics around 224x224; feeding it 64x64 thumbnails
            produces less meaningful distances.
        normalize_percentile
            Per-image percentile used to normalize the raw LPIPS map to
            ``[0, 1]`` (matching ``LabPixelDiff``).
        batch_size
            Maximum batch size used by ``compute_batch``. Per-pair
            ``compute`` ignores this. Larger values amortize GPU
            kernel-launch overhead but use more VRAM:
            * On CPU, batching helps less (sub-2x typical) and can
              be set low (e.g. 4) to bound RAM.
            * On a small GPU (8-12GB), 16 is a safe default.
            * On a large GPU (24GB+), 32 or 64 may extract more
              throughput, especially with ``net="alex"``.
            If a forward pass OOMs at the configured batch_size,
            ``compute_batch`` does NOT automatically retry at a smaller
            size — the user should reduce this value.
        """
        if not 50.0 <= normalize_percentile <= 100.0:
            raise ValueError("normalize_percentile must be in [50, 100]")
        if net not in ("alex", "vgg", "squeeze"):
            raise ValueError(f"net must be alex/vgg/squeeze, got {net!r}")
        if min_side < 32:
            raise ValueError("min_side must be >= 32 (backbone stride constraints)")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.net = net
        self.device = device
        self.min_side = min_side
        self.normalize_percentile = normalize_percentile
        self.batch_size = batch_size

        # Lazy imports: fail here only when the user actually instantiates
        # the signal, not at module import time. This keeps unit tests
        # that never touch LPIPS fully usable in a torch-free environment.
        try:
            import torch
            import lpips as lpips_lib
        except ImportError as e:
            raise ImportError(
                "LPIPSDiff requires `torch` and `lpips`. Install with "
                "`uv sync --extra lpips` (see pyproject.toml)."
            ) from e

        self._torch = torch
        self._model = lpips_lib.LPIPS(net=net, spatial=True, verbose=False)
        self._model = self._model.to(device).eval()

    # ------------------------------------------------------------------ #
    # DiffSignal contract
    # ------------------------------------------------------------------ #

    def compute(self, real: np.ndarray, edited: np.ndarray) -> np.ndarray:
        if real.shape != edited.shape:
            raise ValueError(
                f"shape mismatch: real={real.shape} edited={edited.shape}"
            )
        if real.ndim != 3 or real.shape[2] != 3:
            raise ValueError(f"expected HxWx3, got {real.shape}")

        torch = self._torch
        orig_h, orig_w = real.shape[:2]
        real_t, up_h, up_w = self._preprocess(real)
        edited_t, _, _ = self._preprocess(edited)

        with torch.no_grad():
            # spatial=True -> output shape (1, 1, up_h, up_w), raw distances.
            dense = self._model(real_t, edited_t)

        return self._postprocess(dense, orig_h, orig_w, up_h, up_w)

    def compute_batch(
        self,
        reals: list[np.ndarray],
        editeds: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Batched LPIPS over a list of image pairs.

        Numerical equivalence with looping ``compute``: identical up to
        floating-point reduction-order differences within the model's
        forward pass (same weights, same inputs, same device). The
        only behavioral difference is throughput.

        Implementation notes
        --------------------
        * **Grouping by shape.** LPIPS forward requires uniform spatial
          size within a batch. We group input pairs by their post-
          upsampling shape ``(up_h, up_w)``, run one batch per group,
          then scatter the results back to the caller's input order.
          For most datasets this produces a single dominant group plus
          a few small ones; for Pico-Banana most images are 512x512
          and group together cleanly.
        * **No automatic OOM retry.** If the configured ``batch_size``
          OOMs the GPU, ``compute_batch`` raises. The user should
          reduce ``batch_size`` rather than have us silently run with
          fewer items per call (which would hide a misconfiguration).
        """
        if len(reals) != len(editeds):
            raise ValueError(
                f"length mismatch: reals={len(reals)} editeds={len(editeds)}"
            )
        if not reals:
            return []

        torch = self._torch
        n = len(reals)
        # Per-pair preprocessing produces (1, 3, up_h, up_w) tensors plus
        # the original (orig_h, orig_w) used for postprocessing. We do
        # this serially because preprocessing is CPU-cheap and trying
        # to batch the resize step adds complexity without payoff.
        prepped: list[tuple] = []  # (real_t, edited_t, orig_h, orig_w, up_h, up_w)
        for i in range(n):
            r, e = reals[i], editeds[i]
            if r.shape != e.shape:
                raise ValueError(
                    f"pair {i} shape mismatch: real={r.shape} edited={e.shape}"
                )
            if r.ndim != 3 or r.shape[2] != 3:
                raise ValueError(f"pair {i}: expected HxWx3, got {r.shape}")
            orig_h, orig_w = r.shape[:2]
            real_t, up_h, up_w = self._preprocess(r)
            edited_t, _, _ = self._preprocess(e)
            prepped.append((real_t, edited_t, orig_h, orig_w, up_h, up_w))

        # Group by post-upsampling shape so each group can be stacked.
        # The dict preserves first-seen order; that's irrelevant for
        # correctness (we scatter by original index below) but keeps
        # the iteration deterministic for debugging.
        from collections import defaultdict
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, (_, _, _, _, up_h, up_w) in enumerate(prepped):
            groups[(up_h, up_w)].append(i)

        results: list[np.ndarray | None] = [None] * n
        for (up_h, up_w), indices in groups.items():
            # Run one or more batches of size <= self.batch_size.
            for chunk_start in range(0, len(indices), self.batch_size):
                chunk = indices[chunk_start : chunk_start + self.batch_size]
                real_stack = torch.cat(
                    [prepped[i][0] for i in chunk], dim=0
                )  # (B, 3, up_h, up_w)
                edited_stack = torch.cat(
                    [prepped[i][1] for i in chunk], dim=0
                )

                with torch.no_grad():
                    dense_batch = self._model(real_stack, edited_stack)
                # dense_batch shape: (B, 1, up_h, up_w)

                # Scatter back, applying per-pair postprocessing
                # (downsample to original resolution + percentile
                # normalization).
                for k, i in enumerate(chunk):
                    _, _, orig_h, orig_w, _, _ = prepped[i]
                    # Slice to (1, 1, up_h, up_w) so _postprocess sees
                    # the same shape it does in the per-pair path.
                    dense_i = dense_batch[k : k + 1]
                    results[i] = self._postprocess(
                        dense_i, orig_h, orig_w, up_h, up_w
                    )

        # All slots must have been filled.
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _preprocess(self, arr: np.ndarray) -> tuple:
        """uint8 HxWx3 -> ((1, 3, up_h, up_w) tensor in [-1, 1], up_h, up_w).

        Handles:
        * device move + dtype conversion
        * channel reorder + batch-dim add
        * normalization to LPIPS's expected ``[-1, 1]`` range
        * optional bilinear upsampling for tiny inputs (``min_side``).
        """
        torch = self._torch
        orig_h, orig_w = arr.shape[:2]
        t = torch.from_numpy(arr.copy()).to(self.device, dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()  # 1x3xHxW
        t = t / 127.5 - 1.0

        up_h, up_w = orig_h, orig_w
        if min(orig_h, orig_w) < self.min_side:
            scale = self.min_side / min(orig_h, orig_w)
            up_h = int(round(orig_h * scale))
            up_w = int(round(orig_w * scale))
            t = torch.nn.functional.interpolate(
                t, size=(up_h, up_w), mode="bilinear", align_corners=False
            )
        return t, up_h, up_w

    def _postprocess(
        self,
        dense: "torch.Tensor",
        orig_h: int,
        orig_w: int,
        up_h: int,
        up_w: int,
    ) -> np.ndarray:
        """(1, 1, up_h, up_w) tensor -> (orig_h, orig_w) float32 numpy in [0, 1].

        Handles bilinear-downsampling back to the caller's resolution
        if ``compute`` had upsampled, plus the per-image percentile
        normalization that all signals share.
        """
        torch = self._torch
        if (up_h, up_w) != (orig_h, orig_w):
            dense = torch.nn.functional.interpolate(
                dense, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
        arr = dense.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        denom = float(np.percentile(arr, self.normalize_percentile))
        if denom <= 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip(arr / denom, 0.0, 1.0).astype(np.float32)


class DINOv2Diff(DiffSignal):
    """Dense semantic-difference signal using DINOv2 patch features.

    Runs a pretrained DINOv2 ViT on each image, extracts the per-patch
    feature grid (``x_norm_patchtokens``), computes cosine distance
    between corresponding patches, and upsamples the per-patch distance
    map back to image resolution.

    Why this signal
    ---------------
    LAB pixel diff catches color/luminance changes. LPIPS catches
    textural/perceptual changes. Both can miss *semantic* changes where
    the surface statistics are preserved — e.g., a dog replaced with a
    cat at the same color and fur texture, or an inpainting that
    matches local statistics but changes what the region depicts.
    DINOv2's self-supervised features are specifically sensitive to
    object identity and part structure, so the three signals are
    complementary failure modes of each other.

    Patch-grid specifics
    --------------------
    DINOv2 uses a 14x14 patch tokenizer. Inputs must be divisible by 14
    on both spatial axes. We resize to a fixed multiple of 14 (default
    ``image_size=224``, giving a 16x16 patch grid) before the forward
    pass, then bilinear-upsample the 16x16 cosine-distance grid back to
    the caller's original resolution. Users never have to think about
    the divisibility constraint.

    Compute profile
    ---------------
    One ViT forward pass per image pair. On a single GPU, the ``vits14``
    backbone runs in single-digit milliseconds for 224x224. CPU inference
    is possible (~100-300ms) but not recommended for full-scale runs.

    Dependencies
    ------------
    Requires ``torch`` and ``torchvision``. Install via the ``dinov2``
    optional extra in ``pyproject.toml``. The imports are lazy so the
    module is safely importable without torch.
    """

    name = "dinov2"

    # Valid DINOv2 variants exposed by torch.hub.
    _VALID_BACKBONES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")

    # ImageNet mean/std in [0,1] — DINOv2 was trained with this normalization.
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    # The DINOv2 patch size (14 for all shipped variants).
    _PATCH_SIZE = 14

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        device: str = "cpu",
        image_size: int = 224,
        normalize_percentile: float = 99.0,
        hub_repo: str = "facebookresearch/dinov2",
    ) -> None:
        """
        Parameters
        ----------
        backbone
            DINOv2 variant: one of ``dinov2_vits14`` (21M, default fastest),
            ``dinov2_vitb14`` (86M), ``dinov2_vitl14`` (300M),
            ``dinov2_vitg14`` (1.1B). Larger backbones give sharper
            semantic distinctions at proportionally higher cost.
        device
            Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``,
            ``"mps"``). Default CPU; override for production.
        image_size
            Side length (px) the input pair is resized to before the
            forward pass. Must be a multiple of 14 (DINOv2's patch size).
            224 -> 16x16 patch grid; 336 -> 24x24; 448 -> 32x32.
        normalize_percentile
            Per-image percentile used to normalize cosine distances to
            ``[0, 1]`` (matches the convention of the other signals).
        hub_repo
            Overridable for offline mirrors or forks of the DINOv2
            repository. Default is Meta's official.
        """
        if backbone not in self._VALID_BACKBONES:
            raise ValueError(
                f"backbone must be one of {self._VALID_BACKBONES}, got {backbone!r}"
            )
        if image_size % self._PATCH_SIZE != 0:
            raise ValueError(
                f"image_size must be a multiple of {self._PATCH_SIZE} "
                f"(DINOv2 patch size), got {image_size}"
            )
        if image_size < self._PATCH_SIZE * 4:
            raise ValueError(
                f"image_size must be >= {self._PATCH_SIZE * 4} to produce a "
                f"meaningful patch grid, got {image_size}"
            )
        if not 50.0 <= normalize_percentile <= 100.0:
            raise ValueError("normalize_percentile must be in [50, 100]")

        self.backbone = backbone
        self.device = device
        self.image_size = image_size
        self.normalize_percentile = normalize_percentile
        self.hub_repo = hub_repo

        # Lazy imports: fail here if the user actually tries to use
        # DINOv2 without torch/torchvision installed, rather than at
        # module import time.
        try:
            import torch
            import torchvision  # noqa: F401 -- verifies presence
        except ImportError as e:
            raise ImportError(
                "DINOv2Diff requires `torch` and `torchvision`. Install with "
                "`uv sync --extra dinov2` (see pyproject.toml)."
            ) from e

        self._torch = torch
        # Loading from hub downloads weights on first call and caches
        # them for subsequent constructions.
        self._model = torch.hub.load(hub_repo, backbone, verbose=False)
        self._model = self._model.to(device).eval()

        # Precompute normalization tensors (moved to device once, reused).
        self._mean = torch.tensor(self._IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(self._IMAGENET_STD, device=device).view(1, 3, 1, 1)

        # Patch-grid side length derived once from image_size.
        self._grid_side = image_size // self._PATCH_SIZE

    # ------------------------------------------------------------------ #
    # DiffSignal contract
    # ------------------------------------------------------------------ #

    def compute(self, real: np.ndarray, edited: np.ndarray) -> np.ndarray:
        if real.shape != edited.shape:
            raise ValueError(
                f"shape mismatch: real={real.shape} edited={edited.shape}"
            )
        if real.ndim != 3 or real.shape[2] != 3:
            raise ValueError(f"expected HxWx3, got {real.shape}")

        torch = self._torch
        orig_h, orig_w = real.shape[:2]

        # --- preprocess: uint8 HxWx3 -> float NCHW normalized to ImageNet --
        def _to_tensor(arr: np.ndarray) -> "torch.Tensor":
            t = torch.from_numpy(arr).to(self.device, dtype=torch.float32)
            t = t.permute(2, 0, 1).unsqueeze(0).contiguous()  # 1x3xHxW
            t = t / 255.0
            # Resize to image_size x image_size (ViT input constraint).
            if (t.shape[-2], t.shape[-1]) != (self.image_size, self.image_size):
                t = torch.nn.functional.interpolate(
                    t,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            # ImageNet normalization is required — DINOv2 was trained with it.
            t = (t - self._mean) / self._std
            return t

        real_t = _to_tensor(real)
        edited_t = _to_tensor(edited)

        # --- forward through DINOv2 in eval + no-grad ----------------------
        # `forward_features` returns a dict including:
        #   x_norm_clstoken   : (B, D)
        #   x_norm_patchtokens: (B, N_patches, D)
        # We want the patch tokens; they are already L2-normalized.
        with torch.no_grad():
            feats_real = self._model.forward_features(real_t)
            feats_edited = self._model.forward_features(edited_t)

        pt_real = feats_real["x_norm_patchtokens"]    # (1, N, D)
        pt_edited = feats_edited["x_norm_patchtokens"]

        # --- per-patch cosine distance ------------------------------------
        # IMPORTANT: Despite the "norm" prefix, DINOv2's x_norm_patchtokens
        # are LayerNorm-normalized, NOT L2-normalized. They are not unit
        # vectors; the DINOv2 paper additionally documents a subset of
        # "high-norm" artifact tokens whose magnitudes are far from 1.
        # Computing cosine similarity as a plain dot product therefore
        # produces a numerically meaningless map whose scale is dominated
        # by token norms rather than semantic similarity.
        #
        # We L2-normalize both token sets explicitly and then compute the
        # dot product, which IS genuine cosine similarity. The resulting
        # distance lives in [0, 2], which the downstream normalization
        # handles correctly.
        eps = 1e-8
        pt_real_norm = torch.nn.functional.normalize(pt_real, p=2, dim=-1, eps=eps)
        pt_edited_norm = torch.nn.functional.normalize(pt_edited, p=2, dim=-1, eps=eps)
        cos_sim = (pt_real_norm * pt_edited_norm).sum(dim=-1)   # (1, N)
        dist = 1.0 - cos_sim                                     # (1, N), in [0, 2]

        # Reshape from (1, N) -> (1, 1, G, G) where G = grid_side.
        # DINOv2's patchtokens are laid out in row-major order.
        g = self._grid_side
        dist_grid = dist.view(1, 1, g, g)

        # --- upsample back to original resolution --------------------------
        dist_full = torch.nn.functional.interpolate(
            dist_grid,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        # --- postprocess: tensor -> (H, W) numpy float32 in [0, 1] --------
        arr = (
            dist_full.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        )

        # Cosine distances are in [0, 2] mathematically but in practice
        # DINOv2 patch distances sit roughly in [0, 1]. Per-image
        # percentile normalization keeps the scale consistent with the
        # other signals for max-pool combination.
        denom = float(np.percentile(arr, self.normalize_percentile))
        if denom <= 1e-6:
            return np.zeros_like(arr, dtype=np.float32)

        return np.clip(arr / denom, 0.0, 1.0).astype(np.float32)


class SSIMDiff(DiffSignal):
    """Dense (1 - SSIM) signal as a difference map.

    Why this signal
    ---------------
    SSIM (Wang et al., 2004) compares two images via local windows,
    measuring agreement of luminance, contrast, and structure. The
    per-pixel SSIM map is high where local structure agrees and low
    where it disagrees, so ``1 - sim_map`` is a natural difference
    signal in ``[0, 2]`` (clipped to ``[0, 1]`` after percentile
    normalization to match the other signals).

    SSIM's distinguishing property among the signals in this codebase
    is **invariance to global luminance and contrast shifts**. By
    construction it normalizes each window by local means and
    variances, so a uniform brightening of the whole image produces
    near-zero (1-SSIM) everywhere. This is good news for catching
    *local* edits that disturb local structure (object swaps, replaced
    regions, inpainting whose statistics don't quite match) and bad
    news for catching *global* photometric edits (color grading,
    brightness shifts) — for those, the SSIM map says "everywhere is
    fine" while LAB says "everything changed."

    The complement to LAB is therefore the design intent: LAB catches
    color/luminance changes, SSIM catches local structural changes.
    For a forensic detector trained against datasets where local edits
    dominate (e.g., MagicBrush), this pair is a CPU-only candidate
    that competes with LAB+LPIPS without the GPU dependency.

    Implementation notes
    --------------------
    * **Grayscale by default.** scikit-image's SSIM produces a
      single-channel map even for color inputs (in channel-axis mode
      it returns per-channel maps which then need aggregation; for
      structural localization the grayscale formulation is most
      consistent with how SSIM is reported in the literature). Color
      sensitivity is the LAB signal's job.
    * **Gaussian-weighted window by default.** Matches the original
      SSIM paper and is more numerically stable than a uniform window.
      Note: with ``gaussian_weights=True``, scikit-image **ignores**
      the ``win_size`` parameter entirely — the filter's effective
      support is determined by ``sigma`` instead. Set
      ``gaussian_weights=False`` to make ``win_size`` meaningful.
    * **Per-image percentile normalization** mirrors the LAB and LPIPS
      signals so that max-pool combination is well-scaled.

    Empirical caveat
    ----------------
    Among edit categories represented in MagicBrush and Pico-Banana,
    SSIM is most informative for object-level localized changes and
    least informative for full-image style/photometric edits. If your
    benchmark mix tilts heavily toward style transfers, this signal
    will under-fire on those edits — that's not a bug, it's a property
    of SSIM, and the routing in `MaskGenerator._threshold` should still
    correctly classify those as global via the LAB component.
    """

    name = "ssim"

    def __init__(
        self,
        win_size: int = 7,
        gaussian_weights: bool = True,
        sigma: float = 1.5,
        normalize_percentile: float = 99.0,
    ) -> None:
        """
        Parameters
        ----------
        win_size
            Side length of the SSIM comparison window. Must be odd and
            >= 3. Only takes effect when ``gaussian_weights=False``;
            with the default gaussian path, scikit-image determines
            the effective support from ``sigma`` and ignores this.
        gaussian_weights
            If True (default), use a Gaussian filter as the comparison
            window — matches the original SSIM paper. If False, use a
            uniform ``win_size x win_size`` window.
        sigma
            Standard deviation of the Gaussian window when
            ``gaussian_weights=True``. The default (1.5) matches
            scikit-image's default and the SSIM paper. Larger sigma
            produces smoother diff maps with broader edge "leak"
            outside true edit regions; smaller sigma is sharper but
            noisier.
        normalize_percentile
            Per-image percentile used to rescale the diff map into
            ``[0, 1]``. Robust to single-pixel outliers; matches the
            other signals in this module.
        """
        if win_size < 3 or win_size % 2 == 0:
            raise ValueError(
                f"win_size must be an odd integer >= 3, got {win_size}"
            )
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not 50.0 <= normalize_percentile <= 100.0:
            raise ValueError("normalize_percentile must be in [50, 100]")
        self.win_size = win_size
        self.gaussian_weights = gaussian_weights
        self.sigma = sigma
        self.normalize_percentile = normalize_percentile

    def compute(self, real: np.ndarray, edited: np.ndarray) -> np.ndarray:
        if real.shape != edited.shape:
            raise ValueError(
                f"shape mismatch: real={real.shape} edited={edited.shape}"
            )
        if real.ndim != 3 or real.shape[2] != 3:
            raise ValueError(f"expected HxWx3, got {real.shape}")
        h, w = real.shape[:2]
        if min(h, w) < self.win_size:
            raise ValueError(
                f"image too small for win_size={self.win_size}: "
                f"got HxW={h}x{w}"
            )

        # Lazy import keeps this module importable in environments where
        # scikit-image is partially available.
        from skimage.color import rgb2gray
        from skimage.metrics import structural_similarity as ssim

        real_gray = rgb2gray(real.astype(np.float32) / 255.0)
        edited_gray = rgb2gray(edited.astype(np.float32) / 255.0)

        # full=True returns the (mean_sim, sim_map) tuple; we discard
        # the scalar. With gaussian_weights=True, sigma controls the
        # filter; win_size is ignored by skimage in that path. With
        # gaussian_weights=False, win_size is the uniform window side.
        ssim_kwargs: dict = {
            "data_range": 1.0,
            "gaussian_weights": self.gaussian_weights,
            "use_sample_covariance": not self.gaussian_weights,
            "full": True,
        }
        if self.gaussian_weights:
            ssim_kwargs["sigma"] = self.sigma
        else:
            ssim_kwargs["win_size"] = self.win_size

        _mean_sim, sim_map = ssim(real_gray, edited_gray, **ssim_kwargs)

        # SSIM map is in [-1, 1] mathematically but [0, 1] for
        # near-natural-image content. (1 - sim) is in [0, 2] in the
        # worst case; clip to [0, ∞) before normalization since
        # negative differences make no sense for our usage.
        diff = np.clip(1.0 - sim_map, 0.0, None).astype(np.float32)

        # Per-image percentile normalization, matching LAB / LPIPS / DINOv2.
        denom = float(np.percentile(diff, self.normalize_percentile))
        if denom <= 1e-6:
            return np.zeros(diff.shape, dtype=np.float32)

        return np.clip(diff / denom, 0.0, 1.0).astype(np.float32)


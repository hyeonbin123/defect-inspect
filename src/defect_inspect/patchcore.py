"""PatchCore: patch features, a coreset memory bank and nearest-neighbour anomaly scores."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from defect_inspect.backbones import to_tensor
from defect_inspect.coreset import kcenter_greedy

# Largest [query rows, bank rows] block built at once while searching the bank (2**27 fp32 = 512 MB).
_MAX_DIST_ELEMS = 1 << 27


@dataclass
class ScoreResult:
    image_scores: np.ndarray  # float32 [N]
    maps: np.ndarray  # float16 [N, map_size, map_size]


def _check_images(images: np.ndarray, batch_size: int) -> None:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected images [N, S, S, 3], got {images.shape}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")


def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    """fp16 autocast on CUDA, plain fp32 elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _patch_features(extractor: torch.nn.Module, images: np.ndarray, device: torch.device) -> torch.Tensor:
    """One batch of uint8 images -> fp16 patch features [B, H, W, D] on the device."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    with _autocast(device):
        feats = extractor(to_tensor(images, device, dtype))
    # Everything downstream (bank rows and queries alike) sees the same fp16-rounded values.
    feats = feats.to(torch.float16)
    if not bool(torch.isfinite(feats).all()):
        raise FloatingPointError("patch features overflowed fp16")
    return feats


@torch.no_grad()
def collect_features(
    extractor: torch.nn.Module,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """uint8 images [N, S, S, 3] -> (CPU fp16 patch features [N * H * W, D], grid (H, W)).

    Rows are ordered by image, then row-major over the patch grid.
    """
    _check_images(images, batch_size)
    device = torch.device(device)
    n = len(images)
    if n == 0:
        return torch.empty((0, int(extractor.dim)), dtype=torch.float16), (0, 0)
    extractor = extractor.to(device).eval()
    out: torch.Tensor | None = None
    grid = (0, 0)
    pos = 0
    for start in range(0, n, batch_size):
        feats = _patch_features(extractor, images[start : start + batch_size], device)
        b, h, w, dim = feats.shape
        if out is None:
            # Allocated once, so the peak stays at one copy of the features.
            grid = (h, w)
            out = torch.empty((n * h * w, dim), dtype=torch.float16)
        out[pos : pos + b * h * w] = feats.reshape(b * h * w, dim).cpu()
        pos += b * h * w
    assert out is not None and pos == out.shape[0]
    return out, grid


def build_bank(
    features: torch.Tensor, ratio: float = 0.1, *, seed: int = 0, device: str | torch.device = "cuda"
) -> torch.Tensor:
    """Greedy k-center coreset of `features` [N, D] as a CPU fp16 tensor in selection order.

    Keeps `max(1, round(ratio * N))` rows (`ratio >= 1` keeps all). `bank[:m]` is the coreset of size m.
    """
    n = features.shape[0]
    n_select = n if ratio >= 1 else max(1, round(ratio * n))
    index = kcenter_greedy(features, n_select, seed=seed, device=device)
    return features[index].to(torch.float16)


def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    """Normalised fp32 Gaussian kernel of size `2 * int(4 * sigma + 0.5) + 1` (a single 1 if sigma <= 0)."""
    if sigma <= 0:
        return torch.ones(1, dtype=torch.float32)
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).to(torch.float32)


def _blur(maps: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of [B, 1, H, W] with reflect padding."""
    pad = kernel.numel() // 2
    if pad == 0:
        return maps
    if pad >= min(maps.shape[-2:]):
        raise ValueError(f"blur radius {pad} needs a map larger than {tuple(maps.shape[-2:])}")
    maps = F.pad(maps, (pad, pad, pad, pad), mode="reflect")
    maps = F.conv2d(maps, kernel.view(1, 1, -1, 1))
    return F.conv2d(maps, kernel.view(1, 1, 1, -1))


def _nearest(
    query: torch.Tensor, bank: torch.Tensor, bank_sq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euclidean distance to, and index of, the nearest bank row for each query row (fp32, one device).

    The search uses the expanded form in blocks of at most `_MAX_DIST_ELEMS` values; the distance to
    the row it finds is then recomputed directly, so it does not suffer from cancellation near zero.
    """
    n_query = query.shape[0]
    step = max(1, _MAX_DIST_ELEMS // bank.shape[0])
    dist = torch.empty(n_query, dtype=torch.float32, device=query.device)
    index = torch.empty(n_query, dtype=torch.int64, device=query.device)
    for start in range(0, n_query, step):
        q = query[start : start + step]
        # ||b||^2 - 2 q.b orders the bank rows like the squared distance (||q||^2 is constant per row).
        block = torch.addmm(bank_sq.unsqueeze(0), q, bank.T, alpha=-2.0)
        nearest = block.argmin(dim=1)
        del block
        index[start : start + step] = nearest
        dist[start : start + step] = torch.linalg.vector_norm(q - bank[nearest], dim=1)
    return dist, index


def _image_scores(
    patch_scores: torch.Tensor,
    nearest: torch.Tensor,
    query: torch.Tensor,
    bank: torch.Tensor,
    bank_sq: torch.Tensor,
    reweight_k: int,
) -> torch.Tensor:
    """PatchCore image score: the largest patch score, re-weighted by its neighbourhood in the bank.

    `patch_scores` and `nearest` are [B, P]; `query` is [B, P, D].
    """
    s_star, p_star = patch_scores.max(dim=1)
    if reweight_k <= 1:
        return s_star
    k = min(reweight_k, bank.shape[0])
    rows = torch.arange(patch_scores.shape[0], device=patch_scores.device)
    feat_star = query[rows, p_star]  # [B, D]: the most anomalous patch of each image
    m_star = nearest[rows, p_star]  # [B]: its nearest bank row
    block = torch.addmm(bank_sq.unsqueeze(0), bank[m_star], bank.T, alpha=-2.0)  # [B, R]
    block[rows, m_star] = float("-inf")  # m* itself comes first, even if the bank has duplicates of it
    support = block.topk(k, dim=1, largest=False).indices  # [B, k], nearest first
    dist = torch.linalg.vector_norm(feat_star.unsqueeze(1) - bank[support], dim=2)  # [B, k]
    weight = 1.0 - torch.softmax(dist, dim=1)[:, 0]
    return weight * s_star


@torch.no_grad()
def score_images(
    extractor: torch.nn.Module,
    bank: torch.Tensor,
    images: np.ndarray,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    reweight_k: int = 9,
    sigma: float = 4.0,
    map_size: int = 256,
) -> ScoreResult:
    """Score uint8 images [N, S, S, 3] against a memory bank [R, D].

    Patch score = Euclidean distance to the nearest bank row. Image score = re-weighted largest patch
    score. Map = patch scores upsampled to S (bilinear), blurred (Gaussian `sigma`), resized to `map_size`.
    """
    _check_images(images, batch_size)
    if bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError(f"expected a non-empty bank [R, D], got {tuple(bank.shape)}")
    device = torch.device(device)
    n = len(images)
    image_scores = np.empty(n, dtype=np.float32)
    maps = np.empty((n, map_size, map_size), dtype=np.float16)
    if n == 0:
        return ScoreResult(image_scores, maps)

    extractor = extractor.to(device).eval()
    # One fp32 copy of the bank on the device. Centring leaves distances unchanged and keeps the
    # expanded form used for the search accurate (post-ReLU features share a large common offset).
    bank_dev = bank.to(device=device, dtype=torch.float32, copy=True)
    centre = bank_dev.mean(dim=0, keepdim=True)
    bank_dev -= centre
    bank_sq = torch.linalg.vector_norm(bank_dev, dim=1).square_()
    kernel = gaussian_kernel1d(sigma).to(device)
    size = (images.shape[1], images.shape[2])

    for start in range(0, n, batch_size):
        feats = _patch_features(extractor, images[start : start + batch_size], device)
        b, h, w, dim = feats.shape
        if dim != bank_dev.shape[1]:
            raise ValueError(f"feature dim {dim} does not match the bank ({bank_dev.shape[1]})")
        query = feats.reshape(b * h * w, dim).to(torch.float32).sub_(centre)
        del feats
        dist, nearest = _nearest(query, bank_dev, bank_sq)
        patch_scores = dist.view(b, h * w)
        scores = _image_scores(
            patch_scores, nearest.view(b, h * w), query.view(b, h * w, dim), bank_dev, bank_sq, reweight_k
        )
        image_scores[start : start + b] = scores.cpu().numpy()

        batch_maps = F.interpolate(
            patch_scores.view(b, 1, h, w), size=size, mode="bilinear", align_corners=False
        )
        batch_maps = _blur(batch_maps, kernel)
        if size != (map_size, map_size):
            batch_maps = F.interpolate(
                batch_maps, size=(map_size, map_size), mode="bilinear", align_corners=False
            )
        maps[start : start + b] = batch_maps[:, 0].to(torch.float16).cpu().numpy()
    return ScoreResult(image_scores, maps)

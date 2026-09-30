"""Greedy k-center coreset selection for PatchCore memory banks."""

from __future__ import annotations

import math

import torch

# Rows projected per step: bounds what is on the device besides the [N, proj_dim] projection.
_PROJECT_CHUNK = 32768


def _project(features: torch.Tensor, proj: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    """Return the fp32 [N, d] array the selection runs on, moving `features` to the device in chunks."""
    n = features.shape[0]
    out_dim = features.shape[1] if proj is None else proj.shape[1]
    z = torch.empty((n, out_dim), dtype=torch.float32, device=device)
    for start in range(0, n, _PROJECT_CHUNK):
        chunk = features[start : start + _PROJECT_CHUNK].to(device=device, dtype=torch.float32)
        z[start : start + _PROJECT_CHUNK] = chunk if proj is None else chunk @ proj
    return z


def _select(z: torch.Tensor, z_sq: torch.Tensor, first: torch.Tensor, n_select: int) -> torch.Tensor:
    """Greedy k-center loop over centred points `z` [N, d] with squared norms `z_sq` [N].

    `first` is the int64 [1] index of the first centre; everything is on one device, the result too.
    The current centre stays a 1-element device tensor and is only used through index_select /
    index_copy_: indexing with it (or `.item()`) would read a value back from the device in every
    iteration, and on CUDA that synchronisation costs more than the arithmetic of the iteration.
    """
    n = z.shape[0]
    selected = torch.empty(n_select, dtype=torch.int64, device=z.device)
    min_dist = torch.full((n,), float("inf"), dtype=torch.float32, device=z.device)
    dist = torch.empty(n, dtype=torch.float32, device=z.device)
    taken = torch.full((1,), -1.0, dtype=torch.float32, device=z.device)
    centre = first
    for i in range(n_select):
        selected[i : i + 1] = centre
        if i + 1 == n_select:
            break
        # Squared distance of every point to the last centre: ||z||^2 + ||c||^2 - 2 z.c
        torch.addmv(z_sq, z, z.index_select(0, centre)[0], alpha=-2.0, out=dist)
        dist.add_(z_sq.index_select(0, centre)).clamp_(min=0.0)
        torch.minimum(min_dist, dist, out=min_dist)
        # A chosen point is never chosen again, even among exact duplicates.
        min_dist.index_copy_(0, centre, taken)
        centre = torch.argmax(min_dist, dim=0, keepdim=True)
    return selected


@torch.no_grad()
def kcenter_greedy(
    features: torch.Tensor,
    n_select: int,
    *,
    seed: int = 0,
    proj_dim: int = 128,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Greedy k-center selection over the rows of `features` [N, D] (CPU, fp16 or fp32).

    Returns int64 indices in selection order, so the first m entries equal the result for
    `n_select=m`. Only the [N, proj_dim] projection and a few [N] vectors live on `device`.
    Non-finite features raise ValueError. The selection is reproducible on one device; another
    device may break near-ties differently, so build banks that are compared on the same device.
    """
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"expected a non-empty [N, D] tensor, got {tuple(features.shape)}")
    if n_select < 1:
        raise ValueError(f"n_select must be at least 1, got {n_select}")
    device = torch.device(device)
    n, dim = features.shape
    n_select = min(int(n_select), n)

    generator = torch.Generator().manual_seed(seed)
    proj = None
    if dim > proj_dim:
        proj = torch.randn(dim, proj_dim, generator=generator) / math.sqrt(proj_dim)
        proj = proj.to(device)
    first = torch.randint(n, (1,), generator=generator)

    z = _project(features, proj, device)
    # Centring leaves distances unchanged and keeps the expanded form in the loop accurate in fp32.
    z -= z.mean(dim=0, keepdim=True)
    z_sq = torch.linalg.vector_norm(z, dim=1).square_()
    # One NaN or inf anywhere reaches every row through the mean. Left alone, the loop's argmax
    # would return the same indices again and again instead of failing.
    if not bool(torch.isfinite(z_sq).all()):
        raise ValueError("features contain non-finite values")

    return _select(z, z_sq, first.to(device), n_select).cpu()

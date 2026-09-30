"""Bootstrap resampling, intervals and the pre-registered verdict rule (docs/experiments.md)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import beta, rankdata


def stratified_indices(sizes: Sequence[int], n_boot: int = 2000, seed: int = 0) -> list[np.ndarray]:
    """Resample every group separately: one int64 array [n_boot, size] per group, from a single rng.

    The same sizes and seed always give the same indices, which is what makes comparisons paired.
    """
    rng = np.random.default_rng(seed)
    out = []
    for size in sizes:
        size = int(size)
        if size < 0:
            raise ValueError(f"group size must be non-negative, got {size}")
        if size == 0:
            out.append(np.zeros((n_boot, 0), dtype=np.int64))
        else:
            out.append(rng.integers(0, size, size=(n_boot, size), dtype=np.int64))
    return out


def cluster_indices(cluster_ids: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[np.ndarray]:
    """Cluster bootstrap: draw whole clusters with replacement, as many as there are distinct clusters.

    Returns `n_boot` int64 index arrays into the original rows. Rows of a drawn cluster keep their
    original order, and a cluster drawn twice contributes its rows twice.
    """
    ids = np.asarray(cluster_ids).ravel()
    rng = np.random.default_rng(seed)
    if ids.size == 0:
        return [np.zeros(0, dtype=np.int64) for _ in range(n_boot)]
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    rows = np.argsort(inverse, kind="stable")  # rows grouped by cluster
    starts = np.cumsum(counts) - counts
    draws = rng.integers(0, counts.size, size=(n_boot, counts.size), dtype=np.int64)
    out = []
    for draw in draws:
        lengths = counts[draw]
        ends = np.cumsum(lengths)
        # position in `rows` of every output element: start of its cluster + offset within the cluster
        position = np.repeat(starts[draw] - (ends - lengths), lengths) + np.arange(ends[-1])
        out.append(rows[position].astype(np.int64))
    return out


def percentile_ci(samples: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """Central percentile interval of the samples, ignoring nan. (nan, nan) when nothing is left."""
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    values = np.asarray(samples, dtype=np.float64).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return float(lo), float(hi)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial interval for k successes out of n."""
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    tail = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lo, hi


def macro_auroc_bootstrap(
    neg_by_cat: Sequence[np.ndarray],
    pos_by_cat: Sequence[np.ndarray],
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution [n_boot] of the mean over categories of the image AUROC.

    Groups are, per category in order, its negatives then its positives. A category without negatives
    or positives has no AUROC, so the mean (every resample) is nan.
    """
    if len(neg_by_cat) != len(pos_by_cat):
        raise ValueError("neg_by_cat and pos_by_cat must have the same number of categories")
    negs = [np.asarray(a, dtype=np.float64).ravel() for a in neg_by_cat]
    poss = [np.asarray(a, dtype=np.float64).ravel() for a in pos_by_cat]
    if any(np.isnan(a).any() for a in negs + poss):
        raise ValueError("scores contain NaN")
    if not negs:
        return np.full(n_boot, np.nan)
    sizes = [a.size for pair in zip(negs, poss, strict=True) for a in pair]
    indices = stratified_indices(sizes, n_boot, seed)
    total = np.zeros(n_boot)
    for c, (neg, pos) in enumerate(zip(negs, poss, strict=True)):
        if neg.size == 0 or pos.size == 0:
            total += np.nan
            continue
        sample = np.concatenate([neg[indices[2 * c]], pos[indices[2 * c + 1]]], axis=1)
        ranks = rankdata(sample, axis=1)  # average ranks within each resample
        u = ranks[:, neg.size :].sum(axis=1) - pos.size * (pos.size + 1) / 2
        total += u / (neg.size * pos.size)
    return total / len(negs)


def pooled_rate_bootstrap(
    flags_by_cat: Sequence[np.ndarray], n_boot: int = 2000, seed: int = 0
) -> np.ndarray:
    """Bootstrap distribution [n_boot] of the pooled rate: total flagged / total count over categories.

    Each category is resampled separately (its size stays fixed). nan when there is nothing to count.
    """
    flags = [np.asarray(f).ravel() != 0 for f in flags_by_cat]
    sizes = [f.size for f in flags]
    total = sum(sizes)
    if total == 0:
        return np.full(n_boot, np.nan)
    indices = stratified_indices(sizes, n_boot, seed)
    flagged = np.zeros(n_boot, dtype=np.int64)
    for f, idx in zip(flags, indices, strict=True):
        flagged += f[idx].sum(axis=1)
    return flagged / total


def verdict(diff_samples: np.ndarray, point: float, min_effect: float) -> str:
    """Pre-registered verdict for a difference: its 95% percentile interval and a minimum effect size.

    An undefined (NaN) point estimate or interval is inconclusive, never a difference.
    """
    if math.isnan(point):
        return "판정 불가"
    lo, hi = percentile_ci(diff_samples, 0.95)
    if not (lo > 0 or hi < 0):  # interval contains 0 (or is undefined)
        return "판정 불가"
    if abs(point) >= min_effect:
        return "차이 있음"
    return "차이 작음"

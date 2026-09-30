"""Image- and pixel-level metrics (definitions fixed in docs/experiments.md).

Two decision rules appear here on purpose:
- ``score >= t`` for the curve metrics (AUROC, AUPRO, FPR at a given TPR), where ``t`` is itself a score;
- ``score > threshold`` for fixed thresholds (`rates_at_threshold`), the rule used everywhere else.

Scores must not contain NaN (a NaN score is an upstream bug, so it raises instead of being ranked).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata


def _scores(values: np.ndarray, name: str) -> np.ndarray:
    """Flatten image-level scores to float64 (exact for float16/32 inputs) and reject NaN."""
    out = np.asarray(values, dtype=np.float64).ravel()
    if np.isnan(out).any():
        raise ValueError(f"{name} contains NaN")
    return out


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(pos > neg) + 0.5 * P(pos == neg), from average ranks. Empty `neg` or `pos` gives nan."""
    neg = _scores(neg, "neg")
    pos = _scores(pos, "pos")
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([neg, pos]))
    u = ranks[neg.size :].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (neg.size * pos.size))


def fpr_at_tpr(neg: np.ndarray, pos: np.ndarray, tpr: float = 0.95) -> tuple[float, float, int]:
    """False positive rate at the highest threshold that still catches `tpr` of the positives.

    Rule ``score >= t`` with ``t`` the k-th largest positive score, ``k = ceil(tpr * len(pos))``.
    Returns ``(fp / len(neg), t, fp)``. The rate is nan when `neg` is empty; an empty `pos` raises.
    """
    neg = _scores(neg, "neg")
    pos = _scores(pos, "pos")
    if not 0.0 < tpr <= 1.0:
        raise ValueError(f"tpr must be in (0, 1], got {tpr}")
    if pos.size == 0:
        raise ValueError("fpr_at_tpr needs at least one positive score")
    # Exact rational arithmetic: 0.95 * 20 must give 19, never 20 because of a rounding error.
    k = math.ceil(Fraction(str(float(tpr))) * pos.size)
    k = min(max(k, 1), pos.size)
    t = float(np.partition(pos, pos.size - k)[pos.size - k])
    fp = int(np.count_nonzero(neg >= t))
    rate = fp / neg.size if neg.size else float("nan")
    return rate, t, fp


def rates_at_threshold(neg: np.ndarray, pos: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """Counts at a fixed threshold with the rule ``score > threshold``: ``(fp, n_neg, tp, n_pos)``."""
    neg = _scores(neg, "neg")
    pos = _scores(pos, "pos")
    threshold = float(threshold)
    if math.isnan(threshold):
        raise ValueError("threshold is NaN")
    fp = int(np.count_nonzero(neg > threshold))
    tp = int(np.count_nonzero(pos > threshold))
    return fp, int(neg.size), tp, int(pos.size)


def _check_maps(maps: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate [N, H, W] score maps and masks; return the maps and a boolean defect mask."""
    maps = np.asarray(maps)
    masks = np.asarray(masks)
    if maps.ndim != 3 or maps.shape != masks.shape:
        raise ValueError(f"maps and masks must both be [N, H, W], got {maps.shape} and {masks.shape}")
    if maps.dtype.kind != "f":
        maps = maps.astype(np.float64)
    elif maps.dtype == np.float16:
        maps = maps.astype(np.float32)  # exact, and float32 sorts several times faster
    if np.isnan(maps).any():
        raise ValueError("maps contains NaN")
    defect = masks if masks.dtype == np.bool_ else masks != 0
    return maps, defect


def _auroc_sorted(neg: np.ndarray, pos: np.ndarray) -> float:
    """AUROC by sorting both sides and counting with binary search (no pair matrix, no rank array)."""
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    neg = np.sort(neg)
    pos = np.sort(pos)
    # sum over positives of (#neg < p) + (#neg <= p) = 2 * (#pairs won) + (#pairs tied), in exact integers
    below = np.searchsorted(neg, pos, side="left").sum(dtype=np.int64)
    below_or_equal = np.searchsorted(neg, pos, side="right").sum(dtype=np.int64)
    return (int(below) + int(below_or_equal)) / (2 * neg.size * pos.size)


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """AUROC over all pixels of all images. nan when there are no defect or no normal pixels."""
    maps, defect = _check_maps(maps, masks)
    return _auroc_sorted(maps[~defect], maps[defect])


# Connects the 8 neighbours inside an image and nothing across images (axis 0).
_STRUCTURE = np.zeros((3, 3, 3), dtype=bool)
_STRUCTURE[1] = True


def _label_components(defect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 8-connected regions per image with global ids 1..C; also return the image of each region."""
    labels, n_comp = ndimage.label(defect, structure=_STRUCTURE)
    flat = np.flatnonzero(labels)
    comp_image = np.zeros(n_comp, dtype=np.int64)
    comp_image[labels.ravel()[flat] - 1] = flat // (defect.shape[1] * defect.shape[2])
    return labels, comp_image


def _area_under(x: np.ndarray, y: np.ndarray, x_max: float) -> np.ndarray:
    """Trapezoid area under polylines along the last axis, over [x[0], x_max], interpolating at x_max.

    `x` must be non-decreasing along the last axis. Vertical segments (equal x) contribute nothing.
    """
    x0, x1, y0, y1 = x[..., :-1], x[..., 1:], y[..., :-1], y[..., 1:]
    c0 = np.minimum(x0, x_max)
    c1 = np.minimum(x1, x_max)
    dx = x1 - x0
    frac = np.divide(c1 - x0, dx, out=np.ones_like(dx), where=dx > 0)
    y_end = y0 + (y1 - y0) * frac
    return np.sum((c1 - c0) * (y0 + y_end) / 2, axis=-1)


def _check_max_fpr(max_fpr: float) -> None:
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")


def aupro(maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3) -> float:
    """Exact area under the per-region-overlap curve for FPR in [0, max_fpr], divided by max_fpr.

    Every distinct score is a threshold (rule ``score >= t``). PRO only changes at scores of defect
    pixels, so the curve is evaluated there: for each distinct defect score d, the curve runs flat up to
    FPR(> d) and then joins (FPR(>= d), PRO(d)) with a straight segment, which is exactly the polyline
    through all distinct scores. nan when there are no regions or no normal pixels.
    """
    _check_max_fpr(max_fpr)
    maps, defect = _check_maps(maps, masks)
    labels, comp_image = _label_components(defect)
    n_comp = comp_image.size
    neg = np.sort(maps[~defect])
    if n_comp == 0 or neg.size == 0:
        return float("nan")

    comp = labels[defect] - 1
    scores = maps[defect]
    del labels
    sizes = np.bincount(comp, minlength=n_comp)
    order = np.argsort(scores, kind="stable")[::-1]
    scores = scores[order]
    # cumulative mean overlap, going down the defect scores
    overlap = np.cumsum(1.0 / sizes[comp[order]]) / n_comp
    last = np.flatnonzero(np.append(scores[1:] != scores[:-1], True))  # last pixel of each tie group
    thresholds = scores[last]
    pro = np.minimum(overlap[last], 1.0)
    pro[-1] = 1.0  # every region is fully covered at the lowest defect score
    fpr_above = (neg.size - np.searchsorted(neg, thresholds, side="right")) / neg.size
    fpr_at = (neg.size - np.searchsorted(neg, thresholds, side="left")) / neg.size

    x = np.empty(2 * thresholds.size + 2)
    y = np.empty_like(x)
    x[0], y[0] = 0.0, 0.0
    x[1:-1:2], y[1:-1:2] = fpr_above, np.concatenate([[0.0], pro[:-1]])
    x[2:-1:2], y[2:-1:2] = fpr_at, pro
    x[-1], y[-1] = 1.0, 1.0
    return float(_area_under(x, y, max_fpr)) / max_fpr


@dataclass
class ProHistograms:
    """Binned score counts from which AUPRO can be recomputed for any resampling of the images."""

    edges: np.ndarray  # [bins + 1]
    normal: np.ndarray  # int64 [n_images, bins]: histogram of scores on normal pixels, per image
    components: np.ndarray  # int64 [n_components, bins]
    component_image: np.ndarray  # int64 [n_components]: image index of each component


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each value: edges[i] <= v < edges[i + 1]; the last bin is closed, outliers are clipped."""
    bins = edges.size - 1
    lo, hi = edges[0], edges[-1]
    values = np.clip(values, lo, hi)
    idx = np.floor((values - lo) * (bins / (hi - lo))).astype(np.int64)
    np.clip(idx, 0, bins - 1, out=idx)
    # repair rounding at bin boundaries so that the counts match `edges` exactly
    idx -= (values < edges[idx]) & (idx > 0)
    idx += (values >= edges[idx + 1]) & (idx < bins - 1)
    return idx


def pro_histograms(
    maps: np.ndarray,
    masks: np.ndarray,
    bins: int = 2000,
    lo: float | None = None,
    hi: float | None = None,
) -> ProHistograms:
    """Histogram the scores of normal pixels (per image) and of each defect region.

    `lo`/`hi` default to the min/max of `maps`; scores outside [lo, hi] go to the first/last bin.
    """
    maps, defect = _check_maps(maps, masks)
    if bins < 1:
        raise ValueError(f"bins must be positive, got {bins}")
    if maps.size == 0 and (lo is None or hi is None):
        raise ValueError("lo and hi are required for empty maps")
    lo = float(maps.min()) if lo is None else float(lo)
    hi = float(maps.max()) if hi is None else float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"histogram range must be finite, got [{lo}, {hi}]")
    if hi < lo:
        raise ValueError(f"hi ({hi}) is below lo ({lo})")
    if hi == lo:  # constant maps: any positive width works, everything lands in one bin
        hi = lo + max(1.0, abs(lo))
    edges = np.linspace(lo, hi, bins + 1)

    labels, comp_image = _label_components(defect)
    n_comp = comp_image.size
    normal = np.zeros((maps.shape[0], bins), dtype=np.int64)
    comp_parts: list[np.ndarray] = []
    for i in range(maps.shape[0]):
        idx = _bin_index(maps[i].astype(np.float64).ravel(), edges)
        is_defect = defect[i].ravel()
        if is_defect.any():
            normal[i] = np.bincount(idx[~is_defect], minlength=bins)
            comp_parts.append((labels[i].ravel()[is_defect] - 1).astype(np.int64) * bins + idx[is_defect])
        else:
            normal[i] = np.bincount(idx, minlength=bins)
    flat = np.concatenate(comp_parts) if comp_parts else np.zeros(0, dtype=np.int64)
    components = np.bincount(flat, minlength=n_comp * bins).astype(np.int64).reshape(n_comp, bins)
    return ProHistograms(edges=edges, normal=normal, components=components, component_image=comp_image)


def _binned_aupro(h: ProHistograms, counts: np.ndarray, max_fpr: float) -> np.ndarray:
    """Binned AUPRO for each row of `counts` (float64 [B, n_images]); all scores in a bin count as tied."""
    normal = counts @ h.normal.astype(np.float64)
    n_normal = normal.sum(axis=1)
    weights = counts[:, h.component_image]
    n_regions = weights.sum(axis=1)
    out = np.full(counts.shape[0], np.nan)
    valid = (n_regions > 0) & (n_normal > 0)
    if not valid.any():
        return out
    sizes = h.components.sum(axis=1)
    normal = normal[valid]
    overlap = (weights[valid] / sizes) @ h.components.astype(np.float64) / n_regions[valid, None]
    # cumulative sums from the top bin downwards: the threshold is the lower edge of each bin
    x = np.zeros((normal.shape[0], normal.shape[1] + 1))
    y = np.zeros_like(x)
    x[:, 1:] = np.cumsum(normal[:, ::-1], axis=1) / n_normal[valid, None]
    y[:, 1:] = np.minimum(np.cumsum(overlap[:, ::-1], axis=1), 1.0)
    x[:, -1], y[:, -1] = 1.0, 1.0
    out[valid] = _area_under(x, y, max_fpr) / max_fpr
    return out


def _check_counts(h: ProHistograms, image_counts: np.ndarray, ndim: int) -> np.ndarray:
    counts = np.asarray(image_counts, dtype=np.float64)
    n_images = h.normal.shape[0]
    if counts.ndim != ndim or counts.shape[-1] != n_images:
        shape = f"({n_images},)" if ndim == 1 else f"[n_boot, {n_images}]"
        raise ValueError(f"image_counts must have shape {shape}, got {counts.shape}")
    if not (counts >= 0).all():
        raise ValueError("image_counts must be non-negative")
    return counts


def aupro_from_histograms(
    h: ProHistograms, image_counts: np.ndarray | None = None, max_fpr: float = 0.3
) -> float:
    """AUPRO from binned counts; all scores in a bin are treated as tied.

    `image_counts[i]` is how often image i occurs in the (bootstrap) sample; its normal pixels and its
    regions are counted that many times. nan when the sample has no region or no normal pixel.
    """
    _check_max_fpr(max_fpr)
    if image_counts is None:
        counts = np.ones(h.normal.shape[0])
    else:
        counts = _check_counts(h, image_counts, ndim=1)
    return float(_binned_aupro(h, counts[None, :], max_fpr)[0])


def aupro_bootstrap(h: ProHistograms, image_counts: np.ndarray, max_fpr: float = 0.3) -> np.ndarray:
    """`aupro_from_histograms` for many resamples at once: `image_counts` is [n_boot, n_images].

    Same value per row as the single call, without converting the histograms once per resample.
    """
    _check_max_fpr(max_fpr)
    counts = _check_counts(h, image_counts, ndim=2)
    chunk = 256  # bounds the [chunk, bins] work arrays
    parts = [_binned_aupro(h, counts[i : i + chunk], max_fpr) for i in range(0, counts.shape[0], chunk)]
    return np.concatenate(parts) if parts else np.zeros(0)

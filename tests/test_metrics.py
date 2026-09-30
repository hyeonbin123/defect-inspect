"""Metric tests against independent references: scikit-learn and brute-force sweeps written here."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from defect_inspect.metrics import (
    ProHistograms,
    aupro,
    aupro_bootstrap,
    aupro_from_histograms,
    auroc,
    fpr_at_tpr,
    pixel_auroc,
    pro_histograms,
    rates_at_threshold,
)

# ---------------------------------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------------------------------


def sklearn_auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    return float(roc_auc_score(y, np.concatenate([neg, pos]).astype(np.float64)))


def pairwise_auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos.tolist() for n in neg.tolist())
    return wins / (len(neg) * len(pos))


def fpr_at_tpr_bruteforce(neg: np.ndarray, pos: np.ndarray, tpr: float) -> tuple[float, float, int]:
    """Try every score as a threshold; keep the highest one that catches at least `tpr` of `pos`."""
    target = Fraction(str(tpr))
    best = None
    for t in sorted(set(pos.tolist()) | set(neg.tolist())):
        if Fraction(int(np.sum(pos >= t)), len(pos)) >= target:
            best = t
    assert best is not None
    fp = int(np.sum(neg >= best))
    return fp / len(neg), best, fp


def flood_fill_components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    """Connected regions of a 2-D mask by flood fill, in raster order of their first pixel."""
    h, w = mask.shape
    if connectivity == 8:
        steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seen = np.zeros((h, w), dtype=bool)
    regions = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack, region = [(r, c)], []
            while stack:
                y, x = stack.pop()
                region.append((y, x))
                for dy, dx in steps:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            regions.append(region)
    return regions


def clipped_area(points: list[tuple[float, float]], x_max: float) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=True):
        if x0 >= x_max:
            break
        if x1 <= x_max:
            area += (x1 - x0) * (y0 + y1) / 2
        else:
            y_mid = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
            area += (x_max - x0) * (y0 + y_mid) / 2
    return area


def aupro_bruteforce(
    maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3, connectivity: int = 8
) -> float:
    """Sweep every distinct score as a threshold (score >= t) and integrate the PRO curve."""
    maps = np.asarray(maps, dtype=np.float64)
    masks = np.asarray(masks).astype(bool)
    regions = []
    for image, mask in zip(maps, masks, strict=True):
        for region in flood_fill_components(mask, connectivity):
            regions.append(np.array([image[y, x] for y, x in region]))
    normal = maps[~masks]
    if not regions or normal.size == 0:
        return float("nan")
    points = [(0.0, 0.0)]
    for t in sorted(set(maps.ravel().tolist()), reverse=True):
        fpr = float(np.sum(normal >= t)) / normal.size
        pro = sum(float(np.sum(r >= t)) / r.size for r in regions) / len(regions)
        points.append((fpr, pro))
    assert points[-1] == (1.0, pytest.approx(1.0))
    return clipped_area(points, max_fpr) / max_fpr


def aupro_fullsort(maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3) -> float:
    """Second reference for larger inputs: label image by image, sort all pixels, cumulative sums."""
    maps = np.asarray(maps, dtype=np.float64)
    masks = np.asarray(masks).astype(bool)
    weight = np.zeros(maps.shape)
    n_regions = 0
    for i in range(len(maps)):
        labels, n = ndimage.label(masks[i], structure=np.ones((3, 3)))
        for j in range(1, n + 1):
            weight[i][labels == j] = 1.0 / np.sum(labels == j)
        n_regions += n
    order = np.argsort(-maps.ravel(), kind="stable")
    scores = maps.ravel()[order]
    fp = np.cumsum(~masks.ravel()[order]) / np.sum(~masks)
    pro = np.cumsum(weight.ravel()[order]) / n_regions
    last = np.flatnonzero(np.append(scores[1:] != scores[:-1], True))
    points = [(0.0, 0.0)] + list(zip(fp[last].tolist(), pro[last].tolist(), strict=True))
    return clipped_area(points, max_fpr) / max_fpr


def random_case(seed: int, kind: str, n: int = 5, size: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Small random maps/masks: scattered defect pixels (many regions, diagonal contacts), clean images."""
    rng = np.random.default_rng(seed)
    masks = rng.random((n, size, size)) < rng.choice([0.08, 0.2, 0.4])
    masks[rng.integers(n)] = False  # at least one image without defects
    signal = rng.normal(size=(n, size, size)) + rng.choice([0.0, 1.0, 2.5]) * masks
    if kind == "continuous":
        maps = signal.astype(np.float32)
    elif kind == "float16":
        maps = np.round(signal, 1).astype(np.float16)
    elif kind == "few_values":
        maps = np.clip(np.round(signal), -1, 2).astype(np.float32)
    elif kind == "binary":
        maps = (signal > 0.5).astype(np.float32)
    else:
        raise ValueError(kind)
    return maps, masks


KINDS = ("continuous", "float16", "few_values", "binary")


# ---------------------------------------------------------------------------------------------------
# auroc
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_auroc_matches_sklearn_continuous(seed):
    rng = np.random.default_rng(seed)
    neg = rng.normal(size=rng.integers(1, 300))
    pos = rng.normal(loc=0.8, size=rng.integers(1, 60))
    assert auroc(neg, pos) == pytest.approx(sklearn_auroc(neg, pos), abs=1e-12)
    assert auroc(neg.astype(np.float32), pos.astype(np.float32)) == pytest.approx(
        sklearn_auroc(neg.astype(np.float32), pos.astype(np.float32)), abs=1e-12
    )


@pytest.mark.parametrize("seed", range(6))
def test_auroc_matches_sklearn_and_pairs_with_ties(seed):
    rng = np.random.default_rng(100 + seed)
    neg = rng.integers(0, 5, size=rng.integers(1, 80)).astype(np.float64)
    pos = rng.integers(2, 7, size=rng.integers(1, 40)).astype(np.float64)
    expected = pairwise_auroc(neg, pos)
    assert auroc(neg, pos) == pytest.approx(expected, abs=1e-12)
    assert sklearn_auroc(neg, pos) == pytest.approx(expected, abs=1e-12)


def test_auroc_simple_values():
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.full(7, 0.3), np.full(4, 0.3)) == 0.5  # constant scores
    # one tie out of four pairs: (1 + 1 + 1 + 0.5) / 4
    assert auroc(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx(0.875)


def test_auroc_accepts_any_shape_and_int_scores():
    neg = np.array([[0, 1], [1, 2]])
    pos = np.array([[2, 3]])
    assert auroc(neg, pos) == pytest.approx(pairwise_auroc(neg.ravel(), pos.ravel()))


def test_auroc_empty_is_nan_and_nan_raises():
    assert math.isnan(auroc(np.array([]), np.array([1.0])))
    assert math.isnan(auroc(np.array([1.0]), np.array([])))
    with pytest.raises(ValueError, match="NaN"):
        auroc(np.array([0.0, np.nan]), np.array([1.0]))


# ---------------------------------------------------------------------------------------------------
# fpr_at_tpr, rates_at_threshold
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("tpr", [0.95, 0.9, 0.5, 1.0])
def test_fpr_at_tpr_matches_bruteforce(seed, tpr):
    rng = np.random.default_rng(seed)
    n_pos = int(rng.choice([1, 7, 20, 40, 41]))
    if seed % 2:  # heavy ties
        neg = rng.integers(0, 6, size=120).astype(np.float64)
        pos = rng.integers(2, 9, size=n_pos).astype(np.float64)
    else:
        neg = rng.normal(size=120).astype(np.float32)
        pos = rng.normal(loc=1.0, size=n_pos).astype(np.float32)
    rate, t, fp = fpr_at_tpr(neg, pos, tpr)
    ref_rate, ref_t, ref_fp = fpr_at_tpr_bruteforce(neg, pos, tpr)
    assert (t, fp) == (ref_t, ref_fp)
    assert rate == pytest.approx(ref_rate)
    assert np.mean(pos >= t) >= tpr - 1e-12


def test_fpr_at_tpr_rank_is_exact():
    # 0.95 * 20 = 19 and 0.95 * 40 = 38: the threshold is the 19th / 38th largest positive, never the next.
    pos20 = np.arange(1.0, 21.0)
    assert fpr_at_tpr(np.array([0.0]), pos20, 0.95)[1] == 2.0
    pos40 = np.arange(1.0, 41.0)
    assert fpr_at_tpr(np.array([0.0]), pos40, 0.95)[1] == 3.0
    # 39 positives need ceil(37.05) = 38 of them
    assert fpr_at_tpr(np.array([0.0]), np.arange(1.0, 40.0), 0.95)[1] == 2.0
    # tpr = 1 uses the smallest positive
    assert fpr_at_tpr(np.array([0.0]), pos40, 1.0)[1] == 1.0


def test_fpr_at_tpr_default_is_95_percent():
    # docs/experiments.md: FPR at TPR 95%. With 40 positives that needs 38 of them (t = 3);
    # 90% would need only 36 (t = 5) and flag fewer negatives.
    neg = np.array([0.0, 3.5, 4.5, 10.0])
    pos = np.arange(1.0, 41.0)
    assert fpr_at_tpr(neg, pos) == (0.75, 3.0, 3)
    assert fpr_at_tpr(neg, pos) == fpr_at_tpr(neg, pos, 0.95)
    assert fpr_at_tpr(neg, pos, 0.9) == (0.25, 5.0, 1)


def test_fpr_at_tpr_counts_ties_as_flagged():
    neg = np.array([1.0, 2.0, 2.0, 3.0])
    pos = np.array([2.0, 2.0, 5.0, 6.0])
    rate, t, fp = fpr_at_tpr(neg, pos, 0.75)  # k = 3 -> t = 2.0, and both tied negatives are flagged
    assert (rate, t, fp) == (0.75, 2.0, 3)
    rate, t, fp = fpr_at_tpr(np.full(5, 1.0), np.full(3, 1.0), 0.95)  # constant scores
    assert (rate, t, fp) == (1.0, 1.0, 5)


def test_fpr_at_tpr_edge_cases():
    rate, t, fp = fpr_at_tpr(np.array([]), np.array([1.0, 2.0]), 0.95)
    assert math.isnan(rate) and t == 1.0 and fp == 0
    with pytest.raises(ValueError):
        fpr_at_tpr(np.array([1.0]), np.array([]), 0.95)
    with pytest.raises(ValueError):
        fpr_at_tpr(np.array([1.0]), np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        fpr_at_tpr(np.array([1.0]), np.array([1.0]), 1.5)


def test_rates_at_threshold_is_strict():
    neg = np.array([0.1, 0.5, 0.5, 0.9])
    pos = np.array([0.5, 0.7, 1.0])
    assert rates_at_threshold(neg, pos, 0.5) == (1, 4, 2, 3)  # scores equal to the threshold pass
    assert rates_at_threshold(neg, pos, 0.0) == (4, 4, 3, 3)
    assert rates_at_threshold(neg, pos, 1.0) == (0, 4, 0, 3)
    assert rates_at_threshold(np.array([]), np.array([]), 0.5) == (0, 0, 0, 0)
    assert all(type(v) is int for v in rates_at_threshold(neg, pos, 0.5))


def test_rates_at_threshold_float32_scores_compare_exactly():
    # The threshold is compared in float64: a float32 score equal to float32(0.1) is above 0.1.
    scores = np.array([0.1], dtype=np.float32)
    assert float(scores[0]) > 0.1
    assert rates_at_threshold(scores, scores, 0.1) == (1, 1, 1, 1)
    assert rates_at_threshold(scores, scores, float(scores[0])) == (0, 1, 0, 1)


# ---------------------------------------------------------------------------------------------------
# pixel_auroc
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(4))
def test_pixel_auroc_matches_sklearn(kind, seed):
    maps, masks = random_case(seed, kind, n=6, size=12)
    expected = roc_auc_score(masks.ravel(), maps.ravel().astype(np.float64))
    assert pixel_auroc(maps, masks) == pytest.approx(expected, abs=1e-12)
    assert pixel_auroc(maps, masks.astype(np.uint8)) == pytest.approx(expected, abs=1e-12)


def test_pixel_auroc_more_defect_than_normal_pixels():
    rng = np.random.default_rng(3)
    masks = rng.random((3, 10, 10)) < 0.8
    maps = np.round(rng.normal(size=masks.shape) + masks, 1)
    assert pixel_auroc(maps, masks) == pytest.approx(roc_auc_score(masks.ravel(), maps.ravel()), abs=1e-12)


def test_pixel_auroc_degenerate():
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    assert pixel_auroc(np.ones((2, 4, 4), dtype=np.float32), masks) == 0.5  # constant maps
    assert pixel_auroc(masks.astype(np.float32), masks) == 1.0
    assert pixel_auroc(1.0 - masks, masks) == 0.0
    assert math.isnan(pixel_auroc(np.ones((2, 4, 4)), np.zeros((2, 4, 4), dtype=bool)))
    assert math.isnan(pixel_auroc(np.ones((2, 4, 4)), np.ones((2, 4, 4), dtype=bool)))


def test_pixel_metrics_reject_bad_input():
    maps = np.zeros((2, 4, 4), dtype=np.float32)
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    for fn in (pixel_auroc, aupro, pro_histograms):
        with pytest.raises(ValueError):
            fn(maps, masks[:1])
        with pytest.raises(ValueError):
            fn(maps[0], masks[0])
        bad = maps.copy()
        bad[1, 2, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            fn(bad, masks)
    with pytest.raises(ValueError):
        aupro(maps, masks, max_fpr=0.0)
    with pytest.raises(ValueError):
        aupro(maps, masks, max_fpr=1.2)


# ---------------------------------------------------------------------------------------------------
# aupro (exact)
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(8))
def test_aupro_matches_bruteforce(kind, seed):
    maps, masks = random_case(seed, kind)
    for max_fpr in (0.3, 0.05, 1.0):
        expected = aupro_bruteforce(maps, masks, max_fpr)
        assert aupro(maps, masks, max_fpr) == pytest.approx(expected, abs=1e-10), (kind, seed, max_fpr)


def test_aupro_bruteforce_cases_are_not_trivial():
    # The random cases must exercise ties between normal and defect pixels and multi-region images.
    values = [aupro_bruteforce(*random_case(seed, kind)) for kind in KINDS for seed in range(8)]
    assert len({round(v, 6) for v in values}) > 20
    assert min(values) < 0.3 and max(values) > 0.6
    for kind in KINDS:
        maps, masks = random_case(0, kind)
        regions_per_image = [len(flood_fill_components(m)) for m in masks]
        assert max(regions_per_image) >= 2 and min(regions_per_image) == 0
    maps, masks = random_case(0, "few_values")
    assert set(np.unique(maps[masks])) & set(np.unique(maps[~masks]))


@pytest.mark.parametrize("seed", range(3))
def test_aupro_matches_fullsort_reference_on_larger_maps(seed):
    rng = np.random.default_rng(seed)
    blobs = ndimage.gaussian_filter(rng.normal(size=(40, 48, 48)), sigma=(0, 2.5, 2.5))
    masks = blobs > np.quantile(blobs, 0.96)
    masks[::3] = False
    noise = ndimage.gaussian_filter(rng.normal(size=masks.shape), sigma=(0, 1.5, 1.5))
    for dtype, decimals in ((np.float32, None), (np.float16, 2)):
        maps = noise + 0.25 * ndimage.gaussian_filter(masks.astype(float), sigma=(0, 2, 2))
        maps = (maps if decimals is None else np.round(maps, decimals)).astype(dtype)
        for max_fpr in (0.3, 0.01):
            expected = aupro_fullsort(maps, masks, max_fpr)
            assert aupro(maps, masks, max_fpr) == pytest.approx(expected, abs=1e-9)


def test_aupro_hand_computed():
    mask = np.array([[[1, 0, 0, 0]]])
    # defect pixel scores highest: PRO = 1 at FPR = 0
    assert aupro(np.array([[[4.0, 3.0, 2.0, 1.0]]]), mask) == pytest.approx(1.0)
    # one normal pixel scores higher: PRO jumps to 1 at FPR = 1/3, after the [0, 0.3] window
    maps = np.array([[[3.0, 4.0, 2.0, 1.0]]])
    assert aupro(maps, mask) == pytest.approx(0.0)
    assert aupro(maps, mask, max_fpr=0.5) == pytest.approx((0.5 - 1 / 3) / 0.5)
    # tie between the defect pixel and one normal pixel: straight line from (0, 0) to (1/3, 1)
    assert aupro(np.array([[[2.0, 2.0, 1.0, 1.0]]]), mask) == pytest.approx(0.3 * 0.9 / 2 / 0.3)
    # the defect pixel scores lowest: PRO = 0 until FPR = 1
    assert aupro(np.array([[[0.0, 3.0, 2.0, 1.0]]]), mask, max_fpr=1.0) == pytest.approx(0.0)


def test_aupro_constant_maps():
    masks = np.zeros((3, 6, 6), dtype=bool)
    masks[0, 1:3, 1:4] = True
    masks[1, 4, 4] = True
    # a single threshold: the curve is the diagonal, area 0.3^2 / 2 over 0.3
    for dtype in (np.float32, np.float16, np.float64):
        assert aupro(np.full(masks.shape, 0.7, dtype=dtype), masks) == pytest.approx(0.15)
    assert aupro(np.zeros(masks.shape), masks, max_fpr=1.0) == pytest.approx(0.5)
    assert aupro_bruteforce(np.zeros(masks.shape), masks) == pytest.approx(0.15)


def test_aupro_averages_over_regions_not_pixels():
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 0, 0] = True  # region of 1 pixel, found
    masks[0, 4:7, 4:7] = True  # region of 9 pixels, missed
    maps = np.full((1, 8, 8), 0.5)
    maps[0, 0, 0] = 1.0
    maps[0, 4:7, 4:7] = 0.0
    assert aupro(maps, masks) == pytest.approx(0.5)  # pixel-weighted overlap would give 0.1
    assert aupro(maps, masks) == pytest.approx(aupro_bruteforce(maps, masks))


def test_aupro_uses_8_connectivity():
    # Three pixels in a row plus one pixel touching only diagonally: one region of 4 pixels.
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 0, 0:3] = True
    masks[0, 1, 3] = True
    maps = np.zeros((1, 4, 4))
    maps[0][~masks[0]] = np.arange(1.0, 13.0)
    maps[0, 1, 3] = 100.0  # only the diagonal pixel is found
    assert aupro_bruteforce(maps, masks, connectivity=8) == pytest.approx(0.25)
    assert aupro_bruteforce(maps, masks, connectivity=4) == pytest.approx(0.5)
    assert aupro(maps, masks) == pytest.approx(0.25)
    h = pro_histograms(maps, masks, bins=10)
    assert h.components.shape[0] == 1 and h.components.sum() == 4


def test_aupro_regions_never_span_images():
    masks = np.zeros((3, 5, 5), dtype=bool)
    masks[:, 2, 2] = True  # same pixel in consecutive images: three regions, not one
    rng = np.random.default_rng(0)
    maps = rng.normal(size=masks.shape)
    assert aupro(maps, masks) == pytest.approx(aupro_bruteforce(maps, masks))
    h = pro_histograms(maps, masks, bins=16)
    assert h.component_image.tolist() == [0, 1, 2]
    assert h.components.sum(axis=1).tolist() == [1, 1, 1]


def test_aupro_counts_normal_images_and_normal_pixels_of_defect_images():
    rng = np.random.default_rng(5)
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 2:5, 2:5] = True
    maps = rng.normal(size=masks.shape) + 1.5 * masks
    base = aupro(maps, masks)
    assert base == pytest.approx(aupro_bruteforce(maps, masks))
    # high-scoring defect-free images add false positives and lower the value
    extra = np.full((4, 8, 8), 5.0) + rng.normal(size=(4, 8, 8)) * 0.01
    maps2 = np.concatenate([maps, extra])
    masks2 = np.concatenate([masks, np.zeros((4, 8, 8), dtype=bool)])
    assert aupro(maps2, masks2) == pytest.approx(aupro_bruteforce(maps2, masks2))
    assert aupro(maps2, masks2) < base
    # high scores on the normal pixels of the defect image itself count too
    maps3 = maps.copy()
    maps3[0, 7, :] = 50.0
    assert aupro(maps3, masks) == pytest.approx(aupro_bruteforce(maps3, masks))
    assert aupro(maps3, masks) < base


def test_aupro_without_regions_or_normal_pixels_is_nan():
    maps = np.random.default_rng(0).normal(size=(2, 4, 4))
    assert math.isnan(aupro(maps, np.zeros((2, 4, 4), dtype=bool)))
    assert math.isnan(aupro(maps, np.ones((2, 4, 4), dtype=bool)))
    h = pro_histograms(maps, np.zeros((2, 4, 4), dtype=bool), bins=8)
    assert h.components.shape == (0, 8) and h.component_image.shape == (0,)
    assert math.isnan(aupro_from_histograms(h))
    assert math.isnan(aupro_from_histograms(pro_histograms(maps, np.ones((2, 4, 4), dtype=bool), bins=8)))


def test_aupro_accepts_uint8_masks_and_does_not_modify_inputs():
    maps, masks = random_case(1, "continuous")
    maps_copy, masks_copy = maps.copy(), masks.copy()
    value = aupro(maps, masks)
    assert aupro(maps, masks.astype(np.uint8)) == value
    assert aupro(maps, masks.astype(np.uint8) * 255) == value
    pixel_auroc(maps, masks)
    pro_histograms(maps, masks, bins=32)
    assert np.array_equal(maps, maps_copy) and np.array_equal(masks, masks_copy)


# ---------------------------------------------------------------------------------------------------
# binned AUPRO
# ---------------------------------------------------------------------------------------------------


def smooth_case(seed: int, n: int = 30, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blobs = ndimage.gaussian_filter(rng.normal(size=(n, size, size)), sigma=(0, 3, 3))
    masks = blobs > np.quantile(blobs, 0.97)
    masks[: n // 2] = False
    maps = ndimage.gaussian_filter(rng.normal(size=masks.shape), sigma=(0, 2, 2))
    maps = maps + 0.15 * ndimage.gaussian_filter(masks.astype(float), sigma=(0, 2, 2))
    return maps.astype(np.float32), masks


@pytest.mark.parametrize("kind", KINDS)
def test_pro_histograms_layout(kind):
    maps, masks = random_case(2, kind)
    bins = 25
    h = pro_histograms(maps, masks, bins=bins)
    assert isinstance(h, ProHistograms)
    assert h.edges.shape == (bins + 1,)
    assert h.edges[0] == float(maps.min()) and h.edges[-1] == float(maps.max())
    assert h.normal.shape == (len(maps), bins) and h.normal.dtype == np.int64
    assert h.components.dtype == np.int64 and h.component_image.dtype == np.int64
    expected_regions = []
    for i, (image, mask) in enumerate(zip(maps, masks, strict=True)):
        # per-image counts agree with numpy's histogram on the same edges (last bin closed)
        expected = np.histogram(image[~mask].astype(np.float64), bins=h.edges)[0]
        assert np.array_equal(h.normal[i], expected)
        for region in flood_fill_components(mask):
            scores = np.array([image[y, x] for y, x in region], dtype=np.float64)
            expected_regions.append((i, np.histogram(scores, bins=h.edges)[0]))
    assert h.components.shape == (len(expected_regions), bins)
    assert h.component_image.tolist() == [i for i, _ in expected_regions]
    for row, (_, expected) in zip(h.components, expected_regions, strict=True):
        assert np.array_equal(row, expected)
    assert h.normal.sum() + h.components.sum() == maps.size


def test_pro_histograms_range_handling():
    maps = np.array([[[0.0, 1.0, 2.0, 4.0]]])
    masks = np.array([[[0, 0, 0, 1]]])
    h = pro_histograms(maps, masks, bins=4)
    assert h.edges.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert h.normal.tolist() == [[1, 1, 1, 0]]
    assert h.components.tolist() == [[0, 0, 0, 1]]  # the maximum falls in the last bin
    # explicit range: values outside are clipped into the first / last bin
    h = pro_histograms(maps, masks, bins=2, lo=1.0, hi=3.0)
    assert h.edges.tolist() == [1.0, 2.0, 3.0]
    assert h.normal.tolist() == [[2, 1]]
    assert h.components.tolist() == [[0, 1]]
    with pytest.raises(ValueError):
        pro_histograms(maps, masks, bins=4, lo=2.0, hi=1.0)
    with pytest.raises(ValueError):
        pro_histograms(maps, masks, bins=0)


def test_pro_histograms_default_is_2000_bins_over_the_score_range():
    maps, masks = random_case(3, "continuous")
    lo, hi = float(maps.min()), float(maps.max())
    h = pro_histograms(maps, masks)
    assert h.edges.shape == (2001,)
    assert np.array_equal(h.edges, np.linspace(lo, hi, 2001))
    assert h.normal.shape == (len(maps), 2000) and h.components.shape[1] == 2000
    explicit = pro_histograms(maps, masks, bins=2000, lo=lo, hi=hi)
    assert np.array_equal(h.normal, explicit.normal) and np.array_equal(h.components, explicit.components)


def test_pro_histograms_scores_on_bin_edges():
    # linspace(0, 1, 11) has edges such as 0.30000000000000004. The scores 0.3, 0.6 and 0.7 lie just below
    # their edge, so they belong to the bin before it; floor((v - lo) * bins / (hi - lo)) alone puts them
    # one bin too high.
    values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    expected = [1, 1, 2, 0, 1, 2, 1, 0, 1, 2]
    maps = np.stack([values.reshape(1, 11), values.reshape(1, 11)])
    masks = np.zeros((2, 1, 11), dtype=bool)
    masks[1] = True  # the second image is one defect region with the same scores
    h = pro_histograms(maps, masks, bins=10)
    assert np.histogram(values, bins=h.edges)[0].tolist() == expected
    assert h.normal.tolist() == [expected, [0] * 10]
    assert h.components.tolist() == [expected]


@pytest.mark.parametrize(
    "lo,hi,bins",
    [
        (0.0, 1.0, 10),  # plain floor() is one bin too high for some scores
        (3.0, 3.3, 100),  # ... one bin too low
        (0.1, 0.7, 49),  # ... both
        (0.0, 1.0, 2000),
        (-0.219, 36.266, 2000),
    ],
)
def test_pro_histograms_counts_match_edges_for_scores_next_to_an_edge(lo, hi, bins):
    # Every edge, and the floats just below and just above it: the counts must describe `edges` exactly
    # (edges[i] <= v < edges[i + 1], last bin closed, scores outside [lo, hi] in the first / last bin).
    edges = np.linspace(lo, hi, bins + 1)
    values = np.concatenate([edges, np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf)])
    bin_of = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bins - 1)
    expected = np.bincount(bin_of, minlength=bins)
    # the plain floor() formula really is wrong for some of these values: the case is not vacuous
    plain = np.floor((np.clip(values, lo, hi) - lo) * (bins / (hi - lo))).astype(np.int64)
    assert (np.clip(plain, 0, bins - 1) != bin_of).any()

    maps = np.stack([values.reshape(1, -1), values.reshape(1, -1)])
    masks = np.zeros(maps.shape, dtype=bool)
    masks[1] = True
    h = pro_histograms(maps, masks, bins=bins, lo=lo, hi=hi)
    assert np.array_equal(h.edges, edges)
    assert np.array_equal(h.normal[0], expected) and not h.normal[1].any()
    assert h.components.shape == (1, bins) and np.array_equal(h.components[0], expected)


def test_binned_aupro_constant_maps():
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1:3, 1:3] = True
    h = pro_histograms(np.full(masks.shape, 3.0, dtype=np.float32), masks, bins=50)
    assert h.normal.sum() == 46 and h.components.sum() == 4
    assert aupro_from_histograms(h) == pytest.approx(0.15)


@pytest.mark.parametrize("seed", range(5))
def test_binned_aupro_is_exact_when_every_bin_holds_one_value(seed):
    rng = np.random.default_rng(seed)
    masks = rng.random((6, 10, 10)) < 0.15
    masks[0] = False
    maps = np.clip(np.round(rng.normal(4.5, 2.0, size=masks.shape) + 2.0 * masks), 0, 9).astype(np.float32)
    maps[0, 0, 0], maps[0, 0, 1] = 0.0, 9.0
    h = pro_histograms(maps, masks, bins=10)  # edges 0, 0.9, ..., 9: one integer per bin
    for max_fpr in (0.3, 1.0):
        assert aupro_from_histograms(h, max_fpr=max_fpr) == pytest.approx(
            aupro(maps, masks, max_fpr), abs=1e-12
        )
        assert aupro_from_histograms(h, max_fpr=max_fpr) == pytest.approx(
            aupro_bruteforce(maps, masks, max_fpr), abs=1e-10
        )


@pytest.mark.parametrize("seed", range(4))
def test_binned_aupro_close_to_exact_on_smooth_maps(seed):
    maps, masks = smooth_case(seed)
    exact = aupro(maps, masks)
    assert 0.2 < exact < 0.98  # an informative case, not a saturated one
    binned = aupro_from_histograms(pro_histograms(maps, masks))
    assert abs(binned - exact) < 0.002
    # fewer bins are visibly coarser but still converge
    coarse = aupro_from_histograms(pro_histograms(maps, masks, bins=20))
    assert abs(coarse - exact) < 0.05


@pytest.mark.parametrize("seed", range(4))
def test_binned_aupro_image_counts_equal_replicated_images(seed):
    maps, masks = smooth_case(seed, n=16, size=40)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(maps), size=len(maps))  # one bootstrap sample of images
    counts = np.bincount(idx, minlength=len(maps))
    assert (counts == 0).any() and (counts > 1).any()
    h = pro_histograms(maps, masks)
    value = aupro_from_histograms(h, counts)
    # same histogram range on the replicated images: identical up to rounding
    lo, hi = float(maps.min()), float(maps.max())
    replicated = pro_histograms(maps[idx], masks[idx], lo=lo, hi=hi)
    assert value == pytest.approx(aupro_from_histograms(replicated), abs=1e-12)
    # and close to the exact value on the replicated images
    assert abs(value - aupro(maps[idx], masks[idx])) < 0.002
    # counts of one are the default
    assert aupro_from_histograms(h, np.ones(len(maps), dtype=int)) == aupro_from_histograms(h)


def test_binned_aupro_image_counts_edge_cases():
    maps, masks = smooth_case(0, n=8, size=32)
    h = pro_histograms(maps, masks, bins=200)
    has_defect = masks.any(axis=(1, 2))
    assert has_defect.any() and not has_defect.all()
    assert math.isnan(aupro_from_histograms(h, (~has_defect).astype(int)))  # no region in the sample
    assert math.isnan(aupro_from_histograms(h, np.zeros(len(maps), dtype=int)))
    with pytest.raises(ValueError):
        aupro_from_histograms(h, np.ones(len(maps) + 1))
    with pytest.raises(ValueError):
        aupro_from_histograms(h, -np.ones(len(maps)))
    # doubling every image changes nothing
    assert aupro_from_histograms(h, np.full(len(maps), 2)) == pytest.approx(
        aupro_from_histograms(h), abs=1e-12
    )


def binned_aupro_loop(h: ProHistograms, counts: np.ndarray, max_fpr: float = 0.3) -> float:
    """Plain-loop reference for the binned curve: add one bin at a time, from the top bin downwards."""
    n_regions = sum(int(counts[i]) for i in h.component_image.tolist())
    n_normal = sum(int(c) * int(row.sum()) for c, row in zip(counts.tolist(), h.normal, strict=True))
    if n_regions == 0 or n_normal == 0:
        return float("nan")
    points, fp, overlap = [(0.0, 0.0)], 0, 0.0
    for b in range(h.normal.shape[1] - 1, -1, -1):
        fp += sum(int(c) * int(row[b]) for c, row in zip(counts.tolist(), h.normal, strict=True))
        for row, image in zip(h.components, h.component_image.tolist(), strict=True):
            overlap += int(counts[image]) * int(row[b]) / int(row.sum())
        points.append((fp / n_normal, overlap / n_regions))
    return clipped_area(points, max_fpr) / max_fpr


@pytest.mark.parametrize("seed", range(3))
def test_binned_aupro_matches_plain_loop(seed):
    maps, masks = random_case(seed, "float16", n=6, size=12)
    h = pro_histograms(maps, masks, bins=7)  # coarse bins: many ties between normal and defect pixels
    rng = np.random.default_rng(seed)
    for max_fpr in (0.3, 1.0):
        ones = np.ones(len(maps), dtype=int)
        assert aupro_from_histograms(h, max_fpr=max_fpr) == pytest.approx(
            binned_aupro_loop(h, ones, max_fpr), abs=1e-12
        )
        for _ in range(5):
            counts = rng.integers(0, 4, size=len(maps))
            expected = binned_aupro_loop(h, counts, max_fpr)
            value = aupro_from_histograms(h, counts, max_fpr)
            assert (math.isnan(value) and math.isnan(expected)) or value == pytest.approx(expected, abs=1e-12)


def test_aupro_bootstrap_matches_single_calls():
    maps, masks = smooth_case(1, n=12, size=32)
    has_defect = masks.any(axis=(1, 2))
    h = pro_histograms(maps, masks, bins=300)
    rng = np.random.default_rng(0)
    n_boot = 300  # more than one internal chunk
    idx = rng.integers(0, len(maps), size=(n_boot, len(maps)))
    counts = np.stack([np.bincount(row, minlength=len(maps)) for row in idx])
    counts[7] = ~has_defect  # a resample without any region
    counts[8] = 0
    out = aupro_bootstrap(h, counts)
    assert out.shape == (n_boot,) and out.dtype == np.float64
    assert np.isnan(out[7]) and np.isnan(out[8]) and np.isnan(out).sum() <= 2 + (n_boot // 50)
    for b in range(n_boot):
        single = aupro_from_histograms(h, counts[b])
        assert (math.isnan(single) and math.isnan(out[b])) or out[b] == pytest.approx(single, abs=1e-12)
    # rows are independent of each other and of the chunking
    assert np.array_equal(aupro_bootstrap(h, counts[5:6]), out[5:6])
    # against the exact value on explicitly replicated images
    for b in (0, 1, 2):
        assert abs(out[b] - aupro(maps[idx[b]], masks[idx[b]])) < 0.01
    assert np.allclose(
        aupro_bootstrap(h, counts, max_fpr=1.0)[:3],
        [aupro_from_histograms(h, counts[b], max_fpr=1.0) for b in range(3)],
        atol=1e-12,
        rtol=0,
    )
    assert aupro_bootstrap(h, np.zeros((0, len(maps)))).shape == (0,)
    with pytest.raises(ValueError):
        aupro_bootstrap(h, counts[0])
    with pytest.raises(ValueError):
        aupro_bootstrap(h, counts[:, :-1])
    with pytest.raises(ValueError):
        aupro_bootstrap(h, -counts)

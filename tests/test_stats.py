"""Tests for bootstrap resampling, intervals and the verdict rule."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom, binomtest
from sklearn.metrics import roc_auc_score

from defect_inspect.metrics import auroc
from defect_inspect.stats import (
    clopper_pearson,
    cluster_indices,
    macro_auroc_bootstrap,
    percentile_ci,
    pooled_rate_bootstrap,
    stratified_indices,
    verdict,
)

# ---------------------------------------------------------------------------------------------------
# stratified_indices
# ---------------------------------------------------------------------------------------------------


def test_stratified_indices_shapes_and_ranges():
    sizes = [7, 0, 3, 1]
    out = stratified_indices(sizes, n_boot=50, seed=3)
    assert len(out) == len(sizes)
    for idx, size in zip(out, sizes, strict=True):
        assert idx.shape == (50, size)
        assert idx.dtype == np.int64
        if size:
            assert idx.min() >= 0 and idx.max() < size
    assert set(np.unique(out[0])) == set(range(7))
    assert (out[3] == 0).all()


def test_stratified_indices_are_deterministic_and_paired():
    a = stratified_indices([300, 40, 200, 40], n_boot=20, seed=0)
    b = stratified_indices([300, 40, 200, 40], n_boot=20, seed=0)
    c = stratified_indices([300, 40, 200, 40], n_boot=20, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0], c[0])
    # groups are drawn in order from one generator: an empty group does not shift the others
    d = stratified_indices([300, 0, 40], n_boot=20, seed=0)
    assert np.array_equal(d[0], a[0]) and np.array_equal(d[2], a[1])
    with pytest.raises(ValueError):
        stratified_indices([3, -1])


def test_stratified_indices_groups_are_independent_draws_from_one_stream():
    # The contract: one default_rng(seed), then one [n_boot, size] draw per group, in order.
    sizes = [40, 40, 7, 40]
    out = stratified_indices(sizes, n_boot=25, seed=11)
    rng = np.random.default_rng(11)
    for idx, size in zip(out, sizes, strict=True):
        assert np.array_equal(idx, rng.integers(0, size, size=(25, size)))
    # groups of equal size must not share their indices (that would correlate the categories)
    assert not np.array_equal(out[0], out[1]) and not np.array_equal(out[1], out[3])
    corr = np.corrcoef(out[0].ravel(), out[1].ravel())[0, 1]
    assert abs(corr) < 0.1


def test_stratified_indices_resample_uniformly():
    idx = stratified_indices([5], n_boot=20000, seed=0)[0]
    freq = np.bincount(idx.ravel(), minlength=5) / idx.size
    assert np.allclose(freq, 0.2, atol=0.01)
    # a bootstrap sample of size 5 contains on average 5 * (1 - 0.8^5) distinct items
    distinct = np.mean([len(set(row)) for row in idx.tolist()])
    assert distinct == pytest.approx(5 * (1 - 0.8**5), abs=0.03)


def test_stratified_indices_defaults_are_2000_resamples_with_seed_0():
    # docs/experiments.md: 2,000 bootstrap resamples, seed 0.
    out = stratified_indices([5, 3])
    rng = np.random.default_rng(0)
    assert np.array_equal(out[0], rng.integers(0, 5, size=(2000, 5)))
    assert np.array_equal(out[1], rng.integers(0, 3, size=(2000, 3)))
    assert not np.array_equal(out[0], stratified_indices([5, 3], seed=1)[0])


# ---------------------------------------------------------------------------------------------------
# cluster_indices
# ---------------------------------------------------------------------------------------------------


def test_cluster_indices_draws_whole_clusters():
    ids = np.array(["b", "a", "b", "c", "a", "b", "d"])
    rows_of = {c: np.flatnonzero(ids == c).tolist() for c in "abcd"}
    out = cluster_indices(ids, n_boot=300, seed=0)
    assert len(out) == 300
    lengths = set()
    for idx in out:
        assert idx.dtype == np.int64
        # split the result back into whole clusters, in draw order
        pos, drawn = 0, []
        while pos < len(idx):
            cluster = ids[idx[pos]]
            rows = rows_of[cluster]
            assert idx[pos : pos + len(rows)].tolist() == rows
            drawn.append(cluster)
            pos += len(rows)
        assert len(drawn) == 4  # as many draws as distinct clusters
        lengths.add(len(idx))
    assert len(lengths) > 1  # lengths vary because cluster sizes differ
    assert min(lengths) >= 4 and max(lengths) <= 12


def test_cluster_indices_deterministic_and_uniform_over_clusters():
    ids = np.repeat(np.arange(6), [1, 2, 3, 1, 2, 3])
    a = cluster_indices(ids, n_boot=4000, seed=0)
    b = cluster_indices(ids, n_boot=4000, seed=0)
    c = cluster_indices(ids, n_boot=4000, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert any(not np.array_equal(x, y) for x, y in zip(a, c, strict=True))
    # each cluster is drawn once per resample on average, whatever its size
    first_rows = [np.flatnonzero(ids == k)[0] for k in range(6)]
    draws = np.array([[np.sum(idx == r) for r in first_rows] for idx in a])
    assert np.allclose(draws.mean(axis=0), 1.0, atol=0.06)
    assert (draws.sum(axis=1) == 6).all()


def test_cluster_indices_single_rows_and_empty():
    out = cluster_indices(np.arange(5), n_boot=10, seed=0)
    assert all(len(idx) == 5 and idx.min() >= 0 and idx.max() < 5 for idx in out)
    out = cluster_indices(np.array([]), n_boot=3, seed=0)
    assert [idx.tolist() for idx in out] == [[], [], []]
    out = cluster_indices(np.array([7, 7, 7]), n_boot=4, seed=0)
    assert all(idx.tolist() == [0, 1, 2] for idx in out)


def test_cluster_indices_defaults_are_2000_resamples_with_seed_0():
    # One-row clusters with sorted ids: the rows of a resample are the cluster draws themselves.
    out = cluster_indices(np.arange(5))
    assert len(out) == 2000
    assert np.array_equal(np.stack(out), np.random.default_rng(0).integers(0, 5, size=(2000, 5)))
    assert not np.array_equal(np.stack(out), np.stack(cluster_indices(np.arange(5), seed=1)))


# ---------------------------------------------------------------------------------------------------
# percentile_ci, clopper_pearson
# ---------------------------------------------------------------------------------------------------


def test_percentile_ci():
    samples = np.arange(101.0)
    assert percentile_ci(samples) == pytest.approx((2.5, 97.5))
    assert percentile_ci(samples, conf=0.5) == pytest.approx((25.0, 75.0))
    with_nan = np.concatenate([samples, [np.nan, np.nan]])
    assert percentile_ci(with_nan) == pytest.approx((2.5, 97.5))
    lo, hi = percentile_ci(np.array([np.nan, np.nan]))
    assert math.isnan(lo) and math.isnan(hi)
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert percentile_ci(x) == pytest.approx(tuple(np.percentile(x, [2.5, 97.5])))
    assert all(type(v) is float for v in percentile_ci(x))
    with pytest.raises(ValueError):
        percentile_ci(x, conf=1.0)


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (1, 10), (5, 40), (38, 40), (192, 3848), (0, 1), (1, 1)])
@pytest.mark.parametrize("conf", [0.95, 0.9])
def test_clopper_pearson_matches_scipy_binomtest(k, n, conf):
    lo, hi = clopper_pearson(k, n, conf)
    ref = binomtest(k, n).proportion_ci(confidence_level=conf, method="exact")
    assert lo == pytest.approx(ref.low, abs=1e-12)
    assert hi == pytest.approx(ref.high, abs=1e-12)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


@pytest.mark.parametrize("k,n", [(1, 10), (5, 40), (38, 40), (192, 3848)])
def test_clopper_pearson_tail_definition(k, n):
    # By definition: P(X >= k | p = lo) = 0.025 and P(X <= k | p = hi) = 0.025.
    lo, hi = clopper_pearson(k, n)
    assert binom.sf(k - 1, n, lo) == pytest.approx(0.025, rel=1e-6)
    assert binom.cdf(k, n, hi) == pytest.approx(0.025, rel=1e-6)


def test_clopper_pearson_edges():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 10))
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10))
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson(-1, 2)


# ---------------------------------------------------------------------------------------------------
# macro_auroc_bootstrap
# ---------------------------------------------------------------------------------------------------


def make_scores(seed: int, n_cat: int, ties: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    negs, poss = [], []
    for c in range(n_cat):
        neg = rng.normal(size=int(rng.integers(5, 40)))
        pos = rng.normal(loc=0.3 + 0.4 * c, size=int(rng.integers(3, 15)))
        if ties:
            neg, pos = np.round(neg), np.round(pos)
        negs.append(neg.astype(np.float32))
        poss.append(pos.astype(np.float32))
    return negs, poss


@pytest.mark.parametrize("ties", [False, True])
def test_macro_auroc_bootstrap_matches_explicit_loop(ties):
    negs, poss = make_scores(0, n_cat=3, ties=ties)
    n_boot = 60
    out = macro_auroc_bootstrap(negs, poss, n_boot=n_boot, seed=5)
    assert out.shape == (n_boot,)
    sizes = [len(a) for pair in zip(negs, poss, strict=True) for a in pair]
    indices = stratified_indices(sizes, n_boot=n_boot, seed=5)
    expected = np.zeros(n_boot)
    for b in range(n_boot):
        values = []
        for c, (neg, pos) in enumerate(zip(negs, poss, strict=True)):
            neg_b, pos_b = neg[indices[2 * c][b]], pos[indices[2 * c + 1][b]]
            y = np.concatenate([np.zeros(len(neg_b)), np.ones(len(pos_b))])
            reference = roc_auc_score(y, np.concatenate([neg_b, pos_b]))
            assert auroc(neg_b, pos_b) == pytest.approx(reference, abs=1e-12)
            values.append(reference)
        expected[b] = np.mean(values)
    assert np.allclose(out, expected, atol=1e-12, rtol=0)


def test_macro_auroc_bootstrap_is_paired_and_centred():
    negs, poss = make_scores(1, n_cat=4, ties=False)
    a = macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=0)
    assert np.array_equal(a, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=0))
    assert not np.array_equal(a, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=1))
    point = np.mean([auroc(n, p) for n, p in zip(negs, poss, strict=True)])
    assert a.mean() == pytest.approx(point, abs=0.01)
    lo, hi = percentile_ci(a)
    assert lo < point < hi
    # a second method scored on the same images uses the same resamples: a constant shift of the scores
    # leaves every resample's AUROC unchanged, so the paired difference is exactly zero
    shifted = macro_auroc_bootstrap([n + 3 for n in negs], [p + 3 for p in poss], n_boot=2000, seed=0)
    assert np.array_equal(a, shifted)


def test_macro_auroc_bootstrap_defaults_are_2000_resamples_with_seed_0():
    negs, poss = make_scores(2, n_cat=2, ties=False)
    out = macro_auroc_bootstrap(negs, poss)
    assert out.shape == (2000,)
    assert np.array_equal(out, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=0))
    assert not np.array_equal(out, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=1))
    # the seed-0 stream itself: one default_rng(0), groups in the order neg_0, pos_0, neg_1, pos_1
    rng = np.random.default_rng(0)
    (neg0, pos0), (neg1, pos1) = zip(negs, poss, strict=True)
    i0, j0, i1, j1 = (rng.integers(0, len(a), size=(2000, len(a))) for a in (neg0, pos0, neg1, pos1))
    for b in (0, 1, 1999):
        expected = (auroc(neg0[i0[b]], pos0[j0[b]]) + auroc(neg1[i1[b]], pos1[j1[b]])) / 2
        assert out[b] == pytest.approx(expected, abs=1e-12)


def test_macro_auroc_bootstrap_degenerate_inputs():
    # perfectly separated and constant categories
    out = macro_auroc_bootstrap([np.zeros(5), np.ones(4)], [np.ones(3), np.ones(2)], n_boot=30, seed=0)
    assert np.allclose(out, (1.0 + 0.5) / 2)
    # a category without positives has no AUROC
    out = macro_auroc_bootstrap([np.zeros(5), np.zeros(4)], [np.ones(3), np.array([])], n_boot=10, seed=0)
    assert out.shape == (10,) and np.isnan(out).all()
    assert np.isnan(macro_auroc_bootstrap([], [], n_boot=5)).all()
    with pytest.raises(ValueError):
        macro_auroc_bootstrap([np.zeros(3)], [], n_boot=5)
    with pytest.raises(ValueError, match="NaN"):
        macro_auroc_bootstrap([np.array([0.0, np.nan])], [np.ones(2)], n_boot=5)


# ---------------------------------------------------------------------------------------------------
# pooled_rate_bootstrap
# ---------------------------------------------------------------------------------------------------


def test_pooled_rate_bootstrap_matches_explicit_loop():
    rng = np.random.default_rng(0)
    flags = [rng.random(n) < p for n, p in ((30, 0.1), (50, 0.4), (8, 0.0), (0, 0.5))]
    n_boot = 80
    out = pooled_rate_bootstrap(flags, n_boot=n_boot, seed=2)
    indices = stratified_indices([len(f) for f in flags], n_boot=n_boot, seed=2)
    expected = [
        sum(int(f[idx[b]].sum()) for f, idx in zip(flags, indices, strict=True)) / 88 for b in range(n_boot)
    ]
    assert out.shape == (n_boot,)
    assert np.allclose(out, expected, atol=1e-15, rtol=0)
    # 0/1 integers and floats are accepted as flags
    assert np.array_equal(out, pooled_rate_bootstrap([f.astype(int) for f in flags], n_boot=n_boot, seed=2))
    assert np.array_equal(out, pooled_rate_bootstrap([f.astype(float) for f in flags], n_boot=n_boot, seed=2))


def test_pooled_rate_bootstrap_distribution():
    rng = np.random.default_rng(1)
    flags = [rng.random(300) < 0.05, rng.random(200) < 0.10]
    point = (flags[0].sum() + flags[1].sum()) / 500
    out = pooled_rate_bootstrap(flags)
    assert out.shape == (2000,)
    assert out.mean() == pytest.approx(point, abs=0.002)
    # stratified: the variance is the sum of the per-category binomial variances
    p0, p1 = flags[0].mean(), flags[1].mean()
    expected_sd = math.sqrt(300 * p0 * (1 - p0) + 200 * p1 * (1 - p1)) / 500
    assert out.std() == pytest.approx(expected_sd, rel=0.1)
    assert np.array_equal(out, pooled_rate_bootstrap(flags, seed=0))
    assert (pooled_rate_bootstrap([np.zeros(10, dtype=bool)], n_boot=20) == 0).all()
    assert (pooled_rate_bootstrap([np.ones(10, dtype=bool)], n_boot=20) == 1).all()
    assert np.isnan(pooled_rate_bootstrap([np.array([], dtype=bool)], n_boot=20)).all()
    assert np.isnan(pooled_rate_bootstrap([], n_boot=20)).all()


# ---------------------------------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------------------------------


def test_verdict_three_outcomes():
    rng = np.random.default_rng(0)
    big = rng.normal(0.05, 0.005, size=2000)
    small = rng.normal(0.005, 0.001, size=2000)
    unclear = rng.normal(0.05, 0.05, size=2000)
    assert verdict(big, 0.05, 0.01) == "차이 있음"
    assert verdict(-big, -0.05, 0.01) == "차이 있음"  # direction does not matter
    assert verdict(small, 0.005, 0.01) == "차이 작음"
    assert verdict(-small, -0.005, 0.01) == "차이 작음"
    assert verdict(unclear, 0.05, 0.01) == "판정 불가"
    assert verdict(big, 0.01, 0.01) == "차이 있음"  # exactly the minimum effect counts


def test_verdict_interval_touching_zero_is_inconclusive():
    samples = np.concatenate([np.zeros(100), np.full(1900, 0.1)])  # 2.5th percentile is exactly 0
    assert percentile_ci(samples)[0] == 0.0
    assert verdict(samples, 0.1, 0.01) == "판정 불가"
    assert verdict(np.full(10, np.nan), 0.1, 0.01) == "판정 불가"
    assert verdict(np.zeros(100), 0.0, 0.01) == "판정 불가"


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("across,excludes_zero", [(40, True), (50, True), (51, False), (60, False)])
def test_verdict_uses_the_95_percent_interval(sign, across, excludes_zero):
    # 2,000 resamples of which `across` lie on the other side of zero. The 2.5th percentile sits between
    # the 50th and 51st smallest, so 2% (and exactly 2.5%) across still excludes zero and 3% does not.
    # A 99% interval would call the 2% case inconclusive; a 90% interval would call the 3% case a difference.
    samples = sign * np.concatenate([np.full(across, -1.0), np.full(2000 - across, 1.0)])
    lo, hi = percentile_ci(samples)
    assert (lo > 0 or hi < 0) == excludes_zero
    assert verdict(samples, sign * 1.0, 0.01) == ("차이 있음" if excludes_zero else "판정 불가")
    assert verdict(samples, sign * 0.005, 0.01) == ("차이 작음" if excludes_zero else "판정 불가")


def test_verdict_nan_point_is_inconclusive():
    # An undefined point estimate is never reported as a difference, not even a small one.
    clear = np.full(100, 0.5)
    assert verdict(clear, 0.5, 0.01) == "차이 있음"
    assert verdict(clear, 0.005, 0.01) == "차이 작음"
    for nan in (float("nan"), np.float64("nan"), np.float32("nan")):
        assert verdict(clear, nan, 0.01) == "판정 불가"
        assert verdict(-clear, nan, 0.01) == "판정 불가"
    assert verdict(clear, float("nan"), 0.0) == "판정 불가"

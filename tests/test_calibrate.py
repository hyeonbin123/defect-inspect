"""Threshold calibration: exact rank arithmetic, and the theory claims checked by seeded simulation."""

from __future__ import annotations

import dataclasses
import json
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from defect_inspect.calibrate import (
    Threshold,
    conformal_rank,
    conformal_threshold,
    fpr_beta_params,
    pooled_fpr_band,
    subsample_fpr_curve,
)

# ---------------------------------------------------------------------------------------------------------
# Rank arithmetic
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "alpha", "expected"),
    [
        (19, 0.05, 19),  # 20 * 0.95 is exactly 19: must not round up to 20
        (99, 0.01, 99),  # 100 * 0.99 is exactly 99
        (39, 0.05, 38),
        (199, 0.01, 198),
        (999, 0.001, 999),
        (9, 0.1, 9),
        (20, 0.05, 20),  # ceil(19.95)
        (50, 0.05, 49),  # ceil(48.45)
        (100, 0.05, 96),  # ceil(95.95)
        (10, 0.05, 11),  # ceil(10.45) > n: not attainable
        (0, 0.05, 1),
    ],
)
def test_conformal_rank_values(n: int, alpha: float, expected: int) -> None:
    rank = conformal_rank(n, alpha)
    assert rank == expected
    assert isinstance(rank, int)


def test_conformal_rank_is_exact_where_float_ceil_is_not() -> None:
    # 150 * 0.82 is exactly 123, but the float product lands just above it.
    assert math.ceil(150 * (1 - 0.18)) == 124
    assert conformal_rank(149, 0.18) == 123
    assert conformal_rank(249, 0.18) == 205


def test_conformal_rank_matches_integer_arithmetic() -> None:
    cases = [
        (0.05, 95, 100),
        (0.01, 99, 100),
        (0.1, 9, 10),
        (0.2, 8, 10),
        (0.001, 999, 1000),
        (0.18, 82, 100),
    ]
    for alpha, num, den in cases:
        for n in range(0, 1200):
            assert conformal_rank(n, alpha) == -(-(n + 1) * num // den), (n, alpha)


def test_conformal_rank_accepts_numpy_and_fraction_alpha() -> None:
    assert conformal_rank(19, np.float64(0.05)) == 19
    assert conformal_rank(19, Fraction(1, 20)) == 19
    assert conformal_rank(19, Decimal("0.05")) == 19
    assert conformal_rank(19, "0.05") == 19
    assert conformal_rank(19, np.array(0.05)) == 19
    # Lower precision floats are read in their own shortest decimal form (float32(0.01) is below 0.01).
    assert float(np.float32(0.01)) < 0.01
    assert conformal_rank(99, np.float32(0.01)) == 99
    assert conformal_rank(19, np.float32(0.05)) == 19
    assert conformal_rank(np.int64(19), 0.05) == 19


@pytest.mark.parametrize(
    ("n", "alpha", "expected"),
    [
        (19, 0.15 - 0.1, 19),  # 0.04999999999999999: must behave like 0.05, not give l = 20 > n
        (39, 0.15 - 0.1, 38),
        (599, 0.15 - 0.1, 570),
        (19, 0.06 - 0.01, 19),
        (19, 0.3 - 0.25, 19),
        (9, 1 - 0.9, 9),  # 0.09999999999999998
        (599, 1 - 0.9, 540),
        (19, 1 - 0.95, 19),  # 0.050000000000000044
        (99, 1 - 0.99, 99),  # 0.010000000000000009
        (99, 0.11 - 0.1, 99),  # 0.009999999999999995
    ],
)
def test_conformal_rank_tidies_computed_alphas(n: int, alpha: float, expected: int) -> None:
    assert alpha not in (0.05, 0.1, 0.01)  # the float really is off its decimal value
    assert conformal_rank(n, alpha) == expected


def test_computed_alpha_keeps_the_guarantee_and_the_beta_law() -> None:
    alpha = 0.15 - 0.1
    thr = conformal_threshold(np.arange(19), alpha)
    assert (thr.value, thr.rank, thr.n, thr.guaranteed) == (18.0, 19, 19, True)
    assert fpr_beta_params(19, alpha) == (1.0, 19.0)
    assert fpr_beta_params(100, alpha) == fpr_beta_params(100, 0.05) == (5.0, 96.0)


def test_conformal_rank_keeps_exact_rationals_exact() -> None:
    # 3 * (1 - 1/3) is exactly 2; a 12-digit decimal rendering of 1/3 would give 3.
    assert conformal_rank(2, Fraction(1, 3)) == 2
    assert conformal_rank(8, Fraction(1, 3)) == 6
    assert conformal_rank(6, Fraction(1, 7)) == 6
    # A float that is close to but genuinely different from a round value is not snapped to it:
    # twelve significant digits are kept, the thirteenth and beyond are treated as float noise.
    assert conformal_rank(19, 0.0499) == 20
    assert conformal_rank(10**6 - 1, 1e-6) == 10**6 - 1
    assert conformal_rank(10**8 - 1, 0.05000001) == 94_999_999  # 0.05 would give 95_000_000
    assert conformal_rank(10**13 - 1, 0.0500000000001) == 9_499_999_999_999  # 12 digits survive
    assert conformal_rank(19, 0.05 - 1e-14) == 19  # 0.04999999999999: the 13th digit does not


@pytest.mark.parametrize(
    "alpha", [0, 1, 0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), True, False, None, "x", [0.05]]
)
def test_conformal_rank_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        conformal_rank(10, alpha)


def test_conformal_rank_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        conformal_rank(-1, 0.05)


@pytest.mark.parametrize("n", [19.5, 19.0, "19", None, np.float64(19.0)])
def test_conformal_rank_rejects_non_integer_n(n: object) -> None:
    # A fractional count used to be truncated silently.
    with pytest.raises(TypeError):
        conformal_rank(n, 0.05)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fpr_beta_params(n, 0.05)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------------
# conformal_threshold
# ---------------------------------------------------------------------------------------------------------


def test_threshold_is_the_lth_smallest_score() -> None:
    rng = np.random.default_rng(0)
    scores = rng.permutation(np.arange(1, 101)).astype(np.float32)  # 1..100, shuffled
    thr = conformal_threshold(scores, 0.05)
    assert thr == Threshold(value=96.0, rank=96, n=100, alpha=0.05, guaranteed=True)
    assert isinstance(thr.value, float)
    # Decision rule is strictly greater: exactly 4 of the calibration scores are flagged.
    assert int((scores > thr.value).sum()) == 4
    assert conformal_threshold(scores, 0.2).value == 81.0  # ceil(101 * 0.8) = 81


def test_threshold_exact_integer_edges() -> None:
    thr = conformal_threshold(np.arange(19, 0, -1), 0.05)  # n = 19: l = 19 (the maximum), still guaranteed
    assert (thr.rank, thr.value, thr.guaranteed) == (19, 19.0, True)
    thr = conformal_threshold(np.arange(1, 100), 0.01)  # n = 99: l = 99
    assert (thr.rank, thr.value, thr.guaranteed) == (99, 99.0, True)
    thr = conformal_threshold(np.arange(1, 21), 0.05)  # n = 20: l = ceil(19.95) = 20
    assert (thr.rank, thr.value, thr.guaranteed) == (20, 20.0, True)


def test_threshold_not_guaranteed_uses_the_maximum() -> None:
    scores = np.array([0.3, 0.9, 0.1, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8, 0.0])
    thr = conformal_threshold(scores, 0.05)  # n = 10: l = 11 > n
    assert thr == Threshold(value=0.9, rank=10, n=10, alpha=0.05, guaranteed=False)
    assert fpr_beta_params(10, 0.05) is None
    # n = 18 is the largest n without a guarantee at alpha = 0.05, n = 98 at alpha = 0.01.
    assert not conformal_threshold(np.arange(18), 0.05).guaranteed
    assert conformal_threshold(np.arange(19), 0.05).guaranteed
    assert not conformal_threshold(np.arange(98), 0.01).guaranteed
    assert conformal_threshold(np.arange(99), 0.01).guaranteed


def test_threshold_with_ties_and_input_is_untouched() -> None:
    scores = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0])
    before = scores.copy()
    thr = conformal_threshold(scores, 0.5)  # l = ceil(10 * 0.5) = 5 -> sorted[4] = 2.0
    assert (thr.rank, thr.value) == (5, 2.0)
    assert np.array_equal(scores, before)
    # Ties only make the rule more conservative: the flagged fraction stays below alpha.
    assert np.mean(scores > thr.value) <= 0.5


def test_threshold_is_frozen_and_validates_input() -> None:
    thr = conformal_threshold(np.arange(30.0), 0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        thr.value = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        conformal_threshold(np.array([]), 0.05)
    with pytest.raises(ValueError):
        conformal_threshold(np.array([0.1, np.nan]), 0.05)
    with pytest.raises(ValueError):
        conformal_threshold(np.arange(30.0), 0.0)


def test_fpr_beta_params() -> None:
    assert fpr_beta_params(19, 0.05) == (1.0, 19.0)
    assert fpr_beta_params(99, 0.01) == (1.0, 99.0)
    assert fpr_beta_params(50, 0.05) == (2.0, 49.0)
    assert fpr_beta_params(100, 0.05) == (5.0, 96.0)
    assert fpr_beta_params(18, 0.05) is None
    assert fpr_beta_params(10, 0.05) is None


def test_theoretical_mean_is_within_one_step_below_alpha() -> None:
    # E[FPR] = (n + 1 - l) / (n + 1) is in (alpha - 1/(n+1), alpha], exactly, whenever l <= n.
    for alpha in (0.01, 0.05, 0.1, 0.2):
        target = Fraction(str(alpha))
        for n in range(1, 700):
            params = fpr_beta_params(n, alpha)
            if params is None:
                continue
            a, b = params
            assert a + b == n + 1
            mean = Fraction(int(a), n + 1)
            assert target - Fraction(1, n + 1) < mean <= target, (n, alpha)


# ---------------------------------------------------------------------------------------------------------
# Simulation with uniform scores
# ---------------------------------------------------------------------------------------------------------


def _simulate(n: int, alpha: float, reps: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per repetition: the exact conditional FPR (1 - threshold for U(0,1) scores) and the number of
    false positives among m fresh test scores."""
    rng = np.random.default_rng(seed)
    conditional = np.empty(reps)
    false_pos = np.empty(reps, dtype=np.int64)
    for i in range(reps):
        thr = conformal_threshold(rng.random(n), alpha)
        conditional[i] = 1.0 - thr.value
        false_pos[i] = int((rng.random(m) > thr.value).sum())
    return conditional, false_pos


@pytest.mark.parametrize(("n", "alpha"), [(19, 0.05), (50, 0.05), (100, 0.05), (120, 0.1), (200, 0.01)])
def test_marginal_fpr_is_bounded_by_alpha(n: int, alpha: float) -> None:
    reps, m = 4000, 20
    _, false_pos = _simulate(n, alpha, reps, m, seed=n)
    rates = false_pos / m
    marginal = rates.mean()
    se = rates.std(ddof=1) / math.sqrt(reps)
    a, b = fpr_beta_params(n, alpha)
    assert abs(marginal - a / (a + b)) < 4 * se
    assert marginal <= alpha + 4 * se
    assert marginal >= alpha - 1 / (n + 1) - 4 * se


@pytest.mark.parametrize(("n", "alpha"), [(50, 0.05), (100, 0.05), (200, 0.01), (19, 0.05)])
def test_conditional_fpr_follows_beta(n: int, alpha: float) -> None:
    reps, m = 4000, 400
    conditional, false_pos = _simulate(n, alpha, reps, m, seed=1000 + n)
    a, b = fpr_beta_params(n, alpha)
    law = stats.beta(a, b)

    assert abs(conditional.mean() - law.mean()) < 4 * law.std() / math.sqrt(reps)
    assert abs(conditional.std(ddof=1) / law.std() - 1) < 0.08
    for q in (0.05, 0.5, 0.95):
        tol = 4 * math.sqrt(q * (1 - q) / reps)
        assert abs(law.cdf(np.quantile(conditional, q)) - q) < tol, q
    assert stats.kstest(conditional, law.cdf).pvalue > 1e-3

    # With real test scores the false positive count is BetaBinomial(m, n + 1 - l, l).
    counts = stats.betabinom(m, a, b)
    assert abs(false_pos.mean() - counts.mean()) < 4 * counts.std() / math.sqrt(reps)
    assert abs(false_pos.std(ddof=1) / counts.std() - 1) < 0.08
    for q in (0.05, 0.5, 0.95):
        tol = 4 * math.sqrt(q * (1 - q) / reps)
        k = int(np.quantile(false_pos, q, method="inverted_cdf"))
        assert counts.cdf(k) >= q - tol, q
        assert counts.cdf(k - 1) <= q + tol, q


def test_not_guaranteed_fpr_follows_beta_1_n() -> None:
    # n = 10, alpha = 0.05: the threshold is the maximum of 10 scores, whose FPR is Beta(1, 10).
    reps = 4000
    conditional, _ = _simulate(10, 0.05, reps, 1, seed=7)
    law = stats.beta(1, 10)
    assert abs(conditional.mean() - 1 / 11) < 4 * law.std() / math.sqrt(reps)
    assert conditional.mean() > 0.05  # the target is missed on average
    assert stats.kstest(conditional, law.cdf).pvalue > 1e-3


# ---------------------------------------------------------------------------------------------------------
# pooled_fpr_band
# ---------------------------------------------------------------------------------------------------------


def test_pooled_band_covers_direct_simulation() -> None:
    n_cal = [60, 100, 241, 10]  # the last category has no guarantee at alpha = 0.05
    m_test = [200, 300, 402, 150]
    alpha = 0.05
    reps = 3000
    rng = np.random.default_rng(11)
    total = sum(m_test)
    direct = np.empty(reps)
    for i in range(reps):
        fp = 0
        for n, m in zip(n_cal, m_test, strict=True):
            thr = conformal_threshold(rng.random(n), alpha)
            fp += int((rng.random(m) > thr.value).sum())
        direct[i] = fp / total

    lo, hi, mean = pooled_fpr_band(n_cal, m_test, alpha)
    assert lo < mean < hi
    expected_mean = (3 / 61 * 200 + 5 / 101 * 300 + 12 / 242 * 402 + 1 / 11 * 150) / total
    assert abs(mean - expected_mean) < 5e-4
    assert abs(direct.mean() - mean) < 4 * direct.std(ddof=1) / math.sqrt(reps)
    coverage = np.mean((direct >= lo) & (direct <= hi))
    assert 0.935 <= coverage <= 0.97
    d_lo, d_hi = np.quantile(direct, [0.025, 0.975])
    assert abs(d_lo - lo) < 0.004
    assert abs(d_hi - hi) < 0.004


def test_pooled_band_single_category_approaches_beta_quantiles() -> None:
    # With a huge test set the binomial noise vanishes and the band is the Beta interval.
    lo, hi, mean = pooled_fpr_band([99], [10_000_000], 0.05)
    law = stats.beta(5, 95)
    assert abs(lo - law.ppf(0.025)) < 1e-3
    assert abs(hi - law.ppf(0.975)) < 2e-3
    assert abs(mean - 0.05) < 5e-4


def test_pooled_band_is_deterministic_and_nested_in_conf() -> None:
    args = ([60, 120], [200, 300], 0.05)
    first = pooled_fpr_band(*args, n_sim=20_000)
    assert first == pooled_fpr_band(*args, n_sim=20_000)
    assert first != pooled_fpr_band(*args, n_sim=20_000, seed=1)
    wide = pooled_fpr_band(*args, n_sim=20_000, conf=0.99)
    assert wide[0] <= first[0] < first[1] <= wide[1]
    assert all(isinstance(v, float) for v in first)


def test_pooled_band_draws_categories_independently() -> None:
    # Twelve identical categories. Independent draws: the variance of the pooled count is the sum of the
    # twelve BetaBinomial variances. Reusing one random stream for every category (perfectly correlated
    # draws) would make the band sqrt(12) times wider, which categories of different sizes do not reveal.
    k, n, m = 12, 100, 300
    lo, hi, mean = pooled_fpr_band([n] * k, [m] * k, 0.05)
    counts = stats.betabinom(m, 5, 96)  # l = ceil(101 * 0.95) = 96
    sd_independent = math.sqrt(k * counts.var()) / (k * m)
    assert abs(mean - 5 / 101) < 3e-4
    assert (hi - lo) / 2 == pytest.approx(1.96 * sd_independent, rel=0.10)
    # The same twelve categories as one: the band of a single category is sqrt(12) times wider.
    one_lo, one_hi, _ = pooled_fpr_band([n], [m], 0.05)
    assert (one_hi - one_lo) / (hi - lo) == pytest.approx(math.sqrt(k), rel=0.15)


def test_pooled_band_validates_input() -> None:
    with pytest.raises(ValueError):
        pooled_fpr_band([50, 60], [100], 0.05)
    with pytest.raises(ValueError):
        pooled_fpr_band([], [], 0.05)
    with pytest.raises(ValueError):
        pooled_fpr_band([50], [0], 0.05)
    with pytest.raises(ValueError):
        pooled_fpr_band([0], [100], 0.05)
    with pytest.raises(ValueError):
        pooled_fpr_band([50], [-1], 0.05)


@pytest.mark.parametrize("n_sim", [0, -5])
def test_pooled_band_rejects_non_positive_n_sim(n_sim: int) -> None:
    with pytest.raises(ValueError, match="n_sim"):
        pooled_fpr_band([50], [100], 0.05, n_sim=n_sim)


def test_pooled_band_rejects_fractional_counts() -> None:
    # Counts used to be truncated silently (60.7 -> 60).
    with pytest.raises(TypeError):
        pooled_fpr_band([60.7], [200], 0.05, n_sim=100)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        pooled_fpr_band([60], [200.9], 0.05, n_sim=100)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        pooled_fpr_band(np.array([60.0]), np.array([200.0]), 0.05, n_sim=100)
    with pytest.raises(TypeError):
        pooled_fpr_band([60], [200], 0.05, n_sim=100.0)  # type: ignore[arg-type]
    # Integer types other than int are fine, and give the same stream.
    expected = pooled_fpr_band([60, 120], [200, 300], 0.05, n_sim=2000)
    assert pooled_fpr_band(np.array([60, 120]), np.array([200, 300]), 0.05, n_sim=2000) == expected
    assert pooled_fpr_band((60, 120), (np.int32(200), np.int64(300)), 0.05, n_sim=np.int64(2000)) == expected


# ---------------------------------------------------------------------------------------------------------
# subsample_fpr_curve
# ---------------------------------------------------------------------------------------------------------

_CURVE_KEYS = {"n", "pool", "m", "mean", "p5", "p95", "theory_mean", "theory_p5", "theory_p95", "guaranteed"}


def _replay_curve(
    cal: np.ndarray, test: np.ndarray, alpha: float, ns: tuple[int, ...], draws: int, seed: int
) -> list[tuple[int, float, float, float, bool]]:
    """subsample_fpr_curve redone one subset at a time from the documented random stream: one generator
    seeded once, then for each usable n in order ``draws`` calls of ``choice(len(cal), n, replace=False)``.
    """
    cal = np.asarray(cal, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = []
    for n in ns:
        if n > cal.size:
            continue  # skipped sizes do not consume random numbers
        fpr = np.empty(draws)
        guaranteed = True
        for i in range(draws):
            thr = conformal_threshold(cal[rng.choice(cal.size, size=n, replace=False)], alpha)
            guaranteed = thr.guaranteed
            fpr[i] = np.count_nonzero(test > thr.value) / test.size
        p5, p95 = np.percentile(fpr, [5, 95])
        out.append((n, float(fpr.mean()), float(p5), float(p95), guaranteed))
    return out


def test_subsample_curve_matches_theory_on_exchangeable_scores() -> None:
    # Both populations are fine regular grids on (0, 1), so that the only randomness left is which
    # subset is drawn: the observed spread can then be held to Monte Carlo tolerances.
    cal = (np.arange(4000) + 0.5) / 4000
    test = (np.arange(5000) + 0.5) / 5000
    draws = 2000
    alpha = 0.05
    curve = subsample_fpr_curve(cal, test, alpha, ns=(10, 20, 50, 100, 200, 5000), draws=draws)
    assert [row["n"] for row in curve] == [10, 20, 50, 100, 200]  # 5000 > len(cal) is skipped
    assert [row["guaranteed"] for row in curve] == [False, True, True, True, True]
    for row in curve:
        assert set(row) == _CURVE_KEYS
        assert (row["pool"], row["m"]) == (4000, 5000)
        assert row["p5"] <= row["mean"] <= row["p95"]
        assert row["theory_p5"] < row["theory_mean"] < row["theory_p95"]
        params = fpr_beta_params(row["n"], alpha)
        law = stats.beta(*params) if params is not None else stats.beta(1, row["n"])
        assert row["theory_mean"] == pytest.approx(law.mean())
        assert row["theory_p5"] == pytest.approx(law.ppf(0.05))
        assert row["theory_p95"] == pytest.approx(law.ppf(0.95))
        assert abs(row["mean"] - law.mean()) < 4 * law.std() / math.sqrt(draws), row["n"]
        tol = 4 * math.sqrt(0.05 * 0.95 / draws)
        assert abs(law.cdf(row["p5"]) - 0.05) < tol, row["n"]
        assert abs(law.cdf(row["p95"]) - 0.95) < tol, row["n"]

    by_n = {row["n"]: row for row in curve}
    assert by_n[10]["theory_mean"] == pytest.approx(1 / 11)  # Beta(1, 10)
    assert by_n[10]["theory_p95"] == pytest.approx(stats.beta(1, 10).ppf(0.95))
    assert by_n[20]["theory_mean"] == pytest.approx(1 / 21)  # l = 20 -> Beta(1, 20)
    assert by_n[50]["theory_mean"] == pytest.approx(2 / 51)
    assert by_n[100]["theory_mean"] == pytest.approx(5 / 101)
    assert by_n[200]["theory_mean"] == pytest.approx(10 / 201)
    assert by_n[200]["theory_p5"] == pytest.approx(stats.beta(10, 191).ppf(0.05))
    # More calibration scores -> a narrower spread around the target.
    assert by_n[200]["p95"] - by_n[200]["p5"] < by_n[20]["p95"] - by_n[20]["p5"]


def test_subsample_curve_full_set_equals_single_threshold() -> None:
    rng = np.random.default_rng(5)
    cal = rng.normal(size=80)
    test = rng.normal(size=333)
    expected = float(np.mean(test > conformal_threshold(cal, 0.1).value))
    (row,) = subsample_fpr_curve(cal, test, 0.1, ns=(80,), draws=25)
    assert row["mean"] == pytest.approx(expected)
    assert row["p5"] == pytest.approx(expected)
    assert row["p95"] == pytest.approx(expected)


def _replay_inputs(kind: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    if kind == "continuous":
        return rng.normal(size=150), rng.normal(size=211)
    if kind == "ties":  # twelve distinct values only
        return rng.integers(0, 12, size=150).astype(float), rng.integers(0, 12, size=211).astype(float)
    if kind == "float32":
        return rng.random(150).astype(np.float32), rng.random(97).astype(np.float32)
    return rng.random(150), rng.random(1)  # a single test score


@pytest.mark.parametrize("kind", ["continuous", "ties", "float32", "one_test"])
@pytest.mark.parametrize("alpha", [0.05, 0.01, 0.2])
def test_subsample_curve_replays_the_documented_stream(kind: str, alpha: float) -> None:
    # Exact comparison (no distributional tolerance): the same generator, advanced the same way, must
    # give the same subsets. The sizes are out of order, one of them is skipped (500 > 150) and one is
    # the whole pool, so this also pins that a single generator runs through all of them: re-seeding per
    # n, or drawing for a skipped n, would change every row after the first.
    cal, test = _replay_inputs(kind)
    ns, draws, seed = (40, 10, 500, 150, 120, 20), 60, 3
    curve = subsample_fpr_curve(cal, test, alpha, ns=ns, draws=draws, seed=seed)
    expected = _replay_curve(cal, test, alpha, ns, draws, seed)
    assert [row["n"] for row in curve] == [40, 10, 150, 120, 20]
    assert len(expected) == len(curve)
    for row, (n, mean, p5, p95, guaranteed) in zip(curve, expected, strict=True):
        assert row["n"] == n
        assert row["guaranteed"] is guaranteed
        assert row["mean"] == pytest.approx(mean, abs=1e-12, rel=0)
        assert row["p5"] == pytest.approx(p5, abs=1e-12, rel=0)
        assert row["p95"] == pytest.approx(p95, abs=1e-12, rel=0)


def test_subsample_curve_uses_the_strict_rule_on_ties() -> None:
    # cal = test = 0..9, alpha = 0.5: l = ceil(11 * 0.5) = 6, threshold 5.0. "score > 5" flags 6..9 (0.4);
    # a ">=" rule would also flag the 5 (0.5). n == len(cal), so every draw is the same set.
    scores = np.arange(10.0)
    (row,) = subsample_fpr_curve(scores, scores, 0.5, ns=(10,), draws=7)
    for key in ("mean", "p5", "p95"):
        assert row[key] == pytest.approx(0.4, abs=1e-12, rel=0), key
    assert row["guaranteed"] is True
    # All scores equal: nothing is ever above the threshold.
    (row,) = subsample_fpr_curve(np.ones(30), np.ones(12), 0.05, ns=(20,), draws=9)
    assert (row["mean"], row["p5"], row["p95"]) == (0.0, 0.0, 0.0)
    # Not guaranteed (n = 5 < 19): the threshold is the subset maximum and ties with it are not flagged.
    (row,) = subsample_fpr_curve(np.arange(5.0), np.array([3.0, 4.0, 4.0, 5.0]), 0.05, ns=(5,), draws=3)
    assert row["mean"] == 0.25
    assert row["guaranteed"] is False


def test_subsample_curve_reports_pool_and_test_sizes() -> None:
    rng = np.random.default_rng(12)
    cal, test = rng.random((6, 20)), rng.random(77)  # 2-D input is flattened: pool = 120
    curve = subsample_fpr_curve(cal, test, 0.05, draws=20)
    assert [(row["n"], row["pool"], row["m"]) for row in curve] == [
        (20, 120, 77),
        (50, 120, 77),
        (100, 120, 77),
    ]
    for row in curve:
        assert type(row["pool"]) is int and type(row["m"]) is int
    assert json.loads(json.dumps(curve)) == curve  # plain Python values only


def test_subsample_spread_is_narrower_than_beta_when_n_is_a_large_share_of_the_pool() -> None:
    # Documents a property, not a bug: scores are iid uniform (perfectly exchangeable), yet the observed
    # p5..p95 spread falls short of the Beta spread, because the subsets come without replacement from
    # one fixed pool and are measured on one fixed test set. The shortfall grows with n / pool (roughly
    # sqrt(1 - n / pool)), so the theory_* columns are only comparable to the observed ones through a
    # reference simulation with the same (pool, m, n). Every bound below is at least four standard
    # deviations (of the statistic across seeds) away from its typical value.
    pool, m, alpha, reals, draws = 300, 200, 0.05, 80, 200
    ns = (20, 100, 200, 300)  # n / pool = 0.07, 0.33, 0.67, 1
    rng = np.random.default_rng(21)
    ratios = {n: [] for n in ns}
    means = {n: [] for n in ns}
    for _ in range(reals):
        for row in subsample_fpr_curve(rng.random(pool), rng.random(m), alpha, ns=ns, draws=draws):
            assert (row["pool"], row["m"]) == (pool, m)
            ratios[row["n"]].append((row["p95"] - row["p5"]) / (row["theory_p95"] - row["theory_p5"]))
            means[row["n"]].append(row["mean"] - row["theory_mean"])
    ratio = {n: float(np.mean(v)) for n, v in ratios.items()}

    assert ratio[300] == 0.0  # the whole pool: one threshold, no spread at all, the Beta band unchanged
    assert 0.35 < ratio[200] < 0.8  # n / pool = 0.67: about 0.58 of the Beta width (sqrt(1 / 3) = 0.58)
    assert ratio[20] > 0.8  # n / pool = 0.07: about 0.95, close to the Beta width
    assert ratio[200] < ratio[100] < ratio[20]  # n / pool = 0.33 sits in between (about 0.80)
    assert ratio[20] - ratio[200] > 0.15

    # The per-category mean carries an error that belongs to this pool and this test set and does not
    # shrink with more draws: about sqrt(alpha * (1 - alpha) * (1 / pool + 1 / m)) = 0.020 here, larger
    # than the sd of the Beta law itself at n = 200 (0.015). It averages out over realisations.
    fixed = math.sqrt(alpha * (1 - alpha) * (1 / pool + 1 / m))
    for n in (100, 200, 300):
        sd = float(np.std(means[n], ddof=1))
        assert 0.5 * fixed < sd < 1.6 * fixed, n
        assert abs(float(np.mean(means[n]))) < 4 * sd / math.sqrt(reals), n
    monte_carlo_se = stats.beta(10, 191).std() / math.sqrt(draws)  # all that the draws alone would leave
    assert float(np.std(means[200], ddof=1)) > 8 * monte_carlo_se


def test_subsample_curve_is_deterministic_and_leaves_input_alone() -> None:
    rng = np.random.default_rng(4)
    cal = rng.random(120)
    test = rng.random(300)
    cal_before, test_before = cal.copy(), test.copy()
    first = subsample_fpr_curve(cal, test, 0.05, draws=200)
    assert first == subsample_fpr_curve(cal, test, 0.05, draws=200)
    assert first != subsample_fpr_curve(cal, test, 0.05, draws=200, seed=1)
    assert [row["n"] for row in first] == [20, 50, 100]
    assert np.array_equal(cal, cal_before)
    assert np.array_equal(test, test_before)
    assert subsample_fpr_curve(cal[:10], test, 0.05) == []


def test_subsample_curve_validates_input() -> None:
    with pytest.raises(ValueError):
        subsample_fpr_curve(np.array([]), np.arange(5.0), 0.05)
    with pytest.raises(ValueError):
        subsample_fpr_curve(np.arange(50.0), np.array([]), 0.05)
    with pytest.raises(ValueError):
        subsample_fpr_curve(np.arange(50.0), np.arange(5.0), 0.05, draws=0)

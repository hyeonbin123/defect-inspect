"""Threshold calibration: exact rank arithmetic, and the theory claims checked by seeded simulation."""

# [파일 역할] 임계값 보정 모듈(`defect_inspect.calibrate`)을 검증하는 테스트 묶음이다. 대상은 split-conformal
# 순위 l = ceil((n+1)(1−α))의 정확한 계산, 그 순위로 고른 임계값(conformal_threshold), 실제 오검출률이 따르는
# Beta 법칙(fpr_beta_params), 범주 합산 오검출률의 이론 구간(pooled_fpr_band), 보정 표본 수에 따른 오검출률
# 곡선(subsample_fpr_curve)이다.
# [왜 필요한가] 질문 2('정상 이미지로 고정한 임계값은 목표를 지키나')의 모든 판정이 이 코드에
# 기대고(docs/experiments.md 공통 규칙 '임계값 보정', 1단계 H1~H4), 같은 함수가 2·3·4·6·7단계와 서비스
# 임계값에도 쓰인다. 순위가 부동소수 오차로 하나만 밀려도 n = 19, α = 5%처럼 경계에서 '보장 불가'로 바뀌고,
# 이론 구간을 잘못 모의하면 '유한표본의 흔들림인가 절차의 치우침인가'를 가를 수 없다.
# [흐름] 순위 산술(정확성, 계산된 α 정리, 잘못된 입력) → 임계값 선택(경계, 보장 불가, 동점, 입력 검사) → 균등
# 점수 모의로 이론(주변 오검출률 ≤ α, 조건부 Beta, BetaBinomial) 확인 → 합산 이론 구간 → 부분표본 곡선(문서에
# 적힌 난수 흐름 재현, 엄격한 부등호, 비복원 추출의 성질).

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


# 확인: 순위 l = ceil((n+1)(1−α))의 대표값. (n+1)(1−α)가 정확히 정수인 경우(20 × 0.95 = 19, 100 × 0.99 = 99)에
# 올림되지 않아야 하고, l > n(n = 10, α = 5%면 11)도 그대로 돌려준다(보장 불가 판단은 호출자 몫). 결과가
# 파이썬 int인지도 본다.
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


# 확인: 부동소수로 곱하면 150 × 0.82가 123을 살짝 넘어 ceil이 124가 되는 경우에도 conformal_rank는 정확히
# 123을 낸다. Fraction으로 계산하는 이유를 보여 주는 반례다.
def test_conformal_rank_is_exact_where_float_ceil_is_not() -> None:
    # 150 * 0.82 is exactly 123, but the float product lands just above it.
    assert math.ceil(150 * (1 - 0.18)) == 124
    assert conformal_rank(149, 0.18) == 123
    assert conformal_rank(249, 0.18) == 205


# 확인: 여러 α에서 n = 0..1199 전부, 순위가 정수 산술의 올림 나눗셈 −(−(n+1)·num // den)과 같다. 특정 값만이
# 아니라 범위 전체에서 부동소수 오차가 없음을 확인한다.
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


# 확인: α를 numpy float, Fraction, Decimal, 문자열, 0차원 배열로 줘도 같은 순위다. float32(0.01)은 실제로
# 0.01보다 조금 작지만, 그 정밀도의 가장 짧은 십진 표기('0.01')로 읽어 순위가 밀리지 않는다. n도 numpy 정수를
# 받는다(설정 파일·배열에서 온 값을 그대로 넘기는 경우).
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


# 확인: 0.15 − 0.1(= 0.04999…)처럼 계산 과정에서 생긴 α도 의도한 십진값(0.05)처럼 다룬다. 그렇지 않으면 n =
# 19에서 l = 20 > n이 돼 보장이 사라진다. 12자리 유효숫자로 반올림해 정리한다(_alpha_fraction docstring).
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


# 확인: 계산된 α로도 n = 19의 임계값이 최댓값(18)이면서 보장되고, Beta 매개변수가 α = 0.05일 때와 같다.
def test_computed_alpha_keeps_the_guarantee_and_the_beta_law() -> None:
    alpha = 0.15 - 0.1
    thr = conformal_threshold(np.arange(19), alpha)
    assert (thr.value, thr.rank, thr.n, thr.guaranteed) == (18.0, 19, 19, True)
    assert fpr_beta_params(19, alpha) == (1.0, 19.0)
    assert fpr_beta_params(100, alpha) == fpr_beta_params(100, 0.05) == (5.0, 96.0)


# 확인: Fraction(1, 3) 같은 정확한 유리수는 십진 반올림을 거치지 않아 3 × (2/3) = 2가 정확히 나온다. 반대로
# 0.0499나 0.05000001처럼 진짜로 다른 값은 0.05로 붙이지 않는다. 12번째 자리까지는 살리고 13번째 이후만
# 부동소수 잡음으로 본다.
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


# 확인: α가 0, 1, 음수, 1 초과, NaN, 무한대, bool, None, 문자열 'x', 리스트면 모두 ValueError다. bool은 숫자로
# 읽혀(True = 1) 조용히 통과하면 안 된다.
@pytest.mark.parametrize(
    "alpha", [0, 1, 0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), True, False, None, "x", [0.05]]
)
def test_conformal_rank_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        conformal_rank(10, alpha)


# 확인: 보정 점수 수 n이 음수면 ValueError다.
def test_conformal_rank_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        conformal_rank(-1, 0.05)


# 확인: n이 19.5, 19.0, '19', None, numpy float이면 TypeError다. 예전에는 소수 개수가 조용히 잘렸다(기존
# 주석). 개수 자리에 실수가 들어오는 것은 다른 값을 잘못 넘긴 신호이므로 거부한다.
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


# 확인: 1..100을 섞은 보정 점수에서 α = 5%의 임계값은 96번째로 작은 96.0이고, 판정 규칙이 '점수 > 임계값'이라
# 보정 점수 가운데 정확히 4개만 불량으로 표시된다. 반환값은 float이다.
def test_threshold_is_the_lth_smallest_score() -> None:
    rng = np.random.default_rng(0)
    scores = rng.permutation(np.arange(1, 101)).astype(np.float32)  # 1..100, shuffled
    thr = conformal_threshold(scores, 0.05)
    assert thr == Threshold(value=96.0, rank=96, n=100, alpha=0.05, guaranteed=True)
    assert isinstance(thr.value, float)
    # Decision rule is strictly greater: exactly 4 of the calibration scores are flagged.
    assert int((scores > thr.value).sum()) == 4
    assert conformal_threshold(scores, 0.2).value == 81.0  # ceil(101 * 0.8) = 81


# 확인: 순위가 정확히 정수로 떨어지는 경계(n = 19, α = 5% → l = 19는 최댓값이지만 보장됨, n = 99, α = 1% → l =
# 99, n = 20 → l = 20)에서 올바른 값과 보장 여부가 나온다.
def test_threshold_exact_integer_edges() -> None:
    thr = conformal_threshold(np.arange(19, 0, -1), 0.05)  # n = 19: l = 19 (the maximum), still guaranteed
    assert (thr.rank, thr.value, thr.guaranteed) == (19, 19.0, True)
    thr = conformal_threshold(np.arange(1, 100), 0.01)  # n = 99: l = 99
    assert (thr.rank, thr.value, thr.guaranteed) == (99, 99.0, True)
    thr = conformal_threshold(np.arange(1, 21), 0.05)  # n = 20: l = ceil(19.95) = 20
    assert (thr.rank, thr.value, thr.guaranteed) == (20, 20.0, True)


# 확인: l > n이면 최댓값을 쓰고 rank = n, guaranteed False로 표시하며 Beta 매개변수는 None이다. α = 5%에서는 n
# = 18이, 1%에서는 n = 98이 보장되지 않는 가장 큰 n이다.
# 왜: 공통 규칙이 'l > n이면 최댓값을 쓰고 보장 불가로 표시한다'로 정했다. 범주마다 보정 점수 수가 달라 실제
# 리포트에 보장된 범주 수를 함께 적는다.
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


# 확인: 동점이 많은 점수에서도 l번째 값을 고르고, 동점은 엄격한 부등호 때문에 오히려 표시 비율을 α 아래로
# 만든다(보수적). 입력 배열은 바뀌지 않는다(np.partition은 사본에서 돈다).
def test_threshold_with_ties_and_input_is_untouched() -> None:
    scores = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0])
    before = scores.copy()
    thr = conformal_threshold(scores, 0.5)  # l = ceil(10 * 0.5) = 5 -> sorted[4] = 2.0
    assert (thr.rank, thr.value) == (5, 2.0)
    assert np.array_equal(scores, before)
    # Ties only make the rule more conservative: the flagged fraction stays below alpha.
    assert np.mean(scores > thr.value) <= 0.5


# 확인: Threshold는 frozen dataclass라 값을 바꿀 수 없고, 빈 점수·NaN 점수·α = 0은 ValueError다. 만든 뒤
# 임계값이 실수로 덮어써지는 일을 막는다.
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


# 확인: 교환 가능한 점수에서 보정 집합이 주어졌을 때 실제 오검출률은 Beta(n+1−l, l)을 따른다(공통 규칙
# '이론값'). 대표값과 보장 불가(None)인 경우를 본다.
def test_fpr_beta_params() -> None:
    assert fpr_beta_params(19, 0.05) == (1.0, 19.0)
    assert fpr_beta_params(99, 0.01) == (1.0, 99.0)
    assert fpr_beta_params(50, 0.05) == (2.0, 49.0)
    assert fpr_beta_params(100, 0.05) == (5.0, 96.0)
    assert fpr_beta_params(18, 0.05) is None
    assert fpr_beta_params(10, 0.05) is None


# 확인: 이론 평균 (n+1−l)/(n+1)이 모든 n과 α에서 (α − 1/(n+1), α] 안에 있다(분수로 정확히 비교). conformal
# 임계값이 평균적으로 목표를 넘지 않고, 넘치는 쪽이 아니라 많아야 한 칸(1/(n+1)) 보수적이라는 성질이다.
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


# 균등분포 U(0,1) 점수 모의. 정상 점수가 균등하면 임계값 t의 조건부 오검출률이 정확히 1 − t라 Beta 법칙과 직접
# 비교할 수 있다. 반복마다 새 보정 점수 n개로 임계값을 정하고, 새 테스트 정상 m개 가운데 몇 개가 넘는지도
# 센다.
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


# 확인: 반복 4,000번의 평균 오검출률(주변 오검출률)이 Beta 이론 평균과 표준오차 4배 안에서 같고, α 이하이며, α
# − 1/(n+1)보다 크다. conformal 보정의 '평균적으로 목표를 넘지 않음' 보장을 모의로 확인한다.
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


# 확인: 조건부 오검출률의 분포가 Beta(n+1−l, l)과 평균·표준편차·분위수·KS 검정에서 맞고, 실제 테스트 정상
# m개의 오검출 수는 BetaBinomial(m, n+1−l, l)과 맞는다.
# 왜: 1단계 H2의 이론 구간과 pooled_fpr_band가 바로 이 BetaBinomial 모형으로 만들어진다(공통 규칙 '이론값').
# 모형이 실제 절차와 맞아야 '이론 구간 안'이라는 판정이 의미가 있다.
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


# 확인: 보장 불가(n = 10, α = 5%)일 때 임계값은 10개의 최댓값이라 오검출률이 Beta(1, 10)을 따르고 평균 1/11 ≈
# 9%로 목표를 넘는다. 보정 점수가 적은 범주에서 '보장 불가' 표시가 왜 필요한지 보여 준다.
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


# 확인: 범주 네 개(하나는 보장 불가)의 합산 오검출률 이론 구간이, 실제로 임계값을 정하고 테스트 정상을 세는
# 직접 모의와 평균·포함률(93.5~97%)·양 끝(0.4%p 안)에서 맞는다. 평균은 범주별 Beta 평균을 테스트 수로 가중한
# 값과 같다.
# 왜: 1단계 H2(홀드아웃 실제 오검출률이 이론 구간 안에 있다)의 기준이 이 함수다. README의 '유한표본 이론
# 구간(3.6~6.3%)'이 여기서 나온다.
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


# 확인: 범주 하나에 테스트가 아주 많으면 이항 잡음이 사라져 구간이 Beta(5, 95)의 2.5·97.5% 분위수에
# 가까워진다(모형의 극한 동작 점검).
def test_pooled_band_single_category_approaches_beta_quantiles() -> None:
    # With a huge test set the binomial noise vanishes and the band is the Beta interval.
    lo, hi, mean = pooled_fpr_band([99], [10_000_000], 0.05)
    law = stats.beta(5, 95)
    assert abs(lo - law.ppf(0.025)) < 1e-3
    assert abs(hi - law.ppf(0.975)) < 2e-3
    assert abs(mean - 0.05) < 5e-4


# 확인: 같은 seed면 같은 결과, 다른 seed면 다른 결과이고, 99% 구간이 95% 구간을 감싸며, 반환값은 파이썬
# float다(seed 고정 결정성 규칙).
def test_pooled_band_is_deterministic_and_nested_in_conf() -> None:
    args = ([60, 120], [200, 300], 0.05)
    first = pooled_fpr_band(*args, n_sim=20_000)
    assert first == pooled_fpr_band(*args, n_sim=20_000)
    assert first != pooled_fpr_band(*args, n_sim=20_000, seed=1)
    wide = pooled_fpr_band(*args, n_sim=20_000, conf=0.99)
    assert wide[0] <= first[0] < first[1] <= wide[1]
    assert all(isinstance(v, float) for v in first)


# 확인: 같은 크기의 범주 12개를 넣으면 구간 폭이 '범주별 BetaBinomial 분산의 합'으로 계산한 폭과 맞고, 범주
# 하나짜리 구간보다 √12배 좁다.
# 왜: 범주마다 같은 난수 흐름을 다시 쓰면(완전히 상관된 추출) 구간이 √12배 넓어지는데, 범주 크기가 다르면 이
# 실수가 드러나지 않는다(기존 주석). 공통 규칙은 '범주별로 독립 추출한 합'으로 정했다.
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


# 확인: 길이가 다른 목록, 빈 목록, 테스트 정상 0장, 보정 점수 0개, 음수 테스트 수는 ValueError다.
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


# 확인: 모의 횟수가 0 이하이면 'n_sim' ValueError다.
@pytest.mark.parametrize("n_sim", [0, -5])
def test_pooled_band_rejects_non_positive_n_sim(n_sim: int) -> None:
    with pytest.raises(ValueError, match="n_sim"):
        pooled_fpr_band([50], [100], 0.05, n_sim=n_sim)


# 확인: 개수(n_cal, m_test, n_sim)에 소수나 float 배열이 오면 TypeError다(예전에는 60.7이 60으로 조용히
# 잘렸다). numpy 정수형·튜플은 받고 int와 같은 난수 흐름을 낸다.
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

# subsample_fpr_curve의 각 행이 가져야 하는 키(관측 평균·5%·95%, 이론 평균·5%·95%, 풀·테스트 크기, 보장 여부).
_CURVE_KEYS = {"n", "pool", "m", "mean", "p5", "p95", "theory_mean", "theory_p5", "theory_p95", "guaranteed"}


# subsample_fpr_curve를 docstring에 적힌 난수 흐름 그대로(생성기 하나를 한 번만 seed, 쓸 수 있는 n마다
# 순서대로 draws번 비복원 choice) 한 부분집합씩 다시 계산한다. 구현과 비트 단위로 비교하기 위한 기준이다.
# 건너뛴 n은 난수를 쓰지 않는다.
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


# 확인: 보정·테스트 점수를 (0,1)의 촘촘한 격자로 두면 남는 무작위성은 '어느 부분집합을 뽑나'뿐이라, 관측
# 평균과 5·95% 분위가 Beta 이론과 몬테카를로 허용 범위 안에서 맞는다. 풀보다 큰 n(5000)은 건너뛰고, n = 10은
# 보장 불가(Beta(1, 10))로 표시한다. 보정 표본이 많을수록 퍼짐이 좁다.
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


# 확인: n이 풀 전체면 모든 추출이 같은 집합이라 평균·5%·95%가 모두 전체 보정 점수로 정한 임계값 하나의
# 오검출률과 같다.
def test_subsample_curve_full_set_equals_single_threshold() -> None:
    rng = np.random.default_rng(5)
    cal = rng.normal(size=80)
    test = rng.normal(size=333)
    expected = float(np.mean(test > conformal_threshold(cal, 0.1).value))
    (row,) = subsample_fpr_curve(cal, test, 0.1, ns=(80,), draws=25)
    assert row["mean"] == pytest.approx(expected)
    assert row["p5"] == pytest.approx(expected)
    assert row["p95"] == pytest.approx(expected)


# 재현 테스트용 입력 네 종류: 연속 정규 점수, 값이 12개뿐인 동점 많은 점수, float32 점수, 테스트 점수가
# 하나뿐인 경우.
def _replay_inputs(kind: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    if kind == "continuous":
        return rng.normal(size=150), rng.normal(size=211)
    if kind == "ties":  # twelve distinct values only
        return rng.integers(0, 12, size=150).astype(float), rng.integers(0, 12, size=211).astype(float)
    if kind == "float32":
        return rng.random(150).astype(np.float32), rng.random(97).astype(np.float32)
    return rng.random(150), rng.random(1)  # a single test score


# 확인: 구현 결과가 문서화된 난수 흐름을 다시 돌린 결과와 분포 허용 오차 없이 같다. 크기 순서를 섞고, 하나는
# 건너뛰고(500 > 150), 하나는 풀 전체로 두어, 생성기 하나가 모든 n을 차례로 지나간다는 것까지 고정한다(n마다
# 다시 seed하거나 건너뛴 n에서 난수를 쓰면 뒤 행이 모두 바뀐다).
# 왜: 1단계 '보정 표본 수 n에 따른 실제 오검출률'은 seed 0의 이 흐름으로 잰 값이고, 같은 결과를 다시 만들 수
# 있어야 한다.
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


# 확인: 부분표본 곡선도 '점수 > 임계값' 규칙을 쓴다. 0..9에서 α = 50%의 임계값 5.0이면 6..9만 표시돼
# 0.4이고(≥였다면 0.5), 점수가 모두 같으면 0이며, 보장 불가면 부분집합 최댓값과 같은 점수는 표시하지 않는다.
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


# 확인: 2차원 보정 입력은 펼쳐 풀 크기가 120이 되고, 기본 n(20, 50, 100, 200) 가운데 풀보다 큰 200은 빠진다.
# pool·m은 파이썬 int이고 결과 전체가 JSON으로 그대로 저장된다(리포트에 쓰이므로).
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


# 확인(버그가 아니라 성질): 완전히 교환 가능한 균등 점수에서도 n이 풀의 큰 몫이면 관측 5~95% 폭이 Beta 폭보다
# 좁다(대략 √(1 − n/pool) 배, n = pool이면 0). 또 범주별 평균에는 그 풀과 테스트 집합 고유의 오차(약
# √(α(1−α)(1/pool + 1/m)))가 실려 추출을 늘려도 줄지 않는다.
# 왜: 이 때문에 관측 곡선을 Beta 분위와 직접 비교하면 흔들림과 치우침을 가를 수 없다. 그래서 비교 기준을 같은
# (pool, m, n)의 모의로 바꿨다(docs/experiments.md 규칙 변경 2). 이 테스트가 그 근거를 수치로 고정한다.
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


# 확인: 같은 seed면 같은 곡선, 다른 seed면 다른 곡선이고, 입력 배열이 바뀌지 않으며, 보정 점수가 기본 n보다
# 모두 적으면(10개) 빈 목록이다.
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


# 확인: 빈 보정 점수, 빈 테스트 점수, draws 0은 ValueError다.
def test_subsample_curve_validates_input() -> None:
    with pytest.raises(ValueError):
        subsample_fpr_curve(np.array([]), np.arange(5.0), 0.05)
    with pytest.raises(ValueError):
        subsample_fpr_curve(np.arange(50.0), np.array([]), 0.05)
    with pytest.raises(ValueError):
        subsample_fpr_curve(np.arange(50.0), np.arange(5.0), 0.05, draws=0)

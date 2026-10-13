"""Split-conformal thresholds for a target false positive rate, and the theory they are checked against.

Decision rule everywhere: an item is a defect iff ``score > threshold.value``.
"""

# [파일 역할] 정상 이미지의 점수만으로 "오검출률 α(예: 5%)"에 맞춘 임계값을 정하는 split-conformal 규칙과, 그
# 임계값의 실제 오검출률이 이론상 어떻게 흔들리는지를 계산하는 모듈이다. 질문 2("정상 이미지로 고정한 임계값은
# 목표를 지키나")의 핵심 코드다.
# [규칙] 보정 점수 n개와 α가 주어지면 l = ceil((n+1)(1−α))번째로 작은 점수를 임계값으로 쓰고, 점수가
# 임계값보다 크면 불량이다. l > n이면(표본이 너무 적음) 최댓값을 쓰고 "보장 불가"로
# 표시한다(docs/experiments.md 공통 규칙 "임계값 보정").
# [이론] 보정 점수와 테스트 정상 점수가 교환 가능하면 실제 오검출률은 Beta(n+1−l, l)을 따른다. 그래서 측정된
# 오검출률이 이 분포의 범위 안이면 "표본이 유한해서 생기는 흔들림", 밖이면 "절차의 치우침"으로 가를 수 있다.
# [연결] conformal_threshold는 분석(analyze, compare, analyze_perturb, analyze_export, analyze_stage6,
# analyze_m2ad_enrol), 실행(run_m2ad, export, dinomaly_serving calibrate), 검사기와 서비스(inspector,
# service의 /calibrate), runtime_compare, figures가 모두 같은 함수로 쓴다. pooled_fpr_band와
# subsample_fpr_curve는 1단계 분석(analyze.py)이 쓴다.

from __future__ import annotations

# Fraction·Decimal·Rational은 α를 정확한 유리수로 다뤄 순위 l의 ceil 계산에 부동소수 오차가 끼지 않게 하려고
# 쓴다. operator.index는 개수 인자가 정수인지 확인하는 데 쓴다.
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from numbers import Rational

# numpy는 점수 배열 연산과 시드를 받은 난수 생성기(default_rng)에, scipy.stats는 Beta 분포의 분위수에 쓴다.
import numpy as np
from scipy import stats


# 보정 결과 하나. value는 임계값, rank는 쓴 순위 l(1부터), n은 보정 점수 수, alpha는 목표 오검출률,
# guaranteed는 l ≤ n이라 이론 보장이 성립하는지다. 분석 리포트와 아티팩트에 이 정보를 같이 남겨, 보장 불가로
# 정한 임계값이 섞였는지 나중에 알 수 있게 한다.
@dataclass(frozen=True)
class Threshold:
    """A threshold picked from calibration scores.

    ``rank`` is the 1-based order statistic that was used. ``guaranteed`` is False when the rank the rule
    asks for exceeds ``n`` (too few calibration scores for this ``alpha``) and the maximum was used instead.
    """

    value: float
    rank: int
    n: int
    alpha: float
    guaranteed: bool


# α를 "10진수로 적힌 그대로"의 유리수로 바꾼다(0.05 → 1/20). α가 계산 결과라 0.04999999999999999처럼 들어와도
# 유효숫자 12자리로 반올림해 1/20로 보므로, 순위가 하나 밀려 임계값이 달라지는 일을 막는다. (0, 1) 밖이면
# ValueError.
def _alpha_fraction(alpha: float) -> Fraction:
    """Rational value of ``alpha`` as it reads in decimal (0.05 -> 1/20).

    Binary floats are rounded to 12 significant digits first, so that a computed alpha such as
    ``0.15 - 0.1`` (0.04999999999999999) behaves like 0.05 instead of pushing the rank up by one.
    ``Fraction``, ``Decimal`` and ``str`` values are already exact and are taken as they are.
    """
    try:
        if isinstance(alpha, (Rational, Decimal, str)):
            frac = Fraction(alpha)
        else:
            # str() first: the shortest decimal form in the value's own precision (float32(0.01) -> "0.01").
            frac = Fraction(format(float(str(alpha)), ".12g"))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"alpha must be a number in (0, 1), got {alpha!r}") from exc
    if not 0 < frac < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return frac


# 개수 인자를 int로 받는다. 2.5 같은 실수를 조용히 잘라 쓰지 않고 TypeError를 내, 호출하는 쪽의 실수를 바로
# 드러낸다.
def _as_count(value: int, name: str) -> int:
    """``value`` as an int. Fractional counts are a TypeError instead of being truncated silently."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc


# conformal 순위 l = ceil((n+1)(1−α))를 정확한 유리수 산술로 계산한다. 예를 들어 α = 5%에서 n = 19면 l = 19라
# 보장이 되고, n = 18이면 l = 19 > 18이라 보장 불가다(6단계 서비스 임계값은 12개 범주 모두 이 조건 n ≥ 19를
# 만족했다). service.py는 이 함수로 /calibrate의 장수가 보장 조건을 채우는지 확인하고 최소 장수를 계산한다.
def conformal_rank(n: int, alpha: float) -> int:
    """Return ``ceil((n + 1) * (1 - alpha))``, computed exactly (no floating point error in the ceil)."""
    n = _as_count(n, "n")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return math.ceil((n + 1) * (1 - _alpha_fraction(alpha)))


# 점수를 1차원 float64로 펴고, 비었거나 NaN이 있으면 거절한다. NaN이 섞이면 정렬·순위가 의미를 잃어 임계값이
# 조용히 틀어지기 때문이다.
def _as_scores(scores: np.ndarray, name: str) -> np.ndarray:
    """Flatten to float64 and reject empty or NaN input."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN")
    return arr


# 보정 점수에서 임계값을 고르는 핵심 함수. 홀드아웃이면 빼 둔 정상의 점수, 교차 적합이면 겹마다 그 겹을 뺀
# 뱅크로 매긴 점수를 넣는다(어느 점수를 넣느냐가 보정 방식이다).
# np.partition은 전체 정렬 없이 l번째 값만 제자리에 놓아 O(n)으로 찾는다. l > n이면 최댓값과
# guaranteed=False를 돌려준다. 이때 실제 오검출률은 목표보다 클 수 있다(1단계: 홀드아웃의 α = 1%에서 범주당
# 점수가 99개 미만인 범주가 이 경우였다).
def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> Threshold:
    """Threshold = the l-th smallest calibration score, ``l = conformal_rank(n, alpha)``.

    When ``l > n`` the maximum score is used, ``rank = n`` and ``guaranteed = False``.
    """
    scores = _as_scores(cal_scores, "cal_scores")
    n = int(scores.size)
    rank = conformal_rank(n, alpha)
    if rank <= n:
        value = float(np.partition(scores, rank - 1)[rank - 1])
        return Threshold(value=value, rank=rank, n=n, alpha=float(alpha), guaranteed=True)
    return Threshold(value=float(scores.max()), rank=n, n=n, alpha=float(alpha), guaranteed=False)


# 보정 집합이 주어졌을 때 실제 오검출률이 따르는 Beta 분포의 모수 (n+1−l, l). 평균은 (n+1−l)/(n+1)로 α 이하다.
# 점수가 교환 가능하고 동점이 없다는 가정이 필요하고, l > n이면 None.
def fpr_beta_params(n: int, alpha: float) -> tuple[float, float] | None:
    """Parameters ``(n + 1 - l, l)`` of the Beta law of the false positive rate given the calibration set.

    Holds for exchangeable calibration and test scores without ties. None when ``l > n``.
    """
    rank = conformal_rank(n, alpha)
    if rank > n:
        return None
    return float(n + 1 - rank), float(rank)


# pooled_fpr_band와 subsample_fpr_curve가 쓰는 도우미. 보장 가능한 순위면 위의 Beta 모수를, 아니면(최댓값을
# 임계값으로 씀) 최댓값의 오검출률 분포인 Beta(1, n)을 돌려준다. 세 번째 값은 보장 여부다.
def _fpr_law(n: int, alpha: float) -> tuple[float, float, bool]:
    """Beta parameters of the FPR and whether the rank was attainable.

    If it was not, the threshold is the maximum of n scores, whose FPR follows Beta(1, n).
    """
    if n < 1:
        raise ValueError(f"need at least one calibration score, got n={n}")
    params = fpr_beta_params(n, alpha)
    if params is None:
        return 1.0, float(n), False
    return params[0], params[1], True


# 12개 범주를 합친(합산) 오검출률의 이론 구간을 모의실험으로 구한다. 범주마다 p_c ~ Beta를 뽑고 오검출 수 fp_c
# ~ Binomial(m_c, p_c)를 뽑아(곧 BetaBinomial) 합친 뒤 비율을 낸다. n_sim번(기본 100,000, seed 0) 반복한
# 분포의 가운데 conf(95%) 구간과 평균을 돌려준다(docs/experiments.md 공통 규칙의 이론 구간).
# 왜 모의실험인가: 범주마다 n과 m이 달라 합산 비율의 분포를 닫힌 식으로 쓰기 어렵다. 1단계 홀드아웃의 실제
# 오검출률 5.56%가 이 구간 [3.56, 6.34] 안이어서 H2가 지지됐다. 호출 위치: analyze.py.
def pooled_fpr_band(
    n_cal: Sequence[int],
    m_test: Sequence[int],
    alpha: float,
    n_sim: int = 100_000,
    seed: int = 0,
    conf: float = 0.95,
) -> tuple[float, float, float]:
    """Theoretical band of the pooled false positive rate over categories.

    Category c has ``n_cal[c]`` calibration scores and ``m_test[c]`` test normals. Each simulation draws
    ``p_c ~ Beta`` and ``fp_c ~ Binomial(m_test[c], p_c)`` independently per category and pools
    ``sum(fp) / sum(m)``. Returns ``(lo, hi, mean)`` for the central ``conf`` interval.

    One generator seeded once runs through the categories in order, so categories of equal size still get
    independent draws. Counts must be integers (TypeError otherwise).
    """
    # 입력 검사: 범주 수가 맞는지, 개수가 정수이고 음수가 아닌지, 테스트 정상이 한 장 이상인지, conf가 (0,
    # 1)인지 확인한다.
    if len(n_cal) != len(m_test):
        raise ValueError("n_cal and m_test must have the same length")
    if len(n_cal) == 0:
        raise ValueError("need at least one category")
    n_cal = [_as_count(n, "n_cal") for n in n_cal]
    m_test = [_as_count(m, "m_test") for m in m_test]
    n_sim = _as_count(n_sim, "n_sim")
    if n_sim < 1:
        raise ValueError(f"n_sim must be positive, got {n_sim}")
    if any(m < 0 for m in m_test):
        raise ValueError("m_test must be non-negative")
    total = sum(m_test)
    if total <= 0:
        raise ValueError("need at least one test normal")
    if not 0 < conf < 1:
        raise ValueError(f"conf must be in (0, 1), got {conf}")

    # 생성기 하나를 범주 순서대로 이어 쓰므로, 크기가 같은 범주끼리도 서로 다른(독립인) 난수를 받는다. 범주별
    # 오검출 수를 더해 합산 비율의 표본 n_sim개를 만든다.
    rng = np.random.default_rng(seed)
    fp = np.zeros(n_sim, dtype=np.int64)
    for n, m in zip(n_cal, m_test, strict=True):
        a, b, _ = _fpr_law(n, alpha)
        p = rng.beta(a, b, size=n_sim)
        fp += rng.binomial(m, p)
    rates = fp / total
    lo, hi = np.quantile(rates, [(1 - conf) / 2, (1 + conf) / 2])
    return float(lo), float(hi), float(rates.mean())


# "보정 표본 수 n에 따른 실제 오검출률"(1단계의 판정 없는 보조 표)을 계산한다. 보정 점수 풀에서 n개(20, 50,
# 100, 200)를 비복원으로 draws번(1,000) 뽑아 임계값을 정하고, 각 임계값의 테스트 오검출률 분포(평균, 5·95
# 백분위)를 Beta 이론값과 나란히 돌려준다.
# 주의(docstring): 하나의 고정된 풀과 테스트 집합에서 뽑으므로 관측 폭이 Beta 폭보다 좁고 풀 고유의 오차가
# 섞인다. 그래서 analyze.py는 Beta와 직접 비교하지 않고, 균등 난수로 같은 (풀, m, n)을 만든 기준 모의(이
# 함수를 다시 호출)와 비교한다(규칙 변경 기록 2).
def subsample_fpr_curve(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    ns: Sequence[int] = (20, 50, 100, 200),
    draws: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Test FPR of thresholds calibrated on random subsets of ``cal_scores``, next to the Beta theory.

    For each n in ``ns`` (skipped when ``n > len(cal_scores)``) draw ``draws`` subsets of size n without
    replacement, take the conformal threshold of each and measure ``mean(test_scores > threshold)``.
    Random stream: one ``default_rng(seed)``; for each usable n in the order given, ``draws`` calls of
    ``rng.choice(len(cal_scores), size=n, replace=False)``. A skipped n consumes nothing.

    Each row has ``n``, ``pool`` (= ``len(cal_scores)``), ``m`` (= ``len(test_scores)``), the observed
    ``mean, p5, p95`` over the draws, ``theory_mean, theory_p5, theory_p95`` of Beta(n + 1 - l, l)
    (Beta(1, n) when the rank is not attainable) and ``guaranteed``.

    The observed spread is narrower than the Beta spread when n is a large share of the pool (roughly by
    ``sqrt(1 - n / pool)``, down to zero at ``n == pool``), because the subsets are drawn without
    replacement from one fixed pool and scored on one fixed test set. The per-category ``mean`` also
    carries an error specific to that pool and test set (about
    ``sqrt(alpha * (1 - alpha) * (1 / pool + 1 / m))``, not reduced by more draws), so the observed and
    ``theory_*`` columns are only comparable through a reference simulation with the same ``(pool, m, n)``.
    """
    # 테스트 점수는 한 번 정렬해 두고, 임계값마다 searchsorted로 "임계값보다 큰 개수"를 이분 탐색으로 센다.
    # 1,000개 임계값 × 테스트 수백 장을 매번 비교하는 것보다 빠르다.
    cal = _as_scores(cal_scores, "cal_scores")
    test_sorted = np.sort(_as_scores(test_scores, "test_scores"))
    pool = int(cal.size)
    m = int(test_sorted.size)
    draws = _as_count(draws, "draws")
    if draws < 1:
        raise ValueError(f"draws must be positive, got {draws}")

    rng = np.random.default_rng(seed)
    out: list[dict] = []
    # 풀보다 큰 n은 건너뛴다(난수도 쓰지 않아, 다른 n의 결과가 바뀌지 않는다). 보장 불가인 n이면 최댓값(순위
    # n)을 임계값으로 쓴다.
    for n in ns:
        n = _as_count(n, "subset size")
        if n < 1:
            raise ValueError(f"subset sizes must be positive, got {n}")
        if n > cal.size:
            continue
        a, b, guaranteed = _fpr_law(n, alpha)
        rank = conformal_rank(n, alpha) if guaranteed else n
        # One subset per row. Drawn one at a time so memory stays at draws x n, not draws x len(cal).
        picks = np.stack([rng.choice(cal.size, size=n, replace=False) for _ in range(draws)])
        # 행(부분표본)마다 l번째로 작은 값이 그 부분표본의 임계값이다. side="right"라 임계값과 같은 점수는
        # 정상으로 센다(규칙: 점수 > 임계값만 불량).
        thresholds = np.partition(cal[picks], rank - 1, axis=1)[:, rank - 1]
        fpr = (m - np.searchsorted(test_sorted, thresholds, side="right")) / m
        p5, p95 = np.percentile(fpr, [5, 95])
        law = stats.beta(a, b)
        out.append(
            {
                "n": n,
                "pool": pool,
                "m": m,
                "mean": float(fpr.mean()),
                "p5": float(p5),
                "p95": float(p95),
                "theory_mean": float(a / (a + b)),
                "theory_p5": float(law.ppf(0.05)),
                "theory_p95": float(law.ppf(0.95)),
                "guaranteed": guaranteed,
            }
        )
    return out

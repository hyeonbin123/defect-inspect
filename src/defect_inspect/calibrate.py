"""Split-conformal thresholds for a target false positive rate, and the theory they are checked against.

Decision rule everywhere: an item is a defect iff ``score > threshold.value``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from numbers import Rational

import numpy as np
from scipy import stats


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


def _as_count(value: int, name: str) -> int:
    """``value`` as an int. Fractional counts are a TypeError instead of being truncated silently."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc


def conformal_rank(n: int, alpha: float) -> int:
    """Return ``ceil((n + 1) * (1 - alpha))``, computed exactly (no floating point error in the ceil)."""
    n = _as_count(n, "n")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return math.ceil((n + 1) * (1 - _alpha_fraction(alpha)))


def _as_scores(scores: np.ndarray, name: str) -> np.ndarray:
    """Flatten to float64 and reject empty or NaN input."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN")
    return arr


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


def fpr_beta_params(n: int, alpha: float) -> tuple[float, float] | None:
    """Parameters ``(n + 1 - l, l)`` of the Beta law of the false positive rate given the calibration set.

    Holds for exchangeable calibration and test scores without ties. None when ``l > n``.
    """
    rank = conformal_rank(n, alpha)
    if rank > n:
        return None
    return float(n + 1 - rank), float(rank)


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

    rng = np.random.default_rng(seed)
    fp = np.zeros(n_sim, dtype=np.int64)
    for n, m in zip(n_cal, m_test, strict=True):
        a, b, _ = _fpr_law(n, alpha)
        p = rng.beta(a, b, size=n_sim)
        fp += rng.binomial(m, p)
    rates = fp / total
    lo, hi = np.quantile(rates, [(1 - conf) / 2, (1 + conf) / 2])
    return float(lo), float(hi), float(rates.mean())


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
    cal = _as_scores(cal_scores, "cal_scores")
    test_sorted = np.sort(_as_scores(test_scores, "test_scores"))
    pool = int(cal.size)
    m = int(test_sorted.size)
    draws = _as_count(draws, "draws")
    if draws < 1:
        raise ValueError(f"draws must be positive, got {draws}")

    rng = np.random.default_rng(seed)
    out: list[dict] = []
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

"""Bootstrap resampling, intervals and the pre-registered verdict rule (docs/experiments.md)."""

# [파일 역할] 이 프로젝트의 통계 도구 모음: 층화 부트스트랩 인덱스, 군집 부트스트랩 인덱스, 백분위 구간,
# Clopper-Pearson 이항 구간, 범주 평균 AUROC와 합산 비율의 부트스트랩 분포, 미리 정한 판정 문구(차이 있음/차이
# 작음/판정 불가).
# [왜 필요한가] docs/experiments.md "신뢰구간과 판정" 규칙(부트스트랩 2,000번, seed 0, 범주 안에서 정상과
# 이상을 따로 복원 추출, 2.5~97.5 백분위, 최소 효과)을 모든 분석 모듈이 똑같이 쓰게 하려고 한곳에 모았다. 같은
# 크기·같은 시드면 늘 같은 인덱스가 나오므로, 두 방법에 같은 추출을 써서 "짝지은 차이"의 구간을 낼 수 있다.
# [연결] analyze.py(1단계 보정 비교), compare.py(2단계 방법 비교), analyze_perturb.py, analyze_m2ad.py,
# analyze_m2ad_enrol.py가 가져다 쓴다.

from __future__ import annotations

# numpy: 재표본 인덱스를 [n_boot, 크기] 배열로 한 번에 만들어 반복문 없이 계산한다. scipy.stats.beta:
# Clopper-Pearson 구간의 베타 분위수, rankdata: 재표본마다 평균 순위를 매겨 AUROC를 Mann-Whitney U로 구한다.
import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import beta, rankdata


# 묶음(층)마다 따로 복원 추출한 인덱스. 입력은 묶음 크기 목록, 출력은 묶음마다 [n_boot, 크기] int64 배열이다.
# 분석 코드는 범주마다 정상과 이상을 각각 한 묶음으로 넘겨 "범주 안 층화" 규칙을 지킨다.
# 난수 생성기 하나에서 묶음 순서대로 뽑으므로 크기 목록과 시드가 같으면 결과가 같다. 두 방법의 점수에 같은
# 인덱스를 쓰면 재표본마다 같은 이미지 묶음으로 둘을 비교하게 된다(짝지은 부트스트랩, 차이의 구간을 내는
# 규칙).
# 호출 위치: analyze.py, compare.py, analyze_perturb.py, analyze_m2ad.py, analyze_m2ad_enrol.py, 이 파일의 두
# _bootstrap 함수.
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
        # 빈 묶음은 rng.integers(0, 0)이 오류를 내므로 열이 0개인 배열을 따로 만든다. 이때 난수를 소비하지
        # 않는다.
        if size == 0:
            out.append(np.zeros((n_boot, 0), dtype=np.int64))
        else:
            out.append(rng.integers(0, size, size=(n_boot, size), dtype=np.int64))
    return out


# 군집(예: 같은 시편의 여러 이미지) 단위 부트스트랩. 군집을 군집 수만큼 복원 추출하고, 뽑힌 군집의 행 전부를
# 원래 순서대로 이어 붙인 인덱스를 돌려준다. 같은 군집의 이미지끼리는 독립이 아니므로 군집째 뽑아야 구간이
# 지나치게 좁아지지 않는다.
# 참고: 현재 src에서 부르는 곳은 없고 tests/test_stats.py만 쓴다(M2AD의 시편 단위 구간은 analyze_m2ad 등이
# stratified_indices로 시편을 뽑아 계산한다).
def cluster_indices(cluster_ids: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[np.ndarray]:
    """Cluster bootstrap: draw whole clusters with replacement, as many as there are distinct clusters.

    Returns `n_boot` int64 index arrays into the original rows. Rows of a drawn cluster keep their
    original order, and a cluster drawn twice contributes its rows twice.
    """
    ids = np.asarray(cluster_ids).ravel()
    rng = np.random.default_rng(seed)
    if ids.size == 0:
        return [np.zeros(0, dtype=np.int64) for _ in range(n_boot)]
    # np.unique로 군집마다 번호(inverse)와 크기(counts)를 구하고, 행을 군집별로 모은 순서(rows)와 각 군집이
    # rows에서 시작하는 자리(starts)를 미리 계산한다. 안정 정렬이라 군집 안의 원래 행 순서가 유지된다.
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    rows = np.argsort(inverse, kind="stable")  # rows grouped by cluster
    starts = np.cumsum(counts) - counts
    # 재표본 n_boot개 각각에서 군집 번호를 군집 수만큼 뽑는다. 아래 반복은 뽑힌 군집들의 행 위치를 np.repeat와
    # arange로 한 번에 계산해 파이썬 이중 반복을 피한다.
    draws = rng.integers(0, counts.size, size=(n_boot, counts.size), dtype=np.int64)
    out = []
    for draw in draws:
        lengths = counts[draw]
        ends = np.cumsum(lengths)
        # position in `rows` of every output element: start of its cluster + offset within the cluster
        position = np.repeat(starts[draw] - (ends - lengths), lengths) + np.arange(ends[-1])
        out.append(rows[position].astype(np.int64))
    return out


# 부트스트랩 분포의 가운데 conf 구간(기본 95%면 2.5~97.5 백분위). NaN(정의되지 않은 재표본, 예: 결함이 하나도
# 안 뽑힌 경우)은 빼고 계산하며, 남는 값이 없으면 (nan, nan)이다.
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


# k/n 비율의 정확한(Clopper-Pearson) 이항 95% 구간. 단일 오검출률을 이론 구간과 비교할 때 부트스트랩 구간과
# 함께 적는 규칙(docs/experiments.md 신뢰구간 절) 때문에 둔다. 베타 분포 분위수로 구하고, k=0이면 하한 0,
# k=n이면 상한 1로 둔다(그 끝에서 베타 분위수가 정의되지 않음).
# 호출 위치: analyze.py(fpr_clopper_pearson).
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


# 12개 범주 평균 이미지 AUROC(대표 지표)의 부트스트랩 분포. 범주마다 정상(neg)과 결함(pos)을 따로 복원
# 추출하고, 재표본마다 범주 AUROC를 구해 평균한다. 정상이나 결함이 없는 범주가 하나라도 있으면 평균이 정의되지
# 않으므로 전부 NaN이 된다.
# 호출 위치: analyze.py.
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
    # 점수에 NaN이 있으면 순위가 의미 없어지므로 계산 전에 멈춘다.
    if any(np.isnan(a).any() for a in negs + poss):
        raise ValueError("scores contain NaN")
    if not negs:
        return np.full(n_boot, np.nan)
    # 묶음 순서는 범주마다 [정상, 결함]이다. 그래서 범주 c의 인덱스는 indices[2c](정상),
    # indices[2c+1](결함)이다.
    sizes = [a.size for pair in zip(negs, poss, strict=True) for a in pair]
    indices = stratified_indices(sizes, n_boot, seed)
    total = np.zeros(n_boot)
    for c, (neg, pos) in enumerate(zip(negs, poss, strict=True)):
        if neg.size == 0 or pos.size == 0:
            total += np.nan
            continue
        # AUROC = Mann-Whitney U / (정상 수 × 결함 수). 재표본 행마다 평균 순위(동점은 평균, 지표 규칙과
        # 같음)를 매기고, 결함 쪽 순위 합에서 최소값 pos(pos+1)/2를 빼면 U다. 2,000개 재표본을 한 번의
        # rankdata 호출로 처리한다.
        sample = np.concatenate([neg[indices[2 * c]], pos[indices[2 * c + 1]]], axis=1)
        ranks = rankdata(sample, axis=1)  # average ranks within each resample
        u = ranks[:, neg.size :].sum(axis=1) - pos.size * (pos.size + 1) / 2
        total += u / (neg.size * pos.size)
    return total / len(negs)


# 합산 비율(12개 범주의 표시 수 합 / 전체 장수)의 부트스트랩 분포. 고정 임계값의 실제 오검출률·검출률을 범주
# 합산으로 내는 규칙에 맞춘 것이다. 범주마다 크기를 고정한 채 따로 다시 뽑으므로 분모(total)는 늘 같다.
# 호출 위치: analyze.py.
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


# 미리 정한 세 갈래 판정. 차이의 95% 구간이 0을 포함하면 "판정 불가", 포함하지 않고 점추정 크기가 최소 효과
# 이상이면 "차이 있음", 그보다 작으면 "차이 작음"이다(docs/experiments.md 신뢰구간과 판정).
# 점추정이나 구간이 NaN이면 차이로 보지 않고 "판정 불가"로 둔다. 비교 결과가 정의되지 않았는데 차이 있음으로
# 잘못 나오는 것을 막는다.
# 호출 위치: compare.difference(analyze_grid, analyze_export, analyze_stage6도 그 함수를 거친다).
# tests/test_stats.py가 검증한다.
def verdict(diff_samples: np.ndarray, point: float, min_effect: float) -> str:
    """Pre-registered verdict for a difference: its 95% percentile interval and a minimum effect size.

    An undefined (NaN) point estimate or interval is inconclusive, never a difference.
    """
    if math.isnan(point):
        return "판정 불가"
    lo, hi = percentile_ci(diff_samples, 0.95)
    # 구간이 0을 포함하거나 NaN이면 비교식이 모두 거짓이 되어 이 분기로 들어온다.
    if not (lo > 0 or hi < 0):  # interval contains 0 (or is undefined)
        return "판정 불가"
    if abs(point) >= min_effect:
        return "차이 있음"
    return "차이 작음"

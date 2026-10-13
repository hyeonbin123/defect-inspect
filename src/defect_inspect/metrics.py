"""Image- and pixel-level metrics (definitions fixed in docs/experiments.md).

Two decision rules appear here on purpose:
- ``score >= t`` for the curve metrics (AUROC, AUPRO, FPR at a given TPR), where ``t`` is itself a score;
- ``score > threshold`` for fixed thresholds (`rates_at_threshold`), the rule used everywhere else.

Scores must not contain NaN (a NaN score is an upstream bug, so it raises instead of being ranked).
"""

# [파일 역할] 이미지·픽셀 수준 지표(이미지 AUROC, FPR@TPR95, 고정 임계값의 오검출·검출 수, 픽셀 AUROC,
# AUPRO)를 numpy/scipy만으로 계산한다. 정의는 docs/experiments.md 공통 규칙 '지표'에 고정돼 있다.
# [왜 필요한가] 모든 단계의 판정이 이 수치로 내려지므로, 라이브러리 구현에 기대지 않고 직접 써서 정의를 문서와
# 정확히 맞춘다. 지표 코드는 scikit-learn·전수 계산과 대조하는 단위 테스트로 확인한다(experiments.md 1단계
# '구현 점검', tests/test_metrics.py). torch 없이 돌아 CI에서도 검증된다.
# [주요 구성] (1) 이미지 지표: auroc, fpr_at_tpr, rates_at_threshold. (2) 픽셀 지표: pixel_auroc, aupro(정확한
# 값). (3) 부트스트랩용 AUPRO: pro_histograms로 점수를 구간별 개수로 줄여 두고, aupro_from_histograms /
# aupro_bootstrap이 이미지 재표본마다 AUPRO를 다시 계산한다.
# [연결] run_patchcore, run_grid, run_dinomaly, run_supervised가 aupro·pixel_auroc·pro_histograms를 부르고,
# analyze·compare가 auroc·fpr_at_tpr·aupro_bootstrap으로 구간을 낸다. analyze_perturb, analyze_m2ad,
# runtime_compare, supervised도 auroc를 쓴다.

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

# scipy.ndimage.label: 결함 마스크의 8방향 연결 성분(AUPRO의 '결함 영역'). scipy.stats.rankdata: 동점에 평균
# 순위를 주는 순위 계산(AUROC의 동점 규칙).
import numpy as np
from scipy import ndimage
from scipy.stats import rankdata


# 점수를 1차원 float64로 펴고 NaN이 있으면 거절한다. float16/32 값은 float64로 정확히 표현되므로 변환으로
# 순위가 바뀌지 않는다. NaN 점수는 상류의 버그이므로 순위에 섞지 않고 바로 예외를 낸다(모듈 docstring).
def _scores(values: np.ndarray, name: str) -> np.ndarray:
    """Flatten image-level scores to float64 (exact for float16/32 inputs) and reject NaN."""
    out = np.asarray(values, dtype=np.float64).ravel()
    if np.isnan(out).any():
        raise ValueError(f"{name} contains NaN")
    return out


# 이미지 AUROC = P(결함 점수 > 정상 점수) + 0.5 × P(같음). 순위 합으로 구하는 Mann-Whitney U 방식이라 모든
# 쌍을 비교하지 않고 O(n log n)에 계산한다. 동점은 평균 순위로 처리한다(experiments.md '지표': 동점은 평균
# 순위).
# 입력: 정상 점수(neg), 결함 점수(pos). 한쪽이 비면 nan(예: 결함이 없는 재표본).
def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(pos > neg) + 0.5 * P(pos == neg), from average ranks. Empty `neg` or `pos` gives nan."""
    neg = _scores(neg, "neg")
    pos = _scores(pos, "pos")
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([neg, pos]))
    # U = (결함 쪽 순위 합) - n_pos(n_pos+1)/2. 이것을 전체 쌍 수로 나누면 AUROC다.
    u = ranks[neg.size :].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (neg.size * pos.size))


# 검출률 95%에서의 오검출률(FPR@TPR95): 결함의 tpr 이상을 잡는 가장 높은 임계값 t(결함 점수 중 k번째로 큰 값,
# k = ceil(tpr × 결함 수))에서 score >= t인 정상의 비율.
# 임계값을 테스트 라벨로 정하므로 운영 가능한 수치가 아니라 분리도의 요약이다(experiments.md '지표'). 돌려주는
# fp 개수는 12개 범주를 합산할 때(합산 오검출률) 쓰인다.
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
    # 0.95 × 20 같은 곱을 부동소수로 하면 19.000000000000004가 되어 ceil이 20이 될 수 있다. 문자열을 거쳐
    # Fraction(유리수)으로 바꿔 정확히 계산한다.
    # Exact rational arithmetic: 0.95 * 20 must give 19, never 20 because of a rounding error.
    k = math.ceil(Fraction(str(float(tpr))) * pos.size)
    k = min(max(k, 1), pos.size)
    # 전체 정렬 대신 np.partition으로 k번째로 큰 값만 찾는다(오름차순으로 n-k번째).
    t = float(np.partition(pos, pos.size - k)[pos.size - k])
    fp = int(np.count_nonzero(neg >= t))
    rate = fp / neg.size if neg.size else float("nan")
    return rate, t, fp


# 고정 임계값에서의 개수(fp, 정상 수, tp, 결함 수). 판정 규칙은 score > threshold(임계값보다 커야 불량)로,
# 곡선 지표의 >= 규칙과 일부러 다르다(모듈 docstring). 실험 규칙이 '점수가 임계값보다 크면
# 불량'이고(experiments.md '임계값 보정'), 보정 점수의 l번째 값을 임계값으로 쓰는 유한표본 보장(실제
# 오검출률이 목표 이하)도 이 엄격한 부등호에서 성립한다.
# 이 저장소의 src에서는 이 함수를 직접 부르는 곳이 없고 tests/test_metrics.py가 규칙을 확인한다.
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


# 픽셀 지표 입력 점검: 점수 맵과 마스크가 같은 [N, H, W]인지, NaN이 없는지 확인하고 마스크를 bool(0이 아니면
# 결함)로 바꾼다.
# float16 맵은 float32로 올린다. 값은 정확히 보존되고, 정렬이 float16보다 몇 배 빠르다(기존 주석). 정수 맵은
# float64로 바꾼다.
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


# 픽셀 AUROC용 AUROC. 픽셀은 수백만 개라 rankdata의 순위 배열이나 쌍 행렬 대신, 양쪽을 정렬하고 이진
# 탐색(searchsorted)으로 '각 결함 픽셀보다 작은/작거나 같은 정상 수'를 센다.
# (작은 수 + 작거나 같은 수)의 합 = 2 × 이긴 쌍 + 동점 쌍이므로 2 × 전체 쌍으로 나누면 동점 0.5 규칙의 AUROC가
# 된다. 합은 int64 정수로 해서 반올림 오차가 없다.
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


# 모든 이미지의 모든 픽셀을 한 묶음으로 본 픽셀 AUROC(보조 지표). 픽셀 지표는 모든 구성에서 256×256 격자로
# 계산한다(experiments.md '전처리'). 결함 픽셀이나 정상 픽셀이 없으면 nan.
def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """AUROC over all pixels of all images. nan when there are no defect or no normal pixels."""
    maps, defect = _check_maps(maps, masks)
    return _auroc_sorted(maps[~defect], maps[defect])


# ndimage.label에 줄 3×3×3 연결 구조. 가운데 판(_STRUCTURE[1])만 True라 같은 이미지 안의 8방향 이웃만 잇고,
# 0번 축(이미지 사이)으로는 잇지 않는다. 그래서 [N, H, W] 전체를 한 번에 라벨링해도 영역이 이미지 경계를 넘지
# 않는다.
# Connects the 8 neighbours inside an image and nothing across images (axis 0).
_STRUCTURE = np.zeros((3, 3, 3), dtype=bool)
_STRUCTURE[1] = True


# 모든 이미지의 결함 영역에 전역 번호 1..C를 붙이고, 각 영역이 몇 번 이미지에 속하는지(comp_image)도 돌려준다.
# 영역의 이미지 번호는 평탄화 인덱스를 이미지 한 장의 픽셀 수(H×W)로 나눈 몫이다. 부트스트랩에서 이미지를 다시
# 뽑을 때 그 이미지의 영역도 함께 세려면 이 대응이 필요하다.
def _label_components(defect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 8-connected regions per image with global ids 1..C; also return the image of each region."""
    labels, n_comp = ndimage.label(defect, structure=_STRUCTURE)
    flat = np.flatnonzero(labels)
    comp_image = np.zeros(n_comp, dtype=np.int64)
    comp_image[labels.ravel()[flat] - 1] = flat // (defect.shape[1] * defect.shape[2])
    return labels, comp_image


# 꺾은선 아래 넓이를 사다리꼴 공식으로 구하되, x가 x_max(AUPRO의 FPR 상한 0.3)를 넘는 구간은 잘라 x_max에서의
# y를 선형 보간한다. 마지막 축을 따라 계산하므로 부트스트랩 여러 줄을 한 번에 처리할 수 있다.
# x가 같은 수직 구간(dx = 0)은 넓이 0이고, 0으로 나누지 않도록 frac을 1로 둔다.
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


# AUPRO의 FPR 상한은 (0, 1] 안이어야 한다. 기본값 0.3(experiments.md '지표': 오검출률 0~30% 구간에서 적분해
# 0.3으로 나눈다).
def _check_max_fpr(max_fpr: float) -> None:
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")


# AUPRO의 정확한 값(근사 없음). PRO(per-region overlap) = 결함 영역마다 '임계값 이상인 픽셀 비율'을 구해 영역
# 평균을 낸 값이다. 큰 결함과 작은 결함을 같은 무게로 보므로, 픽셀 수가 많은 큰 결함이 지표를 지배하는 픽셀
# AUROC의 약점을 보완한다.
# 방법: 모든 서로 다른 점수를 임계값으로 보되, PRO는 결함 픽셀의 점수에서만 바뀌므로 결함 점수에서만 곡선 점을
# 계산한다. 각 결함 점수 d에서 (FPR(> d), 이전 PRO)까지 평평하게 갔다가 (FPR(>= d), PRO(d))로 직선으로 잇는다.
# 그러면 모든 점수를 지나는 꺾은선과 정확히 같다(docstring).
# 호출 위치: run_patchcore, run_grid, run_dinomaly, run_supervised가 범주별로 부른다. 영역이나 정상 픽셀이
# 없으면 nan.
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

    # 결함 픽셀마다 영역 번호(0부터)와 점수를 뽑고, 큰 라벨 배열은 바로 지워 메모리를 아낀다. sizes는 영역별
    # 픽셀 수다.
    comp = labels[defect] - 1
    scores = maps[defect]
    del labels
    sizes = np.bincount(comp, minlength=n_comp)
    # 결함 픽셀을 점수 내림차순으로 정렬한다. 임계값을 높은 점수부터 내려가며 '임계값 이상'에 들어오는 픽셀을
    # 누적하기 위해서다.
    order = np.argsort(scores, kind="stable")[::-1]
    scores = scores[order]
    # cumulative mean overlap, going down the defect scores
    # 픽셀 하나가 들어올 때마다 그 영역의 겹침 비율이 1/영역 크기만큼 오르고, 영역 평균이므로 다시 영역 수로
    # 나눈다. 누적합이 곧 각 임계값의 PRO다.
    overlap = np.cumsum(1.0 / sizes[comp[order]]) / n_comp
    # 같은 점수(동점) 묶음의 마지막 픽셀 위치만 남긴다. score >= t 규칙에서는 동점 픽셀이 한꺼번에 들어오므로
    # 묶음 끝의 누적값이 그 임계값의 PRO다.
    last = np.flatnonzero(np.append(scores[1:] != scores[:-1], True))  # last pixel of each tie group
    thresholds = scores[last]
    # 부동소수 누적 오차로 1을 조금 넘는 것을 막고, 가장 낮은 결함 점수에서는 모든 영역이 다 덮이므로 정확히
    # 1로 둔다.
    pro = np.minimum(overlap[last], 1.0)
    pro[-1] = 1.0  # every region is fully covered at the lowest defect score
    # 정상 픽셀은 정렬해 두었으므로 이진 탐색으로 FPR을 구한다. fpr_above는 d보다 큰 정상 비율, fpr_at은 d
    # 이상인 정상 비율(동점 정상까지 포함)이다.
    fpr_above = (neg.size - np.searchsorted(neg, thresholds, side="right")) / neg.size
    fpr_at = (neg.size - np.searchsorted(neg, thresholds, side="left")) / neg.size

    # 곡선 점 배열: (0, 0)에서 시작해 임계값마다 (fpr_above, 이전 PRO), (fpr_at, PRO) 두 점을 넣고 (1, 1)로
    # 끝낸다. 0~max_fpr 구간 넓이를 max_fpr로 나눠 0~1로 맞춘다.
    x = np.empty(2 * thresholds.size + 2)
    y = np.empty_like(x)
    x[0], y[0] = 0.0, 0.0
    x[1:-1:2], y[1:-1:2] = fpr_above, np.concatenate([[0.0], pro[:-1]])
    x[2:-1:2], y[2:-1:2] = fpr_at, pro
    x[-1], y[-1] = 1.0, 1.0
    return float(_area_under(x, y, max_fpr)) / max_fpr


# AUPRO 부트스트랩을 위해 점수 맵을 구간별 개수로 줄인 묶음. 맵 전체(수백만 픽셀 × 2,000번 재표본)를 매번
# 정렬하지 않고, 이미지별 정상 픽셀 히스토그램과 영역별 히스토그램만 저장해 두면 어떤 이미지 재표본이든 개수를
# 더해 AUPRO를 다시 계산할 수 있다.
# 실행 스크립트(run_patchcore 등)가 PRO_BINS = 2000 구간으로 만들어 저장하고, analyze·compare가 읽어 구간을
# 낸다. 같은 구간 안 점수는 동점으로 보므로 정확한 aupro와는 근사 차이가 있다.
@dataclass
class ProHistograms:
    """Binned score counts from which AUPRO can be recomputed for any resampling of the images."""

    edges: np.ndarray  # [bins + 1]
    normal: np.ndarray  # int64 [n_images, bins]: histogram of scores on normal pixels, per image
    components: np.ndarray  # int64 [n_components, bins]
    component_image: np.ndarray  # int64 [n_components]: image index of each component


# 각 값이 들어갈 구간 번호: edges[i] <= v < edges[i+1], 마지막 구간은 닫혀 있고 범위 밖 값은 양 끝 구간으로
# 자른다. (v - lo) × bins / (hi - lo)를 내림하는 빠른 계산이 구간 경계에서 부동소수 오차로 한 칸 어긋날 수
# 있어, 아래 두 줄에서 실제 edges와 비교해 한 칸씩 고친다.
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


# 점수 맵과 마스크로 ProHistograms를 만든다. 정상 픽셀은 이미지별로, 결함 픽셀은 영역별로 구간 개수를 센다.
# lo/hi를 주지 않으면 맵 전체의 최솟값·최댓값을 쓴다.
# 호출 위치: run_patchcore, run_grid, run_dinomaly, run_supervised(모두 bins=PRO_BINS=2000).
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
    # 모든 점수가 같으면 구간 폭이 0이 되어 나눗셈이 깨지므로 임의의 양수 폭을 준다. 모든 값이 한 구간에
    # 들어간다.
    if hi == lo:  # constant maps: any positive width works, everything lands in one bin
        hi = lo + max(1.0, abs(lo))
    edges = np.linspace(lo, hi, bins + 1)

    labels, comp_image = _label_components(defect)
    n_comp = comp_image.size
    normal = np.zeros((maps.shape[0], bins), dtype=np.int64)
    comp_parts: list[np.ndarray] = []
    # 이미지 한 장씩 구간 번호를 구해 메모리를 아낀다. 결함이 있는 이미지는 정상 픽셀만 정상 히스토그램에
    # 넣고, 결함 픽셀은 '영역 번호 × bins + 구간 번호'라는 하나의 정수로 모아 두었다가 마지막에 bincount 한
    # 번으로 [영역, 구간] 표로 바꾼다.
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


# 구간 개수로 AUPRO를 계산한다. counts[b, i]는 재표본 b에서 이미지 i가 뽑힌 횟수다. 정상 히스토그램은 횟수만큼
# 가중합(counts @ normal)하고, 영역은 그 영역이 속한 이미지의 횟수만큼 무게를 준다(같은 이미지가 두 번 뽑히면
# 그 영역도 두 번 센다).
# PRO는 영역마다 '임계값 이상 비율'의 가중 평균이다: (무게 / 영역 크기) × 영역 히스토그램을 더하고 무게
# 합(n_regions)으로 나눈다. 영역이나 정상이 없는 재표본은 nan으로 둔다.
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
    # 맨 위 구간부터 아래로 누적해, 각 구간의 아래 경계를 임계값으로 했을 때의 FPR(x)과 PRO(y)를 만든다.
    # 마지막 점은 (1, 1)로 고정한다. 구간 안 점수는 모두 동점으로 보므로 곡선이 구간 단위로 거칠다.
    # cumulative sums from the top bin downwards: the threshold is the lower edge of each bin
    x = np.zeros((normal.shape[0], normal.shape[1] + 1))
    y = np.zeros_like(x)
    x[:, 1:] = np.cumsum(normal[:, ::-1], axis=1) / n_normal[valid, None]
    y[:, 1:] = np.minimum(np.cumsum(overlap[:, ::-1], axis=1), 1.0)
    x[:, -1], y[:, -1] = 1.0, 1.0
    out[valid] = _area_under(x, y, max_fpr) / max_fpr
    return out


# 재표본 횟수 배열의 모양(1차원은 (이미지 수,), 2차원은 [n_boot, 이미지 수])과 음수가 없는지 확인한다.
def _check_counts(h: ProHistograms, image_counts: np.ndarray, ndim: int) -> np.ndarray:
    counts = np.asarray(image_counts, dtype=np.float64)
    n_images = h.normal.shape[0]
    if counts.ndim != ndim or counts.shape[-1] != n_images:
        shape = f"({n_images},)" if ndim == 1 else f"[n_boot, {n_images}]"
        raise ValueError(f"image_counts must have shape {shape}, got {counts.shape}")
    if not (counts >= 0).all():
        raise ValueError("image_counts must be non-negative")
    return counts


# 히스토그램으로 AUPRO 하나를 계산한다. image_counts가 없으면 모든 이미지를 한 번씩 센다(원 표본).
# analyze.py가 원 표본의 구간 근사 AUPRO를 계산할 때 쓴다.
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


# 여러 재표본의 AUPRO를 한 번에 계산한다(행마다 aupro_from_histograms와 같은 값). 재표본마다 히스토그램을
# float로 바꾸지 않고 행렬곱으로 묶어 처리해 2,000번 부트스트랩을 빠르게 한다(experiments.md '신뢰구간과
# 판정': 부트스트랩 2,000번).
# 호출 위치: analyze.py, compare.py.
def aupro_bootstrap(h: ProHistograms, image_counts: np.ndarray, max_fpr: float = 0.3) -> np.ndarray:
    """`aupro_from_histograms` for many resamples at once: `image_counts` is [n_boot, n_images].

    Same value per row as the single call, without converting the histograms once per resample.
    """
    _check_max_fpr(max_fpr)
    counts = _check_counts(h, image_counts, ndim=2)
    # 재표본 256개씩 나눠 [chunk, bins] 작업 배열의 메모리를 묶어 둔다.
    chunk = 256  # bounds the [chunk, bins] work arrays
    parts = [_binned_aupro(h, counts[i : i + chunk], max_fpr) for i in range(0, counts.shape[0], chunk)]
    return np.concatenate(parts) if parts else np.zeros(0)

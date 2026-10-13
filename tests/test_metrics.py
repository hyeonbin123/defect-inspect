"""Metric tests against independent references: scikit-learn and brute-force sweeps written here."""

# [파일 역할] src/defect_inspect/metrics.py의 지표(이미지 AUROC, FPR@TPR95, 고정 임계값 개수, 픽셀 AUROC,
# AUPRO, 부트스트랩용 구간 AUPRO)를 독립된 기준값과 대조하는 테스트 묶음이다.
# [왜 필요한가] 모든 단계의 판정이 이 지표로 내려지는데, 지표는 라이브러리에 기대지 않고 docs/experiments.md
# '지표' 절의 정의대로 직접 구현했다. 그래서 '지표 코드는 scikit-learn 및 전수 계산과 대조하는 단위 테스트로
# 확인한다'(experiments.md 1단계 구현 점검)는 규칙을 이 파일이 맡는다. 특히 동점 처리, 순위 계산의 반올림,
# 8방향 연결 영역, 구간 경계 같은 곳에서 생기는 미세한 오류를 잡는다.
# [주요 구성] (1) 기준 구현(references): scikit-learn AUROC, 모든 쌍 비교, 모든 임계값 전수 탐색, flood fill로
# 찾은 영역과 전수 AUPRO, 전체 정렬 AUPRO. (2) auroc (3) fpr_at_tpr·rates_at_threshold (4) pixel_auroc (5)
# 정확한 aupro (6) 구간(히스토그램) AUPRO와 부트스트랩.

from __future__ import annotations

import math
from fractions import Fraction

# scipy.ndimage는 큰 테스트 맵을 만들 때(가우시안 흐림으로 덩어리 모양 결함)와 두 번째 기준 구현의 영역
# 라벨링에 쓴다. scikit-learn은 dev 의존성 그룹에 '지표 테스트용 기준 구현'으로만 들어 있고(pyproject.toml)
# 패키지 코드는 쓰지 않는다.
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


# 기준 1: scikit-learn roc_auc_score로 계산한 AUROC. 정상은 0, 결함은 1로 라벨을 붙인다.
def sklearn_auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    return float(roc_auc_score(y, np.concatenate([neg, pos]).astype(np.float64)))


# 기준 2: 모든 (결함, 정상) 쌍을 직접 비교해 이기면 1, 같으면 0.5를 세는 AUROC 정의 그대로의 계산. 느리지만
# 동점 규칙을 가장 직접 보여 준다.
def pairwise_auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos.tolist() for n in neg.tolist())
    return wins / (len(neg) * len(pos))


# FPR@TPR의 기준 구현: 나타난 모든 점수를 임계값으로 시도해 결함의 tpr 이상을 잡는(score >= t) 가장 높은 t를
# 고른다. 비율 비교를 Fraction으로 해 부동소수 오차가 끼지 않게 한다.
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


# 2차원 마스크의 연결 영역을 스택 기반 flood fill로 직접 찾는다. scipy.ndimage.label과 독립된 기준이며,
# connectivity로 8방향(정의)과 4방향을 바꿀 수 있어 8방향 규칙을 검증하는 데 쓴다. 영역은 첫 화소의 래스터
# 순서로 나온다.
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


# 꺾은선 아래 넓이를 x_max까지 사다리꼴로 더하고, x_max를 걸치는 선분은 그 지점의 y를 선형 보간해 자른다.
# metrics._area_under의 반복문판 기준이다.
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


# AUPRO 기준 구현(작은 입력용): 서로 다른 모든 점수를 높은 것부터 임계값으로 써서 (정상 픽셀 FPR, 영역 평균
# 겹침 PRO) 점을 모두 찍고 0~max_fpr 구간 넓이를 max_fpr로 나눈다. 마지막 점이 (1, 1)인지도 단언한다. 영역이나
# 정상 픽셀이 없으면 nan으로 구현과 같은 약속을 따른다.
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


# 더 큰 입력용 두 번째 AUPRO 기준. 이미지마다 영역을 라벨링해 결함 픽셀에 1/영역 크기 무게를 주고, 모든 픽셀을
# 점수 내림차순으로 정렬해 누적합으로 FPR과 PRO를 만든다. 동점 묶음의 끝 위치에서만 점을 찍는다. 구현(결함
# 점수에서만 곡선을 계산)과 방식이 달라 독립 대조가 된다.
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


# 작은 무작위 맵·마스크를 만든다. 결함 픽셀을 흩뿌려 영역이 여러 개이고 대각선으로만 닿는 경우가 생기게 하며,
# 결함 없는 이미지를 적어도 한 장 둔다. kind로 점수 분포를 바꾼다: continuous(연속값), float16(소수 한 자리
# 반올림 후 fp16, 실제 저장 맵 형식), few_values(-1~2 정수, 정상·결함 사이 동점이 많음), binary(0/1). 동점
# 처리를 집중적으로 시험하려는 것이다.
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


# 연속값 점수에서 auroc가 scikit-learn과 1e-12까지 같은지, 표본 크기를 무작위(정상 1~299, 결함 1~59)로 바꿔
# 가며 본다. float32 입력도 float64로 정확히 바꿔 같은 값이 나와야 한다.
@pytest.mark.parametrize("seed", range(6))
def test_auroc_matches_sklearn_continuous(seed):
    rng = np.random.default_rng(seed)
    neg = rng.normal(size=rng.integers(1, 300))
    pos = rng.normal(loc=0.8, size=rng.integers(1, 60))
    assert auroc(neg, pos) == pytest.approx(sklearn_auroc(neg, pos), abs=1e-12)
    assert auroc(neg.astype(np.float32), pos.astype(np.float32)) == pytest.approx(
        sklearn_auroc(neg.astype(np.float32), pos.astype(np.float32)), abs=1e-12
    )


# 정수 점수로 동점을 많이 만들었을 때 auroc가 모든 쌍 비교(동점 0.5)와 같은지 본다. scikit-learn도 같은 값을
# 내는지 함께 확인해 세 계산이 한 정의로 모이는지 본다(experiments.md: 동점은 평균 순위).
@pytest.mark.parametrize("seed", range(6))
def test_auroc_matches_sklearn_and_pairs_with_ties(seed):
    rng = np.random.default_rng(100 + seed)
    neg = rng.integers(0, 5, size=rng.integers(1, 80)).astype(np.float64)
    pos = rng.integers(2, 7, size=rng.integers(1, 40)).astype(np.float64)
    expected = pairwise_auroc(neg, pos)
    assert auroc(neg, pos) == pytest.approx(expected, abs=1e-12)
    assert sklearn_auroc(neg, pos) == pytest.approx(expected, abs=1e-12)


# 손으로 계산할 수 있는 경우: 완전 분리 1, 완전 역순 0, 모든 점수 같음 0.5, 네 쌍 중 하나만 동점이면 0.875.
def test_auroc_simple_values():
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.full(7, 0.3), np.full(4, 0.3)) == 0.5  # constant scores
    # one tie out of four pairs: (1 + 1 + 1 + 0.5) / 4
    assert auroc(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx(0.875)


# 2차원 배열과 정수 점수도 펴서 받아들이는지 본다.
def test_auroc_accepts_any_shape_and_int_scores():
    neg = np.array([[0, 1], [1, 2]])
    pos = np.array([[2, 3]])
    assert auroc(neg, pos) == pytest.approx(pairwise_auroc(neg.ravel(), pos.ravel()))


# 한쪽이 비면 nan을 돌려주고(예: 결함이 뽑히지 않은 재표본), 점수에 NaN이 있으면 ValueError다. NaN 점수는 상류
# 버그이므로 순위에 섞지 않고 멈춘다(metrics 모듈 docstring).
def test_auroc_empty_is_nan_and_nan_raises():
    assert math.isnan(auroc(np.array([]), np.array([1.0])))
    assert math.isnan(auroc(np.array([1.0]), np.array([])))
    with pytest.raises(ValueError, match="NaN"):
        auroc(np.array([0.0, np.nan]), np.array([1.0]))


# ---------------------------------------------------------------------------------------------------
# fpr_at_tpr, rates_at_threshold
# ---------------------------------------------------------------------------------------------------


# fpr_at_tpr를 전수 탐색과 대조한다. 결함 수를 1, 7, 20, 40, 41로 바꿔 ceil 경계를 지나게 하고, 홀수 seed는
# 정수 점수로 동점을 많이 만든다. 임계값·오검출 수가 정확히 같아야 하고, 고른 임계값이 실제로 결함의 tpr
# 이상을 잡아야 한다.
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


# k = ceil(tpr × 결함 수)를 부동소수로 계산하면 0.95 × 20이 19.000000000000004가 되어 k가 20으로 밀릴 수 있다.
# 구현은 Fraction으로 정확히 계산하므로 20장이면 19번째, 40장이면 38번째로 큰 결함 점수가 임계값이어야 한다.
# 39장은 ceil(37.05) = 38, tpr = 1이면 가장 작은 결함 점수다.
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


# 기본 tpr이 docs/experiments.md의 정의(검출률 95%)와 같은지 본다. 90%였다면 임계값이 높아져 오검출이 적게
# 나오므로 값으로 구별된다.
def test_fpr_at_tpr_default_is_95_percent():
    # docs/experiments.md: FPR at TPR 95%. With 40 positives that needs 38 of them (t = 3);
    # 90% would need only 36 (t = 5) and flag fewer negatives.
    neg = np.array([0.0, 3.5, 4.5, 10.0])
    pos = np.arange(1.0, 41.0)
    assert fpr_at_tpr(neg, pos) == (0.75, 3.0, 3)
    assert fpr_at_tpr(neg, pos) == fpr_at_tpr(neg, pos, 0.95)
    assert fpr_at_tpr(neg, pos, 0.9) == (0.25, 5.0, 1)


# 곡선 지표의 규칙은 score >= t라서 임계값과 같은 정상도 오검출로 센다. 모든 점수가 같으면 정상 전부가
# 오검출(1.0)이다.
def test_fpr_at_tpr_counts_ties_as_flagged():
    neg = np.array([1.0, 2.0, 2.0, 3.0])
    pos = np.array([2.0, 2.0, 5.0, 6.0])
    rate, t, fp = fpr_at_tpr(neg, pos, 0.75)  # k = 3 -> t = 2.0, and both tied negatives are flagged
    assert (rate, t, fp) == (0.75, 2.0, 3)
    rate, t, fp = fpr_at_tpr(np.full(5, 1.0), np.full(3, 1.0), 0.95)  # constant scores
    assert (rate, t, fp) == (1.0, 1.0, 5)


# 정상이 없으면 비율은 nan(오검출 수 0), 결함이 없거나 tpr이 (0, 1] 밖이면 ValueError다.
def test_fpr_at_tpr_edge_cases():
    rate, t, fp = fpr_at_tpr(np.array([]), np.array([1.0, 2.0]), 0.95)
    assert math.isnan(rate) and t == 1.0 and fp == 0
    with pytest.raises(ValueError):
        fpr_at_tpr(np.array([1.0]), np.array([]), 0.95)
    with pytest.raises(ValueError):
        fpr_at_tpr(np.array([1.0]), np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        fpr_at_tpr(np.array([1.0]), np.array([1.0]), 1.5)


# 고정 임계값의 판정은 score > threshold로, 곡선 지표(>=)와 일부러 다르다(metrics 모듈 docstring). 임계값과
# 같은 점수는 통과(정상 판정)해야 한다. split-conformal 보정(calibrate 모듈 docstring의 판정 규칙)과 서비스
# 판정이 쓰는 것과 같은 규칙이다. 빈 입력과 반환 형식(파이썬 int)도 본다.
def test_rates_at_threshold_is_strict():
    neg = np.array([0.1, 0.5, 0.5, 0.9])
    pos = np.array([0.5, 0.7, 1.0])
    assert rates_at_threshold(neg, pos, 0.5) == (1, 4, 2, 3)  # scores equal to the threshold pass
    assert rates_at_threshold(neg, pos, 0.0) == (4, 4, 3, 3)
    assert rates_at_threshold(neg, pos, 1.0) == (0, 4, 0, 3)
    assert rates_at_threshold(np.array([]), np.array([]), 0.5) == (0, 0, 0, 0)
    assert all(type(v) is int for v in rates_at_threshold(neg, pos, 0.5))


# 점수는 float32인데 임계값은 파이썬 float로 들어올 수 있다. 비교를 float64로 하므로 float32(0.1)은 0.1보다
# 크다(결함 판정). 임계값을 그 float32 값 자체로 주면 같은 값이라 통과다. 임계값을 float32로 반올림해 비교하면
# 이 구별이 사라진다.
def test_rates_at_threshold_float32_scores_compare_exactly():
    # The threshold is compared in float64: a float32 score equal to float32(0.1) is above 0.1.
    scores = np.array([0.1], dtype=np.float32)
    assert float(scores[0]) > 0.1
    assert rates_at_threshold(scores, scores, 0.1) == (1, 1, 1, 1)
    assert rates_at_threshold(scores, scores, float(scores[0])) == (0, 1, 0, 1)


# ---------------------------------------------------------------------------------------------------
# pixel_auroc
# ---------------------------------------------------------------------------------------------------


# pixel_auroc(정렬 + 이진 탐색으로 쌍을 세는 구현)가 모든 점수 분포에서 scikit-learn과 같은지 본다. 마스크가
# bool이든 uint8이든 같아야 한다.
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(4))
def test_pixel_auroc_matches_sklearn(kind, seed):
    maps, masks = random_case(seed, kind, n=6, size=12)
    expected = roc_auc_score(masks.ravel(), maps.ravel().astype(np.float64))
    assert pixel_auroc(maps, masks) == pytest.approx(expected, abs=1e-12)
    assert pixel_auroc(maps, masks.astype(np.uint8)) == pytest.approx(expected, abs=1e-12)


# 결함 픽셀이 정상보다 많은(80%) 드문 경우에도 맞는지 본다. 구현이 양쪽 크기를 가정하지 않는지 확인한다.
def test_pixel_auroc_more_defect_than_normal_pixels():
    rng = np.random.default_rng(3)
    masks = rng.random((3, 10, 10)) < 0.8
    maps = np.round(rng.normal(size=masks.shape) + masks, 1)
    assert pixel_auroc(maps, masks) == pytest.approx(roc_auc_score(masks.ravel(), maps.ravel()), abs=1e-12)


# 퇴화 사례: 상수 맵 0.5, 마스크와 같은 맵 1.0, 뒤집힌 맵 0.0, 결함 픽셀이 없거나 정상 픽셀이 없으면 nan.
def test_pixel_auroc_degenerate():
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    assert pixel_auroc(np.ones((2, 4, 4), dtype=np.float32), masks) == 0.5  # constant maps
    assert pixel_auroc(masks.astype(np.float32), masks) == 1.0
    assert pixel_auroc(1.0 - masks, masks) == 0.0
    assert math.isnan(pixel_auroc(np.ones((2, 4, 4)), np.zeros((2, 4, 4), dtype=bool)))
    assert math.isnan(pixel_auroc(np.ones((2, 4, 4)), np.ones((2, 4, 4), dtype=bool)))


# 세 픽셀 함수(pixel_auroc, aupro, pro_histograms)가 맵·마스크 모양 불일치, [N, H, W]가 아닌 입력, NaN 맵을
# 거절하는지 본다. aupro의 max_fpr는 (0, 1] 밖이면 거절한다.
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


# 정확한 aupro를 전수 기준과 1e-10까지 맞춘다. 네 가지 점수 분포 × seed 8개 × FPR 상한 세 가지(정의값 0.3,
# 좁은 0.05, 전체 1.0). 구현은 결함 점수에서만 곡선을 계산하는 지름길을 쓰므로, 모든 점수를 지나는 꺾은선과
# 정확히 같은지가 핵심이다.
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(8))
def test_aupro_matches_bruteforce(kind, seed):
    maps, masks = random_case(seed, kind)
    for max_fpr in (0.3, 0.05, 1.0):
        expected = aupro_bruteforce(maps, masks, max_fpr)
        assert aupro(maps, masks, max_fpr) == pytest.approx(expected, abs=1e-10), (kind, seed, max_fpr)


# 위 무작위 사례가 쉬운 경우만 만들지 않았는지 확인하는 메타 테스트다. 값이 충분히 다양하고(20가지 이상, 0.3
# 미만부터 0.6 초과까지), 한 이미지에 영역이 2개 이상인 경우와 결함 없는 이미지가 있으며, 정상·결함 픽셀 사이
# 동점이 실제로 생겨야 한다. 이 조건이 없으면 대조 테스트가 통과해도 의미가 약하다.
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


# 실제 맵에 가까운 큰 입력(40장 × 48×48, 흐린 덩어리 결함, 셋 중 하나는 결함 없음)에서 전체 정렬 기준과
# 맞춘다. float32와, 소수 둘째 자리로 반올림한 float16(저장된 맵 형식, 동점이 많음)을 모두 보고, FPR 상한
# 0.01처럼 좁은 구간도 본다.
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


# 픽셀 4개짜리 손 계산 사례로 곡선 모양을 확인한다: 결함이 가장 높으면 1, 정상 하나가 더 높으면 PRO가 FPR
# 1/3에서야 1로 뛰어 0~0.3 구간 넓이 0, 결함과 정상이 동점이면 (0, 0)에서 (1/3, 1)로 가는 직선, 결함이 가장
# 낮으면 0.
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


# 모든 점수가 같으면 임계값이 하나뿐이라 곡선은 (0, 0)→(1, 1) 대각선이고, 0~0.3 넓이 0.045를 0.3으로 나눈
# 0.15가 된다. dtype에 관계없이 같아야 한다.
def test_aupro_constant_maps():
    masks = np.zeros((3, 6, 6), dtype=bool)
    masks[0, 1:3, 1:4] = True
    masks[1, 4, 4] = True
    # a single threshold: the curve is the diagonal, area 0.3^2 / 2 over 0.3
    for dtype in (np.float32, np.float16, np.float64):
        assert aupro(np.full(masks.shape, 0.7, dtype=dtype), masks) == pytest.approx(0.15)
    assert aupro(np.zeros(masks.shape), masks, max_fpr=1.0) == pytest.approx(0.5)
    assert aupro_bruteforce(np.zeros(masks.shape), masks) == pytest.approx(0.15)


# PRO는 영역 평균이다. 1픽셀 영역은 찾고 9픽셀 영역은 놓치면 0.5여야 하고, 픽셀 수로 가중하면 0.1이 된다. 큰
# 결함이 지표를 지배하지 않게 하는 AUPRO의 정의(experiments.md '지표')를 확인한다.
def test_aupro_averages_over_regions_not_pixels():
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 0, 0] = True  # region of 1 pixel, found
    masks[0, 4:7, 4:7] = True  # region of 9 pixels, missed
    maps = np.full((1, 8, 8), 0.5)
    maps[0, 0, 0] = 1.0
    maps[0, 4:7, 4:7] = 0.0
    assert aupro(maps, masks) == pytest.approx(0.5)  # pixel-weighted overlap would give 0.1
    assert aupro(maps, masks) == pytest.approx(aupro_bruteforce(maps, masks))


# 대각선으로만 닿은 픽셀이 같은 영역에 묶이는지(8방향 연결, experiments.md 정의) 본다. 4방향이면 영역 2개라
# PRO 0.5, 8방향이면 4픽셀 영역 하나라 0.25다. 히스토그램 쪽(pro_histograms)도 같은 영역 하나로 세야 한다.
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


# [N, H, W]를 한 번에 라벨링하므로 연속한 이미지의 같은 위치 픽셀이 한 영역으로 이어질 위험이 있다. 구현의
# 연결 구조(_STRUCTURE)는 이미지 축으로 잇지 않으므로 영역 3개, 각 영역의 이미지 번호 0, 1, 2여야 한다.
def test_aupro_regions_never_span_images():
    masks = np.zeros((3, 5, 5), dtype=bool)
    masks[:, 2, 2] = True  # same pixel in consecutive images: three regions, not one
    rng = np.random.default_rng(0)
    maps = rng.normal(size=masks.shape)
    assert aupro(maps, masks) == pytest.approx(aupro_bruteforce(maps, masks))
    h = pro_histograms(maps, masks, bins=16)
    assert h.component_image.tolist() == [0, 1, 2]
    assert h.components.sum(axis=1).tolist() == [1, 1, 1]


# FPR의 분모에는 결함 없는 이미지의 픽셀과 결함 이미지 안의 정상 픽셀이 모두 들어가야 한다. 점수가 높은 결함
# 없는 이미지를 더하거나, 결함 이미지의 정상 줄 하나에 높은 점수를 주면 AUPRO가 낮아져야 하고, 그 값도 전수
# 기준과 같아야 한다.
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


# 결함 영역이 없거나 정상 픽셀이 없으면 정확한 aupro와 구간 AUPRO 모두 nan이다. 영역이 없을 때 히스토그램 표가
# (0, bins) 모양으로 비어 있는지도 본다.
def test_aupro_without_regions_or_normal_pixels_is_nan():
    maps = np.random.default_rng(0).normal(size=(2, 4, 4))
    assert math.isnan(aupro(maps, np.zeros((2, 4, 4), dtype=bool)))
    assert math.isnan(aupro(maps, np.ones((2, 4, 4), dtype=bool)))
    h = pro_histograms(maps, np.zeros((2, 4, 4), dtype=bool), bins=8)
    assert h.components.shape == (0, 8) and h.component_image.shape == (0,)
    assert math.isnan(aupro_from_histograms(h))
    assert math.isnan(aupro_from_histograms(pro_histograms(maps, np.ones((2, 4, 4), dtype=bool), bins=8)))


# 마스크가 0/1 또는 0/255 uint8이어도 같은 값이 나오는지(0이 아니면 결함), 지표 함수들이 입력 배열을
# 제자리에서 바꾸지 않는지 본다. 같은 맵을 여러 지표가 이어서 쓰므로 입력 변경은 다른 수치를 오염시킨다.
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


# 실제 점수 맵처럼 매끄러운 사례: 흐린 잡음의 상위 3%를 결함 덩어리로 하고, 앞쪽 절반 이미지는 결함 없음으로
# 둔다. 맵은 흐린 잡음에 결함 위치를 조금 더해 AUPRO가 포화되지 않게 한다. 구간 AUPRO의 근사 오차를 볼 때
# 쓴다.
def smooth_case(seed: int, n: int = 30, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blobs = ndimage.gaussian_filter(rng.normal(size=(n, size, size)), sigma=(0, 3, 3))
    masks = blobs > np.quantile(blobs, 0.97)
    masks[: n // 2] = False
    maps = ndimage.gaussian_filter(rng.normal(size=masks.shape), sigma=(0, 2, 2))
    maps = maps + 0.15 * ndimage.gaussian_filter(masks.astype(float), sigma=(0, 2, 2))
    return maps.astype(np.float32), masks


# pro_histograms의 배치를 확인한다: edges가 맵 최솟값~최댓값을 bins 칸으로 나누고, 정상 히스토그램은
# 이미지별로 numpy.histogram(마지막 칸 닫힘)과 같으며, 영역 히스토그램은 flood fill로 찾은 영역 순서·이미지
# 번호와 맞는다. 모든 픽셀이 정확히 한 번씩 세어졌는지(합 = 맵 크기)도 본다.
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


# 범위 처리: 최댓값은 마지막 칸에 들어가고, lo/hi를 직접 주면 범위 밖 점수는 첫·마지막 칸으로 잘린다. hi < lo,
# bins = 0은 거절한다.
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


# 기본값이 2,000칸이고 범위가 맵의 최솟값~최댓값인지 본다. 실행 스크립트들이 PRO_BINS = 2000으로 히스토그램을
# 저장하므로 기본값과 같아야 한다.
def test_pro_histograms_default_is_2000_bins_over_the_score_range():
    maps, masks = random_case(3, "continuous")
    lo, hi = float(maps.min()), float(maps.max())
    h = pro_histograms(maps, masks)
    assert h.edges.shape == (2001,)
    assert np.array_equal(h.edges, np.linspace(lo, hi, 2001))
    assert h.normal.shape == (len(maps), 2000) and h.components.shape[1] == 2000
    explicit = pro_histograms(maps, masks, bins=2000, lo=lo, hi=hi)
    assert np.array_equal(h.normal, explicit.normal) and np.array_equal(h.components, explicit.components)


# 구간 경계 바로 아래 점수의 칸 번호를 고치는 처리(metrics._bin_index의 보정 두 줄)를 확인한다. 정상 이미지와
# 같은 점수의 결함 영역 이미지를 나란히 두어 두 히스토그램이 numpy.histogram과 똑같이 나와야 한다.
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


# 모든 경계값과 그 바로 아래·위 부동소수(nextafter)를 넣어 칸 번호가 edges의 정의(edges[i] <= v < edges[i+1],
# 마지막 칸 닫힘)와 정확히 맞는지 본다. 사례마다 단순 floor 공식이 실제로 틀리는 값이 있는지 먼저 단언해,
# 테스트가 헛돌지 않게 한다. 마지막 범위(-0.219, 36.266)는 실제 점수 범위와 비슷한 비정형 값이다.
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


# 상수 맵(폭이 0인 범위)도 히스토그램을 만들 수 있고, 구간 AUPRO가 정확한 값과 같은 0.15인지 본다. 구현은 hi
# == lo면 임의의 양수 폭을 줘 모든 값을 한 칸에 넣는다.
def test_binned_aupro_constant_maps():
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1:3, 1:3] = True
    h = pro_histograms(np.full(masks.shape, 3.0, dtype=np.float32), masks, bins=50)
    assert h.normal.sum() == 46 and h.components.sum() == 4
    assert aupro_from_histograms(h) == pytest.approx(0.15)


# 점수가 0~9 정수이고 칸 하나에 정수 하나만 들어가도록 10칸을 잡으면 구간화로 잃는 정보가 없으므로, 구간
# AUPRO가 정확한 aupro와 전수 기준에 일치해야 한다. maps[0]에 0과 9를 넣어 범위를 정확히 0~9로 고정한다.
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


# 매끄러운 맵에서 2,000칸 구간 AUPRO가 정확한 값과 0.002 안으로 가까운지 본다. 부트스트랩 구간은 이 근사로
# 계산하므로 근사 오차가 판정 기준(최소 효과 1.0%p)보다 훨씬 작아야 한다. 20칸처럼 거칠면 오차가 커지지만 0.05
# 안이다.
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


# 부트스트랩 재표본을 '이미지별 뽑힌 횟수'(counts)로 주는 계산이, 이미지를 실제로 그 횟수만큼 복제해
# 히스토그램을 다시 만든 것과 같은지 본다. 같은 범위(lo, hi)를 써야 칸이 같으므로 명시한다. 복제한 이미지의
# 정확한 aupro와도 0.002 안이어야 하고, counts가 모두 1이면 기본값과 같다. 재표본에 빠진 이미지와 두 번 이상
# 뽑힌 이미지가 다 있는지 먼저 확인한다.
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


# counts의 예외 처리: 결함 없는 이미지만 뽑힌 재표본은 영역이 없어 nan, 모두 0이어도 nan, 길이가 다르거나
# 음수면 ValueError. 모든 이미지를 두 배로 세면 비율이 같으므로 값이 바뀌지 않아야 한다.
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


# 구간 AUPRO의 반복문 기준 구현. 맨 위 칸부터 한 칸씩 내려오며 정상 오검출 수와 영역 겹침을 counts만큼 가중해
# 더하고, 칸마다 곡선 점을 찍는다. 구현의 행렬곱·누적합 계산과 독립적으로 같은 정의를 따른다.
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


# 칸을 7개로 거칠게 잡아 정상·결함 픽셀이 같은 칸에 많이 섞이게 한 뒤, 구간 AUPRO를 반복문 기준과 1e-12까지
# 맞춘다. 무작위 counts(0~3회) 다섯 가지도 보고, 둘 다 nan인 경우는 같은 것으로 본다.
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


# aupro_bootstrap(재표본 여러 개를 한 번에)이 행마다 aupro_from_histograms 한 번 부른 값과 같은지 본다.
# analyze.py와 compare.py가 2,000번 부트스트랩 구간을 이 함수로 낸다(experiments.md '신뢰구간과 판정').
# 재표본 300개는 내부 묶음 크기 256을 넘겨 묶음 경계를 지나게 한다. 7번 행은 결함 없는 이미지만, 8번 행은 빈
# 재표본으로 만들어 nan을 확인하고, 우연히 영역이 없는 재표본으로 생기는 nan도 소수여야 한다. 한 행만 넘겨도
# 같은 값(행끼리·묶음 나누기와 무관), 복제한 이미지의 정확한 값과 0.01 안, max_fpr = 1.0에서도 일치, 0행
# 입력은 빈 결과, 1차원·이미지 수 불일치·음수 counts는 거절한다.
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

"""부트스트랩 재표본 추출, 구간, 판정 규칙의 테스트."""

# [파일 역할] 통계 도구 `defect_inspect.stats`(층화·군집 부트스트랩 인덱스, 백분위 구간, Clopper-Pearson 구간,
# 범주 평균 AUROC와 합산 비율의 부트스트랩 분포, 미리 정한 판정 문구)를 검증한다.
# [왜 필요한가] 모든 단계의 구간과 판정(차이 있음/차이 작음/판정 불가)이 이 함수들에서
# 나온다(docs/experiments.md 신뢰구간과 판정). 그래서 (1) 독립 구현(scikit-learn, scipy)과 숫자가 같은지, (2)
# 규칙의 기본값(2,000번, seed 0)과 난수 사용 순서가 고정돼 '같은 추출로 짝지은 비교'가 가능한지, (3)
# 경계(구간이 0에 닿음, NaN, 빈 묶음)에서 판정이 규칙대로인지 확인한다.
# [설계 메모] torch 없이 기본 패키지와 dev 그룹(scikit-learn)만으로 돌아 CI에서 실행된다.

from __future__ import annotations

import math

# 외부 라이브러리. scipy의 binomtest(정확한 이항 구간)와 binom(꼬리 확률), scikit-learn의
# roc_auc_score(AUROC)는 기대값을 따로 계산하는 참조 구현이다. scikit-learn은 지표 테스트의 참조용으로만 dev
# 그룹에 있다(pyproject.toml).
import numpy as np
import pytest
from scipy.stats import binom, binomtest
from sklearn.metrics import roc_auc_score

# 프로젝트의 AUROC(metrics.auroc)와 시험 대상 함수들.
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


# 묶음마다 [n_boot, 크기] int64 배열이 나오고 값이 0..크기-1 범위인지, 크기 0인 묶음도 열 0개 배열로
# 처리되는지 본다.
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


# 같은 크기·시드면 같은 인덱스, 다른 시드면 다른 인덱스가 나온다(두 방법을 같은 추출로 비교하는 짝지은
# 부트스트랩의 전제). 빈 묶음은 난수를 쓰지 않아 뒤 묶음의 인덱스를 밀지 않는다. 음수 크기는 ValueError다.
def test_stratified_indices_are_deterministic_and_paired():
    a = stratified_indices([300, 40, 200, 40], n_boot=20, seed=0)
    b = stratified_indices([300, 40, 200, 40], n_boot=20, seed=0)
    c = stratified_indices([300, 40, 200, 40], n_boot=20, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0], c[0])
    # 묶음은 생성기 하나에서 순서대로 뽑힌다: 빈 묶음은 다른 묶음의 인덱스를 밀지 않는다
    d = stratified_indices([300, 0, 40], n_boot=20, seed=0)
    assert np.array_equal(d[0], a[0]) and np.array_equal(d[2], a[1])
    with pytest.raises(ValueError):
        stratified_indices([3, -1])


# 구현 계약: default_rng(seed) 하나에서 묶음 순서대로 [n_boot, 크기]를 한 번씩 뽑는다. 크기가 같은 묶음끼리
# 인덱스를 공유하면 범주 사이에 상관이 생기므로, 서로 다르고 상관이 거의 없는지도 본다.
def test_stratified_indices_groups_are_independent_draws_from_one_stream():
    sizes = [40, 40, 7, 40]
    out = stratified_indices(sizes, n_boot=25, seed=11)
    rng = np.random.default_rng(11)
    for idx, size in zip(out, sizes, strict=True):
        assert np.array_equal(idx, rng.integers(0, size, size=(25, size)))
    # 크기가 같은 묶음도 인덱스를 공유하면 안 된다(공유하면 범주 사이에 상관이 생긴다)
    assert not np.array_equal(out[0], out[1]) and not np.array_equal(out[1], out[3])
    corr = np.corrcoef(out[0].ravel(), out[1].ravel())[0, 1]
    assert abs(corr) < 0.1


# 복원 추출이 균등한지: 각 값의 빈도가 1/5 근처이고, 크기 5 표본에 든 서로 다른 값 수의 평균이 이론값 5 × (1 -
# 0.8^5)와 맞는다(한 값이 다섯 번 모두 빠질 확률이 0.8^5).
def test_stratified_indices_resample_uniformly():
    idx = stratified_indices([5], n_boot=20000, seed=0)[0]
    freq = np.bincount(idx.ravel(), minlength=5) / idx.size
    assert np.allclose(freq, 0.2, atol=0.01)
    # 크기 5인 부트스트랩 표본에는 평균 5 * (1 - 0.8^5)개의 서로 다른 값이 들어 있다
    distinct = np.mean([len(set(row)) for row in idx.tolist()])
    assert distinct == pytest.approx(5 * (1 - 0.8**5), abs=0.03)


# 기본값이 규칙대로 2,000번·seed 0인지, 그 난수열을 직접 만들어 비교한다.
def test_stratified_indices_defaults_are_2000_resamples_with_seed_0():
    # docs/experiments.md: 부트스트랩 재표본 2,000번, seed 0.
    out = stratified_indices([5, 3])
    rng = np.random.default_rng(0)
    assert np.array_equal(out[0], rng.integers(0, 5, size=(2000, 5)))
    assert np.array_equal(out[1], rng.integers(0, 3, size=(2000, 3)))
    assert not np.array_equal(out[0], stratified_indices([5, 3], seed=1)[0])


# ---------------------------------------------------------------------------------------------------
# cluster_indices
# ---------------------------------------------------------------------------------------------------


# 군집 부트스트랩이 군집을 통째로 뽑는지: 결과를 군집 단위로 다시 잘라 보면 매번 원래 행 순서 그대로인 군집이
# 군집 수(4)만큼 이어져 있다. 군집 크기가 달라 결과 길이가 재표본마다 달라지고(최소 4, 최대 12) 범위를
# 벗어나지 않는다.
def test_cluster_indices_draws_whole_clusters():
    ids = np.array(["b", "a", "b", "c", "a", "b", "d"])
    rows_of = {c: np.flatnonzero(ids == c).tolist() for c in "abcd"}
    out = cluster_indices(ids, n_boot=300, seed=0)
    assert len(out) == 300
    lengths = set()
    for idx in out:
        assert idx.dtype == np.int64
        # 결과를 뽑힌 순서대로 다시 온전한 군집 단위로 자른다
        pos, drawn = 0, []
        while pos < len(idx):
            cluster = ids[idx[pos]]
            rows = rows_of[cluster]
            assert idx[pos : pos + len(rows)].tolist() == rows
            drawn.append(cluster)
            pos += len(rows)
        assert len(drawn) == 4  # 서로 다른 군집 수만큼 뽑는다
        lengths.add(len(idx))
    assert len(lengths) > 1  # 군집 크기가 달라 길이가 달라진다
    assert min(lengths) >= 4 and max(lengths) <= 12


# 같은 시드면 같은 결과이고, 각 군집은 크기와 무관하게 재표본마다 평균 한 번 뽑힌다(행이 아니라 군집이 추출
# 단위). 재표본마다 뽑힌 군집 수의 합은 군집 수(6)다.
def test_cluster_indices_deterministic_and_uniform_over_clusters():
    ids = np.repeat(np.arange(6), [1, 2, 3, 1, 2, 3])
    a = cluster_indices(ids, n_boot=4000, seed=0)
    b = cluster_indices(ids, n_boot=4000, seed=0)
    c = cluster_indices(ids, n_boot=4000, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert any(not np.array_equal(x, y) for x, y in zip(a, c, strict=True))
    # 각 군집은 크기와 상관없이 재표본마다 평균 한 번 뽑힌다
    first_rows = [np.flatnonzero(ids == k)[0] for k in range(6)]
    draws = np.array([[np.sum(idx == r) for r in first_rows] for idx in a])
    assert np.allclose(draws.mean(axis=0), 1.0, atol=0.06)
    assert (draws.sum(axis=1) == 6).all()


# 경계: 군집마다 한 행이면 보통 부트스트랩과 같고, 빈 입력은 빈 인덱스, 군집이 하나뿐이면 늘 그 군집의 행
# 전부다.
def test_cluster_indices_single_rows_and_empty():
    out = cluster_indices(np.arange(5), n_boot=10, seed=0)
    assert all(len(idx) == 5 and idx.min() >= 0 and idx.max() < 5 for idx in out)
    out = cluster_indices(np.array([]), n_boot=3, seed=0)
    assert [idx.tolist() for idx in out] == [[], [], []]
    out = cluster_indices(np.array([7, 7, 7]), n_boot=4, seed=0)
    assert all(idx.tolist() == [0, 1, 2] for idx in out)


# 기본값 2,000번·seed 0 확인. 한 행 군집에 정렬된 id면 결과가 군집 추출 난수 그 자체라 직접 비교할 수 있다.
def test_cluster_indices_defaults_are_2000_resamples_with_seed_0():
    out = cluster_indices(np.arange(5))
    assert len(out) == 2000
    assert np.array_equal(np.stack(out), np.random.default_rng(0).integers(0, 5, size=(2000, 5)))
    assert not np.array_equal(np.stack(out), np.stack(cluster_indices(np.arange(5), seed=1)))


# ---------------------------------------------------------------------------------------------------
# percentile_ci, clopper_pearson
# ---------------------------------------------------------------------------------------------------


# 백분위 구간: 0~100의 95% 구간은 (2.5, 97.5), 50%는 (25, 75)이고, NaN은 빼고 계산하며 남는 값이 없으면 (nan,
# nan)이다. numpy.percentile과 같은 값을 파이썬 float로 돌려주고, conf는 (0, 1) 안이어야 한다.
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


# Clopper-Pearson 구간이 scipy binomtest의 정확 구간과 1e-12 안에서 같은지, 끝값(k = 0, k = n)과 봉인 테스트
# 정상 3,848장 규모(192장, 약 5%)를 포함해 확인한다. 구간은 늘 점 추정 k/n을 품는다.
@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (1, 10), (5, 40), (38, 40), (192, 3848), (0, 1), (1, 1)])
@pytest.mark.parametrize("conf", [0.95, 0.9])
def test_clopper_pearson_matches_scipy_binomtest(k, n, conf):
    lo, hi = clopper_pearson(k, n, conf)
    ref = binomtest(k, n).proportion_ci(confidence_level=conf, method="exact")
    assert lo == pytest.approx(ref.low, abs=1e-12)
    assert hi == pytest.approx(ref.high, abs=1e-12)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


# 정의로 확인: 하한 p에서 'k 이상' 확률과 상한 p에서 'k 이하' 확률이 모두 0.025다.
@pytest.mark.parametrize("k,n", [(1, 10), (5, 40), (38, 40), (192, 3848)])
def test_clopper_pearson_tail_definition(k, n):
    # 정의: P(X >= k | p = lo) = 0.025, P(X <= k | p = hi) = 0.025.
    lo, hi = clopper_pearson(k, n)
    assert binom.sf(k - 1, n, lo) == pytest.approx(0.025, rel=1e-6)
    assert binom.cdf(k, n, hi) == pytest.approx(0.025, rel=1e-6)


# 끝값의 닫힌 식: k = 0이면 하한 0, 상한 1 - 0.025^(1/n). k = n이면 그 반대. n = 0이면 (0, 1). k가 범위 밖이면
# ValueError.
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


# 범주별 정상·결함 점수를 크기와 분리도를 달리해 만든다. ties=True면 반올림으로 동점을 많이 만들어 평균 순위
# 처리를 시험한다.
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


# 벡터화한 macro_auroc_bootstrap(재표본 전체를 rankdata 한 번으로 계산)이, 같은 인덱스로 재표본마다 sklearn
# roc_auc_score를 부르는 느린 반복과 같은 값을 내는지 동점 유무 둘 다에서 확인한다. 프로젝트 auroc도 sklearn과
# 같은지 함께 본다.
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


# 같은 시드면 같은 분포(짝지은 비교), 분포의 평균이 점 추정 근처이고 95% 구간이 점 추정을 품는다. 같은
# 이미지에 상수를 더한 '두 번째 방법'은 재표본마다 AUROC가 같아 짝지은 차이가 정확히 0이다(같은 추출을 쓴다는
# 증거).
def test_macro_auroc_bootstrap_is_paired_and_centred():
    negs, poss = make_scores(1, n_cat=4, ties=False)
    a = macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=0)
    assert np.array_equal(a, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=0))
    assert not np.array_equal(a, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=1))
    point = np.mean([auroc(n, p) for n, p in zip(negs, poss, strict=True)])
    assert a.mean() == pytest.approx(point, abs=0.01)
    lo, hi = percentile_ci(a)
    assert lo < point < hi
    # 같은 이미지로 점수를 낸 두 번째 방법은 같은 재표본을 쓴다: 점수에 상수를 더해도 재표본마다의
    # AUROC가 바뀌지 않으므로 짝지은 차이가 정확히 0이다
    shifted = macro_auroc_bootstrap([n + 3 for n in negs], [p + 3 for p in poss], n_boot=2000, seed=0)
    assert np.array_equal(a, shifted)


# 기본값 2,000번·seed 0과 묶음 순서(범주마다 정상, 결함: neg_0, pos_0, neg_1, pos_1)를 난수열을 직접 만들어
# 확인한다. 다른 분석 모듈이 같은 순서로 인덱스를 만들어야 방법 사이 비교가 짝지어진다.
def test_macro_auroc_bootstrap_defaults_are_2000_resamples_with_seed_0():
    negs, poss = make_scores(2, n_cat=2, ties=False)
    out = macro_auroc_bootstrap(negs, poss)
    assert out.shape == (2000,)
    assert np.array_equal(out, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=0))
    assert not np.array_equal(out, macro_auroc_bootstrap(negs, poss, n_boot=2000, seed=1))
    # seed 0 난수열 자체: default_rng(0) 하나, 묶음 순서는 neg_0, pos_0, neg_1, pos_1
    rng = np.random.default_rng(0)
    (neg0, pos0), (neg1, pos1) = zip(negs, poss, strict=True)
    i0, j0, i1, j1 = (rng.integers(0, len(a), size=(2000, len(a))) for a in (neg0, pos0, neg1, pos1))
    for b in (0, 1, 1999):
        expected = (auroc(neg0[i0[b]], pos0[j0[b]]) + auroc(neg1[i1[b]], pos1[j1[b]])) / 2
        assert out[b] == pytest.approx(expected, abs=1e-12)


# 경계: 완전히 갈린 범주(1.0)와 점수가 모두 같은 범주(0.5)의 평균, 결함이 없는 범주가 있으면 전부 NaN, 범주가
# 없으면 NaN, 정상·결함 범주 수가 다르면 ValueError, 점수에 NaN이 있으면 ValueError.
def test_macro_auroc_bootstrap_degenerate_inputs():
    # 완전히 갈린 범주와 점수가 모두 같은 범주
    out = macro_auroc_bootstrap([np.zeros(5), np.ones(4)], [np.ones(3), np.ones(2)], n_boot=30, seed=0)
    assert np.allclose(out, (1.0 + 0.5) / 2)
    # 결함(양성)이 없는 범주는 AUROC가 없다
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


# 합산 비율의 부트스트랩이 범주별로 다시 뽑아 표시 수를 더하고 전체 장수(88)로 나눈 반복 계산과 같은지
# 확인한다. 빈 범주도 섞여 있고, 표시는 bool·0/1 정수·실수 어느 것으로 줘도 같다.
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
    # 표시로 0/1 정수와 실수도 받는다
    assert np.array_equal(out, pooled_rate_bootstrap([f.astype(int) for f in flags], n_boot=n_boot, seed=2))
    assert np.array_equal(out, pooled_rate_bootstrap([f.astype(float) for f in flags], n_boot=n_boot, seed=2))


# 분포의 성질: 평균이 점 추정 근처이고, 범주 안에서 따로 뽑으므로 분산이 범주별 이항 분산의 합에 맞는다. 모두
# 0이면 0, 모두 1이면 1, 셀 것이 없으면 NaN이다.
def test_pooled_rate_bootstrap_distribution():
    rng = np.random.default_rng(1)
    flags = [rng.random(300) < 0.05, rng.random(200) < 0.10]
    point = (flags[0].sum() + flags[1].sum()) / 500
    out = pooled_rate_bootstrap(flags)
    assert out.shape == (2000,)
    assert out.mean() == pytest.approx(point, abs=0.002)
    # 층화 추출: 분산은 범주별 이항 분산의 합이다
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


# 판정 세 갈래: 구간이 0을 포함하지 않고 크기가 최소 효과 이상이면 '차이 있음'(방향 무관, 최소 효과와 같아도
# 포함), 최소 효과보다 작으면 '차이 작음', 구간이 0을 포함하면 '판정 불가'다(docs/experiments.md 신뢰구간과
# 판정).
def test_verdict_three_outcomes():
    rng = np.random.default_rng(0)
    big = rng.normal(0.05, 0.005, size=2000)
    small = rng.normal(0.005, 0.001, size=2000)
    unclear = rng.normal(0.05, 0.05, size=2000)
    assert verdict(big, 0.05, 0.01) == "차이 있음"
    assert verdict(-big, -0.05, 0.01) == "차이 있음"  # 방향은 상관없다
    assert verdict(small, 0.005, 0.01) == "차이 작음"
    assert verdict(-small, -0.005, 0.01) == "차이 작음"
    assert verdict(unclear, 0.05, 0.01) == "판정 불가"
    assert verdict(big, 0.01, 0.01) == "차이 있음"  # 최소 효과와 정확히 같아도 포함한다


# 구간 끝이 정확히 0이면 0을 포함한 것으로 보아 '판정 불가'다. 표본이 모두 NaN이거나 차이가 0뿐이어도 '판정
# 불가'다.
def test_verdict_interval_touching_zero_is_inconclusive():
    samples = np.concatenate([np.zeros(100), np.full(1900, 0.1)])  # 2.5 백분위수가 정확히 0
    assert percentile_ci(samples)[0] == 0.0
    assert verdict(samples, 0.1, 0.01) == "판정 불가"
    assert verdict(np.full(10, np.nan), 0.1, 0.01) == "판정 불가"
    assert verdict(np.zeros(100), 0.0, 0.01) == "판정 불가"


# 판정이 정확히 95% 구간을 쓰는지 경계에서 확인한다. 2,000개 가운데 반대쪽에 놓인 수가 40·50개(2%·2.5%)면 0을
# 배제하고, 51·60개면 배제하지 않는다. 99%나 90% 구간을 잘못 쓰면 이 경계가 바뀌어 드러난다(기존 주석). 양·음
# 방향 모두 같다.
@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("across,excludes_zero", [(40, True), (50, True), (51, False), (60, False)])
def test_verdict_uses_the_95_percent_interval(sign, across, excludes_zero):
    # 재표본 2,000개 가운데 `across`개가 0의 반대쪽에 놓인다. 2.5 백분위수는 50번째와 51번째로 작은 값
    # 사이에 있으므로, 반대쪽이 2%(그리고 정확히 2.5%)면 여전히 0을 배제하고 3%면 배제하지 않는다.
    # 99% 구간을 쓰면 2% 경우를 '판정 불가'로, 90% 구간을 쓰면 3% 경우를 '차이 있음'으로 판정하게 된다.
    samples = sign * np.concatenate([np.full(across, -1.0), np.full(2000 - across, 1.0)])
    lo, hi = percentile_ci(samples)
    assert (lo > 0 or hi < 0) == excludes_zero
    assert verdict(samples, sign * 1.0, 0.01) == ("차이 있음" if excludes_zero else "판정 불가")
    assert verdict(samples, sign * 0.005, 0.01) == ("차이 작음" if excludes_zero else "판정 불가")


# 점 추정이 NaN이면(파이썬 float, numpy float64·float32 모두) 구간이 분명해도 '판정 불가'다. 정의되지 않은
# 비교가 '차이 있음'이나 '차이 작음'으로 보고되는 일을 막는다.
def test_verdict_nan_point_is_inconclusive():
    clear = np.full(100, 0.5)
    assert verdict(clear, 0.5, 0.01) == "차이 있음"
    assert verdict(clear, 0.005, 0.01) == "차이 작음"
    for nan in (float("nan"), np.float64("nan"), np.float32("nan")):
        assert verdict(clear, nan, 0.01) == "판정 불가"
        assert verdict(-clear, nan, 0.01) == "판정 불가"
    assert verdict(clear, float("nan"), 0.0) == "판정 불가"

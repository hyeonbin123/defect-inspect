# [파일 역할] 2단계 방법 비교 모듈(`defect_inspect.compare`)을 검증하는 테스트 묶음이다. 비지도 방법끼리(P0,
# D-S, DM)와 지도 학습 헤드 대 기준 비지도 방법을 같은 층화 부트스트랩 추출로 짝지어 비교하고, 가설 H5~H8과
# 교차점을 판정하는 코드다.
# [왜 필요한가] 판정 문구('차이 있음/차이 작음/판정 불가', '지지/기각')와 기준(구간이 0을 포함하는지, 최소
# 효과 1.0%p, 보지 못한 결함 유형의 최소 장수)은 docs/experiments.md 공통 규칙과 2단계 '가설과 판정'에 측정
# 전에 등록됐다. 이 코드가 규칙과 다르게 판정하면 README의 질문 1 답이 틀어진다.
# [설계] 실제 실행 결과 대신 정상 N(0,1), 결함 N(분리도,1) 점수와 결함 위치에 값을 더한 8×8 맵으로 만든 합성
# 범주 세 개를 쓴다. 분리도를 바꿔 방법 사이의 순서를 미리 알 수 있게 만든 뒤 판정이 그 순서를 따르는지 본다.

import numpy as np
import pytest

from defect_inspect import compare
from defect_inspect.metrics import aupro, pixel_auroc, pro_histograms

# 합성 범주 세 개, 범주마다 정상 80장·결함 24장, 결함 유형 세 가지(결함 i의 유형은 TYPES[i % 3]).
CATEGORIES = ("alpha", "beta", "gamma")
N_NEG, N_POS = 80, 24
TYPES = ["scratch", "hole", "stain"]


# 모든 테스트에서 부트스트랩 횟수를 등록값 2,000에서 200으로 줄여 빠르게 돈다(autouse). 구간의 정밀도는
# 떨어지지만 판정 논리를 보는 데는 충분하다.
@pytest.fixture(autouse=True)
def _few_resamples(monkeypatch):
    monkeypatch.setattr(compare, "N_BOOT", 200)


# 합성 범주 하나를 compare가 읽는 공통 형식(dict)으로 만든다. separation이 클수록 결함 점수가 정상에서 멀어져
# 이미지 AUROC가 높아진다. 보정 점수(cal_score)는 테스트 정상과 같은 N(0,1)이라 고정 임계값의 실제 오검출률이
# 목표 5% 근처가 돼야 한다. 픽셀 지표(pixel AUROC, AUPRO, PRO 히스토그램)는 metrics 모듈로 실제 계산해 넣는다.
# train_types를 주면 지도 학습 실행처럼 학습에 쓴 결함 유형을 기록한다.
def _category(name, rng, separation, *, train_types=None, cal=60):
    labels = np.array([0] * N_NEG + [1] * N_POS, dtype=np.int8)
    scores = np.concatenate([rng.normal(0, 1, N_NEG), rng.normal(separation, 1, N_POS)]).astype(np.float32)
    maps = rng.random((N_NEG + N_POS, 8, 8)).astype(np.float32)
    masks = np.zeros((N_NEG + N_POS, 8, 8), dtype=bool)
    masks[N_NEG:, 2:4, 2:4] = True
    maps[N_NEG:, 2:4, 2:4] += 0.6
    hist = pro_histograms(maps, masks, bins=100)
    cat = {
        "eval_images": np.array([f"{name}/{i}.JPG" for i in range(N_NEG + N_POS)]),
        "eval_labels": labels,
        "eval_defect_types": np.array([""] * N_NEG + [TYPES[i % 3] for i in range(N_POS)]),
        "eval_score": scores,
        "cal_score": rng.normal(0, 1, cal).astype(np.float32),
        "pixel_auroc": np.float64(pixel_auroc(maps, masks)),
        "aupro": np.float64(aupro(maps, masks)),
        "pro_edges": hist.edges,
        "pro_normal": hist.normal,
        "pro_components": hist.components,
        "pro_component_image": hist.component_image,
    }
    if train_types is not None:
        cat["train_defect_types"] = np.array(train_types)
    return cat


# 세 범주를 같은 분리도로 묶은 MethodRun. calibration은 cal_score를 얻은 방식(crossfit=PatchCore 교차 적합,
# holdout=Dinomaly·지도 학습, none=보정 점수 없음)이다.
def _run(name, separation, seed, calibration="crossfit", **kwargs):
    rng = np.random.default_rng(seed)
    run = compare.MethodRun(name, calibration)
    for c in CATEGORIES:
        run.cats[c] = _category(c, rng, separation, **kwargs)
    return run


# 확인: 분리도가 큰 방법의 평균 이미지 AUROC가 더 높고, 점 추정값이 부트스트랩 구간 안에 있으며, 고정 임계값
# 요약(교차 적합, 보정 점수 3 × 60 = 180개, 실제 오검출률이 목표 근처)이 나온다. 기준선 P0와의 짝지은 차이가
# '차이 있음'이고, 지도 학습이 없으면 supervised·hypotheses 키가 없다.
# 왜: 같은 추출을 두 방법에 써서 차이의 구간을 낸다는 공통 규칙(신뢰구간과 판정)이 실제로 짝지은 차이를
# 만드는지 본다.
def test_summary_and_paired_difference():
    weak, strong = _run("p0", 1.0, 1), _run("d-s", 3.0, 2)
    report = compare.build_report([weak, strong], {}, reference_name="d-s")
    s = report["unsupervised"]
    assert s["d-s"]["macro_image_auroc"] > s["p0"]["macro_image_auroc"]
    for v in s.values():
        lo, hi = v["macro_image_auroc_ci"]
        assert lo <= v["macro_image_auroc"] <= hi
        lo, hi = v["macro_aupro_ci"]
        assert lo - 0.01 <= v["macro_aupro"] <= hi + 0.01
        fixed = v["fixed_threshold"]
        assert fixed["calibration"] == "crossfit" and fixed["n_cal"] == 180
        # Calibration scores and test normals are both N(0, 1): the rate is near the 5% target.
        assert 0.0 <= fixed["fpr"] < 0.15
    d = report["vs_baseline"]["pairs"]["d-s"]
    assert d["verdict"] == "차이 있음" and d["diff"] > 0.1
    assert "supervised" not in report and "hypotheses" not in report
    assert "d-s - p0" in compare.format_report(report)


# 확인: 같은 점수를 가진 두 방법은 차이가 정확히 0이고 '판정 불가'다. 짝지은 추출이 아니면 같은 방법끼리도
# 구간이 흔들려 차이가 나올 수 있다.
def test_identical_methods_are_not_different():
    a, b = _run("p0", 2.0, 5), _run("d-s", 2.0, 5)
    d = compare.build_report([a, b], {}, "p0")["vs_baseline"]["pairs"]["d-s"]
    assert d["diff"] == 0.0 and d["verdict"] == "판정 불가"


# 확인: 등록한 세 갈래 문구. 구간이 0을 포함하지 않고 크기가 최소 효과(1.0%p) 이상이면 '차이 있음', 0을
# 포함하지 않지만 1.0%p보다 작으면 '차이 작음', 0을 포함하면 '판정 불가'다. 음수 방향도 대칭이다.
def test_difference_wording():
    rng = np.random.default_rng(0)
    clear = rng.normal(0.03, 0.005, 500)
    assert compare.difference(0.03, clear)["verdict"] == "차이 있음"
    assert compare.difference(0.004, rng.normal(0.004, 0.001, 500))["verdict"] == "차이 작음"
    assert compare.difference(0.01, rng.normal(0.01, 0.02, 500))["verdict"] == "판정 불가"
    assert compare.difference(-0.03, -clear)["verdict"] == "차이 있음"


# 확인: 한 범주라도 평가 이미지 순서가 다른 실행을 섞으면 'different images' ValueError다. 같은 인덱스로
# 부트스트랩을 짝짓기 때문에 이미지가 어긋나면 비교 자체가 무의미해진다.
def test_different_evaluation_images_are_rejected():
    a, b = _run("p0", 2.0, 1), _run("d-s", 2.0, 2)
    b.cats["beta"]["eval_images"] = b.cats["beta"]["eval_images"][::-1].copy()
    with pytest.raises(ValueError, match="different images"):
        compare.build_report([a, b], {}, "p0")


# 확인: unseen_masks가 '학습 결함의 유형과 하나도 겹치지 않는 테스트 결함'만 고른다. 학습 유형이 scratch,
# hole|scratch(여러 유형은 |로 구분)면 stain만 남고(범주당 8장), 세 유형을 모두 학습했으면 남는 결함이 없다.
# 왜: H7(학습에서 보지 못한 결함 유형)은 실행마다 학습에 쓴 k장의 유형으로 부분집합을
# 정한다(docs/experiments.md 2단계 H7).
def test_unseen_masks_follow_the_training_types():
    sup = _run("sup", 2.0, 3, calibration="holdout", train_types=["scratch", "hole|scratch"])
    masks, total = compare.unseen_masks(sup)
    # Only "stain" defects (every third one) share no type with the training defects.
    assert set(masks) == set(CATEGORIES) and total == 3 * 8
    assert masks["alpha"].tolist() == [i % 3 == 2 for i in range(N_POS)]
    seen_all = _run("sup", 2.0, 3, calibration="holdout", train_types=["scratch", "hole", "stain"])
    assert compare.unseen_masks(seen_all) == ({}, 0)


# k마다 시드 두 개(0, 1)의 지도 학습 실행을 만든다. 분리도로 k가 클수록 지도 학습이 강해지는 상황을 흉내 내고,
# 보정은 홀드아웃(cal 40장)이다.
def _supervised(separations, train_types):
    return {
        (k, s): _run(
            f"sup-k{k}-s{s}", sep, 100 + 10 * k + s, calibration="holdout", train_types=train_types, cal=40
        )
        for k, sep in separations.items()
        for s in (0, 1)
    }


# 확인: 지도 학습 행의 라벨 비용이 k + 20(검증 20장 포함, 라벨 회계 규칙)이고, H5(k = 5·10에서 지도 학습이
# 낮음), H6(k = 40에서 높음), H8(D-S > P0)이 '지지'이며, 교차점이 '차이 있음'으로 앞서는 가장 작은 k(여기서는
# 40)다. 보지 못한 유형에서 k = 40의 지도 학습이 앞서면 H7은 그 k를 적은 '기각'이 된다.
# monkeypatch: 합성 데이터의 보지 못한 결함이 3범주 × 8 = 24장이라 등록 최소치 30장(MIN_UNSEEN_POOLED)에 못
# 미친다. 판정 경로를 타게 하려고 이 테스트에서만 10으로 낮춘다.
def test_supervised_crossing_and_hypotheses(monkeypatch):
    monkeypatch.setattr(compare, "MIN_UNSEEN_POOLED", 10)
    unsup = [_run("p0", 1.5, 1), _run("d-s", 2.2, 2)]
    sup = _supervised({5: 0.8, 10: 1.2, 20: 1.9, 40: 4.0}, ["scratch", "hole"])
    report = compare.build_report(unsup, sup, reference_name="d-s")
    rows = report["supervised"]
    assert rows["5"]["labels_with_validation"] == 25 and rows["5"]["seeds"] == [0, 1]
    assert rows["5"]["all_defects"]["diff"] < 0 < rows["40"]["all_defects"]["diff"]
    h = report["hypotheses"]
    assert h["H5 (k=5)"] == "지지" and h["H5 (k=10)"] == "지지" and h["H6 (k=40)"] == "지지"
    assert h["교차점"] == "k = 40"
    assert h["H8 (d-s > P0)"] == "지지"
    # Unseen types: 8 "stain" defects per category are judged at every k, and at k=40 supervised is ahead.
    assert rows["40"]["unseen_types"]["judged"] and rows["40"]["unseen_types"]["unseen_defects_mean"] == 24
    assert h["H7"].startswith("기각") and "40" in h["H7"]
    text = compare.format_report(report)
    assert "교차점" in text and "unseen-type" in text
    assert set(report["supervised_runs"]) == {f"k{k}-s{s}" for k in (5, 10, 20, 40) for s in (0, 1)}


# 확인: 등록 최소치(합계 30장)를 그대로 두면 24장으로는 판정하지 않고 H7이 '판정 불가'다. 표본이 적을 때
# 결론을 내지 않는 규칙을 지키는지 본다.
def test_unseen_types_need_enough_defects():
    unsup = [_run("p0", 1.5, 1), _run("d-s", 2.2, 2)]
    sup = _supervised({5: 1.0, 40: 3.0}, ["scratch", "hole"])  # 24 unseen defects < the pooled minimum of 30
    report = compare.build_report(unsup, sup, reference_name="d-s")
    assert not report["supervised"]["5"]["unseen_types"]["judged"]
    assert report["hypotheses"]["H7"].startswith("판정 불가")


# 확인: k = 40에서도 지도 학습이 기준 비지도보다 낮으면 교차점은 'k = 40까지 교차 없음', H6는
# '기각'이다(등록한 문구).
def test_no_crossing_is_reported():
    unsup = [_run("p0", 1.5, 1), _run("d-s", 3.5, 2)]
    sup = _supervised({5: 0.5, 40: 1.0}, ["scratch", "hole", "stain"])
    h = compare.build_report(unsup, sup, reference_name="d-s")["hypotheses"]
    assert h["교차점"] == "k = 40까지 교차 없음" and h["H6 (k=40)"] == "기각"


# 확인: 보정 점수가 없는 실행(calibration none)은 고정 임계값 요약이 None이고, 표에서는 그 행의
# 보정·오검출률·검출률 칸이 '-'로 나온다. 보정 점수 없이 임계값을 지어내지 않는다.
def test_method_without_calibration_scores_has_no_fixed_threshold():
    a = _run("p0", 2.0, 1)
    b = _run("dm", 2.0, 2, calibration="none")
    report = compare.build_report([a, b], {}, "p0")
    assert report["unsupervised"]["dm"]["fixed_threshold"] is None
    assert "| dm |" in compare.format_report(report)


# 확인: test 절차에서는 --reference(dev에서 고른 기준 비지도 방법)가 꼭 있어야 한다.
# 왜: 기준 비지도 방법은 dev 평균 AUROC로 정한다(2단계 '기준 비지도 방법'). test에서 가장 높은 방법을 자동으로
# 고르면 봉인 테스트를 보고 기준을 정하는 셈이 된다.
def test_cli_needs_a_reference_for_the_test_protocol():
    with pytest.raises(SystemExit):
        compare.main(["--protocol", "test"])

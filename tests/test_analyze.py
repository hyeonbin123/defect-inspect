# [파일 역할] analyze.py(1단계 표: 정확도, 임계값 보정 세 방식, 가설 H1~H4, 보정 표본 수 곡선, 점수 CSV)를
# 검증하는 테스트 묶음.
# [왜 필요한가] 1단계의 핵심 결론(재대입 86%, 홀드아웃이 이론 구간 안, 교차 적합이 보수적)은 이 모듈의
# 계산에서 나온다. 데이터 없이 돌도록, 분포를 아는 합성 점수로 run_patchcore와 같은 형식의 출력 폴더를 만들어
# 넣고 (1) 리포트 구조와 구간이 앞뒤가 맞는지 (2) conformal 임계값이 손 계산과 같은지 (3) 가설 판정 규칙이
# 등록한 문장대로 갈리는지 (4) 기준 모의가 교환 가능한 점수에서는 이론과 맞고 치우친 점수는 잡아내는지
# 확인한다.
# [구성] 합성 범주 3개(alpha, beta, gamma), 부트스트랩·기준 모의 횟수는 monkeypatch로 줄여 빠르게 돈다.

import csv
import json

# metrics의 픽셀 지표 함수는 합성 점수 맵으로 npz의 pixel_auroc·aupro·PRO 히스토그램을 실제 실행과 같은
# 방식으로 채우는 데 쓴다.
import numpy as np
import pytest

from defect_inspect import analyze
from defect_inspect.metrics import aupro, pixel_auroc, pro_histograms

CATEGORIES = ("alpha", "beta", "gamma")


# 합성 범주 하나를 run_patchcore와 같은 열 이름의 npz로 쓴다.
# - 평가: 정상 120장 N(0, 1), 결함 30장 N(2.5, 1). 홀드아웃 뱅크 점수는 전체 뱅크 점수에 작은 잡음을 더한 것
# - 정상 풀 200장: dev 절차처럼 겹 1~4(겹당 50장). 교차 적합 점수는 평가 정상과 같은 N(0, 1)이라 교환
#   가능하고, 재대입 점수는 resub_shift(기본 −2)만큼 낮게 둔다(자기 이미지로 만든 뱅크라 점수가 낮게 나오는
#   상황을 흉내)
# - 점수 맵: 무작위 맵에서 결함 이미지의 4×4 영역만 0.5 높여 픽셀 지표가 의미 있는 값을 갖게 한다
def _write_category(run_dir, name, rng, *, resub_shift=-2.0, n_pool=200, n_neg=120, n_pos=30):
    """합성 범주 하나: 정상 점수 ~ N(0, 1), 결함 ~ N(2.5, 1), 재대입 점수는 지나치게 낮다."""
    folds = np.arange(n_pool) % 4 + 1  # dev 절차: 겹 1..4
    labels = np.array([0] * n_neg + [1] * n_pos, dtype=np.int8)
    eval_full = np.concatenate([rng.normal(0, 1, n_neg), rng.normal(2.5, 1, n_pos)]).astype(np.float32)
    eval_holdout = eval_full + rng.normal(0, 0.05, n_neg + n_pos).astype(np.float32)
    maps = rng.random((n_neg + n_pos, 16, 16)).astype(np.float32)
    masks = np.zeros((n_neg + n_pos, 16, 16), dtype=bool)
    for i in range(n_neg, n_neg + n_pos):
        masks[i, 4:8, 4:8] = True
        maps[i, 4:8, 4:8] += 0.5
    hist = pro_histograms(maps, masks, bins=200)
    np.savez(
        run_dir / f"{name}.npz",
        eval_images=np.array([f"{name}/eval/{i}.JPG" for i in range(n_neg + n_pos)]),
        eval_labels=labels,
        eval_defect_types=np.array([""] * n_neg + ["scratch"] * n_pos),
        eval_score_full=eval_full,
        eval_score_holdout=eval_holdout,
        pool_images=np.array([f"{name}/pool/{i}.JPG" for i in range(n_pool)]),
        pool_folds=folds,
        pool_score_resub=rng.normal(resub_shift, 1, n_pool).astype(np.float32),
        pool_score_oof=rng.normal(0, 1, n_pool).astype(np.float32),
        pixel_auroc=np.float64(pixel_auroc(maps, masks)),
        aupro=np.float64(aupro(maps, masks)),
        pro_edges=hist.edges,
        pro_normal=hist.normal,
        pro_components=hist.components,
        pro_component_image=hist.component_image,
    )


# 합성 범주 3개와 run.json(구성 이름 fake, dev 절차)을 임시 폴더에 만든다. rng 시드를 고정해 매번 같은
# 데이터가 나온다.
@pytest.fixture
def run_dir(tmp_path):
    rng = np.random.default_rng(0)
    for name in CATEGORIES:
        _write_category(tmp_path, name, rng)
    meta = {
        "config": {"name": "fake"},
        "protocol": "dev",
        "commit": "abc1234",
        "categories": [{"category": c} for c in CATEGORIES],
    }
    (tmp_path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


# 모든 테스트에서 보정 표본 수 곡선의 기준 모의(실현 100번 × 추출 300번)를 20 × 100으로 줄여 시간을 아낀다.
# 기준 모의를 직접 검사하는 테스트만 다시 늘린다.
@pytest.fixture(autouse=True)
def _small_reference(monkeypatch):
    monkeypatch.setattr(analyze, "REF_REALISATIONS", 20)
    monkeypatch.setattr(analyze, "REF_DRAWS", 100)


# 리포트 전체가 앞뒤가 맞는지 본다: 평균 AUROC가 구간 안에 있고, 히스토그램으로 다시 계산한 AUPRO가 정확한
# AUPRO와 0.01 안에서 같으며(구간 계산에 쓰는 근사가 점 추정과 어긋나지 않음), 두 목표(5%, 1%)·세 방식 모두
# 오검출률이 부트스트랩 구간과 Clopper-Pearson 구간 안에 있다.
# 특히 홀드아웃은 한 겹(dev 절차의 겹 1, 범주당 50장 × 3 = 150장)만으로, 교차 적합은 풀 전부(600장)로
# 보정하는지 확인한다. 보정 방식끼리 보정 점수가 섞이는 회귀를 막는다.
def test_report_structure_and_ranges(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "N_BOOT", 200)
    report = analyze.build_report(run_dir)
    acc = report["accuracy"]
    assert set(acc["per_category"]) == set(CATEGORIES)
    lo, hi = acc["macro_image_auroc_ci"]
    assert 0.9 < lo <= acc["macro_image_auroc"] <= hi <= 1.0
    lo, hi = acc["macro_aupro_ci"]
    assert lo <= acc["macro_aupro_binned"] <= hi
    assert abs(acc["macro_aupro"] - acc["macro_aupro_binned"]) < 0.01
    for alpha in ("0.05", "0.01"):
        cal = report["calibration"][alpha]
        for strategy in analyze.STRATEGIES:
            v = cal[strategy]
            assert v["n_normal"] == 360 and v["n_defect"] == 90
            assert v["fpr"] == v["fp"] / v["n_normal"]
            assert v["fpr_ci"][0] <= v["fpr"] <= v["fpr_ci"][1]
            assert v["fpr_clopper_pearson"][0] <= v["fpr"] <= v["fpr_clopper_pearson"][1]
        # 홀드아웃 방식은 한 겹만으로 보정한다.
        assert cal["holdout"]["n_cal_total"] == 150 and cal["crossfit"]["n_cal_total"] == 600
    text = analyze.format_report(report)
    assert "resubstitution" in text and "H1" in text


# 교차 적합 임계값과 오검출·검출 수를 손으로 다시 계산해 비교한다: 보정 점수를 정렬해 l = ceil((n+1)(1−α))번째
# 값이 임계값이고, 점수가 임계값보다 "클 때" 불량이다(docs/experiments.md 임계값 보정 규칙). 홀드아웃 임계값은
# dev 절차에서 겹 1의 점수로 정하는지도 본다. 규칙의 등호 방향이나 겹 번호가 바뀌는 실수를 잡는다.
def test_calibration_matches_a_direct_computation(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "N_BOOT", 50)
    _, cats = analyze.load_run(run_dir)
    cal = analyze.calibration(cats, "dev", 0.05)
    fp = tp = 0
    for cat in cats.values():
        scores = np.sort(cat["pool_score_oof"])
        rank = int(np.ceil((len(scores) + 1) * 0.95))
        threshold = scores[rank - 1]
        full = cat["eval_score_full"]
        fp += int((full[cat["eval_labels"] == 0] > threshold).sum())
        tp += int((full[cat["eval_labels"] == 1] > threshold).sum())
    assert cal["crossfit"]["fp"] == fp and cal["crossfit"]["tp"] == tp
    # 홀드아웃 임계값은 겹 1(dev 절차)에서 정하고 홀드아웃 모델의 점수에 적용한다.
    cat = cats["alpha"]
    held = np.sort(cat["pool_score_oof"][cat["pool_folds"] == 1])
    rank = int(np.ceil((len(held) + 1) * 0.95))
    assert cal["holdout"]["per_category"]["alpha"]["threshold"] == pytest.approx(float(held[rank - 1]))


# 재대입 점수가 평가 정상보다 2 표준편차 낮으면 임계값이 너무 낮아 정상 대부분이 불량으로 나오고 H1이 "지지"로
# 판정되는지 본다. 실제 1단계에서 재대입 오검출률 86%가 나온 메커니즘을 합성으로 재현해 판정 경로를 확인한다.
def test_resubstitution_with_too_low_scores_supports_h1(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "N_BOOT", 200)
    report = analyze.build_report(run_dir)
    # 보정 점수가 평가 정상보다 2 표준편차 낮다: 거의 모든 것이 불량으로 표시된다.
    assert report["calibration"]["0.05"]["resubstitution"]["fpr"] > 0.3
    assert report["hypotheses"]["H1"] == "지지"


# hypotheses()가 읽는 값만 담은 최소 보정 결과를 만든다(구간 경계값을 직접 넣어 판정 규칙만 시험하기 위해).
def _cal(resub_ci, hold_fpr, band, cross_ci, diff_ci):
    return {
        "resubstitution": {"fpr_ci": resub_ci},
        "holdout": {"fpr": hold_fpr, "theory_band": band},
        "crossfit": {"fpr_ci": cross_ci},
        "tpr_crossfit_minus_holdout": {"ci": diff_ci},
    }


# H1~H4 판정 규칙을 경계 근처 값으로 세 경우씩 확인한다(목표 5%).
# - 첫째: 재대입 하한 11% ≥ 10%, 홀드아웃 5%가 이론 구간 안, 교차 적합 상한 4.9% ≤ 5%, 검출률 차이 하한 −1%p >
#   −2%p → 모두 지지
# - 둘째: 모두 기각. 홀드아웃은 구간보다 높아 방향까지 적는다
# - 셋째: 구간이 기준선을 걸쳐 판정 불가, 홀드아웃은 구간보다 낮아 기각(낮음)
# 등록한 판정 문장(docs/experiments.md 1단계)과 코드의 부등호가 어긋나지 않게 지킨다.
@pytest.mark.parametrize(
    ("cal", "expected"),
    [
        (
            _cal([0.11, 0.15], 0.05, [0.04, 0.06], [0.03, 0.049], [-0.01, 0.03]),
            {"H1": "지지", "H2": "지지", "H3": "지지", "H4": "지지"},
        ),
        (
            _cal([0.04, 0.09], 0.07, [0.04, 0.06], [0.051, 0.07], [-0.06, -0.03]),
            {"H1": "기각", "H2": "기각 (이론 구간보다 높음)", "H3": "기각", "H4": "기각"},
        ),
        (
            _cal([0.08, 0.12], 0.03, [0.04, 0.06], [0.04, 0.06], [-0.03, 0.01]),
            {"H1": "판정 불가", "H2": "기각 (이론 구간보다 낮음)", "H3": "판정 불가", "H4": "판정 불가"},
        ),
    ],
)
def test_hypothesis_rules(cal, expected):
    assert analyze.hypotheses(cal, 0.05) == expected


# 이미지별 점수 CSV가 커밋용 형식(LF 줄 끝, \r 없음)이고 행 수(범주 3 × (평가 150 + 풀 200)), 라벨, 소수 6자리
# 점수, 풀의 겹 번호가 원래 배열과 같게 다시 읽히는지 본다. 이 CSV만으로 수치를 다시 계산할 수 있어야 한다는
# 모듈의 약속을 지키는 테스트다.
def test_scores_csv_round_trip(run_dir, tmp_path):
    _, cats = analyze.load_run(run_dir)
    path = tmp_path / "scores.csv"
    analyze.write_scores_csv(cats, path)
    assert b"\r" not in path.read_bytes()
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * (150 + 200)
    eval_rows = [r for r in rows if r["category"] == "beta" and r["set"] == "eval"]
    assert [r["label"] for r in eval_rows].count("anomaly") == 30
    assert float(eval_rows[0]["score_full"]) == pytest.approx(
        float(cats["beta"]["eval_score_full"][0]), abs=1e-6
    )
    pool_rows = [r for r in rows if r["category"] == "beta" and r["set"] == "pool"]
    assert {r["fold"] for r in pool_rows} == {"1", "2", "3", "4"}


# 보정 표본 수 곡선(n = 20, 50, 100, 200)의 기준 모의를 확인한다. 교환 가능한 균등 난수로 같은 절차를 돌린
# 기준 평균이 conformal 규칙의 이론 평균과 0.006 안에서 맞아야 하고(모의 자체가 옳은지), 이 합성 데이터의 교차
# 적합 점수는 평가 정상과 같은 분포라 관측 평균이 대부분의 n에서 기준 구간 안에 들어야 한다. 규칙 변경 2(Beta
# 분위 대신 같은 크기의 모의와 비교)의 구현을 지키는 테스트다.
def test_finite_sample_reference_covers_exchangeable_scores(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "REF_REALISATIONS", 60)
    _, cats = analyze.load_run(run_dir)
    rows = analyze.finite_sample(cats, 0.05)
    assert [r["n"] for r in rows] == [20, 50, 100, 200]
    for r in rows:
        lo, hi = r["reference_interval"]
        assert lo < r["reference_mean"] < hi
        # 교환 가능한 모의: 그 평균은 conformal 규칙의 주변(marginal) 비율에 놓인다.
        assert r["reference_mean"] == pytest.approx(r["theory_mean"], abs=0.006)
    # 픽스처의 교차 적합 점수와 평가 정상은 같은 분포에서 나온다.
    assert sum(r["within_reference"] for r in rows) >= 3


# 교차 적합 점수를 1만큼 올려(임계값이 너무 엄격해짐) 모든 n에서 관측 평균이 기준 구간보다 낮게 나와
# "치우침"으로 표시되는지 본다. 실제 1단계에서 교차 적합이 보수 쪽으로 치우친 것을 이 비교로 가려냈으므로, 그
# 감지가 작동하는지 확인한다.
def test_finite_sample_flags_a_conservative_shift(run_dir):
    _, cats = analyze.load_run(run_dir)
    for cat in cats.values():
        cat["pool_score_oof"] = (
            cat["pool_score_oof"] + 1.0
        )  # 보정 점수가 너무 높다: 임계값이 지나치게 엄격해진다
    rows = analyze.finite_sample(cats, 0.05)
    assert all(not r["within_reference"] and r["mean"] < r["reference_interval"][0] for r in rows)

# [파일 역할] 3-A 분석 모듈(`defect_inspect.analyze_perturb`)을 검증하는 테스트 묶음이다. run_perturb가 저장한
# '조건 × 이미지' 점수에서 조건마다 평균 이미지 AUROC와, 원본 실행의 보정 점수로 한 번 정한 임계값의 합산
# 오검출률·검출률, 교란 없는 조건과의 짝지은 차이를 표로 만들고 H9를 판정하는 코드다.
# [왜 필요한가] H9는 'P0에서 평균 이미지 AUROC가 1.0%p 미만으로 떨어지는데 합산 오검출률이 10%(목표의 두 배)
# 이상인 조건이 하나라도 있다'를 점 추정값으로 판정한다(docs/experiments.md 3-A). README의 '어둡게 하면
# AUROC는 그대로인데 오검출률이 15%, 41%로 뛴다'는 결론이 이 판정에서 나온다. 그래서 경계(정확히 1.0%p, 정확히
# 10%), '증가분이 아니라 비율 자체', '점수 > 임계값', 범주 합산(비율의 평균이 아님), 조건 사이의 짝지은
# 부트스트랩, 12개 범주 전부일 때만 판정한다는 규칙을 하나씩 고정한다.
# [흐름] 합성 실행 만들기 → 표 모양과 점 추정 → 경계 규칙 → H9 판정 → 부트스트랩 대조 → 보정 점수 없는
# 실행·잘못된 실행 → 파일 읽기와 출력 형식 → CLI(방법 순서, 일부 범주 실행, 프로토콜 확인).

import json

import numpy as np
import pytest

from defect_inspect import analyze_perturb
from defect_inspect.calibrate import conformal_threshold
from defect_inspect.conditions import condition_names
from defect_inspect.metrics import auroc
from defect_inspect.stats import percentile_ci, stratified_indices

# 범주별 (정상 수, 결함 수). 합계 정상 250장, 결함 64장이다. NAMES는 등록한 조건 16개(clean + 15)이고, 그
# 가운데 세 조건만 점수를 바꿔 효과를 미리 안다.
# Categories of different sizes, so that a pooled rate differs from a mean of per-category rates.
SIZES = {"alpha": (80, 24), "beta": (50, 10), "gamma": (120, 30)}
NAMES = condition_names()
SHIFTED, BLURRED, BROKEN = "brightness-2", "blur-3", "jpeg-3"


# 점수를 1/1024 단위로 반올림해 두면 +2.0, −2.5를 더해도 float32에서 정확해져 'AUROC 차이가 정확히 0' 같은
# 단언이 가능하다. meta에는 run_perturb의 run.json처럼 방법, 프로토콜, 원본 실행, 범주별
# clean_max_rel_diff(교란 없는 재채점이 원본 점수와 얼마나 같은지)를 넣는다.
def _run(seed=0, cal=100, method="p0", protocol="dev"):
    """A synthetic perturbation run.

    Normals and calibration scores are N(0, 1), defects N(3, 1). `SHIFTED` adds 2 to every score (the
    ranking stays, the fixed thresholds are overrun), `BLURRED` pulls the defects down among the normals
    (accuracy falls, the normals do not move), `BROKEN` does both. Every other condition equals clean.
    """
    rng = np.random.default_rng(seed)
    cats = {}
    for name, (n_neg, n_pos) in SIZES.items():
        labels = np.array([0] * n_neg + [1] * n_pos, dtype=np.int8)
        clean = np.concatenate([rng.normal(0, 1, n_neg), rng.normal(3, 1, n_pos)])
        clean = np.round(clean * 1024) / 1024  # the shifts below are then exact in float32
        scores = np.tile(clean, (len(NAMES), 1))
        scores[NAMES.index(SHIFTED)] += 2.0
        scores[NAMES.index(BLURRED), n_neg:] -= 2.5
        scores[NAMES.index(BROKEN)] += 2.0
        scores[NAMES.index(BROKEN), n_neg:] -= 2.5
        cats[name] = {
            "eval_images": np.array([f"{name}/{i}.JPG" for i in range(n_neg + n_pos)]),
            "eval_labels": labels,
            "conditions": np.array(NAMES),
            "scores": scores.astype(np.float32),
            "cal_score": rng.normal(0, 1, cal).astype(np.float32),
        }
    meta = {
        "method": method,
        "protocol": protocol,
        "commit": "abc1234",
        "device": "cpu",
        "source": f"outputs/{method}-{protocol}",
        "source_commit": "src1234",
        "conditions": NAMES,
        "categories": [{"category": c, "clean_max_rel_diff": 1e-5 * (i + 1)} for i, c in enumerate(cats)],
    }
    return {"meta": meta, "cats": cats}


# 합성 실행을 run_perturb 출력과 같은 폴더 구조(run.json + 범주별 npz)로 쓴다. load와 CLI를 실제 파일로 검사할
# 때 쓴다.
def _write(run, run_dir):
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(run["meta"]), encoding="utf-8")
    for name, cat in run["cats"].items():
        np.savez(run_dir / f"{name}.npz", **cat)


# 표에서 조건 이름으로 행 하나를 찾는다.
def _row(table, name):
    return next(row for row in table if row["condition"] == name)


# 범주 하나짜리 실행을 손으로 만든다. 조건마다 정상 점수만 바꾸고 결함 점수는 같게 두어, 경계 규칙을 정확한
# 숫자로 시험한다.
def _tiny(normals, defects, cal):
    """A one-category run written out by hand: `normals` maps each condition to its normal scores.

    The defects score the same under every condition.
    """
    names = list(normals)
    n_neg = len(normals[names[0]])
    scores = np.array([[*normals[name], *defects] for name in names], dtype=np.float32)
    cat = {
        "eval_images": np.array([f"only/{i}.JPG" for i in range(n_neg + len(defects))]),
        "eval_labels": np.array([0] * n_neg + [1] * len(defects), dtype=np.int8),
        "conditions": np.array(names),
        "scores": scores,
        "cal_score": np.asarray(cal, dtype=np.float32),
    }
    return {"meta": {"method": "p0", "categories": [{"category": "only"}]}, "cats": {"only": cat}}


# 보정 점수 0..99(목표 5%)의 임계값은 95다. 정상 100장 가운데 정확히 count장만 그 위에 오도록 점수를 만들어
# 오검출률을 count%로 맞춘다.
def _above(count, of=100):
    """Normal scores of which exactly `count` lie above the threshold 95 of `np.arange(100)`."""
    return [100.0] * count + [0.0] * (of - count)


# 확인: 조건마다 한 행이 등록 순서로 나오고(clean이 먼저), clean 행에는 차이 키가 없으며 나머지 행에는
# AUROC·오검출률·검출률의 차이와 구간이 있다. 모든 점 추정값이 자기 구간 안에 있고, 보정 점수와 정상이 같은
# 분포라 clean 오검출률이 목표 5% 근처다.
def test_table_has_one_row_per_condition_with_clean_first():
    table = analyze_perturb.condition_table(_run(), n_boot=100)
    assert [row["condition"] for row in table] == NAMES
    clean = table[0]
    assert set(clean) == {
        "condition",
        "macro_image_auroc",
        "macro_image_auroc_ci",
        "fpr",
        "fpr_ci",
        "tpr",
        "tpr_ci",
    }
    for row in table[1:]:
        assert set(row) == set(clean) | {"d_auroc", "d_auroc_ci", "d_fpr", "d_fpr_ci", "d_tpr", "d_tpr_ci"}
    for row in table:
        for key in ("macro_image_auroc", "fpr", "tpr"):
            lo, hi = row[f"{key}_ci"]
            assert lo <= row[key] <= hi
    # Calibration scores and clean normals are both N(0, 1): the clean rate is near the 5% target.
    assert 0.0 < clean["fpr"] < 0.12 and clean["macro_image_auroc"] > 0.95


# 확인: 점 추정값을 직접 계산한 값과 맞춘다. 평균 AUROC는 범주별 AUROC의 평균이고, 오검출률·검출률은 범주별
# 비율의 평균이 아니라 개수를 합쳐 전체 정상 250장·결함 64장으로 나눈 합산 비율이다(공통 규칙의 '합산').
# 목표를 1%로 엄격하게 하면 오검출률이 줄어 α가 실제로 임계값을 정하는지도 본다.
# 왜: 범주 크기가 다르므로(SIZES) 합산과 평균이 다른 값이 된다. 이 차이를 잡으려고 크기를 일부러 다르게 했다.
def test_point_estimates_match_a_direct_computation():
    run = _run()
    table = analyze_perturb.condition_table(run, n_boot=20)
    for name in ("clean", SHIFTED, BLURRED):
        k = NAMES.index(name)
        aucs, fp, tp = [], 0, 0
        for cat in run["cats"].values():
            labels, scores = cat["eval_labels"], cat["scores"][k]
            threshold = conformal_threshold(cat["cal_score"], 0.05).value
            aucs.append(auroc(scores[labels == 0], scores[labels == 1]))
            fp += int((scores[labels == 0] > threshold).sum())
            tp += int((scores[labels == 1] > threshold).sum())
        row = _row(table, name)
        assert row["macro_image_auroc"] == pytest.approx(np.mean(aucs), abs=1e-12)
        assert row["fpr"] == fp / 250 and row["tpr"] == tp / 64  # pooled counts, not a mean of rates
    # The target decides the thresholds: a stricter one flags fewer normals.
    strict = analyze_perturb.condition_table(run, alpha=0.01, n_boot=20)
    assert strict[0]["fpr"] < table[0]["fpr"]


# 확인: 판정 규칙은 '점수 > 임계값'이라 임계값(95)과 정확히 같은 점수는 불량이 아니다. 점 추정과 부트스트랩
# 표본 모두 같은 규칙을 쓰는지, 같은 추출을 직접 만들어 구간까지 맞춘다.
# 왜: conformal 임계값은 보정 점수 가운데 하나이므로 같은 값이 실제로 나온다. ≥로 바꾸면 오검출률이 이론값보다
# 커진다.
def test_a_score_equal_to_its_threshold_is_not_flagged():
    # 100 calibration scores 0..99 at a 5% target: the 96th smallest, 95, is the threshold.
    cal = np.arange(100)
    assert conformal_threshold(cal, 0.05).value == 95.0
    normals, defects = [95.0, 95.0, 96.0, 10.0], [95.0, 200.0]
    run = _tiny({"clean": normals, "same": normals}, defects, cal)
    n_boot, seed = 40, 2
    table = analyze_perturb.condition_table(run, n_boot=n_boot, seed=seed)
    # The rule is `score > threshold`: the three scores of exactly 95 are not defects.
    assert table[0]["fpr"] == 0.25 and table[0]["tpr"] == 0.5
    # The resamples use the same rule.
    neg_idx, pos_idx = stratified_indices([4, 2], n_boot=n_boot, seed=seed)
    fpr = (np.array(normals)[neg_idx] > 95.0).mean(axis=1)
    tpr = (np.array(defects)[pos_idx] > 95.0).mean(axis=1)
    assert table[0]["fpr_ci"] == pytest.approx(percentile_ci(fpr), abs=1e-12)
    assert table[0]["tpr_ci"] == pytest.approx(percentile_ci(tpr), abs=1e-12)


# 확인: 세 가지 전형적인 교란 효과가 표에 그대로 나타난다.
# - 모든 점수를 같이 올림(SHIFTED): 순위가 그대로라 AUROC 차이와 구간이 정확히 0인데 오검출률은 50%를 넘는다.
#   H9가 겨냥한 '정확도로는 보이지 않는 고장'이다.
# - 결함 점수만 내림(BLURRED): AUROC와 검출률이 떨어지고 정상은 그대로라 오검출률 차이가 정확히 0이다.
# - 둘 다(BROKEN): 오검출률은 SHIFTED와 같지만 AUROC 하락이 보이므로 H9 조건이 아니다.
# 바뀌지 않은 조건은 차이와 구간이 모두 정확히 0이다(조건 사이에 같은 추출을 쓰기 때문). 결국 H9는 SHIFTED
# 하나로 '지지'다.
def test_upward_shift_keeps_accuracy_and_breaks_the_threshold_h9_supported():
    table = analyze_perturb.condition_table(_run(), n_boot=200)
    clean, shifted = table[0], _row(table, SHIFTED)
    # Adding a constant changes no rank: the AUROC is exactly the clean one, also in every resample.
    assert shifted["macro_image_auroc"] == clean["macro_image_auroc"]
    assert shifted["d_auroc"] == 0.0 and shifted["d_auroc_ci"] == [0.0, 0.0]
    assert shifted["fpr"] > 0.5 and shifted["d_fpr"] > 0.45
    assert shifted["d_fpr_ci"][0] > 0.3 and shifted["d_tpr"] >= 0.0
    # Lost defects: accuracy falls and the normals stay where they were.
    blurred = _row(table, BLURRED)
    assert blurred["d_auroc"] < -0.1 and blurred["d_auroc_ci"][1] < -0.05
    assert blurred["d_fpr"] == 0.0 and blurred["d_fpr_ci"] == [0.0, 0.0] and blurred["d_tpr"] < -0.3
    # Both at once: the false alarms are there, but so is a drop in accuracy that would be noticed.
    broken = _row(table, BROKEN)
    assert broken["fpr"] == shifted["fpr"] and broken["d_auroc"] == pytest.approx(blurred["d_auroc"])
    # An unchanged condition differs from clean by exactly nothing (the draws are shared).
    same = _row(table, "gamma-1")
    assert (same["d_auroc"], same["d_fpr"], same["d_tpr"]) == (0.0, 0.0, 0.0)
    assert same["d_auroc_ci"] == same["d_fpr_ci"] == same["d_tpr_ci"] == [0.0, 0.0]

    assert analyze_perturb.h9(table) == {"verdict": "지지", "conditions": [SHIFTED]}


# 확인: H9의 경계를 손으로 만든 행으로 짚는다. AUROC 하락이 1.0%p '미만'(정확히 −1.0%p는 해당 안 됨)이고
# 오검출률이 목표의 두 배 '이상'(정확히 10%는 해당)인 조건만 센다. clean 행은 비율이 커도 후보가 아니다. 두 배
# 기준은 α를 따라 바뀌고(α 10%면 20%, 1%면 2%), 보정 점수가 없는 행이 있으면 ValueError다. 또 기준은 clean
# 대비 증가분이 아니라 오검출률 자체다(clean 7%에서 12%면 해당, 9%면 해당 안 됨).
def test_h9_rule_on_hand_made_tables():
    def row(name, d_auroc, fpr):
        return {"condition": name, "d_auroc": d_auroc, "fpr": fpr}

    clean = {"condition": "clean", "fpr": 0.5}  # never a candidate, whatever its rate
    assert analyze_perturb.h9([clean]) == {"verdict": "기각", "conditions": []}
    table = [
        clean,
        row("a", -0.005, 0.20),  # accuracy holds, rate doubled: counts
        row("b", -0.02, 0.30),  # accuracy fell by 2 points: the drop would be noticed
        row("c", 0.0, 0.0999),  # rate just under twice the target
        row("d", 0.003, 0.10),  # exactly twice the target counts ("at least")
        row("e", -0.01, 0.50),  # exactly 1.0 point is not "less than 1.0 point"
        row("f", -0.0099, 0.15),
    ]
    assert analyze_perturb.h9(table) == {"verdict": "지지", "conditions": ["a", "d", "f"]}
    assert analyze_perturb.h9([clean, table[2], table[3], table[5]]) == {"verdict": "기각", "conditions": []}
    # The limit follows the target.
    assert analyze_perturb.h9(table, alpha=0.1)["conditions"] == ["a"]
    assert analyze_perturb.h9(table, alpha=0.01)["conditions"] == ["a", "c", "d", "f"]
    with pytest.raises(ValueError, match="fixed thresholds"):
        analyze_perturb.h9([clean, row("a", 0.0, None)])

    # The rate itself is compared with twice the target, not its increase over clean: rows as
    # `condition_table` makes them carry both.
    full = [
        {"condition": "clean", "fpr": 0.07},
        {"condition": "g", "d_auroc": 0.0, "fpr": 0.12, "d_fpr": 0.05},  # rate 12%: counts
        {"condition": "h", "d_auroc": 0.0, "fpr": 0.09, "d_fpr": 0.02},  # rate 9%: does not
        {"condition": "i", "d_auroc": 0.0, "fpr": 0.03, "d_fpr": -0.04},
    ]
    assert analyze_perturb.h9(full) == {"verdict": "지지", "conditions": ["g"]}


# 확인: 결함이 모든 정상보다 높아 AUROC가 늘 1.0인 실행에서, clean 오검출률 7%에 대해 12%와 정확히 10%인
# 조건은 증가분이 10%p가 안 돼도 H9 조건이고 9%는 아니다. 그런 조건이 없으면 clean 비율이 커도 '기각'이다.
# condition_table이 만든 실제 표에서 등록 문구('합산 오검출률은 10% 이상')를 그대로 따르는지 본다.
def test_h9_is_judged_on_the_rate_not_on_its_increase():
    # A clean rate of 7%. Every defect scores above every normal under every condition, so the AUROC
    # never moves and only the pooled false positive rate decides.
    run = _tiny(
        {"clean": _above(7), "to-12": _above(12), "to-9": _above(9), "to-10": _above(10)},
        [200.0] * 20,
        np.arange(100),
    )
    table = analyze_perturb.condition_table(run, n_boot=50)
    assert [row["macro_image_auroc"] for row in table] == [1.0] * 4
    assert [row["fpr"] for row in table] == [0.07, 0.12, 0.09, 0.10]
    assert [row["d_fpr"] for row in table[1:]] == pytest.approx([0.05, 0.02, 0.03])
    assert [row["d_auroc"] for row in table[1:]] == [0.0] * 3
    # 12% and exactly 10% are "at least twice the target" although neither rose by 10 points; 9% is not.
    assert analyze_perturb.h9(table) == {"verdict": "지지", "conditions": ["to-12", "to-10"]}

    # Without such a condition the hypothesis is rejected, however large the clean rate is.
    run = _tiny({"clean": _above(9), "to-9": _above(9), "to-8": _above(8)}, [200.0] * 20, np.arange(100))
    assert analyze_perturb.h9(analyze_perturb.condition_table(run, n_boot=50)) == {
        "verdict": "기각",
        "conditions": [],
    }


# 확인: 기본값이 등록값(부트스트랩 2,000번, seed 0, 목표 5%, AUROC 하락 기준 1.0%p, H9는 P0)이고, 인자를
# 생략한 결과가 등록값을 명시한 결과와 같으며 횟수나 seed를 바꾸면 달라진다.
def test_defaults_are_the_registered_ones():
    assert (analyze_perturb.N_BOOT, analyze_perturb.SEED, analyze_perturb.ALPHA) == (2000, 0, 0.05)
    assert analyze_perturb.MAX_AUROC_DROP == 0.01 and analyze_perturb.H9_METHOD == "p0"
    run = _run()
    table = analyze_perturb.condition_table(run)
    assert table == analyze_perturb.condition_table(run, 0.05, 2000, 0)
    assert table != analyze_perturb.condition_table(run, 0.05, 200, 0)
    assert table != analyze_perturb.condition_table(run, 0.05, 2000, 1)


# 확인: 구간을 반복문으로 직접 계산한 부트스트랩과 맞춘다. 추출은 생성기 하나로 '범주마다 정상, 그다음 결함'
# 순서의 묶음을 뽑고, 모든 조건에 같은 추출을 쓴다. 차이의 구간은 '두 구간의 차'가 아니라 '표본마다 차이를 낸
# 뒤의 구간'(짝지은 차이)이다. 같은 seed면 결과가 같고 다른 seed면 다르다.
# 왜: 공통 규칙은 범주 안 정상·이상을 따로 복원 추출하고, 두 조건을 비교할 때 같은 추출로 차이의 구간을 내도록
# 정했다. 짝짓지 않으면 같은 이미지의 흔들림이 차이 구간을 불필요하게 넓힌다.
def test_intervals_and_paired_differences_match_a_direct_bootstrap():
    run = _run(seed=3)
    n_boot, seed = 60, 5
    table = analyze_perturb.condition_table(run, n_boot=n_boot, seed=seed)

    # One generator, groups in order: normals then defects of each category.
    sizes = [n for pair in SIZES.values() for n in pair]
    draws = stratified_indices(sizes, n_boot=n_boot, seed=seed)
    thresholds = [conformal_threshold(cat["cal_score"], 0.05).value for cat in run["cats"].values()]

    def resampled(name):
        k = NAMES.index(name)
        macro, fpr, tpr = np.zeros(n_boot), np.zeros(n_boot), np.zeros(n_boot)
        for b in range(n_boot):
            aucs, fp, tp = [], 0, 0
            for i, cat in enumerate(run["cats"].values()):
                labels, scores = cat["eval_labels"], cat["scores"][k].astype(np.float64)
                neg = scores[labels == 0][draws[2 * i][b]]
                pos = scores[labels == 1][draws[2 * i + 1][b]]
                aucs.append(auroc(neg, pos))
                fp += int((neg > thresholds[i]).sum())
                tp += int((pos > thresholds[i]).sum())
            macro[b], fpr[b], tpr[b] = np.mean(aucs), fp / 250, tp / 64
        return macro, fpr, tpr

    base = resampled("clean")
    for key, samples in zip(("macro_image_auroc", "fpr", "tpr"), base, strict=True):
        assert table[0][f"{key}_ci"] == pytest.approx(percentile_ci(samples), abs=1e-12)
    for name in (SHIFTED, BLURRED, BROKEN):
        row = _row(table, name)
        other = resampled(name)
        for key, full, mine, clean in zip(
            ("auroc", "fpr", "tpr"), ("macro_image_auroc", "fpr", "tpr"), other, base, strict=True
        ):
            assert row[f"{full}_ci"] == pytest.approx(percentile_ci(mine), abs=1e-12)
            # Paired: the interval of the per-resample differences, not a difference of intervals.
            assert row[f"d_{key}_ci"] == pytest.approx(percentile_ci(mine - clean), abs=1e-12)
            assert row[f"d_{key}"] == pytest.approx(row[full] - table[0][full], abs=1e-12)

    # Deterministic for a seed; another seed gives other draws.
    assert analyze_perturb.condition_table(run, n_boot=n_boot, seed=seed) == table
    assert analyze_perturb.condition_table(run, n_boot=n_boot, seed=6) != table


# 확인: 보정 점수를 10만큼 올리면 모든 조건의 오검출률이 0이 되고 AUROC는 그대로다. 임계값은 원본 실행의 보정
# 점수로만 정하고 평가 점수나 교란 점수로 다시 잡지 않는다(3-A 규칙: 임계값을 그대로 쓰고 다시 보정하지
# 않는다).
def test_thresholds_are_fixed_from_the_calibration_scores_only():
    run = _run()
    table = analyze_perturb.condition_table(run, n_boot=20)
    # Moving the calibration scores moves every rate; the evaluation scores never feed a threshold.
    for cat in run["cats"].values():
        cat["cal_score"] = cat["cal_score"] + np.float32(10.0)
    moved = analyze_perturb.condition_table(run, n_boot=20)
    assert all(row["fpr"] == 0.0 for row in moved) and table[0]["fpr"] > 0.0
    assert [row["macro_image_auroc"] for row in moved] == [row["macro_image_auroc"] for row in table]


# 확인: 보정 점수가 없는 실행은 AUROC만 표에 싣고 오검출률·검출률과 그 차이를 None으로, 출력 표에서는 '-'로
# 둔다. H9는 고정 임계값이 필요하므로 ValueError이고, 보장된 범주 수도 None이다. 일부 범주에만 보정 점수가
# 없으면 '임계값 없는 실행'이 아니라 잘못된 실행으로 보고 그 범주 이름을 담은 ValueError를 낸다.
def test_run_without_calibration_scores_has_accuracy_only():
    run = _run(cal=0)
    table = analyze_perturb.condition_table(run, n_boot=50)
    for row in table:
        assert row["fpr"] is None and row["fpr_ci"] is None and row["tpr"] is None and row["tpr_ci"] is None
        assert 0.0 < row["macro_image_auroc"] <= 1.0
    shifted, blurred = _row(table, SHIFTED), _row(table, BLURRED)
    assert shifted["d_fpr"] is None and shifted["d_fpr_ci"] is None
    assert shifted["d_tpr"] is None and shifted["d_tpr_ci"] is None
    assert shifted["d_auroc"] == 0.0 and blurred["d_auroc"] < -0.1
    assert "d_fpr" not in table[0]
    with pytest.raises(ValueError, match="fixed thresholds"):
        analyze_perturb.h9(table)
    text = analyze_perturb.format_tables({"dm": table}, None)
    assert "| clean |" in text and "| - | - | - | - |" in text and "H9" not in text
    assert analyze_perturb.describe(run)["categories_guaranteed"] is None

    # Calibration scores for some categories only: a broken run, not a run without thresholds.
    partly = _run()
    partly["cats"]["beta"]["cal_score"] = np.empty(0, dtype=np.float32)
    with pytest.raises(ValueError, match="beta"):
        analyze_perturb.condition_table(partly, n_boot=20)


# 확인: 범주마다 조건 목록이 다르거나, clean 조건이 없거나, 점수 배열 모양이 '조건 × 이미지'와 맞지 않거나,
# 점수에 NaN이 있거나, 범주가 하나도 없으면 각각 알맞은 ValueError다. 잘못 만든 실행으로 표를 조용히 만들지
# 않는다.
def test_inconsistent_runs_are_rejected():
    run = _run()
    run["cats"]["beta"]["conditions"] = run["cats"]["beta"]["conditions"][::-1].copy()
    with pytest.raises(ValueError, match="other conditions"):
        analyze_perturb.condition_table(run, n_boot=20)

    run = _run()
    for cat in run["cats"].values():
        cat["conditions"] = cat["conditions"][1:]
        cat["scores"] = cat["scores"][1:]
    with pytest.raises(ValueError, match="clean"):
        analyze_perturb.condition_table(run, n_boot=20)

    run = _run()
    run["cats"]["alpha"]["scores"] = run["cats"]["alpha"]["scores"][:, :-1]
    with pytest.raises(ValueError, match="do not match"):
        analyze_perturb.condition_table(run, n_boot=20)

    run = _run()
    run["cats"]["gamma"]["scores"][3, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        analyze_perturb.condition_table(run, n_boot=20)

    with pytest.raises(ValueError, match="no categories"):
        analyze_perturb.condition_table({"meta": {}, "cats": {}}, n_boot=20)


# 확인: load가 run.json의 범주 순서대로 npz를 읽어 메모리의 실행과 같은 배열·같은 표를 만든다. describe는
# 이미지·보정 점수 수, 보장된 범주 수, clean_max_rel_diff의 최댓값, 원본 실행의 commit을 요약한다(리포트에
# '표가 무엇을 바탕으로 했는지'를 남긴다).
def test_load_reads_a_run_directory_in_run_order(tmp_path):
    run = _run()
    _write(run, tmp_path / "run")
    loaded = analyze_perturb.load(tmp_path / "run")
    assert loaded["meta"] == run["meta"] and list(loaded["cats"]) == list(SIZES)
    for name, cat in run["cats"].items():
        assert set(loaded["cats"][name]) == set(cat)
        for key, value in cat.items():
            assert np.array_equal(loaded["cats"][name][key], value)
    assert analyze_perturb.condition_table(loaded, n_boot=30) == analyze_perturb.condition_table(
        run, n_boot=30
    )
    info = analyze_perturb.describe(loaded)
    assert (info["n_normal"], info["n_defect"], info["n_cal"]) == (250, 64, 300)
    assert info["categories"] == list(SIZES) and info["categories_guaranteed"] == 3
    assert info["clean_max_rel_diff"] == pytest.approx(3e-5) and info["source_commit"] == "src1234"


# 확인: 출력 표가 방법마다 하나씩 나오고, clean 행의 변화 칸 세 개는 '-'이며, 변화는 부호가 붙은 %p로 적히고,
# 마지막 줄이 H9 판정(조건 이름 또는 none)이다.
def test_format_tables():
    tables = {m: analyze_perturb.condition_table(_run(method=m), n_boot=50) for m in ("p0", "d-s")}
    verdict = {"method": "p0", **analyze_perturb.h9(tables["p0"])}
    text = analyze_perturb.format_tables(tables, verdict)
    lines = text.splitlines()
    assert lines[0] == "method p0" and "method d-s" in lines
    assert sum(line.startswith("| clean |") for line in lines) == 2
    assert sum(line.startswith(f"| {SHIFTED} |") for line in lines) == 2
    clean = next(line for line in lines if line.startswith("| clean |"))
    assert clean.count(" - ") == 3  # the clean row has nothing to be compared with
    shifted = next(line for line in lines if line.startswith(f"| {SHIFTED} |"))
    assert "+0.00 [+0.00, +0.00]" in shifted  # the AUROC change, signed and in points
    assert lines[-1] == f"H9 (p0): 지지 (conditions: {SHIFTED})"
    rejected = analyze_perturb.format_tables(tables, {"verdict": "기각", "conditions": []})
    assert rejected.splitlines()[-1] == "H9: 기각 (conditions: none)"


# CLI 테스트용 fixture. outputs·reports 경로를 tmp_path로 돌리고 부트스트랩을 100번으로 줄이며, 합성 범주 세
# 개를 '전체 범주'(원래는 VisA 12개)로 바꿔 H9 판정 경로를 탈 수 있게 한다.
@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_perturb.paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(analyze_perturb.paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(analyze_perturb, "N_BOOT", 100)
    # The synthetic categories stand for the full set that conclusions are drawn over.
    monkeypatch.setattr(analyze_perturb, "CATEGORIES", tuple(SIZES))
    return tmp_path


# 일부 범주만 다시 돌린 실행 흉내: run.json에는 그 범주만 남고, 다른 범주의 npz 파일은 디스크에 그대로 남아
# 있는 상황이다.
def _only(run, categories):
    """The run as `run_perturb --categories ... --overwrite` leaves it: run.json lists only these."""
    meta = dict(run["meta"])
    meta["categories"] = [info for info in meta["categories"] if info["category"] in categories]
    return {"meta": meta, "cats": run["cats"]}


# 확인: CLI가 세 방법(P0, D-S, DM)의 표를 모두 출력·저장하고 H9는 P0로만 판정한다. 리포트는 줄바꿈이 LF이고,
# 프로토콜·α·부트스트랩 횟수·방법별 이미지 수·commit을 담으며, 보정 점수가 없는 DM은 오검출률이 None이다.
# 저장된 P0 표는 파일을 읽어 직접 만든 표와 같고, 표가 어떤 범주와 이미지 수를 바탕으로 했는지 함께 출력한다.
def test_cli_writes_every_table_and_judges_h9_on_p0(project, capsys):
    outputs = project / "outputs"
    _write(_run(method="p0"), outputs / "perturb-p0-dev")
    _write(_run(seed=1, method="d-s"), outputs / "perturb-d-s-dev")
    _write(_run(seed=2, cal=0, method="dm"), outputs / "perturb-dm-dev")
    analyze_perturb.main(["--protocol", "dev"])
    text = capsys.readouterr().out
    assert "method p0" in text and "method d-s" in text and "method dm" in text
    assert f"H9 (p0): 지지 (conditions: {SHIFTED})" in text

    path = project / "reports" / "stage3" / "perturb-dev.json"
    assert b"\r" not in path.read_bytes()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["protocol"] == "dev" and report["alpha"] == 0.05
    assert (report["n_boot"], report["seed"]) == (100, 0)
    assert list(report["tables"]) == ["p0", "d-s", "dm"] == list(report["runs"])
    assert all(len(table) == 16 for table in report["tables"].values())
    assert report["h9"] == {"method": "p0", "verdict": "지지", "conditions": [SHIFTED]}
    assert report["runs"]["p0"]["n_cal"] == 300 and report["runs"]["dm"]["n_cal"] == 0
    assert report["runs"]["d-s"]["protocol"] == "dev" and report["runs"]["d-s"]["commit"] == "abc1234"
    assert report["tables"]["dm"][1]["fpr"] is None
    expected = analyze_perturb.condition_table(analyze_perturb.load(outputs / "perturb-p0-dev"))
    assert report["tables"]["p0"] == json.loads(json.dumps(expected))
    # What the tables stand on is said next to them.
    assert report["categories"] == list(SIZES) and report["complete"] is True
    assert "categories (3 of 3): alpha, beta, gamma" in text and "partial" not in text
    assert "p0: 250 normal and 64 defect images, 300 calibration scores" in text


# 확인: --methods 순서가 D-S, P0여도 H9는 P0 표로만 판정한다. D-S 표만 보면 지지인데 P0(임계값이 모든 점수
# 위)는 기각이므로, 첫 번째 방법을 쓰는 실수를 잡는다(H9의 판정 대상은 P0로 등록됐다).
def test_cli_judges_h9_on_p0_whatever_the_order_of_the_methods(project, capsys):
    outputs = project / "outputs"
    # p0 with thresholds far above every score: no false positive under any condition, so H9 is
    # rejected for it. The d-s run alone would support it.
    p0 = _run(method="p0")
    for cat in p0["cats"].values():
        cat["cal_score"] = cat["cal_score"] + np.float32(10.0)
    _write(p0, outputs / "perturb-p0-dev")
    _write(_run(seed=1, method="d-s"), outputs / "perturb-d-s-dev")
    analyze_perturb.main(["--protocol", "dev", "--methods", "d-s", "p0"])
    text = capsys.readouterr().out
    report = json.loads((project / "reports" / "stage3" / "perturb-dev.json").read_text(encoding="utf-8"))
    assert list(report["tables"]) == ["d-s", "p0"]
    assert analyze_perturb.h9(report["tables"]["d-s"])["conditions"] == [SHIFTED]
    assert report["h9"] == {"method": "p0", "verdict": "기각", "conditions": []}
    assert text.splitlines()[-1] == "H9 (p0): 기각 (conditions: none)"


# 확인: test 절차에서는 run.json이 일부 범주만 담은 실행(다른 범주 npz가 디스크에 남아 있어도)이나 범주 순서가
# 등록 순서와 다른 실행을 거부하고 리포트를 쓰지 않는다.
# 왜: 봉인 테스트 결과는 12개 범주 전체에 대해 말하고, 부트스트랩은 등록한 범주 순서로 묶음을 뽑으므로 순서가
# 바뀌면 같은 추출이 아니게 된다(analyze_perturb의 CATEGORIES 주석).
def test_cli_refuses_a_test_run_that_covers_part_of_the_categories(project, capsys):
    outputs = project / "outputs"
    # A finished three-category run redone for one category: the other npz files are still on disk,
    # run.json lists `alpha` only.
    _write(_only(_run(method="p0", protocol="test"), ["alpha"]), outputs / "perturb-p0-test")
    assert (outputs / "perturb-p0-test" / "gamma.npz").exists()
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "test", "--methods", "p0"])
    err = capsys.readouterr()
    assert "['alpha']" in err.err and "beta" in err.err and "H9" not in err.out
    assert not (project / "reports").exists()

    # The same set but in another order is not the registered resampling either.
    run = _run(method="d-s", protocol="test")
    run["meta"]["categories"] = run["meta"]["categories"][::-1]
    _write(run, outputs / "perturb-d-s-test")
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "test", "--methods", "d-s"])
    assert "gamma" in capsys.readouterr().err and not (project / "reports").exists()


# 확인: dev 절차에서는 일부 범주만 돈 실행(코드 점검)도 표를 만들되, 'partial run'이라고 적고 H9를 판정하지
# 않으며 지지·기각 문구를 출력하지 않는다. 표는 그 두 범주만으로 직접 만든 표와 같다.
def test_cli_dev_run_on_part_of_the_categories_is_marked_and_not_judged(project, capsys):
    outputs = project / "outputs"
    _write(_only(_run(method="p0"), ["alpha", "beta"]), outputs / "perturb-p0-dev")
    _write(_only(_run(seed=1, method="d-s"), ["alpha", "beta"]), outputs / "perturb-d-s-dev")
    analyze_perturb.main(["--protocol", "dev", "--methods", "p0", "d-s"])
    text = capsys.readouterr().out
    # The code check may run on a few categories, but it says so and gives no verdict.
    assert "categories (2 of 3): alpha, beta" in text
    assert "partial run: H9 is not judged" in text and "H9 (p0)" not in text
    assert "지지" not in text and "기각" not in text
    report = json.loads((project / "reports" / "stage3" / "perturb-dev.json").read_text(encoding="utf-8"))
    assert report["categories"] == ["alpha", "beta"] and report["complete"] is False
    assert report["h9"] is None
    assert report["runs"]["p0"]["categories"] == ["alpha", "beta"] and report["runs"]["p0"]["n_normal"] == 130
    # The table of the two categories is still what it says it is.
    direct = _run(method="p0")
    del direct["cats"]["gamma"]
    assert report["tables"]["p0"] == json.loads(json.dumps(analyze_perturb.condition_table(direct)))


# 확인: 방법마다 범주 목록이 다르면 표를 나란히 둘 수 없으므로 build_report와 CLI가 'different categories'
# ValueError를 내고 아무것도 출력·저장하지 않는다. 각 방법을 따로 넣으면 그 범주만큼의 표가 나오고, 범주가
# 모자란 쪽은 complete False·판정 없음이다.
def test_methods_must_cover_the_same_categories(project, capsys):
    outputs = project / "outputs"
    _write(_only(_run(method="p0"), ["alpha"]), outputs / "perturb-p0-dev")
    _write(_run(seed=1, method="d-s"), outputs / "perturb-d-s-dev")
    dirs = {m: outputs / f"perturb-{m}-dev" for m in ("p0", "d-s")}
    with pytest.raises(ValueError, match="different categories"):
        analyze_perturb.build_report(dirs, n_boot=20)
    with pytest.raises(ValueError, match="different categories"):
        analyze_perturb.main(["--protocol", "dev", "--methods", "p0", "d-s"])
    assert capsys.readouterr().out == "" and not (project / "reports").exists()
    # Each on its own is a table of what it covers.
    assert analyze_perturb.build_report({"d-s": dirs["d-s"]}, n_boot=20)["complete"] is True
    alone = analyze_perturb.build_report({"p0": dirs["p0"]}, n_boot=20)
    assert alone["complete"] is False and alone["h9"] is None and len(alone["tables"]["p0"]) == 16


# 확인: '전체 범주'는 VisA의 12개 범주(visa.CATEGORIES)다. 합성 범주 세 개로는 판정하지 않고, 같은 데이터를
# 12개 실제 이름으로 붙이면 complete True가 돼 H9가 판정된다.
def test_the_full_set_is_the_twelve_visa_categories(tmp_path):
    from defect_inspect.visa import CATEGORIES

    assert analyze_perturb.CATEGORIES == CATEGORIES and len(CATEGORIES) == 12
    # Three made-up categories are not the set H9 is registered on.
    _write(_run(method="p0"), tmp_path / "three")
    report = analyze_perturb.build_report({"p0": tmp_path / "three"}, n_boot=20)
    assert report["complete"] is False and report["h9"] is None

    # The same data under the twelve real names (each synthetic category four times) is.
    three = _run(method="p0")
    twelve = {"meta": dict(three["meta"]), "cats": {}}
    for i, category in enumerate(CATEGORIES):
        twelve["cats"][category] = three["cats"][list(SIZES)[i % 3]]
    twelve["meta"]["categories"] = [{"category": c} for c in CATEGORIES]
    _write(twelve, tmp_path / "twelve")
    report = analyze_perturb.build_report({"p0": tmp_path / "twelve"}, n_boot=20)
    assert report["complete"] is True and report["categories"] == list(CATEGORIES)
    assert report["h9"] == {"method": "p0", "verdict": "지지", "conditions": [SHIFTED]}


# 확인: P0 없이 D-S만 넣으면 H9를 판정하지 않고, --no-write면 출력만 하고 리포트 파일을 만들지 않는다.
def test_cli_without_p0_gives_no_verdict_and_no_write_prints_only(project, capsys):
    outputs = project / "outputs"
    _write(_run(method="d-s", protocol="test"), outputs / "perturb-d-s-test")
    analyze_perturb.main(["--protocol", "test", "--methods", "d-s", "--no-write"])
    text = capsys.readouterr().out
    assert "method d-s" in text and "H9" not in text
    assert not (project / "reports").exists()


# 확인: 아직 돌리지 않은 방법의 실행 폴더가 없으면 그 경로를 알려 주며 멈추고, --methods가 비었거나 모르는
# 방법이거나 --protocol이 없으면 거부한다. 폴더 이름과 run.json의 방법·프로토콜이 다르면(perturb-p0-test 안에
# D-S 실행, test 폴더 안에 dev 실행) 거부하고 리포트를 쓰지 않는다. 다른 실행의 점수를 엉뚱한 이름으로
# 보고하는 실수를 막는다.
def test_cli_rejects_missing_and_mismatched_runs(project, capsys):
    outputs = project / "outputs"
    _write(_run(method="p0"), outputs / "perturb-p0-dev")
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "dev"])  # d-s and dm have not been run
    assert "perturb-d-s-dev" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "dev", "--methods"])
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "dev", "--methods", "p1"])
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--methods", "p0"])  # the protocol must be named

    # A run directory that holds another method or protocol than its name says.
    _write(_run(method="d-s"), outputs / "perturb-p0-test")
    with pytest.raises(ValueError, match="not of 'p0'"):
        analyze_perturb.main(["--protocol", "test", "--methods", "p0"])
    _write(_run(method="d-s", protocol="dev"), outputs / "perturb-d-s-test")
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "test", "--methods", "d-s"])
    assert not (project / "reports").exists()

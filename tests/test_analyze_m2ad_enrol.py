"""Stage 7 analysis on hand-made arrays (no torch, no data)."""

# [파일 역할] 7단계 분석 모듈(`defect_inspect.analyze_m2ad_enrol`)을 검증하는 테스트 묶음이다. M2AD 학습
# 시편만 쓰는 검증에서 후보(E0 그대로, E1 이미지별 중심화, E2 조명 여럿 등록, E3 둘 다) 고르기, 봉인 test의
# H18·H19 판정과 검출률 가드, 경보 뒤 자동 재등록(폐루프) 표, 3-B 실행과의 일치 확인(stage3b_match), VisA
# dev의 E1 − E0 표를 다룬다.
# [왜 필요한가] 선택 지표(못 본 조명의 합산 오검출률, 동점이면 단순한 쪽), H18(구간 상한이 −50%p보다 작으면
# 지지, 하한이 −50%p보다 크면 기각), 가드(I01 검출률 하락 10%p 이내일 때만 '개선'), H19(등록 조명 오검출률 6%
# 이하)는 docs/experiments.md 7단계에 측정 전에 등록됐다. 구간은 시편 단위로 다시 뽑는 부트스트랩이고, I01은
# 두 팔 모두에서 등록 조명으로 센다. 이런 집계 규칙 하나가 틀려도 README의 7단계 수치가 달라진다.
# [설계] torch도 데이터도 없이(모듈 docstring) 손으로 만든 배열로 실행 결과를 흉내 낸다. 점수를 임계값 1보다
# 크거나(2.0, 3.0) 작게(0.5) 넣어 오검출·검출 수를 미리 알 수 있게 한다.

import json

import numpy as np
import pytest

from defect_inspect import analyze_m2ad_enrol as an
from defect_inspect.m2ad import ILLUMINATIONS
from defect_inspect.run_m2ad_enrol import LOOP_ARMS, enrolled_matrix, loop_keys

# 검증용 학습 시편 수(실제는 범주당 30개). 겹은 i % 5다.
N_SPEC = 10


# build_test_report는 paths.OUTPUTS/m2ad-<방법>에서 3-B 실행을 찾아 stage3b_match를 계산한다. 모든 테스트에서
# OUTPUTS를 tmp_path로 돌려, 로컬 outputs/에 있는 실제 3-B 실행 결과가 테스트에 섞이지 않게 한다(autouse).
@pytest.fixture(autouse=True)
def no_real_outputs(tmp_path, monkeypatch):
    """The stage 3-B comparison must not find the real runs of this machine."""
    monkeypatch.setattr(an.paths, "OUTPUTS", tmp_path / "outputs")


# 검사기는 (범주, 시점)마다 하나다. 같은 범주(Motor)의 두 시점이 같은 시편을 공유하는 경우를 넣어,
# 부트스트랩이 범주 단위로 시편을 뽑는지 본다.
INSPECTORS = ["Motor_000", "Motor_120", "Bird_000"]


# run_m2ad_enrol val이 저장하는 배열 모양을 흉내 낸다. 팔(E0·E1은 ref 하나, E2·E3는 A·B 둘)마다 열 조명 × 시편
# 점수를 두고, 등록한 조명은 모두 0.5(임계값 1 아래), 못 본 조명은 flag_unseen 비율만큼 2.0(임계값 위)으로
# 만든다. 그래서 못 본 조명의 오검출률이 대략 flag_unseen이 된다.
def _val_arrays(candidate: str, flag_unseen: float, seed: int = 0) -> dict:
    """Validation arrays where a share `flag_unseen` of unseen images (and none of the seen) exceed."""
    rng = np.random.default_rng(seed)
    enrolled = enrolled_matrix(candidate)
    n_arms = len(enrolled)
    folds = np.arange(N_SPEC) % 5
    thresholds = np.ones((n_arms, 5))
    scores = np.full((n_arms, 10, N_SPEC), 0.5, dtype=np.float32)
    for a in range(n_arms):
        for li in range(10):
            if not enrolled[a, li]:
                scores[a, li] = np.where(rng.random(N_SPEC) < flag_unseen, 2.0, 0.5)
    return {
        "specimens": np.array([f"s{i:02d}" for i in range(N_SPEC)]),
        "folds": folds,
        "lights": np.array(ILLUMINATIONS),
        "arms": np.array(["ref"] if n_arms == 1 else ["A", "B"]),
        "enrolled": enrolled,
        "scores": scores,
        "thresholds": thresholds,
        "cal_n": np.full((n_arms, 5), 8),
    }


# 후보 이름 → 못 본 조명 오검출 비율을 받아 load_val이 돌려주는 모양({meta, cands})의 검증 실행을 만든다.
def _val_run(shares: dict[str, float]) -> dict:
    return {
        "meta": {"commit": "abc", "device": "cpu", "groups": None, "inspectors": INSPECTORS},
        "cands": {
            c: {n: _val_arrays(c, share, seed=i) for i, n in enumerate(INSPECTORS)}
            for c, share in shares.items()
        },
    }


# 확인: 모든 후보가 같은 못 본 조명 이미지 수(9조명 × 시편 × 검사기)로 채점되고, E2는 팔마다 못 본 묶음을 합쳐
# 같은 수가 된다. 등록 조명 수는 E0가 I01뿐, E2는 팔 A 5조명 + 팔 B 6조명(I01은 팔마다 한 번씩)이다.
# 오검출률이 낮은 E2가 뽑히고, 조명별 표에는 I01을 뺀 아홉 조명이 나온다.
# 왜: 규칙은 '후보마다 같은 1,620장'으로 비교하도록 정했다(7단계 검증). 후보마다 분모가 다르면 선택 지표를
# 비교할 수 없다.
def test_validation_counts_unseen_images_of_every_arm():
    run = _val_run({"e0": 1.0, "e2": 0.0})
    report = an.build_val_report({"p0": run}, n_boot=50)
    rows = report["methods"]["p0"]["candidates"]
    # Every candidate has the same unseen images: 9 lights x specimens x inspectors.
    assert rows["e0"]["unseen"]["n_normal"] == rows["e2"]["unseen"]["n_normal"] == 9 * N_SPEC * 3
    assert rows["e0"]["unseen"]["fpr"] == 1.0 and rows["e2"]["unseen"]["fpr"] == 0.0
    assert rows["e0"]["seen"]["n_normal"] == N_SPEC * 3
    assert rows["e2"]["seen"]["n_normal"] == (5 + 6) * N_SPEC * 3  # I01 counts once in each arm
    assert report["methods"]["p0"]["pick"] == "e2"
    assert set(rows["e2"]["unseen_by_light"]) == set(ILLUMINATIONS[1:])


# 확인: 오검출 수가 같으면 단순한 쪽(E0 < E1 < E2 < E3)을 고르고, 더 낮은 후보가 있으면 복잡해도 그것을
# 고른다. 비율은 Fraction으로 정확히 비교해 부동소수 반올림이 동점을 깨지 않는다.
# 왜: 실제 검증에서 P0의 E2와 E3가 865장으로 동점이라 단순한 E2가 뽑혔다(docs/experiments.md 7단계 검증 결과).
def test_pick_ties_go_to_the_simpler_candidate():
    rows = {
        c: {"unseen": {"false_positives": fp, "n_normal": 100}} for c, fp in (("e3", 5), ("e1", 5), ("e2", 6))
    }
    assert an.choose(rows) == "e1"
    rows["e0"] = {"unseen": {"false_positives": 5, "n_normal": 100}}
    assert an.choose(rows) == "e0"
    rows["e3"]["unseen"]["false_positives"] = 4
    assert an.choose(rows) == "e3"


# 확인: 참고 구간은 범주 안 시편을 다시 뽑는 부트스트랩이라 점 추정 0.5를 감싸는 0~1 사이 구간이 나온다. 같은
# 범주의 다른 시점이 시편 순서가 다르면 ValueError다.
# 왜: 한 시편의 여러 조명·시점 이미지는 서로 독립이 아니므로 시편을 단위로 뽑아야 구간이 지나치게 좁아지지
# 않는다. 시점마다 시편 순서가 다르면 같은 추출 가중치를 쓸 수 없다.
def test_validation_interval_resamples_specimens_per_category():
    run = _val_run({"e0": 0.5})
    report = an.build_val_report({"p0": run}, n_boot=200)
    lo, hi = report["methods"]["p0"]["candidates"]["e0"]["unseen"]["fpr_ci"]
    assert 0.0 < lo < 0.5 < hi < 1.0
    bad = _val_run({"e0": 0.5})
    bad["cands"]["e0"]["Motor_120"]["specimens"] = bad["cands"]["e0"]["Motor_120"]["specimens"][::-1].copy()
    with pytest.raises(ValueError):
        an.build_val_report({"p0": bad}, n_boot=10)


# ---------------------------------------------------------------- test


# 봉인 test 흉내: 검사기마다 정상 시편 4개, 불량 시편 3개(그 가운데 하나는 모든 조명에서 결함이 보이지 않아
# 라벨 −1).
N_NORMAL, N_DEFECT = 4, 3


# run_m2ad_enrol test가 저장하는 배열 흉내. 정상 점수는 등록 조명에서 0.5, 못 본 조명에서
# unseen_normal_score이고, 결함 점수는 I01(조명 0번)에서만 ref_defect_score, 나머지 조명에서 3.0이다. 라벨
# −1인 결함은 정상·결함 어느 쪽에도 세지 않는다.
def _test_arrays(candidate: str, unseen_normal_score: float, ref_defect_score: float = 3.0) -> dict:
    enrolled = enrolled_matrix(candidate)
    n_arms, n = len(enrolled), N_NORMAL + N_DEFECT
    labels = np.array([[0] * N_NORMAL + [1] * N_DEFECT] * 10, dtype=np.int8)
    labels[:, -1] = -1  # one defect never visible
    scores = np.zeros((n_arms, 10, n), dtype=np.float32)
    for a in range(n_arms):
        for li in range(10):
            normal = unseen_normal_score if not enrolled[a, li] else 0.5
            scores[a, li, :N_NORMAL] = normal
            scores[a, li, N_NORMAL:] = ref_defect_score if li == 0 else 3.0
    return {
        "specimens": np.array([f"n{i}" for i in range(N_NORMAL)] + [f"d{i}" for i in range(N_DEFECT)]),
        "object_anomaly": np.array([0] * N_NORMAL + [1] * N_DEFECT, dtype=np.int8),
        "lights": np.array(ILLUMINATIONS),
        "arms": np.array(["ref"] if n_arms == 1 else ["A", "B"]),
        "enrolled": enrolled,
        "scores": scores,
        "labels": labels,
        "thresholds": np.ones(n_arms),
    }


# E0(못 본 조명 정상이 모두 2.0, 즉 전부 오검출)와 검증에서 고른 후보 하나로 test 실행을 만든다. 실제 test
# 실행도 E0와 선택 후보를 함께 잰다(7단계 봉인 test). loop=True면 폐루프 팔(LOOP_ARMS)의 점수와 거르기
# 기록(섞인 결함 수, 버린 결함·정상 수)을 meta에 같이 넣는다.
def _test_run(pick: str, pick_unseen: float, pick_ref_defect: float = 3.0, loop: bool = False) -> dict:
    cands = {"e0": {n: _test_arrays("e0", 2.0) for n in INSPECTORS}}
    if pick != "e0":
        cands[pick] = {n: _test_arrays(pick, pick_unseen, pick_ref_defect) for n in INSPECTORS}
    meta = {"commit": "abc", "device": "cpu", "candidates": list(cands), "inspectors": INSPECTORS}
    run = {"meta": meta, "cands": cands, "loop": None}
    if loop:
        keys = loop_keys()
        labels = np.array([[0] * N_NORMAL + [1] * N_DEFECT] * len(keys), dtype=np.int8)
        scores = np.tile(np.array([0.5] * N_NORMAL + [3.0] * N_DEFECT, dtype=np.float32), (len(keys), 1))
        run["loop"] = {
            n: {
                "loop_keys": np.array(keys),
                "scores": scores,
                "labels": labels,
                "thresholds": np.ones(len(keys)),
            }
            for n in INSPECTORS
        }
        info = {
            key: {
                "defects_in_batch": int(key.split(":")[2]),
                "dropped_defects": int(key.split(":")[2]),
                "dropped_normals": 4 - int(key.split(":")[2]) if "trim" in key else 0,
            }
            for key in keys
        }
        meta["loop_runs"] = [
            {"category": n.split("_")[0], "view": n.split("_")[1], "loop": info} for n in INSPECTORS
        ]
    return run


# 확인: 선택 후보가 못 본 조명 오검출률을 100%에서 0%로 낮추면(차이 −100%p) H18 '지지', I01 검출률이 그대로라
# 가드 통과·'개선', 등록 조명 오검출률 0%라 H19 '지지'다. 판정은 P0에만 하고 D-S 결과에는 H18·H19·improved
# 키가 없다. 분모도 본다: 못 본 조명 정상 9 × 4 × 3, 결함은 보이는 2장만, 등록 조명 11 × 4 × 3, I01 결함은 두
# 팔에서 한 번씩. 마지막으로 strict_json을 거친 리포트가 JSON으로 저장되는지 확인한다.
# 왜: 'H18·H19는 P0만 판정한다'(7단계 브리프에서 바꾼 것 3)와 'E2·E3의 I01 검출률은 두 팔을 합산한다'가
# 규칙이다.
def test_h18_supported_when_the_pick_removes_the_unseen_false_alarms():
    report = an.build_test_report({"p0": _test_run("e2", 0.5), "d-s": _test_run("e2", 0.5)}, n_boot=100)
    v = report["methods"]["p0"]["verdicts"]
    assert v["pick"] == "e2" and v["d_unseen_fpr"] == -1.0 and v["H18"] == "지지"
    assert v["guard_ok"] and v["improved"] and v["H19"] == "지지" and v["seen_fpr"] == 0.0
    assert report["hypotheses"] == {"H18": "지지", "H19": "지지"}
    # The second method is reported without verdicts.
    assert (
        "H18" not in report["methods"]["d-s"]["verdicts"]
        and "improved" not in report["methods"]["d-s"]["verdicts"]
    )
    cand = report["methods"]["p0"]["candidates"]["e2"]
    assert cand["unseen"]["n_normal"] == 9 * N_NORMAL * 3 and cand["unseen"]["n_defect"] == 9 * 2 * 3
    assert cand["seen"]["n_normal"] == 11 * N_NORMAL * 3
    assert cand["reference"]["n_defect"] == 2 * 2 * 3  # I01 in both arms
    assert report["methods"]["p0"]["stage3b_match"] is None
    json.dumps(an.strict_json(report))


# 확인: 세 가지 판정 분기.
# - 선택 후보(E3)가 I01 결함을 모두 놓치면 오검출이 사라져 H18은 '지지'여도 I01 검출률이 −100%p라 가드를 넘어
#   '개선'이 아니다(규칙: '오검출은 줄었지만 결함도 그만큼 놓친다'로 적는다).
# - 선택 후보(E1)의 못 본 조명 정상이 E0처럼 모두 임계값을 넘으면 차이가 0이라 구간 하한이 −50%p보다 커서
#   '기각'이다.
# - 검증에서 E0 자신이 뽑히면 비교할 것이 없어 H18은 '기각'이고 H19는 E0의 등록 조명(I01) 오검출률로 판정한다.
def test_the_detection_guard_and_the_other_verdicts():
    # The pick misses every defect under I01: the false alarms go, but it is no improvement.
    report = an.build_test_report({"p0": _test_run("e3", 0.5, pick_ref_defect=0.5)}, n_boot=50)
    v = report["methods"]["p0"]["verdicts"]
    assert v["H18"] == "지지" and v["d_reference_tpr"] == -1.0 and not v["guard_ok"] and not v["improved"]
    # Every unseen normal is still flagged, as under E0: the drop is 0, above -50 points, so H18 is rejected.
    report = an.build_test_report({"p0": _test_run("e1", 2.0)}, n_boot=50)
    assert report["methods"]["p0"]["verdicts"]["H18"] == "기각"
    # The pick was e0 itself: nothing to compare, rejected.
    report = an.build_test_report({"p0": _test_run("e0", 0.0)}, n_boot=50)
    v = report["methods"]["p0"]["verdicts"]
    assert v["pick"] == "e0" and v["H18"] == "기각" and v["H19"] == "지지"


# 확인: 폐루프 표가 LOOP_ARMS 순서(거르기 k = 0, 1, 2와 거르지 않음 k = 2)로 한 줄씩 나오고, 아홉 새 조명 ×
# 검사기를 합산한다(정상 9 × 4 × 3장, 섞인 결함 9 × 3 × k장). 규칙의 지표는 조명 L의 test 정상 오검출률과
# 검출률, 섞인 결함 가운데 거르기가 버린 수다(7단계 폐루프).
def test_loop_table_pools_the_nine_illuminations():
    report = an.build_test_report({"p0": _test_run("e2", 0.5, loop=True)}, n_boot=20)
    rows = report["methods"]["p0"]["loop"]
    assert [(r["filter"], r["k"]) for r in rows] == [tuple(arm) for arm in LOOP_ARMS]
    for row in rows:
        assert row["n_normal"] == 9 * N_NORMAL * 3 and row["fpr"] == 0.0 and row["tpr"] == 1.0
        assert row["defects_in_batches"] == 9 * 3 * row["k"]


# 확인: E0 점수·임계값을 3-B 실행 파일과 비교해 최대 점수 차이와 임계값 일치를 낸다. 같은 값을 써 두면 차이
# 0·일치이고, 3-B 실행이 없으면 None, 3-B 파일의 시편 목록이 다르면 'other test specimens' ValueError다.
# 왜: E0는 3-B와 같은 절차라 같은 장치에서 점수와 임계값이 같아야 한다(규칙: 차이를 stage3b_match로 적는다).
# 실제 결과는 두 방법 모두 최대 차이 0이었다.
def test_stage3b_match(tmp_path):
    run = _test_run("e0", 0.0)
    ref = tmp_path / "m2ad-p0"
    ref.mkdir()
    (ref / "run.json").write_text("{}", encoding="utf-8")
    conditions = ["S", *(f"R:{light}" for light in ILLUMINATIONS[1:])]
    for name, arr in run["cands"]["e0"].items():
        np.savez(
            ref / f"{name}.npz",
            conditions=np.array(conditions),
            scores=arr["scores"][0],
            threshold=np.float64(1.0),
        )
    assert an.stage3b_match(run, ref) == {"max_abs_score_diff": 0.0, "thresholds_equal": True}
    assert an.stage3b_match(run, tmp_path / "missing") is None
    np.savez(ref / "Bird_000.npz", specimens=np.array(["x"]), conditions=np.array(conditions))
    with pytest.raises(ValueError, match="other test specimens"):
        an.stage3b_match(run, ref)


# ---------------------------------------------------------------- VisA dev


# run_perturb 실행 흉내(범주 두 개, 정상 30·결함 10). 조건은 VISA_CONDITIONS(교란 없음, 밝기 0.7, 감마 1.5)
# 순서이고, 두 번째 조건은 모든 점수에 shift를 더해 '순서는 그대로인 채 점수 전체가 밀린' 상황을 만든다.
def _perturb_run(shift: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cats = {}
    for c in ("candle", "pcb1"):
        labels = np.array([0] * 30 + [1] * 10, dtype=np.int8)
        clean = np.concatenate([rng.normal(0, 1, 30), rng.normal(2, 1, 10)])
        scores = np.stack([clean, clean + shift, clean + 0.5]).astype(np.float32)
        cats[c] = {
            "eval_images": np.array([f"{c}/{i}.JPG" for i in range(40)]),
            "eval_labels": labels,
            "conditions": np.array(an.VISA_CONDITIONS),
            "scores": scores,
            "cal_score": rng.normal(0, 1, 60).astype(np.float32),
        }
    return {"meta": {"protocol": "dev", "commit": "x"}, "cats": cats}


# 확인: E0(밝기 조건에서 점수가 1.0 밀림)와 E1(밀리지 않음)을 같은 이미지로 짝지어 비교한다. 교란 없는 조건은
# 점수가 같아 차이와 구간이 정확히 0이고, 밝기 조건에서 AUROC는 그대로인데 E0의 오검출률만 올라 E1 − E0
# 오검출률 차이와 구간 상한이 음수다. 두 실행의 평가 이미지 순서가 다르면 ValueError다.
# 왜: E1(이미지별 중심화)이 겨냥한 현상이 3-A의 '밝기 0.7에서 AUROC는 그대로인데 오검출률 41.4%'였다(7단계
# 근거). 이 표가 그 효과를 짝지은 구간으로 보여 준다.
def test_visa_dev_table_pairs_the_two_runs():
    e0, e1 = _perturb_run(shift=1.0), _perturb_run(shift=0.0)
    table = an.visa_dev_table(e0, e1, n_boot=100)
    assert [row["condition"] for row in table] == list(an.VISA_CONDITIONS)
    clean, bright = table[0], table[1]
    # Same clean scores: no difference at all, and a uniform shift leaves the AUROC alone.
    assert clean["d_auroc"] == 0.0 and clean["d_fpr"] == 0.0 and clean["d_auroc_ci"] == [0.0, 0.0]
    assert bright["e0_auroc"] == bright["e1_auroc"] and bright["e0_d_auroc_vs_clean"] == 0.0
    assert bright["e0_fpr"] > bright["e1_fpr"] and bright["d_fpr"] < 0 and bright["d_fpr_ci"][1] < 0
    other = _perturb_run(shift=0.0)
    other["cats"]["pcb1"]["eval_images"] = other["cats"]["pcb1"]["eval_images"][::-1].copy()
    with pytest.raises(ValueError):
        an.visa_dev_table(e0, other, n_boot=10)

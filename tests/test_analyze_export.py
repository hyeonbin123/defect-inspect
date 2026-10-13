# [파일 역할] analyze_export.py(4단계: CPU로 내보낸 PatchCore의 FP32·INT8 점수가 GPU torch 점수와 얼마나
# 다른지, INT8에 FP32 임계값을 그대로 써도 되는지)를 검증하는 테스트 묶음.
# [왜 필요한가] 4단계 H14("FP32 임계값을 INT8 점수에 그대로 쓰면 합산 오검출률이 2.0%p 넘게 달라진다")의
# 판정이 이 모듈에서 나온다. 판정 규칙의 경계(정확히 2.0%p, 구간 끝이 정확히 0), 어느 파이프라인의 임계값을
# 쓰는지, INT8이 없는 실행의 처리, 출력 폴더 읽기와 명령행까지 합성 점수로 확인한다.
# [구성] 판정 규칙 표 테스트 → 합성 파이프라인(torch, fp32, int8)으로 build_report → 실제 폴더 형식으로 load와
# main.

import json

import numpy as np
import pytest

# ae: 테스트 대상. compare는 analyze_export가 부트스트랩·차이 계산을 빌려 쓰는 모듈이라 반복 횟수를 줄이려고,
# paths는 리포트 출력 폴더를 임시 폴더로 바꾸려고 가져온다.
from defect_inspect import analyze_export as ae
from defect_inspect import compare, paths

SUPPORTED, REJECTED, UNDECIDED = "지지", "기각", "판정 불가"


# 등록한 상수(목표 오검출률 5%, 최소 효과 2.0%p)가 코드에 그대로 있는지 확인한다. 누가 값을 바꾸면 이 테스트가
# 먼저 깨진다.
def test_the_registered_constants():
    # docs/experiments.md, stage 4: thresholds at a target of 5%, minimum effect 2.0 points.
    assert ae.ALPHA == 0.05
    assert ae.MIN_EFFECT_FPR == 0.02


# H14 판정 규칙(shift_verdict)을 오검출 "수"의 경계값으로 시험한다(정상 100장이면 2.0%p = 2장, 3,848장이면
# 76.96장).
# - 지지: 구간이 0을 포함하지 않고 변화가 2.0%p 이상. 정확히 2.0%p도 "이상"에 들고, 감소 방향도 변화로 본다
# - 기각: 구간 전체가 ±2.0%p 안쪽(끝이 정확히 2.0%p면 안쪽이 아님)
# - 판정 불가: 그 밖. 구간 끝이 정확히 0이면 0을 제외하지 않은 것이고, 구간이 NaN이어도 판정 불가다
# 규칙을 비율이 아니라 장수(Fraction)로 비교하는 이유(12/100 − 10/100이 부동소수로 0.01999…가 되는 문제)가
# 경계에서 실제로 지켜지는지 확인한다.
@pytest.mark.parametrize(
    ("fp_diff", "ci", "n_neg", "expected"),
    [
        # Supported: the interval excludes 0 and the change is at least 2.0 points.
        (3, (1, 5), 100, SUPPORTED),
        (-3, (-5, -1), 100, SUPPORTED),
        (2, (1, 3), 100, SUPPORTED),  # exactly 2.0 points counts as "at least"
        (-2, (-3, -1), 100, SUPPORTED),
        (3, (2.5, 3.0), 100, SUPPORTED),
        (77, (40, 110), 3848, SUPPORTED),  # 2.0 points of 3,848 normals are 76.96 images
        # Rejected: the whole interval lies strictly inside +-2.0 points.
        (1, (0.5, 1.5), 100, REJECTED),
        (0, (-1, 1), 100, REJECTED),
        (0, (0, 0), 100, REJECTED),
        (1, (0.1, 1.99), 100, REJECTED),
        (76, (40, 76.9), 3848, REJECTED),
        (-76, (-76.9, -40), 3848, REJECTED),
        # Undecided: everything else.
        (1, (0.5, 3), 100, UNDECIDED),  # excludes 0, too small, but may be larger than 2.0 points
        (3, (-0.5, 6), 100, UNDECIDED),  # large, but the interval holds 0
        (1, (-3, 1.5), 100, UNDECIDED),
        (1, (0.1, 2), 100, UNDECIDED),  # an end exactly at 2.0 points is not inside
        (-1, (-2, -0.1), 100, UNDECIDED),
        (3, (0, 6), 100, UNDECIDED),  # an end exactly at 0 does not exclude 0
        (-3, (-6, 0), 100, UNDECIDED),
        (76, (40, 77), 3848, UNDECIDED),
        (0, (float("nan"), float("nan")), 100, UNDECIDED),
    ],
)
def test_shift_verdict_follows_the_registered_rule(fp_diff, ci, n_neg, expected):
    assert ae.shift_verdict(fp_diff, list(ci), n_neg) == expected


# 범주 하나를 analyze_export의 공통 모양(평가 이미지·라벨·점수, 보정 점수)으로 만드는 도우미.
def _category(name, neg, pos, cal):
    scores = np.concatenate([neg, pos]).astype(np.float32)
    return {
        "eval_images": np.array([f"{name}/{i}" for i in range(len(scores))]),
        "eval_labels": np.array([0] * len(neg) + [1] * len(pos), dtype=np.int8),
        "eval_score": scores,
        "cal_score": np.asarray(cal, dtype=np.float32),
    }


# 모든 범주의 평가 점수와 보정 점수를 일정량 옮긴 사본. 양자화로 점수 전체가 밀리는 상황을 흉내 낸다.
def _shifted(cats, eval_shift=0.0, cal_shift=0.0):
    return {
        name: {
            **cat,
            "eval_score": cat["eval_score"] + np.float32(eval_shift),
            "cal_score": cat["cal_score"] + np.float32(cal_shift),
        }
        for name, cat in cats.items()
    }


# 합성 파이프라인 하나: 범주 a(정상 200, 결함 40, 보정 301), b(정상 120, 결함 20, 보정 97). 정상과 보정 점수는
# 같은 분포 N(1.0, 0.2), 결함은 N(1.6, 0.3)이다.
def _base(seed=0, sizes=(("a", 200, 40, 301), ("b", 120, 20, 97))):
    rng = np.random.default_rng(seed)
    return {
        name: _category(
            name, rng.normal(1.0, 0.2, n_neg), rng.normal(1.6, 0.3, n_pos), rng.normal(1.0, 0.2, n_cal)
        )
        for name, n_neg, n_pos, n_cal in sizes
    }


# build_report가 받는 실행 모양(메타 + 파이프라인 표)으로 감싼다.
def _run(pipelines):
    return {"meta": {"artifacts": "x", "ratio": 0.01, "protocol": "dev"}, "pipelines": pipelines}


# 부트스트랩 횟수를 2,000에서 300으로 줄여 테스트를 빠르게 한다(analyze_export는 compare.Bootstrap을 쓴다).
@pytest.fixture(autouse=True)
def few_draws(monkeypatch):
    monkeypatch.setattr(compare, "N_BOOT", 300)


# 검증용 독립 구현: conformal 순위 l = ceil((n+1)(1−α))를 직접 써서(부동소수 오차를 round로 정리, l > n이면
# 최댓값) 합산 오검출률·검출률을 계산한다. 모듈의 결과를 이것과 비교한다.
def _independent_rates(scores, thresholds_from, alpha=0.05):
    """Pooled false alarm and detection rates with the conformal rank written out."""
    fp = tp = n_neg = n_pos = 0
    for name, cat in scores.items():
        cal = np.sort(thresholds_from[name]["cal_score"].astype(np.float64))
        rank = int(np.ceil(round((len(cal) + 1) * (1 - alpha), 9)))
        thr = cal[rank - 1] if rank <= len(cal) else cal[-1]
        labels = cat["eval_labels"]
        fp += int((cat["eval_score"][labels == 0] > thr).sum())
        tp += int((cat["eval_score"][labels == 1] > thr).sum())
        n_neg += int((labels == 0).sum())
        n_pos += int((labels == 1).sum())
    return fp / n_neg, tp / n_pos


# INT8 점수가 FP32와 똑같으면 차이가 0이고 구간도 [0, 0]이라 H14는 기각이어야 한다. 세 칸(FP32, FP32 임계값의
# INT8, 다시 잡은 INT8)도 같아야 한다.
def test_identical_int8_scores_reject_the_claim():
    base = _base()
    report = ae.build_report(_run({"torch": base, "fp32": base, "int8": base}))
    shift = report["int8_at_fp32_thresholds"]
    assert shift["verdict"] == REJECTED
    assert shift["d_fpr"] == 0.0 and shift["d_fpr_ci"] == [0.0, 0.0]
    assert shift["fp32"] == shift["int8_at_fp32_thresholds"] == shift["int8_recalibrated"]
    assert report["int8_scored"] is True


# INT8 점수 전체를 0.15 올리면 FP32 임계값에서 오검출률이 5%p 넘게 늘고 구간 하한도 2%p를 넘어 H14가
# 지지된다(실제 4단계에서 4.18% → 68.48%로 지지된 현상). 비율은 독립 구현과 같아야 한다. INT8 자신의 보정
# 점수로 다시 잡으면 모든 점수가 같은 만큼 움직였으므로 FP32 비율로 돌아와야 한다. 아래로 밀린 경우(오검출과
# 검출이 같이 줄어듦)도 변화로 지지되는지 본다.
def test_a_clear_shift_of_the_int8_scores_supports_the_claim_and_recalibration_undoes_it():
    base = _base()
    int8 = _shifted(base, eval_shift=0.15, cal_shift=0.15)
    report = ae.build_report(_run({"torch": base, "fp32": base, "int8": int8}))
    shift = report["int8_at_fp32_thresholds"]
    fp32_fpr, fp32_tpr = _independent_rates(base, base)
    int8_fpr, int8_tpr = _independent_rates(int8, base)
    assert shift["fp32"]["fpr"] == pytest.approx(fp32_fpr) and shift["fp32"]["tpr"] == pytest.approx(fp32_tpr)
    assert shift["int8_at_fp32_thresholds"]["fpr"] == pytest.approx(int8_fpr)
    assert shift["int8_at_fp32_thresholds"]["tpr"] == pytest.approx(int8_tpr)
    assert shift["d_fpr"] == pytest.approx(int8_fpr - fp32_fpr) and shift["d_fpr"] > 0.05
    assert shift["d_tpr"] == pytest.approx(int8_tpr - fp32_tpr)
    lo, hi = shift["d_fpr_ci"]
    assert 0.02 < lo < shift["d_fpr"] < hi
    assert shift["verdict"] == SUPPORTED
    # Every score moved by the same amount: thresholds from the INT8 scores give the fp32 rates back.
    assert shift["int8_recalibrated"]["fpr"] == pytest.approx(fp32_fpr)
    assert shift["int8_recalibrated"]["tpr"] == pytest.approx(fp32_tpr)
    # A shift downwards is a change as well (fewer false alarms, fewer detections).
    lower = ae.build_report(_run({"torch": base, "fp32": base, "int8": _shifted(base, eval_shift=-0.3)}))
    assert lower["int8_at_fp32_thresholds"]["d_fpr"] < -0.02
    assert lower["int8_at_fp32_thresholds"]["verdict"] == SUPPORTED


# 정상 20장 중 1장만 FP32 임계값을 넘으면 점 추정은 +5%p로 크지만, 그 한 장이 재표본에서 빠질 확률이 0.95^20 ≈
# 0.36이라 구간 하한이 0이 된다. 그래서 판정 불가여야 한다. 점 추정만 크다고 지지하지 않고 구간이 0을 제외해야
# 한다는 규칙을 지키는지 본다.
def test_one_flipped_image_among_few_normals_is_undecided():
    # 20 normals below the threshold 2.0 (rank 96 of 100 calibration scores), 5 defects above it.
    cal = np.linspace(1.0, 2.0 + 4 / 95, 100)
    assert np.sort(cal)[95] == pytest.approx(2.0)
    fp32 = {"a": _category("a", np.full(20, 1.5), np.full(5, 3.0), cal)}
    int8 = {"a": {**fp32["a"], "eval_score": fp32["a"]["eval_score"].copy()}}
    int8["a"]["eval_score"][7] = 2.5  # one normal crosses the fp32 threshold: +5 points
    report = ae.build_report(_run({"torch": fp32, "fp32": fp32, "int8": int8}))
    shift = report["int8_at_fp32_thresholds"]
    assert shift["fp32"]["fpr"] == 0.0 and shift["d_fpr"] == pytest.approx(0.05)
    # A resample misses that one image with probability 0.95 ** 20 = 0.36: the interval starts at 0.
    assert shift["d_fpr_ci"][0] == 0.0 and shift["d_fpr_ci"][1] > 0.02
    assert shift["verdict"] == UNDECIDED


# 임계값의 출처가 맞는지 확인한다. INT8 자신의 보정 점수는 크게 올려(자기 임계값이면 아무것도 표시 안 함),
# torch 보정 점수는 크게 내려(torch 임계값이면 전부 표시) 둔다. H14의 "FP32 임계값의 INT8"은 둘 중 어느 것도
# 아닌 FP32 임계값을 써야 하므로 FP32와 같은 비율이 나와야 한다. 표의 각 칸(자기 임계값, torch 임계값)도
# 의도한 출처를 쓰는지 본다.
def test_the_int8_rates_are_taken_at_the_fp32_thresholds():
    base = _base()
    # INT8 evaluation scores are unchanged; only its own calibration scores moved far up.
    int8 = _shifted(base, cal_shift=10.0)
    torch_side = _shifted(base, cal_shift=-10.0)  # thresholds below every score: everything is flagged
    report = ae.build_report(_run({"torch": torch_side, "fp32": base, "int8": int8}))
    shift = report["int8_at_fp32_thresholds"]
    assert shift["fp32"]["fpr"] > 0.0
    assert shift["int8_at_fp32_thresholds"] == shift["fp32"] and shift["d_fpr"] == 0.0
    assert shift["int8_recalibrated"]["fpr"] == 0.0 and shift["int8_recalibrated"]["tpr"] == 0.0
    rows = {row["pipeline"]: row for row in report["rows"]}
    assert rows["int8"]["own_thresholds"]["fpr"] == 0.0
    assert rows["int8"]["torch_thresholds"]["fpr"] == 1.0 and rows["fp32"]["torch_thresholds"]["fpr"] == 1.0
    assert rows["torch"]["own_thresholds"]["fpr"] == 1.0
    assert rows["fp32"]["d_fpr_vs_torch_at_torch_thresholds"]["diff"] == 0.0
    assert rows["fp32"]["own_thresholds"]["fpr"] == pytest.approx(_independent_rates(base, base)[0])


# 최소 효과를 코드에 숫자로 박지 않고 MIN_EFFECT_FPR 상수에서 읽는지 확인한다. 상수를 0.9로 바꾸면 같은 변화가
# 기각으로 바뀌어야 한다.
def test_the_minimum_effect_is_read_from_the_registered_constant(monkeypatch):
    base = _base()
    int8 = _shifted(base, eval_shift=0.15)
    assert (
        ae.build_report(_run({"torch": base, "fp32": base, "int8": int8}))["int8_at_fp32_thresholds"][
            "verdict"
        ]
        == SUPPORTED
    )
    monkeypatch.setattr(ae, "MIN_EFFECT_FPR", 0.9)
    assert (
        ae.build_report(_run({"torch": base, "fp32": base, "int8": int8}))["int8_at_fp32_thresholds"][
            "verdict"
        ]
        == REJECTED
    )


# INT8 점수가 없는 실행(DINOv2 설정은 INT8로 양자화할 수 없다, docs/experiments.md 규칙 변경 5)이면 H14 항목을
# 만들지 않고 int8_scored를 False로 두며, 출력에 "INT8을 채점하지 않아 판정 없음"을 적어야 한다. 판정이 조용히
# 빠지는 것을 막는다.
def test_a_run_without_int8_has_no_verdict_and_says_so():
    base = _base()
    report = ae.build_report(_run({"torch": base, "fp32": base}))
    assert "int8_at_fp32_thresholds" not in report and report["int8_scored"] is False
    assert [row["pipeline"] for row in report["rows"]] == ["torch", "fp32"]
    assert "INT8 was not scored" in ae.format_report(report)


# run_export_eval의 출력 폴더 형식(fp32/·int8/ 아래 범주별 npz, 같은 이미지의 torch 점수 torch_* 포함,
# run.json)을 임시로 만들어 load가 torch·fp32·int8 세 파이프라인을 복원하는지, main이
# reports/stage4/export-<절차>.json을 쓰는지(paths.REPORTS는 임시 폴더로 바꿈) 확인한다. FP32는 torch와 같은
# 배열이라 점수 차이 0, INT8은 5% 넘게 달라야 하고, --no-write는 출력만 한다.
def test_load_report_and_cli_on_a_run_folder(tmp_path, monkeypatch, capsys):
    base = _base()
    int8 = _shifted(base, eval_shift=0.15, cal_shift=0.15)
    out = tmp_path / "export-x-dev"
    for precision, cats in (("fp32", base), ("int8", int8)):
        (out / precision).mkdir(parents=True)
        for name, cat in cats.items():
            np.savez(
                out / precision / f"{name}.npz",
                eval_images=cat["eval_images"],
                eval_labels=cat["eval_labels"],
                eval_score_full=cat["eval_score"],
                pool_images=np.array([f"{name}/p{i}" for i in range(len(cat["cal_score"]))]),
                pool_folds=np.zeros(len(cat["cal_score"]), dtype=np.int64),
                pool_score_oof=cat["cal_score"],
                torch_eval_score=base[name]["eval_score"],
                torch_pool_score_oof=base[name]["cal_score"],
            )
    run = {"artifacts": "x", "ratio": 0.01, "protocol": "dev", "precisions": ["fp32", "int8"]}
    run["categories"] = list(base)
    (out / "run.json").write_text(json.dumps(run), encoding="utf-8")

    loaded = ae.load(out)
    assert list(loaded["pipelines"]) == ["torch", "fp32", "int8"]
    np.testing.assert_array_equal(loaded["pipelines"]["int8"]["a"]["eval_score"], int8["a"]["eval_score"])
    np.testing.assert_array_equal(loaded["pipelines"]["torch"]["b"]["cal_score"], base["b"]["cal_score"])

    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    ae.main([str(out)])
    text = capsys.readouterr().out
    assert f"verdict: {SUPPORTED}" in text
    saved = json.loads((tmp_path / "reports" / "stage4" / "export-dev.json").read_text(encoding="utf-8"))
    assert saved["int8_at_fp32_thresholds"]["verdict"] == SUPPORTED
    assert saved["protocol"] == "dev" and saved["n_boot"] == 300 and saved["alpha"] == 0.05
    rows = {row["pipeline"]: row for row in saved["rows"]}
    assert rows["fp32"]["score_rel_diff_vs_torch"] == {"median": 0.0, "max": 0.0}
    assert rows["int8"]["score_rel_diff_vs_torch"]["max"] > 0.05
    ae.main([str(out), "--no-write"])  # prints only

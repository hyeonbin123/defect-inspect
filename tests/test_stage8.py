# [파일 역할] 8단계 실행기 세 개(run_m2ad_threshold, run_m2ad_loop, run_m2ad_dinomaly)와 분석
# 모듈(analyze_stage8)을 가짜 백본·가짜 캐시로 검증한다.
# [왜 필요한가] 8단계는 GPU로 한 번 재고, 8-B·8-C는 봉인된 M2AD test를 읽는다(docs/experiments.md 8단계). 그
# 전에 같은 코드 경로를 작은 합성 자료로 돌려 (1) 8-A가 3-B의 겹 뱅크를 그대로 다시 만드는지(동일성 관문), (2)
# 8-B의 재현 팔이 7단계 폐루프와, none n = 30이 3-B 재보정 n = 30과 비트 단위로 같은지(재현 관문), (3) 사람
# 확인 거르기가 상위 m장 가운데 결함만 빼는지, (4) 8-C의 학습 이미지 고르기·채점·장부 기록, (5) 분석의
# 판정선과 팔 대응이 규칙대로인지 확인한다.
# [연결] 가짜 자료(make_rows, FakeCache, GridMean, CFG)는 7단계 테스트의 것을 그대로 가져다 쓴다.

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_run_m2ad_enrol import CFG, N_TRAIN, FakeCache, GridMean, load, make_rows, tools  # noqa: E402

from defect_inspect import (  # noqa: E402
    analyze_m2ad,
    analyze_stage8,
    ledger,
    paths,
    run_dinomaly,
    run_m2ad,
    run_m2ad_dinomaly,
    run_m2ad_enrol,
    run_m2ad_loop,
    run_m2ad_threshold,
)
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.m2ad import (  # noqa: E402
    ILLUMINATIONS,
    illumination_groups,
    recal_specimens,
    specimen_folds,
)
from defect_inspect.splits import SealedTestError  # noqa: E402


@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    for module in (run_m2ad_threshold, run_m2ad_loop, run_m2ad_dinomaly):
        monkeypatch.setattr(module, "git_commit", lambda root=None: "abc1234")
    return path


# 3-B 실행(검사기 하나)을 가짜 자료로 만든다. 재보정 n = 8, 30까지 돌려 분석이 뱅크+임계값 행을 쓸 수 있게
# 하고, analyze_m2ad.load가 읽는 run.json도 쓴다.
def _stage3b(out, rows, inspectors=(("Motor", "000"),)):
    out.mkdir(parents=True, exist_ok=True)
    infos = []
    for category, view in inspectors:
        infos.append(
            run_m2ad.run_inspector(
                CFG, category, view, rows, FakeCache(), GridMean(), "cpu", out, allow_test=True
            )
        )
    (out / "run.json").write_text(json.dumps({"inspectors": infos}), encoding="utf-8")
    return out


# ---------------------------------------------------------------- 8-A


@pytest.fixture(scope="module")
def threshold_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage8a")
    rows = make_rows()
    ref = _stage3b(root / "m2ad-p0", rows)
    out = root / "m2ad-threshold-p0"
    out.mkdir()
    info = run_m2ad_threshold.rebuild_inspector(tools(), "Motor", "000", rows, FakeCache(), out, ref)
    (out / "run.json").write_text(json.dumps({"inspectors": [info], "no_gate_stop": False}), encoding="utf-8")
    return root, rows, info


# 동일성 관문: 다시 만든 겹 뱅크의 I01 교차 적합 점수가 3-B와 정확히 같다. 새 조명 점수는 그 시편의 겹을 뺀
# 뱅크(I01 24장)로 채점한 값이다(손으로 다시 계산해 대조).
def test_threshold_run_rebuilds_the_stage3b_fold_banks(threshold_runs):
    root, rows, info = threshold_runs
    assert info["gate"] == {"max_rel_diff": 0.0, "tolerance": 1e-6, "threshold_equal": True, "ok": True}
    z = load(root / "m2ad-threshold-p0" / "Motor_000.npz")
    old = load(root / "m2ad-p0" / "Motor_000.npz")
    assert np.array_equal(z["ref_oof"], old["cal_score"])
    t = tools()
    names = z["specimens"].tolist()
    folds = np.array([specimen_folds(names)[n] for n in names])
    train, _ = run_m2ad_enrol._train_rows(rows, "Motor", "000")
    ref_images = FakeCache().images(train["01"])
    feats = t.features(ref_images)
    li = z["lights"].tolist().index("05")
    images = FakeCache().images(train["05"])
    for fold in range(5):
        held = folds == fold
        bank = t.build(feats[torch.from_numpy(~held)].reshape(-1, feats.shape[2]))
        assert np.array_equal(z["cal_scores"][li, held], t.score(bank, images[held]))


# 관문이 어긋나면(저장된 점수를 바꿔 흉내) 멈추고, 점검용(stop_on_gate=False)이면 기록만 하고 계속한다.
def test_threshold_gate_stops_on_a_mismatch(threshold_runs, tmp_path):
    root, rows, _ = threshold_runs
    bad = tmp_path / "bad"
    bad.mkdir()
    old = load(root / "m2ad-p0" / "Motor_000.npz")
    old["cal_score"] = old["cal_score"] * 1.01
    np.savez(bad / "Motor_000.npz", **old)
    with pytest.raises(RuntimeError, match="differ from stage 3-B"):
        run_m2ad_threshold.rebuild_inspector(tools(), "Motor", "000", rows, FakeCache(), tmp_path, bad)
    info = run_m2ad_threshold.rebuild_inspector(
        tools(), "Motor", "000", rows, FakeCache(), tmp_path, bad, stop_on_gate=False
    )
    assert info["gate"]["ok"] is False and (tmp_path / "Motor_000.npz").exists()


# 분석: 팔 Tn의 임계값은 새 조명 학습 시편 앞 n개 점수의 conformal 임계값이다. n = 8은 보장 불가, 19와 30은
# 보장. 점검용 실행(--no-gate-stop)과 관문 실패 실행은 받지 않는다.
def test_calibrate_analysis_follows_the_rule(threshold_runs, monkeypatch):
    root, _, _ = threshold_runs
    monkeypatch.setattr(paths, "OUTPUTS", root)
    report = analyze_stage8.calibrate_report(["p0"], n_boot=50)
    arms = report["methods"]["p0"]["arms"]
    assert arms["T8"]["thresholds"]["guaranteed"] == 0 and arms["T8"]["thresholds"]["rank"] == [8]  # 최댓값
    assert arms["T19"]["thresholds"]["guaranteed"] == 9 and arms["T30"]["thresholds"]["rank"] == [30]
    run = analyze_stage8.load_threshold_run(root / "m2ad-threshold-p0")
    ref = analyze_m2ad.load(root / "m2ad-p0")
    cell, thresholds = analyze_stage8.threshold_cell(ref, run, "03", 30)
    z = run["inspectors"]["Motor_000"]
    li = z["lights"].tolist().index("03")
    assert cell[0][2] == float(z["cal_scores"][li].max()) == thresholds[0].value
    assert report["hypotheses"]["H23"] in ("지지", "기각", "판정 불가")
    meta = json.loads((root / "m2ad-threshold-p0" / "run.json").read_text(encoding="utf-8"))
    meta["no_gate_stop"] = True
    (root / "check").mkdir(exist_ok=True)
    (root / "check" / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="code-path check"):
        analyze_stage8.load_threshold_run(root / "check")


# 8-A 팔마다 검사기·조명 평균 이미지 AUROC가 붙는다(저장된 점수로만 계산한다). 임계값만 팔(T8·T19·T30)은 test
# 점수가 재보정 없음(none, 3-B E0 점수)과 같으므로 AUROC와 구간도 none과 같고, 뱅크+임계값 팔은 3-B 재보정
# 점수로 잰다. 새 키(팔의 auroc·auroc_ci, h23의 d_auroc_t30_bank30)를 빼면 나머지는 키를 더하기 전 출력과
# 값·키 순서까지 같다. 이미 커밋한 reports/stage8/calibrate.json을 다시 만들 때 기존 값이 바뀌지 않는다는
# 근거다.
def test_calibrate_arms_carry_the_image_auroc(threshold_runs):
    from defect_inspect.analyze_m2ad_enrol import _public, _rates
    from defect_inspect.metrics import auroc

    root, _, _ = threshold_runs
    ref = analyze_m2ad.load(root / "m2ad-p0")
    run = analyze_stage8.load_threshold_run(root / "m2ad-threshold-p0")
    boot = analyze_m2ad.ClusterBootstrap(ref, 50, 0)
    entry = analyze_stage8.calibrate_method(ref, run, boot)
    arms, lights = entry["arms"], analyze_stage8.NEW_LIGHTS
    for name in ("T8", "T19", "T30"):
        assert arms[name]["auroc"] == arms["none"]["auroc"]
        assert arms[name]["auroc_ci"] == arms["none"]["auroc_ci"]

    # 관측값은 조명마다 낸 이미지 AUROC(가짜 자료는 검사기 하나라 그 검사기의 값)의 조명 평균, 구간은 같은
    # 추출에서 조명 평균을 낸 재표집 표본의 백분위 구간이다.
    def observed(cells):
        assert all(len(cell) == 1 for cell in cells)
        values = []
        for cell in cells:
            scores, labels, _ = cell[0]
            values.append(auroc(scores[labels == 0], scores[labels == 1]))
        return float(np.mean(values))

    def samples(cells):
        return np.mean(np.stack([boot.auroc(cell)[1] for cell in cells]), axis=0)

    none_cells = [analyze_m2ad.condition_cell(ref, f"R:{light}") for light in lights]
    bank_cells = {n: [analyze_m2ad.recal_cell(ref, f"{light}:{n}") for light in lights] for n in (8, 30)}
    assert arms["none"]["auroc"] == pytest.approx(observed(none_cells), abs=1e-12)
    assert arms["none"]["auroc_ci"] == analyze_m2ad._ci(samples(none_cells))
    for n, cells in bank_cells.items():
        assert arms[f"bank{n}"]["auroc"] == pytest.approx(observed(cells), abs=1e-12)
        assert arms[f"bank{n}"]["auroc_ci"] == analyze_m2ad._ci(samples(cells))
    d = entry["h23"]["d_auroc_t30_bank30"]
    assert d["diff"] == arms["T30"]["auroc"] - arms["bank30"]["auroc"]
    assert d["ci"] == analyze_m2ad._ci(samples(none_cells) - samples(bank_cells[30]))

    # 키를 더하기 전 출력: 팔은 비율·구간(+ 임계값 요약), H23은 같은 여덟 키, 조명별 표와 관문은 그대로.
    before = {}
    for n in analyze_stage8.THRESHOLD_ARMS:
        cells, thresholds = [], []
        for light in lights:
            cell, thr = analyze_stage8.threshold_cell(ref, run, light, n)
            cells.append(cell)
            thresholds += thr
        summary = analyze_stage8._threshold_summary(thresholds)
        before[f"T{n}"] = _public(_rates(boot, cells) | {"thresholds": summary})
    for n, cells in bank_cells.items():
        before[f"bank{n}"] = _public(_rates(boot, cells))
    before["none"] = _public(_rates(boot, none_cells))
    new_keys = ("auroc", "auroc_ci")
    stripped = {k: {kk: vv for kk, vv in v.items() if kk not in new_keys} for k, v in arms.items()}
    assert json.dumps(stripped) == json.dumps(before)
    assert all(tuple(v)[-2:] == new_keys for v in arms.values())
    assert list(entry["h23"]) == [
        "d_tpr",
        "d_tpr_ci",
        "margin",
        "verdict",
        "t30_fpr",
        "t30_fpr_ci",
        "t30_fpr_upper_at_most_10pct",
        "d_fpr_t30_bank30",
        "d_auroc_t30_bank30",
        "bank30_at_t30_false_positives",  # 2026-10-30 보충(아래 테스트), 끝에 붙는다
    ]
    assert list(entry) == ["arms", "per_light", "h23", "gate"]
    assert all("auroc" not in key for v in entry["per_light"].values() for key in v)


# 8-A 2026-10-30 보충(판정 없음, test 라벨로 맞춘 기술 통계): 뱅크+임계값 n = 30 팔의 임계값을 검사기·조명
# 칸마다 그 칸 test 정상 점수의 k + 1번째로 큰 값으로 옮겨, 같은 칸 T30 팔의 오검출 수 k와 같게 맞춘다. 확인:
# (1) 손으로 만든 칸에서 임계값과 오검출 수, 동점이면 오검출 수가 k보다 작아지고 그 칸을 세는 것, k가 정상
# 수와 같으면 -inf(모두 불량), 라벨이 다른 두 칸은 거절, 라벨 -1 이미지는 세지 않음, (2) 가짜 8-A 자료에서
# 맞춘 팔의 수치가 맞춘 칸을 _rates에 넣은 것과 같고, 칸마다 오검출 수가 T30 이하(동점이 없으면 같음)이며,
# 분리도 몫(T30 − 맞춘 팔)과 운영점 몫(맞춘 팔 − 뱅크+임계값 n = 30)의 합이 H23 차이와 같고 구간이 같은 추출의
# 표본으로 난다.
def test_bank_arm_matched_to_the_t30_false_positives(threshold_runs):
    from defect_inspect.analyze_m2ad_enrol import _public, _rates

    labels = np.array([0, 0, 0, 0, 1, 1, -1])
    # 기준 칸: 정상 점수 0.1, 0.9, 0.8, 0.2 가운데 임계값 0.5를 넘는 것은 두 개(k = 2). 라벨 -1(5.0)은 세지
    # 않는다.
    reference = [(np.array([0.1, 0.9, 0.8, 0.2, 0.95, 0.3, 5.0]), labels, 0.5)]
    target = [(np.array([0.4, 0.7, 0.6, 0.1, 0.65, 0.2, 9.0]), labels, 0.75)]
    matched, ties = analyze_stage8.match_false_positives(reference, target)
    # 맞출 칸의 정상 점수 내림차순 0.7, 0.6, 0.4, 0.1에서 세 번째 값 0.4가 임계값이고, 0.7·0.6만 넘는다.
    assert matched[0][2] == 0.4 and ties == 0
    assert matched[0][0] is target[0][0] and matched[0][1] is labels
    # 동점: 정상 0.6, 0.6, 0.6, 0.1이면 세 번째 값도 0.6이라 넘는 정상이 0개(k = 2보다 적다) → 동점 칸 하나.
    tied = [(np.array([0.6, 0.6, 0.6, 0.1, 0.65, 0.2, 9.0]), labels, 0.75)]
    matched, ties = analyze_stage8.match_false_positives(reference, tied)
    assert matched[0][2] == 0.6 and ties == 1
    # 기준 칸이 모든 정상을 불량으로 두면(k = 4) 맞출 칸도 모두 불량(-inf).
    everything = [(reference[0][0], labels, 0.0)]
    matched, ties = analyze_stage8.match_false_positives(everything, target)
    assert matched[0][2] == float("-inf") and ties == 0
    with pytest.raises(ValueError, match="same test images"):
        analyze_stage8.match_false_positives(reference, [(target[0][0], labels[::-1].copy(), 0.75)])

    root, _, _ = threshold_runs
    ref = analyze_m2ad.load(root / "m2ad-p0")
    run = analyze_stage8.load_threshold_run(root / "m2ad-threshold-p0")
    boot = analyze_m2ad.ClusterBootstrap(ref, 50, 0)
    entry = analyze_stage8.calibrate_method(ref, run, boot)
    h23, arms = entry["h23"], entry["arms"]
    got = h23["bank30_at_t30_false_positives"]
    lights = analyze_stage8.NEW_LIGHTS
    t30 = [analyze_stage8.threshold_cell(ref, run, light, 30)[0] for light in lights]
    bank = [analyze_m2ad.recal_cell(ref, f"{light}:30") for light in lights]
    cells, ties = [], 0
    for t30_cell, bank_cell in zip(t30, bank, strict=True):
        cell, n_ties = analyze_stage8.match_false_positives(t30_cell, bank_cell)
        cells.append(cell)
        ties += n_ties
        for (s_ref, lab, thr_ref), (s, _, thr) in zip(t30_cell, cell, strict=True):
            k = int((s_ref[lab == 0] > thr_ref).sum())
            assert int((s[lab == 0] > thr).sum()) == k or n_ties > 0
    rates = _rates(boot, cells)
    expected = _public(rates) | {"tied_cells": ties}
    assert {k: got[k] for k in expected} == expected
    assert got["n_normal"] == arms["T30"]["n_normal"] and got["n_defect"] == arms["T30"]["n_defect"]
    if ties == 0:
        assert got["false_positives"] == arms["T30"]["false_positives"]
    t30_rates, bank_rates = _rates(boot, t30), _rates(boot, bank)
    sep, op = got["d_tpr_separation"], got["d_tpr_operating_point"]
    assert sep["diff"] == t30_rates["tpr"] - rates["tpr"] and op["diff"] == rates["tpr"] - bank_rates["tpr"]
    assert sep["diff"] + op["diff"] == pytest.approx(h23["d_tpr"], abs=1e-12)
    assert sep["ci"] == analyze_m2ad._ci(t30_rates["tpr_samples"] - rates["tpr_samples"])
    assert op["ci"] == analyze_m2ad._ci(rates["tpr_samples"] - bank_rates["tpr_samples"])
    assert list(got)[-3:] == ["tied_cells", "d_tpr_separation", "d_tpr_operating_point"]


def test_verdict_lines():
    assert analyze_stage8._verdict_below(-0.4, -0.3, -0.2) == "지지"
    assert analyze_stage8._verdict_below(-0.1, 0.0, -0.2) == "기각"
    assert analyze_stage8._verdict_below(-0.3, -0.1, -0.2) == "판정 불가"
    assert analyze_stage8._verdict_limit(0.02, 0.10, 0.10) == "지지"
    assert analyze_stage8._verdict_limit(0.11, 0.2, 0.10) == "기각"
    assert analyze_stage8._verdict_limit(0.05, 0.12, 0.10) == "판정 불가"


# ---------------------------------------------------------------- 8-B


def test_dropped_indices():
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.7, 0.2])
    defect = np.array([False, True, False, False, True, False])
    assert run_m2ad_loop.dropped_indices("none", None, defect).tolist() == []
    # trim: ⌈0.2 × 6⌉ = 2장(가장 높은 0.9, 0.8)
    assert run_m2ad_loop.dropped_indices("trim", scores, defect).tolist() == [1, 3]
    # label: 상위 4장(0.9, 0.8, 0.7, 0.5) 가운데 실제 결함(1번, 4번)만 뺀다.
    assert run_m2ad_loop.dropped_indices("label", scores, defect).tolist() == [1, 4]
    with pytest.raises(ValueError):
        run_m2ad_loop.dropped_indices("label", None, defect)


# 재현 관문의 전제: 8-B의 R(trim, k = 0, n = 20)은 7단계 폐루프와, none n = 30·k = 0은 3-B 재보정 n = 30과
# 점수·mixed 임계값이 정확히 같다. label은 상위 4장 가운데 결함만 빼고(사람 라벨 4개), cond 임계값은 더한
# 이미지 점수만으로 정한다(n = 20이면 20장 중 20번째).
def test_loop_reproduces_stage7_and_stage3b(tmp_path):
    rows = make_rows()
    with pytest.raises(SealedTestError):
        run_m2ad_loop.loop_inspector(tools(), "Motor", "000", rows, FakeCache(), tmp_path)
    info = run_m2ad_loop.loop_inspector(tools(), "Motor", "000", rows, FakeCache(), tmp_path, allow_test=True)
    z = load(tmp_path / "Motor_000.npz")
    keys = z["arm_keys"].tolist()
    s7 = tmp_path / "s7"
    s7.mkdir()
    run_m2ad_enrol.loop_inspector(tools(), "Motor", "000", rows, FakeCache(), s7, allow_test=True)
    old = load(s7 / "Motor_000_loop.npz")
    ref = _stage3b(tmp_path / "m2ad-p0", rows)
    b = load(ref / "Motor_000.npz")
    for light in [x for x in ILLUMINATIONS if x != "01"]:
        j, j7 = keys.index(f"{light}:20:0:trim"), old["loop_keys"].tolist().index(f"{light}:trim:0")
        assert np.array_equal(z["scores"][j], old["scores"][j7])
        assert z["thresholds_mixed"][j] == old["thresholds"][j7]
        j30, jb = keys.index(f"{light}:30:0:none"), b["recal_keys"].tolist().index(f"{light}:30")
        assert np.array_equal(z["scores"][j30], b["recal_scores"][jb])
        assert z["thresholds_mixed"][j30] == b["recal_thresholds"][jb]
        entry = info["arms"][f"{light}:20:0:none"]
        assert entry["threshold_cond"]["rank"] == 20 and entry["threshold_cond"]["n"] == 20
        lab = info["arms"][f"{light}:30:2:label"]
        assert lab["human_labels"] == 4 and lab["dropped_normals"] == 0
        assert lab["dropped_defects"] == 2  # 가짜 결함은 묶음에서 점수가 가장 높다
    # 분석의 재현 관문이 이 실행들을 통과시킨다.
    run = analyze_stage8.load_loop_run(tmp_path) if (tmp_path / "run.json").exists() else None
    assert run is None  # loop_inspector는 run.json을 쓰지 않는다(CLI가 쓴다)
    (tmp_path / "run.json").write_text(
        json.dumps({"inspectors": [info], "arms": run_m2ad_loop.ARMS}), encoding="utf-8"
    )
    run = analyze_stage8.load_loop_run(tmp_path)
    gates = analyze_stage8.loop_gates(run, analyze_m2ad.load(ref), s7)
    assert gates["ok"] is True and gates["stage7_trim"]["max_rel_diff"] == 0.0
    boot = analyze_m2ad.ClusterBootstrap(analyze_m2ad.load(ref), 30, 0)
    arm = analyze_stage8.loop_arm(run, boot, 30, 2, "label", "cond")
    assert arm["human_labels"] == 4 * 9 and arm["defects_in_batch"] == 2 * 9


# ---------------------------------------------------------------- 8-C


def test_dinomaly_arms_and_training_rows():
    a, b = illumination_groups()
    assert run_m2ad_dinomaly.arm_lights("e0") == ("01",)
    assert run_m2ad_dinomaly.arm_lights("e2-a") == ("01", *a) and run_m2ad_dinomaly.arm_lights("e2-b") == (
        "01",
        *b,
    )
    rows = make_rows(categories=("Motor", "Bird"))
    plan = run_m2ad_dinomaly.train_plan(rows)
    full = run_m2ad_dinomaly.training_rows(plan, "e0", "full")
    assert len(full) == 6 * N_TRAIN and {r.illumination for r in full} == {"01"}
    fold0 = run_m2ad_dinomaly.training_rows(plan, "e2-b", "fold0")
    held = {(k, n) for k, p in plan.items() for n, f in zip(p["names"], p["folds"], strict=True) if f == 0}
    assert len(fold0) == 6 * (N_TRAIN - 6) * 6
    assert not any(((r.category, r.view), r.specimen) in held for r in fold0)
    with pytest.raises(ValueError):
        run_m2ad_dinomaly.arm_lights("e1")


# score-train과 test의 경로를 가짜 모델로 돌린다: 모델 이름마다 다른 상수를 더한 밝기 평균을 점수로 낸다.
# test는 모든 팔의 학습·score-train이 끝나야 돌고, 장부에 한 줄 남긴다. 이어서 분석이 등록·못 본 조명을
# 7단계처럼 가른다.
class _Scores:
    def __init__(self, scores):
        self.image_scores = scores


def test_dinomaly_scoring_paths_and_analysis(tmp_path, monkeypatch, temporary_ledger):
    rows = make_rows(categories=("Motor", "Bird"))
    cache = FakeCache()
    monkeypatch.setattr(run_m2ad_dinomaly, "read_meta", lambda path: rows)
    monkeypatch.setattr(run_m2ad_dinomaly, "M2adCache", lambda root, size: cache)
    monkeypatch.setattr(run_m2ad_dinomaly, "_load", lambda out_dir, name, device: (out_dir.name, name))
    offsets = {"e0": 0.0, "e2-a": -0.05, "e2-b": -0.05}

    def fake_predict(model, images, **kwargs):
        arm, _ = model
        return _Scores(
            (1.0 - images.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0 + offsets[arm]).astype(np.float32)
        )

    monkeypatch.setattr(run_dinomaly, "predict", fake_predict)
    out = tmp_path / "outputs" / "m2ad-dms"
    for arm in run_m2ad_dinomaly.ARMS:
        (out / arm).mkdir(parents=True)
        for name in run_m2ad_dinomaly.MODELS:
            info = {
                "model": name,
                "train_images": 1,
                "final_loss": 0.1,
                "skipped_steps": 0,
                "peak_vram_mb": 0,
                "train_s": 1,
            }
            (out / arm / f"{name}.json").write_text(json.dumps(info), encoding="utf-8")
        run_m2ad_dinomaly.main(["score-train", "--arm", arm, "--out", str(out), "--device", "cpu"])
        z = load(out / arm / "train_scores.npz")
        assert z["scores"].shape == (6, 10, N_TRAIN) and np.isfinite(z["scores"]).all()
    assert not temporary_ledger.exists() and set(cache.splits) == {"train"}
    with pytest.raises(SystemExit):
        run_m2ad_dinomaly.main(["test", "--out", str(out), "--device", "cpu"])
    run_m2ad_dinomaly.main(
        ["test", "--out", str(out), "--device", "cpu", "--allow-test", "--stage", "8-m2ad"]
    )
    line = json.loads(temporary_ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert line["stage"] == "8-m2ad" and line["config"] == "m2ad-dms-280-car"
    # 분석: 3-B 실행(시편·부트스트랩 기준)을 가짜로 만들고 리포트를 낸다.
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    _stage3b(tmp_path / "outputs" / "m2ad-p0", rows, run_m2ad_dinomaly.INSPECTORS)
    train = analyze_stage8.dms_train_report()
    assert train["arms"]["e0"]["unseen_counts"][1] == 9 * 6 * N_TRAIN
    assert train["arms"]["e2-a"]["enrolled_counts"][1] == 5 * 6 * N_TRAIN
    report = analyze_stage8.dms_test_report(n_boot=30)
    e0, e2 = report["candidates"]["e0"], report["candidates"]["e2"]
    # 못 본 조명 정상: 조명 9 × 검사기 6 × 정상 시편 4. E2도 아홉 조명이 한 번씩.
    assert e0["unseen"]["n_normal"] == e2["unseen"]["n_normal"] == 9 * 6 * 4
    assert e2["reference"]["n_normal"] == 2 * 6 * 4
    assert report["hypotheses"]["H24"] in ("지지", "기각", "판정 불가")
    assert set(report["closed_loop_e0"]) == {"none:0", "trim:0", "label:0", "none:2", "trim:2", "label:2"}
    # E0의 임계값만 다시 잡기 T30은 학습 시편 점수의 최댓값이다(30장 중 30번째).
    tr = analyze_stage8.dms_train_scores(out, "e0")
    li = ILLUMINATIONS.index("04")
    specimens = tr["specimens"][0].tolist()
    idx = [specimens.index(s) for s in recal_specimens(specimens, 30)]
    assert conformal_threshold(tr["scores"][0, li, idx], 0.05).value == float(tr["scores"][0, li].max())


def test_e2_arm_mapping():
    a, b = illumination_groups()
    assert analyze_stage8._e2_arms("01", True) == ["e2-a", "e2-b"]
    assert analyze_stage8._e2_arms(a[0], False) == ["e2-b"] and analyze_stage8._e2_arms(a[0], True) == [
        "e2-a"
    ]
    assert analyze_stage8._e2_arms(b[0], False) == ["e2-a"] and analyze_stage8._e2_arms(b[0], True) == [
        "e2-b"
    ]


# ---------------------------------------------------------------- 8-D(한 조명 재등록: E0 모델 미세조정)


# 8-D의 새 조명(I01이 아닌 아홉 개)과 검사기 이름(3-B와 같은 순서).
FT_LIGHTS = tuple(light for light in ILLUMINATIONS if light != "01")
INSPECTOR_NAMES = [f"{c}_{v}" for c, v in run_m2ad_dinomaly.INSPECTORS]


# 가짜 3-B 실행(검사기 6개)을 모듈에서 한 번만 만든다. 8-D 분석은 시편 목록·부트스트랩·P0 뱅크+임계값 행을 이
# 실행에서 가져온다. run_inspector는 장부를 쓰지 않으므로(장부는 명령줄이 쓴다) 모듈 범위에서 만들어도 진짜
# 장부에 줄이 생기지 않는다.
@pytest.fixture(scope="module")
def stage8d_ref(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage8d")
    rows = make_rows(categories=("Motor", "Bird"))
    _stage3b(root / "m2ad-p0", rows, run_m2ad_dinomaly.INSPECTORS)
    return root, rows


# 미세조정 조명은 (I01, L)이고 L은 아홉 새 조명 가운데 하나다. 학습 이미지는 검사기 6개 × (I01, L) × 시편이고,
# 순서는 검사기 → 조명(I01 다음 L) → 시편 이름 순이다. 겹 f 모델의 이미지에는 겹 f 시편이 어느 조명으로도
# 없어야 한다(출발점인 E0 겹 f 모델도 그 시편을 본 적이 없으므로, 교차 적합 점수가 끝까지 깨끗하다).
def test_finetune_lights_and_training_rows():
    assert run_m2ad_dinomaly.NEW_LIGHTS == FT_LIGHTS
    assert run_m2ad_dinomaly.finetune_lights("05") == ("01", "05")
    for light in ("01", "11", 5):
        with pytest.raises(ValueError):
            run_m2ad_dinomaly.finetune_lights(light)
    rows = make_rows(categories=("Motor", "Bird"))
    plan = run_m2ad_dinomaly.train_plan(rows)
    full = run_m2ad_dinomaly.finetune_rows(plan, "05", "full")
    assert len(full) == 6 * 2 * N_TRAIN
    assert full == [
        row
        for key in run_m2ad_dinomaly.INSPECTORS
        for light in ("01", "05")
        for row in plan[key]["rows"][light]
    ]
    fold0 = run_m2ad_dinomaly.finetune_rows(plan, "05", "fold0")
    held = {(k, n) for k, p in plan.items() for n, f in zip(p["names"], p["folds"], strict=True) if f == 0}
    assert len(fold0) == 6 * 2 * (N_TRAIN - 6)
    assert {r.illumination for r in fold0} == {"01", "05"}
    assert not any(((r.category, r.view), r.specimen) in held for r in fold0)
    with pytest.raises(ValueError):
        run_m2ad_dinomaly.finetune_rows(plan, "05", "fold9")
    # 8-C 팔의 학습 이미지 고르기는 그대로다(같은 도우미를 나눠 쓴다).
    assert run_m2ad_dinomaly.training_rows(plan, "e0", "full") == [
        row for key in run_m2ad_dinomaly.INSPECTORS for row in plan[key]["rows"]["01"]
    ]


# finetune 명령의 경로: 조명마다 모델마다 8-C E0의 같은 이름 모델(<out>/e0/<모델>.pt)에서 출발해, 그 모델의
# 미세조정 이미지로 run_dinomaly.train을 8-D 일정(FINETUNE_SCHEDULE)·배치 16·seed 0으로 부르고, 결과를
# <out>/ft/<조명>/에 쓴다. 기록에는 출발 모델(이름·sha256·스텝·학습 커밋), 실제 옵티마이저·스케줄러 설정, 학습
# 이미지 수가 남는다. 진짜 train(가짜 작은 모델)으로 돌려 일정이 로그까지 이어지는지 보고, 끝난 모델은 다시
# 부르면 건너뛰며(--overwrite면 다시 함), 출발 모델이 8-C 규칙의 그 모델(dms-280-car, 3,000스텝, e0 팔)이
# 아니면 학습 전에 멈추는지 본다.
# global_step(8-D 규칙 2026-10-30 보충): 모델이 받는 global_step은 출발 모델의 학습 스텝 수(3,000)부터 이어
# 센다(3스텝이면 3000, 3001, 3002). Dinomaly 손실의 hard-mining 비율 p가 8-C 수렴 때 값(0.9) 그대로이게 하려는
# 것이고, 시작값은 기록의 config.global_step_start에 남는다.
def test_finetune_starts_from_the_e0_models_with_the_fine_tuning_schedule(tmp_path, monkeypatch):
    pytest.importorskip("anomalib")  # 진짜 train()이 anomalib의 StableAdamW·WarmCosineScheduler를 쓴다
    from test_run_dinomaly import TinyModel

    from defect_inspect.download import sha256_file

    rows = make_rows(categories=("Motor", "Bird"))
    cache = FakeCache()
    monkeypatch.setattr(run_m2ad_dinomaly, "read_meta", lambda path: rows)
    monkeypatch.setattr(run_m2ad_dinomaly, "M2adCache", lambda root, size: cache)
    monkeypatch.setattr(
        run_dinomaly, "encoder_weights", lambda encoder: {"hf_hub_id": f"timm/{encoder}.fake"}
    )
    cfg = run_dinomaly.CONFIGS[run_m2ad_dinomaly.CONFIG]
    out = tmp_path / "m2ad-dms"
    start = out / "e0"
    start.mkdir(parents=True)
    for name in run_m2ad_dinomaly.MODELS:
        (start / f"{name}.pt").write_bytes(f"e0 {name}".encode())
        (start / f"{name}.json").write_text(
            json.dumps({"arm": "e0", "model": name, "commit": "47b5569"}), encoding="utf-8"
        )
    start_meta = {
        "encoder": cfg.encoder,
        "steps": 3000,
        "amp": True,
        "config": cfg.name,
        "options": cfg.build_options(),
    }
    loaded = []

    def fake_load(path):
        loaded.append(path.relative_to(out).as_posix())
        torch.manual_seed(0)
        return TinyModel(), dict(start_meta)

    calls = []
    real_train = run_dinomaly.train

    def spy_train(model, images, **kwargs):
        calls.append((images.copy(), kwargs, model))
        return real_train(model, images, **kwargs)

    monkeypatch.setattr(run_dinomaly, "load_model", fake_load)
    monkeypatch.setattr(run_dinomaly, "train", spy_train)
    common = ["--out", str(out), "--device", "cpu", "--steps", "3"]
    argv = ["finetune", "--light", "05", "08", "--models", "full", "fold2", *common]
    run_m2ad_dinomaly.main(argv)

    assert loaded == ["e0/full.pt", "e0/fold2.pt", "e0/full.pt", "e0/fold2.pt"]
    plan = run_m2ad_dinomaly.train_plan(rows)
    order = [("05", "full"), ("05", "fold2"), ("08", "full"), ("08", "fold2")]
    for (images, kwargs, model), (light, name) in zip(calls, order, strict=True):
        assert np.array_equal(images, cache.images(run_m2ad_dinomaly.finetune_rows(plan, light, name)))
        assert kwargs["schedule"] == run_dinomaly.FINETUNE_SCHEDULE
        assert (kwargs["steps"], kwargs["batch_size"], kwargs["seed"], kwargs["amp"]) == (3, 16, 0, False)
        # 손실의 hard-mining 일정은 출발 모델의 3,000스텝에서 이어진다(0부터 다시 세지 않는다).
        assert kwargs["global_step_start"] == 3000
        assert [c[1] for c in model.calls] == [3000, 3001, 3002]
    info = json.loads((out / "ft" / "05" / "full.json").read_text(encoding="utf-8"))
    assert (info["arm"], info["light"], info["model"], info["lights"]) == ("ft", "05", "full", ["01", "05"])
    assert info["start"] == {
        "arm": "e0",
        "model": "full",
        "file": "e0/full.pt",
        "sha256": sha256_file(start / "full.pt"),
        "steps": 3000,
        "commit": "47b5569",
    }
    config = info["config"]
    assert (config["name"], config["steps"], config["batch_size"], config["seed"]) == (cfg.name, 3, 16, 0)
    assert config["global_step_start"] == 3000
    assert config["optimizer"] == json.loads(
        json.dumps({"name": "StableAdamW", **run_dinomaly.OPTIMIZER, "lr": 2e-4})
    )
    assert config["scheduler"] == {
        "name": "WarmCosineScheduler",
        "total_iters": 3,
        "base_value": 2e-4,
        "final_value": 2e-4,
        "warmup_iters": 10,
    }
    assert info["train_images"] == 6 * 2 * N_TRAIN
    fold2 = json.loads((out / "ft" / "08" / "fold2.json").read_text(encoding="utf-8"))
    assert fold2["train_images"] == 6 * 2 * (N_TRAIN - 6) and fold2["start"]["file"] == "e0/fold2.pt"
    assert info["env"] == run_dinomaly.runtime_env("cpu") and info["skipped_steps"] == 0
    # 저장한 모델에는 미세조정 스텝 수와 실제 정밀도가, 로그에는 마지막 스텝 한 줄(기본 10스텝마다)이 남는다.
    # 3스텝이면 warmup 중이라 마지막 학습률은 linspace(0, 2e-4, 10)의 셋째 값이다.
    saved = torch.load(out / "ft" / "05" / "full.pt", map_location="cpu", weights_only=True)
    assert (saved["steps"], saved["amp"], saved["config"]) == (3, False, cfg.name)
    lines = (out / "ft" / "05" / "full_log.jsonl").read_text(encoding="utf-8").splitlines()
    log = [json.loads(line) for line in lines]
    assert [e["step"] for e in log] == [3] and log[0]["lr"] == pytest.approx(2e-4 * 2 / 9, rel=1e-12)

    # 끝난 모델은 건너뛰고, --overwrite면 다시 학습한다.
    run_m2ad_dinomaly.main(argv)
    assert len(calls) == 4
    run_m2ad_dinomaly.main(["finetune", "--light", "05", "--models", "full", "--overwrite", *common])
    assert len(calls) == 5
    # 출발 모델이 8-C 규칙의 그 모델이 아니면(점검용 짧은 학습 등) 학습 전에 멈춘다.
    start_meta["steps"] = 2
    with pytest.raises(ValueError, match="start model"):
        run_m2ad_dinomaly.main(["finetune", "--light", "06", "--models", "fold0", *common])
    start_meta["steps"] = 3000
    (start / "fold1.json").write_text(json.dumps({"arm": "e2-a", "model": "fold1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="start model"):
        run_m2ad_dinomaly.main(["finetune", "--light", "06", "--models", "fold1", *common])
    (start / "fold3.json").unlink()
    with pytest.raises(FileNotFoundError):
        run_m2ad_dinomaly.main(["finetune", "--light", "06", "--models", "fold3", *common])
    assert len(calls) == 5 and not (out / "ft" / "06" / "fold0.json").exists()
    # 같은 조명을 두 번 적거나 기준 조명을 주면 명령줄이 거절한다.
    for light in (["06", "06"], ["01"]):
        with pytest.raises(SystemExit):
            run_m2ad_dinomaly.main(["finetune", "--light", *light, *common])


# 8-C 저장물(세 팔의 학습 시편 점수와 봉인 test 점수)을 가짜 모델로 만든다. 8-D 분석은 여기서 E0 비교
# 팔(임계값만 T30, I01)을 다시 만든다. predict와 _load는 부르는 쪽이 가짜로 바꿔 둔다.
def _dms_outputs(out):
    for arm in run_m2ad_dinomaly.ARMS:
        (out / arm).mkdir(parents=True, exist_ok=True)
        for name in run_m2ad_dinomaly.MODELS:
            info = {
                "model": name,
                "train_images": 1,
                "final_loss": 0.1,
                "skipped_steps": 0,
                "peak_vram_mb": 0,
                "train_s": 1,
            }
            (out / arm / f"{name}.json").write_text(json.dumps(info), encoding="utf-8")
        run_m2ad_dinomaly.main(["score-train", "--arm", arm, "--out", str(out), "--device", "cpu"])
    run_m2ad_dinomaly.main(
        ["test", "--out", str(out), "--device", "cpu", "--allow-test", "--stage", "8-m2ad"]
    )


def _ledger_lines(path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    )


# score-finetune, test-finetune과 두 분석(dms-ft-train, dms-ft)의 경로를 가짜 모델로 돌린다. 가짜 점수는 밝기
# 평균에 모델마다 다른 상수를 더한 것이라(겹 f 모델은 −0.001 × (f + 1)) 각 학습 시편이 자기 겹 모델로
# 채점됐는지 값으로 확인할 수 있다. score-finetune은 학습 시편만 읽고 장부에 줄을 남기지 않으며 점수 모양은
# [검사기 6, (I01, L), 시편 30]이다. test-finetune은 플래그·모든 모델·모든 학습 시편 점수가 있어야 돌고,
# 장부에 한 줄 (stage 8-m2ad-ft, config m2ad-dms-280-car-ft) 남긴 뒤 아홉 FT 전체 모델로 I01과 L test를
# 채점한다. 분석의 E0 비교 팔은 8-C 분석(dms_test_report)의 임계값만 T30·I01 행과 같은 추출에서 같은 값이어야
# 하고, 기록된 8-C 리포트가 있으면 개수를 대조해 다르면 멈춘다.
def test_finetune_scoring_the_sealed_test_and_the_reports(
    tmp_path, monkeypatch, temporary_ledger, stage8d_ref
):
    root, rows = stage8d_ref
    cache = FakeCache()
    monkeypatch.setattr(run_m2ad_dinomaly, "read_meta", lambda path: rows)
    monkeypatch.setattr(run_m2ad_dinomaly, "M2adCache", lambda root, size: cache)
    monkeypatch.setattr(run_m2ad_dinomaly, "_load", lambda out_dir, name, device: (out_dir.name, name))
    monkeypatch.setattr(run_dinomaly, "encoder_weights", lambda encoder: {"hf_hub_id": "fake"})
    offsets = {"e0": 0.0, "e2-a": -0.05, "e2-b": -0.05}

    def shift(folder: str, name: str) -> float:
        if folder in offsets:
            return offsets[folder]
        return -0.02 - (0.001 * (int(name.removeprefix("fold")) + 1) if name.startswith("fold") else 0.0)

    def brightness(images):
        return 1.0 - images.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0

    def fake_predict(model, images, **kwargs):
        return _Scores((brightness(images) + shift(*model)).astype(np.float32))

    monkeypatch.setattr(run_dinomaly, "predict", fake_predict)
    out = tmp_path / "outputs" / "m2ad-dms"
    _dms_outputs(out)
    ft = out / "ft"
    for light in FT_LIGHTS:
        (ft / light).mkdir(parents=True)
        for name in run_m2ad_dinomaly.MODELS:
            info = {
                "model": name,
                "train_images": 1,
                "final_loss": 0.1,
                "skipped_steps": 0,
                "peak_vram_mb": 10.0,
                "train_s": 2.0,
                "start": {"sha256": "0" * 64},
            }
            (ft / light / f"{name}.json").write_text(json.dumps(info), encoding="utf-8")
    before = len(_ledger_lines(temporary_ledger))
    cache.splits.clear()
    run_m2ad_dinomaly.main(["score-finetune", "--light", *FT_LIGHTS, "--out", str(out), "--device", "cpu"])
    assert set(cache.splits) == {"train"} and len(_ledger_lines(temporary_ledger)) == before
    plan = run_m2ad_dinomaly.train_plan(rows)
    for light in FT_LIGHTS:
        z = load(ft / light / "train_scores.npz")
        assert z["scores"].shape == (6, 2, N_TRAIN) and np.isfinite(z["scores"]).all()
        assert z["lights"].tolist() == ["01", light] and z["inspectors"].tolist() == INSPECTOR_NAMES
        assert z["specimens"].tolist() == [plan[k]["names"] for k in run_m2ad_dinomaly.INSPECTORS]
        assert np.array_equal(z["folds"], np.stack([plan[k]["folds"] for k in run_m2ad_dinomaly.INSPECTORS]))
    z = load(ft / "07" / "train_scores.npz")
    key = run_m2ad_dinomaly.INSPECTORS[4]
    for li, light in enumerate(("01", "07")):
        for j, row in enumerate(plan[key]["rows"][light]):
            expected = brightness(cache.images([row]))[0] - 0.02 - 0.001 * (int(plan[key]["folds"][j]) + 1)
            assert z["scores"][4, li, j] == pytest.approx(expected, rel=1e-6)

    flags = ["--allow-test", "--stage", "8-m2ad-ft"]
    test_argv = ["test-finetune", "--out", str(out), "--device", "cpu"]
    with pytest.raises(SystemExit):
        run_m2ad_dinomaly.main(test_argv)
    missing = ft / "07" / "train_scores.npz"
    kept = missing.read_bytes()
    missing.unlink()
    with pytest.raises(SystemExit):
        run_m2ad_dinomaly.main([*test_argv, *flags])
    missing.write_bytes(kept)
    (ft / "09" / "fold4.json").rename(ft / "09" / "fold4.json.bak")
    with pytest.raises(SystemExit):
        run_m2ad_dinomaly.main([*test_argv, *flags])
    (ft / "09" / "fold4.json.bak").rename(ft / "09" / "fold4.json")
    assert len(_ledger_lines(temporary_ledger)) == before and not (ft / "test").exists()
    cache.splits.clear()
    run_m2ad_dinomaly.main([*test_argv, *flags])
    lines = _ledger_lines(temporary_ledger)
    assert len(lines) == before + 1
    assert (lines[-1]["stage"], lines[-1]["config"]) == ("8-m2ad-ft", "m2ad-dms-280-car-ft")
    assert set(cache.splits) == {"test"}
    for light in FT_LIGHTS:
        for name in INSPECTOR_NAMES:
            z = load(ft / "test" / f"{light}_{name}.npz")
            assert z["scores"].shape == z["labels"].shape == (2, len(z["specimens"]))
            assert z["lights"].tolist() == ["01", light]
    z = load(ft / "test" / "04_Bird_120.npz")
    category, view = "Bird", "120"
    test_rows, _ = run_m2ad_enrol._test_rows(rows, category, view)
    assert np.allclose(z["scores"][1], brightness(cache.images(test_rows["04"])) - 0.02, rtol=1e-6)
    run = json.loads((ft / "test" / "run.json").read_text(encoding="utf-8"))
    assert run["lights"] == list(FT_LIGHTS) and run["ledger_config"] == "m2ad-dms-280-car-ft"
    assert run["inspectors"] == INSPECTOR_NAMES and run["model"] == "full"
    with pytest.raises(SystemExit):  # 끝난 실행이 있으면 --overwrite 없이 다시 읽지 않는다
        run_m2ad_dinomaly.main([*test_argv, *flags])
    assert len(_ledger_lines(temporary_ledger)) == before + 1

    # 분석. 3-B 실행은 모듈 고정 자료(root/m2ad-p0)를 쓴다.
    monkeypatch.setattr(paths, "OUTPUTS", root)
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    record = analyze_stage8.dms_ft_train_report(out)
    per = record["lights"]["05"]
    assert set(per["models"]) == set(run_m2ad_dinomaly.MODELS) and record["training"]["models"] == 54
    first = per["thresholds"][0]
    assert first["inspector"] == "Motor_000"
    assert (first["light"]["rank"], first["light"]["n"], first["light"]["guaranteed"]) == (30, 30, True)
    assert (first["reference"]["rank"], first["reference"]["n"]) == (30, 30)
    assert (first["mixed"]["rank"], first["mixed"]["n"], first["mixed"]["guaranteed"]) == (58, 60, True)
    assert per["light_at_reference"]["n"] == 6 * N_TRAIN
    assert record["total"]["light_at_reference"]["n"] == 9 * 6 * N_TRAIN
    assert record["total"]["mixed_reference"]["n"] == 9 * 6 * N_TRAIN
    # E0 짝은 8-C 학습 시편 기록과 같은 정의(E0 겹 모델 점수, E0의 I01 임계값)라 그 조명별 수와 같다.
    c8_train = analyze_stage8.dms_train_report(out)
    for light in FT_LIGHTS:
        e0_light = record["lights"][light]["e0_light_at_reference"]
        assert e0_light["false_positives"] == c8_train["arms"]["e0"]["per_light"][light]["false_positives"]
    # 가짜 FT 점수는 8-C E0보다 0.02 이상 낮으므로 같은 I01 임계값에서 L 학습 시편 오검출이 늘 수 없다.
    total = record["total"]
    assert total["light_at_reference"]["false_positives"] <= total["e0_light_at_reference"]["false_positives"]

    report = analyze_stage8.dms_ft_report(out, n_boot=30)
    c8 = analyze_stage8.dms_test_report(out, n_boot=30)
    for k in ("false_positives", "n_normal", "detected", "n_defect", "fpr", "tpr", "fpr_ci", "tpr_ci"):
        assert report["e0_t30"][k] == c8["threshold_only"]["e0:T30"][k]
        assert report["e0_reference"][k] == c8["candidates"]["e0"]["reference"][k]
    assert report["e0_t30"]["auroc"] == pytest.approx(c8["candidates"]["e0"]["unseen_auroc_mean"], rel=1e-12)
    assert report["ft"]["n_normal"] == c8["candidates"]["e0"]["unseen"]["n_normal"]
    assert report["ft"]["n_defect"] == c8["candidates"]["e0"]["unseen"]["n_defect"]
    assert report["ft_reference"]["n_defect"] == 9 * c8["candidates"]["e0"]["reference"]["n_defect"]
    assert report["hypotheses"]["H28"] in ("지지", "기각", "판정 불가")
    assert report["h28"]["verdict"] == report["hypotheses"]["H28"]
    assert isinstance(report["h28"]["forgetting"], bool)
    assert set(report["per_light"]) == set(FT_LIGHTS)
    assert report["thresholds"]["light"] == {"cells": 54, "guaranteed": 54, "n": [30], "rank": [30]}
    assert report["thresholds"]["mixed"] == {"cells": 54, "guaranteed": 54, "n": [60], "rank": [58]}
    assert report["side_by_side"]["dms_e2_unseen"] is None and report["recorded_check"] is None
    # P0 뱅크+임계값 n = 30(판정 없이 나란히)은 3-B 저장값에서 같은 추출로 낸다.
    ref = analyze_m2ad.load(root / "m2ad-p0")
    boot = analyze_m2ad.ClusterBootstrap(ref, 30, 0)
    bank30 = analyze_stage8._rates(boot, [analyze_m2ad.recal_cell(ref, f"{light}:30") for light in FT_LIGHTS])
    for k in ("false_positives", "n_normal", "detected", "n_defect"):
        assert report["side_by_side"]["p0_bank30"][k] == bank30[k]

    # 8-C 리포트가 기록으로 있으면 비교 팔의 개수를 대조한다: 같으면 통과, 다르면 결과를 내지 않는다.
    recorded = tmp_path / "reports" / "stage8" / "dms-test.json"
    analyze_stage8._write(recorded, c8)
    report = analyze_stage8.dms_ft_report(out, n_boot=30)
    assert report["recorded_check"]["ok"] is True
    assert report["side_by_side"]["dms_e2_unseen"]["fpr"] == c8["candidates"]["e2"]["unseen"]["fpr"]
    assert analyze_stage8.check_e0_comparison(out)["ok"] is True
    tampered = json.loads(recorded.read_text(encoding="utf-8"))
    tampered["threshold_only"]["e0:T30"]["detected"] += 1
    recorded.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="do not reproduce"):
        analyze_stage8.dms_ft_report(out, n_boot=30)
    with pytest.raises(ValueError, match="do not reproduce"):
        analyze_stage8.check_e0_comparison(out)

    # 명령줄: 두 리포트를 reports/stage8/에 쓴다.
    recorded.unlink()
    monkeypatch.setattr(analyze_stage8, "_dms_dir", lambda: out)
    analyze_stage8.main(["dms-ft-train"])
    analyze_stage8.main(["dms-ft"])
    written = json.loads((tmp_path / "reports" / "stage8" / "dms-ft.json").read_text(encoding="utf-8"))
    assert written["hypotheses"]["H28"] in ("지지", "기각", "판정 불가") and written["n_boot"] == 2000
    assert (tmp_path / "reports" / "stage8" / "dms-ft-train.json").exists()


# 판정선 단위 확인(H28, docs/experiments.md 8-D): 차이 하한이 +20%p보다 크고 FT 오검출률 상한이 10% 이하이면
# 지지, 차이 상한이 +20%p보다 작으면 기각, 나머지는 판정 불가. 경계와 같은 값은 넘지 않은 것으로 본다(하한
# 0.20은 '크다'가 아니고 상한 0.20은 '작다'가 아니다). 오검출률 상한은 10%와 같아도 된다. 망각 가드는 I01
# 검출률 차이 점 추정이 −10%p보다 낮을 때만 참이다(−10%p 그대로는 망각이 아니다, H18·H24 가드와 같은 선).
def test_h28_verdict_lines_and_the_forgetting_guard():
    verdict = analyze_stage8._verdict_h28
    assert verdict(0.21, 0.35, 0.08) == "지지"
    assert verdict(0.21, 0.35, 0.10) == "지지"
    assert verdict(0.21, 0.35, 0.11) == "판정 불가"
    assert verdict(0.20, 0.35, 0.05) == "판정 불가"
    assert verdict(0.15, 0.25, 0.05) == "판정 불가"
    assert verdict(0.05, 0.19, 0.05) == "기각"
    assert verdict(0.05, 0.19, 0.50) == "기각"
    assert verdict(0.10, 0.20, 0.05) == "판정 불가"
    assert analyze_stage8.H28_MARGIN == 0.20 and analyze_stage8.LIMIT == 0.10
    forgetting = analyze_stage8._forgetting
    assert forgetting(-0.11) is True
    assert forgetting(-0.10) is False and forgetting(-0.05) is False and forgetting(0.02) is False


# 합성 저장물로 dms-ft 리포트의 판정을 끝까지 확인한다. 학습 시편 점수는 모두 0.5라 조건별·혼합 임계값이 모두
# 0.5(최댓값)다. test 정상은 0.4(오검출 없음), 결함(라벨 1)은 팔마다 정한 점수다: 0.9면 모두 검출, 0.4면 모두
# 놓친다. 시편·라벨은 가짜 3-B 실행에서 가져와 부트스트랩과 맞춘다(라벨 −1인 이미지는 어느 쪽에도 세지
# 않는다).
def _synthetic_ft(out, ref, *, e0_defect, ft_defect, ft_normal=0.4, e0_ref_defect=0.9, ft_ref_defect=0.9):
    names = list(ref["inspectors"])
    specimens = np.array([[f"{2 * i:03d}" for i in range(N_TRAIN)]] * len(names))
    folds = np.zeros(specimens.shape, dtype=np.int64)
    (out / "e0").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "e0" / "train_scores.npz",
        inspectors=np.array(names),
        lights=np.array(ILLUMINATIONS),
        enrolled=np.array([light == "01" for light in ILLUMINATIONS]),
        specimens=specimens,
        folds=folds,
        scores=np.full((len(names), len(ILLUMINATIONS), N_TRAIN), 0.5, dtype=np.float32),
    )

    def scores_for(labels, normal, defect):
        return np.where(labels == 1, defect, normal).astype(np.float32)

    for name, arr in ref["inspectors"].items():
        labels = arr["labels"][arr["conditions"].tolist().index("S")]
        e0 = [
            scores_for(labels, 0.4, e0_ref_defect if light == "01" else e0_defect) for light in ILLUMINATIONS
        ]
        np.savez(
            out / "test" / f"e0_{name}.npz",
            specimens=arr["specimens"],
            object_anomaly=arr["object_anomaly"],
            lights=np.array(ILLUMINATIONS),
            scores=np.stack(e0),
            labels=np.stack([labels] * len(ILLUMINATIONS)),
        )
    for light in FT_LIGHTS:
        (out / "ft" / light).mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "ft" / light / "train_scores.npz",
            inspectors=np.array(names),
            lights=np.array(["01", light]),
            specimens=specimens,
            folds=folds,
            scores=np.full((len(names), 2, N_TRAIN), 0.5, dtype=np.float32),
        )
    (out / "ft" / "test").mkdir(parents=True, exist_ok=True)
    for light in FT_LIGHTS:
        for name, arr in ref["inspectors"].items():
            labels = arr["labels"][arr["conditions"].tolist().index("S")]
            np.savez(
                out / "ft" / "test" / f"{light}_{name}.npz",
                specimens=arr["specimens"],
                object_anomaly=arr["object_anomaly"],
                lights=np.array(["01", light]),
                scores=np.stack(
                    [scores_for(labels, 0.4, ft_ref_defect), scores_for(labels, ft_normal, ft_defect)]
                ),
                labels=np.stack([labels, labels]),
            )


# 합성 자료로 세 판정과 가드가 리포트에 그대로 나오는지 본다. (가) FT가 결함을 모두 잡고(검출률 100%) E0 T30은
# 모두 놓치면(0%) 차이가 +100%p, 구간 [1, 1]이고 FT 오검출률 0%라 지지. (나) FT도 모두 놓치면 차이 0이라 기각.
# (다) FT가 모두 잡지만 정상도 모두 불량으로 판정하면(오검출률 100%) 차이 하한이 +20%p를 넘어도 판정 불가.
# (라) FT의 I01 결함을 모두 놓치고 E0은 모두 잡으면 I01 검출률 차이 −100%p로 망각. 판정은 망각과 따로 적는다.
@pytest.mark.parametrize(
    ("e0_defect", "ft_defect", "ft_normal", "ft_ref_defect", "verdict", "forgetting"),
    [
        (0.4, 0.9, 0.4, 0.9, "지지", False),
        (0.4, 0.4, 0.4, 0.9, "기각", False),
        (0.4, 0.9, 0.9, 0.9, "판정 불가", False),
        (0.4, 0.9, 0.4, 0.4, "지지", True),
    ],
)
def test_dms_ft_report_verdicts_on_synthetic_scores(
    tmp_path, monkeypatch, stage8d_ref, e0_defect, ft_defect, ft_normal, ft_ref_defect, verdict, forgetting
):
    root, _ = stage8d_ref
    monkeypatch.setattr(paths, "OUTPUTS", root)
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    ref = analyze_m2ad.load(root / "m2ad-p0")
    out = tmp_path / "m2ad-dms"
    _synthetic_ft(
        out, ref, e0_defect=e0_defect, ft_defect=ft_defect, ft_normal=ft_normal, ft_ref_defect=ft_ref_defect
    )
    report = analyze_stage8.dms_ft_report(out, n_boot=50)
    h28 = report["h28"]
    assert report["hypotheses"] == {"H28": verdict} and h28["verdict"] == verdict
    assert h28["forgetting"] is forgetting
    assert report["e0_t30"]["tpr"] == 0.0 and report["e0_reference"]["tpr"] == 1.0
    assert report["ft"]["tpr"] == (1.0 if ft_defect > 0.5 else 0.0)
    assert report["ft"]["fpr"] == (1.0 if ft_normal > 0.5 else 0.0)
    assert h28["d_tpr"] == report["ft"]["tpr"] - report["e0_t30"]["tpr"]
    assert h28["d_reference_tpr"] == report["ft_reference"]["tpr"] - report["e0_reference"]["tpr"]
    assert h28["margin"] == 0.20 and h28["guard"] == -0.10 and h28["fpr_limit"] == 0.10
    assert h28["ft_fpr_upper_at_most_10pct"] is (ft_normal < 0.5)

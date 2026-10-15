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

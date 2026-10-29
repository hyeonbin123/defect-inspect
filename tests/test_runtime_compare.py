# [파일 역할] CPU 런타임 대조(onnxruntime R0·R1 대 OpenVINO R2) 모듈 `defect_inspect.runtime_compare`를
# 검증하는 테스트 묶음이다. 블록 순서 돌리기, 점수 동일성 관문 (가)~(라), 지연 요약과 H22 판정, 한가함 판정과
# 기다리기, CPU 사용률 계산(Windows GetSystemTimes와 Linux /proc/stat), 리포트 기록 시각의 형식, latency
# 리포트의 한가함·실행 시각 기록, 호출 시간 재기, 세 팔이 같은 ONNX 파일을 읽는지를 확인한다.
# [왜 필요한가] 이 대조는 규칙(docs/experiments.md 'CPU 런타임 대조', 2026-10-04 측정 전 등록)에 정한 관문과
# 기준(10% 이득)으로 판정한다. 관문 계산이나 블록 순서가 틀리면 판정 자체가 틀리므로, 실제 모델과 데이터 없이
# 작은 점수 배열과 가짜 함수로 규칙의 각 조건을 하나씩 시험한다.
# [설계 메모] torch가 필요 없어 CI(Linux)에서도 돈다. 그래서 CPU 사용률을 실제로 읽는 테스트는 이
# PC(Windows)와 CI에서 서로 다른 경로를 지난다. 마지막 테스트만 onnx·onnxruntime·openvino가 있을 때 아주 작은
# ONNX 모델을 만들어 세 팔에 실제로 돌린다.

import json
import os
import re
import subprocess
import types
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

# inspector는 작은 아티팩트의 meta.json 버전 값을, calibrate.conformal_threshold는 기대 임계값(목표 5%)을
# 만드는 데 쓴다. bench는 latency 명령이 함수 안에서 가져오는 리사이즈 시간·CPU 이름 함수를 가짜로 바꾸려고
# 가져온다.
from defect_inspect import bench, inspector, runtime_compare
from defect_inspect.calibrate import conformal_threshold

# 리포트 기록 시각의 형식: 초 단위 ISO 8601에 UTC 오프셋(예: 2026-10-29T21:05:13+09:00, CI에서는 +00:00).
ISO_WITH_OFFSET = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")


# 지연 측정의 블록 순서: 블록 b는 팔 목록을 b만큼 돌린다. 세 블록이면 모든 팔이 모든 자리(처음·가운데·끝)를 한
# 번씩 맡아, 실행 순서에 따른 영향이 특정 팔에 몰리지 않는다(규칙의 블록 1: R0 R1 R2, 2: R1 R2 R0, 3: R2 R0
# R1). 팔이 둘이면 번갈아 돈다.
def test_block_orders_rotate_every_arm_through_every_position():
    orders = runtime_compare.block_orders(["R0", "R1", "R2"], 3)
    assert orders == [["R0", "R1", "R2"], ["R1", "R2", "R0"], ["R2", "R0", "R1"]]
    for position in range(3):
        assert sorted(order[position] for order in orders) == ["R0", "R1", "R2"]
    assert runtime_compare.block_orders(["R0", "R2"], 3) == [["R0", "R2"], ["R2", "R0"], ["R0", "R2"]]


# 시험용 점수: 정상(평균 1.0)과 결함(평균 1.6) 점수를 고정 시드 정규분포로 만들고 라벨(0 정상, 1 결함)을
# 붙인다. R0의 점수 역할을 한다.
def _scores(seed=0, normal=60, defect=12):
    rng = np.random.default_rng(seed)
    ref = np.concatenate([rng.normal(1.0, 0.1, normal), rng.normal(1.6, 0.2, defect)]).astype(np.float32)
    labels = np.array([0] * normal + [1] * defect, dtype=np.int8)
    return ref, labels


# R0와 똑같은 점수를 내는 팔은 모든 확인을 통과해야 한다: 상대 차이 0, AUROC 같음, 겹 0 정상 표시 같음, 저장
# 임계값 판정 뒤집힘 0. 관문을 범주 두 개로 모으면 (라)가 두 범주 모두에서 같다고 센다.
def test_identical_scores_pass_every_check():
    ref, labels = _scores()
    stored = conformal_threshold(ref[labels == 0], 0.05).value
    cat = runtime_compare.category_parity(ref, ref.copy(), labels, stored)
    assert cat["max_rel_diff"] == 0.0 and cat["fold0_flags_same"]
    assert cat["auroc_ref"] == cat["auroc_arm"] and cat["verdict_flips_stored"] == 0
    assert cat["fold0_flags_ref"] == int((ref[labels == 0] > stored).sum())
    gate = runtime_compare.parity_gate({"a": cat, "b": cat}, map_abs=0.0)
    assert gate["pass"] and gate["checks"]["d_fold0_flags"]["same_in"] == 2


# 규칙에서 (라)를 '각 팔 자신의 임계값'으로 정한 이유를 그대로 시험한다. 임계값 자리의 정상 이미지 점수를
# 마지막 비트(1 ulp)만 올리면, 그 팔 자신의 임계값도 함께 움직여 표시는 같지만, R0 점수로 정한 저장 임계값과
# 비교하면 그 한 장의 판정이 뒤집힌다. 관문은 통과하고 뒤집힌 수는 판정 없이 기록된다.
def test_last_bit_noise_passes_the_own_threshold_check_but_can_flip_the_stored_one():
    ref, labels = _scores(seed=1)
    normal = np.flatnonzero(labels == 0)
    thr = conformal_threshold(ref[normal], 0.05)
    at_threshold = normal[np.argmin(np.abs(ref[normal] - thr.value))]
    arm = ref.copy()
    arm[at_threshold] = np.nextafter(arm[at_threshold], np.float32(np.inf))  # 1 ulp 올림
    cat = runtime_compare.category_parity(ref, arm, labels, thr.value)
    assert cat["fold0_flags_same"]  # 팔 자신의 임계값이 그 이미지와 함께 움직인다
    assert cat["verdict_flips_stored"] == 1  # R0의 임계값은 이제 그 점수보다 1 ulp 아래
    assert runtime_compare.parity_gate({"a": cat}, map_abs=1e-7)["pass"]


# 관문의 네 확인이 각각 따로 실패할 수 있는지 본다. (가) 점수를 0.2% 키우면 상대 차이 1e-3을 넘는다. (나) 맵
# 절대 차이 2e-3은 1e-3을 넘는다. (다) 가장 높은 정상과 가장 낮은 결함의 점수를 맞바꾸면 AUROC가 0.01%p 넘게
# 바뀐다. (라) 임계값 바로 아래·위 정상 두 장의 점수를 맞바꾸면 AUROC 등은 거의 그대로여도 표시되는 정상이
# 달라진다. 확인 하나가 다른 확인에 가려 통과하는 버그를 막는다.
def test_each_check_can_fail_on_its_own():
    ref, labels = _scores(seed=2)
    stored = conformal_threshold(ref[labels == 0], 0.05).value

    rel = runtime_compare.category_parity(ref, ref * np.float32(1.002), labels, stored)
    gate = runtime_compare.parity_gate({"a": rel}, map_abs=0.0)
    assert not gate["checks"]["a_score_rel"]["pass"] and not gate["pass"]

    same = runtime_compare.category_parity(ref, ref.copy(), labels, stored)
    gate = runtime_compare.parity_gate({"a": same}, map_abs=2e-3)
    assert not gate["checks"]["b_map_abs"]["pass"] and not gate["pass"]

    # 정상 하나와 결함 하나의 점수를 맞바꾼다: AUROC 변화는 작지만 상대 차이는 크다.
    arm = ref.copy()
    normal_top = np.flatnonzero(labels == 0)[np.argmax(ref[labels == 0])]
    defect_low = np.flatnonzero(labels == 1)[np.argmin(ref[labels == 1])]
    arm[[normal_top, defect_low]] = arm[[defect_low, normal_top]]
    gate = runtime_compare.parity_gate({"a": runtime_compare.category_parity(ref, arm, labels, stored)}, 0.0)
    assert not gate["checks"]["c_macro_auroc_pp"]["pass"]

    # 겹 0 정상 하나가 임계값을 넘나든다: (가)~(다)는 통과할 상황에서도 (라)는 실패한다.
    normal = np.flatnonzero(labels == 0)
    thr = conformal_threshold(ref[normal], 0.05).value
    order = normal[np.argsort(ref[normal])]
    below = order[np.searchsorted(ref[order], thr) - 1]  # 임계값을 넘지 않는 가장 큰 정상
    above = order[np.searchsorted(ref[order], thr, side="right")]  # 임계값을 넘는 가장 작은 정상
    arm = ref.copy()
    arm[below], arm[above] = ref[above], ref[below]
    cat = runtime_compare.category_parity(ref, arm, labels, stored)
    assert not cat["fold0_flags_same"]
    assert not runtime_compare.parity_gate({"a": cat}, 0.0)["checks"]["d_fold0_flags"]["pass"]


# 지연 요약: 팔의 지연은 모든 블록의 시간 기록을 합친 중앙값에 리사이즈 시간을 더한 값이고(블록 중앙값의
# 평균이 아님), 블록별 중앙값도 함께 남긴다. 속도 이득은 1 - (팔 지연 / R0 지연)이다(규칙 (2)).
def test_summarise_latency_pools_blocks_and_adds_the_resize():
    runs = [
        {"arm": "R0", "block": 1, "ms": [100.0, 102.0, 104.0]},
        {"arm": "R2", "block": 1, "ms": [80.0, 81.0, 82.0]},
        {"arm": "R2", "block": 2, "ms": [90.0, 91.0, 92.0]},
        {"arm": "R0", "block": 2, "ms": [101.0, 103.0, 105.0]},
    ]
    out = runtime_compare.summarise_latency(runs, resize_ms=8.0, arms=["R0", "R2"])
    assert out["R0"]["inference_median_ms"] == 102.5 and out["R0"]["timed_calls"] == 6
    assert out["R0"]["total_with_resize_ms"] == 110.5
    assert out["R2"]["block_medians_ms"] == [81.0, 91.0]
    assert out["R2"]["total_with_resize_ms"] == 8.0 + 86.0
    assert out["R2"]["speedup_vs_r0"] == pytest.approx(1 - 94.0 / 110.5)
    assert out["R0"]["speedup_vs_r0"] == 0.0


# H22 판정: 팔이 관문을 통과하고, R0가 6단계 기록과 같은 점수를 냈으며(환경이 바뀌지 않았다는 전제), 속도
# 이득이 10% 이상일 때만 후보다. 셋 중 하나라도 어긋나면 후보가 아니다. 관문 리포트가 없으면 관문 결과를
# None(모름)으로 둔다. R0 자신은 판정 대상이 아니다.
def test_h22_needs_parity_the_record_match_and_ten_percent():
    latency = {
        "R0": {"speedup_vs_r0": 0.0},
        "R1": {"speedup_vs_r0": -0.05},
        "R2": {"speedup_vs_r0": 0.10},
    }
    parity = {"reference": {"matches_record": True}, "arms": {"R1": {"pass": True}, "R2": {"pass": True}}}
    out = runtime_compare.h22(parity, latency)
    assert out["R2"]["candidate"] and not out["R1"]["candidate"] and "R0" not in out
    failed = {"reference": {"matches_record": True}, "arms": {"R2": {"pass": False}}}
    assert not runtime_compare.h22(failed, latency)["R2"]["candidate"]
    off_record = {"reference": {"matches_record": False}, "arms": {"R2": {"pass": True}}}
    assert not runtime_compare.h22(off_record, latency)["R2"]["candidate"]
    assert runtime_compare.h22(None, latency)["R2"]["parity_pass"] is None


# 한가함 규칙: 1분 평균 CPU 10% 미만이고 GPU 10% 미만(GPU를 읽을 수 없으면 CPU만 본다)이어야 한다. CPU를 읽을
# 수 없으면 한가하지 않은 것으로 본다. wait_idle은 1분씩 재다가 한가해지면 멈추고, 최대 횟수가 지나면 한가하지
# 않음과 기록을 돌려준다. 돌려주는 dict는 리포트의 idle 절이 되며, CPU를 읽은 방법(cpu_probe)을 함께 담는다.
# 마지막 부분은 각 기록의 끝 시각(end)을 그 1분을 잰 뒤에 읽는지 확인한다. 2026-10-04 검토에서 측정 시각 기록
# 순서를 고친 것(runtime_compare.wait_idle의 2026-10-04 수정)의 회귀를 막는다. CPU를 읽을 수 있는지(probe)는
# 1분을 재기 전에 한 번만 본다.
def test_idle_rule_and_wait():
    assert runtime_compare.is_idle({"cpu_pct": 9.9, "gpu_pct": 9.9})
    assert not runtime_compare.is_idle({"cpu_pct": 10.0, "gpu_pct": 0.0})
    assert not runtime_compare.is_idle({"cpu_pct": 2.0, "gpu_pct": 35.0})
    assert runtime_compare.is_idle({"cpu_pct": 2.0, "gpu_pct": None})  # 읽을 GPU가 없음
    assert not runtime_compare.is_idle({"cpu_pct": None, "gpu_pct": 0.0})

    minutes = iter([{"cpu_pct": 40.0, "gpu_pct": 0.0}, {"cpu_pct": 5.0, "gpu_pct": 1.0}])
    result = runtime_compare.wait_idle(
        5, minute=lambda: next(minutes), clock=lambda: "t", probe=lambda: "GetSystemTimes"
    )
    assert result["reached"] and result["cpu_probe"] == "GetSystemTimes"
    assert [r["idle"] for r in result["wait"]] == [False, True]
    assert [(r["minute"], r["end"]) for r in result["wait"]] == [(1, "t"), (2, "t")]
    result = runtime_compare.wait_idle(
        2, minute=lambda: {"cpu_pct": 50.0, "gpu_pct": 0.0}, clock=lambda: "t", probe=lambda: "/proc/stat"
    )
    assert not result["reached"] and result["cpu_probe"] == "/proc/stat" and len(result["wait"]) == 2

    # `end`는 그 1분을 잰 뒤에 읽는다. CPU를 읽을 수 있는지는 그 전에 한 번 본다.
    events = []

    def probe():
        events.append("probe")
        return "GetSystemTimes"

    def minute():
        events.append("minute")
        return {"cpu_pct": 1.0, "gpu_pct": 0.0}

    def clock():
        events.append("clock")
        return "t"

    runtime_compare.wait_idle(1, minute=minute, clock=clock, probe=probe)
    assert events == ["probe", "minute", "clock"]


# CPU 사용률 계산: (유휴 시간, 전체 시간) 두 측정 사이의 증가분으로 100 × (1 - 유휴 증가 / 전체 증가)를 낸다.
# 측정이 없거나 시간이 흐르지 않았으면 None이다. cpu_percent는 실제로 짧게 재 보고(Windows도 Linux도 아니면
# None) 범위만 확인한다.
def test_busy_percent():
    assert runtime_compare.busy_percent((50, 100), (80, 200)) == pytest.approx(70.0)
    assert runtime_compare.busy_percent(None, (1, 2)) is None
    assert runtime_compare.busy_percent((5, 10), (5, 10)) is None
    value = runtime_compare.cpu_percent(0.05)
    assert value is None or 0.0 <= value <= 100.0


# Linux의 CPU 사용률: /proc/stat 첫 줄(모든 CPU의 합계)에서 (유휴 + iowait, 합계) 틱을 읽는다. 열 순서는
# proc(5)의 user nice system idle iowait irq softirq steal guest guest_nice다. guest·guest_nice는 커널이
# user·nice에 이미 더해 두므로 합계에서 빼고 센다(psutil·htop과 같은 계산). 두 번 세면 아래 두 표본의 바쁨이
# 25%로 부풀려진다(guest 30틱이 user 60틱 안에 이미 들어 있다). 첫 표본은 WSL2 커널(6.18)의 실제 첫 줄이다.
# iowait 이후 열이 없는 옛 커널 형식도 읽고, cpu 줄이 아니거나(cpu0, intr) 열이 모자라거나 숫자가 아니면
# None이다. 파일에서는 첫 줄만 읽고, 파일이 없으면 None이다.
def test_proc_stat_first_line_gives_idle_and_total_ticks(tmp_path):
    sample = "cpu  19317 0 18284 3754496 125378 0 6083 0 0 0\n"
    assert runtime_compare.parse_proc_stat(sample) == (
        3754496 + 125378,
        19317 + 0 + 18284 + 3754496 + 125378 + 0 + 6083 + 0,
    )
    before = runtime_compare.parse_proc_stat("cpu  100 20 30 400 50 6 7 8 90 10")
    after = runtime_compare.parse_proc_stat("cpu  160 20 50 700 80 6 7 8 120 10")
    assert before == (450, 621) and after == (780, 1031)
    assert runtime_compare.busy_percent(before, after) == pytest.approx(100.0 * 80 / 410)
    assert runtime_compare.parse_proc_stat("cpu  100 20 30 400") == (400, 550)  # iowait 이후 열이 없는 형식
    for bad in ("", "\n", "cpu0 1 2 3 4 5", "intr 1 2 3 4 5", "cpu  1 2 3", "cpu  1 2 x 4 5"):
        assert runtime_compare.parse_proc_stat(bad) is None, bad

    stat = tmp_path / "stat"
    stat.write_text(sample + "cpu0 1 0 1 1 0 0 0 0 0 0\nintr 5 6 7\n", encoding="ascii")
    assert runtime_compare._proc_stat_times(stat) == runtime_compare.parse_proc_stat(sample)
    assert runtime_compare._proc_stat_times(tmp_path / "missing") is None


# 이 PC의 CPU 사용률을 실제로 읽는다. Windows는 GetSystemTimes, /proc/stat이 있는 곳(Linux, CI)은
# /proc/stat이고, 둘 다 없으면 unavailable이다. 읽은 값은 (유휴, 합계) 정수 쌍이고 유휴는 합계를 넘지 않는다.
def test_cpu_probe_names_how_this_platform_reads_the_cpu():
    if os.name == "nt":
        expected = "GetSystemTimes"
    else:
        expected = "/proc/stat" if Path("/proc/stat").exists() else "unavailable"
    assert runtime_compare.cpu_probe() == expected
    times = runtime_compare._system_times()
    if expected == "unavailable":
        assert times is None
    else:
        idle, total = times
        assert isinstance(idle, int) and isinstance(total, int) and 0 <= idle <= total


# CPU 사용률을 읽을 수 없는 플랫폼에서는 한가함을 판정할 수 없다. 예전에는 1분씩 --wait-max-min번(기본 60분)을
# 아무 알림 없이 기다린 뒤 '한가하지 않음'으로 쟀다. 이제 wait_idle은 1분 측정을 한 번도 하지 않고 곧바로
# 돌아오며, 돌려주는 기록(리포트의 idle 절)에 cpu_probe: unavailable이 남고 표준 오류에 알림이 나간다.
def test_wait_idle_returns_at_once_when_the_cpu_load_cannot_be_read(monkeypatch, capsys):
    monkeypatch.setattr(runtime_compare, "_system_times", lambda: None)

    def minute():
        raise AssertionError("a minute was measured although the CPU load cannot be read")

    result = runtime_compare.wait_idle(60, minute=minute, clock=lambda: "t")
    assert result == {"reached": False, "cpu_probe": "unavailable", "wait": []}
    assert "unavailable" in capsys.readouterr().err


# 리포트 기록 시각은 이 PC의 지역 시각에 UTC 오프셋을 붙인 ISO 8601이다. 2026-10-04 리포트는 오프셋 없는 지역
# 시각('2026-10-04 22:08:28')이라 UTC로 적는 장부와 맞춰 보기 어려웠다. 오프셋을 붙여 읽은 시각은 지금(UTC)과
# 몇 초 안이고, wait_idle의 기본 시계도 이 형식이다.
def test_report_times_carry_the_utc_offset():
    stamp = runtime_compare.now_iso()
    assert ISO_WITH_OFFSET.fullmatch(stamp), stamp
    assert abs((datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds()) < 5
    result = runtime_compare.wait_idle(
        1, minute=lambda: {"cpu_pct": 1.0, "gpu_pct": 0.0}, probe=lambda: "GetSystemTimes"
    )
    assert ISO_WITH_OFFSET.fullmatch(result["wait"][0]["end"]), result


# latency 명령의 리포트 배선을 실제 측정 없이 본다(5초 CPU, nvidia-smi, 리사이즈, 팔마다 새 프로세스, 세션 뒤
# 1분을 모두 가짜로 바꿔 순식간에 끝난다). CPU를 읽을 수 없으면 한가함을 기다리지 않고(1분 측정은 세션 뒤의
# 한 번뿐) idle 절에 cpu_probe: unavailable과 reached: false를 남긴다. 세션 시작 시각과 팔 실행마다 새
# 프로세스의 시작·끝 시각을 UTC 오프셋이 붙은 ISO 8601로 적는다. 2026-10-04 리포트에는 실행별 시각이 없어 블록
# 시각을 실행 시간의 합으로 추정해야 했다(docs/experiments.md CPU 런타임 대조 결과의 정정). --wait-max-min 1은
# 기다림을 잘못 고쳤을 때 테스트가 오래 매달리지 않게 하는 상한이다.
def test_latency_report_records_the_cpu_probe_and_each_run_time(tmp_path, monkeypatch):
    art = tmp_path / "art"
    (art / "pcb1").mkdir(parents=True)
    (art / "pcb1" / "meta.json").write_text(json.dumps({"img_size": 8}), encoding="utf-8")
    (art / "model_fp32.onnx").write_bytes(b"not a model: only its sha256 is recorded")
    after_minutes = []
    launched = []

    def after_minute():
        after_minutes.append(runtime_compare.now_iso())
        return {"cpu_pct": None, "gpu_pct": None, "gpu_samples": 0}

    def run_arm(cmd, **kwargs):
        arm = cmd[cmd.index("--arm") + 1]
        launched.append(arm)
        line = json.dumps({"arm": arm, "load_s": 0.1, "ms": [10.0, 11.0, 12.0], "runtime": "fake"})
        return subprocess.CompletedProcess(cmd, 0, stdout="loading\n" + line + "\n", stderr="")

    monkeypatch.setattr(runtime_compare, "_system_times", lambda: None)
    monkeypatch.setattr(runtime_compare, "cpu_percent", lambda seconds: None)
    monkeypatch.setattr(runtime_compare, "gpu_percent", lambda: None)
    monkeypatch.setattr(runtime_compare, "load_minute", after_minute)
    monkeypatch.setattr(runtime_compare, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(runtime_compare, "subprocess", types.SimpleNamespace(run=run_arm))
    monkeypatch.setattr(bench, "time_resize", lambda size: 8.0)
    monkeypatch.setattr(bench, "cpu_name", lambda: "test cpu")
    out = tmp_path / "latency.json"
    runtime_compare.main(
        ["latency", "--artifacts", str(art), "--arms", "R0", "R2", "--blocks", "2", "--wait-max-min", "1"]
        + ["--parity", str(tmp_path / "no-parity.json"), "--out", str(out)]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["idle"] == {
        "reached": False,
        "cpu_probe": "unavailable",
        "cpu_limit_pct": runtime_compare.CPU_IDLE_PCT,
        "gpu_limit_pct": runtime_compare.GPU_IDLE_PCT,
        "wait": [],
    }
    assert len(after_minutes) == 1 and launched == ["R0", "R2", "R2", "R0"]
    assert ISO_WITH_OFFSET.fullmatch(report["started"]), report["started"]
    session = datetime.fromisoformat(report["started"])
    previous_end = session
    for run in report["runs"]:
        assert ISO_WITH_OFFSET.fullmatch(run["started"]) and ISO_WITH_OFFSET.fullmatch(run["ended"]), run
        started, ended = datetime.fromisoformat(run["started"]), datetime.fromisoformat(run["ended"])
        assert previous_end <= started <= ended  # 실행은 차례로 돈다
        previous_end = ended
    assert previous_end <= datetime.fromisoformat(after_minutes[0])
    assert [(r["block"], r["arm"]) for r in report["runs"]] == [(1, "R0"), (1, "R2"), (2, "R2"), (2, "R0")]
    assert report["arms"]["R0"]["timed_calls"] == 6 and report["H22"]["R2"]["parity_pass"] is None


# time_calls는 워밍업 호출을 시간에서 빼고, 이미지를 순서대로 돌려 가며 쓴다(워밍업 2 + 측정 5 = 이미지
# 0,1,2,3,0,1,2). 반복 횟수가 0이면 ValueError다.
def test_time_calls_cycles_images_after_the_warmup():
    seen = []

    class Insp:
        def run(self, image):
            seen.append(int(image[0, 0, 0]))

    images = np.zeros((4, 2, 2, 3), np.uint8)
    images[:, 0, 0, 0] = np.arange(4)
    ms = runtime_compare.time_calls(Insp(), images, warmup=2, repeats=5)
    assert len(ms) == 5 and all(v >= 0 for v in ms)
    assert seen == [0, 1, 2, 3, 0, 1, 2]
    with pytest.raises(ValueError):
        runtime_compare.time_calls(Insp(), images, warmup=0, repeats=0)


# 작은 ONNX 모델의 입력 크기(8x8).
SIZE = 8


# 세 팔이 모두 읽을 수 있는 아주 작은 ONNX 모델을 만든다. 입력 image [1, 3, S, S], 출력 score [1]와 map [1, S,
# S]로, ReconstructionInspector가 확인하는 입출력 이름·모양에 맞춘다. onnx 패키지가 없으면 테스트를 건너뛴다.
def _tiny_model(path):
    """image [1, 3, S, S] -> map = 채널 평균 [1, S, S], score = 맵 최댓값 [1] (opset 13)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("ReduceMean", ["image"], ["map"], axes=[1], keepdims=0),
        helper.make_node("ReduceMax", ["map"], ["score"], axes=[1, 2], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [
            helper.make_tensor_value_info("score", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("map", TensorProto.FLOAT, [1, SIZE, SIZE]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


# 세 팔(R0, R1 onnxruntime, R2 OpenVINO)이 같은 아티팩트 폴더(meta.json + 상위 폴더의 model_fp32.onnx)를 읽어
# FP32 검사기를 만들고, 같은 이미지에서 점수와 맵이 R0와 1e-6 안에서 같은지 본다. arm_info가 런타임 이름을
# 맞게 적는지, OpenVINO가 규칙대로 NUM_STREAMS=1로 컴파일됐는지도 확인한다. onnxruntime이나 openvino가 없는
# 환경(CI 등)에서는 건너뛴다.
def test_every_arm_reads_the_same_artifact(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("openvino")
    _tiny_model(tmp_path / "model_fp32.onnx")
    meta = {
        "version": inspector.ARTIFACT_VERSION,
        "kind": "reconstruction",
        "name": "tiny",
        "category": "toy",
        "img_size": SIZE,
        "map_size": SIZE,
        "threshold": 0.0,
    }
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    image = np.random.default_rng(0).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    outputs = {}
    for arm in runtime_compare.ARMS:
        insp = runtime_compare.load_arm(arm, tmp_path / "toy")
        assert insp.precision == "fp32" and insp.size == SIZE
        outputs[arm] = insp.run(image)
        info = runtime_compare.arm_info(arm, insp)
        assert info["runtime"] == runtime_compare.ARMS[arm]["runtime"]
    ov_info = runtime_compare.arm_info("R2", runtime_compare.load_arm("R2", tmp_path / "toy"))
    assert ov_info["compiled"]["NUM_STREAMS"] == "1"
    ref_score, ref_map = outputs["R0"]
    for arm in ("R1", "R2"):
        score, amap = outputs[arm]
        assert score == pytest.approx(ref_score, rel=1e-6)
        np.testing.assert_allclose(amap, ref_map, atol=1e-6)

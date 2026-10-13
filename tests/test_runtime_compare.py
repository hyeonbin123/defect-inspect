# [파일 역할] CPU 런타임 대조(onnxruntime R0·R1 대 OpenVINO R2) 모듈 `defect_inspect.runtime_compare`를
# 검증하는 테스트 묶음이다. 블록 순서 돌리기, 점수 동일성 관문 (가)~(라), 지연 요약과 H22 판정, 한가함 판정과
# 기다리기, CPU 사용률 계산, 호출 시간 재기, 세 팔이 같은 ONNX 파일을 읽는지를 확인한다.
# [왜 필요한가] 이 대조는 규칙(docs/experiments.md 'CPU 런타임 대조', 2026-10-04 측정 전 등록)에 정한 관문과
# 기준(10% 이득)으로 판정한다. 관문 계산이나 블록 순서가 틀리면 판정 자체가 틀리므로, 실제 모델과 데이터 없이
# 작은 점수 배열과 가짜 함수로 규칙의 각 조건을 하나씩 시험한다.
# [설계 메모] torch가 필요 없어 CI에서도 돈다. 마지막 테스트만 onnx·onnxruntime·openvino가 있을 때 아주 작은
# ONNX 모델을 만들어 세 팔에 실제로 돌린다.

import json

import numpy as np
import pytest

# inspector는 작은 아티팩트의 meta.json 버전 값을, calibrate.conformal_threshold는 기대 임계값(목표 5%)을
# 만드는 데 쓴다.
from defect_inspect import inspector, runtime_compare
from defect_inspect.calibrate import conformal_threshold


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
    arm[at_threshold] = np.nextafter(arm[at_threshold], np.float32(np.inf))  # one ulp up
    cat = runtime_compare.category_parity(ref, arm, labels, thr.value)
    assert cat["fold0_flags_same"]  # the arm's own threshold moves with the image
    assert cat["verdict_flips_stored"] == 1  # R0's threshold value now sits one ulp below it
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

    # Swap a normal and a defect score: tiny AUROC change, large relative differences.
    arm = ref.copy()
    normal_top = np.flatnonzero(labels == 0)[np.argmax(ref[labels == 0])]
    defect_low = np.flatnonzero(labels == 1)[np.argmin(ref[labels == 1])]
    arm[[normal_top, defect_low]] = arm[[defect_low, normal_top]]
    gate = runtime_compare.parity_gate({"a": runtime_compare.category_parity(ref, arm, labels, stored)}, 0.0)
    assert not gate["checks"]["c_macro_auroc_pp"]["pass"]

    # A fold-0 normal moves across the threshold: (d) fails although (a) to (c) would pass.
    normal = np.flatnonzero(labels == 0)
    thr = conformal_threshold(ref[normal], 0.05).value
    order = normal[np.argsort(ref[normal])]
    below = order[np.searchsorted(ref[order], thr) - 1]  # the largest normal not above the threshold
    above = order[np.searchsorted(ref[order], thr, side="right")]  # the smallest normal above it
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
# 않음과 기록을 돌려준다.
# 마지막 부분은 각 기록의 끝 시각(end)을 그 1분을 잰 뒤에 읽는지 확인한다. 2026-10-04 검토에서 측정 시각 기록
# 순서를 고친 것(소스 주석 '2026-10-04 fix')의 회귀를 막는다.
def test_idle_rule_and_wait():
    assert runtime_compare.is_idle({"cpu_pct": 9.9, "gpu_pct": 9.9})
    assert not runtime_compare.is_idle({"cpu_pct": 10.0, "gpu_pct": 0.0})
    assert not runtime_compare.is_idle({"cpu_pct": 2.0, "gpu_pct": 35.0})
    assert runtime_compare.is_idle({"cpu_pct": 2.0, "gpu_pct": None})  # no GPU to read
    assert not runtime_compare.is_idle({"cpu_pct": None, "gpu_pct": 0.0})

    minutes = iter([{"cpu_pct": 40.0, "gpu_pct": 0.0}, {"cpu_pct": 5.0, "gpu_pct": 1.0}])
    idle, log = runtime_compare.wait_idle(5, minute=lambda: next(minutes), clock=lambda fmt: "t")
    assert idle and [r["idle"] for r in log] == [False, True]
    idle, log = runtime_compare.wait_idle(2, minute=lambda: {"cpu_pct": 50.0, "gpu_pct": 0.0}, clock=str)
    assert not idle and len(log) == 2

    # `end` is read after the minute was measured.
    events = []

    def minute():
        events.append("minute")
        return {"cpu_pct": 1.0, "gpu_pct": 0.0}

    def clock(fmt):
        events.append("clock")
        return "t"

    runtime_compare.wait_idle(1, minute=minute, clock=clock)
    assert events == ["minute", "clock"]


# CPU 사용률 계산: (유휴 시간, 전체 시간) 두 측정 사이의 증가분으로 100 × (1 - 유휴 증가 / 전체 증가)를 낸다.
# 측정이 없거나 시간이 흐르지 않았으면 None이다. cpu_percent는 실제로 짧게 재 보고(Windows가 아니면 None)
# 범위만 확인한다.
def test_busy_percent():
    assert runtime_compare.busy_percent((50, 100), (80, 200)) == pytest.approx(70.0)
    assert runtime_compare.busy_percent(None, (1, 2)) is None
    assert runtime_compare.busy_percent((5, 10), (5, 10)) is None
    value = runtime_compare.cpu_percent(0.05)
    assert value is None or 0.0 <= value <= 100.0


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
    """image [1, 3, S, S] -> map = channel mean [1, S, S], score = map maximum [1] (opset 13)."""
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

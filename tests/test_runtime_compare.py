import json

import numpy as np
import pytest

from defect_inspect import inspector, runtime_compare
from defect_inspect.calibrate import conformal_threshold


def test_block_orders_rotate_every_arm_through_every_position():
    orders = runtime_compare.block_orders(["R0", "R1", "R2"], 3)
    assert orders == [["R0", "R1", "R2"], ["R1", "R2", "R0"], ["R2", "R0", "R1"]]
    for position in range(3):
        assert sorted(order[position] for order in orders) == ["R0", "R1", "R2"]
    assert runtime_compare.block_orders(["R0", "R2"], 3) == [["R0", "R2"], ["R2", "R0"], ["R0", "R2"]]


def _scores(seed=0, normal=60, defect=12):
    rng = np.random.default_rng(seed)
    ref = np.concatenate([rng.normal(1.0, 0.1, normal), rng.normal(1.6, 0.2, defect)]).astype(np.float32)
    labels = np.array([0] * normal + [1] * defect, dtype=np.int8)
    return ref, labels


def test_identical_scores_pass_every_check():
    ref, labels = _scores()
    stored = conformal_threshold(ref[labels == 0], 0.05).value
    cat = runtime_compare.category_parity(ref, ref.copy(), labels, stored)
    assert cat["max_rel_diff"] == 0.0 and cat["fold0_flags_same"]
    assert cat["auroc_ref"] == cat["auroc_arm"] and cat["verdict_flips_stored"] == 0
    assert cat["fold0_flags_ref"] == int((ref[labels == 0] > stored).sum())
    gate = runtime_compare.parity_gate({"a": cat, "b": cat}, map_abs=0.0)
    assert gate["pass"] and gate["checks"]["d_fold0_flags"]["same_in"] == 2


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


def test_busy_percent():
    assert runtime_compare.busy_percent((50, 100), (80, 200)) == pytest.approx(70.0)
    assert runtime_compare.busy_percent(None, (1, 2)) is None
    assert runtime_compare.busy_percent((5, 10), (5, 10)) is None
    value = runtime_compare.cpu_percent(0.05)
    assert value is None or 0.0 <= value <= 100.0


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


SIZE = 8


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

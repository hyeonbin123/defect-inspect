import json

import numpy as np
import pytest

from defect_inspect import analyze_stage6 as a6
from defect_inspect.calibrate import conformal_threshold
from defect_inspect.metrics import pro_histograms

CATS = ("alpha", "beta")


def _latency(**ms):
    """Latency file entries `dms-<size>-untrained` with the given registered CPU latency."""
    return {
        f"dms-{size}-untrained": {"cpu_fp32": {"total_with_resize_ms": value}}
        for size, value in ((int(k[1:]), v) for k, v in ms.items())
    }


@pytest.mark.parametrize(
    ("ms", "train"),
    [
        (
            {"s252": 120, "s280": 150, "s294": 170, "s308": 190},
            ["dms-252", "dms-280", "dms-308", "dms-280-car"],
        ),
        (
            {"s252": 120, "s280": 150, "s294": 195, "s308": 205},
            ["dms-252", "dms-280", "dms-294", "dms-280-car"],
        ),
        ({"s252": 120, "s280": 150, "s294": 201, "s308": 230}, ["dms-252", "dms-280", "dms-280-car"]),
        ({"s252": 180, "s280": 200.5, "s294": 220, "s308": 240}, ["dms-252", "dms-252-car"]),
        ({"s252": 200, "s280": 200.0, "s294": 220, "s308": 240}, ["dms-252", "dms-280", "dms-280-car"]),
    ],
)
def test_gate_picks_the_registered_training_set(ms, train):
    result = a6.gate(_latency(**ms))
    assert result["train"] == train and not result["blocked"]
    assert result["cpu_ms"]["dms-252"] == ms["s252"]


def test_gate_blocks_when_nothing_fits_and_needs_every_size():
    result = a6.gate(_latency(s252=201, s280=230, s294=240, s308=250))
    assert result["blocked"] and result["train"] == [] and result["passed"] == []
    with pytest.raises(ValueError, match="294"):
        a6.gate(_latency(s252=120, s280=150, s308=190))


def test_choose_takes_the_best_dev_auroc_within_budget_and_the_smaller_input_on_ties():
    rows = [
        {"name": "dms-252", "cpu_ms": 120.0, "macro_image_auroc": 0.93},
        {"name": "dms-280", "cpu_ms": 150.0, "macro_image_auroc": 0.95},
        {"name": "dms-280-car", "cpu_ms": 150.0, "macro_image_auroc": 0.95},
        {"name": "dms-308", "cpu_ms": 201.0, "macro_image_auroc": 0.99},
    ]
    pick = a6.choose(rows)
    assert pick["name"] == "dms-280" and pick["candidates"] == ["dms-252", "dms-280", "dms-280-car"]
    rows[2]["macro_image_auroc"] = 0.951
    assert a6.choose(rows)["name"] == "dms-280-car"
    rows[0]["name"], rows[0]["macro_image_auroc"] = "dms-294", 0.951  # same AUROC, larger input
    assert a6.choose(rows)["name"] == "dms-280-car"
    assert a6.choose([{"name": "dms-308", "cpu_ms": 230.0, "macro_image_auroc": 0.99}]) is None


def test_h16_and_h17_rules():
    good = {"macro_image_auroc": 0.95, "fixed_threshold": {"fpr": 0.05}}
    assert a6.h16(good, 199.0, 0.9679)["verdict"] == "지지"
    result = a6.h16({"macro_image_auroc": 0.9478, "fixed_threshold": {"fpr": 0.061}}, 200.1, 0.9679)
    assert result["verdict"] == "기각" and result["failed"] == ["a_cpu_ms", "b_fpr", "c_auroc_vs_dm"]
    assert a6.h16(good, None, 0.9679)["failed"] == ["a_cpu_ms"]
    rng = np.random.default_rng(0)
    assert a6.h17(0.05, 0.05 + rng.normal(0, 0.005, 2000))["verdict"] == "지지"
    assert a6.h17(0.005, 0.005 + rng.normal(0, 0.001, 2000))["verdict"] == "판정 불가"  # above 0, too small
    assert a6.h17(0.01, 0.01 + rng.normal(0, 0.02, 2000))["verdict"] == "판정 불가"
    assert a6.h17(-0.03, -0.03 + rng.normal(0, 0.005, 2000))["verdict"] == "기각"


def _write_run(
    out, name, *, protocol, shift=0.0, seed=0, with_pixels=False, condition="clean", cal=True, pc=False
):
    """A run in the common format (or the stage 4 CPU format with `pc`): normals ~N(0,1), defects ~N(2,1)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for c in CATS:
        neg = rng.normal(0, 1, 60).astype(np.float32)
        pos = (rng.normal(2, 1, 20) + shift).astype(np.float32)
        images = np.array([f"{c}/n{i}" for i in range(60)] + [f"{c}/d{i}" for i in range(20)])
        labels = np.array([0] * 60 + [1] * 20, dtype=np.int8)
        score = np.concatenate([neg, pos])
        cal_score = rng.normal(0, 1, 40).astype(np.float32) if cal else np.empty(0, np.float32)
        arrays = {"eval_images": images, "eval_labels": labels}
        if pc:
            arrays.update(eval_score_full=score, pool_score_oof=cal_score)
        else:
            arrays.update(eval_score=score, cal_score=cal_score)
        if with_pixels:
            maps = np.random.default_rng(99).random((80, 8, 8)).astype(np.float32)
            masks = np.zeros((80, 8, 8), dtype=bool)
            masks[60:, 2:5, 2:5] = True
            maps[60:, 2:5, 2:5] += 0.5
            hist = pro_histograms(maps, masks, bins=50)
            arrays.update(
                aupro=np.float64(0.8),
                pixel_auroc=np.float64(0.9),
                pro_edges=hist.edges,
                pro_normal=hist.normal,
                pro_components=hist.components,
                pro_component_image=hist.component_image,
            )
        np.savez(out / f"{c}.npz", **arrays)
    (out / "run.json").write_text(
        json.dumps({"protocol": protocol, "config": {"name": name, "condition": condition}}), encoding="utf-8"
    )
    return out


def test_dev_report_tables_the_candidates_and_the_car_pair(tmp_path):
    runs = [
        _write_run(tmp_path / "a", "dms-252", protocol="dev", seed=1, with_pixels=True, cal=False),
        _write_run(tmp_path / "b", "dms-280", protocol="dev", seed=1, shift=0.5, with_pixels=True, cal=False),
        _write_run(
            tmp_path / "c", "dms-280-car", protocol="dev", seed=1, shift=1.5, with_pixels=True, cal=False
        ),
    ]
    latency = _latency(s252=120, s280=150, s294=170, s308=230)
    latency["dms-280-car-untrained"] = {"cpu_fp32": {"total_with_resize_ms": 151.0}}
    report = a6.dev_report(runs, latency)
    assert [r["name"] for r in report["rows"]] == ["dms-252", "dms-280", "dms-280-car"]
    assert report["rows"][1]["macro_image_auroc"] > report["rows"][0]["macro_image_auroc"]
    assert report["pick"]["name"] == "dms-280-car" and report["rows"][0]["cpu_ms"] == 120
    assert report["gate"]["train"] == ["dms-252", "dms-280", "dms-294", "dms-280-car"]
    car = report["car_effect"]
    assert car["pair"] == ["dms-280-car", "dms-280"] and car["diff"] > 0 and len(car["ci"]) == 2


def test_sealed_report_judges_h16_and_h17(tmp_path):
    torch_dir = _write_run(tmp_path / "t", "dms-280", protocol="test", seed=2, shift=2.0, with_pixels=True)
    onnx_dir = _write_run(tmp_path / "o", "dms-280", protocol="test", seed=2, shift=2.0)
    pc_dir = _write_run(tmp_path / "p", "x", protocol="test", seed=3, pc=True)
    report = a6.sealed_report(
        torch_dir=torch_dir, onnx_dir=onnx_dir, patchcore_dir=pc_dir, cpu_ms=150.0, dm_auroc=0.9
    )
    assert report["name"] == "dms-280" and report["H17"]["verdict"] == "지지"
    assert report["onnx_fp32"]["macro_image_auroc"] > report["serving_patchcore_auroc"]
    assert report["score_rel_diff_onnx_vs_torch"]["max"] == 0.0
    # The registered torch metrics include the mean pixel AUROC next to the per-category values.
    per_cat = [v["pixel_auroc"] for v in report["torch_per_category"].values()]
    assert report["torch"]["macro_pixel_auroc"] == pytest.approx(np.mean(per_cat))
    assert report["torch"]["macro_pixel_auroc"] == pytest.approx(0.9)
    # The same scores and calibration normals: the torch thresholds give the ONNX pipeline's own rates.
    own = report["onnx_fp32"]["fixed_threshold"]
    assert report["onnx_at_torch_thresholds"]["fpr"] == pytest.approx(own["fpr"])
    h16 = report["H16"]
    assert h16["conditions"]["a_cpu_ms"]["met"] and h16["conditions"]["c_auroc_vs_dm"]["met"]
    assert h16["conditions"]["b_fpr"]["met"] == (own["fpr"] <= 0.06)
    with np.load(onnx_dir / "alpha.npz") as z:
        thr = conformal_threshold(z["cal_score"], 0.05).value
        assert z["eval_score"][:60][z["eval_score"][:60] > thr].size <= 60


def test_supplement_report_uses_the_clean_normals_for_thresholds(tmp_path):
    clean = _write_run(tmp_path / "clean", "dms-280", protocol="dev", seed=4, cal=False)
    shifted = _write_run(
        tmp_path / "shift", "dms-280", protocol="dev", seed=4, cal=False, condition="shift-1"
    )
    # Move every normal score up: more normals cross the clean thresholds.
    for c in CATS:
        with np.load(shifted / f"{c}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        arrays["eval_score"] = arrays["eval_score"] + np.where(arrays["eval_labels"] == 0, 1.5, 0).astype(
            np.float32
        )
        np.savez(shifted / f"{c}.npz", **arrays)
    int8 = _write_run(tmp_path / "int8", "dms-280", protocol="dev", seed=4, cal=False, shift=-0.2)
    report = a6.supplement_report(clean, [shifted], int8)
    rows = {r["condition"]: r for r in report["rows"]}
    assert rows["clean"]["fpr"] <= 0.05 and rows["shift-1"]["fpr"] > 0.3
    assert report["int8_dynamic"]["diff"] < 0

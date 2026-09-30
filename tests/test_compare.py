import numpy as np
import pytest

from defect_inspect import compare
from defect_inspect.metrics import aupro, pixel_auroc, pro_histograms

CATEGORIES = ("alpha", "beta", "gamma")
N_NEG, N_POS = 80, 24
TYPES = ["scratch", "hole", "stain"]


@pytest.fixture(autouse=True)
def _few_resamples(monkeypatch):
    monkeypatch.setattr(compare, "N_BOOT", 200)


def _category(name, rng, separation, *, train_types=None, cal=60):
    labels = np.array([0] * N_NEG + [1] * N_POS, dtype=np.int8)
    scores = np.concatenate([rng.normal(0, 1, N_NEG), rng.normal(separation, 1, N_POS)]).astype(np.float32)
    maps = rng.random((N_NEG + N_POS, 8, 8)).astype(np.float32)
    masks = np.zeros((N_NEG + N_POS, 8, 8), dtype=bool)
    masks[N_NEG:, 2:4, 2:4] = True
    maps[N_NEG:, 2:4, 2:4] += 0.6
    hist = pro_histograms(maps, masks, bins=100)
    cat = {
        "eval_images": np.array([f"{name}/{i}.JPG" for i in range(N_NEG + N_POS)]),
        "eval_labels": labels,
        "eval_defect_types": np.array([""] * N_NEG + [TYPES[i % 3] for i in range(N_POS)]),
        "eval_score": scores,
        "cal_score": rng.normal(0, 1, cal).astype(np.float32),
        "pixel_auroc": np.float64(pixel_auroc(maps, masks)),
        "aupro": np.float64(aupro(maps, masks)),
        "pro_edges": hist.edges,
        "pro_normal": hist.normal,
        "pro_components": hist.components,
        "pro_component_image": hist.component_image,
    }
    if train_types is not None:
        cat["train_defect_types"] = np.array(train_types)
    return cat


def _run(name, separation, seed, calibration="crossfit", **kwargs):
    rng = np.random.default_rng(seed)
    run = compare.MethodRun(name, calibration)
    for c in CATEGORIES:
        run.cats[c] = _category(c, rng, separation, **kwargs)
    return run


def test_summary_and_paired_difference():
    weak, strong = _run("p0", 1.0, 1), _run("d-s", 3.0, 2)
    report = compare.build_report([weak, strong], {}, reference_name="d-s")
    s = report["unsupervised"]
    assert s["d-s"]["macro_image_auroc"] > s["p0"]["macro_image_auroc"]
    for v in s.values():
        lo, hi = v["macro_image_auroc_ci"]
        assert lo <= v["macro_image_auroc"] <= hi
        lo, hi = v["macro_aupro_ci"]
        assert lo - 0.01 <= v["macro_aupro"] <= hi + 0.01
        fixed = v["fixed_threshold"]
        assert fixed["calibration"] == "crossfit" and fixed["n_cal"] == 180
        # Calibration scores and test normals are both N(0, 1): the rate is near the 5% target.
        assert 0.0 <= fixed["fpr"] < 0.15
    d = report["vs_baseline"]["pairs"]["d-s"]
    assert d["verdict"] == "차이 있음" and d["diff"] > 0.1
    assert "supervised" not in report and "hypotheses" not in report
    assert "d-s - p0" in compare.format_report(report)


def test_identical_methods_are_not_different():
    a, b = _run("p0", 2.0, 5), _run("d-s", 2.0, 5)
    d = compare.build_report([a, b], {}, "p0")["vs_baseline"]["pairs"]["d-s"]
    assert d["diff"] == 0.0 and d["verdict"] == "판정 불가"


def test_difference_wording():
    rng = np.random.default_rng(0)
    clear = rng.normal(0.03, 0.005, 500)
    assert compare.difference(0.03, clear)["verdict"] == "차이 있음"
    assert compare.difference(0.004, rng.normal(0.004, 0.001, 500))["verdict"] == "차이 작음"
    assert compare.difference(0.01, rng.normal(0.01, 0.02, 500))["verdict"] == "판정 불가"
    assert compare.difference(-0.03, -clear)["verdict"] == "차이 있음"


def test_different_evaluation_images_are_rejected():
    a, b = _run("p0", 2.0, 1), _run("d-s", 2.0, 2)
    b.cats["beta"]["eval_images"] = b.cats["beta"]["eval_images"][::-1].copy()
    with pytest.raises(ValueError, match="different images"):
        compare.build_report([a, b], {}, "p0")


def test_unseen_masks_follow_the_training_types():
    sup = _run("sup", 2.0, 3, calibration="holdout", train_types=["scratch", "hole|scratch"])
    masks, total = compare.unseen_masks(sup)
    # Only "stain" defects (every third one) share no type with the training defects.
    assert set(masks) == set(CATEGORIES) and total == 3 * 8
    assert masks["alpha"].tolist() == [i % 3 == 2 for i in range(N_POS)]
    seen_all = _run("sup", 2.0, 3, calibration="holdout", train_types=["scratch", "hole", "stain"])
    assert compare.unseen_masks(seen_all) == ({}, 0)


def _supervised(separations, train_types):
    return {
        (k, s): _run(
            f"sup-k{k}-s{s}", sep, 100 + 10 * k + s, calibration="holdout", train_types=train_types, cal=40
        )
        for k, sep in separations.items()
        for s in (0, 1)
    }


def test_supervised_crossing_and_hypotheses(monkeypatch):
    monkeypatch.setattr(compare, "MIN_UNSEEN_POOLED", 10)
    unsup = [_run("p0", 1.5, 1), _run("d-s", 2.2, 2)]
    sup = _supervised({5: 0.8, 10: 1.2, 20: 1.9, 40: 4.0}, ["scratch", "hole"])
    report = compare.build_report(unsup, sup, reference_name="d-s")
    rows = report["supervised"]
    assert rows["5"]["labels_with_validation"] == 25 and rows["5"]["seeds"] == [0, 1]
    assert rows["5"]["all_defects"]["diff"] < 0 < rows["40"]["all_defects"]["diff"]
    h = report["hypotheses"]
    assert h["H5 (k=5)"] == "지지" and h["H5 (k=10)"] == "지지" and h["H6 (k=40)"] == "지지"
    assert h["교차점"] == "k = 40"
    assert h["H8 (d-s > P0)"] == "지지"
    # Unseen types: 8 "stain" defects per category are judged at every k, and at k=40 supervised is ahead.
    assert rows["40"]["unseen_types"]["judged"] and rows["40"]["unseen_types"]["unseen_defects_mean"] == 24
    assert h["H7"].startswith("기각") and "40" in h["H7"]
    text = compare.format_report(report)
    assert "교차점" in text and "unseen-type" in text
    assert set(report["supervised_runs"]) == {f"k{k}-s{s}" for k in (5, 10, 20, 40) for s in (0, 1)}


def test_unseen_types_need_enough_defects():
    unsup = [_run("p0", 1.5, 1), _run("d-s", 2.2, 2)]
    sup = _supervised({5: 1.0, 40: 3.0}, ["scratch", "hole"])  # 24 unseen defects < the pooled minimum of 30
    report = compare.build_report(unsup, sup, reference_name="d-s")
    assert not report["supervised"]["5"]["unseen_types"]["judged"]
    assert report["hypotheses"]["H7"].startswith("판정 불가")


def test_no_crossing_is_reported():
    unsup = [_run("p0", 1.5, 1), _run("d-s", 3.5, 2)]
    sup = _supervised({5: 0.5, 40: 1.0}, ["scratch", "hole", "stain"])
    h = compare.build_report(unsup, sup, reference_name="d-s")["hypotheses"]
    assert h["교차점"] == "k = 40까지 교차 없음" and h["H6 (k=40)"] == "기각"


def test_method_without_calibration_scores_has_no_fixed_threshold():
    a = _run("p0", 2.0, 1)
    b = _run("dm", 2.0, 2, calibration="none")
    report = compare.build_report([a, b], {}, "p0")
    assert report["unsupervised"]["dm"]["fixed_threshold"] is None
    assert "| dm |" in compare.format_report(report)


def test_cli_needs_a_reference_for_the_test_protocol():
    with pytest.raises(SystemExit):
        compare.main(["--protocol", "test"])

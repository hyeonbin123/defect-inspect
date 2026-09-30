import csv
import json

import numpy as np
import pytest

from defect_inspect import analyze
from defect_inspect.metrics import aupro, pixel_auroc, pro_histograms

CATEGORIES = ("alpha", "beta", "gamma")


def _write_category(run_dir, name, rng, *, resub_shift=-2.0, n_pool=200, n_neg=120, n_pos=30):
    """One synthetic category: normal scores ~ N(0, 1), defects ~ N(2.5, 1), resubstitution scores too low."""
    folds = np.arange(n_pool) % 4 + 1  # dev protocol: folds 1..4
    labels = np.array([0] * n_neg + [1] * n_pos, dtype=np.int8)
    eval_full = np.concatenate([rng.normal(0, 1, n_neg), rng.normal(2.5, 1, n_pos)]).astype(np.float32)
    eval_holdout = eval_full + rng.normal(0, 0.05, n_neg + n_pos).astype(np.float32)
    maps = rng.random((n_neg + n_pos, 16, 16)).astype(np.float32)
    masks = np.zeros((n_neg + n_pos, 16, 16), dtype=bool)
    for i in range(n_neg, n_neg + n_pos):
        masks[i, 4:8, 4:8] = True
        maps[i, 4:8, 4:8] += 0.5
    hist = pro_histograms(maps, masks, bins=200)
    np.savez(
        run_dir / f"{name}.npz",
        eval_images=np.array([f"{name}/eval/{i}.JPG" for i in range(n_neg + n_pos)]),
        eval_labels=labels,
        eval_defect_types=np.array([""] * n_neg + ["scratch"] * n_pos),
        eval_score_full=eval_full,
        eval_score_holdout=eval_holdout,
        pool_images=np.array([f"{name}/pool/{i}.JPG" for i in range(n_pool)]),
        pool_folds=folds,
        pool_score_resub=rng.normal(resub_shift, 1, n_pool).astype(np.float32),
        pool_score_oof=rng.normal(0, 1, n_pool).astype(np.float32),
        pixel_auroc=np.float64(pixel_auroc(maps, masks)),
        aupro=np.float64(aupro(maps, masks)),
        pro_edges=hist.edges,
        pro_normal=hist.normal,
        pro_components=hist.components,
        pro_component_image=hist.component_image,
    )


@pytest.fixture
def run_dir(tmp_path):
    rng = np.random.default_rng(0)
    for name in CATEGORIES:
        _write_category(tmp_path, name, rng)
    meta = {
        "config": {"name": "fake"},
        "protocol": "dev",
        "commit": "abc1234",
        "categories": [{"category": c} for c in CATEGORIES],
    }
    (tmp_path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _small_reference(monkeypatch):
    monkeypatch.setattr(analyze, "REF_REALISATIONS", 20)
    monkeypatch.setattr(analyze, "REF_DRAWS", 100)


def test_report_structure_and_ranges(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "N_BOOT", 200)
    report = analyze.build_report(run_dir)
    acc = report["accuracy"]
    assert set(acc["per_category"]) == set(CATEGORIES)
    lo, hi = acc["macro_image_auroc_ci"]
    assert 0.9 < lo <= acc["macro_image_auroc"] <= hi <= 1.0
    lo, hi = acc["macro_aupro_ci"]
    assert lo <= acc["macro_aupro_binned"] <= hi
    assert abs(acc["macro_aupro"] - acc["macro_aupro_binned"]) < 0.01
    for alpha in ("0.05", "0.01"):
        cal = report["calibration"][alpha]
        for strategy in analyze.STRATEGIES:
            v = cal[strategy]
            assert v["n_normal"] == 360 and v["n_defect"] == 90
            assert v["fpr"] == v["fp"] / v["n_normal"]
            assert v["fpr_ci"][0] <= v["fpr"] <= v["fpr_ci"][1]
            assert v["fpr_clopper_pearson"][0] <= v["fpr"] <= v["fpr_clopper_pearson"][1]
        # The hold-out strategy calibrates on one fold only.
        assert cal["holdout"]["n_cal_total"] == 150 and cal["crossfit"]["n_cal_total"] == 600
    text = analyze.format_report(report)
    assert "resubstitution" in text and "H1" in text


def test_calibration_matches_a_direct_computation(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "N_BOOT", 50)
    _, cats = analyze.load_run(run_dir)
    cal = analyze.calibration(cats, "dev", 0.05)
    fp = tp = 0
    for cat in cats.values():
        scores = np.sort(cat["pool_score_oof"])
        rank = int(np.ceil((len(scores) + 1) * 0.95))
        threshold = scores[rank - 1]
        full = cat["eval_score_full"]
        fp += int((full[cat["eval_labels"] == 0] > threshold).sum())
        tp += int((full[cat["eval_labels"] == 1] > threshold).sum())
    assert cal["crossfit"]["fp"] == fp and cal["crossfit"]["tp"] == tp
    # Hold-out thresholds come from fold 1 (dev protocol) and apply to the hold-out model's scores.
    cat = cats["alpha"]
    held = np.sort(cat["pool_score_oof"][cat["pool_folds"] == 1])
    rank = int(np.ceil((len(held) + 1) * 0.95))
    assert cal["holdout"]["per_category"]["alpha"]["threshold"] == pytest.approx(float(held[rank - 1]))


def test_resubstitution_with_too_low_scores_supports_h1(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "N_BOOT", 200)
    report = analyze.build_report(run_dir)
    # Calibration scores two standard deviations below the test normals: almost everything is flagged.
    assert report["calibration"]["0.05"]["resubstitution"]["fpr"] > 0.3
    assert report["hypotheses"]["H1"] == "지지"


def _cal(resub_ci, hold_fpr, band, cross_ci, diff_ci):
    return {
        "resubstitution": {"fpr_ci": resub_ci},
        "holdout": {"fpr": hold_fpr, "theory_band": band},
        "crossfit": {"fpr_ci": cross_ci},
        "tpr_crossfit_minus_holdout": {"ci": diff_ci},
    }


@pytest.mark.parametrize(
    ("cal", "expected"),
    [
        (
            _cal([0.11, 0.15], 0.05, [0.04, 0.06], [0.03, 0.049], [-0.01, 0.03]),
            {"H1": "지지", "H2": "지지", "H3": "지지", "H4": "지지"},
        ),
        (
            _cal([0.04, 0.09], 0.07, [0.04, 0.06], [0.051, 0.07], [-0.06, -0.03]),
            {"H1": "기각", "H2": "기각 (이론 구간보다 높음)", "H3": "기각", "H4": "기각"},
        ),
        (
            _cal([0.08, 0.12], 0.03, [0.04, 0.06], [0.04, 0.06], [-0.03, 0.01]),
            {"H1": "판정 불가", "H2": "기각 (이론 구간보다 낮음)", "H3": "판정 불가", "H4": "판정 불가"},
        ),
    ],
)
def test_hypothesis_rules(cal, expected):
    assert analyze.hypotheses(cal, 0.05) == expected


def test_scores_csv_round_trip(run_dir, tmp_path):
    _, cats = analyze.load_run(run_dir)
    path = tmp_path / "scores.csv"
    analyze.write_scores_csv(cats, path)
    assert b"\r" not in path.read_bytes()
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * (150 + 200)
    eval_rows = [r for r in rows if r["category"] == "beta" and r["set"] == "eval"]
    assert [r["label"] for r in eval_rows].count("anomaly") == 30
    assert float(eval_rows[0]["score_full"]) == pytest.approx(
        float(cats["beta"]["eval_score_full"][0]), abs=1e-6
    )
    pool_rows = [r for r in rows if r["category"] == "beta" and r["set"] == "pool"]
    assert {r["fold"] for r in pool_rows} == {"1", "2", "3", "4"}


def test_finite_sample_reference_covers_exchangeable_scores(run_dir, monkeypatch):
    monkeypatch.setattr(analyze, "REF_REALISATIONS", 60)
    _, cats = analyze.load_run(run_dir)
    rows = analyze.finite_sample(cats, 0.05)
    assert [r["n"] for r in rows] == [20, 50, 100, 200]
    for r in rows:
        lo, hi = r["reference_interval"]
        assert lo < r["reference_mean"] < hi
        # Exchangeable simulation: its mean sits at the marginal rate of the conformal rule.
        assert r["reference_mean"] == pytest.approx(r["theory_mean"], abs=0.006)
    # The fixture's cross-fitted scores and test normals come from the same distribution.
    assert sum(r["within_reference"] for r in rows) >= 3


def test_finite_sample_flags_a_conservative_shift(run_dir):
    _, cats = analyze.load_run(run_dir)
    for cat in cats.values():
        cat["pool_score_oof"] = (
            cat["pool_score_oof"] + 1.0
        )  # calibration scores too high: thresholds too strict
    rows = analyze.finite_sample(cats, 0.05)
    assert all(not r["within_reference"] and r["mean"] < r["reference_interval"][0] for r in rows)

import json

import numpy as np
import pytest

from defect_inspect import analyze_export as ae
from defect_inspect import compare, paths

SUPPORTED, REJECTED, UNDECIDED = "지지", "기각", "판정 불가"


def test_the_registered_constants():
    # docs/experiments.md, stage 4: thresholds at a target of 5%, minimum effect 2.0 points.
    assert ae.ALPHA == 0.05
    assert ae.MIN_EFFECT_FPR == 0.02


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


def _category(name, neg, pos, cal):
    scores = np.concatenate([neg, pos]).astype(np.float32)
    return {
        "eval_images": np.array([f"{name}/{i}" for i in range(len(scores))]),
        "eval_labels": np.array([0] * len(neg) + [1] * len(pos), dtype=np.int8),
        "eval_score": scores,
        "cal_score": np.asarray(cal, dtype=np.float32),
    }


def _shifted(cats, eval_shift=0.0, cal_shift=0.0):
    return {
        name: {
            **cat,
            "eval_score": cat["eval_score"] + np.float32(eval_shift),
            "cal_score": cat["cal_score"] + np.float32(cal_shift),
        }
        for name, cat in cats.items()
    }


def _base(seed=0, sizes=(("a", 200, 40, 301), ("b", 120, 20, 97))):
    rng = np.random.default_rng(seed)
    return {
        name: _category(
            name, rng.normal(1.0, 0.2, n_neg), rng.normal(1.6, 0.3, n_pos), rng.normal(1.0, 0.2, n_cal)
        )
        for name, n_neg, n_pos, n_cal in sizes
    }


def _run(pipelines):
    return {"meta": {"artifacts": "x", "ratio": 0.01, "protocol": "dev"}, "pipelines": pipelines}


@pytest.fixture(autouse=True)
def few_draws(monkeypatch):
    monkeypatch.setattr(compare, "N_BOOT", 300)


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


def test_identical_int8_scores_reject_the_claim():
    base = _base()
    report = ae.build_report(_run({"torch": base, "fp32": base, "int8": base}))
    shift = report["int8_at_fp32_thresholds"]
    assert shift["verdict"] == REJECTED
    assert shift["d_fpr"] == 0.0 and shift["d_fpr_ci"] == [0.0, 0.0]
    assert shift["fp32"] == shift["int8_at_fp32_thresholds"] == shift["int8_recalibrated"]
    assert report["int8_scored"] is True


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


def test_a_run_without_int8_has_no_verdict_and_says_so():
    base = _base()
    report = ae.build_report(_run({"torch": base, "fp32": base}))
    assert "int8_at_fp32_thresholds" not in report and report["int8_scored"] is False
    assert [row["pipeline"] for row in report["rows"]] == ["torch", "fp32"]
    assert "INT8 was not scored" in ae.format_report(report)


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

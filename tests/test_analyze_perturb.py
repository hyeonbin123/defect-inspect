import json

import numpy as np
import pytest

from defect_inspect import analyze_perturb
from defect_inspect.calibrate import conformal_threshold
from defect_inspect.conditions import condition_names
from defect_inspect.metrics import auroc
from defect_inspect.stats import percentile_ci, stratified_indices

# Categories of different sizes, so that a pooled rate differs from a mean of per-category rates.
SIZES = {"alpha": (80, 24), "beta": (50, 10), "gamma": (120, 30)}
NAMES = condition_names()
SHIFTED, BLURRED, BROKEN = "brightness-2", "blur-3", "jpeg-3"


def _run(seed=0, cal=100, method="p0", protocol="dev"):
    """A synthetic perturbation run.

    Normals and calibration scores are N(0, 1), defects N(3, 1). `SHIFTED` adds 2 to every score (the
    ranking stays, the fixed thresholds are overrun), `BLURRED` pulls the defects down among the normals
    (accuracy falls, the normals do not move), `BROKEN` does both. Every other condition equals clean.
    """
    rng = np.random.default_rng(seed)
    cats = {}
    for name, (n_neg, n_pos) in SIZES.items():
        labels = np.array([0] * n_neg + [1] * n_pos, dtype=np.int8)
        clean = np.concatenate([rng.normal(0, 1, n_neg), rng.normal(3, 1, n_pos)])
        clean = np.round(clean * 1024) / 1024  # the shifts below are then exact in float32
        scores = np.tile(clean, (len(NAMES), 1))
        scores[NAMES.index(SHIFTED)] += 2.0
        scores[NAMES.index(BLURRED), n_neg:] -= 2.5
        scores[NAMES.index(BROKEN)] += 2.0
        scores[NAMES.index(BROKEN), n_neg:] -= 2.5
        cats[name] = {
            "eval_images": np.array([f"{name}/{i}.JPG" for i in range(n_neg + n_pos)]),
            "eval_labels": labels,
            "conditions": np.array(NAMES),
            "scores": scores.astype(np.float32),
            "cal_score": rng.normal(0, 1, cal).astype(np.float32),
        }
    meta = {
        "method": method,
        "protocol": protocol,
        "commit": "abc1234",
        "device": "cpu",
        "source": f"outputs/{method}-{protocol}",
        "source_commit": "src1234",
        "conditions": NAMES,
        "categories": [{"category": c, "clean_max_rel_diff": 1e-5 * (i + 1)} for i, c in enumerate(cats)],
    }
    return {"meta": meta, "cats": cats}


def _write(run, run_dir):
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(run["meta"]), encoding="utf-8")
    for name, cat in run["cats"].items():
        np.savez(run_dir / f"{name}.npz", **cat)


def _row(table, name):
    return next(row for row in table if row["condition"] == name)


def _tiny(normals, defects, cal):
    """A one-category run written out by hand: `normals` maps each condition to its normal scores.

    The defects score the same under every condition.
    """
    names = list(normals)
    n_neg = len(normals[names[0]])
    scores = np.array([[*normals[name], *defects] for name in names], dtype=np.float32)
    cat = {
        "eval_images": np.array([f"only/{i}.JPG" for i in range(n_neg + len(defects))]),
        "eval_labels": np.array([0] * n_neg + [1] * len(defects), dtype=np.int8),
        "conditions": np.array(names),
        "scores": scores,
        "cal_score": np.asarray(cal, dtype=np.float32),
    }
    return {"meta": {"method": "p0", "categories": [{"category": "only"}]}, "cats": {"only": cat}}


def _above(count, of=100):
    """Normal scores of which exactly `count` lie above the threshold 95 of `np.arange(100)`."""
    return [100.0] * count + [0.0] * (of - count)


def test_table_has_one_row_per_condition_with_clean_first():
    table = analyze_perturb.condition_table(_run(), n_boot=100)
    assert [row["condition"] for row in table] == NAMES
    clean = table[0]
    assert set(clean) == {
        "condition",
        "macro_image_auroc",
        "macro_image_auroc_ci",
        "fpr",
        "fpr_ci",
        "tpr",
        "tpr_ci",
    }
    for row in table[1:]:
        assert set(row) == set(clean) | {"d_auroc", "d_auroc_ci", "d_fpr", "d_fpr_ci", "d_tpr", "d_tpr_ci"}
    for row in table:
        for key in ("macro_image_auroc", "fpr", "tpr"):
            lo, hi = row[f"{key}_ci"]
            assert lo <= row[key] <= hi
    # Calibration scores and clean normals are both N(0, 1): the clean rate is near the 5% target.
    assert 0.0 < clean["fpr"] < 0.12 and clean["macro_image_auroc"] > 0.95


def test_point_estimates_match_a_direct_computation():
    run = _run()
    table = analyze_perturb.condition_table(run, n_boot=20)
    for name in ("clean", SHIFTED, BLURRED):
        k = NAMES.index(name)
        aucs, fp, tp = [], 0, 0
        for cat in run["cats"].values():
            labels, scores = cat["eval_labels"], cat["scores"][k]
            threshold = conformal_threshold(cat["cal_score"], 0.05).value
            aucs.append(auroc(scores[labels == 0], scores[labels == 1]))
            fp += int((scores[labels == 0] > threshold).sum())
            tp += int((scores[labels == 1] > threshold).sum())
        row = _row(table, name)
        assert row["macro_image_auroc"] == pytest.approx(np.mean(aucs), abs=1e-12)
        assert row["fpr"] == fp / 250 and row["tpr"] == tp / 64  # pooled counts, not a mean of rates
    # The target decides the thresholds: a stricter one flags fewer normals.
    strict = analyze_perturb.condition_table(run, alpha=0.01, n_boot=20)
    assert strict[0]["fpr"] < table[0]["fpr"]


def test_a_score_equal_to_its_threshold_is_not_flagged():
    # 100 calibration scores 0..99 at a 5% target: the 96th smallest, 95, is the threshold.
    cal = np.arange(100)
    assert conformal_threshold(cal, 0.05).value == 95.0
    normals, defects = [95.0, 95.0, 96.0, 10.0], [95.0, 200.0]
    run = _tiny({"clean": normals, "same": normals}, defects, cal)
    n_boot, seed = 40, 2
    table = analyze_perturb.condition_table(run, n_boot=n_boot, seed=seed)
    # The rule is `score > threshold`: the three scores of exactly 95 are not defects.
    assert table[0]["fpr"] == 0.25 and table[0]["tpr"] == 0.5
    # The resamples use the same rule.
    neg_idx, pos_idx = stratified_indices([4, 2], n_boot=n_boot, seed=seed)
    fpr = (np.array(normals)[neg_idx] > 95.0).mean(axis=1)
    tpr = (np.array(defects)[pos_idx] > 95.0).mean(axis=1)
    assert table[0]["fpr_ci"] == pytest.approx(percentile_ci(fpr), abs=1e-12)
    assert table[0]["tpr_ci"] == pytest.approx(percentile_ci(tpr), abs=1e-12)


def test_upward_shift_keeps_accuracy_and_breaks_the_threshold_h9_supported():
    table = analyze_perturb.condition_table(_run(), n_boot=200)
    clean, shifted = table[0], _row(table, SHIFTED)
    # Adding a constant changes no rank: the AUROC is exactly the clean one, also in every resample.
    assert shifted["macro_image_auroc"] == clean["macro_image_auroc"]
    assert shifted["d_auroc"] == 0.0 and shifted["d_auroc_ci"] == [0.0, 0.0]
    assert shifted["fpr"] > 0.5 and shifted["d_fpr"] > 0.45
    assert shifted["d_fpr_ci"][0] > 0.3 and shifted["d_tpr"] >= 0.0
    # Lost defects: accuracy falls and the normals stay where they were.
    blurred = _row(table, BLURRED)
    assert blurred["d_auroc"] < -0.1 and blurred["d_auroc_ci"][1] < -0.05
    assert blurred["d_fpr"] == 0.0 and blurred["d_fpr_ci"] == [0.0, 0.0] and blurred["d_tpr"] < -0.3
    # Both at once: the false alarms are there, but so is a drop in accuracy that would be noticed.
    broken = _row(table, BROKEN)
    assert broken["fpr"] == shifted["fpr"] and broken["d_auroc"] == pytest.approx(blurred["d_auroc"])
    # An unchanged condition differs from clean by exactly nothing (the draws are shared).
    same = _row(table, "gamma-1")
    assert (same["d_auroc"], same["d_fpr"], same["d_tpr"]) == (0.0, 0.0, 0.0)
    assert same["d_auroc_ci"] == same["d_fpr_ci"] == same["d_tpr_ci"] == [0.0, 0.0]

    assert analyze_perturb.h9(table) == {"verdict": "지지", "conditions": [SHIFTED]}


def test_h9_rule_on_hand_made_tables():
    def row(name, d_auroc, fpr):
        return {"condition": name, "d_auroc": d_auroc, "fpr": fpr}

    clean = {"condition": "clean", "fpr": 0.5}  # never a candidate, whatever its rate
    assert analyze_perturb.h9([clean]) == {"verdict": "기각", "conditions": []}
    table = [
        clean,
        row("a", -0.005, 0.20),  # accuracy holds, rate doubled: counts
        row("b", -0.02, 0.30),  # accuracy fell by 2 points: the drop would be noticed
        row("c", 0.0, 0.0999),  # rate just under twice the target
        row("d", 0.003, 0.10),  # exactly twice the target counts ("at least")
        row("e", -0.01, 0.50),  # exactly 1.0 point is not "less than 1.0 point"
        row("f", -0.0099, 0.15),
    ]
    assert analyze_perturb.h9(table) == {"verdict": "지지", "conditions": ["a", "d", "f"]}
    assert analyze_perturb.h9([clean, table[2], table[3], table[5]]) == {"verdict": "기각", "conditions": []}
    # The limit follows the target.
    assert analyze_perturb.h9(table, alpha=0.1)["conditions"] == ["a"]
    assert analyze_perturb.h9(table, alpha=0.01)["conditions"] == ["a", "c", "d", "f"]
    with pytest.raises(ValueError, match="fixed thresholds"):
        analyze_perturb.h9([clean, row("a", 0.0, None)])

    # The rate itself is compared with twice the target, not its increase over clean: rows as
    # `condition_table` makes them carry both.
    full = [
        {"condition": "clean", "fpr": 0.07},
        {"condition": "g", "d_auroc": 0.0, "fpr": 0.12, "d_fpr": 0.05},  # rate 12%: counts
        {"condition": "h", "d_auroc": 0.0, "fpr": 0.09, "d_fpr": 0.02},  # rate 9%: does not
        {"condition": "i", "d_auroc": 0.0, "fpr": 0.03, "d_fpr": -0.04},
    ]
    assert analyze_perturb.h9(full) == {"verdict": "지지", "conditions": ["g"]}


def test_h9_is_judged_on_the_rate_not_on_its_increase():
    # A clean rate of 7%. Every defect scores above every normal under every condition, so the AUROC
    # never moves and only the pooled false positive rate decides.
    run = _tiny(
        {"clean": _above(7), "to-12": _above(12), "to-9": _above(9), "to-10": _above(10)},
        [200.0] * 20,
        np.arange(100),
    )
    table = analyze_perturb.condition_table(run, n_boot=50)
    assert [row["macro_image_auroc"] for row in table] == [1.0] * 4
    assert [row["fpr"] for row in table] == [0.07, 0.12, 0.09, 0.10]
    assert [row["d_fpr"] for row in table[1:]] == pytest.approx([0.05, 0.02, 0.03])
    assert [row["d_auroc"] for row in table[1:]] == [0.0] * 3
    # 12% and exactly 10% are "at least twice the target" although neither rose by 10 points; 9% is not.
    assert analyze_perturb.h9(table) == {"verdict": "지지", "conditions": ["to-12", "to-10"]}

    # Without such a condition the hypothesis is rejected, however large the clean rate is.
    run = _tiny({"clean": _above(9), "to-9": _above(9), "to-8": _above(8)}, [200.0] * 20, np.arange(100))
    assert analyze_perturb.h9(analyze_perturb.condition_table(run, n_boot=50)) == {
        "verdict": "기각",
        "conditions": [],
    }


def test_defaults_are_the_registered_ones():
    assert (analyze_perturb.N_BOOT, analyze_perturb.SEED, analyze_perturb.ALPHA) == (2000, 0, 0.05)
    assert analyze_perturb.MAX_AUROC_DROP == 0.01 and analyze_perturb.H9_METHOD == "p0"
    run = _run()
    table = analyze_perturb.condition_table(run)
    assert table == analyze_perturb.condition_table(run, 0.05, 2000, 0)
    assert table != analyze_perturb.condition_table(run, 0.05, 200, 0)
    assert table != analyze_perturb.condition_table(run, 0.05, 2000, 1)


def test_intervals_and_paired_differences_match_a_direct_bootstrap():
    run = _run(seed=3)
    n_boot, seed = 60, 5
    table = analyze_perturb.condition_table(run, n_boot=n_boot, seed=seed)

    # One generator, groups in order: normals then defects of each category.
    sizes = [n for pair in SIZES.values() for n in pair]
    draws = stratified_indices(sizes, n_boot=n_boot, seed=seed)
    thresholds = [conformal_threshold(cat["cal_score"], 0.05).value for cat in run["cats"].values()]

    def resampled(name):
        k = NAMES.index(name)
        macro, fpr, tpr = np.zeros(n_boot), np.zeros(n_boot), np.zeros(n_boot)
        for b in range(n_boot):
            aucs, fp, tp = [], 0, 0
            for i, cat in enumerate(run["cats"].values()):
                labels, scores = cat["eval_labels"], cat["scores"][k].astype(np.float64)
                neg = scores[labels == 0][draws[2 * i][b]]
                pos = scores[labels == 1][draws[2 * i + 1][b]]
                aucs.append(auroc(neg, pos))
                fp += int((neg > thresholds[i]).sum())
                tp += int((pos > thresholds[i]).sum())
            macro[b], fpr[b], tpr[b] = np.mean(aucs), fp / 250, tp / 64
        return macro, fpr, tpr

    base = resampled("clean")
    for key, samples in zip(("macro_image_auroc", "fpr", "tpr"), base, strict=True):
        assert table[0][f"{key}_ci"] == pytest.approx(percentile_ci(samples), abs=1e-12)
    for name in (SHIFTED, BLURRED, BROKEN):
        row = _row(table, name)
        other = resampled(name)
        for key, full, mine, clean in zip(
            ("auroc", "fpr", "tpr"), ("macro_image_auroc", "fpr", "tpr"), other, base, strict=True
        ):
            assert row[f"{full}_ci"] == pytest.approx(percentile_ci(mine), abs=1e-12)
            # Paired: the interval of the per-resample differences, not a difference of intervals.
            assert row[f"d_{key}_ci"] == pytest.approx(percentile_ci(mine - clean), abs=1e-12)
            assert row[f"d_{key}"] == pytest.approx(row[full] - table[0][full], abs=1e-12)

    # Deterministic for a seed; another seed gives other draws.
    assert analyze_perturb.condition_table(run, n_boot=n_boot, seed=seed) == table
    assert analyze_perturb.condition_table(run, n_boot=n_boot, seed=6) != table


def test_thresholds_are_fixed_from_the_calibration_scores_only():
    run = _run()
    table = analyze_perturb.condition_table(run, n_boot=20)
    # Moving the calibration scores moves every rate; the evaluation scores never feed a threshold.
    for cat in run["cats"].values():
        cat["cal_score"] = cat["cal_score"] + np.float32(10.0)
    moved = analyze_perturb.condition_table(run, n_boot=20)
    assert all(row["fpr"] == 0.0 for row in moved) and table[0]["fpr"] > 0.0
    assert [row["macro_image_auroc"] for row in moved] == [row["macro_image_auroc"] for row in table]


def test_run_without_calibration_scores_has_accuracy_only():
    run = _run(cal=0)
    table = analyze_perturb.condition_table(run, n_boot=50)
    for row in table:
        assert row["fpr"] is None and row["fpr_ci"] is None and row["tpr"] is None and row["tpr_ci"] is None
        assert 0.0 < row["macro_image_auroc"] <= 1.0
    shifted, blurred = _row(table, SHIFTED), _row(table, BLURRED)
    assert shifted["d_fpr"] is None and shifted["d_fpr_ci"] is None
    assert shifted["d_tpr"] is None and shifted["d_tpr_ci"] is None
    assert shifted["d_auroc"] == 0.0 and blurred["d_auroc"] < -0.1
    assert "d_fpr" not in table[0]
    with pytest.raises(ValueError, match="fixed thresholds"):
        analyze_perturb.h9(table)
    text = analyze_perturb.format_tables({"dm": table}, None)
    assert "| clean |" in text and "| - | - | - | - |" in text and "H9" not in text
    assert analyze_perturb.describe(run)["categories_guaranteed"] is None

    # Calibration scores for some categories only: a broken run, not a run without thresholds.
    partly = _run()
    partly["cats"]["beta"]["cal_score"] = np.empty(0, dtype=np.float32)
    with pytest.raises(ValueError, match="beta"):
        analyze_perturb.condition_table(partly, n_boot=20)


def test_inconsistent_runs_are_rejected():
    run = _run()
    run["cats"]["beta"]["conditions"] = run["cats"]["beta"]["conditions"][::-1].copy()
    with pytest.raises(ValueError, match="other conditions"):
        analyze_perturb.condition_table(run, n_boot=20)

    run = _run()
    for cat in run["cats"].values():
        cat["conditions"] = cat["conditions"][1:]
        cat["scores"] = cat["scores"][1:]
    with pytest.raises(ValueError, match="clean"):
        analyze_perturb.condition_table(run, n_boot=20)

    run = _run()
    run["cats"]["alpha"]["scores"] = run["cats"]["alpha"]["scores"][:, :-1]
    with pytest.raises(ValueError, match="do not match"):
        analyze_perturb.condition_table(run, n_boot=20)

    run = _run()
    run["cats"]["gamma"]["scores"][3, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        analyze_perturb.condition_table(run, n_boot=20)

    with pytest.raises(ValueError, match="no categories"):
        analyze_perturb.condition_table({"meta": {}, "cats": {}}, n_boot=20)


def test_load_reads_a_run_directory_in_run_order(tmp_path):
    run = _run()
    _write(run, tmp_path / "run")
    loaded = analyze_perturb.load(tmp_path / "run")
    assert loaded["meta"] == run["meta"] and list(loaded["cats"]) == list(SIZES)
    for name, cat in run["cats"].items():
        assert set(loaded["cats"][name]) == set(cat)
        for key, value in cat.items():
            assert np.array_equal(loaded["cats"][name][key], value)
    assert analyze_perturb.condition_table(loaded, n_boot=30) == analyze_perturb.condition_table(
        run, n_boot=30
    )
    info = analyze_perturb.describe(loaded)
    assert (info["n_normal"], info["n_defect"], info["n_cal"]) == (250, 64, 300)
    assert info["categories"] == list(SIZES) and info["categories_guaranteed"] == 3
    assert info["clean_max_rel_diff"] == pytest.approx(3e-5) and info["source_commit"] == "src1234"


def test_format_tables():
    tables = {m: analyze_perturb.condition_table(_run(method=m), n_boot=50) for m in ("p0", "d-s")}
    verdict = {"method": "p0", **analyze_perturb.h9(tables["p0"])}
    text = analyze_perturb.format_tables(tables, verdict)
    lines = text.splitlines()
    assert lines[0] == "method p0" and "method d-s" in lines
    assert sum(line.startswith("| clean |") for line in lines) == 2
    assert sum(line.startswith(f"| {SHIFTED} |") for line in lines) == 2
    clean = next(line for line in lines if line.startswith("| clean |"))
    assert clean.count(" - ") == 3  # the clean row has nothing to be compared with
    shifted = next(line for line in lines if line.startswith(f"| {SHIFTED} |"))
    assert "+0.00 [+0.00, +0.00]" in shifted  # the AUROC change, signed and in points
    assert lines[-1] == f"H9 (p0): 지지 (conditions: {SHIFTED})"
    rejected = analyze_perturb.format_tables(tables, {"verdict": "기각", "conditions": []})
    assert rejected.splitlines()[-1] == "H9: 기각 (conditions: none)"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_perturb.paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(analyze_perturb.paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(analyze_perturb, "N_BOOT", 100)
    # The synthetic categories stand for the full set that conclusions are drawn over.
    monkeypatch.setattr(analyze_perturb, "CATEGORIES", tuple(SIZES))
    return tmp_path


def _only(run, categories):
    """The run as `run_perturb --categories ... --overwrite` leaves it: run.json lists only these."""
    meta = dict(run["meta"])
    meta["categories"] = [info for info in meta["categories"] if info["category"] in categories]
    return {"meta": meta, "cats": run["cats"]}


def test_cli_writes_every_table_and_judges_h9_on_p0(project, capsys):
    outputs = project / "outputs"
    _write(_run(method="p0"), outputs / "perturb-p0-dev")
    _write(_run(seed=1, method="d-s"), outputs / "perturb-d-s-dev")
    _write(_run(seed=2, cal=0, method="dm"), outputs / "perturb-dm-dev")
    analyze_perturb.main(["--protocol", "dev"])
    text = capsys.readouterr().out
    assert "method p0" in text and "method d-s" in text and "method dm" in text
    assert f"H9 (p0): 지지 (conditions: {SHIFTED})" in text

    path = project / "reports" / "stage3" / "perturb-dev.json"
    assert b"\r" not in path.read_bytes()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["protocol"] == "dev" and report["alpha"] == 0.05
    assert (report["n_boot"], report["seed"]) == (100, 0)
    assert list(report["tables"]) == ["p0", "d-s", "dm"] == list(report["runs"])
    assert all(len(table) == 16 for table in report["tables"].values())
    assert report["h9"] == {"method": "p0", "verdict": "지지", "conditions": [SHIFTED]}
    assert report["runs"]["p0"]["n_cal"] == 300 and report["runs"]["dm"]["n_cal"] == 0
    assert report["runs"]["d-s"]["protocol"] == "dev" and report["runs"]["d-s"]["commit"] == "abc1234"
    assert report["tables"]["dm"][1]["fpr"] is None
    expected = analyze_perturb.condition_table(analyze_perturb.load(outputs / "perturb-p0-dev"))
    assert report["tables"]["p0"] == json.loads(json.dumps(expected))
    # What the tables stand on is said next to them.
    assert report["categories"] == list(SIZES) and report["complete"] is True
    assert "categories (3 of 3): alpha, beta, gamma" in text and "partial" not in text
    assert "p0: 250 normal and 64 defect images, 300 calibration scores" in text


def test_cli_judges_h9_on_p0_whatever_the_order_of_the_methods(project, capsys):
    outputs = project / "outputs"
    # p0 with thresholds far above every score: no false positive under any condition, so H9 is
    # rejected for it. The d-s run alone would support it.
    p0 = _run(method="p0")
    for cat in p0["cats"].values():
        cat["cal_score"] = cat["cal_score"] + np.float32(10.0)
    _write(p0, outputs / "perturb-p0-dev")
    _write(_run(seed=1, method="d-s"), outputs / "perturb-d-s-dev")
    analyze_perturb.main(["--protocol", "dev", "--methods", "d-s", "p0"])
    text = capsys.readouterr().out
    report = json.loads((project / "reports" / "stage3" / "perturb-dev.json").read_text(encoding="utf-8"))
    assert list(report["tables"]) == ["d-s", "p0"]
    assert analyze_perturb.h9(report["tables"]["d-s"])["conditions"] == [SHIFTED]
    assert report["h9"] == {"method": "p0", "verdict": "기각", "conditions": []}
    assert text.splitlines()[-1] == "H9 (p0): 기각 (conditions: none)"


def test_cli_refuses_a_test_run_that_covers_part_of_the_categories(project, capsys):
    outputs = project / "outputs"
    # A finished three-category run redone for one category: the other npz files are still on disk,
    # run.json lists `alpha` only.
    _write(_only(_run(method="p0", protocol="test"), ["alpha"]), outputs / "perturb-p0-test")
    assert (outputs / "perturb-p0-test" / "gamma.npz").exists()
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "test", "--methods", "p0"])
    err = capsys.readouterr()
    assert "['alpha']" in err.err and "beta" in err.err and "H9" not in err.out
    assert not (project / "reports").exists()

    # The same set but in another order is not the registered resampling either.
    run = _run(method="d-s", protocol="test")
    run["meta"]["categories"] = run["meta"]["categories"][::-1]
    _write(run, outputs / "perturb-d-s-test")
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "test", "--methods", "d-s"])
    assert "gamma" in capsys.readouterr().err and not (project / "reports").exists()


def test_cli_dev_run_on_part_of_the_categories_is_marked_and_not_judged(project, capsys):
    outputs = project / "outputs"
    _write(_only(_run(method="p0"), ["alpha", "beta"]), outputs / "perturb-p0-dev")
    _write(_only(_run(seed=1, method="d-s"), ["alpha", "beta"]), outputs / "perturb-d-s-dev")
    analyze_perturb.main(["--protocol", "dev", "--methods", "p0", "d-s"])
    text = capsys.readouterr().out
    # The code check may run on a few categories, but it says so and gives no verdict.
    assert "categories (2 of 3): alpha, beta" in text
    assert "partial run: H9 is not judged" in text and "H9 (p0)" not in text
    assert "지지" not in text and "기각" not in text
    report = json.loads((project / "reports" / "stage3" / "perturb-dev.json").read_text(encoding="utf-8"))
    assert report["categories"] == ["alpha", "beta"] and report["complete"] is False
    assert report["h9"] is None
    assert report["runs"]["p0"]["categories"] == ["alpha", "beta"] and report["runs"]["p0"]["n_normal"] == 130
    # The table of the two categories is still what it says it is.
    direct = _run(method="p0")
    del direct["cats"]["gamma"]
    assert report["tables"]["p0"] == json.loads(json.dumps(analyze_perturb.condition_table(direct)))


def test_methods_must_cover_the_same_categories(project, capsys):
    outputs = project / "outputs"
    _write(_only(_run(method="p0"), ["alpha"]), outputs / "perturb-p0-dev")
    _write(_run(seed=1, method="d-s"), outputs / "perturb-d-s-dev")
    dirs = {m: outputs / f"perturb-{m}-dev" for m in ("p0", "d-s")}
    with pytest.raises(ValueError, match="different categories"):
        analyze_perturb.build_report(dirs, n_boot=20)
    with pytest.raises(ValueError, match="different categories"):
        analyze_perturb.main(["--protocol", "dev", "--methods", "p0", "d-s"])
    assert capsys.readouterr().out == "" and not (project / "reports").exists()
    # Each on its own is a table of what it covers.
    assert analyze_perturb.build_report({"d-s": dirs["d-s"]}, n_boot=20)["complete"] is True
    alone = analyze_perturb.build_report({"p0": dirs["p0"]}, n_boot=20)
    assert alone["complete"] is False and alone["h9"] is None and len(alone["tables"]["p0"]) == 16


def test_the_full_set_is_the_twelve_visa_categories(tmp_path):
    from defect_inspect.visa import CATEGORIES

    assert analyze_perturb.CATEGORIES == CATEGORIES and len(CATEGORIES) == 12
    # Three made-up categories are not the set H9 is registered on.
    _write(_run(method="p0"), tmp_path / "three")
    report = analyze_perturb.build_report({"p0": tmp_path / "three"}, n_boot=20)
    assert report["complete"] is False and report["h9"] is None

    # The same data under the twelve real names (each synthetic category four times) is.
    three = _run(method="p0")
    twelve = {"meta": dict(three["meta"]), "cats": {}}
    for i, category in enumerate(CATEGORIES):
        twelve["cats"][category] = three["cats"][list(SIZES)[i % 3]]
    twelve["meta"]["categories"] = [{"category": c} for c in CATEGORIES]
    _write(twelve, tmp_path / "twelve")
    report = analyze_perturb.build_report({"p0": tmp_path / "twelve"}, n_boot=20)
    assert report["complete"] is True and report["categories"] == list(CATEGORIES)
    assert report["h9"] == {"method": "p0", "verdict": "지지", "conditions": [SHIFTED]}


def test_cli_without_p0_gives_no_verdict_and_no_write_prints_only(project, capsys):
    outputs = project / "outputs"
    _write(_run(method="d-s", protocol="test"), outputs / "perturb-d-s-test")
    analyze_perturb.main(["--protocol", "test", "--methods", "d-s", "--no-write"])
    text = capsys.readouterr().out
    assert "method d-s" in text and "H9" not in text
    assert not (project / "reports").exists()


def test_cli_rejects_missing_and_mismatched_runs(project, capsys):
    outputs = project / "outputs"
    _write(_run(method="p0"), outputs / "perturb-p0-dev")
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "dev"])  # d-s and dm have not been run
    assert "perturb-d-s-dev" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "dev", "--methods"])
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "dev", "--methods", "p1"])
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--methods", "p0"])  # the protocol must be named

    # A run directory that holds another method or protocol than its name says.
    _write(_run(method="d-s"), outputs / "perturb-p0-test")
    with pytest.raises(ValueError, match="not of 'p0'"):
        analyze_perturb.main(["--protocol", "test", "--methods", "p0"])
    _write(_run(method="d-s", protocol="dev"), outputs / "perturb-d-s-test")
    with pytest.raises(SystemExit):
        analyze_perturb.main(["--protocol", "test", "--methods", "d-s"])
    assert not (project / "reports").exists()

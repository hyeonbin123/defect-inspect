"""analyze_grid on synthetic grid runs (numpy only: this file runs in the torch-free CI environment).

Every number of `grid_table` is recomputed from the registered rules with code that shares nothing with
compare.py except the random draws (seed 0, normals then defects of each category from one generator).
"""

import json
import math
from fractions import Fraction

import numpy as np
import pytest

from defect_inspect import analyze_grid, compare, paths

N_BOOT = 200
CATS = {"a": (31, 12, 90), "b": (45, 10, 130)}  # evaluation normals, defects, pool normals
RATIOS = [0.1, 0.01, 0.001]


def write_grid(root, setting, ratios, seed, cats=CATS, listed=None, protocol="dev"):
    """A folder in the run_grid layout. Smaller ratios separate normals and defects less well."""
    out = root / f"grid-{setting}-{protocol}"
    rng = np.random.default_rng(seed)
    base = {
        c: (rng.normal(0, 1, n), rng.normal(1.5, 1, p), rng.normal(0, 1, m)) for c, (n, p, m) in cats.items()
    }
    for k, ratio in enumerate(sorted(ratios, reverse=True)):
        folder = out / f"r{ratio!r}"
        folder.mkdir(parents=True)
        noise = np.random.default_rng(1000 * seed + k)
        for c, (n, p, m) in cats.items():
            neg, pos, cal = base[c]
            # One decimal: many ties between normals, defects and the thresholds.
            scores = np.round(np.concatenate([neg, pos - 0.3 * k]) + noise.normal(0, 0.2 * k, n + p), 1)
            np.savez(
                folder / f"{c}.npz",
                eval_images=np.array([f"{c}/e{i}.JPG" for i in range(n + p)]),
                eval_labels=np.array([0] * n + [1] * p, dtype=np.int8),
                eval_defect_types=np.array([""] * n + ["hole"] * p),
                eval_score_full=scores.astype(np.float32),
                pool_images=np.array([f"{c}/p{i}.JPG" for i in range(m)]),
                pool_folds=np.arange(m) % 4 + 1,
                pool_score_oof=np.round(cal + noise.normal(0, 0.2 * k, m), 1).astype(np.float32),
                pixel_auroc=np.float64(0.9),
                aupro=np.float64(0.8 - 0.05 * k),
                pro_edges=np.linspace(0, 1, 11),
                pro_normal=noise.integers(1, 9, (n + p, 10)),
                pro_components=noise.integers(1, 9, (p, 10)),
                pro_component_image=np.arange(n, n + p),
                bank_rows=np.int64(round(ratio * m * 100)),
            )
        with open(folder / "run.json", "w", encoding="utf-8") as f:
            json.dump({"config": {"name": "x"}, "categories": [{"category": c} for c in cats]}, f)
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(
            {"setting": setting, "ratios": listed or sorted(ratios, reverse=True), "protocol": protocol}, f
        )
    return out


def _auroc(neg, pos):
    d = pos[:, None].astype(np.float64) - neg[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def _reference(run_dir, ratios, alpha):
    data = {r: {c: dict(np.load(run_dir / f"r{r!r}" / f"{c}.npz")) for c in CATS} for r in ratios}
    rng = np.random.default_rng(0)
    draws = {
        c: (rng.integers(0, n, (N_BOOT, n)), rng.integers(0, p, (N_BOOT, p))) for c, (n, p, _) in CATS.items()
    }
    out, boots = {}, {}
    for ratio, cats in data.items():
        point, boot = [], np.zeros(N_BOOT)
        fp = tp = n_neg = n_pos = fp95 = 0
        fp_b, tp_b = np.zeros(N_BOOT), np.zeros(N_BOOT)
        for c, z in cats.items():
            neg = z["eval_score_full"][z["eval_labels"] == 0]
            pos = z["eval_score_full"][z["eval_labels"] == 1]
            ni, pi = draws[c]
            point.append(_auroc(neg, pos))
            boot += np.array([_auroc(neg[ni[b]], pos[pi[b]]) for b in range(N_BOOT)]) / len(cats)
            cal = np.sort(z["pool_score_oof"].astype(np.float64))
            thr = cal[math.ceil((len(cal) + 1) * (1 - Fraction(str(alpha)))) - 1]
            fp, tp = fp + (neg > thr).sum(), tp + (pos > thr).sum()
            n_neg, n_pos = n_neg + len(neg), n_pos + len(pos)
            fp_b += (neg > thr)[ni].sum(axis=1)
            tp_b += (pos > thr)[pi].sum(axis=1)
            # The highest threshold that still catches 95% of the defects, rule score >= threshold.
            t95 = np.sort(pos)[::-1][math.ceil(Fraction("0.95") * len(pos)) - 1]
            fp95 += (neg >= t95).sum()
        boots[ratio] = boot
        out[ratio] = {
            "macro_image_auroc": np.mean(point),
            "macro_image_auroc_ci": np.percentile(boot, [2.5, 97.5]),
            "pooled_fpr_at_tpr95": fp95 / n_neg,
            "fpr": fp / n_neg,
            "fpr_ci": np.percentile(fp_b / n_neg, [2.5, 97.5]),
            "tpr": tp / n_pos,
            "tpr_ci": np.percentile(tp_b / n_pos, [2.5, 97.5]),
            "bank_rows_mean": np.mean([int(z["bank_rows"]) for z in cats.values()]),
            "macro_aupro": np.mean([float(z["aupro"]) for z in cats.values()]),
        }
    top = max(ratios)
    for ratio in ratios:
        if ratio != top:
            out[ratio]["diff"] = out[ratio]["macro_image_auroc"] - out[top]["macro_image_auroc"]
            out[ratio]["diff_ci"] = np.percentile(boots[ratio] - boots[top], [2.5, 97.5])
    return out


@pytest.fixture
def grids(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "N_BOOT", N_BOOT)
    return {
        "wrn50-256": write_grid(tmp_path, "wrn50-256", RATIOS, 1),
        "dinov2_vits14-392": write_grid(tmp_path, "dinov2_vits14-392", RATIOS, 2),
    }


@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_rows_match_the_registered_rules(grids, alpha):
    runs = {s: analyze_grid.load_grid(d) for s, d in grids.items()}
    rows = analyze_grid.grid_table(runs, alpha=alpha)
    assert [(r["setting"], r["ratio"]) for r in rows] == [(s, r) for s in grids for r in RATIOS]
    assert [r["key"] for r in rows[:3]] == ["wrn50-256-r0.1", "wrn50-256-r0.01", "wrn50-256-r0.001"]
    exact = ("macro_image_auroc", "macro_image_auroc_ci", "pooled_fpr_at_tpr95", "fpr", "fpr_ci", "tpr")
    for setting, run_dir in grids.items():
        want = _reference(run_dir, RATIOS, alpha)
        for row in (r for r in rows if r["setting"] == setting):
            ref = want[row["ratio"]]
            for key in (*exact, "tpr_ci"):
                np.testing.assert_allclose(row[key], ref[key], rtol=0, atol=1e-12, err_msg=key)
            assert row["bank_rows_mean"] == ref["bank_rows_mean"]
            assert row["macro_aupro"] == pytest.approx(ref["macro_aupro"])
            if row["ratio"] == 0.1:
                assert "vs_largest_ratio" not in row
            else:  # smaller ratio minus the largest one, from the same draws
                d = row["vs_largest_ratio"]
                assert d["diff"] == pytest.approx(ref["diff"], abs=1e-12) and d["diff"] < 0
                np.testing.assert_allclose(d["ci"], ref["diff_ci"], rtol=0, atol=1e-12)
    assert rows[0]["fpr"] != rows[0]["tpr"]  # the reference would not notice a swap of equal numbers


def test_rows_carry_the_numbers_of_compare_summarize(grids):
    runs = {s: analyze_grid.load_grid(d) for s, d in grids.items()}
    rows = analyze_grid.grid_table(runs)
    flat = [run for ratios in runs.values() for run in ratios.values()]
    boot = compare.Bootstrap(flat[0])
    assert [r["key"] for r in rows] == [run.name for run in flat]
    for row, run in zip(rows, flat, strict=True):
        summary = compare.summarize(run, boot)
        for key in ("macro_image_auroc", "macro_image_auroc_ci", "macro_aupro", "macro_aupro_ci"):
            assert row[key] == summary[key], key
        assert row["pooled_fpr_at_tpr95"] == summary["pooled_fpr_at_tpr95"]
        for key in ("fpr", "fpr_ci", "tpr", "tpr_ci"):
            assert row[key] == summary["fixed_threshold"][key], key


def test_every_row_uses_the_same_draws(grids):
    runs = {s: analyze_grid.load_grid(d) for s, d in grids.items()}
    forward = {r["key"]: r for r in analyze_grid.grid_table(runs)}
    backward = {r["key"]: r for r in analyze_grid.grid_table({s: runs[s] for s in reversed(list(runs))})}
    assert forward == backward


def test_each_run_is_resampled_once(grids, monkeypatch):
    """The interval of a row and the paired differences share one bootstrap of the macro AUROC."""
    runs = {"wrn50-256": analyze_grid.load_grid(grids["wrn50-256"])}
    calls = []
    real = compare.Bootstrap.macro_auroc

    def counting(self, run, keep_pos=None):
        calls.append(run.name)
        return real(self, run, keep_pos)

    monkeypatch.setattr(compare.Bootstrap, "macro_auroc", counting)
    rows = analyze_grid.grid_table(runs)
    assert sorted(calls) == sorted(r["key"] for r in rows)


def test_identical_runs_have_no_difference(grids):
    run = analyze_grid.load_grid(grids["wrn50-256"])[0.1]
    d = analyze_grid.grid_table({"twin": {0.1: run, 0.01: run}})[1]["vs_largest_ratio"]
    assert d["diff"] == 0 and d["ci"] == [0.0, 0.0] and d["verdict"] == "판정 불가"


def test_runs_over_other_images_are_refused(grids, tmp_path):
    other = write_grid(tmp_path, "wrn50-384", [0.1], 3, cats={"a": CATS["a"]})
    runs = {
        "wrn50-256": analyze_grid.load_grid(grids["wrn50-256"]),
        "wrn50-384": analyze_grid.load_grid(other),
    }
    with pytest.raises(ValueError, match="categories"):
        analyze_grid.grid_table(runs)


def test_load_grid_reads_only_what_the_last_call_wrote(tmp_path):
    out = write_grid(tmp_path, "wrn50-256", [0.1, 0.05, 0.01], 3, listed=[0.1, 0.01])  # r0.05 is stale
    runs = analyze_grid.load_grid(out)
    assert list(runs) == [0.1, 0.01] and runs[0.01].name == "wrn50-256-r0.01"
    (out / "run.json").unlink()  # an unfinished call has no run.json
    with pytest.raises(FileNotFoundError):
        analyze_grid.load_grid(out)


LATENCY = {
    "wrn50-256-r0.01": {"resize_ms": 31.0, "cpu_fp32": {"total_ms": 180.0}, "gpu_fp16": {"total_ms": 9.0}},
    "wrn50-256-r0.001": {"resize_ms": 31.0, "gpu_fp16": {"total_ms": 8.0}},
    "wrn50-256": {"resize_ms": 1.0, "cpu_fp32": {"total_ms": 1.0}},  # no ratio: the key of no row
}


def test_cpu_latency_is_resize_plus_inference():
    rows = [{"key": "wrn50-256-r0.1"}, {"key": "wrn50-256-r0.01"}, {"key": "wrn50-256-r0.001"}]
    analyze_grid.merge_latency(rows, LATENCY)
    assert "latency" not in rows[0] and "cpu_ms" not in rows[0]
    assert rows[1]["latency"] == LATENCY["wrn50-256-r0.01"] and rows[1]["cpu_ms"] == 211.0
    # Only the GPU was timed for this key: there is no CPU latency to report.
    assert rows[2]["latency"]["gpu_fp16"]["total_ms"] == 8.0 and "cpu_ms" not in rows[2]


def test_cpu_entry_without_the_resize_time_is_refused():
    with pytest.raises(ValueError, match="resize_ms"):
        analyze_grid.merge_latency([{"key": "k-r0.1"}], {"k-r0.1": {"cpu_fp32": {"total_ms": 120.0}}})


def test_table_shows_resize_plus_inference(grids):
    rows = analyze_grid.grid_table({"wrn50-256": analyze_grid.load_grid(grids["wrn50-256"])})
    analyze_grid.merge_latency(rows, LATENCY)
    lines = analyze_grid.format_table(rows).splitlines()
    assert lines[0].split(" | ")[-1] == "CPU ms (resize + inference) |"
    cells = {line.split(" | ")[1]: line.split(" | ")[-1] for line in lines[2:]}
    assert cells == {"0.1": "- |", "0.01": "211.0 |", "0.001": "- |"}
    assert all(line.count("|") == lines[0].count("|") for line in lines)


def _row(key, auroc, bank_rows, cpu_ms=None):
    setting, ratio = key.rsplit("-r", 1)
    row = {"key": key, "setting": setting, "ratio": float(ratio)}
    row.update(macro_image_auroc=auroc, bank_rows_mean=float(bank_rows))
    if cpu_ms is not None:
        row["cpu_ms"] = cpu_ms
    return row


def test_serving_choice_follows_the_registered_rule():
    rows = [
        _row("a-r0.1", 0.95, 1000, 200.5),  # the most accurate one is over the budget
        _row("a-r0.01", 0.93, 100, 200.0),  # exactly the budget counts as within
        _row("b-r0.1", 0.93, 50, 120.0),  # same AUROC as a-r0.01: the smaller bank wins
        _row("b-r0.01", 0.90, 5, 60.0),
        _row("c-r0.1", 0.99, 10),  # no CPU latency (e.g. the export failed): not a candidate
    ]
    pick = analyze_grid.choose_serving(rows)
    assert pick["key"] == "b-r0.1" and pick["setting"] == "b" and pick["ratio"] == 0.1
    assert pick["within_budget"] is True and pick["budget_ms"] == 200.0 and pick["cpu_ms"] == 120.0
    assert pick["candidates"] == ["a-r0.01", "b-r0.1", "b-r0.01"] and pick["unmeasured"] == ["c-r0.1"]
    assert analyze_grid.choose_serving(list(reversed(rows)))["key"] == "b-r0.1"

    # Nothing within the budget: the fastest configuration (H15 is then rejected).
    pick = analyze_grid.choose_serving(rows, budget_ms=50.0)
    assert pick["key"] == "b-r0.01" and pick["within_budget"] is False and pick["candidates"] == []
    assert analyze_grid.choose_serving([_row("c-r0.1", 0.99, 10)]) is None


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(compare, "N_BOOT", 50)
    write_grid(tmp_path / "outputs", "dinov2_vits14-392", [0.1, 0.01], 2)
    write_grid(tmp_path / "outputs", "wrn50-256", [0.1, 0.01], 1)
    write_grid(tmp_path / "outputs", "wrn50-32", [0.1], 3)  # not a registered setting
    return tmp_path


def _report(world, protocol="dev"):
    with open(world / "reports" / "stage4" / f"grid-{protocol}.json", encoding="utf-8") as f:
        return json.load(f)


def test_cli_defaults_to_the_registered_settings_that_were_run(world, capsys):
    (world / "reports" / "stage4").mkdir(parents=True)
    with open(world / "reports" / "stage4" / "latency.json", "w", encoding="utf-8") as f:
        json.dump(LATENCY, f)
    analyze_grid.main(["--protocol", "dev"])
    out, err = capsys.readouterr()
    report = _report(world)
    keys = ["wrn50-256-r0.1", "wrn50-256-r0.01", "dinov2_vits14-392-r0.1", "dinov2_vits14-392-r0.01"]
    assert [r["key"] for r in report["rows"]] == keys  # the registered order, not the folder order
    assert report["protocol"] == "dev" and report["alpha"] == 0.05
    assert report["rows"][1]["cpu_ms"] == 211.0 and "cpu_ms" not in report["rows"][0]
    # Only one row has a CPU latency, and it is over the budget.
    serving = report["serving"]
    assert serving["key"] == "wrn50-256-r0.01" and serving["within_budget"] is False
    assert serving["unmeasured"] == [k for k in keys if k != "wrn50-256-r0.01"]
    assert "CPU ms (resize + inference)" in out and "211.0" in out and "wrn50-256-r0.01" in out
    for skipped in ("wrn50-384", "dinov2_vits14-252", "dinov2_vits14-448"):
        assert skipped in err
    assert "wrn50-32" not in out + err


def test_cli_with_explicit_settings_and_without_latency(world, capsys):
    analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-32", "--no-write"])
    out = capsys.readouterr().out
    assert out.count("| wrn50-32 |") == 1 and "wrn50-256" not in out
    assert not (world / "reports").exists()
    analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-32"])
    report = _report(world)
    assert [r["key"] for r in report["rows"]] == ["wrn50-32-r0.1"] and "serving" not in report


def test_cli_refuses_missing_runs_and_a_missing_latency_file(world):
    with pytest.raises(SystemExit):  # no grid run of the test protocol at all
        analyze_grid.main(["--protocol", "test", "--no-write"])
    with pytest.raises(SystemExit):
        analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-256", "wrn50-384", "--no-write"])
    with pytest.raises(SystemExit):
        analyze_grid.main(["--protocol", "dev", "--latency", str(world / "nowhere.json"), "--no-write"])


def test_cli_does_not_choose_a_serving_configuration_on_the_test_protocol(world, capsys):
    write_grid(world / "outputs", "wrn50-256", [0.1, 0.01], 1, protocol="test")
    latency = world / "latency.json"
    with open(latency, "w", encoding="utf-8") as f:
        json.dump(LATENCY, f)
    analyze_grid.main(["--protocol", "test", "--latency", str(latency)])
    report = _report(world, "test")
    assert report["rows"][1]["cpu_ms"] == 211.0 and "serving" not in report
    assert "serving" not in capsys.readouterr().out

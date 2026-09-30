import json

import numpy as np
import pytest

from defect_inspect import analyze_m2ad, metrics, paths
from defect_inspect.analyze_m2ad import (
    ClusterBootstrap,
    condition_cell,
    condition_table,
    h11,
    h12,
    recal_cell,
    recal_table,
)
from defect_inspect.conditions import condition_names
from defect_inspect.stats import percentile_ci, stratified_indices

VIEWS = ("000", "120", "240")
LIGHTS = [f"{i:02d}" for i in range(2, 11)]
SYNTHETIC = [f"P:{name}" for name in condition_names(include_clean=False)]
CONDITIONS = ["S", *SYNTHETIC, *(f"R:{light}" for light in LIGHTS)]
RECAL_KEYS = [f"{light}:{n}" for light in LIGHTS for n in (8, 30)]
N_BOOT = 300


def _meta(categories):
    inspectors = [{"category": c, "view": v} for c in categories for v in VIEWS]
    return {"method": "p0", "commit": "abc1234", "device": "cpu", "check": False, "inspectors": inspectors}


def random_run(seed=0, sizes=(("Motor", 5, 8), ("Bird", 4, 7))) -> dict:
    """Random scores with ties, excluded images and one inspector that has a single defect image."""
    rng = np.random.default_rng(seed)
    inspectors = {}
    for category, n_normal, n_anomalous in sizes:
        n = n_normal + n_anomalous
        object_anomaly = np.array([0] * n_normal + [1] * n_anomalous, dtype=np.int8)
        specimens = np.array(
            [f"{i:03d}" for i in range(n_normal)] + [f"hole_{i}" for i in range(n_anomalous)]
        )
        for view in VIEWS:

            def draw(rows, object_anomaly=object_anomaly, n=n):
                scores = np.round(rng.normal(size=(rows, n)) + object_anomaly, 1).astype(np.float32)
                labels = np.tile(object_anomaly, (rows, 1)).astype(np.int8)
                labels[(rng.random((rows, n)) < 0.25) & (object_anomaly == 1)] = -1
                return scores, labels

            scores, labels = draw(len(CONDITIONS))
            recal_scores, recal_labels = draw(len(RECAL_KEYS))
            if (category, view) == ("Bird", "240"):
                labels[:, n_normal:] = -1
                labels[:, n_normal + 2] = 1  # resamples without this specimen have no defect image here
            inspectors[f"{category}_{view}"] = {
                "category": category,
                "view": view,
                "specimens": specimens,
                "object_anomaly": object_anomaly,
                "conditions": np.array(CONDITIONS),
                "scores": scores,
                "labels": labels,
                "threshold": np.float64(round(float(rng.normal(0.4, 0.2)), 1)),
                "cal_score": rng.normal(size=30).astype(np.float32),
                "recal_keys": np.array(RECAL_KEYS),
                "recal_scores": recal_scores,
                "recal_thresholds": np.round(rng.normal(0.4, 0.2, len(RECAL_KEYS)), 1),
                "recal_labels": recal_labels,
            }
    return {"meta": _meta([s[0] for s in sizes]), "inspectors": inspectors}


def flat_run(n_normal=20, n_anomalous=50, categories=("Motor", "Bird")) -> dict:
    """Threshold 0.5; every normal image scores 0 and every defect image 1, whatever the condition."""
    inspectors = {}
    object_anomaly = np.array([0] * n_normal + [1] * n_anomalous, dtype=np.int8)
    specimens = np.array([f"{i:03d}" for i in range(n_normal)] + [f"hole_{i}" for i in range(n_anomalous)])
    for category in categories:
        for view in VIEWS:
            inspectors[f"{category}_{view}"] = {
                "category": category,
                "view": view,
                "specimens": specimens,
                "object_anomaly": object_anomaly,
                "conditions": np.array(CONDITIONS),
                "scores": np.tile(object_anomaly, (len(CONDITIONS), 1)).astype(np.float32),
                "labels": np.tile(object_anomaly, (len(CONDITIONS), 1)).astype(np.int8),
                "threshold": np.float64(0.5),
                "cal_score": np.zeros(30, dtype=np.float32),
                "recal_keys": np.array(RECAL_KEYS),
                "recal_scores": np.tile(object_anomaly, (len(RECAL_KEYS), 1)).astype(np.float32),
                "recal_thresholds": np.full(len(RECAL_KEYS), 0.5),
                "recal_labels": np.tile(object_anomaly, (len(RECAL_KEYS), 1)).astype(np.int8),
            }
    return {"meta": _meta(categories), "inspectors": inspectors}


def flag(run, names, specimens, views=VIEWS, key="scores", categories=("Motor", "Bird")):
    """Raise the normal specimens `specimens` above the threshold under the conditions `names`."""
    index_key = "conditions" if key == "scores" else "recal_keys"
    for arr in run["inspectors"].values():
        if arr["view"] not in views or arr["category"] not in categories:
            continue
        for name in names:
            arr[key][arr[index_key].tolist().index(name), list(specimens)] = 1.0


def direct_counts(run, cell, n_boot, seed=0, with_auroc=True) -> np.ndarray:
    """Literal resampling: draw specimens, take their images from every view, count.

    Returns [n_boot, 5]: false positives, normal images, detected defects, defect images, and the mean
    inspector AUROC of the resample (nan when `with_auroc` is off).
    """
    inspectors = list(run["inspectors"].values())
    categories = list(dict.fromkeys(arr["category"] for arr in inspectors))
    groups, sizes = [], []
    for category in categories:
        anomalous = next(a for a in inspectors if a["category"] == category)["object_anomaly"]
        groups.append((np.flatnonzero(anomalous == 0), np.flatnonzero(anomalous == 1)))
        sizes += [len(groups[-1][0]), len(groups[-1][1])]
    draws = stratified_indices(sizes, n_boot=n_boot, seed=seed)
    out = np.empty((n_boot, 5))
    for b in range(n_boot):
        fp = n_neg = tp = n_pos = 0
        aurocs = []
        for i, category in enumerate(categories):
            drawn = np.concatenate([groups[i][0][draws[2 * i][b]], groups[i][1][draws[2 * i + 1][b]]])
            for arr, (scores, labels, threshold) in zip(inspectors, cell, strict=True):
                if arr["category"] != category:
                    continue
                s, lab = scores[drawn].astype(np.float64), labels[drawn]
                fp += int(((s > threshold) & (lab == 0)).sum())
                n_neg += int((lab == 0).sum())
                tp += int(((s > threshold) & (lab == 1)).sum())
                n_pos += int((lab == 1).sum())
                if with_auroc and (lab == 1).any():
                    aurocs.append(metrics.auroc(s[lab == 0], s[lab == 1]))
        out[b] = (fp, n_neg, tp, n_pos, np.mean(aurocs) if aurocs else np.nan)
    return out


def direct_samples(run, cell, n_boot, seed=0) -> np.ndarray:
    """[n_boot, 3] from the literal loop: pooled FPR, pooled detection rate, mean inspector AUROC."""
    counts = direct_counts(run, cell, n_boot, seed)
    return np.stack([counts[:, 0] / counts[:, 1], counts[:, 2] / counts[:, 3], counts[:, 4]], axis=1)


def direct_pooled(run, cells, n_boot) -> tuple[np.ndarray, np.ndarray]:
    """Pooled FPR and detection rate samples over several cells, from the literal loop."""
    counts = sum(direct_counts(run, cell, n_boot, with_auroc=False)[:, :4] for cell in cells)
    return counts[:, 0] / counts[:, 1], counts[:, 2] / counts[:, 3]


def write_run(run: dict, path) -> None:
    path.mkdir(parents=True)
    for name, arr in run["inspectors"].items():
        np.savez(path / f"{name}.npz", **{k: v for k, v in arr.items() if k not in ("category", "view")})
    (path / "run.json").write_text(json.dumps(run["meta"]), encoding="utf-8")


# ------------------------------------------------------------------------------------------ the bootstrap


def test_registered_resampling_constants():
    assert analyze_m2ad.N_BOOT == 2000 and analyze_m2ad.SEED == 0
    assert (analyze_m2ad.H12_N, analyze_m2ad.H12_LIMIT, analyze_m2ad.JUDGED_METHOD) == (30, 0.10, "p0")
    # Without an explicit number the bootstrap takes the registered 2,000 resamples with seed 0.
    run = random_run()
    boot = ClusterBootstrap(run)
    assert boot.n_boot == 2000 and boot.weights["Motor"].shape == (2000, 13)
    assert np.array_equal(boot.weights["Bird"], ClusterBootstrap(run, n_boot=2000, seed=0).weights["Bird"])


def test_cluster_bootstrap_matches_a_direct_loop():
    run = random_run()
    boot = ClusterBootstrap(run, n_boot=N_BOOT, seed=0)
    skipped = False
    for name in ("S", "P:blur-2", "R:07"):
        cell = condition_cell(run, name)
        direct = direct_samples(run, cell, N_BOOT)
        point, samples = boot.counts([cell])
        np.testing.assert_allclose(samples[:, 0] / samples[:, 1], direct[:, 0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(samples[:, 2] / samples[:, 3], direct[:, 1], rtol=0, atol=1e-12)
        value, auroc_samples = boot.auroc(cell)
        np.testing.assert_allclose(auroc_samples, direct[:, 2], rtol=0, atol=1e-12)
        # Observed values: plain counts and the mean of the inspectors' AUROC.
        flags = [(s.astype(np.float64) > t, lab) for s, lab, t in cell]
        assert point.tolist() == [
            sum(int((f & (lab == 0)).sum()) for f, lab in flags),
            sum(int((lab == 0).sum()) for _, lab in flags),
            sum(int((f & (lab == 1)).sum()) for f, lab in flags),
            sum(int((lab == 1).sum()) for _, lab in flags),
        ]
        assert value == pytest.approx(
            np.mean([metrics.auroc(s[lab == 0], s[lab == 1]) for s, lab, _ in cell])
        )
        # The Bird 240 inspector drops out of the resamples that miss its only defect image.
        weights = boot.weights["Bird"][:, 4 + 2]
        skipped = skipped or bool((weights == 0).any())
    assert skipped


def test_cluster_bootstrap_draws_specimens_not_images():
    run = random_run()
    boot = ClusterBootstrap(run, n_boot=N_BOOT, seed=0)
    assert boot.categories == ["Motor", "Bird"] and len(boot.names) == 6
    for category, n_normal, n_anomalous in (("Motor", 5, 8), ("Bird", 4, 7)):
        weights = boot.weights[category]
        assert weights.shape == (N_BOOT, n_normal + n_anomalous)
        # Every resample keeps the number of normal and of anomalous specimens.
        assert (weights[:, :n_normal].sum(axis=1) == n_normal).all()
        assert (weights[:, n_normal:].sum(axis=1) == n_anomalous).all()
    # The draws are those of `stratified_indices` over (normals, anomalous) of each category in order.
    draws = stratified_indices([5, 8, 4, 7], n_boot=N_BOOT, seed=0)
    assert boot.weights["Motor"][3, :5].tolist() == np.bincount(draws[0][3], minlength=5).tolist()
    assert boot.weights["Bird"][7, 4:].tolist() == np.bincount(draws[3][7], minlength=7).tolist()
    # Same seed, same draws; another seed, other draws.
    again = ClusterBootstrap(run, n_boot=N_BOOT, seed=0)
    assert np.array_equal(again.weights["Bird"], boot.weights["Bird"])
    other = ClusterBootstrap(run, n_boot=N_BOOT, seed=1)
    assert not np.array_equal(other.weights["Bird"], boot.weights["Bird"])
    # A normal specimen drawn twice counts its images of all three views twice.
    cell = condition_cell(run, "S")
    _, samples = boot.counts([cell])
    assert (samples[:, 1] == 3 * 5 + 3 * 4).all()


def test_condition_table_against_the_direct_loop():
    run = random_run(seed=3)
    table = condition_table(run, n_boot=N_BOOT, seed=0)
    assert [row["condition"] for row in table] == CONDITIONS
    base = direct_samples(run, condition_cell(run, "S"), N_BOOT)
    assert "d_fpr" not in table[0] and "d_auroc_ci" not in table[0]
    for name in ("P:jpeg-3", "R:02", "R:10"):
        row = table[CONDITIONS.index(name)]
        direct = direct_samples(run, condition_cell(run, name), N_BOOT)
        for j, key in enumerate(("fpr", "tpr", "auroc")):
            assert row[f"{key}_ci"] == pytest.approx(percentile_ci(direct[:, j]))
            assert row[f"d_{key}"] == pytest.approx(row[key] - table[0][key])
            assert row[f"d_{key}_ci"] == pytest.approx(percentile_ci(direct[:, j] - base[:, j]))
        assert row["fpr"] == row["false_positives"] / row["n_normal"]
        assert row["tpr"] == row["detected"] / row["n_defect"]
    assert table[0]["n_normal"] == 3 * 5 + 3 * 4


def test_auroc_leaves_out_an_inspector_without_defect_images_everywhere():
    run = random_run()
    arr = run["inspectors"]["Bird_120"]
    k = CONDITIONS.index("R:07")
    arr["labels"][k, arr["object_anomaly"] == 1] = -1  # no defect image under this condition
    cell = condition_cell(run, "R:07")
    defined = [metrics.auroc(s[lab == 0], s[lab == 1]) for s, lab, _ in cell if (lab == 1).any()]
    assert len(defined) == 5
    # The observed value averages the inspectors that have an AUROC, exactly as every resample does.
    boot = ClusterBootstrap(run, n_boot=N_BOOT, seed=0)
    value, samples = boot.auroc(cell)
    assert value == pytest.approx(np.mean(defined))
    direct = direct_samples(run, cell, N_BOOT)
    np.testing.assert_allclose(samples, direct[:, 2], rtol=0, atol=1e-12)
    row = condition_table(run, n_boot=N_BOOT)[k]
    assert row["condition"] == "R:07" and row["auroc"] == pytest.approx(np.mean(defined))
    assert row["auroc_ci"][0] <= row["auroc"] <= row["auroc_ci"][1]
    assert np.isfinite(row["d_auroc"]) and np.isfinite(row["d_auroc_ci"]).all()

    # No inspector has a defect image: there is no AUROC, in the observed value and in the resamples.
    for other in run["inspectors"].values():
        other["labels"][k, other["object_anomaly"] == 1] = -1
    value, samples = boot.auroc(condition_cell(run, "R:07"))
    assert np.isnan(value) and np.isnan(samples).all()


def test_condition_table_of_a_flat_run():
    run = flat_run()
    flag(run, ["R:04"], range(5))
    flag(run, ["P:gamma-1"], [0], views=("000",))
    table = {row["condition"]: row for row in condition_table(run, n_boot=N_BOOT)}
    assert (table["S"]["n_normal"], table["S"]["n_defect"]) == (120, 300)
    assert table["S"]["fpr"] == 0 and table["S"]["tpr"] == 1 and table["S"]["auroc"] == 1
    assert table["S"]["fpr_ci"] == [0, 0]
    assert table["R:04"]["fpr"] == 0.25 and table["R:04"]["d_fpr"] == 0.25
    assert table["R:04"]["d_fpr_ci"][0] > 0
    assert table["R:04"]["d_tpr"] == 0 and table["R:04"]["d_tpr_ci"] == [0, 0]
    assert table["P:gamma-1"]["fpr"] == pytest.approx(2 / 120)  # one view of one specimen per category
    # A quarter of the normals now ties with the defects: AUROC = 0.75 + 0.25 / 2.
    assert table["R:04"]["auroc"] == pytest.approx(0.875)
    assert table["R:04"]["d_auroc"] == pytest.approx(-0.125)
    assert table["R:04"]["d_auroc_ci"][1] < 0


# ----------------------------------------------------------------------------------------------------- H11


def test_h11_supported():
    run = flat_run()
    flag(run, [f"R:{light}" for light in LIGHTS], range(10))  # +50 points under every real illumination
    flag(run, ["P:blur-3"], [0])  # +5 points at most under a synthetic condition
    result = h11(run, n_boot=N_BOOT)
    assert result["max_synthetic_condition"] == "P:blur-3"
    assert result["max_synthetic_increase"] == pytest.approx(0.05)
    assert result["mean_real_increase"] == pytest.approx(0.5)
    assert result["statistic"] == pytest.approx(0.05 - 0.25)
    assert result["ci"][1] < 0 and result["verdict"] == "지지"
    assert (result["n_synthetic"], result["n_real"]) == (15, 9)


def test_h11_rejected():
    run = flat_run()
    flag(run, ["P:jpeg-3"], range(20))
    flag(run, ["P:jpeg-2"], range(10))
    result = h11(run, n_boot=N_BOOT)
    assert result["max_synthetic_condition"] == "P:jpeg-3"
    assert result["statistic"] == pytest.approx(1.0) and result["ci"] == pytest.approx([1.0, 1.0])
    assert result["mean_real_increase"] == 0 and result["verdict"] == "기각"


def test_h11_undecided():
    run = flat_run()
    flag(run, ["P:gamma-2"], [0])  # +5 points
    flag(run, [f"R:{light}" for light in LIGHTS], [1, 2])  # +10 points, half of it is 5
    result = h11(run, n_boot=N_BOOT)
    assert result["statistic"] == pytest.approx(0.0)
    assert result["ci"][0] < 0 < result["ci"][1]
    assert result["verdict"] == "판정 불가"


def test_h11_uses_the_mean_of_the_real_conditions_and_differences_to_s():
    run = flat_run()
    # Specimen 5 is flagged under every condition, S included: it moves no difference.
    flag(run, CONDITIONS, [5])
    flag(run, ["R:02", "R:03", "R:04"], range(12))  # three of nine illuminations: mean increase 0.2
    flag(run, ["P:shift-3"], [0, 1])
    result = h11(run, n_boot=N_BOOT)
    assert result["mean_real_increase"] == pytest.approx(3 * (11 / 20) / 9)
    assert result["max_synthetic_increase"] == pytest.approx(0.1)
    assert result["statistic"] == pytest.approx(0.1 - 0.5 * 3 * (11 / 20) / 9)


def test_h11_takes_the_maximum_again_in_every_resample():
    run = flat_run(categories=("Motor",))
    flag(run, ["P:brightness-1"], [0, 1])
    flag(run, ["P:jpeg-1"], [2, 3, 4])
    flag(run, [f"R:{light}" for light in LIGHTS], [5, 6, 7])
    result = h11(run, n_boot=N_BOOT)
    assert result["max_synthetic_condition"] == "P:jpeg-1"

    draws = stratified_indices([20, 50], n_boot=N_BOOT, seed=0)[0]  # the normal specimens
    counts = np.stack([(draws == k).sum(axis=1) for k in range(20)], axis=1) / 20
    first, second, real = counts[:, :2].sum(axis=1), counts[:, 2:5].sum(axis=1), counts[:, 5:8].sum(axis=1)
    expected = np.maximum(first, second) - 0.5 * real
    assert result["ci"] == pytest.approx(percentile_ci(expected))
    assert result["max_synthetic_increase_ci"] == pytest.approx(percentile_ci(np.maximum(first, second)))
    # Following only the condition that is largest in the observed data would give another interval.
    assert (first > second).any()
    assert result["statistic"] == pytest.approx(0.15 - 0.5 * 0.15)


def test_h11_interval_averages_the_nine_real_conditions_in_every_resample():
    run = flat_run(categories=("Motor",))
    flag(run, ["P:jpeg-1"], [0, 1])
    flag(run, ["R:02", "R:05", "R:09"], [2, 3, 4, 5])  # three of the nine illuminations
    flag(run, ["R:03"], [6])  # and a fourth one with another specimen
    result = h11(run, n_boot=N_BOOT)
    assert result["mean_real_increase"] == pytest.approx((3 * 0.2 + 0.05) / 9)
    assert result["statistic"] == pytest.approx(0.1 - 0.5 * (3 * 0.2 + 0.05) / 9)

    draws = stratified_indices([20, 50], n_boot=N_BOOT, seed=0)[0]  # the normal specimens
    counts = np.stack([(draws == k).sum(axis=1) for k in range(20)], axis=1) / 20
    synthetic = counts[:, :2].sum(axis=1)
    real_mean = (3 * counts[:, 2:6].sum(axis=1) + counts[:, 6]) / 9  # five illuminations add nothing
    assert result["mean_real_increase_ci"] == pytest.approx(percentile_ci(real_mean))
    assert result["ci"] == pytest.approx(percentile_ci(synthetic - 0.5 * real_mean))
    # The largest real increase instead of the mean of the nine would give another interval.
    real_max = np.maximum(counts[:, 2:6].sum(axis=1), counts[:, 6])
    assert result["ci"] != pytest.approx(percentile_ci(synthetic - 0.5 * real_max))


def test_h11_is_undecided_when_the_interval_ends_at_zero():
    # Nothing is flagged anywhere: the statistic is 0 in every resample, the interval is [0, 0]. The rule
    # needs an upper end below 0 to support and a lower end above 0 to reject, so neither holds.
    result = h11(flat_run(), n_boot=N_BOOT)
    assert result["statistic"] == 0 and result["ci"] == [0, 0]
    assert result["verdict"] == "판정 불가"
    # The same with increases that cancel exactly in every resample: real +100 points in both categories,
    # one synthetic condition +100 points in one category (half of the pooled normal images).
    run = flat_run()
    flag(run, [f"R:{light}" for light in LIGHTS], range(20))
    flag(run, ["P:blur-2"], range(20), categories=("Motor",))
    result = h11(run, n_boot=N_BOOT)
    assert result["max_synthetic_increase"] == pytest.approx(0.5) and result["mean_real_increase"] == 1
    assert result["ci"] == pytest.approx([0, 0], abs=1e-12) and result["verdict"] == "판정 불가"


@pytest.mark.parametrize("always", [[("Motor", "000")], [("Motor", "000"), ("Motor", "120")]])
def test_h11_exact_tie_is_not_decided_by_float_rounding(always):
    # Whole (category, view) blocks of normal images are flagged, so every rate is the same in every
    # resample: S = s/120, the synthetic condition (s + 20)/120, every real one (s + 40)/120. The statistic
    # is exactly 0, but in floats it comes out as -5.6e-17 (s = 20) or +2.8e-17 (s = 40).
    real = [f"R:{light}" for light in LIGHTS]
    blocks = [(category, view) for category in ("Motor", "Bird") for view in VIEWS]
    free = [block for block in blocks if block not in always]
    run = flat_run()
    for category, view in always:
        flag(run, CONDITIONS, range(20), views=(view,), categories=(category,))
    flag(run, ["P:blur-2", *real], range(20), views=(free[0][1],), categories=(free[0][0],))
    flag(run, real, range(20), views=(free[1][1],), categories=(free[1][0],))
    result = h11(run, n_boot=N_BOOT)
    assert result["max_synthetic_increase"] == pytest.approx(1 / 6)
    assert result["mean_real_increase"] == pytest.approx(1 / 3)
    assert result["ci"] == pytest.approx([0, 0], abs=1e-12)
    assert result["verdict"] == "판정 불가"


def test_h11_needs_both_kinds_of_conditions():
    run = flat_run()
    for arr in run["inspectors"].values():
        keep = [0, *range(16, 25)]
        arr["conditions"], arr["scores"], arr["labels"] = (
            arr["conditions"][keep],
            arr["scores"][keep],
            arr["labels"][keep],
        )
    with pytest.raises(ValueError, match="H11"):
        h11(run, n_boot=N_BOOT)


# ----------------------------------------------------------------------------------------------------- H12


def keys_of(n):
    return [f"{light}:{n}" for light in LIGHTS]


def test_h12_supported_rejected_undecided():
    run = flat_run()
    result = h12(run, n_boot=N_BOOT)
    assert (result["n"], result["limit"], result["n_normal"]) == (30, 0.10, 9 * 120)
    assert result["fpr"] == 0 and result["fpr_ci"] == [0, 0] and result["verdict"] == "지지"
    assert result["illuminations"] == LIGHTS

    run = flat_run()
    flag(run, keys_of(30), range(20), key="recal_scores")
    result = h12(run, n_boot=N_BOOT)
    assert result["fpr"] == 1 and result["false_positives"] == 1080 and result["verdict"] == "기각"

    run = flat_run()
    flag(run, keys_of(30), [0, 1], key="recal_scores")  # 10% of the normal specimens
    result = h12(run, n_boot=N_BOOT)
    assert result["fpr"] == pytest.approx(0.10)
    assert result["fpr_ci"][0] < 0.10 < result["fpr_ci"][1] and result["verdict"] == "판정 불가"

    run = flat_run()
    flag(run, keys_of(30), range(8), key="recal_scores")  # 40%: clearly above the limit
    result = h12(run, n_boot=N_BOOT)
    assert result["fpr_ci"][0] > 0.10 and result["verdict"] == "기각"

    run = flat_run()
    flag(run, keys_of(30)[:2], [0], key="recal_scores")  # 12 of 1,080 images
    result = h12(run, n_boot=N_BOOT)
    assert result["false_positives"] == 12 and result["fpr_ci"][1] < 0.10 and result["verdict"] == "지지"


def test_h12_boundary_and_size():
    run = flat_run()
    # Every normal specimen is flagged in exactly one of its three views: the pooled rate is 1/3 in every
    # resample, so the interval is the single point 1/3.
    flag(run, keys_of(30), range(20), views=("120",), key="recal_scores")
    assert h12(run, limit=1 / 3, n_boot=N_BOOT)["verdict"] == "지지"  # upper end <= limit
    assert h12(run, limit=0.33, n_boot=N_BOOT)["verdict"] == "기각"
    assert h12(run, limit=0.34, n_boot=N_BOOT)["verdict"] == "지지"
    # n selects the recalibration size; the un-recalibrated scores play no part.
    assert h12(run, n=8, n_boot=N_BOOT)["fpr"] == 0
    flag(run, [f"R:{light}" for light in LIGHTS], range(20))
    assert h12(run, n=8, n_boot=N_BOOT)["verdict"] == "지지"
    with pytest.raises(ValueError):
        h12(run, n=16, n_boot=N_BOOT)


def test_h12_is_undecided_when_the_lower_end_equals_the_limit():
    run = flat_run()
    # Motor: every normal specimen is flagged in one view, 20 of 120 images in every resample. Bird: one
    # specimen is flagged in all three views, so resamples without it stay at 20/120 = 1/6 (more than a
    # third of them: the lower end) and the others lie above.
    flag(run, keys_of(30), range(20), views=("120",), key="recal_scores", categories=("Motor",))
    flag(run, keys_of(30), [0], key="recal_scores", categories=("Bird",))
    result = h12(run, limit=1 / 6, n_boot=N_BOOT)
    assert result["fpr_ci"][0] == pytest.approx(1 / 6) and result["fpr_ci"][1] > 1 / 6 + 0.01
    assert result["verdict"] == "판정 불가"  # rejected only if the lower end is above the limit
    assert h12(run, limit=1 / 6 - 1e-6, n_boot=N_BOOT)["verdict"] == "기각"


def test_recalibrated_images_are_judged_by_the_recalibrated_thresholds():
    run = flat_run()
    for arr in run["inspectors"].values():
        arr["recal_scores"][:, :20] = 0.7  # every normal image: above the base threshold 0.5 ...
        arr["recal_thresholds"][:] = 0.9  # ... and below the threshold set by the recalibration
    result = h12(run, n_boot=N_BOOT)
    assert result["fpr"] == 0 and result["false_positives"] == 0 and result["verdict"] == "지지"
    table = recal_table(run, n_boot=N_BOOT)
    assert [row["fpr"] for row in table] == [0, 0, 0] and [row["tpr"] for row in table] == [1, 1, 1]

    # Each key has its own threshold, in each inspector.
    for arr in run["inspectors"].values():
        arr["recal_thresholds"][[RECAL_KEYS.index(key) for key in keys_of(8)]] = 0.6
    run["inspectors"]["Bird_240"]["recal_thresholds"][RECAL_KEYS.index("04:30")] = 0.65
    run["inspectors"]["Motor_000"]["recal_thresholds"][RECAL_KEYS.index("04:30")] = 1.5  # misses all defects
    assert [t for _, _, t in recal_cell(run, "04:30")] == [1.5, 0.9, 0.9, 0.9, 0.9, 0.65]
    assert [t for _, _, t in recal_cell(run, "04:8")] == [0.6] * 6
    table = recal_table(run, n_boot=N_BOOT)
    assert [row["false_positives"] for row in table] == [0, 1080, 20]
    assert [row["detected"] for row in table] == [2700, 2700, 2700 - 50]
    result = h12(run, n_boot=N_BOOT)
    assert result["false_positives"] == 20 and result["fpr"] == pytest.approx(20 / 1080)
    assert h12(run, n=8, n_boot=N_BOOT)["fpr"] == 1


def test_h12_matches_the_direct_loop():
    run = random_run(seed=5)
    result = h12(run, n_boot=100)
    fpr, _ = direct_pooled(run, [recal_cell(run, key) for key in keys_of(30)], 100)
    assert result["fpr_ci"] == pytest.approx(percentile_ci(fpr))
    assert result["n_normal"] == 9 * (3 * 5 + 3 * 4)
    assert 0 < result["fpr"] < 1 and result["fpr_ci"][0] < result["fpr"] < result["fpr_ci"][1]


# --------------------------------------------------------------------------------------------- recal table


def test_recal_table_pools_the_real_illuminations():
    run = flat_run()
    flag(run, [f"R:{light}" for light in LIGHTS], range(16))  # 80% flagged without recalibration
    flag(run, keys_of(8), range(4), key="recal_scores")  # 20% after n = 8
    flag(run, keys_of(30)[:3], [0], key="recal_scores")
    for arr in run["inspectors"].values():  # recalibration with 30 also loses ten defects everywhere
        for key in keys_of(30):
            arr["recal_scores"][RECAL_KEYS.index(key), 20:30] = 0.0
    table = recal_table(run, n_boot=N_BOOT)
    assert [row["n"] for row in table] == [0, 8, 30]
    assert all(row["illuminations"] == LIGHTS and row["n_normal"] == 1080 for row in table)
    assert [row["fpr"] for row in table] == pytest.approx([0.8, 0.2, 3 * 6 / 1080])
    assert [row["tpr"] for row in table] == pytest.approx([1.0, 1.0, 0.8])
    assert "d_fpr" not in table[0]
    assert table[1]["d_fpr"] == pytest.approx(-0.6) and table[1]["d_fpr_ci"][1] < 0
    assert table[2]["d_tpr"] == pytest.approx(-0.2) and table[2]["d_tpr_ci"][1] < 0
    # n = 0 is the pooling of the R rows of the condition table.
    rows = [r for r in condition_table(run, n_boot=N_BOOT) if r["condition"].startswith("R:")]
    assert table[0]["false_positives"] == sum(r["false_positives"] for r in rows)
    assert table[0]["n_defect"] == sum(r["n_defect"] for r in rows) == 9 * 300


def test_recal_table_matches_the_direct_loop():
    run = random_run(seed=8)
    table = recal_table(run, n_boot=100)
    base_fpr, base_tpr = direct_pooled(run, [condition_cell(run, f"R:{light}") for light in LIGHTS], 100)
    assert table[0]["fpr_ci"] == pytest.approx(percentile_ci(base_fpr))
    assert table[0]["tpr_ci"] == pytest.approx(percentile_ci(base_tpr))
    for row, n in ((table[1], 8), (table[2], 30)):
        fpr, tpr = direct_pooled(run, [recal_cell(run, key) for key in keys_of(n)], 100)
        assert row["fpr_ci"] == pytest.approx(percentile_ci(fpr))
        assert row["tpr_ci"] == pytest.approx(percentile_ci(tpr))
        # Differences to the un-recalibrated rows use the same draws.
        assert row["d_fpr_ci"] == pytest.approx(percentile_ci(fpr - base_fpr))
        assert row["d_tpr_ci"] == pytest.approx(percentile_ci(tpr - base_tpr))


# ------------------------------------------------------------------------------------------ files and CLI


def test_load_reads_a_run_folder(tmp_path):
    run = random_run()
    write_run(run, tmp_path / "m2ad-p0")
    loaded = analyze_m2ad.load(tmp_path / "m2ad-p0")
    assert list(loaded["inspectors"]) == [f"{c}_{v}" for c in ("Motor", "Bird") for v in VIEWS]
    assert loaded["meta"]["commit"] == "abc1234"
    for name, arr in run["inspectors"].items():
        assert loaded["inspectors"][name]["category"] == arr["category"]
        for key in ("scores", "labels", "recal_scores", "recal_thresholds", "specimens"):
            assert np.array_equal(loaded["inspectors"][name][key], arr[key])
    assert condition_table(loaded, n_boot=50) == condition_table(run, n_boot=50)


def test_load_rejects_inconsistent_inspectors(tmp_path):
    run = random_run()
    run["inspectors"]["Motor_120"]["specimens"] = run["inspectors"]["Motor_120"]["specimens"][::-1].copy()
    write_run(run, tmp_path / "a")
    with pytest.raises(ValueError, match="specimens differ"):
        analyze_m2ad.load(tmp_path / "a")

    run = random_run()
    run["inspectors"]["Bird_000"]["labels"][3, 0] = 1  # a normal specimen with a defect label
    write_run(run, tmp_path / "b")
    with pytest.raises(ValueError, match="object_anomaly"):
        analyze_m2ad.load(tmp_path / "b")

    run = random_run()
    arr = run["inspectors"]["Bird_240"]
    arr["conditions"] = arr["conditions"][::-1].copy()
    write_run(run, tmp_path / "c")
    with pytest.raises(ValueError, match="conditions differ"):
        analyze_m2ad.load(tmp_path / "c")


def _cli_runs(tmp_path, monkeypatch, edit=None):
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(analyze_m2ad, "N_BOOT", 100)
    p0 = flat_run()
    flag(p0, [f"R:{light}" for light in LIGHTS], range(10))
    flag(p0, ["P:blur-3"], [0])
    if edit is not None:
        edit(p0)
    ds = flat_run()
    ds["meta"]["method"] = "d-s"
    flag(ds, ["P:jpeg-3"], range(20))
    flag(ds, keys_of(30), range(20), key="recal_scores")
    write_run(p0, tmp_path / "outputs" / "m2ad-p0")
    write_run(ds, tmp_path / "outputs" / "m2ad-d-s")


def test_cli_writes_the_report_and_judges_p0_only(tmp_path, monkeypatch, capsys):
    _cli_runs(tmp_path, monkeypatch)
    analyze_m2ad.main([])
    out = capsys.readouterr().out
    assert "## p0" in out and "## d-s" in out and "| R:10 |" in out and "| P:brightness-1 |" in out
    assert "hypotheses (p0): H11: 지지; H12: 지지" in out
    report = json.loads((tmp_path / "reports" / "stage3" / "m2ad.json").read_text(encoding="utf-8"))
    assert report["hypotheses"] == {"H11": "지지", "H12": "지지"}
    assert report["n_boot"] == 100 and report["judged_method"] == "p0"
    assert list(report["methods"]) == ["p0", "d-s"]
    p0, ds = report["methods"]["p0"], report["methods"]["d-s"]
    assert p0["h11"]["verdict"] == "지지" and p0["h12"]["verdict"] == "지지"
    # The numbers of the other method are there, without a verdict.
    assert "verdict" not in ds["h11"] and "verdict" not in ds["h12"]
    assert ds["h11"]["statistic"] == pytest.approx(1.0) and ds["h12"]["fpr"] == 1
    assert [row["condition"] for row in p0["conditions"]] == CONDITIONS
    assert [row["n"] for row in p0["recal"]] == [0, 8, 30]
    assert p0["per_inspector"]["Motor_000"]["conditions"]["R:02"]["false_positives"] == 10
    assert p0["per_inspector"]["Motor_000"]["threshold"] == 0.5


def test_cli_report_is_strict_json_when_a_value_is_undefined(tmp_path, monkeypatch, capsys):
    k = CONDITIONS.index("R:06")

    def hide_the_defects(p0):
        # No defect image is left under R:06: that condition has no detection rate and no AUROC.
        for arr in p0["inspectors"].values():
            arr["labels"][k, 20:] = -1

    _cli_runs(tmp_path, monkeypatch, edit=hide_the_defects)
    analyze_m2ad.main([])
    out = capsys.readouterr().out
    assert "| R:06 | 120 | 0 | 50.0 [" in out and "nan [nan, nan]" in out
    assert "hypotheses (p0): H11: 지지; H12: 지지" in out

    def refuse(token):
        raise AssertionError(f"{token} is not JSON")

    text = (tmp_path / "reports" / "stage3" / "m2ad.json").read_text(encoding="utf-8")
    report = json.loads(text, parse_constant=refuse)  # NaN or Infinity in the file would end up here
    row = report["methods"]["p0"]["conditions"][k]
    assert (row["condition"], row["n_defect"], row["fpr"]) == ("R:06", 0, 0.5)
    for key in ("tpr", "auroc", "d_tpr", "d_auroc"):
        assert row[key] is None and row[f"{key}_ci"] == [None, None]
    assert report["methods"]["p0"]["per_inspector"]["Bird_120"]["conditions"]["R:06"]["auroc"] is None
    # Everything that is defined is still a number.
    assert row["fpr_ci"][0] < 0.5 < row["fpr_ci"][1] and row["d_fpr"] == 0.5
    other = report["methods"]["p0"]["conditions"][k + 1]
    assert other["tpr"] == 1 and other["auroc"] == pytest.approx(0.75)
    assert other["auroc_ci"][0] < 0.75 < other["auroc_ci"][1]


def test_cli_no_write_and_single_method(tmp_path, monkeypatch, capsys):
    _cli_runs(tmp_path, monkeypatch)
    analyze_m2ad.main(["--methods", "d-s", "--no-write"])
    out = capsys.readouterr().out
    assert "## d-s" in out and "## p0" not in out and "hypotheses" not in out
    assert not (tmp_path / "reports").exists()
    with pytest.raises(FileNotFoundError):
        analyze_m2ad.main(["--methods", "d-b"])


def test_cli_reads_a_check_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(analyze_m2ad, "N_BOOT", 100)
    run = flat_run(categories=("Motor",))
    run["meta"]["inspectors"] = [{"category": "Motor", "view": "000"}]
    run["meta"]["check"] = True
    arr = run["inspectors"]["Motor_000"]
    keep = [0, CONDITIONS.index("R:02")]
    arr["conditions"], arr["scores"], arr["labels"] = (
        arr["conditions"][keep],
        arr["scores"][keep],
        arr["labels"][keep],
    )
    arr["recal_keys"] = np.array([], dtype=str)
    arr["recal_scores"], arr["recal_labels"] = arr["recal_scores"][:0], arr["recal_labels"][:0]
    arr["recal_thresholds"] = arr["recal_thresholds"][:0]
    run["inspectors"] = {"Motor_000": arr}
    arr["scores"][1, :3] = 1.0
    write_run(run, tmp_path / "outputs" / "m2ad-p0-check")
    analyze_m2ad.main(["--methods", "p0", "--check"])
    out = capsys.readouterr().out
    assert "| S | 20 | 50 | 0.0 [0.0, 0.0] |" in out and "| R:02 | 20 | 50 | 15.0" in out
    assert "H11" not in out and "H12" not in out
    assert not (tmp_path / "reports").exists()

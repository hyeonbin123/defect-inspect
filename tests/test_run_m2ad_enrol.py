import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defect_inspect import analyze_m2ad_enrol, ledger, paths, run_m2ad, run_m2ad_enrol  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.m2ad import (  # noqa: E402
    ILLUMINATIONS,
    VIEWS,
    M2adRow,
    illumination_groups,
    recal_specimens,
    specimen_folds,
)
from defect_inspect.splits import SealedTestError, path_key  # noqa: E402

SIZE = 16
N_TRAIN = 30
NORMAL = [f"{2 * i + 1:03d}" for i in range(4)]
ANOMALOUS = ["hole_1_000", "hole_1_001", "hole_1_002", "scratch_1_003", "scratch_2_004"]
NEVER_VISIBLE = "hole_1_001"


class GridMean(torch.nn.Module):
    """Tiny stand-in for a backbone: the mean colour of each cell of a 2x2 grid."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 2).permute(0, 2, 3, 1)


def make_rows(categories=("Motor",)) -> list[M2adRow]:
    rows = []
    for category in categories:
        for view in VIEWS:
            for light in ILLUMINATIONS:
                for i in range(N_TRAIN):
                    name = f"{2 * i:03d}"
                    path = f"{category}/Good/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "train", name, view, light, path, 0, 0))
                for name in NORMAL:
                    path = f"{category}/Good/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "test", name, view, light, path, 0, 0))
                for name in ANOMALOUS:
                    label = -1 if name == NEVER_VISIBLE else 1
                    path = f"{category}/NG/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "test", name, view, light, path, 1, label))
    return sorted(rows, key=lambda r: hashlib.sha256(r.img_path.encode()).hexdigest())


class FakeCache:
    """Specimen noise, a uniform darkening per illumination, a cell of a specimen's own colour for defects."""

    size = SIZE

    def __init__(self):
        self.splits: list[str] = []

    def _image(self, row: M2adRow) -> np.ndarray:
        key = f"{row.category}/{row.specimen}/{row.view}"
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
        img = img - np.uint8(8 * ILLUMINATIONS.index(row.illumination))
        if row.label == 1:
            img[:8, :8] = rng.integers(0, 256, 3, dtype=np.uint8) // 2 * np.array([2, 0, 1], dtype=np.uint8)
        return img

    def images(self, rows):
        self.splits += sorted({r.split for r in rows})
        return np.stack([self._image(r) for r in rows])


CFG = PatchCoreConfig(
    name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=1.0, sigma=1.0, batch_size=16
)


def tools(centre=False):
    extractor = run_m2ad_enrol.centred(GridMean()) if centre else GridMean()
    return run_m2ad_enrol.Tools(CFG, extractor, "cpu")


@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    monkeypatch.setattr(run_m2ad_enrol, "git_commit", lambda root=None: "abc1234")
    return path


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def pick(rows, view, split, light, category="Motor"):
    wanted = (category, view, split, light)
    return sorted(
        (r for r in rows if (r.category, r.view, r.split, r.illumination) == wanted), key=lambda r: r.specimen
    )


# ---------------------------------------------------------------- arms


def test_arms_and_enrolment():
    a, b = illumination_groups()
    assert run_m2ad_enrol.arms("e0") == run_m2ad_enrol.arms("e1") == [("ref", ("01",))]
    assert run_m2ad_enrol.arms("e2") == run_m2ad_enrol.arms("e3") == [("A", ("01", *a)), ("B", ("01", *b))]
    m = run_m2ad_enrol.enrolled_matrix("e2")
    assert m.shape == (2, 10) and m[:, 0].all()
    # Every non-reference illumination is enrolled in exactly one arm, so it is unseen in the other one.
    assert (m[:, 1:].sum(axis=0) == 1).all()
    assert run_m2ad_enrol.enrolled_matrix("e0").tolist() == [[True] + [False] * 9]
    with pytest.raises(ValueError):
        run_m2ad_enrol.arms("e4")
    keys = run_m2ad_enrol.loop_keys()
    assert len(keys) == 36 and keys[:4] == ["02:trim:0", "02:trim:1", "02:trim:2", "02:none:2"]
    assert run_m2ad_enrol.CANDIDATES == ("e0", "e1", "e2", "e3")


# ---------------------------------------------------------------- validation


@pytest.fixture(scope="module")
def val_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("val")
    rows = make_rows()
    cache = FakeCache()
    infos = {}
    for candidate in ("e0", "e1", "e2"):
        t = tools(run_m2ad_enrol.CENTRE[candidate])
        infos[candidate] = run_m2ad_enrol.validate_inspector(t, candidate, "Motor", "000", rows, cache, out)
    return out, rows, cache, infos


def test_validation_reads_train_images_only(val_runs):
    _, _, cache, infos = val_runs
    assert set(cache.splits) == {"train"}
    assert infos["e2"]["arms"] == {
        "A": ["01", *illumination_groups()[0]],
        "B": ["01", *illumination_groups()[1]],
    }


def test_validation_is_nested_over_specimen_folds(val_runs):
    out, rows, _, _ = val_runs
    z = load(out / "Motor_000_e2.npz")
    names = [f"{2 * i:03d}" for i in range(N_TRAIN)]
    assert z["specimens"].tolist() == names
    fold_of = specimen_folds(names)
    assert z["folds"].tolist() == [fold_of[n] for n in names]
    assert z["arms"].tolist() == ["A", "B"] and z["lights"].tolist() == list(ILLUMINATIONS)
    assert z["scores"].shape == (2, 10, N_TRAIN) and z["thresholds"].shape == (2, 5)
    folds = z["folds"]
    # Arm B, fold 2, rebuilt by hand: the enrolled images of the other specimens, a cross-fit over their
    # four folds, and the held-out specimens scored by the bank of all of them.
    _, enrolled = run_m2ad_enrol.arms("e2")[1]
    t = tools()
    keep = folds != 2
    imgs = {light: FakeCache().images(pick(rows, "000", "train", light)) for light in ILLUMINATIONS}
    feats = torch.cat([t.features(imgs[light][keep]) for light in enrolled])
    bank, oof, thr, _ = run_m2ad.crossfit(
        t.build,
        t.score,
        feats,
        np.concatenate([imgs[light][keep] for light in enrolled]),
        np.concatenate([folds[keep]] * len(enrolled)),
    )
    assert z["thresholds"][1, 2] == thr.value == conformal_threshold(oof, 0.05).value
    assert z["cal_n"][1, 2] == len(oof) == len(enrolled) * keep.sum()
    held = np.flatnonzero(folds == 2)
    for li, light in enumerate(ILLUMINATIONS):
        np.testing.assert_allclose(z["scores"][1, li, held], t.score(bank, imgs[light][held]), rtol=1e-6)


def test_enrolment_and_centring_remove_the_false_alarms_of_unseen_light(val_runs):
    out, _, _, _ = val_runs

    def unseen_fpr(candidate):
        z = load(out / f"Motor_000_{candidate}.npz")
        flagged = z["scores"] > z["thresholds"][:, None, z["folds"]]
        unseen = ~z["enrolled"][:, :, None].repeat(N_TRAIN, axis=2)
        return flagged[unseen].mean()

    # A darker image is far from a bank of I01 patches; centring removes a uniform darkening entirely.
    assert unseen_fpr("e0") == 1.0
    assert unseen_fpr("e1") <= 0.2
    z1 = load(out / "Motor_000_e1.npz")
    np.testing.assert_allclose(z1["scores"][0, 5], z1["scores"][0, 0], rtol=1e-4)
    # A bank that holds the other specimens under an enrolled light covers that light for held-out ones.
    z0, z2 = load(out / "Motor_000_e0.npz"), load(out / "Motor_000_e2.npz")
    for a, light in ((0, "07"), (1, "05")):
        li = ILLUMINATIONS.index(light)
        assert z2["enrolled"][a, li]
        assert (z0["scores"][0, li] > z0["thresholds"][0, z0["folds"]]).all()
        assert (z2["scores"][a, li] > z2["thresholds"][a, z2["folds"]]).mean() <= 0.2


# ---------------------------------------------------------------- test


def test_the_test_run_needs_permission(tmp_path):
    with pytest.raises(SealedTestError):
        run_m2ad_enrol.sealed_inspector(tools(), "e0", "Motor", "000", make_rows(), FakeCache(), tmp_path)
    with pytest.raises(SealedTestError):
        run_m2ad_enrol.loop_inspector(tools(), "Motor", "000", make_rows(), FakeCache(), tmp_path)


def test_e0_reproduces_the_stage_3b_scores(tmp_path):
    rows = make_rows()
    info = run_m2ad_enrol.sealed_inspector(
        tools(), "e0", "Motor", "000", rows, FakeCache(), tmp_path, allow_test=True
    )
    z = load(tmp_path / "Motor_000_e0.npz")
    ref = tmp_path / "ref"
    ref.mkdir()
    run_m2ad.run_inspector(
        CFG, "Motor", "000", rows, FakeCache(), GridMean(), "cpu", ref, allow_test=True, recal_sizes=()
    )
    old = load(ref / "Motor_000.npz")
    assert z["specimens"].tolist() == old["specimens"].tolist()
    assert z["thresholds"].tolist() == [float(old["threshold"])]
    assert np.array_equal(z["cal_score_ref"], old["cal_score"])
    conditions = old["conditions"].tolist()
    for li, light in enumerate(ILLUMINATIONS):
        k = conditions.index("S" if light == "01" else f"R:{light}")
        assert np.array_equal(z["scores"][0, li], old["scores"][k]), light
        assert np.array_equal(z["labels"][li], old["labels"][k]), light
    assert info["arms"]["ref"]["threshold"]["n"] == N_TRAIN


def test_multi_enrolment_test_run(tmp_path):
    rows = make_rows()
    info = run_m2ad_enrol.sealed_inspector(
        tools(), "e2", "Motor", "120", rows, FakeCache(), tmp_path, allow_test=True
    )
    z = load(tmp_path / "Motor_120_e2.npz")
    assert z["scores"].shape == (2, 10, 9) and z["thresholds"].shape == (2,)
    assert len(z["cal_score_A"]) == N_TRAIN * 5 and len(z["cal_score_B"]) == N_TRAIN * 6
    assert info["arms"]["B"]["enrolled"] == ["01", *illumination_groups()[1]]
    assert z["object_anomaly"].tolist() == [0] * 4 + [1] * 5


def test_closed_loop_drops_the_mixed_in_defects(tmp_path):
    rows = make_rows()
    cache = FakeCache()
    info = run_m2ad_enrol.loop_inspector(tools(), "Motor", "000", rows, cache, tmp_path, allow_test=True)
    z = load(tmp_path / "Motor_000_loop.npz")
    keys = run_m2ad_enrol.loop_keys()
    assert z["loop_keys"].tolist() == keys and z["scores"].shape == (36, 9)
    names = z["specimens"].tolist()
    visible = sorted((n for n in ANOMALOUS if n != NEVER_VISIBLE), key=path_key)
    for key in keys:
        light, filt, k = key.split(":")
        k = int(k)
        entry = info["loop"][key]
        assert entry["batch"] == 20 and entry["defects_in_batch"] == k
        labels = z["labels"][keys.index(key)]
        # The enrolled defect specimens (first k visible ones in SHA-256 order) leave the detection rate.
        for i, name in enumerate(names):
            if name in visible[:k]:
                assert labels[i] == -1
        assert (labels == -1).sum() == k + 1  # plus the defect that is never visible
        if filt == "trim":
            # The bright defect cells score highest within the batch, so the filter drops them first.
            assert entry["dropped_defects"] == k and entry["dropped_normals"] == 4 - k
            assert entry["enrolled"] == N_TRAIN + 16
        else:
            assert entry["dropped_defects"] == 0 and entry["enrolled"] == N_TRAIN + 20
    # The normal part of the batch is the first 20 - k train specimens in SHA-256 order.
    assert (
        recal_specimens([f"{2 * i:03d}" for i in range(N_TRAIN)], 18)
        == recal_specimens([f"{2 * i:03d}" for i in range(N_TRAIN)], 20)[:18]
    )
    # Re-enrolment brings the normals under the new light back under the threshold.
    j = keys.index("05:trim:2")
    normal = z["labels"][j] == 0
    assert (z["scores"][j][normal] <= z["thresholds"][j]).mean() >= 0.75


# ---------------------------------------------------------------- CLI


@pytest.fixture
def project(tmp_path, monkeypatch):
    rows = make_rows()
    cache = FakeCache()
    monkeypatch.setattr(run_m2ad_enrol, "read_meta", lambda path: rows)
    monkeypatch.setattr(run_m2ad_enrol, "M2adCache", lambda root, size: cache)
    monkeypatch.setattr(run_m2ad_enrol, "_make_extractor", lambda cfg, device: GridMean())
    monkeypatch.setattr(run_m2ad_enrol, "get_config", lambda name: CFG)
    monkeypatch.setattr(run_m2ad_enrol, "INSPECTORS", ("Motor_000", "Motor_120"))
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    return cache


def test_cli_validation(project, tmp_path, temporary_ledger):
    run_m2ad_enrol.main(["val", "--method", "p0", "--device", "cpu", "--candidates", "e0", "e3"])
    out = tmp_path / "outputs" / "m2ad-enrol-val-p0"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["command"] == "val" and run["candidates"] == ["e0", "e3"]
    assert run["inspectors"] == ["Motor_000", "Motor_120"] and run["groups"]["A"] == ["03", "07", "08", "09"]
    assert sorted(p.name for p in out.glob("*.npz")) == [
        "Motor_000_e0.npz",
        "Motor_000_e3.npz",
        "Motor_120_e0.npz",
        "Motor_120_e3.npz",
    ]
    assert set(project.splits) == {"train"} and not temporary_ledger.exists()
    with pytest.raises(SystemExit):
        run_m2ad_enrol.main(["val", "--method", "p0", "--device", "cpu"])  # a finished run stays

    # The analysis reads the run as written and picks the candidate with fewer unseen false alarms.
    report = analyze_m2ad_enrol.build_val_report({"p0": analyze_m2ad_enrol.load_val(out)}, n_boot=20)
    rows = report["methods"]["p0"]["candidates"]
    assert rows["e0"]["unseen"]["n_normal"] == rows["e3"]["unseen"]["n_normal"] == 9 * N_TRAIN * 2
    assert rows["e0"]["unseen"]["fpr"] == 1.0 and rows["e3"]["unseen"]["fpr"] < 0.5
    assert report["methods"]["p0"]["pick"] == "e3"


def test_cli_test_records_the_read_and_runs_e0_and_the_pick(project, tmp_path, temporary_ledger):
    val = tmp_path / "val.json"
    base = ["test", "--method", "p0", "--device", "cpu", "--pick-from", str(val)]
    with pytest.raises(SystemExit):
        run_m2ad_enrol.main(base)  # no --allow-test
    with pytest.raises(SystemExit):
        run_m2ad_enrol.main(base + ["--allow-test", "--stage", "7-m2ad"])  # no val report
    val.write_text(json.dumps({"methods": {"p0": {"pick": "e2"}}}), encoding="utf-8")
    assert project.splits == [] and not temporary_ledger.exists()
    run_m2ad_enrol.main(base + ["--allow-test", "--stage", "7-m2ad", "--loop"])
    entries = [json.loads(line) for line in temporary_ledger.read_text(encoding="utf-8").splitlines()]
    assert [(e["stage"], e["config"], e["commit"]) for e in entries] == [
        ("7-m2ad", "m2ad-enrol-p0", "abc1234")
    ]
    out = tmp_path / "outputs" / "m2ad-enrol-test-p0"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["candidates"] == ["e0", "e2"] and run["loop"]["n"] == 20
    assert len(run["loop_runs"]) == 2
    assert (out / "Motor_120_loop.npz").exists() and (out / "Motor_000_e2.npz").exists()

    report = analyze_m2ad_enrol.build_test_report({"p0": analyze_m2ad_enrol.load_test(out)}, n_boot=20)
    entry = report["methods"]["p0"]
    assert entry["verdicts"]["pick"] == "e2" and entry["candidates"]["e0"]["unseen"]["fpr"] == 1.0
    assert [(row["filter"], row["k"]) for row in entry["loop"]] == [tuple(a) for a in run["loop"]["arms"]]
    assert all(row["dropped_defects"] == row["defects_in_batches"] for row in entry["loop"][:3])


def test_read_pick(tmp_path):
    path = tmp_path / "val.json"
    path.write_text(json.dumps({"methods": {"p0": {"pick": "e1"}}}), encoding="utf-8")
    assert run_m2ad_enrol.read_pick(path, "p0") == "e1"
    with pytest.raises(ValueError):
        run_m2ad_enrol.read_pick(path, "d-s")
    path.write_text(json.dumps({"methods": {"p0": {"pick": "e9"}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        run_m2ad_enrol.read_pick(path, "p0")

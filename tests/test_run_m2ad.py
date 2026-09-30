import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defect_inspect import ledger, paths, run_m2ad  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.conditions import apply_condition, condition_names  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.m2ad import ILLUMINATIONS, VIEWS, M2adRow, recal_specimens, specimen_folds  # noqa: E402
from defect_inspect.splits import SealedTestError  # noqa: E402

SIZE = 16
PATCHES = 4  # GridMean gives a 2x2 grid
N_TRAIN = 30
NORMAL = [f"{2 * i + 1:03d}" for i in range(4)]
ANOMALOUS = ["hole_1_000", "hole_1_001", "hole_1_002", "scratch_1_003", "scratch_2_004"]
NEVER_VISIBLE = "hole_1_001"  # excluded everywhere
HIDDEN_UNDER_05 = "hole_1_002"  # excluded under illumination 05 only
SAME_AS_REFERENCE = "03"  # the fake images under this illumination equal those under 01


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
                    label = 1
                    if name == NEVER_VISIBLE or (name == HIDDEN_UNDER_05 and light == "05"):
                        label = -1
                    path = f"{category}/NG/{name}/A{view}_I{light}.png"
                    rows.append(M2adRow(category, "test", name, view, light, path, 1, label))
    # The run must not depend on the order in which the rows arrive.
    return sorted(rows, key=lambda r: hashlib.sha256(r.img_path.encode()).hexdigest())


class FakeCache:
    """Images made from the row: specimen noise, an offset per illumination, a bright cell for defects."""

    size = SIZE

    def __init__(self):
        self.reads: list[tuple[str, str, int]] = []
        self.on_test_read = None

    def _image(self, row: M2adRow) -> np.ndarray:
        key = f"{row.category}/{row.specimen}/{row.view}"
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
        light = ILLUMINATIONS.index(row.illumination)
        if row.illumination != SAME_AS_REFERENCE:
            img = img - np.uint8(8 * light)  # every new illumination is darker by eight more levels
        if row.label == 1:
            img[:8, :8] = 255  # a bright cell that no normal image has
        return img

    def images(self, rows):
        splits = {r.split for r in rows}
        if "test" in splits and self.on_test_read is not None:
            self.on_test_read()
        for split in sorted(splits):
            lights = sorted({r.illumination for r in rows if r.split == split})
            self.reads.append((split, ",".join(lights), len(rows)))
        return np.stack([self._image(r) for r in rows])


CFG = PatchCoreConfig(
    name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=1.0, sigma=1.0, batch_size=16
)


@pytest.fixture(autouse=True)
def temporary_ledger(tmp_path, monkeypatch):
    path = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "TEST_LEDGER", path)
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "abc1234")
    monkeypatch.setattr(run_m2ad, "git_commit", lambda root=None: "abc1234")
    return path


def _run(tmp_path, view="000", cache=None, cfg=CFG, **kwargs):
    cache = cache or FakeCache()
    info = run_m2ad.run_inspector(
        cfg, "Motor", view, make_rows(), cache, GridMean(), "cpu", tmp_path, **kwargs
    )
    with np.load(tmp_path / f"Motor_{view}.npz") as z:
        return info, {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("m2ad-full")
    cache = FakeCache()
    info, arrays = _run(tmp_path, cache=cache, allow_test=True)
    return info, arrays, cache


def ledger_entries(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def pick(rows, view, split, light, category="Motor") -> list[M2adRow]:
    """The rows of one (category, view, split, illumination) by specimen name, stated without the run."""
    wanted = (category, view, split, light)
    return sorted(
        (r for r in rows if (r.category, r.view, r.split, r.illumination) == wanted), key=lambda r: r.specimen
    )


def direct_scores(bank_rows, shown_rows, cfg=CFG) -> np.ndarray:
    """Image scores of `shown_rows` against a bank of `bank_rows`, straight from the patchcore functions."""
    from defect_inspect.patchcore import build_bank, collect_features, score_images

    cache = FakeCache()
    feats, _ = collect_features(GridMean(), cache.images(bank_rows), batch_size=cfg.batch_size, device="cpu")
    bank = build_bank(feats, cfg.coreset_ratio, seed=cfg.seed, device="cpu")
    return score_images(
        GridMean(),
        bank,
        cache.images(shown_rows),
        batch_size=cfg.batch_size,
        device="cpu",
        reweight_k=cfg.reweight_k,
        sigma=cfg.sigma,
    ).image_scores


def test_condition_and_recalibration_order():
    conditions = run_m2ad.condition_list()
    assert len(conditions) == 25 and conditions[0] == "S"
    assert conditions[1:16] == [f"P:{name}" for name in condition_names(include_clean=False)]
    assert conditions[16:] == [f"R:{light}" for light in ILLUMINATIONS[1:]]
    assert run_m2ad.condition_list(check=True) == ["S", "R:02"]
    keys = run_m2ad.recal_key_list()
    assert len(keys) == 18 and keys[:4] == ["02:8", "02:30", "03:8", "03:30"] and keys[-1] == "10:30"


def test_full_run_shapes_and_order(full):
    info, z, _ = full
    n_test = len(NORMAL) + len(ANOMALOUS)
    assert z["specimens"].tolist() == sorted(NORMAL + ANOMALOUS)
    assert z["object_anomaly"].tolist() == [0] * len(NORMAL) + [1] * len(ANOMALOUS)
    assert z["object_anomaly"].dtype == np.int8
    assert z["conditions"].tolist() == run_m2ad.condition_list()
    assert z["scores"].shape == (25, n_test) and z["scores"].dtype == np.float32
    assert z["labels"].shape == (25, n_test) and z["labels"].dtype == np.int8
    assert z["threshold"].shape == () and z["threshold"].dtype == np.float64
    assert z["cal_score"].shape == (N_TRAIN,) and z["cal_score"].dtype == np.float32
    assert z["recal_keys"].tolist() == run_m2ad.recal_key_list()
    assert z["recal_scores"].shape == (18, n_test) and z["recal_scores"].dtype == np.float32
    assert z["recal_thresholds"].shape == (18,) and z["recal_thresholds"].dtype == np.float64
    assert z["recal_labels"].shape == (18, n_test) and z["recal_labels"].dtype == np.int8
    assert np.isfinite(z["scores"]).all() and np.isfinite(z["recal_scores"]).all()
    assert (info["category"], info["view"]) == ("Motor", "000")
    assert (info["train_specimens"], info["test_specimens"], info["test_anomalous_specimens"]) == (30, 9, 5)
    assert info["grid"] == [2, 2] and info["dim"] == 3
    assert set(info["recal"]) == set(run_m2ad.recal_key_list())


def test_threshold_is_cross_fitted_over_specimens(full):
    info, z, _ = full
    thr = conformal_threshold(z["cal_score"], 0.05)
    assert float(z["threshold"]) == thr.value
    assert info["threshold"] == {"value": thr.value, "rank": 30, "n": 30, "guaranteed": True}
    train = [f"{2 * i:03d}" for i in range(N_TRAIN)]
    assert info["folds"] == specimen_folds(train)
    # With the whole bank kept, an image scored by a bank that holds it would score exactly 0.
    assert z["cal_score"].min() > 0
    assert info["bank_rows"] == {"full": 30 * PATCHES} | {f"minus_fold_{f}": 24 * PATCHES for f in range(5)}


def test_labels_come_from_the_image_shown(full):
    _, z, _ = full
    names = z["specimens"].tolist()
    conditions = z["conditions"].tolist()
    base = np.array([0] * len(NORMAL) + [1, -1, 1, 1, 1], dtype=np.int8)
    assert np.array_equal(z["labels"][0], base)
    for k, condition in enumerate(conditions):
        expected = base.copy()
        if condition == "R:05":
            expected[names.index(HIDDEN_UNDER_05)] = -1
        assert np.array_equal(z["labels"][k], expected), condition
    # Synthetic conditions show the illumination-01 image, so they keep its labels.
    assert (z["labels"][1:16] == base).all()
    for j, key in enumerate(z["recal_keys"].tolist()):
        expected = base.copy()
        if key.startswith("05:"):
            expected[names.index(HIDDEN_UNDER_05)] = -1
        assert np.array_equal(z["recal_labels"][j], expected), key


def test_scores_separate_defects_and_conditions(full):
    _, z, _ = full
    conditions = z["conditions"].tolist()
    s = z["scores"][0]
    threshold = float(z["threshold"])
    visible = z["labels"][0] == 1
    assert s[visible].min() > threshold > 0
    assert s[visible].min() > 5 * s[z["labels"][0] == 0].max()
    # The never-visible defect has no bright cell: it scores like a normal image.
    assert s[z["specimens"].tolist().index(NEVER_VISIBLE)] < s[visible].min() / 5
    # A translation of less than half a pixel at this size is the identity.
    assert np.array_equal(z["scores"][conditions.index("P:shift-1")], s)
    assert not np.allclose(z["scores"][conditions.index("P:brightness-3")], s)
    # The fake illumination 03 equals the reference, the others are darker and darker.
    assert np.array_equal(z["scores"][conditions.index("R:03")], s)
    normal = z["labels"][0] == 0
    darker = [z["scores"][conditions.index(f"R:{light}")][normal].mean() for light in ("02", "06", "10")]
    assert threshold < darker[0] < darker[1] < darker[2]


def test_synthetic_conditions_are_applied_to_the_reference_test_images(tmp_path, full, monkeypatch):
    _, z, _ = full
    from defect_inspect.patchcore import build_bank, collect_features, score_images

    rows = make_rows()
    cache = FakeCache()
    train = run_m2ad.inspector_rows(rows, "Motor", "000", "train", "01")
    test = run_m2ad.inspector_rows(rows, "Motor", "000", "test", "01")
    feats, _ = collect_features(GridMean(), cache.images(train), batch_size=16, device="cpu")
    bank = build_bank(feats, 1.0, seed=0, device="cpu")
    for name in ("brightness-3", "gamma-2", "jpeg-3"):
        images = apply_condition(cache.images(test), name)
        direct = score_images(GridMean(), bank, images, batch_size=16, device="cpu", sigma=1.0).image_scores
        k = z["conditions"].tolist().index(f"P:{name}")
        np.testing.assert_allclose(z["scores"][k], direct, rtol=1e-5)


def test_recalibration_bank_contains_the_new_illumination(full):
    info, z, cache = full
    conditions = z["conditions"].tolist()
    keys = z["recal_keys"].tolist()
    normal = z["labels"][0] == 0
    train = [f"{2 * i:03d}" for i in range(N_TRAIN)]
    folds = specimen_folds(train)
    for light in ("04", "07", "10"):
        before = z["scores"][conditions.index(f"R:{light}")]
        for n in (8, 30):
            key = f"{light}:{n}"
            j = keys.index(key)
            rec = info["recal"][key]
            assert rec["images"] == 30 + n and rec["threshold"]["n"] == 30 + n
            assert rec["bank_rows"]["full"] == (30 + n) * PATCHES
            held = {
                f: sum(folds[s] == f for s in train) + sum(folds[s] == f for s in recal_specimens(train, n))
                for f in range(5)
            }
            assert {f: rec["bank_rows"][f"minus_fold_{f}"] for f in range(5)} == {
                f: (30 + n - held[f]) * PATCHES for f in range(5)
            }
            assert float(z["recal_thresholds"][j]) == rec["threshold"]["value"]
            # Normal images under the new illumination now have neighbours in the bank.
            assert z["recal_scores"][j][normal].max() < before[normal].min() / 2
            # Defects are still far from it.
            visible = z["recal_labels"][j] == 1
            assert z["recal_scores"][j][visible].min() > float(z["recal_thresholds"][j])
    # n = 30 brings every normal back under the threshold; without recalibration all were flagged.
    j = keys.index("10:30")
    assert (z["scores"][conditions.index("R:10")][normal] > float(z["threshold"])).all()
    assert (z["recal_scores"][j][normal] <= float(z["recal_thresholds"][j])).mean() >= 0.75
    # The train images of every new illumination were read once, 30 at a time.
    train_reads = [r for r in cache.reads if r[0] == "train"]
    assert train_reads == [("train", "01", 30)] + [("train", light, 30) for light in ILLUMINATIONS[1:]]


def test_recalibrated_scores_are_those_of_the_new_illumination_test_images(full):
    _, z, _ = full
    rows = make_rows()
    keys = z["recal_keys"].tolist()
    ref_train, ref_test = pick(rows, "000", "train", "01"), pick(rows, "000", "test", "01")
    train = [r.specimen for r in ref_train]
    # 05 hides one defect that the reference illumination shows; 07 only changes the brightness.
    for light in ("05", "07"):
        new_train, new_test = pick(rows, "000", "train", light), pick(rows, "000", "test", light)
        for n in (8, 30):
            chosen = set(recal_specimens(train, n))
            bank_rows = ref_train + [r for r in new_train if r.specimen in chosen]
            assert len(bank_rows) == 30 + n
            got = z["recal_scores"][keys.index(f"{light}:{n}")]
            np.testing.assert_allclose(got, direct_scores(bank_rows, new_test), rtol=1e-5)
            # Neither the reference test images nor the reference bank give these numbers.
            if n == 8 or light == "05":
                assert not np.allclose(got, direct_scores(bank_rows, ref_test), rtol=1e-3)
            assert not np.allclose(got, direct_scores(ref_train, new_test), rtol=1e-3)


def test_an_inspector_uses_the_images_of_its_own_view(tmp_path, monkeypatch):
    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02"))
    info, z = _run(tmp_path, view="120", allow_test=True)
    assert info["view"] == "120" and z["conditions"].tolist()[-1] == "R:02"
    assert z["recal_keys"].tolist() == ["02:8", "02:30"]
    rows = make_rows()
    ref_train, ref_test = pick(rows, "120", "train", "01"), pick(rows, "120", "test", "01")
    new_train, new_test = pick(rows, "120", "train", "02"), pick(rows, "120", "test", "02")
    train = [r.specimen for r in ref_train]

    # Threshold: every train image is scored by the bank of the other folds, all from view 120.
    fold_of = specimen_folds(train)
    expected = np.empty(N_TRAIN, dtype=np.float32)
    for fold in range(5):
        held = [i for i, r in enumerate(ref_train) if fold_of[r.specimen] == fold]
        rest = [r for r in ref_train if fold_of[r.specimen] != fold]
        expected[held] = direct_scores(rest, [ref_train[i] for i in held])
    np.testing.assert_allclose(z["cal_score"], expected, rtol=1e-5)
    assert float(z["threshold"]) == conformal_threshold(z["cal_score"], 0.05).value

    np.testing.assert_allclose(z["scores"][0], direct_scores(ref_train, ref_test), rtol=1e-5)
    np.testing.assert_allclose(z["scores"][-1], direct_scores(ref_train, new_test), rtol=1e-5)
    assert z["labels"][-1].tolist() == [r.label for r in new_test]
    for j, n in enumerate((8, 30)):
        chosen = set(recal_specimens(train, n))
        bank_rows = ref_train + [r for r in new_train if r.specimen in chosen]
        np.testing.assert_allclose(z["recal_scores"][j], direct_scores(bank_rows, new_test), rtol=1e-5)
    # The same specimens seen from view 000 give other numbers, as test images and as bank images.
    other_test, other_train = pick(rows, "000", "test", "02"), pick(rows, "000", "train", "02")
    assert not np.allclose(z["scores"][-1], direct_scores(ref_train, other_test), rtol=1e-3)
    assert not np.allclose(z["recal_scores"][1], direct_scores(ref_train + other_train, new_test), rtol=1e-3)


def test_config_values_reach_every_bank_and_every_scoring_call(tmp_path, monkeypatch):
    from dataclasses import replace

    from defect_inspect import patchcore

    # None of these is the default of PatchCoreConfig or of the patchcore functions.
    cfg = PatchCoreConfig(
        name="toy",
        backbone="gridmean",
        img_size=SIZE,
        coreset_ratio=0.25,
        reweight_k=3,
        sigma=2.0,
        batch_size=7,
        seed=5,
    )
    built, scored, collected = [], [], []
    real_build, real_score, real_collect = (
        patchcore.build_bank,
        patchcore.score_images,
        patchcore.collect_features,
    )

    def build(features, ratio=0.1, **kwargs):
        built.append((ratio, kwargs))
        return real_build(features, ratio, **kwargs)

    def score(extractor, bank, images, **kwargs):
        scored.append(kwargs)
        return real_score(extractor, bank, images, **kwargs)

    def collect(extractor, images, **kwargs):
        collected.append(kwargs)
        return real_collect(extractor, images, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02"))
        patch.setattr(patchcore, "build_bank", build)
        patch.setattr(patchcore, "score_images", score)
        patch.setattr(patchcore, "collect_features", collect)
        _, z = _run(tmp_path, cfg=cfg, allow_test=True)
    # Reference bank and two recalibrated ones, each with its five fold banks.
    assert len(built) == 18 and all(call == (0.25, {"seed": 5, "device": "cpu"}) for call in built)
    # 5 folds + S + 15 P + R:02, then per recalibration 5 folds + the new test images.
    assert len(scored) == 22 + 2 * 6
    for kwargs in scored:
        picked = (kwargs["batch_size"], kwargs["reweight_k"], kwargs["sigma"], kwargs["device"])
        assert picked == (7, 3, 2.0, "cpu")
    assert len(collected) == 2 and all(k == {"batch_size": 7, "device": "cpu"} for k in collected)

    # The numbers are those of a direct call with the config's values, and the values matter.
    rows = make_rows()
    ref_train, ref_test = pick(rows, "000", "train", "01"), pick(rows, "000", "test", "01")
    np.testing.assert_allclose(z["scores"][0], direct_scores(ref_train, ref_test, cfg), rtol=1e-5)
    for other in (replace(cfg, seed=0), replace(cfg, reweight_k=9), replace(cfg, coreset_ratio=0.1)):
        assert not np.allclose(z["scores"][0], direct_scores(ref_train, ref_test, other), rtol=1e-3)


def test_reference_and_new_images_of_a_specimen_are_held_out_together(full):
    info, z, _ = full
    keys = z["recal_keys"].tolist()
    # Under the fake illumination 03 the new image of a specimen equals its reference image. If one of
    # them stayed in the bank while the other was scored, all 60 cross-fitted scores (and so the
    # threshold) would be exactly 0, because the whole bank is kept.
    assert info["recal"]["03:30"]["threshold"]["rank"] == 58
    assert float(z["recal_thresholds"][keys.index("03:30")]) > 0


def test_recalibration_crossfit_gets_aligned_inputs_and_leaks_nothing(tmp_path, monkeypatch):
    from defect_inspect.patchcore import collect_features

    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", SAME_AS_REFERENCE))
    calls = []
    real_crossfit = run_m2ad.crossfit

    def spying(build, score, feats, images, folds):
        result = real_crossfit(build, score, feats, images, folds)
        calls.append((feats, images, folds, result[1]))
        return result

    monkeypatch.setattr(run_m2ad, "crossfit", spying)
    _, z = _run(tmp_path, allow_test=True)

    rows = make_rows()
    cache = FakeCache()
    ref_train = pick(rows, "000", "train", "01")
    new_train = {r.specimen: r for r in pick(rows, "000", "train", SAME_AS_REFERENCE)}
    train = [r.specimen for r in ref_train]
    fold_of = specimen_folds(train)
    assert [len(images) for _, images, _, _ in calls] == [30, 38, 60]
    for (feats, images, folds, oof), n in zip(calls, (0, 8, 30), strict=True):
        added = recal_specimens(train, n)
        # Image i, feature i and fold i belong to one specimen: the 30 reference images, then the new ones.
        assert np.array_equal(images, cache.images(ref_train + [new_train[name] for name in added]))
        direct, _ = collect_features(GridMean(), images, batch_size=16, device="cpu")
        assert torch.equal(feats.reshape(-1, feats.shape[-1]), direct)
        assert folds.tolist() == [fold_of[name] for name in train + added]
        # The new image of a specimen equals its reference image under this fake illumination, and the
        # whole bank is kept: a score of 0 (below 1e-3 after fp16 rounding, see the crossfit test) would
        # mean that one of the two stayed in the bank while the other was scored. Every one of the
        # cross-fitted scores must be clear of that.
        assert oof.shape == (30 + n,) and oof.min() > 5e-3
    assert np.array_equal(calls[0][3], z["cal_score"])


def test_crossfit_holds_out_by_fold():
    from defect_inspect.patchcore import build_bank, collect_features, score_images

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (10, SIZE, SIZE, 3), dtype=np.uint8)
    images[5:] = images[:5]  # image i + 5 is a copy of image i
    feats, _ = collect_features(GridMean(), images, batch_size=4, device="cpu")
    feats = feats.view(10, PATCHES, 3)
    calls = []

    def build(features):
        calls.append(features.shape[0])
        return build_bank(features, 1.0, seed=0, device="cpu")

    def score(bank, batch):
        extractor = GridMean()
        return score_images(extractor, bank, batch, batch_size=4, device="cpu", sigma=1.0).image_scores

    together = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    bank, oof, thr, bank_rows = run_m2ad.crossfit(build, score, feats, images, together)
    assert calls == [40, 32, 32, 32, 32, 32] and bank.shape == (40, 3)
    assert bank_rows == {"full": 40} | {f"minus_fold_{f}": 32 for f in range(5)}
    assert oof.shape == (10,) and thr.value == conformal_threshold(oof, 0.05).value
    assert oof.min() > 0.01  # no image was scored by a bank that holds it or its copy
    assert np.array_equal(oof[:5], oof[5:])  # copies in the same fold get the same score
    # With the copies in different folds every image finds itself in the bank.
    apart = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 0])
    _, leaked, _, _ = run_m2ad.crossfit(build, score, feats, images, apart)
    assert np.abs(leaked).max() < 1e-3
    with pytest.raises(RuntimeError):
        run_m2ad.crossfit(build, score, feats, images, np.array([0, 1, 2, 3, 7, 0, 1, 2, 3, 7]))


def test_coreset_ratio_of_the_config_is_used(tmp_path):
    cfg = PatchCoreConfig(
        name="toy", backbone="gridmean", img_size=SIZE, coreset_ratio=0.25, sigma=1.0, batch_size=16
    )
    info, _ = _run(tmp_path, cfg=cfg, check=True)
    assert info["bank_rows"]["full"] == 30 * PATCHES // 4
    assert info["bank_rows"]["minus_fold_0"] == 24 * PATCHES // 4


def test_run_is_deterministic(tmp_path, full):
    _, z, _ = full
    _, again = _run(tmp_path, allow_test=True)
    assert z.keys() == again.keys()
    for key in z:
        assert np.array_equal(z[key], again[key]), key


def test_check_runs_one_inspector_without_recalibration(tmp_path, full):
    _, z, _ = full
    cache = FakeCache()
    info, c = _run(tmp_path, cache=cache, check=True)
    assert c["conditions"].tolist() == ["S", "R:02"]
    assert c["scores"].shape == (2, 9) and c["labels"].shape == (2, 9)
    assert c["recal_keys"].shape == (0,) and c["recal_thresholds"].shape == (0,)
    assert c["recal_scores"].shape == (0, 9) and c["recal_labels"].shape == (0, 9)
    assert info["recal"] == {}
    # Same inspector, same numbers as in the full run.
    assert float(c["threshold"]) == float(z["threshold"])
    assert np.array_equal(c["scores"][0], z["scores"][0])
    assert np.array_equal(c["scores"][1], z["scores"][z["conditions"].tolist().index("R:02")])
    # Only the two conditions of the check were read from the test split, and only reference train images.
    assert cache.reads == [("train", "01", 30), ("test", "01", 9), ("test", "02", 9)]


def test_test_images_need_permission(tmp_path):
    cache = FakeCache()
    with pytest.raises(SealedTestError):
        _run(tmp_path, cache=cache)
    with pytest.raises(SealedTestError):
        _run(tmp_path, cache=cache, allow_test="yes")
    # The check exempts one inspector only.
    with pytest.raises(SealedTestError):
        _run(tmp_path, view="120", cache=cache, check=True)
    assert cache.reads == [] and not list(tmp_path.glob("*.npz"))
    _run(tmp_path, view="120", cache=cache, check=True, allow_test=True)


def test_specimens_must_match_across_illuminations(tmp_path):
    rows = [
        r for r in make_rows() if not (r.split == "test" and r.illumination == "02" and r.specimen == "001")
    ]
    with pytest.raises(ValueError, match="illumination 02"):
        run_m2ad.run_inspector(
            CFG, "Motor", "000", rows, FakeCache(), GridMean(), "cpu", tmp_path, allow_test=True
        )
    with pytest.raises(ValueError, match="no train images"):
        run_m2ad.run_inspector(
            CFG, "Bird", "000", rows, FakeCache(), GridMean(), "cpu", tmp_path, allow_test=True
        )


# ----------------------------------------------------------------------------------------------------- CLI


@pytest.fixture
def cli(tmp_path, monkeypatch, temporary_ledger):
    cache = FakeCache()
    rows = make_rows(("Motor", "Bird"))
    monkeypatch.setattr(run_m2ad, "read_meta", lambda path: rows)
    monkeypatch.setattr(run_m2ad, "M2adCache", lambda out_dir, size: cache)
    monkeypatch.setattr(run_m2ad, "get_config", lambda name: CFG)
    monkeypatch.setattr(run_m2ad, "_make_extractor", lambda cfg, device: GridMean())
    return cache, ["--method", "p0", "--device", "cpu", "--out", str(tmp_path / "m2ad-p0")]


def test_cli_check_needs_no_permission_and_writes_no_ledger_line(tmp_path, cli, temporary_ledger, capsys):
    cache, argv = cli
    run_m2ad.main([*argv, "--check"])
    out_dir = tmp_path / "m2ad-p0-check"
    assert sorted(p.name for p in out_dir.iterdir()) == ["Motor_000.npz", "run.json"]
    assert not (tmp_path / "m2ad-p0").exists()
    assert not temporary_ledger.exists()
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["method"] == "p0" and meta["check"] is True and meta["device"] == "cpu"
    assert meta["commit"] == "abc1234" and meta["conditions"] == ["S", "R:02"] and meta["recal_keys"] == []
    assert [(i["category"], i["view"]) for i in meta["inspectors"]] == [("Motor", "000")]
    assert meta["inspectors"][0]["seconds"] >= 0 and meta["inspectors"][0]["bank_rows"]["full"] == 120
    assert json.loads(capsys.readouterr().out.splitlines()[0])["view"] == "000"
    # A finished run is not redone without --overwrite.
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--check"])
    run_m2ad.main([*argv, "--check", "--overwrite"])


def test_cli_check_stays_the_check_when_the_full_run_flags_are_given(
    tmp_path, cli, temporary_ledger, monkeypatch
):
    cache, argv = cli
    calls = []
    real_inspector = run_m2ad.run_inspector

    def watching(cfg, category, view, *args, **kwargs):
        calls.append((category, view, kwargs["allow_test"], kwargs["check"]))
        return real_inspector(cfg, category, view, *args, **kwargs)

    monkeypatch.setattr(run_m2ad, "run_inspector", watching)
    run_m2ad.main([*argv, "--check", "--allow-test", "--stage", "3-m2ad"])
    # One inspector, without the permission of the full run, no ledger line, only the check's reads.
    assert calls == [("Motor", "000", False, True)]
    assert not temporary_ledger.exists() and not (tmp_path / "m2ad-p0").exists()
    assert cache.reads == [("train", "01", 30), ("test", "01", 9), ("test", "02", 9)]


def test_cli_full_run_needs_the_flags(tmp_path, cli, temporary_ledger):
    cache, argv = cli
    with pytest.raises(SystemExit):
        run_m2ad.main(argv)
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--allow-test"])
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--stage", "3-m2ad"])
    assert cache.reads == [] and not temporary_ledger.exists()
    assert not (tmp_path / "m2ad-p0").exists()


def test_cli_full_run_records_the_ledger_line_first(tmp_path, cli, temporary_ledger, monkeypatch):
    cache, argv = cli
    # Three illuminations keep the six inspectors quick; the full condition list is tested above.
    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02", "03"))
    seen = []
    cache.on_test_read = lambda: seen.append(len(ledger_entries(temporary_ledger)))
    run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad", "--note", "full"])
    entries = ledger_entries(temporary_ledger)
    assert len(entries) == 1
    assert {k: v for k, v in entries[0].items() if k != "time"} == {
        "commit": "abc1234",
        "stage": "3-m2ad",
        "config": "m2ad-p0",
        "note": "full",
    }
    assert seen and set(seen) == {1}  # the line was there before every read of test images
    out_dir = tmp_path / "m2ad-p0"
    inspectors = [f"{category}_{view}" for category in ("Motor", "Bird") for view in VIEWS]
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        [f"{i}.npz" for i in inspectors] + ["run.json"]
    )
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert [f"{i['category']}_{i['view']}" for i in meta["inspectors"]] == inspectors
    assert meta["check"] is False and meta["config"]["name"] == "toy" and meta["alpha"] == 0.05
    assert meta["conditions"] == ["S", *(f"P:{n}" for n in condition_names(False)), "R:02", "R:03"]
    assert meta["recal_keys"] == ["02:8", "02:30", "03:8", "03:30"]
    for name in inspectors:
        with np.load(out_dir / f"{name}.npz") as z:
            assert z["conditions"].tolist() == meta["conditions"]
            assert z["recal_keys"].tolist() == meta["recal_keys"]
            assert z["scores"].shape == (18, 9) and z["recal_scores"].shape == (4, 9)
    # The two categories are different objects: their inspectors do not share scores.
    with np.load(out_dir / "Motor_120.npz") as a, np.load(out_dir / "Bird_120.npz") as b:
        assert not np.array_equal(a["scores"], b["scores"])


def test_cli_stops_before_the_ledger_when_the_cache_is_missing(tmp_path, monkeypatch, temporary_ledger):
    rows = make_rows()
    monkeypatch.setattr(run_m2ad, "read_meta", lambda path: rows)
    argv = ["--method", "p0", "--device", "cpu", "--out", str(tmp_path / "out"), "--cache", str(tmp_path)]
    with pytest.raises(FileNotFoundError, match="defect_inspect.m2ad --size 256"):
        run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad"])
    assert not temporary_ledger.exists() and not (tmp_path / "out").exists()


def test_cli_one_category_and_overwrite(tmp_path, cli, temporary_ledger, monkeypatch):
    cache, argv = cli
    monkeypatch.setattr(run_m2ad, "ILLUMINATIONS", ("01", "02"))
    full = [*argv, "--allow-test", "--stage", "3-m2ad", "--categories", "Bird"]
    run_m2ad.main(full)
    out_dir = tmp_path / "m2ad-p0"
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert [(i["category"], i["view"]) for i in meta["inspectors"]] == [("Bird", view) for view in VIEWS]
    assert not list(out_dir.glob("Motor_*.npz"))
    with pytest.raises(SystemExit):
        run_m2ad.main(full)
    assert len(ledger_entries(temporary_ledger)) == 1

    # A rerun that dies half-way leaves no run.json behind, so it cannot pass for the finished run.
    def dying(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(run_m2ad, "run_inspector", dying)
    with pytest.raises(RuntimeError, match="stopped"):
        run_m2ad.main([*full, "--overwrite"])
    assert not (out_dir / "run.json").exists()
    assert len(ledger_entries(temporary_ledger)) == 2  # the rerun was allowed to read test images
    with pytest.raises(SystemExit):
        run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad", "--categories", "Car"])


def test_cli_overwrite_keeps_the_finished_run_when_the_ledger_cannot_be_written(
    tmp_path, cli, temporary_ledger, monkeypatch
):
    cache, argv = cli
    out_dir = tmp_path / "m2ad-p0"
    out_dir.mkdir()
    (out_dir / "run.json").write_text('{"finished": true}\n', encoding="utf-8")
    (out_dir / "Motor_000.npz").write_bytes(b"old scores")
    started = []

    def locked(*args, **kwargs):
        raise TimeoutError("could not take the ledger lock")

    monkeypatch.setattr(run_m2ad, "record_test_access", locked)
    monkeypatch.setattr(run_m2ad, "run_inspector", lambda *args, **kwargs: started.append(args[1:3]))
    with pytest.raises(TimeoutError, match="ledger lock"):
        run_m2ad.main([*argv, "--allow-test", "--stage", "3-m2ad", "--overwrite"])
    # The rerun never started: no inspector, no read, and the finished run can still be analysed.
    assert started == [] and cache.reads == [] and not temporary_ledger.exists()
    assert (out_dir / "run.json").read_text(encoding="utf-8") == '{"finished": true}\n'
    assert (out_dir / "Motor_000.npz").read_bytes() == b"old scores"

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defect_inspect import analyze_perturb, backbones, run_patchcore, run_perturb  # noqa: E402
from defect_inspect.conditions import apply_condition, condition_names  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.download import sha256_file  # noqa: E402
from defect_inspect.patchcore import score_images  # noqa: E402
from defect_inspect.splits import (  # noqa: E402
    SEALED_ROLES,
    ManifestRow,
    SealedTestError,
    select,
    write_manifest,
)

SIZE = 32
NPZ_KEYS = {"eval_images", "eval_labels", "conditions", "scores", "cal_score"}
DEV_IMAGES = [f"toy/n{i}.JPG" for i in range(0, 40, 5)] + [f"toy/d{i}.JPG" for i in range(6)]
TEST_IMAGES = [f"toy/tn{i}.JPG" for i in range(10)] + [f"toy/td{i}.JPG" for i in range(4)]


class GridMean(torch.nn.Module):
    """Tiny stand-in for a backbone: the mean colour of each cell of a 4x4 grid."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 4).permute(0, 2, 3, 1)


class Recording(GridMean):
    """GridMean that notes the size of every batch it is given."""

    def __init__(self):
        super().__init__()
        self.batches: list[int] = []

    def forward(self, x):
        self.batches.append(int(x.shape[0]))
        return super().forward(x)


class FakeCache:
    """In-memory stand-in for ImageCache: defects carry a bright square that no normal image has."""

    size = SIZE

    def __init__(self, manifest, ledger=None):
        rng = np.random.default_rng(0)
        self._images, self._masks, self._role = {}, {}, {}
        self._ledger = ledger
        self.requested: list[str] = []
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask
            self._role[row.image] = row.role

    def _note(self, rows):
        if self._ledger is not None and any(self._role[r.image] in SEALED_ROLES for r in rows):
            # The read of sealed images must already be on record.
            assert self._ledger.exists() and self._ledger.read_text(encoding="utf-8").strip()
        self.requested += [r.image for r in rows]

    def images(self, rows):
        self._note(rows)
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        self._note(rows)
        return np.stack([self._masks[r.image] for r in rows])


def _manifest(categories=("toy",)):
    rows = []
    for c in categories:
        for i in range(40):
            rows.append(ManifestRow(f"{c}/n{i}.JPG", "", c, "normal", "pool_normal", i % 5, ""))
        for i in range(6):
            rows.append(ManifestRow(f"{c}/d{i}.JPG", f"{c}/d{i}.png", c, "anomaly", "dev_defect", -1, "hole"))
        for i in range(10):
            rows.append(ManifestRow(f"{c}/tn{i}.JPG", "", c, "normal", "test_normal", -1, ""))
        for i in range(4):
            mask = f"{c}/td{i}.png"
            rows.append(ManifestRow(f"{c}/td{i}.JPG", mask, c, "anomaly", "test_defect", -1, "hole"))
    return rows


# Named like the registered config, so that the CLI (which looks the method up by name) can use it.
# No scoring option is left at the default of `score_images` (9, 4.0, 32): a run that ignored the config
# would not reproduce these scores.
CFG = PatchCoreConfig(
    name="p0", backbone="gridmean", img_size=SIZE, coreset_ratio=0.5, reweight_k=3, sigma=1.0, batch_size=8
)


def _make_source(src, protocol, categories=("toy",)):
    """A finished `run_patchcore` run with its memory banks, as the perturbation run expects it."""
    manifest = _manifest(categories)
    src.mkdir(parents=True)
    cache = FakeCache(manifest)
    infos = [
        run_patchcore.run_category(
            CFG, protocol, c, manifest, cache, GridMean(), "cpu", protocol == "test", src, True
        )
        for c in categories
    ]
    meta = {
        "config": asdict(CFG),
        "protocol": protocol,
        "commit": "src1234",
        "device": "cpu",
        "categories": infos,
    }
    (src / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return manifest


def _scorer(src):
    bank = torch.from_numpy(np.load(src / "toy_bank.npy"))
    return run_perturb.patchcore_scorer(CFG, GridMean(), bank, "cpu")


def _perturb(src, out, protocol="dev", allow_test=False, cache=None, score=None):
    manifest = _manifest()
    out.mkdir(parents=True, exist_ok=True)
    return run_perturb.run_category(
        protocol,
        "toy",
        manifest,
        cache or FakeCache(manifest),
        score or _scorer(src),
        run_perturb.load_source(src, "toy", "p0"),
        allow_test,
        out,
    )


# ---------------------------------------------------------------- one category, PatchCore


def test_every_condition_is_scored_in_the_registered_order(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    manifest = _make_source(src, "dev")
    info = _perturb(src, out)
    assert (info["category"], info["eval_normal"], info["eval_defect"], info["cal"]) == ("toy", 8, 6, 32)
    assert info["clean_max_rel_diff"] == 0.0  # same bank, images and batches on the CPU: identical
    assert set(info["timing"]) == {"perturb_s", "score_s"}
    assert sorted(p.name for p in out.iterdir()) == ["toy.npz"]  # no maps, no pixel metrics

    names = condition_names()
    with np.load(out / "toy.npz") as z, np.load(src / "toy.npz") as source:
        assert set(z.files) == NPZ_KEYS
        assert z["conditions"].tolist() == names and names[0] == "clean" and len(names) == 16
        assert z["eval_images"].tolist() == source["eval_images"].tolist() == DEV_IMAGES
        assert z["eval_labels"].dtype == np.int8 and z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert z["scores"].shape == (16, 14) and z["scores"].dtype == np.float32
        assert np.array_equal(z["scores"][0], source["eval_score_full"])
        # The thresholds of a PatchCore run come from the cross-fitted scores of the normal pool.
        assert z["cal_score"].dtype == np.float32
        assert np.array_equal(z["cal_score"], source["pool_score_oof"])
        scores = z["scores"]

    # Every row is the saved bank scoring the evaluation images under that condition, nothing else.
    rows = select(manifest, protocol="dev", part="eval_normal", category="toy")
    rows += select(manifest, protocol="dev", part="eval_defect", category="toy")
    images = FakeCache(manifest).images(rows)
    bank = torch.from_numpy(np.load(src / "toy_bank.npy"))
    for k, name in enumerate(names):
        direct = score_images(
            GridMean(),
            bank,
            apply_condition(images, name),
            batch_size=CFG.batch_size,
            device="cpu",
            reweight_k=CFG.reweight_k,
            sigma=CFG.sigma,
        )
        assert np.array_equal(scores[k], direct.image_scores), name
    # A darker image is far from a bank of normally lit patches: every normal scores above the clean ones.
    assert scores[names.index("brightness-3")][:8].min() > scores[0][:8].max()
    # The comparison above depends on the config: the default re-weighting gives other scores.
    default_k = score_images(GridMean(), bank, images, batch_size=CFG.batch_size, device="cpu")
    assert not np.allclose(default_k.image_scores, scores[0], rtol=1e-3)


def test_the_scorer_takes_batch_size_and_scoring_options_from_the_config(tmp_path, monkeypatch):
    from defect_inspect import patchcore

    src = tmp_path / "src"
    manifest = _make_source(src, "dev")
    images = FakeCache(manifest).images(manifest[:14])
    bank = torch.from_numpy(np.load(src / "toy_bank.npy"))
    passed = []

    def spy(extractor, bank, images, **options):
        passed.append(options)
        return score_images(extractor, bank, images, **options)

    monkeypatch.setattr(patchcore, "score_images", spy)
    for cfg, batches in (
        (CFG, [8, 6]),
        (PatchCoreConfig("x", "gridmean", SIZE, 0.5, reweight_k=5, sigma=2.0, batch_size=5), [5, 5, 4]),
    ):
        extractor = Recording()
        passed.clear()
        scores = run_perturb.patchcore_scorer(cfg, extractor, bank, "cpu")(images)
        assert extractor.batches == batches  # the images really went through in batches of cfg.batch_size
        assert passed == [
            {"batch_size": cfg.batch_size, "device": "cpu", "reweight_k": cfg.reweight_k, "sigma": cfg.sigma}
        ]
        direct = score_images(
            GridMean(), bank, images, batch_size=cfg.batch_size, device="cpu", reweight_k=cfg.reweight_k
        )
        assert scores.dtype == np.float32 and np.array_equal(scores, direct.image_scores)


def test_a_tampered_bank_fails_the_clean_guard(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_source(src, "dev")
    bank = np.load(src / "toy_bank.npy")
    np.save(src / "toy_bank.npy", (bank + np.float16(0.5)).astype(np.float16))
    calls = []
    real = _scorer(src)

    def counting(images):
        calls.append(len(images))
        return real(images)

    with pytest.raises(run_perturb.CleanScoreMismatch, match="clean scores differ"):
        _perturb(src, out, score=counting)
    assert calls == [14]  # it stops at the clean condition
    assert not (out / "toy.npz").exists()


def test_the_clean_guard_uses_the_relative_limit(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    real = _scorer(src)
    info = _perturb(src, tmp_path / "ok", score=lambda images: real(images) * np.float32(1 + 5e-4))
    assert 4e-4 < info["clean_max_rel_diff"] < 6e-4
    with pytest.raises(run_perturb.CleanScoreMismatch):
        _perturb(src, tmp_path / "bad", score=lambda images: real(images) * np.float32(1 + 2e-3))


@pytest.mark.parametrize("odd", [0, 5, 13])
def test_the_clean_guard_looks_at_every_image(tmp_path, odd):
    src = tmp_path / "src"
    _make_source(src, "dev")
    real = _scorer(src)

    def one_off(images):
        scores = real(images).copy()
        scores[odd] *= np.float32(1 + 2e-3)  # one image out of 14, anywhere in the set
        return scores

    with pytest.raises(run_perturb.CleanScoreMismatch):
        _perturb(src, tmp_path / "out", score=one_off)
    assert not (tmp_path / "out" / "toy.npz").exists()


def test_max_rel_diff():
    assert run_perturb.max_rel_diff(np.array([1.0, 2.0]), np.array([1.0, 2.5])) == pytest.approx(0.2)
    assert run_perturb.max_rel_diff(np.zeros(3), np.zeros(3)) == 0.0
    assert run_perturb.max_rel_diff(np.zeros(0), np.zeros(0)) == 0.0
    assert run_perturb.max_rel_diff(np.array([1e-6]), np.array([0.0])) > 1.0
    with pytest.raises(ValueError):
        run_perturb.max_rel_diff(np.zeros(3), np.zeros(4))


def test_a_source_that_evaluated_other_images_is_rejected(tmp_path):
    src = tmp_path / "src"
    manifest = _make_source(src, "dev")
    source = run_perturb.load_source(src, "toy", "p0")
    swapped = run_perturb.Source(source.eval_images[::-1].copy(), source.eval_score, source.cal_score)
    cache = FakeCache(manifest)
    with pytest.raises(ValueError, match="did not evaluate"):
        run_perturb.run_category("dev", "toy", manifest, cache, _scorer(src), swapped, False, tmp_path)
    assert cache.requested == []


def test_bad_scores_are_rejected(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    with pytest.raises(FloatingPointError):
        _perturb(src, tmp_path / "nan", score=lambda images: np.full(len(images), np.nan, np.float32))
    with pytest.raises(ValueError, match="expected 14 scores"):
        _perturb(src, tmp_path / "short", score=lambda images: np.zeros(3, np.float32))


def test_the_sealed_test_set_needs_permission(tmp_path):
    src = tmp_path / "src"
    manifest = _make_source(src, "test")
    cache = FakeCache(manifest)
    with pytest.raises(SealedTestError):
        _perturb(src, tmp_path / "out", protocol="test", cache=cache)
    assert cache.requested == [] and not (tmp_path / "out" / "toy.npz").exists()

    info = _perturb(src, tmp_path / "out", protocol="test", allow_test=True, cache=cache)
    assert (info["eval_normal"], info["eval_defect"], info["cal"]) == (10, 4, 40)
    assert cache.requested == TEST_IMAGES
    with np.load(tmp_path / "out" / "toy.npz") as z:
        assert z["eval_images"].tolist() == TEST_IMAGES
        assert z["eval_labels"].tolist() == [0] * 10 + [1] * 4
        assert z["scores"].shape == (16, 14) and z["cal_score"].shape == (40,)


def test_load_source_names_a_run_of_the_wrong_kind(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    with pytest.raises(KeyError, match="eval_score"):
        run_perturb.load_source(src, "toy", "dm")  # a PatchCore run is not in the common format


def _rewrite(path, **changes):
    """Replace arrays of an npz file (None drops the array)."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update(changes)
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None})


def test_load_source_rejects_scores_that_cannot_be_compared(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    good = run_perturb.load_source(src, "toy", "p0")
    assert good.eval_score.shape == (14,) and good.cal_score.shape == (32,)

    _rewrite(src / "toy.npz", eval_score_full=good.eval_score[:-1])
    with pytest.raises(ValueError, match="14 evaluation images"):
        run_perturb.load_source(src, "toy", "p0")
    bad = good.eval_score.copy()
    bad[3] = np.nan
    _rewrite(src / "toy.npz", eval_score_full=bad)
    with pytest.raises(ValueError, match="not finite"):
        run_perturb.load_source(src, "toy", "p0")
    cal = good.cal_score.copy()
    cal[0] = np.inf
    _rewrite(src / "toy.npz", eval_score_full=good.eval_score, pool_score_oof=cal)
    with pytest.raises(ValueError, match="not finite"):
        run_perturb.load_source(src, "toy", "p0")
    _rewrite(src / "toy.npz", pool_score_oof=good.cal_score.reshape(4, 8))
    with pytest.raises(ValueError, match="pool_score_oof"):
        run_perturb.load_source(src, "toy", "p0")


# ---------------------------------------------------------------- the CLI, PatchCore


def _project(tmp_path, monkeypatch, categories=("toy",)):
    """The CLI wired to a toy manifest, a fake cache, the stand-in backbone and a ledger under tmp_path."""
    manifest = _manifest(categories)
    ledger = tmp_path / "ledger.jsonl"
    cache = FakeCache(manifest, ledger=ledger)
    outputs = tmp_path / "outputs"
    extractor = Recording()

    def fake_cache(root, size):
        assert size == SIZE
        return cache

    def fake_extractor(name, *, img_size, pretrained=True):
        assert (name, img_size) == ("gridmean", SIZE)
        return extractor

    def fake_config(name):
        assert name == "p0"
        return CFG

    def no_dinomaly():
        raise AssertionError("run_dinomaly must only be imported for the dm method")

    write_manifest(manifest, tmp_path / "visa.csv")
    monkeypatch.setattr(run_perturb.paths, "VISA_MANIFEST", tmp_path / "visa.csv")
    monkeypatch.setattr(run_perturb.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(run_perturb.paths, "OUTPUTS", outputs)
    monkeypatch.setattr(run_perturb.paths, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(run_perturb, "ImageCache", fake_cache)
    monkeypatch.setattr(run_perturb, "get_config", fake_config)
    monkeypatch.setattr(run_perturb, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(run_perturb, "_dinomaly", no_dinomaly)
    monkeypatch.setattr(backbones, "make_extractor", fake_extractor)
    return SimpleNamespace(
        cache=cache, ledger=ledger, outputs=outputs, manifest=manifest, extractor=extractor
    )


@pytest.fixture
def toy_project(tmp_path, monkeypatch):
    return _project(tmp_path, monkeypatch)


@pytest.fixture
def two_project(tmp_path, monkeypatch):
    """The same with two categories, for what must hold for every category and not just the first."""
    return _project(tmp_path, monkeypatch, TWO)


TWO = ("toy", "toz")
DEV = ["--method", "p0", "--protocol", "dev", "--device", "cpu", "--categories", "toy"]
TEST = ["--method", "p0", "--protocol", "test", "--device", "cpu", "--categories", "toy"]
ALLOW = ["--allow-test", "--stage", "3"]


def _untouched(project, out_name):
    """No ledger line, no image read and no output directory: the run stopped before it started."""
    return (
        not project.ledger.exists()
        and project.cache.requested == []
        and not (project.outputs / out_name).exists()
    )


def test_cli_dev_run_and_its_analysis(toy_project, capsys):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-dev", "dev")
    run_perturb.main(DEV)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["category"] for p in printed] == ["toy"]

    out = outputs / "perturb-p0-dev"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert (run["method"], run["protocol"], run["commit"], run["device"]) == ("p0", "dev", "abc1234", "cpu")
    assert run["source"] == str(outputs / "p0-dev") and run["source_commit"] == "src1234"
    assert run["conditions"] == condition_names() and run["config"] == asdict(CFG)
    assert [c["category"] for c in run["categories"]] == ["toy"]
    assert run["categories"][0]["seconds"] >= 0 and run["categories"][0]["clean_max_rel_diff"] == 0.0
    assert b"\r" not in (out / "run.json").read_bytes()
    assert toy_project.cache.requested == DEV_IMAGES  # nothing but the dev evaluation set was read
    assert not toy_project.ledger.exists()
    assert sorted(p.name for p in out.iterdir()) == ["run.json", "toy.npz"]
    # 14 images in batches of the config's size, once per condition.
    assert toy_project.extractor.batches == [8, 6] * 16

    # The analysis reads the run as it is. Darkening leaves the ranking alone (the defects still stand
    # out) but pushes every normal over its threshold: the case H9 is about.
    loaded = analyze_perturb.load(out)
    table = analyze_perturb.condition_table(loaded, n_boot=50)
    assert [row["condition"] for row in table] == condition_names()
    dark = table[condition_names().index("brightness-3")]
    assert table[0]["macro_image_auroc"] == 1.0 and dark["d_auroc"] == 0.0
    assert dark["fpr"] == 1.0 and table[0]["fpr"] < 0.5
    assert "brightness-3" in analyze_perturb.h9(table)["conditions"]

    # A finished run is not overwritten by accident.
    with pytest.raises(SystemExit):
        run_perturb.main(DEV)
    run_perturb.main(DEV + ["--overwrite"])
    assert (out / "run.json").exists()


def test_cli_missing_bank_names_the_fix(toy_project, capsys):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-test", "test")
    (outputs / "p0-test" / "toy_bank.npy").unlink()
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + ["--allow-test", "--stage", "3"])
    err = capsys.readouterr().err
    assert "run_patchcore --config p0 --protocol test --save-bank" in err and "toy" in err
    # Nothing was recorded or read, and nothing was left behind.
    assert not toy_project.ledger.exists() and toy_project.cache.requested == []
    assert not (outputs / "perturb-p0-test").exists()


def test_cli_refuses_the_test_protocol_without_flags(toy_project, monkeypatch):
    _make_source(toy_project.outputs / "p0-test", "test")
    monkeypatch.setattr(run_perturb, "ImageCache", None)  # nothing may be opened before the refusal
    for extra in ([], ["--allow-test"], ["--stage", "3"]):
        with pytest.raises(SystemExit):
            run_perturb.main(TEST + extra)
    assert not toy_project.ledger.exists() and toy_project.cache.requested == []
    assert not (toy_project.outputs / "perturb-p0-test").exists()
    with pytest.raises(SystemExit):
        run_perturb.main(["--method", "p0"])  # the protocol must be named
    with pytest.raises(SystemExit):
        run_perturb.main(["--protocol", "dev"])  # and so must the method


def test_cli_test_protocol_records_the_read_first(toy_project):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-test", "test")
    # FakeCache asserts that the ledger line exists before it hands out a sealed image.
    run_perturb.main(TEST + ["--allow-test", "--stage", "3", "--note", "toy"])
    lines = toy_project.ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert (entry["stage"], entry["config"], entry["commit"], entry["note"]) == (
        "3",
        "perturb-p0",
        "abc1234",
        "toy",
    )
    assert toy_project.cache.requested == TEST_IMAGES
    run = json.loads((outputs / "perturb-p0-test" / "run.json").read_text(encoding="utf-8"))
    assert run["protocol"] == "test" and run["categories"][0]["cal"] == 40
    with np.load(outputs / "perturb-p0-test" / "toy.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 10 + [1] * 4 and z["scores"].shape == (16, 14)


def test_cli_checks_the_source_before_anything_is_read(toy_project, capsys):
    outputs = toy_project.outputs
    allow = ["--allow-test", "--stage", "3"]
    # No source run at all.
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow)
    assert "holds no finished run" in capsys.readouterr().err

    # A dev run cannot lend its thresholds to the test protocol.
    _make_source(outputs / "p0-dev", "dev")
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow + ["--source", str(outputs / "p0-dev")])
    assert "'dev' run" in capsys.readouterr().err

    # A source made with another configuration than the registered one.
    _make_source(outputs / "p0-test", "test")
    meta = json.loads((outputs / "p0-test" / "run.json").read_text(encoding="utf-8"))
    meta["config"]["coreset_ratio"] = 0.25
    (outputs / "p0-test" / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow)
    assert "registered config" in capsys.readouterr().err

    # A memory bank with another number of rows than the source run recorded (left by another run).
    meta["config"] = asdict(CFG)
    (outputs / "p0-test" / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    bank = np.load(outputs / "p0-test" / "toy_bank.npy")
    np.save(outputs / "p0-test" / "toy_bank.npy", bank[: len(bank) // 2])
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow)
    assert "written by another run" in capsys.readouterr().err

    # Unknown categories and a model file for a method without one.
    with pytest.raises(SystemExit):
        run_perturb.main(DEV[:-1] + ["nothing"])
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--model", str(outputs / "model.pt")])
    assert not toy_project.ledger.exists() and toy_project.cache.requested == []


def test_cli_stops_at_a_bank_that_does_not_match(toy_project):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-dev", "dev")
    bank = np.load(outputs / "p0-dev" / "toy_bank.npy")
    np.save(outputs / "p0-dev" / "toy_bank.npy", (bank * np.float16(1.5)).astype(np.float16))
    with pytest.raises(run_perturb.CleanScoreMismatch):
        run_perturb.main(DEV)
    assert not (outputs / "perturb-p0-dev" / "run.json").exists()  # not a finished run


def test_cli_a_redo_that_fails_leaves_no_finished_run(toy_project, capsys):
    outputs = toy_project.outputs
    src, out = outputs / "p0-dev", outputs / "perturb-p0-dev"
    _make_source(src, "dev")
    run_perturb.main(DEV)
    finished = (out / "run.json").read_bytes()

    # Refused before it starts (the source lost a bank): the finished run stays as it is.
    bank = np.load(src / "toy_bank.npy")
    (src / "toy_bank.npy").unlink()
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--overwrite"])
    assert "--save-bank" in capsys.readouterr().err
    assert (out / "run.json").read_bytes() == finished

    # Started and stopped at the guard (another bank with the same number of rows): the directory now
    # holds part of a redo, so it must not pass for a finished run any more.
    np.save(src / "toy_bank.npy", (bank * np.float16(1.5)).astype(np.float16))
    with pytest.raises(run_perturb.CleanScoreMismatch):
        run_perturb.main(DEV + ["--overwrite"])
    assert not (out / "run.json").exists()
    with pytest.raises(FileNotFoundError):
        analyze_perturb.load(out)


def test_cli_checks_every_category_before_the_test_read(two_project, capsys):
    outputs = two_project.outputs
    src = outputs / "p0-test"
    _make_source(src, "test", TWO)
    args = TEST + ["toz"] + ALLOW
    with np.load(src / "toz.npz") as z:
        good = {k: z[k] for k in z.files}

    # Each of these is wrong in the SECOND category only and needs no image to be noticed: no ledger
    # line may be written for a read that does not happen, and the first category must not be read.
    broken = {
        "has no ['pool_score_oof']": {"pool_score_oof": None},
        "did not evaluate": {"eval_images": good["eval_images"][::-1].copy()},
        "14 evaluation images": {"eval_score_full": good["eval_score_full"][:-1]},
        "not finite": {"eval_score_full": np.full(14, np.nan, dtype=np.float32)},
    }
    for message, change in broken.items():
        np.savez(src / "toz.npz", **good)
        _rewrite(src / "toz.npz", **change)
        with pytest.raises(SystemExit):
            run_perturb.main(args)
        err = capsys.readouterr().err
        assert message in err and "toz" in err, message
        assert _untouched(two_project, "perturb-p0-test"), message

    # The same in the first category.
    np.savez(src / "toz.npz", **good)
    _rewrite(src / "toy.npz", pool_score_oof=None)
    with pytest.raises(SystemExit):
        run_perturb.main(args)
    assert "toy.npz" in capsys.readouterr().err and _untouched(two_project, "perturb-p0-test")


def test_cli_two_categories_use_their_own_bank_and_scores(two_project):
    outputs = two_project.outputs
    src, out = outputs / "p0-test", outputs / "perturb-p0-test"
    _make_source(src, "test", TWO)
    run_perturb.main(TEST + ["toz"] + ALLOW)
    assert len(two_project.ledger.read_text(encoding="utf-8").splitlines()) == 1  # one line per run
    assert two_project.cache.requested == TEST_IMAGES + [p.replace("toy/", "toz/") for p in TEST_IMAGES]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert [c["category"] for c in run["categories"]] == list(TWO)
    assert [c["clean_max_rel_diff"] for c in run["categories"]] == [0.0, 0.0]
    for c in TWO:
        with np.load(out / f"{c}.npz") as z, np.load(src / f"{c}.npz") as source:
            assert np.array_equal(z["scores"][0], source["eval_score_full"])
            assert np.array_equal(z["cal_score"], source["pool_score_oof"])
    with np.load(src / "toy.npz") as a, np.load(src / "toz.npz") as b:
        assert not np.array_equal(a["eval_score_full"], b["eval_score_full"])  # the categories differ


def test_cli_refuses_to_write_into_the_source_run(toy_project, capsys):
    outputs = toy_project.outputs
    src = outputs / "p0-dev"
    _make_source(src, "dev")
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    for out in (str(src), str(src / ".." / "p0-dev"), str(src) + "/"):
        for extra in ([], ["--overwrite"]):
            with pytest.raises(SystemExit):
                run_perturb.main(DEV + ["--out", out] + extra)
            assert "source run" in capsys.readouterr().err
    # Nothing of the source run was replaced or removed, and nothing was read.
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before
    assert toy_project.cache.requested == []
    with np.load(src / "toy.npz") as z:
        assert "eval_score_full" in z.files and "pool_score_oof" in z.files

    # The same directory named through --source.
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--source", str(src), "--out", str(src), "--overwrite"])
    assert "source run" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before


# ---------------------------------------------------------------- Dinomaly, with a stand-in


class TinyDm(torch.nn.Module):
    """Stand-in for the trained model: one trainable gain on a brightness score."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(()))


def _dm_module(calls, with_loader=True):
    """What `run_perturb` uses of `run_dinomaly`, without anomalib."""

    def build_model(encoder_name="tiny"):
        calls.append(("build", encoder_name))
        return TinyDm()

    def load_model(path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = TinyDm()
        model.load_state_dict(saved["state"])
        calls.append(("load", str(path)))
        return model.eval(), {"encoder": saved["encoder"], "steps": saved["steps"]}

    def predict(model, images, *, batch_size=32, device="cuda", amp=True, map_size=256):
        calls.append(("predict", len(images), batch_size, device, amp))
        x = torch.from_numpy(images).float().flatten(1)
        scores = float(model.gain.detach()) * (x.amax(dim=1) + x.mean(dim=1)) / 255.0
        return SimpleNamespace(image_scores=scores.numpy().astype(np.float32), maps=None)

    module = SimpleNamespace(IMG_SIZE=SIZE, build_model=build_model, predict=predict)
    if with_loader:
        module.load_model = load_model
    return module


def _save_dm(path, gain, state_key="gain"):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": "tiny", "steps": 3, "state": {state_key: torch.tensor(float(gain))}}, path)


def _make_dm_source(src, protocol, module, gain, *, model_path=None, batch_size=5, amp=False):
    """A finished Dinomaly evaluation in the common stage 2 format, scored by the stand-in."""
    manifest = _manifest()
    cache = FakeCache(manifest)
    model = TinyDm()
    with torch.no_grad():
        model.gain.fill_(gain)
    allow = protocol == "test"
    rows = select(manifest, protocol=protocol, part="eval_normal", category="toy", allow_test=allow)
    rows += select(manifest, protocol=protocol, part="eval_defect", category="toy", allow_test=allow)
    cal_rows = select(manifest, protocol="dev", part="eval_normal", category="toy") if allow else []
    cal = np.empty(0, dtype=np.float32)
    if cal_rows:
        cal = module.predict(model, cache.images(cal_rows), batch_size=batch_size).image_scores
    src.mkdir(parents=True)
    np.savez(
        src / "toy.npz",
        eval_images=np.array([r.image for r in rows]),
        eval_labels=np.array([int(r.label == "anomaly") for r in rows], dtype=np.int8),
        eval_score=module.predict(model, cache.images(rows), batch_size=batch_size).image_scores,
        cal_score=cal,
    )
    config = {"name": "dm", "batch_size": batch_size, "amp": amp}
    if model_path is not None:
        config["model_sha256"] = sha256_file(model_path)
    meta = {
        "method": "dinomaly",
        "protocol": protocol,
        "commit": "src1234",
        "device": "cpu",
        "config": config,
        "categories": [{"category": "toy"}],
    }
    (src / "run.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def dm_project(toy_project, monkeypatch):
    calls = []
    module = _dm_module(calls)
    monkeypatch.setattr(run_perturb, "_dinomaly", lambda: module)
    toy_project.calls = calls
    toy_project.module = module
    toy_project.model = toy_project.outputs / "dm" / "model.pt"
    _save_dm(toy_project.model, 2.0)
    return toy_project


DM_DEV = ["--method", "dm", "--protocol", "dev", "--device", "cpu", "--categories", "toy"]
DM_TEST = ["--method", "dm", "--protocol", "test", "--device", "cpu", "--categories", "toy"]


def test_cli_dinomaly_dev_has_no_calibration_scores(dm_project):
    outputs = dm_project.outputs
    _make_dm_source(outputs / "dm-dev", "dev", dm_project.module, 2.0, model_path=dm_project.model)
    dm_project.calls.clear()
    run_perturb.main(DM_DEV)

    # The default model file, then one prediction per condition with the source run's batch size and
    # precision (so that the clean scores are comparable).
    assert dm_project.calls[0] == ("load", str(dm_project.model))
    assert dm_project.calls[1:] == [("predict", 14, 5, "cpu", False)] * 16
    out = outputs / "perturb-dm-dev"
    with np.load(out / "toy.npz") as z, np.load(outputs / "dm-dev" / "toy.npz") as source:
        assert set(z.files) == NPZ_KEYS
        assert z["conditions"].tolist() == condition_names()
        assert np.array_equal(z["scores"][0], source["eval_score"])
        assert z["cal_score"].shape == (0,) and z["cal_score"].dtype == np.float32
        assert z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert not np.array_equal(z["scores"][condition_names().index("gamma-3")], z["scores"][0])
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["method"] == "dm" and run["source"] == str(outputs / "dm-dev")
    assert run["config"] == {
        "name": "dm",
        "img_size": SIZE,
        "model": str(dm_project.model),
        "model_sha256": sha256_file(dm_project.model),
        "batch_size": 5,
        "amp": False,
    }
    assert not dm_project.ledger.exists()
    # Without calibration scores the table has accuracy only.
    table = analyze_perturb.condition_table(analyze_perturb.load(out), n_boot=20)
    assert table[1]["fpr"] is None and table[1]["d_fpr"] is None and table[1]["d_auroc"] is not None


def test_cli_dinomaly_test_passes_the_holdout_scores_on(dm_project):
    outputs = dm_project.outputs
    model = outputs / "elsewhere" / "model.pt"
    _save_dm(model, 3.0)
    _make_dm_source(outputs / "dm-test", "test", dm_project.module, 3.0, model_path=model)
    dm_project.calls.clear()
    run_perturb.main(DM_TEST + ["--allow-test", "--stage", "3", "--model", str(model)])
    assert dm_project.calls[0] == ("load", str(model))
    entry = json.loads(dm_project.ledger.read_text(encoding="utf-8").splitlines()[0])
    assert (entry["stage"], entry["config"]) == ("3", "perturb-dm")
    assert dm_project.cache.requested == TEST_IMAGES  # the hold-out normals are not scored again
    with (
        np.load(outputs / "perturb-dm-test" / "toy.npz") as z,
        np.load(outputs / "dm-test" / "toy.npz") as source,
    ):
        assert z["cal_score"].shape == (8,) and np.array_equal(z["cal_score"], source["cal_score"])
        assert np.array_equal(z["scores"][0], source["eval_score"])
        assert z["eval_images"].tolist() == TEST_IMAGES


def test_cli_dinomaly_refuses_another_model(dm_project, capsys):
    outputs = dm_project.outputs
    allow = ["--allow-test", "--stage", "3"]
    # The source run recorded the hash of its model: another file is refused before anything is read.
    other = outputs / "other" / "model.pt"
    _save_dm(other, 3.0)
    _make_dm_source(outputs / "dm-test", "test", dm_project.module, 3.0, model_path=other)
    with pytest.raises(SystemExit):
        run_perturb.main(DM_TEST + allow)
    assert "sha256 differs" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_perturb.main(DM_TEST + allow + ["--model", str(outputs / "nowhere.pt")])
    assert "does not exist" in capsys.readouterr().err
    assert not dm_project.ledger.exists() and dm_project.cache.requested == []

    # Without a recorded hash the clean scores still give the wrong model away.
    _make_dm_source(outputs / "dm-dev", "dev", dm_project.module, 3.0)
    with pytest.raises(run_perturb.CleanScoreMismatch):
        run_perturb.main(DM_DEV)
    assert not (outputs / "perturb-dm-dev" / "run.json").exists()


def test_load_dinomaly_without_a_loader_follows_the_model_file_layout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_perturb, "_dinomaly", lambda: _dm_module(calls, with_loader=False))
    _save_dm(tmp_path / "model.pt", 4.0)
    model = run_perturb.load_dinomaly(tmp_path / "model.pt")
    assert calls == [("build", "tiny")] and float(model.gain.detach()) == 4.0 and not model.training

    _save_dm(tmp_path / "odd.pt", 4.0, state_key="bias")
    with pytest.raises(ValueError, match="trainable parameters"):
        run_perturb.load_dinomaly(tmp_path / "odd.pt")

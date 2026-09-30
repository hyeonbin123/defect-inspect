import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from defect_inspect import backbones, paths, run_supervised, supervised  # noqa: E402
from defect_inspect.metrics import aupro, auroc, pixel_auroc  # noqa: E402
from defect_inspect.splits import (  # noqa: E402
    ManifestRow,
    SealedTestError,
    label_subset,
    write_manifest,
)
from defect_inspect.supervised import TrainConfig  # noqa: E402

SIZE = 32
CELLS = 4  # the fake extractor returns a 4x4 grid
CELL = SIZE // CELLS
SEALED = {"test_normal", "test_defect"}
COMMON_KEYS = {
    "eval_images",
    "eval_labels",
    "eval_defect_types",
    "eval_score",
    "cal_score",
    "pixel_auroc",
    "aupro",
    "pro_edges",
    "pro_normal",
    "pro_components",
    "pro_component_image",
}
SUPERVISED_KEYS = {"train_defect_images", "train_defect_types", "best_epoch", "best_val_auroc"}
# Two epochs are enough wherever the test is about bookkeeping, not about the head.
QUICK = TrainConfig(epochs=2, eval_every=1)


class GridMean(torch.nn.Module):
    """Tiny stand-in for a backbone: the mean colour of each cell of a 4x4 grid."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), CELL).permute(0, 2, 3, 1)


class FakeCache:
    """Random grey images; every defect image has one bright cell, marked in its mask."""

    size = SIZE

    def __init__(self, manifest, events):
        rng = np.random.default_rng(0)
        self.events = events
        self._role = {r.image: r.role for r in manifest}
        self._images, self._masks = {}, {}
        for i, row in enumerate(manifest):
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                r, c = i % CELLS, (i // CELLS) % CELLS
                img[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL] = 255
                mask[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask

    def images(self, rows):
        self.events.extend(("images", r.image, self._role[r.image]) for r in rows)
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        self.events.extend(("masks", r.image, self._role[r.image]) for r in rows)
        return np.stack([self._masks[r.image] for r in rows])


def _manifest():
    rows = []
    for i in range(40):
        rows.append(ManifestRow(f"toy/n{i}.JPG", "", "toy", "normal", "pool_normal", i % 5, ""))
    for i in range(6):
        rows.append(ManifestRow(f"toy/d{i}.JPG", f"toy/d{i}.png", "toy", "anomaly", "dev_defect", -1, "hole"))
    for i in range(12):
        types = ("hole", "scratch", "hole|scratch")[i % 3]
        rows.append(ManifestRow(f"toy/p{i}.JPG", f"toy/p{i}.png", "toy", "anomaly", "label_pool", -1, types))
    for i in range(10):
        rows.append(ManifestRow(f"toy/tn{i}.JPG", "", "toy", "normal", "test_normal", -1, ""))
    # 15 test images against 14 validation images: scoring the wrong set shows in the shapes.
    for i in range(5):
        mask = f"toy/td{i}.png"
        rows.append(ManifestRow(f"toy/td{i}.JPG", mask, "toy", "anomaly", "test_defect", -1, "hole"))
    # A second category that a run of "toy" must never touch.
    for i in range(10):
        rows.append(ManifestRow(f"other/n{i}.JPG", "", "other", "normal", "pool_normal", i % 5, ""))
    for i in range(3):
        mask = f"other/p{i}.png"
        rows.append(ManifestRow(f"other/p{i}.JPG", mask, "other", "anomaly", "label_pool", -1, "dent"))
    return rows


def _images(manifest, role, fold=None):
    return [
        r.image
        for r in manifest
        if r.category == "toy" and r.role == role and (fold is None or r.fold in fold)
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Toy manifest, fake cache and extractor, and every project path redirected into tmp_path."""
    manifest = _manifest()
    write_manifest(manifest, tmp_path / "visa.csv")
    events, calls = [], {}
    ledger = tmp_path / "reports" / "test_ledger.jsonl"
    monkeypatch.setattr(paths, "VISA_MANIFEST", tmp_path / "visa.csv")
    monkeypatch.setattr(paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")

    def fake_cache(out_dir, size):
        calls["cache"] = (out_dir, size)
        return FakeCache(manifest, events)

    def fake_extractor(name, *, img_size, pretrained=True):
        calls["extractor"] = (name, img_size, pretrained)
        return GridMean()

    real_record = run_supervised.record_test_access

    def recording_ledger(path, **kwargs):
        events.append(("ledger", str(path), ""))
        return real_record(path, **kwargs)

    monkeypatch.setattr(run_supervised, "ImageCache", fake_cache)
    monkeypatch.setattr(backbones, "make_extractor", fake_extractor)
    monkeypatch.setattr(run_supervised, "record_test_access", recording_ledger)
    monkeypatch.setattr(run_supervised, "git_commit", lambda: "test-commit")
    return SimpleNamespace(
        manifest=manifest, events=events, calls=calls, ledger=ledger, out=tmp_path / "out", tmp=tmp_path
    )


def _run(env, protocol, allow_test=False, ks=(2,), seeds=(0,), train_cfg=QUICK):
    env.out.mkdir(exist_ok=True)
    return run_supervised.run_category(
        protocol,
        "toy",
        env.manifest,
        FakeCache(env.manifest, env.events),
        GridMean(),
        "cpu",
        allow_test,
        env.out,
        ks=ks,
        seeds=seeds,
        train_cfg=train_cfg,
    )


def _spy_on_training(monkeypatch):
    """Record the arguments and the result of every `train_head` call."""
    calls = []
    real = supervised.train_head

    def spy(train_feats, train_targets, val_feats, val_labels, cfg, *, device):
        result = real(train_feats, train_targets, val_feats, val_labels, cfg, device=device)
        calls.append(
            SimpleNamespace(
                train_feats=train_feats.clone(),
                train_targets=train_targets.copy(),
                val_feats=val_feats.clone(),
                val_labels=val_labels.copy(),
                cfg=cfg,
                device=device,
                result=result,
            )
        )
        return result

    monkeypatch.setattr(supervised, "train_head", spy)
    return calls


def _quick_training(monkeypatch):
    """Two epochs instead of 60, for CLI tests that are about the files and not about the head."""
    real = supervised.train_head

    def quick(train_feats, train_targets, val_feats, val_labels, cfg, *, device):
        cfg = replace(cfg, epochs=2, eval_every=1)
        return real(train_feats, train_targets, val_feats, val_labels, cfg, device=device)

    monkeypatch.setattr(supervised, "train_head", quick)


def _rows(manifest, images):
    by_image = {r.image: r for r in manifest}
    return [by_image[image] for image in images]


def _features(manifest, images):
    """Features of the named images, extracted the way the runner extracts them."""
    cache = FakeCache(manifest, [])
    return supervised.extract(GridMean(), cache.images(_rows(manifest, images)), device="cpu")


def _masks(manifest, images):
    return FakeCache(manifest, []).masks(_rows(manifest, images)).astype(bool)


def _files(out_dir):
    return sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())


def test_dev_cli_writes_the_common_format(env, monkeypatch):
    calls = _spy_on_training(monkeypatch)
    args = ["--protocol", "dev", "--categories", "toy", "--ks", "2", "4", "--seeds", "0", "1"]
    run_supervised.main([*args, "--device", "cpu"])
    out = env.tmp / "outputs" / "sup-dev"  # the default output directory
    assert env.calls["extractor"][:2] == ("dinov2_vits14", 448)
    assert env.calls["cache"] == (env.tmp / "cache", 448)
    by_run = {(len(call.train_feats) - 32, call.cfg.seed): call for call in calls}
    assert len(calls) == 4 and set(by_run) == {(2, 0), (2, 1), (4, 0), (4, 1)}

    fold0 = _images(env.manifest, "pool_normal", fold=(0,))
    dev_defects = _images(env.manifest, "dev_defect")
    eval_masks = _masks(env.manifest, fold0 + dev_defects)
    pool = set(_images(env.manifest, "label_pool"))
    subsets = {}
    for k in (2, 4):
        for seed in (0, 1):
            run_dir = out / f"k{k}-s{seed}"
            # What the selected head of this run gives for the evaluation set, image by image.
            call = by_run[(k, seed)]
            scored = supervised.predict(call.result.head, call.val_feats, device="cpu")
            maps = np.load(run_dir / "toy_maps.npy")
            assert maps.shape == (14, 256, 256) and maps.dtype == np.float16
            assert np.isfinite(maps.astype(np.float32)).all()
            # The maps file holds the maps of the evaluation images, in the order of `eval_images`.
            np.testing.assert_array_equal(maps, scored.maps)
            with np.load(run_dir / "toy.npz") as z:
                np.testing.assert_array_equal(z["eval_score"], scored.image_scores)
                # The pixel metrics in the npz are those of the stored maps against the masks.
                stored = maps.astype(np.float32)
                assert float(z["pixel_auroc"]) == pytest.approx(pixel_auroc(stored, eval_masks), abs=1e-12)
                assert float(z["aupro"]) == pytest.approx(aupro(stored, eval_masks), abs=1e-12)
                assert COMMON_KEYS | SUPERVISED_KEYS <= set(z.files)
                # Evaluation set of the dev protocol: fold-0 normals, then the dev defects.
                assert z["eval_images"].tolist() == fold0 + dev_defects
                assert z["eval_labels"].dtype == np.int8
                assert z["eval_labels"].tolist() == [0] * 8 + [1] * 6
                assert z["eval_defect_types"].tolist() == [""] * 8 + ["hole"] * 6
                scores = z["eval_score"]
                assert scores.shape == (14,) and scores.dtype == np.float32
                assert np.isfinite(scores).all()
                # In dev the fold-0 normals are the evaluation set itself: no calibration scores.
                assert z["cal_score"].shape == (0,) and z["cal_score"].dtype == np.float32
                # The bright cell is easy: the head finds every defect, and because the dev evaluation
                # set is the validation set, the stored scores reproduce the selected epoch's AUROC.
                assert float(z["best_val_auroc"]) >= 0.95
                assert auroc(scores[:8], scores[8:]) == pytest.approx(float(z["best_val_auroc"]))
                assert int(z["best_epoch"]) in range(5, 61, 5)
                for key in ("pixel_auroc", "aupro"):
                    assert z[key].shape == () and z[key].dtype == np.float64
                    assert 0.0 <= float(z[key]) <= 1.0
                assert float(z["pixel_auroc"]) > 0.9
                assert z["pro_edges"].shape == (2001,)
                assert z["pro_normal"].shape == (14, 2000)
                assert z["pro_components"].shape == (6, 2000)
                assert z["pro_component_image"].tolist() == list(range(8, 14))
                assert z["history_loss"].shape == (60,) and z["history_val_auroc"].shape == (60,)
                assert np.flatnonzero(~np.isnan(z["history_val_auroc"])).tolist() == list(range(4, 60, 5))
                # Label accounting: exactly k training defects, all from the label pool.
                train_defects = z["train_defect_images"].tolist()
                assert len(train_defects) == k and len(set(train_defects)) == k
                assert set(train_defects) <= pool
                expected = label_subset(env.manifest, "toy", k, seed)
                assert train_defects == [r.image for r in expected]
                assert z["train_defect_types"].tolist() == [r.defect_types for r in expected]
                subsets[(k, seed)] = train_defects
            sub = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            assert sub["method"] == "supervised" and sub["protocol"] == "dev"
            assert sub["torch_threads"] == torch.get_num_threads()
            assert (sub["config"]["k"], sub["config"]["seed"]) == (k, seed)
            assert [c["category"] for c in sub["categories"]] == ["toy"]
    # Nested across k for one seed; the seeds order the pool differently.
    for seed in (0, 1):
        assert subsets[(4, seed)][:2] == subsets[(2, seed)]
    assert subsets[(4, 0)] != subsets[(4, 1)]

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["method"] == "supervised" and run["protocol"] == "dev"
    assert run["commit"] == "test-commit" and run["device"] == "cpu"
    # CPU results are only bit-identical for the same number of threads, so the run records it.
    assert run["torch_threads"] == torch.get_num_threads() and type(run["torch_threads"]) is int
    assert run["config"] == {
        "backbone": "dinov2_vits14",
        "img_size": 448,
        "ks": [2, 4],
        "seeds": [0, 1],
        "train": asdict(TrainConfig()),
    }
    assert run["config"]["train"]["epochs"] == 60
    (category,) = run["categories"]
    assert category["category"] == "toy"
    assert (category["train_normal"], category["val_normal"], category["val_defect"]) == (32, 8, 6)
    assert (category["eval_normal"], category["eval_defect"], category["label_pool"]) == (8, 6, 12)
    assert category["grid"] == [4, 4] and category["dim"] == 3
    assert [(r["category"], r["k"], r["seed"]) for r in run["runs"]] == [
        ("toy", 2, 0),
        ("toy", 2, 1),
        ("toy", 4, 0),
        ("toy", 4, 1),
    ]
    for info in run["runs"]:
        assert {"best_epoch", "best_val_auroc", "n_train_normal", "seconds"} <= set(info)
        assert info["n_train_normal"] == 32 and info["n_train_defect"] == info["k"]
        with np.load(out / f"k{info['k']}-s{info['seed']}" / "toy.npz") as z:
            assert info["best_epoch"] == int(z["best_epoch"])
            assert info["best_val_auroc"] == float(z["best_val_auroc"])
    assert run["total_s"] >= 0

    # The dev protocol never touches the sealed test set, the other category or the ledger.
    assert {role for _, _, role in env.events} == {"pool_normal", "dev_defect", "label_pool"}
    assert all(image.startswith("toy/") for _, image, _ in env.events)
    assert not env.ledger.exists()


def test_training_set_is_the_pool_normals_plus_k_label_pool_defects(env, monkeypatch):
    calls = _spy_on_training(monkeypatch)
    base = TrainConfig(epochs=2, eval_every=1, lr=5e-4)
    info, runs = _run(env, "dev", ks=(1, 3), seeds=(0, 2), train_cfg=base)
    assert [(r["k"], r["seed"]) for r in runs] == [(1, 0), (1, 2), (3, 0), (3, 2)]
    cache, extractor = FakeCache(env.manifest, []), GridMean()
    by_image = {r.image: r for r in env.manifest}
    normals = [by_image[i] for i in _images(env.manifest, "pool_normal", fold=(1, 2, 3, 4))]
    validation = [by_image[i] for i in _images(env.manifest, "pool_normal", fold=(0,))]
    validation += [by_image[i] for i in _images(env.manifest, "dev_defect")]
    normal_feats = supervised.extract(extractor, cache.images(normals), device="cpu")
    val_feats = supervised.extract(extractor, cache.images(validation), device="cpu")
    for call, run in zip(calls, runs, strict=True):
        k, seed = run["k"], run["seed"]
        # The run's seed replaces the seed of the base configuration and nothing else.
        assert call.cfg == TrainConfig(epochs=2, eval_every=1, lr=5e-4, seed=seed)
        assert call.device == "cpu"
        assert call.train_feats.shape == (32 + k, 4, 4, 3) and call.train_feats.dtype == torch.float16
        assert torch.equal(call.train_feats[:32], normal_feats)
        defects = label_subset(env.manifest, "toy", k, seed)
        defect_feats = supervised.extract(extractor, cache.images(defects), device="cpu")
        assert torch.equal(call.train_feats[32:], defect_feats)
        # Targets: nothing on the normals, the one bright cell of each defect image.
        assert call.train_targets.shape == (32 + k, 4, 4) and call.train_targets.dtype == np.bool_
        assert not call.train_targets[:32].any()
        expected = supervised.patch_targets(cache.masks(defects), CELLS)
        np.testing.assert_array_equal(call.train_targets[32:], expected)
        assert expected.reshape(k, -1).sum(axis=1).tolist() == [1] * k
        bright = defect_feats[..., 0].float().reshape(k, -1).argmax(dim=1).numpy()
        assert expected.reshape(k, -1).argmax(axis=1).tolist() == bright.tolist()
        # Validation: fold-0 normals and the dev defects.
        assert torch.equal(call.val_feats, val_feats)
        assert call.val_labels.tolist() == [0] * 8 + [1] * 6
        assert run["n_train_normal"] == 32 and run["n_train_defect"] == k
        assert run["pos_weight"] == pytest.approx(min(((32 + k) * 16 - k) / k, 100.0))
        assert run["best_epoch"] == call.result.best_epoch
    assert info["train_normal"] == 32


def test_test_protocol_needs_permission(env, monkeypatch):
    with pytest.raises(SealedTestError):
        _run(env, "test")
    assert env.events == []  # refused before anything was read
    assert not list(env.out.rglob("*.npz"))

    calls = _spy_on_training(monkeypatch)
    info, runs = _run(env, "test", allow_test=True, ks=(2,), seeds=(1,))
    # Training and validation are the same as in dev: fold 0 stays out of training.
    assert (info["train_normal"], info["val_normal"], info["val_defect"]) == (32, 8, 6)
    assert (info["eval_normal"], info["eval_defect"]) == (10, 5)
    assert runs[0]["n_train_normal"] == 32
    (call,) = calls
    assert call.train_feats.shape[0] == 34 and call.val_feats.shape[0] == 14

    sealed = _images(env.manifest, "test_normal") + _images(env.manifest, "test_defect")
    # What the selected head gives for the features of the test images (not of the validation set).
    scored = supervised.predict(call.result.head, _features(env.manifest, sealed), device="cpu")
    maps = np.load(env.out / "k2-s1" / "toy_maps.npy")
    assert maps.shape == (15, 256, 256)
    np.testing.assert_array_equal(maps, scored.maps)
    with np.load(env.out / "k2-s1" / "toy.npz") as z:
        assert COMMON_KEYS | SUPERVISED_KEYS <= set(z.files)
        assert z["eval_images"].tolist() == sealed
        assert z["eval_labels"].tolist() == [0] * 10 + [1] * 5
        assert z["eval_score"].shape == (15,)
        np.testing.assert_array_equal(z["eval_score"], scored.image_scores)
        assert z["pro_components"].shape[0] == 5
        # Pixel metrics: the stored maps against the masks of the test images.
        stored, masks = maps.astype(np.float32), _masks(env.manifest, sealed)
        assert float(z["pixel_auroc"]) == pytest.approx(pixel_auroc(stored, masks), abs=1e-12)
        assert float(z["aupro"]) == pytest.approx(aupro(stored, masks), abs=1e-12)
        # Calibration scores: the fold-0 normals under the selected head.
        cal = z["cal_score"]
        assert cal.shape == (8,) and cal.dtype == np.float32
        expected = supervised.predict(call.result.head, call.val_feats[:8], device="cpu").image_scores
        np.testing.assert_allclose(cal, expected, rtol=1e-5, atol=1e-6)
        train_defects = z["train_defect_images"].tolist()
    assert train_defects == [r.image for r in label_subset(env.manifest, "toy", 2, 1)]
    # Sealed images are only ever read as the evaluation set, never for training or validation.
    sealed_reads = [image for kind, image, role in env.events if role in SEALED and kind == "images"]
    assert sorted(sealed_reads) == sorted(sealed)


def test_cli_refuses_the_test_protocol_without_flags(env):
    base = ["--protocol", "test", "--categories", "toy", "--ks", "2", "--seeds", "0", "--device", "cpu"]
    with pytest.raises(SystemExit):
        run_supervised.main([*base, "--out", str(env.out)])
    with pytest.raises(SystemExit):
        run_supervised.main([*base, "--out", str(env.out), "--allow-test"])
    with pytest.raises(SystemExit):
        run_supervised.main([*base, "--out", str(env.out), "--stage", "2"])
    assert env.events == [] and not env.ledger.exists() and not env.out.exists()


def test_cli_test_protocol_records_the_ledger_before_reading(env):
    run_supervised.main(
        ["--protocol", "test", "--categories", "toy", "--ks", "2", "--seeds", "0", "--device", "cpu"]
        + ["--out", str(env.out), "--allow-test", "--stage", "2"]
    )
    lines = env.ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["stage"] == "2" and entry["config"] == "supervised"
    assert entry["commit"] == "test-commit"
    assert "ks=[2]" in entry["note"] and "seeds=[0]" in entry["note"]
    # One ledger line, written to the ledger path, before the first sealed image is read.
    kinds = [kind for kind, _, _ in env.events]
    assert kinds.count("ledger") == 1
    assert env.events[kinds.index("ledger")][1] == str(env.ledger)
    first_sealed = next(i for i, (_, _, role) in enumerate(env.events) if role in SEALED)
    assert kinds.index("ledger") < first_sealed

    run = json.loads((env.out / "run.json").read_text(encoding="utf-8"))
    assert run["protocol"] == "test" and run["commit"] == "test-commit"
    assert [(r["k"], r["seed"], r["n_train_normal"]) for r in run["runs"]] == [(2, 0, 32)]
    with np.load(env.out / "k2-s0" / "toy.npz") as z:
        assert z["cal_score"].shape == (8,) and z["eval_labels"].tolist() == [0] * 10 + [1] * 5


def test_previous_results_are_the_files_this_module_writes(tmp_path):
    out = tmp_path / "out"
    assert run_supervised.previous_results(out) == []  # no directory yet
    assert run_supervised.clear_previous_results(out) == 0
    names = ["run.json", "toy.npz", "toy_maps.npy", "notes.txt", "toy_bank.npy"]
    for folder in ("k2-s0", "k10-s12", "k2-s0-old", "xk2-s0", "plots"):
        (out / folder).mkdir(parents=True)
        for name in names:
            (out / folder / name).write_bytes(b"x")
    (out / "k10-s12" / "nested").mkdir()
    (out / "k10-s12" / "nested" / "toy.npz").write_bytes(b"x")
    (out / "k3-s1").mkdir()
    for name in names[:3]:
        (out / "k3-s1" / name).write_bytes(b"x")
        (out / name).write_bytes(b"x")
    before = _files(out)

    found = run_supervised.previous_results(out)
    expected = [f"{folder}/{name}" for folder in ("k10-s12", "k2-s0", "k3-s1") for name in names[:3]]
    assert [p.relative_to(out).as_posix() for p in found] == expected
    assert _files(out) == before  # looking removes nothing

    assert run_supervised.clear_previous_results(out) == 9
    # Only run.json, *.npz and *_maps.npy directly inside a k{k}-s{seed} folder are gone, plus the
    # top-level run.json; a folder that is empty afterwards is removed as well.
    assert _files(out) == sorted(set(before) - set(expected) - {"run.json"})
    assert not (out / "k3-s1").exists() and (out / "k2-s0").is_dir()
    assert run_supervised.previous_results(out) == []


def test_an_interrupted_redo_leaves_nothing_of_the_previous_run(env, monkeypatch):
    _quick_training(monkeypatch)
    args = ["--categories", "toy", "--ks", "2", "4", "--seeds", "0", "--device", "cpu"]
    args += ["--out", str(env.out)]
    run_supervised.main(args)
    run_files = ["run.json", "toy.npz", "toy_maps.npy"]
    assert _files(env.out) == [f"k{k}-s0/{name}" for k in (2, 4) for name in run_files] + ["run.json"]
    sub = json.loads((env.out / "k4-s0" / "run.json").read_text(encoding="utf-8"))
    assert sub["commit"] == "test-commit"

    # A redo under another commit, interrupted while its second run trains.
    monkeypatch.setattr(run_supervised, "git_commit", lambda: "later-commit")
    quick = supervised.train_head
    started = []

    def interrupted(*args, **kwargs):
        started.append(1)
        if len(started) == 2:
            raise KeyboardInterrupt
        return quick(*args, **kwargs)

    monkeypatch.setattr(supervised, "train_head", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_supervised.main([*args, "--overwrite"])
    # No run.json on either level, and no result of the first invocation: the analysis reads
    # k{k}-s{seed}/<category>.npz directly, so an old file there would pass for a result of the redo.
    left = ["k2-s0/toy.npz", "k2-s0/toy_maps.npy"]
    assert _files(env.out) == left

    # The leftovers of the unfinished redo are not removed without --overwrite either.
    n_events = len(env.events)
    with pytest.raises(SystemExit):
        run_supervised.main(args)
    assert _files(env.out) == left and len(env.events) == n_events


def test_a_redo_with_other_ks_removes_the_old_run_folders(env, monkeypatch):
    _quick_training(monkeypatch)
    base = ["--categories", "toy", "--seeds", "0", "--device", "cpu", "--out", str(env.out)]
    run_supervised.main([*base, "--ks", "2", "4"])
    # Files this module did not write stay where they are.
    (env.out / "notes.txt").write_text("keep", encoding="utf-8")
    (env.out / "k2-s0" / "notes.txt").write_text("keep", encoding="utf-8")
    before = _files(env.out)

    # Arguments are checked before anything is removed.
    with pytest.raises(SystemExit):  # the label pool of "toy" has 12 images
        run_supervised.main([*base, "--ks", "13", "--overwrite"])
    with pytest.raises(SystemExit):
        run_supervised.main([*base, "--ks", "6", "6", "--overwrite"])
    assert _files(env.out) == before

    run_supervised.main([*base, "--ks", "6", "--overwrite"])
    run_files = ["run.json", "toy.npz", "toy_maps.npy"]
    assert _files(env.out) == ["k2-s0/notes.txt", *(f"k6-s0/{n}" for n in run_files), "notes.txt", "run.json"]
    run = json.loads((env.out / "run.json").read_text(encoding="utf-8"))
    assert run["config"]["ks"] == [6] and [(r["k"], r["seed"]) for r in run["runs"]] == [(6, 0)]


def test_cli_refuses_to_overwrite_a_finished_run(env):
    args = ["--categories", "toy", "--ks", "2", "--seeds", "0", "--device", "cpu", "--out", str(env.out)]
    run_supervised.main(args)
    first = (env.out / "run.json").read_text(encoding="utf-8")
    with np.load(env.out / "k2-s0" / "toy.npz") as z:
        scores = z["eval_score"].copy()
    n_events = len(env.events)
    with pytest.raises(SystemExit):
        run_supervised.main(args)
    assert len(env.events) == n_events  # refused before anything was read
    assert (env.out / "run.json").read_text(encoding="utf-8") == first

    run_supervised.main([*args, "--overwrite"])
    assert json.loads((env.out / "run.json").read_text(encoding="utf-8"))["method"] == "supervised"
    # Same inputs and seed on CPU: the redo reproduces the scores exactly.
    with np.load(env.out / "k2-s0" / "toy.npz") as z:
        np.testing.assert_array_equal(z["eval_score"], scores)


def test_cli_rejects_bad_arguments(env):
    base = ["--device", "cpu", "--out", str(env.out)]
    with pytest.raises(SystemExit):  # the label pool of "toy" has 12 images
        run_supervised.main([*base, "--categories", "toy", "--ks", "13", "--seeds", "0"])
    with pytest.raises(SystemExit):
        run_supervised.main([*base, "--categories", "toy", "--ks", "2", "2", "--seeds", "0"])
    with pytest.raises(SystemExit):
        run_supervised.main([*base, "--categories", "toy", "--ks", "2", "--seeds", "0", "0"])
    with pytest.raises(SystemExit):
        run_supervised.main([*base, "--categories", "nothing", "--ks", "2", "--seeds", "0"])
    with pytest.raises(SystemExit):  # "other" only has 3 images in its label pool
        run_supervised.main([*base, "--categories", "toy", "other", "--ks", "4", "--seeds", "0"])
    assert env.events == [] and not env.out.exists()

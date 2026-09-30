import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defect_inspect import analyze_grid, compare, paths, run_grid, run_patchcore  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.patchcore import build_bank, collect_features, score_images  # noqa: E402
from defect_inspect.splits import ManifestRow, SealedTestError  # noqa: E402

SIZE = 32
RATIOS = [0.5, 0.25, 0.1]


class GridMean(torch.nn.Module):
    """Tiny stand-in for a backbone: the mean colour of each cell of a 4x4 grid."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 4).permute(0, 2, 3, 1)


class FakeCache:
    size = SIZE

    def __init__(self, manifest, log=None):
        rng = np.random.default_rng(0)
        self._images, self._masks = {}, {}
        self._role = {row.image: row.role for row in manifest}
        self.log = [] if log is None else log  # ("images" | "masks", roles read), in call order
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask

    def images(self, rows):
        self.log.append(("images", sorted({self._role[r.image] for r in rows})))
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        self.log.append(("masks", sorted({self._role[r.image] for r in rows})))
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
            rows.append(
                ManifestRow(f"{c}/td{i}.JPG", f"{c}/td{i}.png", c, "anomaly", "test_defect", -1, "hole")
            )
    return rows


def _cfg(ratio):
    return PatchCoreConfig(
        name="toy",
        backbone="gridmean",
        img_size=SIZE,
        coreset_ratio=ratio,
        sigma=1.0,
        batch_size=8,
        reweight_k=3,
    )


def _run(tmp_path, protocol="dev", allow_test=False, save_banks=False):
    manifest = _manifest()
    return run_grid.run_category(
        _cfg(max(RATIOS)),
        RATIOS,
        protocol,
        "toy",
        manifest,
        FakeCache(manifest),
        GridMean(),
        "cpu",
        allow_test,
        tmp_path,
        save_banks=save_banks,
    )


def test_ratio_names_and_rows():
    assert [run_grid.ratio_name(r) for r in (0.1, 0.01, 0.001, 1)] == ["r0.1", "r0.01", "r0.001", "r1.0"]
    assert run_grid.ratio_rows(512, 0.5) == 256
    assert run_grid.ratio_rows(96, 0.3) == 29  # 28.8 rounds to nearest, like build_bank
    assert run_grid.ratio_rows(94, 0.1) == 9
    assert run_grid.ratio_rows(4, 0.001) == 1
    assert run_grid.ratio_rows(7, 1.0) == 7


def test_largest_ratio_equals_run_patchcore(tmp_path):
    _run(tmp_path / "grid")
    manifest = _manifest()
    (tmp_path / "pc").mkdir()
    run_patchcore.run_category(
        _cfg(0.5), "dev", "toy", manifest, FakeCache(manifest), GridMean(), "cpu", False, tmp_path / "pc"
    )
    with np.load(tmp_path / "grid" / "r0.5" / "toy.npz") as g, np.load(tmp_path / "pc" / "toy.npz") as p:
        np.testing.assert_array_equal(g["eval_images"], p["eval_images"])
        np.testing.assert_array_equal(g["eval_labels"], p["eval_labels"])
        np.testing.assert_allclose(g["eval_score_full"], p["eval_score_full"], rtol=1e-6)
        np.testing.assert_allclose(g["pool_score_oof"], p["pool_score_oof"], rtol=1e-6)
        np.testing.assert_array_equal(g["pool_folds"], p["pool_folds"])
        assert float(g["aupro"]) == pytest.approx(float(p["aupro"]))
        assert float(g["pixel_auroc"]) == pytest.approx(float(p["pixel_auroc"]))


@pytest.mark.parametrize("ratio", RATIOS)
def test_smaller_ratios_equal_a_fresh_selection(tmp_path, ratio):
    """A prefix of the largest bank gives the same scores as building that ratio from scratch."""
    infos = _run(tmp_path)
    manifest = _manifest()
    cache, ext, cfg = FakeCache(manifest), GridMean(), _cfg(ratio)
    pool = [r for r in manifest if r.role == "pool_normal" and r.fold != 0]
    eval_rows = [r for r in manifest if r.role == "pool_normal" and r.fold == 0]
    eval_rows += [r for r in manifest if r.role == "dev_defect"]
    feats, _ = collect_features(ext, cache.images(pool), batch_size=8, device="cpu")
    bank = build_bank(feats, ratio, seed=0, device="cpu")
    assert infos[ratio]["bank_rows"]["full"] == bank.shape[0] == run_grid.ratio_rows(len(pool) * 16, ratio)
    direct = score_images(
        ext,
        bank,
        cache.images(eval_rows),
        batch_size=8,
        device="cpu",
        reweight_k=cfg.reweight_k,
        sigma=cfg.sigma,
    )
    with np.load(tmp_path / run_grid.ratio_name(ratio) / "toy.npz") as z:
        np.testing.assert_allclose(z["eval_score_full"], direct.image_scores, rtol=1e-6)
        assert int(z["bank_rows"]) == bank.shape[0]

        # Cross-fitting at this ratio: fold 2's images against a fresh bank built without fold 2.
        rest = [r for r in pool if r.fold != 2]
        held = [r for r in pool if r.fold == 2]
        feats, _ = collect_features(ext, cache.images(rest), batch_size=8, device="cpu")
        fold_bank = build_bank(feats, ratio, seed=0, device="cpu")
        oof = score_images(
            ext,
            fold_bank,
            cache.images(held),
            batch_size=8,
            device="cpu",
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
        )
        np.testing.assert_allclose(z["pool_score_oof"][z["pool_folds"] == 2], oof.image_scores, rtol=1e-6)


def test_smaller_banks_change_the_scores(tmp_path):
    _run(tmp_path)
    with np.load(tmp_path / "r0.5" / "toy.npz") as a, np.load(tmp_path / "r0.1" / "toy.npz") as b:
        assert not np.allclose(a["eval_score_full"], b["eval_score_full"])
        assert int(a["bank_rows"]) > int(b["bank_rows"])
        # A smaller bank covers the normal patches less well: normal scores do not go down.
        assert b["pool_score_oof"].mean() >= a["pool_score_oof"].mean()
    assert not list(tmp_path.glob("**/*_maps.npy"))


def test_saved_banks_reproduce_every_ratio(tmp_path):
    _run(tmp_path, save_banks=True)
    manifest = _manifest()
    cache, ext, cfg = FakeCache(manifest), GridMean(), _cfg(0.5)
    with open(tmp_path / "banks" / "toy.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["largest_ratio"] == 0.5
    assert meta["n_features"]["full"] == 32 * 16 and meta["n_features"]["minus_fold_3"] == 24 * 16
    full = torch.from_numpy(np.load(tmp_path / "banks" / "toy_full.npy"))
    assert full.dtype == torch.float16
    eval_rows = [r for r in manifest if r.role == "pool_normal" and r.fold == 0]
    eval_rows += [r for r in manifest if r.role == "dev_defect"]
    for ratio in RATIOS:
        rows = run_grid.ratio_rows(meta["n_features"]["full"], ratio)
        res = score_images(
            ext,
            full[:rows],
            cache.images(eval_rows),
            batch_size=8,
            device="cpu",
            reweight_k=cfg.reweight_k,
            sigma=1.0,
        )
        with np.load(tmp_path / run_grid.ratio_name(ratio) / "toy.npz") as z:
            np.testing.assert_allclose(z["eval_score_full"], res.image_scores, rtol=1e-6)
    assert sorted(p.name for p in (tmp_path / "banks").glob("toy_minus_fold_*.npy")) == [
        f"toy_minus_fold_{f}.npy" for f in (1, 2, 3, 4)
    ]

    # The cross-fitted score of a pool image comes from the saved bank that was built without its fold.
    pool = [r for r in manifest if r.role == "pool_normal" and r.fold != 0]
    for fold in (1, 2, 3, 4):
        bank = torch.from_numpy(np.load(tmp_path / "banks" / f"toy_minus_fold_{fold}.npy"))
        n_source = meta["n_features"][f"minus_fold_{fold}"]
        assert bank.dtype == torch.float16 and bank.shape[0] == run_grid.ratio_rows(n_source, 0.5)
        held = cache.images([r for r in pool if r.fold == fold])
        for ratio in RATIOS:
            res = score_images(
                ext,
                bank[: run_grid.ratio_rows(n_source, ratio)],
                held,
                batch_size=8,
                device="cpu",
                reweight_k=cfg.reweight_k,
                sigma=1.0,
            )
            with np.load(tmp_path / run_grid.ratio_name(ratio) / "toy.npz") as z:
                oof = z["pool_score_oof"][z["pool_folds"] == fold]
                np.testing.assert_allclose(oof, res.image_scores, rtol=1e-6)


def test_test_protocol_is_sealed(tmp_path):
    with pytest.raises(SealedTestError):
        _run(tmp_path, protocol="test")
    infos = _run(tmp_path, protocol="test", allow_test=True)
    assert infos[0.5]["pool"] == 40 and infos[0.5]["eval_normal"] == 10 and infos[0.5]["eval_defect"] == 4


def test_largest_ratio_must_be_the_config_ratio(tmp_path):
    manifest = _manifest()
    with pytest.raises(ValueError, match="largest ratio"):
        run_grid.run_category(
            _cfg(0.25),
            RATIOS,
            "dev",
            "toy",
            manifest,
            FakeCache(manifest),
            GridMean(),
            "cpu",
            False,
            tmp_path,
        )


@pytest.fixture
def calls():
    """Ledger writes and cache reads of the CLI runs, in the order they happen."""
    return []


@pytest.fixture
def cli(tmp_path, monkeypatch, calls):
    manifest = _manifest(("toy", "two"))
    # A category with pool normals but no defects to evaluate on.
    manifest += [
        ManifestRow(f"bare/n{i}.JPG", "", "bare", "normal", "pool_normal", i % 5, "") for i in range(10)
    ]
    real_record = run_grid.record_test_access

    def record(*args, **kwargs):
        calls.append(("ledger", kwargs["config"]))
        return real_record(*args, **kwargs)

    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(paths, "TEST_LEDGER", tmp_path / "reports" / "test_ledger.jsonl")
    monkeypatch.setattr(run_grid, "read_manifest", lambda path: manifest)
    monkeypatch.setattr(run_grid, "ImageCache", lambda out_dir, size: FakeCache(manifest, log=calls))
    monkeypatch.setattr(run_grid, "record_test_access", record)
    monkeypatch.setattr("defect_inspect.backbones.make_extractor", lambda name, img_size: GridMean())
    monkeypatch.setattr(compare, "N_BOOT", 100)
    return tmp_path


SETTING = ["--backbone", "wrn50", "--size", "32", "--device", "cpu"]
BASE = [*SETTING, "--categories", "toy", "two", "--ratios", "0.5", "0.1"]


def test_cli_writes_a_run_per_ratio_that_compare_can_read(cli, capsys):
    run_grid.main([*BASE, "--protocol", "dev"])
    out = cli / "outputs" / "grid-wrn50-32-dev"
    with open(out / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["setting"] == "wrn50-32" and meta["ratios"] == [0.5, 0.1] and meta["protocol"] == "dev"
    run = compare.load_patchcore(out / "r0.1")
    assert run.name == "wrn50-32-r0.1" and list(run.cats) == ["toy", "two"]
    assert not (cli / "reports" / "test_ledger.jsonl").exists()

    for ratio in (0.5, 0.1):  # every ratio folder describes its own ratio, not the largest one
        folder = out / run_grid.ratio_name(ratio)
        with open(folder / "run.json", encoding="utf-8") as f:
            ratio_meta = json.load(f)
        config = ratio_meta["config"]
        assert config["name"] == f"wrn50-32-r{ratio}" and config["coreset_ratio"] == ratio
        assert config["backbone"] == "wrn50" and config["img_size"] == 32
        fixed = {"reweight_k": 9, "sigma": 4.0, "batch_size": 32, "seed": 0}
        assert {k: config[k] for k in fixed} == fixed
        assert ratio_meta["protocol"] == "dev" and ratio_meta["device"] == "cpu"
        assert ratio_meta["commit"] == meta["commit"]
        assert [info["category"] for info in ratio_meta["categories"]] == ["toy", "two"]
        for info in ratio_meta["categories"]:
            with np.load(folder / f"{info['category']}.npz") as z:
                assert info["bank_rows"]["full"] == int(z["bank_rows"]) == run_grid.ratio_rows(32 * 16, ratio)

    with pytest.raises(SystemExit):  # a finished run is not overwritten silently
        run_grid.main([*BASE, "--protocol", "dev"])
    run_grid.main([*BASE, "--protocol", "dev", "--overwrite"])

    capsys.readouterr()
    analyze_grid.main(["--protocol", "dev", "--settings", "wrn50-32"])
    text = capsys.readouterr().out
    assert "wrn50-32" in text and "vs largest" in text
    with open(cli / "reports" / "stage4" / "grid-dev.json", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    assert [r["key"] for r in rows] == ["wrn50-32-r0.5", "wrn50-32-r0.1"]
    assert "vs_largest_ratio" not in rows[0] and "vs_largest_ratio" in rows[1]
    assert rows[0]["bank_rows_mean"] == 32 * 16 * 0.5


def test_cli_test_protocol_needs_flags_and_writes_the_ledger(cli, calls):
    ledger = cli / "reports" / "test_ledger.jsonl"
    for extra in ([], ["--allow-test"], ["--stage", "4"]):
        with pytest.raises(SystemExit):
            run_grid.main([*BASE, "--protocol", "test", *extra])
    assert not ledger.exists() and not calls
    run_grid.main([*BASE, "--protocol", "test", "--allow-test", "--stage", "4"])
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["stage"] == "4" and entry["config"] == "grid-wrn50-32"
    # The ledger line exists before the first image of any kind is read, and it is the only one.
    assert calls[0] == ("ledger", "grid-wrn50-32") and [c for c in calls if c[0] == "ledger"] == calls[:1]
    assert ("images", ["test_defect", "test_normal"]) in calls[1:]

    calls.clear()
    with pytest.raises(SystemExit):  # a finished test run is not read again without --overwrite
        run_grid.main([*BASE, "--protocol", "test", "--allow-test", "--stage", "4"])
    assert not calls and len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_cli_dev_protocol_reads_no_sealed_image(cli, calls):
    run_grid.main([*BASE, "--protocol", "dev"])
    assert {role for _, roles in calls for role in roles} == {"pool_normal", "dev_defect"}
    assert not (cli / "reports" / "test_ledger.jsonl").exists()


@pytest.mark.parametrize(
    ("categories", "named"),
    [
        (["toy", "tyo"], "tyo"),  # a typo
        (["bare"], "eval_defect"),  # pool normals but nothing to evaluate on
        (["toy", "toy"], "twice"),
    ],
)
def test_cli_checks_the_categories_before_the_ledger(cli, calls, capsys, categories, named):
    args = [*SETTING, "--ratios", "0.5", "--categories", *categories]
    for protocol in (["--protocol", "test", "--allow-test", "--stage", "4"], ["--protocol", "dev"]):
        with pytest.raises(SystemExit):
            run_grid.main([*args, *protocol])
        assert named in capsys.readouterr().err
    assert not calls and not (cli / "reports" / "test_ledger.jsonl").exists()
    assert not (cli / "outputs").exists()


def test_cli_clears_old_outputs_before_the_ledger(cli, calls, monkeypatch):
    """Old outputs that cannot be deleted (a file in use) stop the run without a ledger line."""

    def in_use(out_dir):
        raise PermissionError(f"{out_dir} is in use")

    monkeypatch.setattr(run_grid, "clear_outputs", in_use)
    with pytest.raises(PermissionError):
        run_grid.main([*BASE, "--protocol", "test", "--allow-test", "--stage", "4"])
    assert not calls and not (cli / "reports" / "test_ledger.jsonl").exists()


def _files(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def test_cli_overwrite_removes_what_the_earlier_call_wrote(cli):
    out = cli / "outputs" / "grid-wrn50-32-dev"
    first = [*SETTING, "--protocol", "dev", "--categories", "toy", "two", "--ratios", "0.5", "0.25", "0.1"]
    run_grid.main([*first, "--save-banks"])
    assert (out / "r0.25" / "two.npz").exists() and (out / "banks" / "two_minus_fold_1.npy").exists()
    # Files that are not grid outputs survive, also in folders that are cleaned or merely start with "r".
    kept = ["notes.txt", "r0.5/notes.txt", "banks/notes.txt", "reports/toy.npz", "r0.5x/run.json"]
    for name in kept:
        (out / name).parent.mkdir(exist_ok=True)
        (out / name).write_text("keep", encoding="utf-8")

    second = [*SETTING, "--protocol", "dev", "--categories", "toy", "--ratios", "0.5", "0.1", "--overwrite"]
    run_grid.main(second)
    written = ["run.json", "r0.5/run.json", "r0.5/toy.npz", "r0.1/run.json", "r0.1/toy.npz"]
    assert _files(out) == sorted(written + kept)
    assert not (out / "r0.25").exists()  # a stale ratio folder no longer loads as a finished run
    assert list(compare.load_patchcore(out / "r0.5").cats) == ["toy"]


def test_interrupted_overwrite_does_not_leave_a_finished_looking_run(cli, monkeypatch):
    out = cli / "outputs" / "grid-wrn50-32-dev"
    run_grid.main([*BASE, "--protocol", "dev"])
    real, seen = run_grid.run_category, []

    def interrupted(*args, **kwargs):
        seen.append(args[3])
        if len(seen) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(run_grid, "run_category", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_grid.main([*BASE, "--protocol", "dev", "--overwrite"])
    assert seen == ["toy", "two"]
    assert _files(out) == ["r0.1/toy.npz", "r0.5/toy.npz"]  # no run.json at any level
    with pytest.raises(FileNotFoundError):
        analyze_grid.load_grid(out)
    with pytest.raises(FileNotFoundError):
        compare.load_patchcore(out / "r0.5")

    # The next call starts from a clean folder; nothing finished is there, so it needs no --overwrite.
    monkeypatch.setattr(run_grid, "run_category", real)
    run_grid.main([*SETTING, "--protocol", "dev", "--categories", "two", "--ratios", "0.5"])
    assert _files(out) == ["r0.5/run.json", "r0.5/two.npz", "run.json"]


def test_cli_rejects_bad_arguments(cli, capsys):
    with pytest.raises(SystemExit):
        run_grid.main(["--backbone", "dinov2_vits14", "--size", "256", "--protocol", "dev", *BASE[6:]])
    assert "multiples of 14" in capsys.readouterr().err
    for ratio in ("0", "1.5", "-0.1", "nan"):
        with pytest.raises(SystemExit):
            run_grid.main([*SETTING, "--categories", "toy", "--protocol", "dev", "--ratios", ratio])
        assert "ratios must be in (0, 1]" in capsys.readouterr().err
    assert not (cli / "outputs").exists()

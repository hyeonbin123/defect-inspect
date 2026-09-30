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

    def __init__(self, manifest):
        rng = np.random.default_rng(0)
        self._images, self._masks = {}, {}
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask

    def images(self, rows):
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
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
def cli(tmp_path, monkeypatch):
    manifest = _manifest(("toy", "two"))
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(paths, "TEST_LEDGER", tmp_path / "reports" / "test_ledger.jsonl")
    monkeypatch.setattr(run_grid, "read_manifest", lambda path: manifest)
    monkeypatch.setattr(run_grid, "ImageCache", lambda out_dir, size: FakeCache(manifest))
    monkeypatch.setattr("defect_inspect.backbones.make_extractor", lambda name, img_size: GridMean())
    monkeypatch.setattr(compare, "N_BOOT", 100)
    return tmp_path


BASE = ["--backbone", "wrn50", "--size", "32", "--device", "cpu", "--categories", "toy", "two"]
BASE += ["--ratios", "0.5", "0.1"]


def test_cli_writes_a_run_per_ratio_that_compare_can_read(cli, capsys):
    run_grid.main([*BASE, "--protocol", "dev"])
    out = cli / "outputs" / "grid-wrn50-32-dev"
    with open(out / "run.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["setting"] == "wrn50-32" and meta["ratios"] == [0.5, 0.1] and meta["protocol"] == "dev"
    run = compare.load_patchcore(out / "r0.1")
    assert run.name == "wrn50-32-r0.1" and list(run.cats) == ["toy", "two"]
    assert not (cli / "reports" / "test_ledger.jsonl").exists()

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


def test_cli_test_protocol_needs_flags_and_writes_the_ledger(cli):
    for extra in ([], ["--allow-test"], ["--stage", "4"]):
        with pytest.raises(SystemExit):
            run_grid.main([*BASE, "--protocol", "test", *extra])
    assert not (cli / "reports" / "test_ledger.jsonl").exists()
    run_grid.main([*BASE, "--protocol", "test", "--allow-test", "--stage", "4"])
    lines = (cli / "reports" / "test_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["stage"] == "4" and entry["config"] == "grid-wrn50-32"


def test_cli_rejects_bad_arguments(cli):
    with pytest.raises(SystemExit):
        run_grid.main(["--backbone", "dinov2_vits14", "--size", "256", "--protocol", "dev"])
    with pytest.raises(SystemExit):
        run_grid.main([*BASE[:6], "--protocol", "dev", "--ratios", "0"])


def test_latency_is_merged_by_key():
    rows = [{"key": "wrn50-256-r0.01"}, {"key": "wrn50-256-r0.1"}]
    analyze_grid.merge_latency(rows, {"wrn50-256-r0.01": {"cpu_fp32": {"total_ms": 120.0}}})
    assert rows[0]["latency"]["cpu_fp32"]["total_ms"] == 120.0 and "latency" not in rows[1]

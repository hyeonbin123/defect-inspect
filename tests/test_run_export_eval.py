import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from defect_inspect import (  # noqa: E402
    analyze_export,
    bench,
    compare,
    export,
    paths,
    run_export_eval,
    run_grid,
)
from defect_inspect.inspector import Inspector  # noqa: E402
from defect_inspect.splits import ManifestRow  # noqa: E402

SIZE = 64
DIM = 12


class TinyPatches(torch.nn.Module):
    name = "tiny"
    dim = DIM

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, DIM, 3, stride=4, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        return self.conv2(x).permute(0, 2, 3, 1).contiguous()


class FakeCache:
    size = SIZE

    def __init__(self, manifest):
        rng = np.random.default_rng(0)
        self._images, self._masks = {}, {}
        for row in manifest:
            img = rng.integers(90, 150, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[16:40, 16:40] = 255
                mask[64:160, 64:160] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask

    def images(self, rows):
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        return np.stack([self._masks[r.image] for r in rows])


def _manifest(categories=("toy", "two")):
    rows = []
    for c in categories:
        for i in range(30):
            rows.append(ManifestRow(f"{c}/n{i}.JPG", "", c, "normal", "pool_normal", i % 5, ""))
        for i in range(6):
            rows.append(ManifestRow(f"{c}/d{i}.JPG", f"{c}/d{i}.png", c, "anomaly", "dev_defect", -1, "hole"))
        for i in range(9):
            rows.append(ManifestRow(f"{c}/tn{i}.JPG", "", c, "normal", "test_normal", -1, ""))
        for i in range(4):
            rows.append(
                ManifestRow(f"{c}/td{i}.JPG", f"{c}/td{i}.png", c, "anomaly", "test_defect", -1, "hole")
            )
    return rows


@pytest.fixture
def world(tmp_path, monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(paths, "TEST_LEDGER", tmp_path / "reports" / "test_ledger.jsonl")
    for module in (run_grid, run_export_eval):
        monkeypatch.setattr(module, "read_manifest", lambda path: manifest)
        monkeypatch.setattr(module, "ImageCache", lambda out_dir, size: FakeCache(manifest))
    monkeypatch.setattr("defect_inspect.splits.read_manifest", lambda path: manifest)
    monkeypatch.setattr("defect_inspect.cache.ImageCache", lambda out_dir, size: FakeCache(manifest))
    monkeypatch.setattr("defect_inspect.backbones.make_extractor", lambda name, img_size: TinyPatches())
    monkeypatch.setattr(compare, "N_BOOT", 100)
    return tmp_path


def _grid(world, protocol, extra=()):
    args = ["--backbone", "wrn50", "--size", str(SIZE), "--device", "cpu", "--categories", "toy", "two"]
    run_grid.main([*args, "--ratios", "0.5", "0.25", "--protocol", protocol, "--save-banks", *extra])
    return world / "outputs" / f"grid-wrn50-{SIZE}-{protocol}"


def test_artifacts_eval_and_analysis_end_to_end(world, capsys):
    grid_dir = _grid(world, "dev")
    art = world / "artifacts" / "tiny"
    info = export.build_artifacts(grid_dir, 0.25, art, calibration_per_category=6)
    assert (art / "model_fp32.onnx").exists() and (art / "model_int8.onnx").exists()
    assert info["calibration_images"] == 12 and [c["category"] for c in info["categories"]] == ["toy", "two"]
    meta = json.loads((art / "toy" / "meta.json").read_text(encoding="utf-8"))
    assert meta["img_size"] == SIZE and meta["grid"] == [8, 8] and meta["dim"] == DIM
    assert meta["calibration"] == {"strategy": "crossfit", "n": 24, "guaranteed": True}
    # The bank is the prefix for ratio 0.25 of the full bank saved at ratio 0.5.
    bank = np.load(art / "toy" / "bank.npy")
    full = np.load(grid_dir / "banks" / "toy_full.npy")
    assert bank.shape[0] == run_grid.ratio_rows(24 * 64, 0.25)
    np.testing.assert_array_equal(bank, full[: bank.shape[0]])
    with np.load(grid_dir / "r0.25" / "toy.npz") as z:
        assert meta["threshold"] == pytest.approx(
            float(np.sort(z["pool_score_oof"])[23])
        )  # rank ceil(25*0.95)=24

    run_export_eval.main(
        ["--grid", str(grid_dir), "--ratio", "0.25", "--artifacts", str(art), "--threads", "1"]
    )
    out = world / "outputs" / "export-tiny-dev"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["precisions"] == ["fp32", "int8"] and run["protocol"] == "dev" and run["ratio"] == 0.25
    with np.load(out / "fp32" / "toy.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 6 + [1] * 6
        # The grid ran with torch on the CPU in fp32: the onnxruntime path gives the same scores.
        np.testing.assert_allclose(z["eval_score_full"], z["torch_eval_score"], rtol=2e-3)
        np.testing.assert_allclose(z["pool_score_oof"], z["torch_pool_score_oof"], rtol=2e-3)
        assert set(z["pool_folds"].tolist()) == {1, 2, 3, 4}
    with np.load(out / "int8" / "toy.npz") as z:
        assert not np.allclose(z["eval_score_full"], z["torch_eval_score"], rtol=1e-6)

    report = analyze_export.build_report(analyze_export.load(out))
    assert [r["pipeline"] for r in report["rows"]] == ["torch", "fp32", "int8"]
    fp32 = report["rows"][1]
    assert abs(fp32["vs_torch_auroc"]["diff"]) < 1e-9
    assert fp32["score_rel_diff_vs_torch"]["max"] < 2e-3
    assert fp32["torch_thresholds"]["fpr"] == report["rows"][0]["own_thresholds"]["fpr"]
    shift = report["int8_at_fp32_thresholds"]
    assert shift["verdict"] in {"지지", "기각", "판정 불가"}
    assert shift["d_fpr"] == pytest.approx(shift["int8_at_fp32_thresholds"]["fpr"] - shift["fp32"]["fpr"])
    assert "INT8 under fp32 thresholds" in analyze_export.format_report(report)

    with pytest.raises(SystemExit):  # finished run
        run_export_eval.main(["--grid", str(grid_dir), "--ratio", "0.25", "--artifacts", str(art)])
    assert not (world / "reports" / "test_ledger.jsonl").exists()

    # The artifact of a category loads as an inspector and agrees with the evaluation run.
    insp = Inspector.load(art / "toy", threads=1)
    manifest = _manifest()
    rows = [r for r in manifest if r.category == "toy" and r.role == "dev_defect"]
    with np.load(out / "fp32" / "toy.npz") as z:
        np.testing.assert_allclose(
            insp.scores(FakeCache(manifest).images(rows)), z["eval_score_full"][6:], rtol=1e-6
        )
    timing = bench.time_inspector(insp, FakeCache(manifest).images(rows), warmup=1, repeats=4)
    assert {
        "backbone_ms",
        "search_ms",
        "map_ms",
        "total_ms",
        "total_p95_ms",
        "bank_rows",
        "precision",
    } <= set(timing)
    assert timing["total_ms"] > 0 and timing["bank_rows"] == bank.shape[0] and timing["precision"] == "fp32"


def test_test_protocol_needs_permission_and_is_logged(world):
    grid_dir = _grid(world, "test", ["--allow-test", "--stage", "4"])
    ledger = world / "reports" / "test_ledger.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    art = world / "artifacts" / "tiny"
    export.build_artifacts(grid_dir, 0.5, art, int8=False)
    base = ["--grid", str(grid_dir), "--ratio", "0.5", "--artifacts", str(art), "--precision", "fp32"]
    for extra in ([], ["--allow-test"], ["--stage", "4"]):
        with pytest.raises(SystemExit):
            run_export_eval.main([*base, *extra])
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    run_export_eval.main([*base, "--allow-test", "--stage", "4"])
    lines = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[1]["config"] == "export-tiny" and lines[1]["stage"] == "4"
    with np.load(world / "outputs" / "export-tiny-test" / "fp32" / "two.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 9 + [1] * 4 and len(z["pool_score_oof"]) == 30


def test_grid_without_saved_banks_is_refused(world):
    args = ["--backbone", "wrn50", "--size", str(SIZE), "--device", "cpu", "--categories", "toy"]
    run_grid.main([*args, "--ratios", "0.5", "--protocol", "dev"])
    grid_dir = world / "outputs" / f"grid-wrn50-{SIZE}-dev"
    with pytest.raises(ValueError, match="save-banks"):
        export.build_artifacts(grid_dir, 0.5, world / "artifacts" / "x")


def test_latency_file_accumulates_entries(tmp_path):
    path = tmp_path / "latency.json"
    bench.merge_into(path, "a", {"cpu_fp32": {"total_ms": 1.0}})
    data = bench.merge_into(path, "a", {"cpu_int8": {"total_ms": 2.0}})
    data = bench.merge_into(path, "b", {"cpu_fp32": {"total_ms": 3.0}})
    assert set(data) == {"a", "b"} and set(data["a"]) == {"cpu_fp32", "cpu_int8"}
    assert bench.time_resize(64, source=(200, 100), repeats=2) > 0
    assert isinstance(bench.cpu_name(), str)

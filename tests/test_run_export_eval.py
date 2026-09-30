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
from defect_inspect.patchcore import score_images  # noqa: E402
from defect_inspect.splits import ManifestRow  # noqa: E402

SIZE = 64
DIM = 12
PATCHES = 64  # 8 x 8 patch features per image
# Every read of the fake cache: (roles of the rows, their image names).
EVENTS: list[tuple[tuple[str, ...], tuple[str, ...]]] = []


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
        self._images, self._masks, self._role = {}, {}, {}
        for row in manifest:
            img = rng.integers(90, 150, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[16:40, 16:40] = 255
                mask[64:160, 64:160] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask
            self._role[row.image] = row.role

    def images(self, rows):
        roles = tuple(sorted({self._role[r.image] for r in rows}))
        EVENTS.append((roles, tuple(r.image for r in rows)))
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        return np.stack([self._masks[r.image] for r in rows])


def _sealed_reads():
    return [names for roles, names in EVENTS if any(role.startswith("test") for role in roles)]


def _manifest(categories=("toy", "two"), normals=30, fold_of=lambda i: i % 5):
    rows = []
    for c in categories:
        for i in range(normals):
            rows.append(ManifestRow(f"{c}/n{i:02d}.JPG", "", c, "normal", "pool_normal", fold_of(i), ""))
        for i in range(6):
            rows.append(ManifestRow(f"{c}/d{i}.JPG", f"{c}/d{i}.png", c, "anomaly", "dev_defect", -1, "hole"))
        for i in range(9):
            rows.append(ManifestRow(f"{c}/tn{i}.JPG", "", c, "normal", "test_normal", -1, ""))
        for i in range(4):
            rows.append(
                ManifestRow(f"{c}/td{i}.JPG", f"{c}/td{i}.png", c, "anomaly", "test_defect", -1, "hole")
            )
    return rows


def _uneven_manifest():
    # 33 normals: folds of 7, 7, 7, 6 and 6 images, and the manifest order is not the fold order.
    return _manifest(normals=33, fold_of=lambda i: (i * 3) % 5)


def _world(tmp_path, monkeypatch, manifest):
    EVENTS.clear()
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


@pytest.fixture
def world(tmp_path, monkeypatch):
    return _world(tmp_path, monkeypatch, _manifest())


@pytest.fixture
def uneven_world(tmp_path, monkeypatch):
    return _world(tmp_path, monkeypatch, _uneven_manifest())


def _grid(world, protocol, extra=(), ratios=("0.5", "0.25")):
    args = ["--backbone", "wrn50", "--size", str(SIZE), "--device", "cpu", "--categories", "toy", "two"]
    run_grid.main([*args, "--ratios", *ratios, "--protocol", protocol, "--save-banks", *extra])
    return world / "outputs" / f"grid-wrn50-{SIZE}-{protocol}"


def _ledger_lines(world):
    ledger = world / "reports" / "test_ledger.jsonl"
    return ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []


def test_artifacts_eval_and_analysis_end_to_end(world, capsys):
    grid_dir = _grid(world, "dev")
    art = world / "artifacts" / "tiny"
    info = export.build_artifacts(grid_dir, 0.25, art, calibration_per_category=6)
    assert (art / "model_fp32.onnx").exists() and (art / "model_int8.onnx").exists()
    assert info["calibration_images"] == 12 and [c["category"] for c in info["categories"]] == ["toy", "two"]
    assert info["int8"] == {"available": True}
    saved = json.loads((art / "artifacts.json").read_text(encoding="utf-8"))
    assert saved["int8"] == {"available": True} and saved["calibration_images"] == 12
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
    assert shift["verdict"] in {"지지", "기각", "판정 불가"}  # the rule itself: tests/test_analyze_export.py
    assert shift["d_fpr"] == pytest.approx(shift["int8_at_fp32_thresholds"]["fpr"] - shift["fp32"]["fpr"])
    assert "INT8 under fp32 thresholds" in analyze_export.format_report(report)

    with pytest.raises(SystemExit):  # finished run
        run_export_eval.main(["--grid", str(grid_dir), "--ratio", "0.25", "--artifacts", str(art)])
    assert not (world / "reports" / "test_ledger.jsonl").exists()
    assert not _sealed_reads()

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
        "threads",
    } <= set(timing)
    assert timing["total_ms"] > 0 and timing["bank_rows"] == bank.shape[0] and timing["precision"] == "fp32"
    assert timing["threads"] == 1

    # The latency CLI on the real artifact: inference and resize + inference are both in the entry.
    latency = world / "reports" / "stage4" / "latency.json"
    bench.main(
        ["--artifacts", str(art), "--category", "toy", "--images", "5", "--key", "k", "--out", str(latency)]
    )
    entry = json.loads(latency.read_text(encoding="utf-8"))["k"]
    assert entry["cpu_fp32"]["repeats"] == 5 and entry["resize_ms"] > 0
    assert entry["cpu_fp32"]["total_with_resize_ms"] == entry["resize_ms"] + entry["cpu_fp32"]["total_ms"]


@pytest.mark.parametrize("protocol", ["dev", "test"])
def test_crossfit_scores_against_an_independent_computation(uneven_world, protocol):
    """Unequal folds in a manifest that is not in fold order: every image meets the bank without its fold."""
    world = uneven_world
    extra = ["--allow-test", "--stage", "4"] if protocol == "test" else []
    grid_dir = _grid(world, protocol, extra, ratios=("0.5", "0.25", "0.05"))
    manifest = _uneven_manifest()
    cache = FakeCache(manifest)
    folds_used = (1, 2, 3, 4) if protocol == "dev" else (0, 1, 2, 3, 4)
    ratio = 0.05
    art = world / "artifacts" / "tiny"
    export.build_artifacts(grid_dir, ratio, art, int8=False)
    EVENTS.clear()
    base = ["--grid", str(grid_dir), "--ratio", str(ratio), "--artifacts", str(art), "--precision", "fp32"]
    run_export_eval.main([*base, "--threads", "1", *extra])
    if protocol == "dev":
        assert not _sealed_reads()
    out = world / "outputs" / f"export-tiny-{protocol}"
    for c in ("toy", "two"):
        pool = [r for r in manifest if r.category == c and r.role == "pool_normal" and r.fold in folds_used]
        folds = np.array([r.fold for r in pool])
        assert len({int((folds == f).sum()) for f in folds_used}) > 1  # the folds are unequal
        meta = json.loads((art / c / "meta.json").read_text(encoding="utf-8"))
        assert meta["calibration"]["n"] == len(pool) and meta["source"]["ratio"] == ratio
        with np.load(out / "fp32" / f"{c}.npz") as z:
            got, eval_got = z["pool_score_oof"].copy(), z["eval_score_full"].copy()
            assert z["pool_images"].tolist() == [r.image for r in pool]
            np.testing.assert_array_equal(z["pool_folds"], folds)
            eval_names = z["eval_images"].tolist()
        expect = np.full(len(pool), np.nan, dtype=np.float32)
        wrong = np.full(len(pool), np.nan, dtype=np.float32)
        for f in folds_used:
            held = np.flatnonzero(folds == f)
            images = cache.images([pool[i] for i in held])
            other = folds_used[(folds_used.index(f) + 1) % len(folds_used)]
            for target, fold in ((expect, f), (wrong, other)):
                rows = max(1, round(ratio * int((folds != fold).sum()) * PATCHES))
                bank = np.load(grid_dir / "banks" / f"{c}_minus_fold_{fold}.npy")[:rows]
                target[held] = score_images(
                    TinyPatches(), torch.from_numpy(bank), images, device="cpu"
                ).image_scores
        np.testing.assert_allclose(got, expect, rtol=2e-3)
        # The comparison has teeth: the bank of a neighbouring fold gives clearly different scores.
        assert np.abs(wrong - expect).max() / np.abs(expect).max() > 0.02

        full = np.load(grid_dir / "banks" / f"{c}_full.npy")[: max(1, round(ratio * len(pool) * PATCHES))]
        if protocol == "dev":
            rows = [r for r in manifest if r.category == c and r.role == "pool_normal" and r.fold == 0]
            rows += [r for r in manifest if r.category == c and r.role == "dev_defect"]
        else:
            rows = [r for r in manifest if r.category == c and r.role == "test_normal"]
            rows += [r for r in manifest if r.category == c and r.role == "test_defect"]
        assert eval_names == [r.image for r in rows]
        ref = score_images(TinyPatches(), torch.from_numpy(full), cache.images(rows), device="cpu")
        np.testing.assert_allclose(eval_got, ref.image_scores, rtol=2e-3)


def test_test_protocol_needs_permission_and_is_logged(world, monkeypatch):
    grid_dir = _grid(world, "test", ["--allow-test", "--stage", "4"])
    assert len(_ledger_lines(world)) == 1
    art = world / "artifacts" / "tiny"
    EVENTS.clear()
    export.build_artifacts(grid_dir, 0.5, art, int8=False)
    assert not EVENTS  # without INT8 calibration no image is read at all
    base = ["--grid", str(grid_dir), "--ratio", "0.5", "--artifacts", str(art), "--precision", "fp32"]
    for extra in ([], ["--allow-test"], ["--stage", "4"]):
        with pytest.raises(SystemExit):
            run_export_eval.main([*base, *extra])
    assert len(_ledger_lines(world)) == 1 and not EVENTS

    # The ledger line is written before the first sealed image is read.
    real = run_export_eval.record_test_access

    def recorded(*args, **kwargs):
        EVENTS.append((("ledger",), ()))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_export_eval, "record_test_access", recorded)
    run_export_eval.main([*base, "--allow-test", "--stage", "4"])
    assert EVENTS[0][0] == ("ledger",) and sum(roles == ("ledger",) for roles, _ in EVENTS) == 1
    assert _sealed_reads()
    lines = [json.loads(x) for x in _ledger_lines(world)]
    assert len(lines) == 2 and lines[1]["config"] == "export-tiny" and lines[1]["stage"] == "4"
    with np.load(world / "outputs" / "export-tiny-test" / "fp32" / "two.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 9 + [1] * 4 and len(z["pool_score_oof"]) == 30


def _rewrite_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _rewrite_npz(path, **replaced):
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update(replaced)
    np.savez(path, **arrays)


def test_inputs_that_do_not_fit_stop_the_run_before_the_ledger_and_the_sealed_images(world, capsys):
    """One sealed measurement per stage: whatever can be checked without an image is checked first."""
    grid_dir = _grid(world, "test", ["--allow-test", "--stage", "4"])
    art = world / "artifacts" / "tiny"
    export.build_artifacts(grid_dir, 0.25, art, int8=False)
    other = world / "artifacts" / "other-ratio"
    export.build_artifacts(grid_dir, 0.5, other, int8=False)
    assert len(_ledger_lines(world)) == 1

    def refused(match, *, ratio="0.25", artifacts=art, precision=("fp32",), categories=()):
        EVENTS.clear()
        args = ["--grid", str(grid_dir), "--ratio", ratio, "--artifacts", str(artifacts)]
        args += ["--precision", *precision, "--allow-test", "--stage", "4"]
        if categories:
            args += ["--categories", *categories]
        capsys.readouterr()
        with pytest.raises(SystemExit):
            run_export_eval.main(args)
        assert match in capsys.readouterr().err
        assert not EVENTS, "images were read"
        assert len(_ledger_lines(world)) == 1, "a ledger line was written"
        assert not (world / "outputs" / f"export-{artifacts.name}-test").exists()

    # A ratio the grid run does not have: between two of its ratios, or above the largest.
    refused("not one of the grid run's ratios", ratio="0.3")
    refused("not one of the grid run's ratios", ratio="0.9")
    # Artifacts built at another ratio of the same grid run.
    refused("was built at ratio 0.5", artifacts=other)
    # No INT8 model in the set.
    refused("has no model_int8.onnx", precision=("fp32", "int8"))
    refused("no categories ['nope']", categories=("toy", "nope"))

    bank = grid_dir / "banks" / "two_minus_fold_3.npy"
    kept = bank.read_bytes()
    bank.unlink()
    refused("two_minus_fold_3.npy")
    np.save(bank, np.load(grid_dir / "banks" / "two_full.npy")[:3])  # too few rows for the ratio
    refused("two_minus_fold_3.npy")
    bank.write_bytes(kept)

    meta_path = art / "two" / "meta.json"
    kept = meta_path.read_bytes()
    for key, value in (("sigma", 2.0), ("reweight_k", 3), ("backbone", "dinov2_vits14"), ("dim", 7)):
        _rewrite_json(meta_path, lambda meta, key=key, value=value: meta.update({key: value}))
        refused("another setting")
        meta_path.write_bytes(kept)
    _rewrite_json(meta_path, lambda meta: meta["source"].update(protocol="dev"))
    refused("built from a dev run")
    _rewrite_json(meta_path, lambda meta: meta.update(threshold=None))
    refused("threshold")
    meta_path.write_bytes(kept)

    # The model does not produce what the artifact meta says (another input size).
    model = art / "model_fp32.onnx"
    kept_model = model.read_bytes()
    export.export_onnx(TinyPatches(), 32, model)
    refused("does not fit the artifact")
    model.write_bytes(b"broken")
    refused("not a loadable ONNX model")
    model.write_bytes(kept_model)

    # The grid run scored other images than the manifest selects now.
    npz = grid_dir / "r0.25" / "two.npz"
    kept = npz.read_bytes()
    with np.load(npz) as z:
        names = z["eval_images"].copy()
    _rewrite_npz(npz, eval_images=names[::-1].copy())
    refused("scored other images")
    npz.unlink()
    refused("two.npz")
    npz.write_bytes(kept)

    # Everything restored: the run goes through, with exactly one more ledger line.
    base = ["--grid", str(grid_dir), "--ratio", "0.25", "--artifacts", str(art), "--precision", "fp32"]
    run_export_eval.main([*base, "--allow-test", "--stage", "4"])
    assert len(_ledger_lines(world)) == 2 and _sealed_reads()


def test_run_category_checks_the_image_lists_before_it_reads_an_image(world):
    grid_dir = _grid(world, "dev")
    art = world / "artifacts" / "tiny"
    export.build_artifacts(grid_dir, 0.25, art, int8=False)
    manifest = _manifest()
    meta = json.loads((art / "toy" / "meta.json").read_text(encoding="utf-8"))
    npz = grid_dir / "r0.25" / "toy.npz"
    with np.load(npz) as z:
        names = z["pool_images"].copy()
    _rewrite_npz(npz, pool_images=names[::-1].copy())
    EVENTS.clear()
    with pytest.raises(ValueError, match="scored other images"):
        run_export_eval.run_category(
            export._session(art / "model_fp32.onnx"),
            meta,
            grid_dir,
            0.25,
            "dev",
            "toy",
            manifest,
            FakeCache(manifest),
            False,
        )
    assert not EVENTS


def test_a_model_that_cannot_be_quantized_still_gives_complete_fp32_artifacts(world, monkeypatch, capsys):
    grid_dir = _grid(world, "dev")
    art = world / "artifacts" / "tiny"
    export.build_artifacts(grid_dir, 0.25, art, calibration_per_category=2)
    assert (art / "model_int8.onnx").exists()

    def refuse(fp32_path, int8_path, images, **kwargs):
        raise NotImplementedError("static INT8 quantization is not available for model_fp32.onnx: reason")

    monkeypatch.setattr(export, "quantize_static_int8", refuse)
    info = export.build_artifacts(grid_dir, 0.25, art, calibration_per_category=2)
    assert info["int8"]["available"] is False and "reason" in info["int8"]["reason"]
    assert "quantize_s" not in info and "calibration_images" not in info
    # The INT8 model of the earlier export is gone: it does not belong to this set.
    assert (art / "model_fp32.onnx").exists() and not (art / "model_int8.onnx").exists()
    saved = json.loads((art / "artifacts.json").read_text(encoding="utf-8"))
    assert saved["int8"] == info["int8"] and [c["category"] for c in saved["categories"]] == ["toy", "two"]
    for category in ("toy", "two"):
        assert Inspector.load(art / category).bank_rows == run_grid.ratio_rows(24 * PATCHES, 0.25)
        with pytest.raises(ValueError, match="missing"):
            Inspector.load(art / category, precision="int8")

    base = ["--grid", str(grid_dir), "--ratio", "0.25", "--artifacts", str(art)]
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run_export_eval.main(base)  # default: fp32 and int8
    err = capsys.readouterr().err
    assert "has no model_int8.onnx" in err and "reason" in err and "--precision fp32" in err
    run_export_eval.main([*base, "--precision", "fp32"])
    report = analyze_export.build_report(analyze_export.load(world / "outputs" / "export-tiny-dev"))
    assert report["int8_scored"] is False and "int8_at_fp32_thresholds" not in report

    # Without INT8 requested the set says so as well.
    info = export.build_artifacts(grid_dir, 0.25, art, int8=False)
    assert info["int8"] == {"available": False, "reason": "not requested"}


def test_grid_without_saved_banks_is_refused(world):
    args = ["--backbone", "wrn50", "--size", str(SIZE), "--device", "cpu", "--categories", "toy"]
    run_grid.main([*args, "--ratios", "0.5", "--protocol", "dev"])
    grid_dir = world / "outputs" / f"grid-wrn50-{SIZE}-dev"
    with pytest.raises(ValueError, match="save-banks"):
        export.build_artifacts(grid_dir, 0.5, world / "artifacts" / "x")
    with pytest.raises(SystemExit):
        run_export_eval.main(["--grid", str(grid_dir), "--ratio", "0.5", "--artifacts", str(world / "x")])

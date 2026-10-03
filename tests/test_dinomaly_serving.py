import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")
pytest.importorskip("timm")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from defect_inspect import dinomaly_serving, run_dinomaly  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.conditions import apply_condition  # noqa: E402
from defect_inspect.dinomaly_model import ServingGraph  # noqa: E402
from defect_inspect.inspector import ReconstructionInspector, preprocess  # noqa: E402
from defect_inspect.splits import SEALED_ROLES, ManifestRow, write_manifest  # noqa: E402

SIZE = 28  # a 2x2 patch grid


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A random-weight ViT-S Dinomaly with a disturbed decoder, exported once for the module."""
    import timm
    import timm.models.vision_transformer as vit

    with pytest.MonkeyPatch.context() as mp:
        real_create = timm.create_model

        def create_without_download(name, *args, **kwargs):
            kwargs["pretrained"] = False
            return real_create(name, *args, **kwargs)

        mp.setattr(timm, "create_model", create_without_download)
        mp.setattr(vit, "resample_abs_pos_embed", vit.resample_abs_pos_embed)
        mp.setattr(dinomaly_serving, "git_commit", lambda: "abc1234")
        torch.manual_seed(0)
        options = {"img_size": SIZE, "encoder_blocks": 10, "attention": "scaled"}
        model = run_dinomaly.build_model(run_dinomaly.ENCODER_S, **options).eval()
        with torch.no_grad():
            for p in model.decoder.parameters():
                p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(1)) * 0.05)
        out = tmp_path_factory.mktemp("art") / "dms-toy"
        info = dinomaly_serving.build_artifacts(
            model,
            name="dms-toy",
            img_size=SIZE,
            out_dir=out,
            source={"config": "dms-toy", "untrained": True},
            categories=["toy", "other"],
            int8_dynamic=True,
        )
        yield {"model": model, "out": out, "info": info}


def _images(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, SIZE, SIZE, 3), dtype=np.uint8)


def test_export_writes_an_artifact_set_that_matches_torch(exported):
    out, info = exported["out"], exported["info"]
    assert (out / "model_fp32.onnx").exists() and (out / "model_int8_dynamic.onnx").exists()
    assert info["parity"]["score_rel_diff"] <= 1e-3 and info["parity"]["map_abs_diff"] <= 1e-3
    assert info["int8_dynamic"]["available"] and info["int8_dynamic"]["op_types"] == ["MatMul"]
    record = json.loads((out / "artifacts.json").read_text(encoding="utf-8"))
    assert record["name"] == "dms-toy" and record["img_size"] == SIZE and record["commit"] == "abc1234"
    meta = json.loads((out / "toy" / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "reconstruction" and meta["map_size"] == 256 and meta["img_size"] == SIZE
    assert meta["threshold"] == 0.0 and meta["calibration"] == {"strategy": "none"}
    assert len(meta["source"]["onnx_sha256"]) == 64 and meta["source"]["untrained"] is True

    graph = ServingGraph(exported["model"]).eval()
    insp = ReconstructionInspector.load(out / "toy")
    for image in _images(3, seed=4):
        score, amap = insp.run(image)
        with torch.no_grad():
            ref_score, ref_map = graph(torch.from_numpy(preprocess(image, SIZE)))
        assert score == pytest.approx(float(ref_score[0]), rel=1e-4)
        assert amap.shape == (256, 256) and float(np.abs(amap - ref_map[0].numpy()).max()) < 1e-4
    quantized = ReconstructionInspector.load(out / "toy", precision="int8-dynamic")
    scores = quantized.scores(_images(3, seed=4))
    assert quantized.precision == "int8-dynamic" and np.isfinite(scores).all()


class FakeCache:
    """Images by manifest row; asserts that a sealed image is only handed out after the ledger line."""

    def __init__(self, manifest, ledger):
        rng = np.random.default_rng(0)
        self._images = {}
        self._role = {}
        self._ledger = ledger
        self.requested = []
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            if row.label == "anomaly":
                img[4:12, 4:12] = 255
            self._images[row.image] = img
            self._role[row.image] = row.role

    def images(self, rows):
        if any(self._role[r.image] in SEALED_ROLES for r in rows):
            assert self._ledger.exists() and self._ledger.read_text(encoding="utf-8").strip()
        self.requested += [r.image for r in rows]
        return np.stack([self._images[r.image] for r in rows])


def _manifest():
    rows = []
    for category in ("toy", "other"):
        for i in range(20):
            rows.append(ManifestRow(f"{category}/n{i}.JPG", "", category, "normal", "pool_normal", i % 5, ""))
        for i in range(3):
            rows.append(
                ManifestRow(f"{category}/d{i}.JPG", "m.png", category, "anomaly", "dev_defect", -1, "hole")
            )
        for i in range(5):
            rows.append(ManifestRow(f"{category}/tn{i}.JPG", "", category, "normal", "test_normal", -1, ""))
        for i in range(2):
            rows.append(
                ManifestRow(f"{category}/td{i}.JPG", "m.png", category, "anomaly", "test_defect", -1, "hole")
            )
    return rows


@pytest.fixture
def project(tmp_path, monkeypatch):
    manifest = _manifest()
    ledger = tmp_path / "ledger.jsonl"
    cache = FakeCache(manifest, ledger)
    sizes = []

    def fake_cache(root, size):
        sizes.append(size)
        return cache

    write_manifest(manifest, tmp_path / "visa.csv")
    monkeypatch.setattr(dinomaly_serving.paths, "VISA_MANIFEST", tmp_path / "visa.csv")
    monkeypatch.setattr(dinomaly_serving.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(dinomaly_serving.paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(dinomaly_serving, "ImageCache", fake_cache)
    monkeypatch.setattr(dinomaly_serving, "git_commit", lambda: "abc1234")
    return {"cache": cache, "ledger": ledger, "outputs": tmp_path / "outputs", "sizes": sizes}


def test_score_cli_dev_test_conditions_and_calibrate(exported, project, tmp_path):
    import shutil

    art = tmp_path / "dms-toy"
    shutil.copytree(exported["out"], art)
    base = ["score", "--artifacts", str(art), "--categories", "toy", "other"]
    dinomaly_serving.main(base + ["--protocol", "dev"])
    dev_dir = project["outputs"] / "dms-toy-onnx-fp32-dev"
    run = json.loads((dev_dir / "run.json").read_text(encoding="utf-8"))
    assert run["protocol"] == "dev" and run["config"]["condition"] == "clean" and run["commit"] == "abc1234"
    assert project["sizes"] == [SIZE] and not project["ledger"].exists()
    insp = ReconstructionInspector.load(art / "toy")
    with np.load(dev_dir / "toy.npz") as z:
        assert z["eval_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 20, 5)] + [
            f"toy/d{i}.JPG" for i in range(3)
        ]
        assert z["eval_labels"].tolist() == [0] * 4 + [1] * 3 and z["cal_score"].shape == (0,)
        rows = [r for r in _manifest() if r.image in set(z["eval_images"].tolist())]
        expected = insp.scores(project["cache"].images(rows))
        np.testing.assert_array_equal(z["eval_score"], expected)
    assert not any("/t" in image for image in project["cache"].requested)

    with pytest.raises(SystemExit):  # the sealed test set needs both flags
        dinomaly_serving.main(base + ["--protocol", "test", "--allow-test"])
    assert not project["ledger"].exists()
    dinomaly_serving.main(base + ["--protocol", "test", "--allow-test", "--stage", "6"])
    line = json.loads(project["ledger"].read_text(encoding="utf-8").splitlines()[-1])
    assert line["stage"] == "6" and line["config"] == "dms-toy-onnx-fp32"
    test_dir = project["outputs"] / "dms-toy-onnx-fp32-test"
    with np.load(test_dir / "toy.npz") as z:
        assert z["cal_images"].tolist() == [f"toy/n{i}.JPG" for i in range(0, 20, 5)]  # fold 0
        assert z["eval_images"].tolist()[:5] == [f"toy/tn{i}.JPG" for i in range(5)]
        cal_score = z["cal_score"]
        clean_eval = z["eval_score"]

    # A condition changes the evaluation images only; the thresholds stay those of the clean normals.
    dinomaly_serving.main(
        base + ["--protocol", "test", "--allow-test", "--stage", "6", "--condition", "brightness-3"]
    )
    with np.load(project["outputs"] / "dms-toy-onnx-fp32-test-brightness-3" / "toy.npz") as z:
        np.testing.assert_array_equal(z["cal_score"], cal_score)
        assert not np.array_equal(z["eval_score"], clean_eval)
        test_rows = [r for r in _manifest() if r.category == "toy" and r.role in SEALED_ROLES]
        blurred = apply_condition(project["cache"].images(test_rows), "brightness-3")
        np.testing.assert_array_equal(z["eval_score"], insp.scores(blurred))
    with pytest.raises(SystemExit):
        dinomaly_serving.main(base + ["--protocol", "dev", "--condition", "fog-1"])

    # Thresholds for the service: hold-out on the fold-0 normals of the clean test run only.
    for run_dir in (dev_dir, project["outputs"] / "dms-toy-onnx-fp32-test-brightness-3"):
        with pytest.raises(SystemExit):
            dinomaly_serving.main(["calibrate", "--artifacts", str(art), "--scores", str(run_dir)])
    dinomaly_serving.main(["calibrate", "--artifacts", str(art), "--scores", str(test_dir)])
    meta = json.loads((art / "toy" / "meta.json").read_text(encoding="utf-8"))
    thr = conformal_threshold(cal_score, 0.05)
    assert meta["threshold"] == float(thr.value) and meta["alpha"] == 0.05
    assert meta["calibration"] == {"strategy": "holdout", "n": 4, "guaranteed": False}
    assert ReconstructionInspector.load(art / "toy").threshold == meta["threshold"]


def test_export_cli_needs_exactly_one_model_source(tmp_path):
    with pytest.raises(SystemExit):
        dinomaly_serving.main(["export", "--config", "dms-280", "--out", str(tmp_path / "a")])
    with pytest.raises(SystemExit):
        dinomaly_serving.main(
            ["export", "--config", "dms-280", "--untrained", "--model", "m.pt", "--out", str(tmp_path / "a")]
        )
    assert not (tmp_path / "a").exists()

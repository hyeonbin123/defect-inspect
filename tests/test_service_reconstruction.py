"""The service with a reconstruction (Dinomaly) artifact set: the model scores by itself, there is no bank."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from defect_inspect import service  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.inspector import ReconstructionInspector  # noqa: E402

HOST = "http://127.0.0.1"
TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}
SIZE = 32
MAP = 256


class ReconSession:
    """Stand-in for the exported Dinomaly graph: the map is the brightness blown up to 256, the score its
    maximum."""

    def run(self, names, feeds):
        assert names == ["score", "map"]
        x = feeds["image"]
        assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
        bright = x.mean(axis=1)
        amap = np.repeat(np.repeat(bright, MAP // SIZE, axis=1), MAP // SIZE, axis=2)
        return [amap.reshape(1, -1).max(axis=1).astype(np.float32), amap.astype(np.float32)]


def _meta(category="toy", **extra) -> dict:
    meta = {
        "version": 1,
        "kind": "reconstruction",
        "name": "dms-toy",
        "category": category,
        "img_size": SIZE,
        "map_size": MAP,
        "threshold": 0.0,
    }
    meta.update(extra)
    return meta


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _normal(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(110, 146, (SIZE, SIZE, 3), dtype=np.uint8)


def _defect(seed: int) -> np.ndarray:
    image = _normal(seed)
    image[8:16, 8:16] = 255
    return image


def _bright(count: int) -> list[np.ndarray]:
    return [np.clip(_normal(100 + s).astype(np.int16) + 60, 0, 255).astype(np.uint8) for s in range(count)]


def _inspector() -> ReconstructionInspector:
    inspector = ReconstructionInspector(ReconSession(), _meta())
    inspector.calibrate([_normal(s) for s in range(24)], alpha=0.05)
    return inspector


def _client(inspector) -> TestClient:
    return TestClient(service.create_app(inspectors={"toy": inspector}, admin_token=TOKEN), base_url=HOST)


def _inspect(client, image: np.ndarray, **params):
    files = {"image": ("x.png", _png(image), "image/png")}
    return client.post("/inspect", files=files, data={"category": "toy"}, params=params)


def test_categories_describe_a_reconstruction_model():
    item = _client(_inspector()).get("/categories").json()[0]
    assert item["kind"] == "reconstruction" and item["model"] == "dms-toy"
    assert item["bank_rows"] == 0 and item["backbone"] is None and item["img_size"] == SIZE
    assert item["calibration"] == {"strategy": "holdout", "n": 24, "guaranteed": True}


def test_inspect_uses_the_models_own_score_and_map():
    inspector = _inspector()
    client = _client(inspector)
    ok = _inspect(client, _normal(50)).json()
    assert ok["is_defect"] is False and ok["score"] == inspector.inspect(_normal(50)).score
    bad = _inspect(client, _defect(50)).json()
    assert bad["is_defect"] is True and bad["score"] > bad["threshold"] == inspector.threshold
    heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(bad["heatmap_png"]))))
    # The map of the model itself, scaled like the PatchCore one: white at 1.5 x the threshold.
    assert heat.shape == (MAP, MAP) and heat[90, 90] == 255 and heat[200:, 200:].max() < 255
    preview = _inspect(client, _defect(50), preview="true").json()
    seen = np.asarray(Image.open(io.BytesIO(base64.b64decode(preview["input_png"]))))
    assert seen.shape == (SIZE, SIZE, 3)


def test_calibrate_only_moves_the_threshold_and_reset_restores_it():
    inspector = _inspector()
    client = _client(inspector)
    before = client.get("/categories").json()[0]
    bright = _bright(30)
    assert _inspect(client, bright[0]).json()["is_defect"] is True
    files = [("images", (f"{i}.png", _png(im), "image/png")) for i, im in enumerate(bright[:25])]
    response = client.post("/calibrate", files=files, data={"category": "toy"}, headers=ADMIN)
    assert response.status_code == 200
    body = response.json()
    # The same rule as the inspector's own calibration, on the scores of the decoded uploads.
    scores = np.array([inspector.inspect(im).score for im in bright[:25]], dtype=np.float32)
    assert body["threshold"] == conformal_threshold(scores, 0.05).value
    assert body["previous_threshold"] == before["threshold"]
    assert body["calibration"] == {"strategy": "holdout", "n": 25, "guaranteed": True}
    assert sum(_inspect(client, im).json()["is_defect"] for im in bright[25:]) <= 1
    assert inspector.session.__class__ is ReconSession  # nothing but the threshold was replaced

    assert client.delete("/calibrate", params={"category": "toy"}).status_code == 403
    reset = client.delete("/calibrate", params={"category": "toy"}, headers=ADMIN).json()
    assert reset["threshold"] == before["threshold"] and reset["calibration"] == before["calibration"]
    assert client.get("/categories").json()[0] == before


def _fake_ort(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    created = []

    class FakeSession(ReconSession):
        def __init__(self, path, sess_options=None, providers=None):
            self.path = Path(path)
            created.append((self.path.parent.name, self.path.name, sess_options.intra_op_num_threads))

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    return created


def _write_set(root: Path, categories=("candle", "pcb1", "pcb2")) -> None:
    (root / "model_fp32.onnx").write_bytes(b"shared")
    for index, name in enumerate(categories):
        (root / name).mkdir()
        meta = _meta(name, threshold=0.25 + index, calibration={"strategy": "holdout", "n": 60})
        (root / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_load_inspectors_reads_a_reconstruction_set(tmp_path, monkeypatch):
    created = _fake_ort(monkeypatch)
    _write_set(tmp_path)
    (tmp_path / "pcb2" / "model_fp32.onnx").write_bytes(b"own")
    (tmp_path / "notes").mkdir()  # no meta.json: not a category

    found = service.load_inspectors(tmp_path, threads=2)
    assert sorted(found) == ["candle", "pcb1", "pcb2"]
    assert all(isinstance(i, ReconstructionInspector) for i in found.values())
    assert found["candle"].session is found["pcb1"].session is not found["pcb2"].session
    assert created == [(tmp_path.name, "model_fp32.onnx", 2), ("pcb2", "model_fp32.onnx", 2)]
    # Each inspector is what `ReconstructionInspector.load` builds for its folder, attribute by attribute.
    for name, inspector in found.items():
        alone = ReconstructionInspector.load(tmp_path / name, threads=2)
        mine, theirs = vars(inspector), vars(alone)
        assert set(mine) == set(theirs)
        for key, value in mine.items():
            if key == "session":
                assert value.path == theirs[key].path
            else:
                assert value == theirs[key], key
        assert (inspector.precision, inspector.threads, inspector.threshold) == ("fp32", 2, alone.threshold)

    # The model file of a precision depends on the kind of the set.
    with pytest.raises(ValueError, match="model_int8_dynamic.onnx is missing"):
        service.load_inspectors(tmp_path, precision="int8-dynamic")
    with pytest.raises(ValueError, match=r"candle.*\['fp32', 'int8-dynamic'\] for a reconstruction artifact"):
        service.load_inspectors(tmp_path, precision="int8")
    with pytest.raises(ValueError, match="precision must be one of"):
        service.load_inspectors(tmp_path, precision="fp16")
    (tmp_path / "model_int8_dynamic.onnx").write_bytes(b"int8")
    (tmp_path / "pcb2" / "model_int8_dynamic.onnx").write_bytes(b"int8")
    assert service.load_inspectors(tmp_path, precision="int8-dynamic")["pcb1"].precision == "int8-dynamic"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"map_size": None}, "map_size"),
        ({"kind": "something-else"}, "unknown artifact kind"),
        ({"threshold": "high"}, "threshold"),
    ],
)
def test_a_bad_reconstruction_artifact_names_its_folder(tmp_path, monkeypatch, change, message):
    _fake_ort(monkeypatch)
    _write_set(tmp_path)
    meta = {**_meta("pcb1"), **change}
    (tmp_path / "pcb1" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=f"pcb1.*{message}"):
        service.load_inspectors(tmp_path)


def test_a_reconstruction_set_needs_no_bank_and_a_patchcore_set_still_does(tmp_path, monkeypatch):
    _fake_ort(monkeypatch)
    _write_set(tmp_path, categories=("pcb1",))
    assert not list(tmp_path.rglob("bank.npy"))
    assert service.load_inspectors(tmp_path)["pcb1"].kind == "reconstruction"
    patchcore = {k: v for k, v in _meta("pcb1").items() if k not in ("kind", "map_size")}
    patchcore.update(backbone="fake", grid=[4, 4], dim=3, reweight_k=9, sigma=1.0)
    (tmp_path / "pcb1" / "meta.json").write_text(json.dumps(patchcore), encoding="utf-8")
    with pytest.raises(ValueError, match="bank.npy is missing"):
        service.load_inspectors(tmp_path)


def _toy_onnx(path: Path) -> None:
    """A real ONNX model with the serving interface: map = channel mean [1, S, S], score = its max [1]."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, SIZE, SIZE])
    score = helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])
    amap = helper.make_tensor_value_info("map", TensorProto.FLOAT, [1, SIZE, SIZE])
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["map"], axes=[1], keepdims=0),
        helper.make_node("ReduceMax", ["map"], ["score"], axes=[1, 2], keepdims=0),
    ]
    graph = helper.make_graph(nodes, "toy", [image], [score, amap])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_a_real_onnx_reconstruction_set_end_to_end(tmp_path):
    pytest.importorskip("onnxruntime")
    _toy_onnx(tmp_path / "model_fp32.onnx")
    for name, threshold in (("pcb1", 0.5), ("pcb2", 0.7)):
        (tmp_path / name).mkdir()
        meta = _meta(name, map_size=SIZE, threshold=threshold)
        (tmp_path / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    app = service.create_app(tmp_path, admin_token=TOKEN, allowed_hosts=["127.0.0.1"])
    assert app.state.inspectors["pcb1"].session is app.state.inspectors["pcb2"].session
    client = TestClient(app, base_url=HOST)
    items = {i["category"]: i for i in client.get("/categories").json()}
    assert {k: (v["kind"], v["threshold"]) for k, v in items.items()} == {
        "pcb1": ("reconstruction", 0.5),
        "pcb2": ("reconstruction", 0.7),
    }
    alone = ReconstructionInspector.load(tmp_path / "pcb1")
    for image in (_normal(7), _defect(7)):
        files = {"image": ("x.png", _png(image), "image/png")}
        body = client.post("/inspect", files=files, data={"category": "pcb1"}).json()
        assert body["score"] == alone.inspect(image).score
        assert body["is_defect"] is (body["score"] > 0.5)
        assert np.asarray(Image.open(io.BytesIO(base64.b64decode(body["heatmap_png"])))).shape == (SIZE, SIZE)

import json
from types import SimpleNamespace

import numpy as np
import pytest

from defect_inspect import bench, inspector
from defect_inspect.inspector import ReconstructionInspector

SIZE = 28
MAP = 16


class FakeSession:
    """Stand-in for the exported Dinomaly graph: the map is the mean brightness, the score its maximum."""

    def __init__(self, score_value=None):
        self.calls = 0
        self.score_value = score_value

    def run(self, names, feeds):
        assert names == ["score", "map"]
        x = feeds["image"]
        assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
        self.calls += 1
        brightness = x.mean(axis=1)  # [1, S, S]
        amap = brightness[:, :: SIZE // MAP + 0, :: SIZE // MAP + 0][:, :MAP, :MAP]
        amap = np.ascontiguousarray(
            np.pad(amap, ((0, 0), (0, MAP - amap.shape[1]), (0, MAP - amap.shape[2])))
        )
        score = amap.reshape(1, -1).max(axis=1) if self.score_value is None else np.array([self.score_value])
        return [score.astype(np.float32), amap.astype(np.float32)]


def _meta(**extra):
    meta = {
        "version": inspector.ARTIFACT_VERSION,
        "kind": "reconstruction",
        "name": "dms-toy",
        "category": "toy",
        "img_size": SIZE,
        "map_size": MAP,
        "threshold": 0.5,
    }
    meta.update(extra)
    return meta


def _images(n, seed=0, defect=False):
    rng = np.random.default_rng(seed)
    images = rng.integers(100, 140, (n, SIZE, SIZE, 3), dtype=np.uint8)
    if defect:
        images[:, 4:12, 4:12] = 255
    return images


def test_inspect_scores_and_calibrate():
    insp = ReconstructionInspector(FakeSession(), _meta())
    assert insp.bank_rows == 0 and insp.kind == "reconstruction" and insp.size == SIZE
    normal = insp.inspect(_images(1)[0])
    defect = insp.inspect(_images(1, defect=True)[0])
    assert defect.score > normal.score and defect.is_defect and not normal.is_defect
    assert normal.heatmap.shape == (MAP, MAP) and normal.heatmap.dtype == np.float32
    assert insp.inspect(_images(1)[0], heatmap=False).heatmap is None
    scores = insp.scores(_images(5, seed=1))
    assert scores.shape == (5,) and scores.dtype == np.float32
    assert scores[0] == np.float32(insp.run(_images(5, seed=1)[0])[0])

    normals = list(_images(40, seed=2))
    value = insp.calibrate(normals, alpha=0.05)
    expected = np.sort(insp.scores(np.stack(normals)))[38]  # rank ceil(41 * 0.95) = 39 of 40
    assert value == insp.threshold == insp.meta["threshold"] == float(expected)
    assert insp.meta["calibration"] == {"strategy": "holdout", "n": 40, "guaranteed": True}
    with pytest.raises(ValueError):
        insp.calibrate([])
    with pytest.raises(ValueError):
        insp.set_threshold(float("nan"))


def test_a_non_finite_score_never_gives_a_verdict():
    insp = ReconstructionInspector(FakeSession(score_value=float("nan")), _meta())
    with pytest.raises(FloatingPointError):
        insp.inspect(_images(1)[0])


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "patchcore"},
        {"version": 99},
        {"img_size": 0},
        {"map_size": 2.5},
        {"threshold": float("inf")},
        {"threshold": None},
    ],
)
def test_bad_meta_is_refused(change):
    with pytest.raises(ValueError):
        ReconstructionInspector(FakeSession(), _meta(**change))
    meta = _meta()
    meta.pop("map_size")
    with pytest.raises(ValueError, match="lacks"):
        inspector.check_reconstruction_meta(meta)


def test_the_declared_model_shapes_must_fit_the_meta():
    def node(name, shape):
        return SimpleNamespace(name=name, shape=shape)

    class Described(FakeSession):
        def __init__(self, inputs, outputs):
            super().__init__()
            self._inputs, self._outputs = inputs, outputs

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

    good_in = [node("image", [1, 3, SIZE, SIZE])]
    good_out = [node("score", [1]), node("map", [1, MAP, MAP])]
    ReconstructionInspector(Described(good_in, good_out), _meta())
    ReconstructionInspector(Described([node("image", ["b", 3, SIZE, SIZE])], good_out), _meta())
    for inputs, outputs in [
        ([node("image", [1, 3, 32, 32])], good_out),
        (good_in, [node("score", [1]), node("map", [1, 8, 8])]),
        (good_in, [node("map", [1, MAP, MAP])]),
        ([node("input", [1, 3, SIZE, SIZE])], good_out),
    ]:
        with pytest.raises(ValueError):
            ReconstructionInspector(Described(inputs, outputs), _meta())


def test_artifact_kind(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "meta.json").write_text('{"img_size": 32}', encoding="utf-8")
    assert inspector.artifact_kind(tmp_path / "a") == "reconstruction"
    assert inspector.artifact_kind(tmp_path / "b") == "patchcore"
    with pytest.raises(ValueError):
        inspector.artifact_kind(tmp_path / "missing")


def test_load_turns_missing_files_into_value_errors(tmp_path):
    pytest.importorskip("onnxruntime")
    art = tmp_path / "dms" / "toy"
    art.mkdir(parents=True)
    with pytest.raises(ValueError, match="missing"):
        ReconstructionInspector.load(art)
    (art / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    (art.parent / "model_fp32.onnx").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="not a loadable ONNX model"):
        ReconstructionInspector.load(art)
    with pytest.raises(ValueError, match="precision"):
        ReconstructionInspector.load(art, precision="int8")


# ---------------------------------------------------------------- bench


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class TimedInspector:
    precision = "fp32"
    threads = None

    def __init__(self, clock):
        self.clock = clock
        self.seen = []

    def run(self, image):
        call = len(self.seen)
        self.seen.append(int(image[0, 0, 0]))
        self.clock.now += (900.0 if call < 5 else 150.0 + (call % 2)) / 1e3
        return 1.0, np.zeros((MAP, MAP), np.float32)


def test_time_reconstruction_reports_the_median_after_the_warmup(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(bench, "time", clock)
    images = np.zeros((3, SIZE, SIZE, 3), np.uint8)
    images[:, 0, 0, 0] = np.arange(3)
    fake = TimedInspector(clock)
    timing = bench.time_reconstruction(fake, images, warmup=5, repeats=10)
    assert len(fake.seen) == 15 and fake.seen[:4] == [0, 1, 2, 0]
    assert timing["total_ms"] == pytest.approx(150.5) and timing["repeats"] == 10
    assert timing["kind"] == "reconstruction" and timing["precision"] == "fp32"
    with pytest.raises(ValueError):
        bench.time_reconstruction(fake, images[:0])


def test_bench_cli_times_a_reconstruction_artifact(tmp_path, monkeypatch):
    from defect_inspect.splits import ManifestRow

    manifest = [
        ManifestRow(f"toy/n{i}.JPG", "", "toy", "normal", "pool_normal", i % 5, "") for i in range(20)
    ]
    read = []

    class FakeCache:
        def __init__(self, out_dir, size):
            assert size == SIZE

        def images(self, rows):
            read.append([r.image for r in rows])
            return _images(len(rows))

    loaded = []

    def fake_load(art_dir, *, precision, threads):
        loaded.append((art_dir.name, precision, threads))
        return SimpleNamespace(precision=precision, threads=threads)

    def fake_time(insp, images, *, repeats):
        assert repeats == len(images) == 6
        return {"total_ms": 180.0 if insp.precision == "fp32" else 120.0, "precision": insp.precision}

    monkeypatch.setattr("defect_inspect.cache.ImageCache", FakeCache)
    monkeypatch.setattr("defect_inspect.splits.read_manifest", lambda path: manifest)
    monkeypatch.setattr(bench.ReconstructionInspector, "load", staticmethod(fake_load))
    monkeypatch.setattr(bench, "time_reconstruction", fake_time)
    monkeypatch.setattr(bench, "time_resize", lambda size: 9.0)
    art = tmp_path / "artifacts" / "dms-toy"
    (art / "toy").mkdir(parents=True)
    (art / "toy" / "meta.json").write_text(json.dumps(_meta(source={"untrained": True})), encoding="utf-8")
    out = tmp_path / "reports" / "stage6" / "latency.json"

    args = ["--artifacts", str(art), "--category", "toy", "--images", "6", "--out", str(out)]
    bench.main([*args, "--precision", "fp32", "int8-dynamic", "--key", "dms-toy-untrained"])
    entry = json.loads(out.read_text(encoding="utf-8"))["dms-toy-untrained"]
    assert entry["cpu_fp32"]["total_with_resize_ms"] == 189.0
    assert entry["cpu_int8-dynamic"]["total_with_resize_ms"] == 129.0
    assert entry["source"] == {"untrained": True} and entry["resize_ms"] == 9.0
    assert loaded == [("toy", "fp32", None), ("toy", "int8-dynamic", None)]
    dev_pool = [r.image for r in manifest if r.fold != 0]
    assert read == [dev_pool[:6]]

    for bad in (["--precision", "int8"], ["--gpu"]):
        with pytest.raises(SystemExit):
            bench.main([*args, "--key", "x", *bad])

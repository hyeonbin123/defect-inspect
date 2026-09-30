import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from defect_inspect import export, inspector  # noqa: E402
from defect_inspect.backbones import WideResNetPatches  # noqa: E402
from defect_inspect.inspector import Inspector  # noqa: E402
from defect_inspect.patchcore import build_bank, collect_features, score_images  # noqa: E402

DIM = 12


class TinyPatches(torch.nn.Module):
    """A small conv extractor with the interface of backbones.py: [B, 3, S, S] -> [B, S/8, S/8, DIM]."""

    name = "tiny"
    dim = DIM

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, DIM, 3, stride=4, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(self.conv2(x), 3, 1, 1)
        return x.permute(0, 2, 3, 1).contiguous()


def _images(n, size, seed=0, defect=False):
    rng = np.random.default_rng(seed)
    images = rng.integers(60, 200, (n, size, size, 3), dtype=np.uint8)
    if defect:
        q = size // 4
        images[:, q : 2 * q, q : 2 * q] = 255
    return images


def _meta(size, grid, **extra):
    meta = {
        "version": inspector.ARTIFACT_VERSION,
        "name": "tiny",
        "category": "toy",
        "backbone": "tiny",
        "img_size": size,
        "grid": [grid, grid],
        "dim": DIM,
        "reweight_k": 3,
        "sigma": 1.0,
        "threshold": 0.0,
    }
    meta.update(extra)
    return meta


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    out = {}
    for size in (64, 256):
        path = tmp_path_factory.mktemp(f"onnx{size}") / "model_fp32.onnx"
        export.export_onnx(TinyPatches(), size, path)
        out[size] = path
    return out


def test_export_matches_the_torch_extractor(tiny_onnx):
    parity = export.check_parity(TinyPatches(), tiny_onnx[64], _images(4, 64, seed=1))
    assert parity["max_abs"] < 1e-4 and parity["rel"] < 1e-4
    session = export._session(tiny_onnx[64])
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == ["features"]
    assert session.get_inputs()[0].shape == [1, 3, 64, 64]
    assert session.get_outputs()[0].shape == [1, 8, 8, DIM]


@pytest.mark.parametrize("size", [64, 256])
def test_inspector_reproduces_the_torch_pipeline(tiny_onnx, size):
    """Image scores and 256-grid maps of the numpy/onnxruntime path against patchcore.score_images on CPU."""
    ext = TinyPatches()
    train = _images(10, size, seed=2)
    feats, grid = collect_features(ext, train, batch_size=4, device="cpu")
    bank = build_bank(feats, 0.25, seed=0, device="cpu")
    images = np.concatenate([_images(5, size, seed=3), _images(5, size, seed=4, defect=True)])
    ref = score_images(ext, bank, images, batch_size=4, device="cpu", reweight_k=3, sigma=1.0, map_size=256)

    insp = Inspector(export._session(tiny_onnx[size]), bank.numpy(), _meta(size, grid[0]))
    scores = insp.scores(images)
    np.testing.assert_allclose(scores, ref.image_scores, rtol=1e-3)
    span = float(ref.maps.astype(np.float32).max() - ref.maps.astype(np.float32).min())
    for i in (0, 7):
        heat = insp.inspect(images[i]).heatmap
        assert np.abs(heat - ref.maps[i].astype(np.float32)).max() < 1e-2 * span


def test_inspector_loads_a_saved_artifact_set(tiny_onnx, tmp_path):
    ext = TinyPatches()
    feats, grid = collect_features(ext, _images(6, 64, seed=5), batch_size=4, device="cpu")
    bank = build_bank(feats, 0.5, seed=0, device="cpu").numpy()
    meta = {k: v for k, v in _meta(64, grid[0], threshold=1.25).items() if k != "version"}
    # Single-folder layout.
    inspector.save_artifact(tmp_path / "one", onnx_fp32=tiny_onnx[64], bank=bank, meta=meta)
    a = Inspector.load(tmp_path / "one", threads=1)
    # Set layout: the model next to the category folders.
    (tmp_path / "set" / "toy").mkdir(parents=True)
    (tmp_path / "set" / "model_fp32.onnx").write_bytes(tiny_onnx[64].read_bytes())
    for name in ("bank.npy", "meta.json"):
        (tmp_path / "set" / "toy" / name).write_bytes((tmp_path / "one" / name).read_bytes())
    b = Inspector.load(tmp_path / "set" / "toy")
    images = _images(3, 64, seed=6)
    np.testing.assert_array_equal(a.scores(images), b.scores(images))
    assert a.threshold == 1.25 and a.bank_rows == bank.shape[0] and a.precision == "fp32"
    assert a.threads == 1 and b.threads is None  # None = onnxruntime's default
    with pytest.raises(ValueError, match="missing"):
        Inspector.load(tmp_path / "one", precision="int8")


def test_int8_quantization_gives_a_loadable_model_close_to_fp32(tiny_onnx, tmp_path):
    int8 = export.quantize_static_int8(tiny_onnx[64], tmp_path / "model_int8.onnx", _images(16, 64, seed=7))
    assert int8.exists() and not (tmp_path / "model_int8.prep.onnx").exists()
    images = _images(4, 64, seed=8)
    fp32_session, int8_session = export._session(tiny_onnx[64]), export._session(int8)
    a = np.concatenate(
        [fp32_session.run(["features"], {"image": inspector.preprocess(im, 64)})[0] for im in images]
    )
    b = np.concatenate(
        [int8_session.run(["features"], {"image": inspector.preprocess(im, 64)})[0] for im in images]
    )
    assert a.shape == b.shape and not np.array_equal(a, b)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98


def test_quantization_needs_a_model_and_calibration_images(tiny_onnx, tmp_path):
    with pytest.raises(FileNotFoundError):
        export.quantize_static_int8(tmp_path / "nothing.onnx", tmp_path / "int8.onnx", _images(2, 64))
    with pytest.raises(ValueError, match="calibration"):
        export.quantize_static_int8(tiny_onnx[64], tmp_path / "int8.onnx", _images(0, 64))
    assert not (tmp_path / "int8.onnx").exists()


def test_a_graph_the_preprocessing_cannot_handle_is_refused_with_the_reason(tiny_onnx, tmp_path, monkeypatch):
    from onnxruntime.quantization import shape_inference

    def broken(*args, **kwargs):
        raise TypeError("object of type 'NoneType' has no len()")

    monkeypatch.setattr(shape_inference, "quant_pre_process", broken)
    with pytest.raises(NotImplementedError, match="pre-processing failed .*TypeError.*NoneType"):
        export.quantize_static_int8(tiny_onnx[64], tmp_path / "model_int8.onnx", _images(2, 64))
    assert list(tmp_path.iterdir()) == []  # neither a model nor the intermediate file

    def disk_full(*args, **kwargs):
        raise OSError("no space left on device")

    # A failure of the machine is not "this graph cannot be quantized": it must not be downgraded.
    monkeypatch.setattr(shape_inference, "quant_pre_process", disk_full)
    with pytest.raises(OSError, match="no space"):
        export.quantize_static_int8(tiny_onnx[64], tmp_path / "model_int8.onnx", _images(2, 64))


def test_dinov2_exports_and_its_int8_quantization_is_refused_or_usable(tmp_path):
    """The ViT graph must export; onnxruntime's static quantization either refuses it or gives a model."""
    pytest.importorskip("timm")
    from defect_inspect.backbones import make_extractor

    ext = make_extractor("dinov2_vits14", img_size=56, pretrained=False)
    path = export.export_onnx(ext, 56, tmp_path / "model_fp32.onnx")
    parity = export.check_parity(ext, path, _images(2, 56, seed=10))
    assert parity["max_abs"] < 1e-4
    session = export._session(path)
    assert session.get_inputs()[0].shape == [1, 3, 56, 56]
    assert session.get_outputs()[0].shape == [1, 4, 4, 384]
    int8 = tmp_path / "model_int8.onnx"
    try:
        export.quantize_static_int8(path, int8, _images(4, 56, seed=11))
    except NotImplementedError as err:
        # onnxruntime 1.30: symbolic shape inference fails at the Expand of the class/register tokens.
        assert "pre-processing failed" in str(err) and not int8.exists()
    else:
        out = export._session(int8).run(["features"], {"image": inspector.preprocess(_images(1, 56)[0], 56)})
        assert out[0].shape == (1, 4, 4, 384) and np.isfinite(out[0]).all()
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".onnx" and "prep" in p.name) == []


def test_export_and_parity_leave_the_callers_module_alone(tmp_path):
    ext = TinyPatches().half()
    ext.train()
    path = export.export_onnx(ext, 64, tmp_path / "model.onnx")
    export.check_parity(ext, path, _images(1, 64))
    assert ext.training and all(p.dtype == torch.float16 for p in ext.parameters())
    # The exported model is the fp32 version of it.
    reference = TinyPatches()
    reference.load_state_dict({k: v.float() for k, v in ext.state_dict().items()})
    assert export.check_parity(reference, path, _images(2, 64, seed=3))["max_abs"] < 1e-4
    # A module that is ready is used as it is (no copy of a large backbone).
    ready = TinyPatches().eval()
    assert export._cpu_fp32(ready) is ready and export._cpu_fp32(ext) is not ext


def test_load_checks_the_meta_against_the_models_declared_shapes(tiny_onnx, tmp_path):
    ext = TinyPatches()
    feats, grid = collect_features(ext, _images(6, 64, seed=5), batch_size=4, device="cpu")
    bank = build_bank(feats, 0.5, seed=0, device="cpu").numpy()
    good = {k: v for k, v in _meta(64, grid[0]).items() if k != "version"}
    bad = {
        "img_size 96 with the 64 px model": {"img_size": 96, "grid": [12, 12]},
        "grid 4x4 with the 8x8 model": {"grid": [4, 4]},
    }
    for i, (name, change) in enumerate(bad.items()):
        art = tmp_path / f"art{i}"
        inspector.save_artifact(art, onnx_fp32=tiny_onnx[64], bank=bank, meta={**good, **change})
        with pytest.raises(ValueError, match="the model's"):
            Inspector.load(art)
            pytest.fail(f"loaded an artifact with {name}")
    art = tmp_path / "dim"
    inspector.save_artifact(art, onnx_fp32=tiny_onnx[64], bank=bank[:, :5], meta={**good, "dim": 5})
    with pytest.raises(ValueError, match="the model's output"):
        Inspector.load(art)
    # The session built by hand goes through the same check.
    with pytest.raises(ValueError, match="the model's"):
        Inspector(export._session(tiny_onnx[64]), bank, _meta(64, 4))
    # A broken model file is a ValueError too, not an onnxruntime exception.
    art = tmp_path / "broken"
    inspector.save_artifact(art, onnx_fp32=tiny_onnx[64], bank=bank, meta=good)
    (art / "model_fp32.onnx").write_bytes(tiny_onnx[64].read_bytes()[:200])
    with pytest.raises(ValueError, match="not a loadable ONNX model"):
        Inspector.load(art)


def test_wide_resnet_export_at_a_small_size(tmp_path):
    ext = WideResNetPatches(pretrained=False)
    path = export.export_onnx(ext, 64, tmp_path / "wrn.onnx")
    parity = export.check_parity(ext, path, _images(1, 64, seed=9))
    assert parity["max_abs"] < 1e-3
    session = export._session(path)
    assert session.get_outputs()[0].shape == [1, 8, 8, 1536]

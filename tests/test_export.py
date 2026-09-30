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


def test_wide_resnet_export_at_a_small_size(tmp_path):
    ext = WideResNetPatches(pretrained=False)
    path = export.export_onnx(ext, 64, tmp_path / "wrn.onnx")
    parity = export.check_parity(ext, path, _images(1, 64, seed=9))
    assert parity["max_abs"] < 1e-3
    session = export._session(path)
    assert session.get_outputs()[0].shape == [1, 8, 8, 1536]

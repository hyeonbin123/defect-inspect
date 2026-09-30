import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from defect_inspect import inspector
from defect_inspect.inspector import Inspector

GRID = 4
SIZE = 32


class FakeSession:
    """Stand-in for an onnxruntime session: the mean colour of each grid cell as a 3-d patch feature."""

    def __init__(self, grid=GRID):
        self.grid = grid
        self.calls = 0

    def run(self, names, feeds):
        assert names == ["features"]
        x = feeds["image"]
        self.calls += 1
        b, c, h, w = x.shape
        cell = h // self.grid
        out = x.reshape(b, c, self.grid, cell, self.grid, cell).mean(axis=(3, 5))
        return [np.ascontiguousarray(out.transpose(0, 2, 3, 1)).astype(np.float32)]


def _meta(**extra):
    meta = {
        "version": inspector.ARTIFACT_VERSION,
        "name": "toy",
        "category": "toy",
        "backbone": "gridmean",
        "img_size": SIZE,
        "grid": [GRID, GRID],
        "dim": 3,
        "reweight_k": 3,
        "sigma": 1.0,
        "threshold": 0.5,
    }
    meta.update(extra)
    return meta


def _images(n, seed=0, defect=False):
    rng = np.random.default_rng(seed)
    images = rng.integers(100, 140, (n, SIZE, SIZE, 3), dtype=np.uint8)
    if defect:
        images[:, 8:16, 8:16] = 255
    return images


def _bank(images):
    session = FakeSession()
    feats = [
        session.run(["features"], {"image": inspector.preprocess(im, SIZE)})[0].reshape(-1, 3)
        for im in images
    ]
    return np.concatenate(feats).astype(np.float16)


def test_preprocess_matches_the_definition():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
    x = inspector.preprocess(array, SIZE)
    assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
    mean, std = np.array(inspector.IMAGENET_MEAN), np.array(inspector.IMAGENET_STD)
    expected = ((array / 255.0 - mean) / std).transpose(2, 0, 1)
    np.testing.assert_allclose(x[0], expected, rtol=1e-5, atol=1e-6)
    # A PIL image of the right size gives the same tensor; another size is resized with BICUBIC.
    np.testing.assert_array_equal(inspector.preprocess(Image.fromarray(array), SIZE), x)
    big = Image.fromarray(rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    resized = np.asarray(big.resize((SIZE, SIZE), Image.Resampling.BICUBIC))
    np.testing.assert_array_equal(inspector.preprocess(big, SIZE), inspector.preprocess(resized, SIZE))
    grey = Image.fromarray(array[..., 0])
    assert inspector.preprocess(grey, SIZE).shape == (1, 3, SIZE, SIZE)
    with pytest.raises(ValueError):
        inspector.preprocess(array[:16], SIZE)
    with pytest.raises(ValueError):
        inspector.preprocess(array.astype(np.float32), SIZE)


def test_nearest_matches_brute_force_and_is_exact_for_bank_members():
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(200, 16)).astype(np.float32) + 5.0
    query = np.concatenate([rng.normal(size=(40, 16)).astype(np.float32) + 5.0, bank[:7]])
    centre = bank.mean(axis=0, keepdims=True)
    b, q = bank - centre, query - centre
    dist, index = inspector.nearest(q, b, np.square(b).sum(axis=1))
    brute = np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2)
    np.testing.assert_array_equal(index, brute.argmin(axis=1))
    np.testing.assert_allclose(dist, brute.min(axis=1), rtol=1e-5, atol=1e-6)
    assert np.all(dist[-7:] == 0.0)


def test_nearest_in_blocks_gives_the_same_result(monkeypatch):
    rng = np.random.default_rng(2)
    bank = rng.normal(size=(50, 8)).astype(np.float32)
    query = rng.normal(size=(33, 8)).astype(np.float32)
    sq = np.square(bank).sum(axis=1)
    whole = inspector.nearest(query, bank, sq)
    monkeypatch.setattr(inspector, "_MAX_DIST_ELEMS", 50 * 4)  # four query rows per block
    blocks = inspector.nearest(query, bank, sq)
    np.testing.assert_array_equal(whole[1], blocks[1])
    np.testing.assert_allclose(whole[0], blocks[0], rtol=1e-6)


@pytest.mark.parametrize("k", [1, 2, 9, 500])
def test_image_score_is_the_reweighted_largest_patch_score(k):
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(60, 6)).astype(np.float32)
    query = rng.normal(size=(16, 6)).astype(np.float32) * 1.5
    sq = np.square(bank).sum(axis=1)
    dist, index = inspector.nearest(query, bank, sq)
    score = inspector.image_score(dist, index, query, bank, sq, reweight_k=k)

    p = int(dist.argmax())
    if k <= 1:
        expected = dist[p]
    else:
        m = index[p]
        to_m = np.linalg.norm(bank - bank[m], axis=1)
        support = np.argsort(to_m, kind="stable")[: min(k, len(bank))]
        assert support[0] == m
        d = np.linalg.norm(query[p] - bank[support], axis=1).astype(np.float64)
        expected = (1.0 - np.exp(d[0]) / np.exp(d).sum()) * dist[p]
    assert score == pytest.approx(float(expected), rel=1e-5)


def test_image_score_puts_m_star_first_among_duplicates():
    bank = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    sq = np.square(bank).sum(axis=1)
    dist = np.array([1.0], dtype=np.float32)
    for m in (0, 1):
        score = inspector.image_score(dist, np.array([m]), query, bank, sq, reweight_k=2)
        # Support = the two duplicate rows, both at distance 1: weight 1 - 1/2.
        assert score == pytest.approx(0.5)


def test_resize_bilinear_matches_the_align_corners_false_formula():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(inspector.resize_bilinear(a[:3, :3], 3), a[:3, :3])
    np.testing.assert_allclose(inspector.resize_bilinear(np.full((5, 5), 2.5, np.float32), 17), 2.5)
    # 2 -> 4 along one axis: source positions -0.25 (clamped to 0), 0.25, 0.75, 1.25 (clamped to 1).
    up = inspector.resize_bilinear(np.array([[0.0, 4.0], [0.0, 4.0]], dtype=np.float32), 4)
    np.testing.assert_allclose(up[0], [0.0, 1.0, 3.0, 4.0])
    # Downsampling without antialiasing: 4 -> 2 samples at positions 0.5 and 2.5.
    down = inspector.resize_bilinear(np.tile(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32), (4, 1)), 2)
    np.testing.assert_allclose(down[0], [0.5, 2.5])


def test_gaussian_kernel_and_score_map():
    assert inspector.gaussian_kernel1d(4.0).shape == (33,)
    assert inspector.gaussian_kernel1d(0.4).shape == (5,)  # int(4 * 0.4 + 0.5) = 2
    assert inspector.gaussian_kernel1d(0.0).tolist() == [1.0]
    assert inspector.gaussian_kernel1d(1.0).sum() == pytest.approx(1.0)

    patch = np.zeros((4, 4), dtype=np.float32)
    patch[1, 2] = 8.0
    out = inspector.score_map(patch, SIZE, sigma=1.0, map_size=64)
    assert out.shape == (64, 64) and out.dtype == np.float32
    r, c = np.unravel_index(int(out.argmax()), out.shape)
    assert 16 <= r < 32 and 32 <= c < 48  # the hot patch's cell on the 64 grid
    np.testing.assert_allclose(
        inspector.score_map(np.full((4, 4), 3.0, np.float32), SIZE, 1.0, 64), 3.0, rtol=1e-5
    )
    unblurred = inspector.score_map(patch, SIZE, sigma=0.0, map_size=SIZE)
    np.testing.assert_allclose(unblurred, inspector.resize_bilinear(patch, SIZE))
    with pytest.raises(ValueError):
        inspector.score_map(patch, 8, sigma=4.0)


def test_inspector_scores_defects_above_normals_and_flags_them():
    bank = _bank(_images(12))
    insp = Inspector(FakeSession(), bank, _meta())
    normal = insp.scores(_images(6, seed=5))
    defect = insp.scores(_images(6, seed=6, defect=True))
    assert normal.dtype == np.float32 and defect.min() > normal.max()
    insp.set_threshold(float(normal.max()))
    result = insp.inspect(_images(1, seed=7, defect=True)[0])
    assert result.is_defect and result.score > result.threshold
    assert result.heatmap.shape == (256, 256) and result.heatmap.dtype == np.float32
    r, c = np.unravel_index(int(result.heatmap.argmax()), result.heatmap.shape)
    assert 48 <= r < 144 and 48 <= c < 144  # around the bright square (rows/cols 64..128 on the 256 grid)
    clean = insp.inspect(_images(1, seed=8)[0], heatmap=False)
    assert not clean.is_defect and clean.heatmap is None
    # A score exactly at the threshold is not a defect (rule: score > threshold).
    insp.set_threshold(clean.score)
    assert not insp.inspect(_images(1, seed=8)[0], heatmap=False).is_defect


def test_a_bank_member_scores_zero_and_features_are_rounded_to_fp16():
    images = _images(3)
    insp = Inspector(FakeSession(), _bank(images), _meta(reweight_k=1))
    np.testing.assert_array_equal(insp.scores(images), 0.0)
    feats = insp.features(images[0])
    np.testing.assert_array_equal(feats, feats.astype(np.float16).astype(np.float32))


def test_calibrate_sets_the_conformal_threshold():
    insp = Inspector(FakeSession(), _bank(_images(12)), _meta())
    normals = list(_images(40, seed=9))
    scores = np.sort(insp.scores(np.stack(normals)))
    value = insp.calibrate(normals, alpha=0.05)
    assert value == pytest.approx(float(scores[38]))  # rank ceil(41 * 0.95) = 39
    assert insp.threshold == value and insp.meta["threshold"] == value
    assert insp.meta["calibration"] == {"strategy": "holdout", "n": 40, "guaranteed": True}
    insp.calibrate(normals[:5], alpha=0.05)
    assert insp.threshold == pytest.approx(float(insp.scores(np.stack(normals[:5])).max()))
    assert insp.meta["calibration"]["guaranteed"] is False
    with pytest.raises(ValueError):
        insp.calibrate([])
    kept = insp.threshold
    for bad in (float("nan"), float("inf"), None, "0.5", True):
        with pytest.raises(ValueError):
            insp.set_threshold(bad)
    assert insp.threshold == kept and insp.meta["threshold"] == kept
    insp.set_threshold(np.float32(1.5))
    assert insp.threshold == 1.5 and isinstance(insp.meta["threshold"], float)


def test_meta_and_bank_are_validated():
    bank = _bank(_images(4))
    with pytest.raises(ValueError, match="lacks"):
        Inspector(FakeSession(), bank, {k: v for k, v in _meta().items() if k != "sigma"})
    with pytest.raises(ValueError, match="version"):
        Inspector(FakeSession(), bank, _meta(version=99))
    with pytest.raises(ValueError, match="bank shape"):
        Inspector(FakeSession(), bank[:, :2], _meta())
    with pytest.raises(ValueError, match="bank shape"):
        Inspector(FakeSession(), bank[:0], _meta())
    wrong_grid = Inspector(FakeSession(grid=2), bank, _meta())
    with pytest.raises(ValueError, match="features"):
        wrong_grid.features(_images(1)[0])


class SpikeSession(FakeSession):
    """A model whose first feature value is `value` (too large for fp16, or not finite)."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def run(self, names, feeds):
        out = super().run(names, feeds)
        out[0][0, 0, 0, 0] = self.value
        return out


def test_features_beyond_fp16_raise_like_the_torch_pipeline():
    # patchcore._patch_features checks after the cast to fp16; so does the numpy path, without warnings.
    insp = Inspector(SpikeSession(1e5), _bank(_images(4)), _meta())
    image = _images(1, seed=3)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for call in (
            lambda: insp.features(image),
            lambda: insp.inspect(image),
            lambda: insp.scores(image[None]),
            lambda: insp.calibrate([image]),
        ):
            with pytest.raises(FloatingPointError, match="overflowed fp16"):
                call()
    assert insp.threshold == 0.5  # a failed calibration leaves the threshold alone
    # The largest fp16 value still passes.
    assert np.isfinite(Inspector(SpikeSession(65504.0), _bank(_images(4)), _meta()).features(image)).all()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_features_never_give_a_verdict(value):
    insp = Inspector(SpikeSession(value), _bank(_images(4)), _meta())
    image = _images(1, seed=3)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FloatingPointError):
            insp.inspect(image)
        with pytest.raises(FloatingPointError):
            insp.scores(image[None])
        # Features handed in from elsewhere get the same treatment: NaN > threshold would read as "normal".
        feats = Inspector(FakeSession(), _bank(_images(4)), _meta()).features(image)
        feats[5, 1] = value
        with pytest.raises(FloatingPointError):
            insp.score_features(feats)


def test_inspect_scores_and_calibrate_use_the_same_float32_score():
    """The threshold comes from float32 scores; the service must judge the same float32 value."""
    insp = Inspector(FakeSession(), _bank(_images(10)), _meta(reweight_k=9))
    images = _images(40, seed=11)
    offline = insp.scores(images)
    assert offline.dtype == np.float32 and len(set(offline.tolist())) > 30
    for image, score in zip(images, offline, strict=True):
        insp.set_threshold(float(score))
        served = insp.inspect(image, heatmap=False)
        assert served.score == float(score)
        assert not served.is_defect  # rule: score > threshold, and this score is the threshold
    value = insp.calibrate(list(images), alpha=0.05)
    assert value == float(np.sort(offline)[38])  # exactly one of the float32 scores, rank ceil(41 * 0.95)
    assert value == float(np.float32(value))


BAD_META = {
    "threshold NaN": {"threshold": float("nan")},
    "threshold inf": {"threshold": float("inf")},
    "threshold None": {"threshold": None},
    "threshold text": {"threshold": "0.5"},
    "threshold bool": {"threshold": True},
    "grid one entry": {"grid": [GRID]},
    "grid three entries": {"grid": [GRID, GRID, GRID]},
    "grid zero": {"grid": [0, 0]},
    "grid negative": {"grid": [GRID, -GRID]},
    "grid fractional": {"grid": [4.5, 4]},
    "grid not a list": {"grid": GRID},
    "img_size zero": {"img_size": 0},
    "img_size text": {"img_size": str(SIZE)},
    "img_size fractional": {"img_size": 32.5},
    "dim zero": {"dim": 0},
    "dim text": {"dim": "3"},
    "reweight_k negative": {"reweight_k": -3},
    "reweight_k fractional": {"reweight_k": 2.5},
    "sigma NaN": {"sigma": float("nan")},
    "sigma negative": {"sigma": -1.0},
    "sigma text": {"sigma": "1"},
    "sigma too large for the input": {"sigma": 8.0},  # blur radius 32 on a 32 px map
    "version as text": {"version": str(inspector.ARTIFACT_VERSION)},
}


@pytest.mark.parametrize("change", BAD_META.values(), ids=BAD_META.keys())
def test_bad_meta_values_are_refused_with_a_value_error(change, tmp_path):
    bank = _bank(_images(4))
    with pytest.raises(ValueError):
        Inspector(FakeSession(), bank, _meta(**change))
    with pytest.raises(ValueError):
        inspector.check_meta(_meta(**change))
    # Nothing invalid gets written either.
    model = tmp_path / "source.onnx"
    model.write_bytes(b"x")
    with pytest.raises(ValueError):
        inspector.save_artifact(tmp_path / "art", onnx_fp32=model, bank=bank, meta=_meta(**change))
    assert not (tmp_path / "art" / "meta.json").exists()


def test_valid_meta_variants_are_accepted():
    bank = _bank(_images(4))
    for change in (
        {"threshold": 0},
        {"threshold": -1.5},
        {"sigma": 0},
        {"sigma": 7.8},  # blur radius int(31.7) = 31 < 32
        {"reweight_k": 0},
        {"grid": (GRID, GRID)},
        {"img_size": np.int64(SIZE)},
    ):
        insp = Inspector(FakeSession(), bank, _meta(**change))
        assert insp.size == SIZE and insp.grid == (GRID, GRID)
        assert isinstance(insp.threshold, float) and isinstance(insp.sigma, float)


def test_a_bank_with_non_finite_values_is_refused():
    bank = _bank(_images(4)).astype(np.float32)
    for value in (np.nan, np.inf):
        bad = bank.copy()
        bad[3, 1] = value
        with pytest.raises(ValueError, match="finite"):
            Inspector(FakeSession(), bad, _meta())
    with pytest.raises(ValueError, match="bank"):
        Inspector(FakeSession(), bank[:, 0], _meta())
    with pytest.raises(ValueError, match="bank"):
        Inspector(FakeSession(), np.array([["a", "b", "c"]]), _meta())


class DescribedSession(FakeSession):
    """A session that declares its input and output like onnxruntime does."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self._inputs, self._outputs = inputs, outputs

    def get_inputs(self):
        return [SimpleNamespace(name=n, shape=s) for n, s in self._inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=n, shape=s) for n, s in self._outputs]


def test_the_declared_model_shapes_must_fit_the_meta():
    bank = _bank(_images(4))
    image, features = ("image", [1, 3, SIZE, SIZE]), ("features", [1, GRID, GRID, 3])
    Inspector(DescribedSession([image], [features]), bank, _meta())
    # Symbolic (str) or unknown (None) dimensions are not compared.
    Inspector(
        DescribedSession([("image", ["batch", 3, SIZE, SIZE])], [("features", [None, GRID, GRID, 3])]),
        bank,
        _meta(),
    )
    bad = {
        "another input size": ([("image", [1, 3, 64, 64])], [features]),
        "another grid": ([image], [("features", [1, 8, 8, 3])]),
        "another dim": ([image], [("features", [1, GRID, GRID, 5])]),
        "a fixed batch of two": ([("image", [2, 3, SIZE, SIZE])], [("features", [2, GRID, GRID, 3])]),
        "channels last": ([("image", [1, SIZE, SIZE, 3])], [features]),
        "three dimensions": ([("image", [3, SIZE, SIZE])], [features]),
        "another input name": ([("x", [1, 3, SIZE, SIZE])], [features]),
        "another output name": ([image], [("out", [1, GRID, GRID, 3])]),
    }
    for name, (inputs, outputs) in bad.items():
        with pytest.raises(ValueError, match="model"):
            Inspector(DescribedSession(inputs, outputs), bank, _meta())
            pytest.fail(f"accepted {name}")
    with pytest.raises(ValueError, match="model"):
        inspector.check_model_io(DescribedSession([image], [("features", [1, 8, 8, 3])]), _meta())


def test_load_turns_broken_files_into_value_errors(tmp_path):
    pytest.importorskip("onnxruntime")
    art = tmp_path / "art"
    art.mkdir()
    np.save(art / "bank.npy", _bank(_images(4)))
    (art / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    (art / "model_fp32.onnx").write_bytes(b"not a real model")
    with pytest.raises(ValueError, match="ONNX"):
        Inspector.load(art)
    (art / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="meta.json"):
        Inspector.load(art)
    (art / "meta.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="meta.json"):
        Inspector.load(art)
    # The meta is checked before the model is opened: the message names the bad value, not the model.
    (art / "meta.json").write_text(json.dumps(_meta(threshold=None)), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        Inspector.load(art)
    (art / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    (art / "bank.npy").write_bytes(b"not an array")
    with pytest.raises(ValueError, match="bank.npy"):
        Inspector.load(art)


def test_save_artifact_writes_the_documented_files(tmp_path):
    bank = _bank(_images(4))
    model = tmp_path / "source.onnx"
    model.write_bytes(b"not a real model")
    meta = {k: v for k, v in _meta().items() if k != "version"}
    inspector.save_artifact(tmp_path / "art", onnx_fp32=model, bank=bank, meta=meta)
    assert sorted(p.name for p in (tmp_path / "art").iterdir()) == [
        "bank.npy",
        "meta.json",
        "model_fp32.onnx",
    ]
    saved = json.loads((tmp_path / "art" / "meta.json").read_text(encoding="utf-8"))
    assert saved["version"] == inspector.ARTIFACT_VERSION and saved["threshold"] == 0.5
    np.testing.assert_array_equal(np.load(tmp_path / "art" / "bank.npy"), bank)
    with pytest.raises(ValueError, match="lacks"):
        inspector.save_artifact(tmp_path / "bad", onnx_fp32=model, bank=bank, meta={"name": "x"})
    with pytest.raises(ValueError, match="missing"):
        Inspector.load(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="precision"):
        Inspector.load(tmp_path / "art", precision="fp8")


def test_module_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, defect_inspect.inspector; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

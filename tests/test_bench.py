import json
from types import SimpleNamespace

import numpy as np
import pytest

from defect_inspect import bench


class FakeClock:
    """Stands in for the `time` module inside bench: time only passes when a fake stage says so."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, ms):
        self.now += ms / 1e3


class FakeInspector:
    """Stage costs in ms per call: the warm-up calls are slow, one timed call is an outlier."""

    size = 32
    sigma = 0.0  # no blur: the map stage costs no fake time
    bank_rows = 7
    precision = "fp32"
    threads = 3

    def __init__(self, clock, slow_first, outlier_at=None):
        self.clock, self.slow_first, self.outlier_at = clock, slow_first, outlier_at
        self.seen = []

    def features(self, image):
        call = len(self.seen)
        self.seen.append(int(image[0, 0, 0]))
        if call < self.slow_first:
            self.clock.advance(800.0)
        elif call == self.outlier_at:
            self.clock.advance(500.0)
        else:
            self.clock.advance(4.0 + 0.125 * (call % 2))  # 4.0 or 4.125
        return np.zeros((16, 3), np.float32)

    def score_features(self, feats):
        self.clock.advance(2.0)
        return 1.0, np.ones((4, 4), np.float32)


def _images(n):
    images = np.zeros((n, 32, 32, 3), np.uint8)
    images[:, 0, 0, 0] = np.arange(n)
    return images


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(bench, "time", clock)
    return clock


def test_time_inspector_leaves_out_the_warmup_and_reports_medians(clock):
    fake = FakeInspector(clock, slow_first=5, outlier_at=9)
    timing = bench.time_inspector(fake, _images(4), warmup=5, repeats=10)
    assert len(fake.seen) == 15 and fake.seen[:6] == [0, 1, 2, 3, 0, 1]  # images are cycled
    # Timed calls 5..14: backbone 4.125, 4.0, 4.125, 4.0, 500 (call 9), 4.0, 4.125, ... -> median, not mean.
    assert timing["backbone_ms"] == pytest.approx(4.0625)
    assert timing["search_ms"] == pytest.approx(2.0)
    assert timing["map_ms"] == pytest.approx(0.0, abs=1e-9)
    assert timing["total_ms"] == pytest.approx(6.0625)
    # p95 of ten values with one outlier: between the ninth and the largest.
    assert 6.125 < timing["total_p95_ms"] < 502.0
    assert timing["backbone_p95_ms"] > 4.125 and timing["search_p95_ms"] == pytest.approx(2.0)
    assert timing["repeats"] == 10 and timing["bank_rows"] == 7
    assert timing["precision"] == "fp32" and timing["threads"] == 3
    assert {"backbone_ms", "search_ms", "map_ms", "total_ms", "total_p95_ms", "threads", "bank_rows"} <= set(
        timing
    )


def test_time_inspector_without_warmup_times_every_call(clock):
    fake = FakeInspector(clock, slow_first=1)
    timing = bench.time_inspector(fake, _images(2), warmup=0, repeats=3)
    assert len(fake.seen) == 3
    assert timing["total_p95_ms"] > 700.0  # the slow first call is in the sample now
    with pytest.raises(ValueError):
        bench.time_inspector(fake, _images(0))
    with pytest.raises(ValueError):
        bench.time_inspector(fake, _images(2), repeats=0)


def test_time_torch_prepares_the_bank_once_and_leaves_out_the_warmup(clock, monkeypatch):
    pytest.importorskip("torch")
    made = []

    class FakeScorer:
        def __init__(self, extractor, bank, *, device, **kwargs):
            clock.advance(300.0)  # the one-off bank upload and centring
            self.calls = 0
            self.kwargs = kwargs
            made.append(self)

        def __call__(self, image):
            assert image.shape == (32, 32, 3)
            clock.advance(900.0 if self.calls < 3 else (5.0 if self.calls != 6 else 70.0))
            self.calls += 1

    monkeypatch.setattr(bench, "_TorchScorer", FakeScorer)
    bank = np.zeros((11, 4), np.float16)
    timing = bench.time_torch(
        None, bank, _images(4), device="cpu", warmup=3, repeats=8, reweight_k=3, sigma=1.0
    )
    assert len(made) == 1 and made[0].calls == 11  # one scorer for every image: the bank is set up once
    assert made[0].kwargs == {"reweight_k": 3, "sigma": 1.0}
    assert timing["total_ms"] == pytest.approx(5.0)  # the 300 ms set-up and the warm-up calls are not in it
    assert 5.0 < timing["total_p95_ms"] < 70.0
    assert timing["bank_setup_ms"] == pytest.approx(300.0)
    assert timing["repeats"] == 8 and timing["bank_rows"] == 11 and timing["device"] == "cpu"
    with pytest.raises(ValueError):
        bench.time_torch(None, bank, _images(0), device="cpu")


def test_the_torch_scorer_is_score_images_with_a_prepared_bank():
    torch = pytest.importorskip("torch")
    from defect_inspect.patchcore import build_bank, collect_features, score_images

    class Tiny(torch.nn.Module):
        dim = 6

        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.conv = torch.nn.Conv2d(3, 6, 3, stride=8, padding=1)

        def forward(self, x):
            return self.conv(x.float()).permute(0, 2, 3, 1).contiguous()

    rng = np.random.default_rng(0)
    extractor = Tiny()
    for size, map_size in ((64, 64), (64, 256)):
        train = rng.integers(60, 200, (6, size, size, 3), dtype=np.uint8)
        images = rng.integers(0, 256, (4, size, size, 3), dtype=np.uint8)
        feats, _ = collect_features(extractor, train, batch_size=4, device="cpu")
        bank = build_bank(feats, 0.25, seed=0, device="cpu")
        kwargs = {"reweight_k": 3, "sigma": 1.0, "map_size": map_size}
        ref = score_images(extractor, bank, images, batch_size=1, device="cpu", **kwargs)
        before = bank.clone()
        scorer = bench._TorchScorer(extractor, bank, device="cpu", **kwargs)
        for i, image in enumerate(images):
            score, heat = scorer(image)
            assert score == float(ref.image_scores[i])
            assert heat.dtype == np.float16 and heat.shape == (map_size, map_size)
            np.testing.assert_array_equal(heat, ref.maps[i])
        assert torch.equal(bank, before)  # the caller's bank is not centred in place
    with pytest.raises(ValueError, match="dim"):
        bench._TorchScorer(extractor, torch.zeros((5, 4), dtype=torch.float16), device="cpu")(images[0])
    with pytest.raises(ValueError, match="bank"):
        bench._TorchScorer(extractor, torch.zeros((0, 6), dtype=torch.float16), device="cpu")

    # The real clock: a run on the CPU returns the documented keys.
    timing = bench.time_torch(
        extractor, bank, images, device="cpu", warmup=1, repeats=3, reweight_k=3, sigma=1.0
    )
    assert timing["total_ms"] > 0 and timing["total_p95_ms"] >= timing["total_ms"]
    assert timing["bank_setup_ms"] > 0 and timing["bank_rows"] == bank.shape[0]


def test_latency_file_accumulates_entries(tmp_path):
    path = tmp_path / "stage4" / "latency.json"
    bench.merge_into(path, "a", {"cpu_fp32": {"total_ms": 1.0}})
    data = bench.merge_into(path, "a", {"cpu_int8": {"total_ms": 2.0}})
    data = bench.merge_into(path, "b", {"cpu_fp32": {"total_ms": 3.0}})
    assert set(data) == {"a", "b"} and set(data["a"]) == {"cpu_fp32", "cpu_int8"}
    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_time_resize_and_cpu_name():
    assert bench.RESIZE_SOURCE == (1500, 1000)  # the registered photo size (width, height)
    assert bench.time_resize(64, source=(200, 100), repeats=2) > 0
    assert isinstance(bench.cpu_name(), str)


def test_cli_adds_the_resize_time_to_the_cpu_latency(tmp_path, monkeypatch):
    """The registered CPU latency is resize + inference; the entry must carry that sum."""
    from defect_inspect import paths
    from defect_inspect.splits import ManifestRow

    manifest = [
        ManifestRow(f"toy/n{i}.JPG", "", "toy", "normal", "pool_normal", i % 5, "") for i in range(20)
    ]
    manifest += [ManifestRow(f"toy/t{i}.JPG", "", "toy", "normal", "test_normal", -1, "") for i in range(5)]
    read = []

    class FakeCache:
        def __init__(self, out_dir, size):
            assert size == 32

        def images(self, rows):
            read.append([r.image for r in rows])
            return _images(len(rows))

    loaded = []

    def fake_load(art_dir, *, precision, threads):
        loaded.append((art_dir.name, precision, threads))
        return SimpleNamespace(precision=precision)

    def fake_time_inspector(inspector, images, *, repeats):
        assert repeats == len(images) == 6
        return {
            "total_ms": 195.0 if inspector.precision == "fp32" else 60.0,
            "precision": inspector.precision,
        }

    monkeypatch.setattr("defect_inspect.cache.ImageCache", FakeCache)
    monkeypatch.setattr("defect_inspect.splits.read_manifest", lambda path: manifest)
    monkeypatch.setattr(bench.Inspector, "load", staticmethod(fake_load))
    monkeypatch.setattr(bench, "time_inspector", fake_time_inspector)
    monkeypatch.setattr(bench, "time_resize", lambda size: 12.0)
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    art = tmp_path / "artifacts" / "tiny"
    (art / "toy").mkdir(parents=True)
    (art / "toy" / "meta.json").write_text('{"img_size": 32}', encoding="utf-8")

    args = ["--artifacts", str(art), "--category", "toy", "--precision", "fp32", "int8", "--images", "6"]
    bench.main([*args, "--key", "wrn50-32-r0.01", "--threads", "2"])
    data = json.loads((tmp_path / "reports" / "stage4" / "latency.json").read_text(encoding="utf-8"))
    entry = data["wrn50-32-r0.01"]
    assert entry["resize_ms"] == 12.0 and entry["resize_source"] == [1500, 1000]
    assert entry["cpu_fp32"]["total_ms"] == 195.0
    assert entry["cpu_fp32"]["total_with_resize_ms"] == 207.0  # over a 200 ms budget although 195 is not
    assert entry["cpu_int8"]["total_with_resize_ms"] == 72.0
    assert entry["threads"] == 2 and entry["category"] == "toy" and "gpu_fp16" not in entry
    assert loaded == [("toy", "fp32", 2), ("toy", "int8", 2)]
    # Timed on the first dev pool normals of the category (folds 1-4): nothing from fold 0 or the test set.
    dev_pool = [r.image for r in manifest if r.role == "pool_normal" and r.fold != 0]
    assert read == [dev_pool[:6]]

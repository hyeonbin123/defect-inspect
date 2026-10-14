# [파일 역할] 검사 경로의 장당 지연을 재는 `defect_inspect.bench`를 검증하는 테스트 묶음이다. CPU onnxruntime
# 경로(time_inspector), GPU torch 경로(time_torch, _TorchScorer), 원본 사진 리사이즈 시간, 지연 리포트
# 누적(merge_into), CLI가 쓰는 '리사이즈 + 추론' 합계를 다룬다.
# [왜 필요한가] 4단계와 6단계의 핵심 판정(CPU 장당 200ms 이하)이 이 숫자로 정해진다. 규칙은 지연을 '워밍업 뒤
# 반복의 중앙값'으로, CPU 지연은 1500×1000 사진의 리사이즈 시간을 더한 값으로 정했다(docs/experiments.md 4단계
# 격자, 규칙 변경 5에서 리사이즈를 빼던 코드를 고침). 워밍업이 섞이거나 평균을 쓰거나 리사이즈를 빼면 예산
# 근처 구성의 판정이 바뀐다.
# [설계] 실제 시간 대신 FakeClock으로 bench 모듈의 time을 바꿔, 가짜 검사기가 '이 단계는 몇 ms 걸렸다'고
# 시계를 직접 움직인다. 그래서 중앙값·p95·워밍업 제외를 정확한 수치로 단언할 수 있다.

import json
from types import SimpleNamespace

import numpy as np
import pytest

from defect_inspect import bench


# bench 안의 time 모듈 대신 쓰는 가짜 시계. perf_counter는 현재 값만 돌려주고, 시간은 가짜 단계가
# advance(ms)를 부를 때만 흐른다.
class FakeClock:
    """bench 안의 `time` 모듈 대역: 시간은 가짜 단계가 흐르게 할 때만 흐른다."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, ms):
        self.now += ms / 1e3


# CPU 경로 Inspector 흉내. 처음 slow_first번 호출은 800ms(워밍업이 느린 상황), outlier_at번째 호출은 500ms
# 이상치, 나머지 특징 단계는 4.0 또는 4.125ms, 탐색 단계는 늘 2ms가 걸린 것처럼 시계를 움직인다. seen에는
# 처리한 이미지 번호가 쌓여 이미지를 돌려 쓰는 순서를 확인할 수 있다. sigma 0이면 맵 단계(score_map)는 실제로
# 계산하지만 가짜 시간이 흐르지 않는다.
class FakeInspector:
    """호출마다 단계별 비용(ms): 워밍업 호출은 느리고, 잰 호출 하나는 이상치다."""

    size = 32
    sigma = 0.0  # 흐림 없음: 맵 단계에는 가짜 시간이 들지 않는다
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
            self.clock.advance(4.0 + 0.125 * (call % 2))  # 4.0 또는 4.125
        return np.zeros((16, 3), np.float32)

    def score_features(self, feats):
        self.clock.advance(2.0)
        return 1.0, np.ones((4, 4), np.float32)


# 왼쪽 위 픽셀 값에 이미지 번호를 넣은 빈 이미지 n장(FakeInspector가 몇 번째 이미지인지 읽는다).
def _images(n):
    images = np.zeros((n, 32, 32, 3), np.uint8)
    images[:, 0, 0, 0] = np.arange(n)
    return images


# monkeypatch로 bench.time을 FakeClock으로 바꾸는 fixture. 테스트가 끝나면 원래 time 모듈로 돌아간다.
@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(bench, "time", clock)
    return clock


# 확인: 워밍업 5회(800ms씩)는 재지 않고, 이미지 4장을 돌려 쓰며 10회를 잰다. 단계별 값은 평균이 아니라
# 중앙값이라 500ms 이상치 하나가 결과를 끌어올리지 않는다(특징 4.0625, 탐색 2.0, 맵 0, 합계 6.0625ms). p95는
# 이상치를 반영하고, 반복 수·뱅크 행 수·정밀도·스레드도 기록된다.
# 왜: 지연 규칙은 '워밍업 5회 뒤 50회의 중앙값'이다(6단계 관문 1, 4단계와 같은 방법). 첫 호출의 세션 준비
# 비용이나 순간적인 방해가 판정 값에 섞이면 안 된다.
def test_time_inspector_leaves_out_the_warmup_and_reports_medians(clock):
    fake = FakeInspector(clock, slow_first=5, outlier_at=9)
    timing = bench.time_inspector(fake, _images(4), warmup=5, repeats=10)
    assert len(fake.seen) == 15 and fake.seen[:6] == [0, 1, 2, 3, 0, 1]  # 이미지를 돌려 쓴다
    # 잰 호출 5..14: 백본 4.125, 4.0, 4.125, 4.0, 500(호출 9), 4.0, 4.125, ... -> 평균이 아니라 중앙값.
    assert timing["backbone_ms"] == pytest.approx(4.0625)
    assert timing["search_ms"] == pytest.approx(2.0)
    assert timing["map_ms"] == pytest.approx(0.0, abs=1e-9)
    assert timing["total_ms"] == pytest.approx(6.0625)
    # 이상치 하나가 낀 값 열 개의 p95: 아홉째 값과 가장 큰 값 사이.
    assert 6.125 < timing["total_p95_ms"] < 502.0
    assert timing["backbone_p95_ms"] > 4.125 and timing["search_p95_ms"] == pytest.approx(2.0)
    assert timing["repeats"] == 10 and timing["bank_rows"] == 7
    assert timing["precision"] == "fp32" and timing["threads"] == 3
    assert {"backbone_ms", "search_ms", "map_ms", "total_ms", "total_p95_ms", "threads", "bank_rows"} <= set(
        timing
    )


# 확인: warmup=0이면 느린 첫 호출까지 표본에 들어가 p95가 700ms를 넘는다(워밍업이 실제로 빠지는지의 대조).
# 이미지가 0장이거나 반복이 0이면 ValueError다.
def test_time_inspector_without_warmup_times_every_call(clock):
    fake = FakeInspector(clock, slow_first=1)
    timing = bench.time_inspector(fake, _images(2), warmup=0, repeats=3)
    assert len(fake.seen) == 3
    assert timing["total_p95_ms"] > 700.0  # 이번에는 느린 첫 호출이 표본에 들어 있다
    with pytest.raises(ValueError):
        bench.time_inspector(fake, _images(0))
    with pytest.raises(ValueError):
        bench.time_inspector(fake, _images(2), repeats=0)


# 확인: GPU torch 경로 측정은 뱅크를 장치로 옮기고 중심화하는 준비를 한 번만 하고(300ms, bank_setup_ms로 따로
# 기록), 워밍업 3회를 뺀 8회의 중앙값(5ms)을 total_ms로 낸다. reweight_k·sigma 인자는 그대로 scorer에
# 전달된다.
# 왜: patchcore.score_images는 호출마다 뱅크를 옮기고 중심화하므로 그대로 장당으로 재면 서비스에서는 한 번만
# 드는 비용을 매 장에 물리게 된다(_TorchScorer docstring). monkeypatch로 _TorchScorer를 가짜로 바꿔 시간만
# 본다.
def test_time_torch_prepares_the_bank_once_and_leaves_out_the_warmup(clock, monkeypatch):
    pytest.importorskip("torch")
    made = []

    class FakeScorer:
        def __init__(self, extractor, bank, *, device, **kwargs):
            clock.advance(300.0)  # 한 번뿐인 뱅크 업로드와 중심화
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
    assert len(made) == 1 and made[0].calls == 11  # 모든 이미지에 채점기 하나: 뱅크 준비는 한 번
    assert made[0].kwargs == {"reweight_k": 3, "sigma": 1.0}
    assert timing["total_ms"] == pytest.approx(5.0)  # 300ms 준비와 워밍업 호출은 들어가지 않는다
    assert 5.0 < timing["total_p95_ms"] < 70.0
    assert timing["bank_setup_ms"] == pytest.approx(300.0)
    assert timing["repeats"] == 8 and timing["bank_rows"] == 11 and timing["device"] == "cpu"
    with pytest.raises(ValueError):
        bench.time_torch(None, bank, _images(0), device="cpu")


# 확인: 장당 채점기 _TorchScorer의 이미지 점수와 fp16 맵이 patchcore.score_images와 비트 단위로 같다(맵 크기
# 64와 256 둘 다). 호출자의 뱅크를 제자리에서 중심화하지 않고, 특징 차원이 뱅크와 다르거나 뱅크가 비면
# ValueError다. 마지막으로 실제 시계로 CPU에서 돌려 문서에 적힌 키가 나오는지 본다.
# 왜: 지연을 잰 경로가 정확도를 잰 경로와 다른 계산이면 지연 수치가 그 구성의 것이 아니게 된다.
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
        assert torch.equal(bank, before)  # 호출자의 뱅크를 제자리에서 중심화하지 않는다
    with pytest.raises(ValueError, match="dim"):
        bench._TorchScorer(extractor, torch.zeros((5, 4), dtype=torch.float16), device="cpu")(images[0])
    with pytest.raises(ValueError, match="bank"):
        bench._TorchScorer(extractor, torch.zeros((0, 6), dtype=torch.float16), device="cpu")

    # 실제 시계: CPU에서 돌리면 문서에 적힌 키가 나온다.
    timing = bench.time_torch(
        extractor, bank, images, device="cpu", warmup=1, repeats=3, reweight_k=3, sigma=1.0
    )
    assert timing["total_ms"] > 0 and timing["total_p95_ms"] >= timing["total_ms"]
    assert timing["bank_setup_ms"] > 0 and timing["bank_rows"] == bank.shape[0]


# 확인: 지연 리포트에 같은 키로 여러 번 쓰면 정밀도별 항목(cpu_fp32, cpu_int8)이 합쳐지고, 다른 키는 따로
# 쌓이며, 파일은 줄바꿈으로 끝난다. FP32와 INT8, 구성 15개를 따로 재도 한 파일(reports/stage4/latency.json)에
# 모인다.
def test_latency_file_accumulates_entries(tmp_path):
    path = tmp_path / "stage4" / "latency.json"
    bench.merge_into(path, "a", {"cpu_fp32": {"total_ms": 1.0}})
    data = bench.merge_into(path, "a", {"cpu_int8": {"total_ms": 2.0}})
    data = bench.merge_into(path, "b", {"cpu_fp32": {"total_ms": 3.0}})
    assert set(data) == {"a", "b"} and set(data["a"]) == {"cpu_fp32", "cpu_int8"}
    assert path.read_text(encoding="utf-8").endswith("}\n")


# 확인: 리사이즈 원본 크기가 등록한 1500×1000(가로, 세로)이고, 리사이즈 시간이 양수이며, CPU 이름을 문자열로
# 얻는다(Windows에서는 레지스트리의 ProcessorNameString).
def test_time_resize_and_cpu_name():
    assert bench.RESIZE_SOURCE == (1500, 1000)  # 등록한 사진 크기(가로, 세로)
    assert bench.time_resize(64, source=(200, 100), repeats=2) > 0
    assert isinstance(bench.cpu_name(), str)


# 확인: CLI가 정밀도마다 total_with_resize_ms = 리사이즈(12ms) + 추론 중앙값을 기록한다. 추론만 195ms인 구성도
# 합계 207ms라 200ms 예산을 넘는다. 스레드 설정이 Inspector.load에 전달되고, 지연용 이미지는 그 범주 dev 풀(겹
# 1~4) 정상의 앞 N장이며 겹 0과 봉인 테스트는 읽지 않는다.
# monkeypatch: bench.main은 ImageCache와 read_manifest를 함수 안에서 import하므로 원래
# 모듈(defect_inspect.cache, defect_inspect.splits)의 이름을 바꿔야 가짜가 쓰인다. 출력 경로(paths.REPORTS)도
# tmp_path로 돌린다.
def test_cli_adds_the_resize_time_to_the_cpu_latency(tmp_path, monkeypatch):
    """등록한 CPU 지연은 리사이즈 + 추론이다. 항목에 그 합이 들어 있어야 한다."""
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
    assert entry["cpu_fp32"]["total_with_resize_ms"] == 207.0  # 195는 아니지만 합계는 200ms 예산을 넘는다
    assert entry["cpu_int8"]["total_with_resize_ms"] == 72.0
    assert entry["threads"] == 2 and entry["category"] == "toy" and "gpu_fp16" not in entry
    assert loaded == [("toy", "fp32", 2), ("toy", "int8", 2)]
    # 범주의 dev 풀 정상(겹 1-4) 앞쪽으로 잰다: 겹 0이나 봉인 테스트 세트에서는 아무것도 읽지 않는다.
    dev_pool = [r.image for r in manifest if r.role == "pool_normal" and r.fold != 0]
    assert read == [dev_pool[:6]]

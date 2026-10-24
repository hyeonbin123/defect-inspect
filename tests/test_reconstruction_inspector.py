# [파일 역할] 6단계에서 서빙 모델로 바꾼 Dinomaly ViT-S용 검사기
# ReconstructionInspector(src/defect_inspect/inspector.py)와, 그 지연을 재는 bench.time_reconstruction·bench
# CLI의 reconstruction 경로를 검증한다.
# [왜 필요한가] Dinomaly는 메모리 뱅크 없이 ONNX 모델 하나가 이미지 점수와 이상 맵을 직접 내고, 12개 범주가
# 모델 하나를 같이 쓰며 범주 폴더에는 임계값만 있다(README 검사 서비스 절). 그래서 PatchCore Inspector와
# 검증할 것이 다르다: meta의 kind, 모델 입출력 이름(score, map), 비유한 점수·맵 차단, 같은 보정 규칙, 그리고
# bench가 이 아티팩트를 PatchCore와 다른 방식으로 재는지다.
# [주요 구성] 앞부분은 가짜 세션으로 검사기 자체를, '# ---- bench' 아래는 가짜 시계·가짜 검사기로 지연 측정
# 코드를 시험한다. 실제 모델·데이터·GPU는 쓰지 않는다.

import json
from types import SimpleNamespace

import numpy as np
import pytest

# bench는 뒤쪽 지연 측정 테스트에서 time 모듈과 적재 함수를 monkeypatch로 바꾸려고 모듈째 가져온다.
from defect_inspect import bench, inspector
from defect_inspect.inspector import ReconstructionInspector

# 가짜 모델의 입력 한 변(28px)과 맵 한 변(16). 실제 서빙 모델(dms-280-car)은 280px이지만 테스트는 작게 잡아
# 빨리 돈다.
SIZE = 28
MAP = 16


# 내보낸 Dinomaly 그래프를 흉내 내는 가짜 세션. 입력이 [1, 3, S, S] float32인지 단언하고, 채널 평균 밝기를
# 맵으로, 그 최댓값을 이미지 점수로 돌려준다. score_value를 주면 점수를 그 값으로 고정해 NaN 같은 깨진 출력을
# 시험할 수 있고, map_poison을 주면 점수는 깨지기 전 맵에서 그대로 구한 채 맵의 한 칸만 그 값(NaN·inf)으로
# 바꿔 맵만 깨진 출력을 시험할 수 있다.
class FakeSession:
    """내보낸 Dinomaly 그래프 대신 쓰는 가짜 세션: 맵은 평균 밝기, 점수는 그 최댓값이다."""

    def __init__(self, score_value=None, map_poison=None):
        self.calls = 0
        self.score_value = score_value
        self.map_poison = map_poison

    def run(self, names, feeds):
        assert names == ["score", "map"]
        x = feeds["image"]
        assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
        self.calls += 1
        brightness = x.mean(axis=1)  # [1, S, S]
        # SIZE // MAP = 1이라 간격 1로 뽑은 뒤 왼쪽 위 MAP×MAP을 잘라 맵으로 쓴다. 다음 줄의 np.pad는 맵이
        # MAP보다 작게 나올 때 0으로 채워 출력 모양을 [1, MAP, MAP]으로 맞추는 안전장치다.
        amap = brightness[:, :: SIZE // MAP + 0, :: SIZE // MAP + 0][:, :MAP, :MAP]
        amap = np.ascontiguousarray(
            np.pad(amap, ((0, 0), (0, MAP - amap.shape[1]), (0, MAP - amap.shape[2])))
        )
        score = amap.reshape(1, -1).max(axis=1) if self.score_value is None else np.array([self.score_value])
        if self.map_poison is not None:
            amap = amap.copy()
            amap[0, 0, 0] = self.map_poison  # 점수는 위에서 깨지기 전 맵으로 이미 구했다
        return [score.astype(np.float32), amap.astype(np.float32)]


# check_reconstruction_meta를 통과하는 기본 meta(RECONSTRUCTION_META_KEYS 전부). kind가 "reconstruction"이어야
# 하고, PatchCore meta와 달리 grid·dim·뱅크 관련 키가 없고 map_size가 있다.
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


# 회색 잡음 uint8 이미지. defect=True면 4~12px 구역을 흰색으로 칠한다. 이 구역은 가짜 맵(왼쪽 위 16×16) 안에
# 들어가므로 점수가 올라간다.
def _images(n, seed=0, defect=False):
    rng = np.random.default_rng(seed)
    images = rng.integers(100, 140, (n, SIZE, SIZE, 3), dtype=np.uint8)
    if defect:
        images[:, 4:12, 4:12] = 255
    return images


# 검사기의 기본 동작을 한 번에 확인한다. bank_rows = 0과 kind는 서비스의 /categories 응답(service.describe)에
# 그대로 나가는 값이다. 결함 이미지가 정상보다 점수가 높고 임계값(0.5)을 넘는지, 히트맵이 [MAP, MAP]
# float32인지, heatmap=False면 None인지, scores가 run과 같은 float32 점수를 내는지 본다.
# 보정은 PatchCore Inspector와 같은 split-conformal 규칙이어야 한다: n = 40, α = 0.05면 39번째 작은 점수이고
# meta에 보정 기록이 남는다. 빈 목록과 NaN 임계값은 거절한다.
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
    expected = np.sort(insp.scores(np.stack(normals)))[38]  # 40개 중 순위 ceil(41 * 0.95) = 39
    assert value == insp.threshold == insp.meta["threshold"] == float(expected)
    assert insp.meta["calibration"] == {"strategy": "holdout", "n": 40, "guaranteed": True}
    with pytest.raises(ValueError):
        insp.calibrate([])
    with pytest.raises(ValueError):
        insp.set_threshold(float("nan"))


# 모델이 NaN 점수를 내면 판정 대신 FloatingPointError가 나는지 본다. NaN > threshold는 False라 막지 않으면
# '정상'으로 통과하기 때문이다(ReconstructionInspector docstring).
def test_a_non_finite_score_never_gives_a_verdict():
    insp = ReconstructionInspector(FakeSession(score_value=float("nan")), _meta())
    with pytest.raises(FloatingPointError):
        insp.inspect(_images(1)[0])


# 모델이 유한하지 않은 맵을 내면(점수는 유한해도) 판정하지 않고 FloatingPointError를 낸다. 연구 경로
# (run_dinomaly.predict)가 점수와 맵을 함께 검사하는 것과 같은 규칙이다. 막지 않으면 NaN 맵이 그대로
# 나가 서비스 히트맵에 점수 0(검정)으로 조용히 그려지고, 맵이 깨진 실행의 점수를 믿게 된다.
# run·inspect(heatmap=False 포함)·scores·calibrate 어느 길로도 값이 나오지 않아야 하고, 실패한 보정은
# 임계값을 건드리지 않아야 한다.
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_map_never_gives_a_verdict(poison):
    insp = ReconstructionInspector(FakeSession(map_poison=poison), _meta())
    with pytest.raises(FloatingPointError, match="map"):
        insp.run(_images(1)[0])
    with pytest.raises(FloatingPointError):
        insp.inspect(_images(1)[0], heatmap=False)
    with pytest.raises(FloatingPointError):
        insp.scores(_images(2))
    with pytest.raises(FloatingPointError):
        insp.calibrate(list(_images(3)))
    assert insp.threshold == 0.5 and insp.meta["threshold"] == 0.5


# 잘못된 meta(PatchCore kind, 다른 버전, 0 크기, 소수 map_size, inf·None 임계값)를 생성자가 거절하는지, 키가
# 빠지면 'lacks' 메시지가 나오는지 본다. 특히 kind 검사는 PatchCore 아티팩트를 이 클래스로 잘못 읽는 일을
# 막는다.
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


# 모델이 선언한 입력 image [1, 3, S, S]와 출력 score [1], map [1, M, M]이 meta와 맞는지 생성 시점에 검사하는지
# 본다. 심볼릭 배치 차원("b")은 비교하지 않는다. 다른 입력 크기, 다른 맵 크기, score 출력 없음, 입력 이름이
# 다름은 모두 ValueError여야 한다. 모델 하나를 범주들이 같이 쓰므로 다른 크기로 내보낸 모델을 잘못 짝짓는
# 경우를 적재할 때 잡는다.
def test_the_declared_model_shapes_must_fit_the_meta():
    # onnxruntime의 입출력 노드처럼 name과 shape 속성만 가진 객체를 만든다.
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


# artifact_kind가 meta.json의 kind를 읽고, kind가 없는 옛 아티팩트는 patchcore로 보는지 본다.
# 서비스(load_inspectors)가 이 값으로 범주 폴더마다 ReconstructionInspector와 PatchCore Inspector 중 하나를
# 고르므로, 4단계 PatchCore 아티팩트를 그대로 계속 읽으려면 기본값이 patchcore여야 한다. meta.json이 없으면
# ValueError다.
def test_artifact_kind(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "meta.json").write_text('{"img_size": 32}', encoding="utf-8")
    assert inspector.artifact_kind(tmp_path / "a") == "reconstruction"
    assert inspector.artifact_kind(tmp_path / "b") == "patchcore"
    with pytest.raises(ValueError):
        inspector.artifact_kind(tmp_path / "missing")


# load가 파일 문제를 모두 ValueError로 바꾸는지 본다: meta.json이 없을 때(missing), 모델 파일이 범주 폴더가
# 아니라 부모 폴더(아티팩트 묶음)에 있지만 ONNX가 아닐 때, PatchCore용 정밀도 이름(int8)을 줬을 때. 두 번째
# 경우는 모델을 부모 폴더에서 찾는다는 것도 함께 보여 준다. onnxruntime이 없으면 건너뛴다.
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


# ---------------------------------------------------------------- bench(지연 측정)


# bench가 쓰는 time.perf_counter를 대신하는 가짜 시계. 실제 시간을 쓰지 않고 now 값을 테스트가 직접 움직여
# 지연 통계를 정확히 맞출 수 있게 한다.
class FakeClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


# 호출될 때마다 가짜 시계를 정해진 만큼 앞으로 돌리는 가짜 검사기. 처음 5번은 900ms(예열), 그 뒤는 150ms와
# 151ms를 번갈아 쓴다. seen에 이미지 첫 화소값을 남겨 이미지를 어떤 순서로 돌려 썼는지 확인한다.
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


# time_reconstruction이 예열 5회를 빼고 반복 10회의 중앙값을 total_ms로 내는지 본다. 예열의 900ms가 섞이면
# 중앙값이 달라지므로, 결과가 150.5ms면 예열이 제외된 것이다. 이미지 3장을 0, 1, 2, 0 순으로 돌려 쓰는지,
# kind·precision이 기록되는지, 이미지가 없으면 ValueError인지도 확인한다. monkeypatch로 bench 모듈의 time을
# 가짜 시계로 바꿔 실제로 기다리지 않는다.
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


# bench CLI가 reconstruction 아티팩트(meta의 kind)를 알아보고 PatchCore와 다른 경로로 재는지 본다.
# 캐시·매니페스트·적재·시간 측정 함수를 모두 가짜로 바꿔 데이터와 모델 없이 돌린다.
# 확인하는 것: 정밀도마다 total_with_resize_ms = 리사이즈(9ms) + 추론이 기록되는지(지연 예산은 원본 사진
# 리사이즈를 포함해 판정한다, bench docstring), meta의 source(학습 전 여부)가 리포트에 옮겨지는지, 이미지는
# dev 정상 풀(겹 0을 뺀 겹 1~4)의 앞 6장만 읽는지(봉인 테스트를 읽지 않음). 마지막으로 PatchCore용
# 정밀도(int8)와 --gpu(PatchCore torch 경로 전용)는 parser.error로 거절해야 한다.
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

    # ReconstructionInspector.load 대신 들어가 호출 인자만 기록하고 precision·threads만 가진 객체를 돌려준다.
    def fake_load(art_dir, *, precision, threads):
        loaded.append((art_dir.name, precision, threads))
        return SimpleNamespace(precision=precision, threads=threads)

    # time_reconstruction 대신 들어가는 함수. 반복 횟수가 --images와 같은지 단언하고 정밀도별로 정해 둔
    # 지연(fp32 180ms, 그 밖 120ms)을 돌려준다.
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

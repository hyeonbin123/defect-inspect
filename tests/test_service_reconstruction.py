"""복원 방식(Dinomaly) 아티팩트 세트를 쓰는 서비스: 모델이 스스로 점수를 내고, 뱅크가 없다."""

# [파일 역할] 검사 서비스가 복원 방식(Dinomaly) 아티팩트를 읽고 서빙하는 경로를 검증한다. 6단계 모델
# dms-280-car로 서비스를 바꾼 뒤(2026-10-04, docs/experiments.md 서비스 모델 교체 점검) 새로 생긴 경로라서
# test_service.py(오프라인 PatchCore 흉내)와 따로 둔다.
# [무엇을 확인하나] /categories가 모델 종류(reconstruction)와 이름을 알리는지, /inspect가 모델 자신의 점수와
# 맵을 쓰는지, 모델이 유한하지 않은 맵을 내면 검정 히트맵 대신 500으로 멈추는지, /calibrate가 임계값만 바꾸고
# 되돌릴 수 있는지, load_inspectors가 범주 폴더(meta.json만, 뱅크 없음)와 상위 폴더의 공용 ONNX 하나를 읽어
# 세션을 나눠 쓰는지, 정밀도별 모델 파일과 잘못된 meta를 어떻게 거절하는지, 시작할 때 대조한 모델 지문이
# /categories의 model_sha256(앞 12자)으로 보이는지, 실제 작은 ONNX 모델로 끝에서 끝까지 도는지(이 길은 시작
# 예열도 실제 세션으로 거친다).
# [설계 메모] 대부분은 ONNX 대신 ReconSession(밝기를 256으로 키운 맵, 그 최댓값이 점수)으로 돌아 onnxruntime
# 없이도 의미가 있다. 마지막 테스트만 onnx·onnxruntime이 있을 때 실제 모델을 만든다.

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# serve 의존성 그룹(FastAPI, python-multipart)이 없는 환경에서는 파일 전체를 건너뛴다.
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

# TestClient로 서버 없이 앱을 부르고, conformal_threshold로 보정 기대값을 직접 계산하며,
# ReconstructionInspector.load로 서비스 로더의 결과와 대조한다.
from fastapi.testclient import TestClient  # noqa: E402

from defect_inspect import service  # noqa: E402
from defect_inspect.calibrate import conformal_threshold  # noqa: E402
from defect_inspect.inspector import ReconstructionInspector  # noqa: E402

# 요청 주소(기본 허용 Host), 관리자 토큰, 모델 입력 크기 32px, 맵 크기 256(서빙 모델의 맵 크기와 같은 값).
HOST = "http://127.0.0.1"
TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}
SIZE = 32
MAP = 256


# 내보낸 Dinomaly 그래프 대신 쓰는 가짜 세션. 서비스가 부르는 출력 이름(score, map)과 입력 모양·dtype을
# assert로 확인하고, 채널 평균 밝기를 256으로 늘린 맵과 그 최댓값을 점수로 돌려준다. 밝은 결함 칸이 그대로
# 높은 점수가 된다.
class ReconSession:
    """내보낸 Dinomaly 그래프의 대역: 맵은 밝기를 256으로 키운 것이고, 점수는 그 최댓값이다."""

    def run(self, names, feeds):
        assert names == ["score", "map"]
        x = feeds["image"]
        assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
        bright = x.mean(axis=1)
        amap = np.repeat(np.repeat(bright, MAP // SIZE, axis=1), MAP // SIZE, axis=2)
        return [amap.reshape(1, -1).max(axis=1).astype(np.float32), amap.astype(np.float32)]


# 복원 방식 범주 폴더의 meta.json 내용(kind reconstruction, 모델 이름, 입력·맵 크기, 임계값). extra로 값을
# 바꾸거나 더한다.
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


# 정상(회색 잡음), 결함(8~16px 흰 칸), 더 밝은 촬영 조건의 정상(+60) 이미지를 만드는 도우미들.
def _normal(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(110, 146, (SIZE, SIZE, 3), dtype=np.uint8)


def _defect(seed: int) -> np.ndarray:
    image = _normal(seed)
    image[8:16, 8:16] = 255
    return image


def _bright(count: int) -> list[np.ndarray]:
    return [np.clip(_normal(100 + s).astype(np.int16) + 60, 0, 255).astype(np.uint8) for s in range(count)]


# 가짜 세션으로 만든 복원 검사기를 정상 24장으로 보정해 돌려준다(오프라인 PatchCore 검사기와 같은 보정 장수).
def _inspector() -> ReconstructionInspector:
    inspector = ReconstructionInspector(ReconSession(), _meta())
    inspector.calibrate([_normal(s) for s in range(24)], alpha=0.05)
    return inspector


# 검사기 하나(범주 toy)와 관리자 토큰으로 서비스 앱을 만든다.
def _client(inspector) -> TestClient:
    return TestClient(service.create_app(inspectors={"toy": inspector}, admin_token=TOKEN), base_url=HOST)


def _inspect(client, image: np.ndarray, **params):
    files = {"image": ("x.png", _png(image), "image/png")}
    return client.post("/inspect", files=files, data={"category": "toy"}, params=params)


# /categories가 복원 모델을 설명하는 방식: kind reconstruction, 모델 이름, 뱅크 행 0, 백본 없음, 입력 크기,
# 보정 기록. kind와 model은 서비스 모델 교체 때 /categories에 더한 필드다(커밋 7d590d2). model_sha256은 시작할
# 때 대조한 모델 파일 지문이라, 아티팩트에서 읽지 않고 직접 넘긴 검사기는 null이다.
def test_categories_describe_a_reconstruction_model():
    item = _client(_inspector()).get("/categories").json()[0]
    assert item["kind"] == "reconstruction" and item["model"] == "dms-toy" and item["model_sha256"] is None
    assert item["bank_rows"] == 0 and item["backbone"] is None and item["img_size"] == SIZE
    assert item["calibration"] == {"strategy": "holdout", "n": 24, "guaranteed": True}


# /inspect가 모델 자신의 점수를 그대로 쓰고(검사기 직접 호출과 같음), 히트맵은 PatchCore와 같은 척도(임계값의
# 1.5배에서 흰색)로 모델의 맵을 그린다. 미리보기는 모델 입력 크기(32px)의 그림이다.
def test_inspect_uses_the_models_own_score_and_map():
    inspector = _inspector()
    client = _client(inspector)
    ok = _inspect(client, _normal(50)).json()
    assert ok["is_defect"] is False and ok["score"] == inspector.inspect(_normal(50)).score
    bad = _inspect(client, _defect(50)).json()
    assert bad["is_defect"] is True and bad["score"] > bad["threshold"] == inspector.threshold
    heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(bad["heatmap_png"]))))
    # 모델 자신의 맵을 PatchCore 맵과 같은 척도로 그린다: 임계값의 1.5배에서 흰색.
    assert heat.shape == (MAP, MAP) and heat[90, 90] == 255 and heat[200:, 200:].max() < 255
    preview = _inspect(client, _defect(50), preview="true").json()
    seen = np.asarray(Image.open(io.BytesIO(base64.b64decode(preview["input_png"]))))
    assert seen.shape == (SIZE, SIZE, 3)


# 모델이 유한하지 않은 맵을 내면(점수는 유한해도) 서비스는 판정을 내지 않는다: 응답은 내부 정보 없는 JSON
# 500이고 보안 헤더가 붙으며 원인은 로그에만 남는다(test_service.py의 처리되지 않은 오류 테스트와 같은
# 형식). 고치기 전에는 검사기가 점수만 검사해 NaN 맵이 히트맵 PNG에 점수 0(검정)으로 조용히 그려진 채
# 200으로 나갔다. heatmap=false여도 같다: 맵이 깨진 실행의 점수는 믿지 않는다(연구 경로
# run_dinomaly.predict와 같은 규칙).
def test_a_map_that_is_not_finite_is_a_500_not_a_black_heatmap(caplog):
    inspector = _inspector()
    real = inspector.session

    class PoisonedMap:
        def run(self, names, feeds):
            score, amap = real.run(names, feeds)
            amap = amap.copy()
            amap[0, 100, 100] = np.nan
            return [score, amap]

    inspector.session = PoisonedMap()
    with pytest.raises(FloatingPointError, match="map"):
        inspector.inspect(_normal(50))
    client = _client(inspector)
    for params in ({}, {"heatmap": "false"}):
        response = _inspect(client, _normal(50), **params)
        assert response.status_code == 500 and response.json() == {"detail": "internal server error"}
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "not finite" not in response.text
    assert "the anomaly map is not finite" in caplog.text


# 복원 모델의 재보정은 뱅크가 없으므로 임계값만 바꾼다. 밝은 조건의 정상 25장으로 보정하면 임계값이 그
# 점수들의 등각 임계값과 같고, 보정에 쓰지 않은 같은 조건 이미지는 대부분 통과하며, 모델 세션은 그대로다. 토큰
# 없는 DELETE는 403이고, 토큰을 주면 원래 상태로 정확히 돌아온다.
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
    # 검사기 자체의 보정과 같은 규칙을 디코딩한 업로드의 점수에 적용한다.
    scores = np.array([inspector.inspect(im).score for im in bright[:25]], dtype=np.float32)
    assert body["threshold"] == conformal_threshold(scores, 0.05).value
    assert body["previous_threshold"] == before["threshold"]
    assert body["calibration"] == {"strategy": "holdout", "n": 25, "guaranteed": True}
    assert sum(_inspect(client, im).json()["is_defect"] for im in bright[25:]) <= 1
    assert inspector.session.__class__ is ReconSession  # 임계값 말고는 아무것도 바뀌지 않았다

    assert client.delete("/calibrate", params={"category": "toy"}).status_code == 403
    reset = client.delete("/calibrate", params={"category": "toy"}, headers=ADMIN).json()
    assert reset["threshold"] == before["threshold"] and reset["calibration"] == before["calibration"]
    assert client.get("/categories").json()[0] == before


# onnxruntime.InferenceSession을 ReconSession 기반 가짜로 바꾸고, 만들어진 세션마다 (폴더 이름, 파일 이름,
# 스레드 수)를 기록한다. 세션 공유와 설정 전달을 실제 모델 없이 확인하려는 것이다.
def _fake_ort(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    created = []

    class FakeSession(ReconSession):
        def __init__(self, path, sess_options=None, providers=None):
            self.path = Path(path)
            created.append((self.path.parent.name, self.path.name, sess_options.intra_op_num_threads))

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    return created


# 복원 방식 아티팩트 세트를 흉내 낸다: 상위 폴더에 공용 model_fp32.onnx 하나, 범주 폴더마다 meta.json(임계값과
# 보정 기록)만 둔다. dinomaly_serving export가 만드는 배치와 같다(모델 하나를 12개 범주가 같이 쓰고 범주
# 폴더에는 임계값만, README 검사 서비스 절).
def _write_set(root: Path, categories=("candle", "pcb1", "pcb2")) -> None:
    (root / "model_fp32.onnx").write_bytes(b"shared")
    for index, name in enumerate(categories):
        (root / name).mkdir()
        meta = _meta(name, threshold=0.25 + index, calibration={"strategy": "holdout", "n": 60})
        (root / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


# load_inspectors가 복원 세트를 읽는지 확인한다. 범주 폴더마다 ReconstructionInspector가 생기고, 공용 모델을
# 쓰는 범주는 세션 하나를 나눠 쓰며 자기 모델이 있는 pcb2만 따로 세션을 만든다. 각 검사기의 모든 속성이
# ReconstructionInspector.load의 결과와 같아야 한다.
# 정밀도 확인: int8-dynamic 모델 파일이 없으면 그 파일 이름으로, 복원 세트에 없는 정밀도(PatchCore용 int8)는
# 허용 목록과 폴더 이름으로, 어디에도 없는 정밀도(fp16)는 전체 목록으로 거절한다. 파일을 넣으면
# int8-dynamic으로 읽힌다.
def test_load_inspectors_reads_a_reconstruction_set(tmp_path, monkeypatch):
    created = _fake_ort(monkeypatch)
    _write_set(tmp_path)
    (tmp_path / "pcb2" / "model_fp32.onnx").write_bytes(b"own")
    (tmp_path / "notes").mkdir()  # meta.json 없음: 범주가 아니다

    found = service.load_inspectors(tmp_path, threads=2)
    assert sorted(found) == ["candle", "pcb1", "pcb2"]
    assert all(isinstance(i, ReconstructionInspector) for i in found.values())
    assert found["candle"].session is found["pcb1"].session is not found["pcb2"].session
    assert created == [(tmp_path.name, "model_fp32.onnx", 2), ("pcb2", "model_fp32.onnx", 2)]
    # 각 검사기는 `ReconstructionInspector.load`가 그 폴더로 만드는 것과 속성 하나하나까지 같다. 예외는 서비스
    # 로더만 붙이는 model_sha256(시작할 때 대조한 모델 지문)이고, 이 세트의 meta에는 지문 기록이 없어
    # None이다. vars()는 인스턴스의 __dict__ 자체라 사본에서 뺀다.
    for name, inspector in found.items():
        alone = ReconstructionInspector.load(tmp_path / name, threads=2)
        mine, theirs = dict(vars(inspector)), vars(alone)
        assert mine.pop("model_sha256") is None
        assert set(mine) == set(theirs)
        for key, value in mine.items():
            if key == "session":
                assert value.path == theirs[key].path
            else:
                assert value == theirs[key], key
        assert (inspector.precision, inspector.threads, inspector.threshold) == ("fp32", 2, alone.threshold)

    # 정밀도마다 쓰는 모델 파일은 세트의 종류(kind)에 따라 다르다.
    with pytest.raises(ValueError, match="model_int8_dynamic.onnx is missing"):
        service.load_inspectors(tmp_path, precision="int8-dynamic")
    with pytest.raises(ValueError, match=r"candle.*\['fp32', 'int8-dynamic'\] for a reconstruction artifact"):
        service.load_inspectors(tmp_path, precision="int8")
    with pytest.raises(ValueError, match="precision must be one of"):
        service.load_inspectors(tmp_path, precision="fp16")
    (tmp_path / "model_int8_dynamic.onnx").write_bytes(b"int8")
    (tmp_path / "pcb2" / "model_int8_dynamic.onnx").write_bytes(b"int8")
    assert service.load_inspectors(tmp_path, precision="int8-dynamic")["pcb1"].precision == "int8-dynamic"


# 잘못된 복원 meta(맵 크기 없음, 모르는 kind, 숫자가 아닌 임계값)는 모두 그 범주 폴더 이름을 담은 ValueError가
# 되어야 한다. 서비스 시작 때 어느 아티팩트가 문제인지 바로 알 수 있게 하려는 것이다.
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


# 복원 세트는 bank.npy 없이 읽히고, 같은 폴더의 meta를 PatchCore 형식으로 바꾸면 여전히 뱅크를 요구한다.
# 서비스가 두 종류(4단계 PatchCore, 6단계 Dinomaly)를 meta의 kind로 갈라 읽는다는 점을 지킨다.
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


# 모델 지문 대조: meta의 source에 모델 파일 SHA-256이 적혀 있으면 서비스가 시작할 때 실제 파일과 대조한다. 왜:
# 모델(ONNX)과 임계값(meta.json)은 다른 파일이라, 모델 파일만 다른 학습본으로 바뀌면 서비스가 조용히 다른
# 모델의 점수를 이 임계값과 비교하게 된다. 확인: (1) 같으면 그대로 읽힌다, (2) 한 바이트라도 다르면 범주 폴더
# 이름과 "does not match"를 담은 ValueError로 멈춘다(ReconstructionInspector.load도 같다), (3) 기록이 64자리
# 16진수가 아니면 깨진 meta로 멈춘다, (4) INT8 지문이 없는 옛 아티팩트는 INT8로 띄울 때 대조를 건너뛰고,
# 있으면 대조한다, (5) 12개 범주가 같이 쓰는 파일의 해시는 한 번만 계산한다, (6) 대조를 통과한 지문은
# 검사기의 model_sha256에 남고 /categories가 그 앞 12자를 보인다(어떤 모델 파일이 떠 있는지 밖에서 확인할 수
# 있게). 지문은 띄운 정밀도의 파일 것이다: INT8로 띄울 때 INT8 기록이 없으면 FP32 기록이 있어도 None(대조하지
# 않은 파일의 지문을 보이지 않는다). /healthz는 지문을 더하지 않고 지금 모양 그대로다.
def test_load_inspectors_checks_the_model_digest_recorded_in_meta(tmp_path, monkeypatch):
    import hashlib

    from defect_inspect import download

    monkeypatch.delenv("DEFECT_INSPECT_PRECISION", raising=False)
    monkeypatch.delenv("DEFECT_INSPECT_THREADS", raising=False)
    _fake_ort(monkeypatch)
    _write_set(tmp_path)
    good = hashlib.sha256(b"shared").hexdigest()
    for name in ("candle", "pcb1", "pcb2"):
        meta = json.loads((tmp_path / name / "meta.json").read_text(encoding="utf-8"))
        meta["source"] = {"onnx_sha256": good}
        (tmp_path / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    hashed = []
    real = download.sha256_file
    monkeypatch.setattr(download, "sha256_file", lambda path: hashed.append(Path(path).name) or real(path))
    found = service.load_inspectors(tmp_path)
    assert sorted(found) == ["candle", "pcb1", "pcb2"]
    assert hashed == ["model_fp32.onnx"]  # 세 범주가 같이 쓰는 파일이라 한 번만
    assert ReconstructionInspector.load(tmp_path / "pcb1").threshold == 1.25
    assert {name: i.model_sha256 for name, i in found.items()} == dict.fromkeys(found, good)
    client = TestClient(service.create_app(tmp_path, admin_token=TOKEN), base_url=HOST)
    shown = {item["category"]: item["model_sha256"] for item in client.get("/categories").json()}
    assert shown == dict.fromkeys(("candle", "pcb1", "pcb2"), good[:12])
    assert client.get("/healthz").json() == {"status": "ok", "categories": ["candle", "pcb1", "pcb2"]}

    (tmp_path / "model_fp32.onnx").write_bytes(b"shared, retrained")
    with pytest.raises(ValueError, match=r"candle.*model_fp32.onnx does not match the artifact meta"):
        service.load_inspectors(tmp_path)
    with pytest.raises(ValueError, match=r"pcb1.*does not match the artifact meta"):
        ReconstructionInspector.load(tmp_path / "pcb1")

    (tmp_path / "model_fp32.onnx").write_bytes(b"shared")
    meta = json.loads((tmp_path / "pcb2" / "meta.json").read_text(encoding="utf-8"))
    meta["source"] = {"onnx_sha256": good.upper()}
    (tmp_path / "pcb2" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=r"pcb2.*not a SHA-256 hex digest"):
        service.load_inspectors(tmp_path)
    meta["source"] = {"onnx_sha256": good}
    (tmp_path / "pcb2" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # INT8: 지문이 없는 옛 아티팩트는 대조 없이 읽히고(대조한 지문이 없으니 model_sha256은 None), 지문을
    # 적으면 대조한다.
    (tmp_path / "model_int8_dynamic.onnx").write_bytes(b"int8")
    int8 = service.load_inspectors(tmp_path, precision="int8-dynamic")["pcb1"]
    assert int8.precision == "int8-dynamic" and int8.model_sha256 is None

    def record_int8(digest: str) -> None:
        for name in ("candle", "pcb1", "pcb2"):
            meta = json.loads((tmp_path / name / "meta.json").read_text(encoding="utf-8"))
            meta["source"]["onnx_int8_dynamic_sha256"] = digest
            (tmp_path / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    record_int8(hashlib.sha256(b"int8, old").hexdigest())
    with pytest.raises(ValueError, match="model_int8_dynamic.onnx does not match the artifact meta"):
        service.load_inspectors(tmp_path, precision="int8-dynamic")
    # FP32는 INT8 지문과 상관없이 자기 지문으로만 대조한다.
    assert sorted(service.load_inspectors(tmp_path)) == ["candle", "pcb1", "pcb2"]
    record_int8(hashlib.sha256(b"int8").hexdigest())
    int8 = service.load_inspectors(tmp_path, precision="int8-dynamic")["pcb1"]
    assert int8.model_sha256 == hashlib.sha256(b"int8").hexdigest() != good


# 서빙 인터페이스(입력 image, 출력 score·map)를 가진 실제 ONNX 모델을 만들어 저장한다. onnx.checker로 형식을
# 검사하고, onnx가 없으면 건너뛴다.
def _toy_onnx(path: Path) -> None:
    """서빙 인터페이스를 가진 실제 ONNX 모델: map = 채널 평균 [1, S, S], score = 그 최댓값 [1]."""
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


# 실제 onnxruntime 세션으로 끝에서 끝까지 확인한다: 경로만 주고 create_app을 부르면 두 범주가 세션 하나를 나눠
# 쓰고, /categories에 범주별 임계값이 보이며, /inspect 점수가 ReconstructionInspector.load로 따로 읽은
# 검사기의 점수와 같고 판정은 '점수 > 임계값'이다. 히트맵은 meta의 맵 크기(여기서는 32)로 나온다.
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

# [파일 역할] FastAPI 검사 서비스(`defect_inspect.service`)를 검증하는 테스트 묶음이다. 모델 파일 없이 도는
# 오프라인 앱(합성 범주 demo, 작은 PatchCore 흉내)으로 /inspect, /calibrate, DELETE /calibrate, /categories,
# /healthz, 데모 페이지와 정적 파일을 시험한다.
# [왜 필요한가] 이 서비스는 업로드를 받아 판정하고 관리자가 임계값을 바꿀 수 있으므로, 점수 정확성뿐 아니라
# 입력 검증(형식·크기·16비트·투명도), 보호 장치(Host 검사, 다른 출처의 쓰기 요청 거절, 관리자 토큰, 본문 크기
# 제한, 보안 헤더), 동시성(범주별 잠금, 작업자 수 제한과 503)이 모두 맞아야 한다. docs/security.md의 DAST
# 판정(예: Anti-CSRF 지적을 오탐으로 본 근거)도 이 파일의 테스트에 기대고 있다.
# [주요 구성] 요청 도우미와 ASGI 직접 호출 → 판정·히트맵·입력 형식(RGB 그림은 복사 없이 디코딩) → 잘못된
# 요청과 크기 제한 → 손상 파일과 디코더 보안 권고 회귀(작은 BMP·APNG) → 16비트·투명도·미리보기 → 재보정과
# 되돌리기 → 관리자 토큰·다른 출처 거절 → 임계값 변경·토큰 거절의 감사 기록(uvicorn 기본 로깅에서 표준
# 출력의 JSON 한 줄) → 잠금과 동시 요청·대기열 → 본문을 받기 전의 입장 제한(503)과 스풀 공간 부족 → Host
# 검사·보안 헤더·정적 파일 → 아티팩트 읽기(load_inspectors, 대조한 모델 지문)와 시작 예열 → 히트맵 PNG 눈금과
# 비유한 맵 거부 → 데모 페이지 JS(node가 있을 때).

import asyncio
import base64
import errno
import inspect as pyinspect
import io
import itertools
import json
import re
import shutil
import struct
import subprocess
import threading
import time
import tracemalloc
import urllib.parse
import zlib
from pathlib import Path

import numpy as np
import PIL
import pytest
from PIL import Image, ImageOps

# 서비스는 serve 의존성 그룹(FastAPI, 업로드 파싱용 python-multipart)이 있어야 돈다. 기본 패키지만 설치한
# 환경에서는 이 파일 전체를 건너뛴다. torch는 필요 없다.
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

# TestClient는 실제 서버를 띄우지 않고 앱을 같은 프로세스에서 HTTP처럼 부른다. HTTPException은 서비스의
# 디코딩 함수(_decode)를 서버 없이 직접 부를 때 400 거절을 받아 보는 데 쓴다. httpx(TestClient가 쓰는 dev
# 의존성)의 AsyncClient + ASGITransport는 요청 여럿을 한 이벤트 루프에서 겹치게 보내고, 앱이 receive()를 부를
# 때만 본문 조각을 넘기므로 "거절된 요청의 본문이 앱에 얼마나 들어갔는가"를 재는 데 쓴다.
import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from defect_inspect import service  # noqa: E402

# 기본 허용 Host(127.0.0.1)로 요청하는 주소, 테스트용 관리자 토큰과 그 헤더, 모든 응답에 붙어야 하는 보안 헤더
# 이름들.
HOST = "http://127.0.0.1"
TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}
SECURITY = ("x-content-type-options", "x-frame-options", "content-security-policy", "referrer-policy")
# 감사 기록(임계값 변경·관리자 토큰 거절)을 쓰는 로거 이름. 서비스는 이 로거로 표준 출력에 JSON 한 줄씩 쓰고,
# 운영자는 로그 설정에서 이 이름으로 감사 줄을 따로 다룰 수 있다(README 검사 서비스 절). AUDIT_TIME은 감사
# 줄의 time 필드 형식: 밀리초까지의 UTC ISO 8601 시각(예: 2026-10-24T12:34:56.789+00:00).
AUDIT_LOGGER = "defect_inspect.audit"
AUDIT_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00")


# 배열이나 PIL 이미지를 원하는 형식의 파일 바이트로 인코딩한다(업로드 본문을 만드는 데 쓴다).
def _encode(array: np.ndarray | Image.Image, fmt: str = "PNG", **options) -> bytes:
    image = array if isinstance(array, Image.Image) else Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def _png(array: np.ndarray) -> bytes:
    return _encode(array)


# 오프라인 검사기의 '정상'과 같은 분포(밝기 110~146 회색 잡음)의 64px 이미지.
def _normal(seed=1, size=64) -> np.ndarray:
    return np.random.default_rng(seed).integers(110, 146, (size, size, 3), dtype=np.uint8)


# 정상 이미지 가운데(20~36px)에 흰 사각형을 넣은 '결함' 이미지.
def _defect(seed=1) -> np.ndarray:
    image = _normal(seed)
    image[20:36, 20:36] = 255
    return image


def _bright(count: int) -> list[np.ndarray]:
    """더 밝은 라인(촬영 조건)의 정상 이미지. 오프라인 검사기의 처음 임계값으로는 모두 불량이 된다."""
    return [np.clip(_normal(seed=s).astype(np.int16) + 60, 0, 255).astype(np.uint8) for s in range(count)]


# /calibrate에 보낼 multipart 파일 목록(필드 이름 images).
def _files(images) -> list:
    return [("images", (f"{i}.png", _png(im), "image/png")) for i, im in enumerate(images)]


# 관리자 토큰을 켠 오프라인 앱의 TestClient. settings로 작업자 수, 대기열, 허용 Host 등을 바꿀 수 있다.
def _client(**settings) -> TestClient:
    settings.setdefault("admin_token", TOKEN)
    return TestClient(service.create_offline_app(**settings), base_url=HOST)


@pytest.fixture
def client():
    return _client()


# /inspect 호출 도우미(이미지 파일 + 범주 폼 필드, 쿼리 파라미터는 heatmap·preview).
def _inspect(client, data: bytes, category="demo", headers=None, **params):
    return client.post(
        "/inspect",
        files={"image": ("x.png", data, "image/png")},
        data={"category": category},
        params=params,
        headers=headers,
    )


# /calibrate 호출 도우미. 기본으로 관리자 토큰 헤더를 붙인다.
def _calibrate(client, images, headers=ADMIN, **fields):
    return client.post(
        "/calibrate", files=_files(images), data={"category": "demo", **fields}, headers=headers
    )


# /categories에서 demo 범주의 현재 임계값 상태(임계값, alpha, 보정 기록 등)를 읽는다.
def _state(client) -> dict:
    return client.get("/categories").json()[0]


# HTTP 클라이언트를 거치지 않고 ASGI 앱을 직접 부르는 도우미. 클라이언트가 경로의 '..'을 정리해 버리거나
# 본문을 한꺼번에 보내는 것을 피해, 경로 탈출 시도와 '본문을 조각으로 보낼 때 앱이 얼마나 읽었는가(pulled)'를
# 그대로 시험할 수 있다. root_path는 uvicorn --root-path로 띄웠을 때의 scope를 흉내 낸다: uvicorn은 path에
# root_path를 앞붙여 넘기므로(path도 root_path로 시작하게 준다) Starlette 라우터는 그것을 떼고 경로를 맞춘다.
# peer는 scope의 client((주소, 포트))다. uvicorn은 믿는 프록시에서 온 연결이면 이 값을
# X-Forwarded-For로 바꾸므로, 감사 줄의 client 필드가 서버 밖에서 온 글자를 담는 경우를 흉내 낼 때 쓴다.
def _asgi(
    app, method: str, path: str, *, headers=(), chunks=(), root_path: str = "", peer=("127.0.0.1", 50000)
):
    """ASGI 앱을 직접 부른다: 경로를 클라이언트가 정리하지 않고, 본문은 조각으로 들어간다.

    (상태 코드, 헤더, 본문, 앱이 요청해 받아 간 본문 조각들의 크기)를 돌려준다.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": root_path,
        "headers": [(b"host", b"127.0.0.1"), *((k.lower().encode(), v.encode()) for k, v in headers)],
        "client": peer,
        "server": ("127.0.0.1", 80),
    }
    body = iter(chunks)
    pulled, sent = [], []

    async def receive():
        chunk = next(body, None)
        if chunk is None:
            return {"type": "http.request", "body": b"", "more_body": False}
        pulled.append(len(chunk))
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    out = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], out, payload, pulled


# multipart 본문의 앞부분만 만든다. 파일 부분이 끝나지 않는 본문을 이어 보내, 크기 제한이 본문을 끝까지 읽기
# 전에 작동하는지 시험하는 데 쓴다.
def _multipart_head(field: str = "image") -> tuple[dict, bytes]:
    """파일 부분이 끝나지 않아도 되는 multipart 본문의 헤더와 시작 부분."""
    head = (
        b'--B\r\nContent-Disposition: form-data; name="category"\r\n\r\ndemo\r\n'
        b'--B\r\nContent-Disposition: form-data; name="' + field.encode() + b'"; filename="x.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
    )
    return {"Content-Type": "multipart/form-data; boundary=B"}, head


# /healthz와 /categories의 기본 응답. 오프라인 검사기는 정상 24장으로 보정돼 있어 보정 기록이 holdout, n 24,
# 보장됨이다. /categories의 model_sha256(시작할 때 대조한 모델 파일 지문의 앞 12자)은 모델 파일이 없는 합성
# 검사기라 null이다. /healthz는 상태와 범주 목록만 내고 그 모양이 정확히 그대로여야 한다(compose의 상태 점검과
# CI docker 작업이 이 응답을 그대로 비교한다). 헤더만 묻는 HEAD 요청(상태 점검 도구가 쓰는 방식)도 200이어야
# 한다.
def test_health_and_categories(client):
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok", "categories": ["demo"]}
    items = client.get("/categories").json()
    assert [i["category"] for i in items] == ["demo"]
    assert items[0]["img_size"] == 64 and items[0]["threshold"] > 0
    assert items[0]["calibration"] == {"strategy": "holdout", "n": 24, "guaranteed": True}
    assert items[0]["model"] == "offline" and items[0]["model_sha256"] is None
    # 헤더만 묻는 상태 점검(HEAD 요청)도 동작한다.
    assert client.head("/healthz").status_code == 200 and client.head("/").status_code == 200


# 정상은 통과하고 결함은 불량으로 판정되며, 히트맵은 256x256 8비트 PNG이고 흰 사각형 자리(256 격자 80~144)가
# 가장 뜨겁다. heatmap=false면 히트맵 없이 같은 점수를 낸다.
def test_a_normal_image_passes_and_a_defect_is_flagged_with_a_heatmap(client):
    ok = _inspect(client, _png(_normal(seed=3)))
    assert ok.status_code == 200
    body = ok.json()
    assert body["category"] == "demo" and body["is_defect"] is False and body["score"] <= body["threshold"]
    assert body["latency_ms"] >= 0

    bad = _inspect(client, _png(_defect(seed=3))).json()
    assert bad["is_defect"] is True and bad["score"] > bad["threshold"]
    heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(bad["heatmap_png"]))))
    assert heat.shape == (256, 256) and heat.dtype == np.uint8
    # 흰 사각형은 256 격자에서 행·열 80..144를 덮는다: 그곳이 가장 뜨겁고, 먼 구석은 차갑다.
    assert heat[112, 112] == heat.max() == 255
    assert heat[200:, 200:].max() < 128

    without = _inspect(client, _png(_defect(seed=3)), heatmap="false").json()
    assert "heatmap_png" not in without and without["score"] == pytest.approx(bad["score"])


# 크기·비율이 다른 회색조, JPEG, 팔레트, BMP, WebP 모두 받아 검사기 자체의 파이프라인과 같은 점수를 내는지
# 본다. 서비스는 디코딩 직후 모델 입력 크기로 줄이는데(큰 원본을 메모리에 두지 않으려고), 그 리사이즈가
# inspector.preprocess와 같은 BICUBIC 호출이라 점수가 바뀌지 않아야 한다.
def test_any_image_size_and_8_bit_mode_is_scored_like_the_inspector_itself(client):
    twin = service.offline_inspector()
    big = np.random.default_rng(0).integers(110, 146, (300, 480), dtype=np.uint8)  # 회색조, 정사각형 아님
    jpeg = _encode(_normal(), "JPEG")
    palette = _encode(Image.fromarray(_normal()).quantize(32))
    for data in (
        _png(big),
        jpeg,
        palette,
        _encode(_normal(), "BMP"),
        _encode(_normal(), "WEBP", lossless=True),
    ):
        response = _inspect(client, data)
        assert response.status_code == 200
        # 서비스는 디코딩 직후 리사이즈한다. 점수는 검사기 자체 파이프라인의 점수와 같다.
        assert response.json()["score"] == pytest.approx(twin.inspect(Image.open(io.BytesIO(data))).score)
    decoded = service._decode(_png(big), 64)
    assert decoded.mode == "RGB" and decoded.size == (64, 64)
    expected = Image.fromarray(big).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
    assert np.array_equal(np.asarray(decoded), np.asarray(expected))


# 디코딩이 이미 RGB인 그림을 복사하지 않는지와, 어느 8비트 모드에서든 결과 화소가 예전 경로(convert("RGB") 뒤
# resize)와 똑같은지 본다. Pillow의 convert는 같은 모드를 요구하면 self.copy()를 돌려주므로, 예전에는 RGB
# 업로드도 원본 크기 사본을 한 번 더 만들어 원본과 나란히 들고 있었다(Pillow는 RGB도 화소당 4바이트라 2,400만
# 화소면 약 96MB가 더 든다).
# 그래서 RGB로 열리는 PNG·JPEG·BMP·TIFF·WebP에서는 convert가 한 번도 불리지 않아야 하고, 다른 모드(L·P·
# 불투명 RGBA·CMYK)는 convert를 거치되 화소는 예전과 같아야 한다. Image.Image.convert를 세는 감싸개로 바꿔
# 호출을 기록하고, 비교 대상(예전 경로의 결과)은 세기 전에 만든다.
def test_decoding_keeps_an_rgb_picture_without_a_copy_and_every_mode_pixel_identical(monkeypatch):
    rgb = _normal()
    opaque = np.dstack([rgb, np.full((64, 64), 255, np.uint8)])
    wide = np.random.default_rng(0).integers(0, 256, (300, 480, 3), dtype=np.uint8)
    uploads = {
        "rgb-png": _png(rgb),
        "rgb-jpeg": _encode(rgb, "JPEG", quality=95),
        "rgb-bmp": _encode(rgb, "BMP"),
        "rgb-tiff": _encode(rgb, "TIFF"),
        "rgb-webp": _encode(rgb, "WEBP", lossless=True),
        "rgb-wide-png": _png(wide),  # 정사각형이 아니고 입력 크기보다 크다: 리사이즈가 실제로 일어난다
        "grey-png": _png(rgb[..., 0]),
        "palette-png": _encode(Image.fromarray(rgb).quantize(32)),
        "opaque-rgba-png": _png(opaque),
        "cmyk-jpeg": _encode(Image.fromarray(rgb).convert("CMYK"), "JPEG", quality=95),
    }
    expected = {
        name: Image.open(io.BytesIO(data)).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
        for name, data in uploads.items()
    }

    converted = []
    real_convert = Image.Image.convert

    def counting_convert(self, mode=None, *args, **kwargs):
        converted.append((self.mode, mode))
        return real_convert(self, mode, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", counting_convert)
    for name, data in uploads.items():
        converted.clear()
        decoded = service._decode(data, 64)
        assert decoded.mode == "RGB" and decoded.size == (64, 64), name
        assert np.array_equal(np.asarray(decoded), np.asarray(expected[name])), name
        if name.startswith("rgb-"):
            assert converted == [], name  # 이미 RGB: 원본 크기 사본을 만들지 않는다
        else:
            assert converted and converted[-1][1] == "RGB", name


# 잘못된 요청의 상태 코드: 모르는 범주 404, 이미지가 아닌 바이트·빈 업로드 400, 필드 누락 422, 64자를 넘는
# 범주 이름 422, 업로드 크기 초과 413, 픽셀 수 초과 400. 크기 제한은 monkeypatch로 작게 바꿔 시험한다.
def test_bad_requests_are_rejected(client, monkeypatch):
    assert _inspect(client, _png(_normal()), category="nope").status_code == 404
    assert _inspect(client, b"this is not an image").status_code == 400
    assert _inspect(client, b"").status_code == 400
    assert client.post("/inspect", data={"category": "demo"}).status_code == 422
    assert (
        client.post("/inspect", files={"image": ("x.png", _png(_normal()), "image/png")}).status_code == 422
    )
    too_long = _inspect(client, _png(_normal()), category="x" * 65)
    assert too_long.status_code == 422

    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 100)
    assert _inspect(client, _png(_normal())).status_code == 413
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(service, "MAX_IMAGE_PIXELS", 1000)
    assert _inspect(client, _png(_normal())).status_code == 400


# 422 검증 오류가 입력값(스크립트 문자열 등)이나 파이썬 타입 이름을 되돌려 보내지 않는지 확인한다. 오류 항목은
# loc·type·msg만 가진다. 입력을 그대로 비추는 응답은 반사형 XSS나 내부 정보 노출의 통로가 될 수 있다.
def test_validation_errors_do_not_echo_the_input(client):
    probe = "<script>alert(1)</script>" * 3
    responses = [
        client.post("/inspect", data={"category": "demo", "image": "hello"}),  # 파일 자리에 텍스트
        _inspect(client, _png(_normal()), category=probe),
        client.post(
            "/calibrate", files=_files([_normal()]), data={"category": "demo", "alpha": probe}, headers=ADMIN
        ),
    ]
    for response in responses:
        assert response.status_code == 422
        assert (
            "hello" not in response.text and "script" not in response.text and "<class" not in response.text
        )
        for item in response.json()["detail"]:
            assert set(item) == {"loc", "type", "msg"} and item["loc"][0] == "body"


# 본문 크기 제한이 Content-Length에 기대지 않는지 확인한다. 큰 Content-Length를 밝히면 본문을 한 바이트도 앱에
# 넘기기 전에 413이고, 길이를 밝히지 않는 chunked 본문도 413이며, 조각으로 계속 들어오면 한도(업로드 한도 +
# multipart 여유) 바로 뒤에서 읽기를 멈추고 Connection: close로 답한다. 어느 거절에도 보안 헤더가 붙는다.
# /calibrate는 여러 장을 받으므로 본문 전체 한도(MAX_REQUEST_BYTES)가 따로 있고, 그 한도를 넘으면 첫 조각 뒤에
# 멈춘다.
def test_the_body_limit_holds_without_a_content_length(monkeypatch):
    app = service.create_offline_app(admin_token=TOKEN)
    client = TestClient(app, base_url=HOST)
    header, head = _multipart_head()
    whole = head + _png(_normal()) + b"\r\n--B--\r\n"
    assert client.post("/inspect", content=whole, headers=header).status_code == 200

    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 2000)
    monkeypatch.setattr(service, "MULTIPART_SLACK_BYTES", 1000)
    # 길이를 밝힌 경우: 본문을 한 바이트도 앱에 넘기기 전에 거절한다.
    status, out, _, pulled = _asgi(
        app, "POST", "/inspect", headers={**header, "Content-Length": "999999999"}.items(), chunks=[head]
    )
    assert status == 413 and pulled == [] and out["x-content-type-options"] == "nosniff"
    # chunked(생성기 본문에는 Content-Length가 없다): 예전 검사는 이런 요청을 아예 보지 않았다.
    chunked = client.post("/inspect", content=iter([whole]), headers=header)
    assert "content-length" not in chunked.request.headers and chunked.status_code == 413
    # 조각으로 들어오는 경우: 서비스는 본문을 계속 받아 쌓지 않고 한도 바로 뒤에서 더 읽기를 멈춘다.
    pieces = itertools.chain([head], itertools.repeat(b"\0" * 500, 400))
    status, out, payload, pulled = _asgi(app, "POST", "/inspect", headers=header.items(), chunks=pieces)
    assert status == 413 and json.loads(payload) == {"detail": "request body too large"}
    assert sum(pulled) <= 3000 + 500 and out["connection"] == "close"
    assert out["x-content-type-options"] == "nosniff"

    # 보정 호출에는 본문 전체에 대한 별도의 더 큰 한도가 있다.
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 20_000)
    monkeypatch.setattr(service, "MULTIPART_SLACK_BYTES", 0)
    assert _calibrate(client, _bright(25)).status_code == 200
    monkeypatch.setattr(service, "MAX_REQUEST_BYTES", 10)
    assert _calibrate(client, _bright(25)).status_code == 413
    header, head = _multipart_head("images")
    pieces = itertools.chain([head], itertools.repeat(b"\0" * 500, 400))
    status, _, _, pulled = _asgi(
        app, "POST", "/calibrate", headers={**header, **ADMIN}.items(), chunks=pieces
    )
    assert status == 413 and len(pulled) == 1


# 손상된 파일(데이터 길이를 반으로 줄인 PNG, 잘린 PNG)과 받지 않는 형식(GIF, PPM)은 500이 아니라 같은 문구의
# 400이어야 한다. Pillow 디코더는 손상 파일에서 OSError 말고도 SyntaxError 등을 내므로, 서비스가 모든 디코딩
# 예외를 400으로 바꾸는지 본다. 보정 요청에 손상 파일이 한 장만 섞여도 400이고, Image.open 자체가 예상 밖
# 예외를 내도 400이다.
def test_damaged_and_unsupported_files_are_a_400_not_a_500(client, monkeypatch):
    rgba = np.dstack([_normal(), np.full((64, 64), 255, np.uint8)])
    damaged = bytearray(_png(rgba))
    # 데이터 청크(IDAT)에 적힌 길이를 반으로 줄인다: 그러면 Pillow가 압축된 바이트 안에서 청크 헤더를
    # 읽어 OSError가 아니라 SyntaxError를 낸다. (길이의 한 바이트를 뒤집으면 길이가 오히려 늘 수도 있고
    # 그러면 여전히 디코딩된다. 늘어나는지는 zlib 빌드에 따라 달라서, 손상을 이렇게 직접 적어 둔다.)
    assert damaged[37:41] == b"IDAT"
    damaged[33:37] = (int.from_bytes(damaged[33:37], "big") // 2).to_bytes(4, "big")
    truncated = _png(_normal())[:200]
    for data in (bytes(damaged), truncated, _encode(_normal(), "GIF"), _encode(_normal(), "PPM")):
        response = _inspect(client, data)
        assert response.status_code == 400 and response.json() == {
            "detail": "the upload is not a readable image"
        }
        assert response.headers["x-content-type-options"] == "nosniff"
    assert _calibrate(client, [_normal()] * 19).status_code == 200
    files = [*_files([_normal()] * 19), ("images", ("bad.png", bytes(damaged), "image/png"))]
    assert client.post("/calibrate", files=files, data={"category": "demo"}, headers=ADMIN).status_code == 400

    def boom(*args, **kwargs):
        raise RuntimeError("a decoder bug")

    monkeypatch.setattr(service.Image, "open", boom)
    assert _inspect(client, _png(_normal())).status_code == 400


# 8비트 팔레트 BMP(BI_RLE8 압축) 파일 바이트를 만든다. 파일 머리 14바이트 + 정보 머리 40바이트
# + 256색 회색 팔레트 1,024바이트 + RLE 자료 6바이트 = 1,084바이트이고, 그림은 폭 `width`·높이 1이다.
# RLE 자료는 "델타(오른쪽 0칸, 위로 `up`줄 건너뛰기: 00 02 00 up)"와 "비트맵 끝(00 01)" 두 명령뿐이라
# 실제 화소 자료는 없다.
def _bmp_rle8(width: int, up: int) -> bytes:
    palette = b"".join(struct.pack("<BBBB", i, i, i, 0) for i in range(256))
    rle = bytes([0, 2, 0, up, 0, 1])
    offset = 14 + 40 + len(palette)
    # 정보 머리(BITMAPINFOHEADER): 크기 40, 폭, 높이 1, 면 1, 화소당 8비트, 압축 1(BI_RLE8), 자료 크기,
    # 해상도 2835 화소/m(72dpi) 두 개, 쓰는 색 256, 중요한 색 0(전부).
    info = struct.pack("<IiiHHIIiiII", 40, width, 1, 1, 8, 1, len(rle), 2835, 2835, 256, 0)
    return b"BM" + struct.pack("<IHHI", offset + len(rle), 0, 0, offset) + info + palette + rle


# PNG 청크 하나(길이 4바이트 + 종류 4바이트 + 자료 + CRC-32)를 만든다.
def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


# 127바이트 APNG를 만든다. 머리(IHDR)에는 side×side 8비트 RGB라고 적고, 애니메이션 프레임은 (0, 0)의
# 1×1 하나뿐이며 그 프레임의 처분 방식이 "배경으로 지움(dispose_op=1)"이다. 실제 화소 자료(IDAT)는
# 1×1 한 줄(필터 바이트 + 3바이트)뿐이다.
def _apng_background_dispose(side: int) -> bytes:
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    animation = struct.pack(">II", 1, 0)  # 프레임 1개, 무한 반복
    # fcTL: 순번 0, 폭 1, 높이 1, 위치 (0, 0), 지연 1/1초, dispose_op 1(배경), blend_op 0(덮어쓰기).
    frame = struct.pack(">IIIIIHHBB", 0, 1, 1, 0, 0, 1, 1, 1, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"acTL", animation)
        + _png_chunk(b"fcTL", frame)
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + _png_chunk(b"IEND", b"")
    )


# 작은 BMP가 디코딩 중 거대한 메모리로 부풀지 않는지 본다(Pillow 공개 보안 권고
# GHSA-24rc-8hj2-v98c, BMP RLE 디코더의 메모리 증폭, Pillow 13.0.0에서 수정). 1,084바이트 파일이
# 폭 200,000·높이 1의 그림이라 화소 수 검사(MAX_IMAGE_PIXELS 2,500만)를 통과한다. Pillow 12.3.0의
# 파이썬 RLE 디코더는 델타가 그림 밖(높이가 1인데 255줄 위)을 가리켜도 건너뛴 칸만큼의 0바이트
# (255 × 200,000 ≈ 51MB)를 파이썬 bytes로 만들어 붙여서, 디코딩 동안 파이썬 메모리 최고치가 약 98MB였다
# (파일 크기의 약 9만 배). 같은 구조로 폭을 화소 수 한도까지 키우면 수 GB가 되어, 메모리 한도가 있는
# 컨테이너(compose의 mem_limit)가 죽을 수 있다.
# 결과는 400(읽을 수 없는 그림)이든 정상 디코딩이든 상관없고, 디코딩 동안 늘어난 파이썬 메모리
# 최고치가 20MB보다 작은지만 본다. 주의: tracemalloc은 파이썬 할당(bytes·bytearray 등)만 센다. 이
# 증폭은 파이썬 디코더가 만드는 bytes라 잡히지만, Pillow가 C에서 잡는 화소 버퍼는 잡히지 않는다(그래서
# APNG 경우는 아래 버전 검사로 지킨다). 다른 곳에서 이미 tracemalloc을 켰다면(예: python -X
# tracemalloc) 끄지 않고, 최고치를 초기화한 뒤 그때의 사용량을 빼서 이 디코딩이 늘린 몫만 본다.
def test_a_tiny_rle_bmp_does_not_blow_up_into_a_huge_allocation():
    data = _bmp_rle8(200_000, up=255)
    assert len(data) == 1084
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        try:
            service._decode(data, 64)
        except HTTPException as error:
            assert error.status_code == 400
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        if started:
            tracemalloc.stop()
    assert peak < 20 * 2**20, f"decoding a 1,084-byte BMP peaked at {peak / 2**20:.1f} MB"


# 설치된 Pillow가 13 이상인지 본다. 서비스가 받는 형식(TIFF·PNG·BMP)의 디코딩 경로에 걸리는 공개
# 보안 권고들이 13.0.0에서 고쳐졌기 때문이다(docs/security.md의 의존성 취약점 대조). 그중 APNG 권고
# (GHSA-5xm2-6676-qcp9)는 처분 방식이 "배경으로 지움"인 프레임이 있으면 Image.open이 머리에 적힌
# 크기 그대로의 바탕 그림을 화소 수 검사보다 먼저 잡는 것이다(12.3.0에서 아래 127바이트 파일로 약
# 137MB). 이 할당은 Pillow의 C 코드가 하므로 tracemalloc에 잡히지 않아, 위 BMP처럼 메모리 최고치로
# 시험할 수 없다. 그래서 잠금 파일이 다시 낮은 버전으로 내려가는 일을 이 버전 단언으로 막는다. 같은
# 파일은 어느 버전에서든 화소 수(6,000 × 6,000 = 3,600만)가 한도를 넘으므로 400이어야 한다.
def test_the_installed_pillow_has_the_decoder_security_fixes():
    assert int(PIL.__version__.split(".")[0]) >= 13, PIL.__version__
    data = _apng_background_dispose(6000)
    assert len(data) == 127
    with pytest.raises(HTTPException) as refused:
        service._decode(data, 64)
    assert refused.value.status_code == 400


# 모델이 예상 밖 오류를 내면 응답은 내용 없는 JSON 500이고 보안 헤더가 붙으며, 오류 메시지는 응답에 나가지
# 않고 로그에만 남는다(_Guard가 처리하지 않은 예외를 직접 감싸는 부분).
def test_an_unhandled_error_is_a_json_500_with_the_security_headers(client, caplog):
    class Broken:
        def run(self, names, feeds):
            raise RuntimeError("the model fell over")

    client.app.state.inspectors["demo"].session = Broken()
    response = _inspect(client, _png(_normal()))
    assert response.status_code == 500 and response.json() == {"detail": "internal server error"}
    for name in SECURITY:
        assert name in response.headers
    assert "the model fell over" not in response.text and "the model fell over" in caplog.text


# 16비트·부동소수 이미지는 거절한다. Pillow가 RGB로 바꿀 때 255로 잘라 이미지 전체가 흰색으로 채점되기
# 때문이다(소스의 DEEP_MODES 주석). 보정 요청에 섞여도 400이고 임계값은 그대로다.
def test_16_bit_images_are_refused_instead_of_scored_as_white(client):
    grey = np.random.default_rng(3).integers(110, 146, (64, 64), dtype=np.uint8)
    assert _inspect(client, _png(grey)).json()["is_defect"] is not None
    deep = grey.astype(np.uint16) * 257
    for data in (_encode(deep, "PNG"), _encode(deep, "TIFF"), _encode(grey.astype(np.float32), "TIFF")):
        response = _inspect(client, data)
        assert response.status_code == 400 and "8 bits" in response.json()["detail"]
    assert _calibrate(client, [deep] * 19).status_code == 400
    assert _state(client)["calibration"]["n"] == 24


# 완전히 불투명한 알파 채널은 무시하고 같은 점수를 낸다. 투명한 픽셀이 있으면(투명 영역 아래 흰 결함, 팔레트
# 투명색) 거절한다. 보는 사람에게는 보이지 않는 픽셀이 채점되는 일을 막으려는 것이다.
def test_transparency_is_refused_and_an_opaque_alpha_channel_is_ignored(client):
    plain = _inspect(client, _png(_normal())).json()
    alpha = np.full((64, 64), 255, np.uint8)
    opaque = _inspect(client, _png(np.dstack([_normal(), alpha])))
    assert opaque.status_code == 200 and opaque.json()["score"] == pytest.approx(plain["score"])
    grey_alpha = _inspect(client, _png(np.dstack([_normal()[..., 0], alpha])))
    assert grey_alpha.status_code == 200

    # 어떤 뷰어에도 보이지 않는 흰 조각: 완전히 투명한 픽셀 아래에 흰색이 있다.
    hidden = np.dstack([_defect(), alpha])
    hidden[20:36, 20:36, 3] = 0
    keyed = _encode(Image.fromarray(_normal()).quantize(8), transparency=0)  # 팔레트 0번 색이 투명색
    for data in (_png(hidden), keyed):
        response = _inspect(client, data)
        assert response.status_code == 400 and "transparency" in response.json()["detail"]


# preview=true의 input_png가 모델이 실제로 본 그림인지 확인한다. EXIF 회전 정보가 있는 JPEG도 회전하지 않은
# 저장 픽셀을 채점하고(모델과 뱅크를 만든 리사이즈 캐시와 같은 방식), 미리보기·점수·히트맵이 모두 같은 그림을
# 가리킨다. 브라우저가 그리지 못하는 TIFF도 미리보기가 오고, 요청하지 않으면 미리보기를 보내지 않는다.
def test_preview_is_the_picture_the_model_saw(client):
    stored = np.random.default_rng(2).integers(110, 146, (48, 80, 3), dtype=np.uint8)
    stored[2:14, 2:20] = 255  # 저장된 픽셀 기준 왼쪽 위
    exif = Image.Exif()
    exif[0x0112] = 6  # 뷰어는 이 파일을 90도 돌려 보여 준다
    rotated = _encode(stored, "JPEG", quality=95, exif=exif)
    plain = _encode(stored, "JPEG", quality=95)
    assert ImageOps.exif_transpose(Image.open(io.BytesIO(rotated))).size == (48, 80)

    body = _inspect(client, rotated, preview="true").json()
    seen = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["input_png"]))))
    expected = Image.open(io.BytesIO(plain)).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
    # 돌려 보인 모습이 아니라 저장된 픽셀: 점수, 히트맵, 미리보기가 모두 같은 그림을 가리킨다.
    assert seen.shape == (64, 64, 3) and np.array_equal(seen, np.asarray(expected))
    assert body["score"] == pytest.approx(_inspect(client, plain).json()["score"])
    heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["heatmap_png"]))))
    assert heat[:128, :128].max() == heat.max() and seen[:16, :16].mean() > 200

    # 브라우저가 그리지 못하는 형식도 미리보기를 받고, 요청하지 않으면 미리보기를 보내지 않는다.
    tiff = _inspect(client, _encode(_defect(), "TIFF"), preview="true")
    assert tiff.status_code == 200 and tiff.json()["is_defect"] is True
    shown = np.asarray(Image.open(io.BytesIO(base64.b64decode(tiff.json()["input_png"]))))
    assert np.array_equal(shown, _defect())
    assert "input_png" not in _inspect(client, _png(_normal())).json()


# 재보정: 더 밝은 촬영 조건에서는 기존 임계값으로 정상도 불량이 된다. 그 조건의 정상 25장으로 /calibrate를
# 부르면 임계값이 올라가고 보정 기록(holdout, n 25, 보장됨)이 바뀌며, 보정에 쓰지 않은 같은 조건 이미지는
# 대부분 통과한다. 서비스는 잠금 밖에서 점수를 매기고 규칙을 직접 적용하므로, 결과가 Inspector.calibrate와
# 같은지도 비교한다.
def test_calibrate_replaces_the_threshold(client):
    before = _state(client)["threshold"]
    # 더 밝은 라인: 새 조건의 이미지는 기존 임계값으로 모두 불량이 된다.
    bright = _bright(30)
    assert _inspect(client, _png(bright[0])).json()["is_defect"] is True

    response = _calibrate(client, bright[:25], alpha="0.05")
    assert response.status_code == 200
    body = response.json()
    assert body["previous_threshold"] == pytest.approx(before) and body["threshold"] > before
    assert body["alpha"] == 0.05
    assert body["calibration"] == {"strategy": "holdout", "n": 25, "guaranteed": True}
    assert _state(client)["threshold"] == pytest.approx(body["threshold"])
    # 보정에 쓰지 않은 새 조건의 이미지는 이제 대부분 통과한다.
    flagged = [_inspect(client, _png(im)).json()["is_defect"] for im in bright[25:]]
    assert sum(flagged) <= 1

    # 서비스는 잠금 밖에서 점수를 매기고 규칙을 직접 적용한다: 결과는 검사기 자체의 보정과 같다.
    twin = service.offline_inspector()
    twin.calibrate([Image.open(io.BytesIO(_png(im))) for im in bright[:25]], alpha=0.05)
    state = _state(client)
    assert state["threshold"] == pytest.approx(twin.threshold)
    assert state["alpha"] == twin.meta["alpha"] and state["calibration"] == twin.meta["calibration"]


# 등각 순위가 장수 안에 들어오지 않을 만큼 적은 이미지(α 5%에서 19장 미만, 10%에서 9장 미만)는 400으로
# 거절하고 필요한 장수를 알려 준다. allow_unguaranteed를 켜면 받아들이되 보정 기록에 '보장 안 됨'으로 남긴다.
# 거절된 호출은 상태를 바꾸지 않는다.
def test_too_few_calibration_images_need_an_explicit_flag(client):
    before = _state(client)
    refused = _calibrate(client, _bright(3))
    assert refused.status_code == 400 and "at least 19" in refused.json()["detail"]
    assert _calibrate(client, _bright(18)).status_code == 400
    assert _calibrate(client, _bright(8), alpha="0.1").status_code == 400  # 10%에서는 9장이 필요하다
    assert _state(client) == before

    few = _calibrate(client, _bright(3), allow_unguaranteed="true")
    assert few.status_code == 200
    assert few.json()["calibration"] == {"strategy": "holdout", "n": 3, "guaranteed": False}
    assert few.json()["alpha"] == 0.05
    assert _calibrate(client, _bright(19)).json()["calibration"]["guaranteed"] is True
    assert _calibrate(client, _bright(9), alpha="0.1").json()["calibration"]["guaranteed"] is True


# 보정 요청 검증: 모르는 범주 404, alpha가 (0, 0.5] 밖이거나 숫자가 아니면 422, 이미지 없음 422, 읽을 수 없는
# 이미지 400, 파일 수 한도 초과 413. 거절된 호출 뒤 임계값은 그대로다.
def test_calibrate_validation(client, monkeypatch):
    assert (
        client.post(
            "/calibrate", files=_files([_normal()]), data={"category": "nope"}, headers=ADMIN
        ).status_code
        == 404
    )
    for alpha in ("0", "-0.1", "0.6", "abc"):
        assert _calibrate(client, [_normal()] * 19, alpha=alpha).status_code == 422
    assert client.post("/calibrate", data={"category": "demo"}, headers=ADMIN).status_code == 422
    bad = [("images", ("0.png", b"junk", "image/png"))] * 19
    assert client.post("/calibrate", files=bad, data={"category": "demo"}, headers=ADMIN).status_code == 400
    monkeypatch.setattr(service, "MAX_CALIBRATION_FILES", 1)
    assert _calibrate(client, [_normal()] * 2, allow_unguaranteed="true").status_code == 413
    # 거절된 호출은 임계값을 그대로 둔다.
    assert _state(client)["calibration"]["n"] == 24


# 관리자 토큰: 토큰을 설정하지 않으면 /calibrate와 DELETE /calibrate가 아예 꺼져 있다(403 disabled). 토큰을
# 설정하면 틀린 토큰·빈 값·다른 헤더는 403이고, 거절은 업로드 본문을 읽기 전에 일어난다(pulled가 비어 있음).
# 토큰은 환경 변수(DEFECT_INSPECT_ADMIN_TOKEN, 컨테이너가 받는 방식)로도 줄 수 있다.
def test_calibration_needs_the_admin_token(monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_ADMIN_TOKEN", raising=False)
    white = [np.full((64, 64, 3), 255, np.uint8)] * 19

    # 토큰을 설정하지 않음: HTTP로는 임계값을 아예 바꿀 수 없다.
    closed = TestClient(service.create_offline_app(), base_url=HOST)
    before = _state(closed)
    for headers in (None, ADMIN, {"X-Admin-Token": ""}):
        response = _calibrate(closed, white, headers=headers)
        assert response.status_code == 403 and "disabled" in response.json()["detail"]
    assert closed.delete("/calibrate", params={"category": "demo"}, headers=ADMIN).status_code == 403
    assert _state(closed) == before
    assert _inspect(closed, _png(_defect())).json()["is_defect"] is True

    client = _client()
    for headers in (
        None,
        {"X-Admin-Token": "wrong"},
        {"X-Admin-Token": TOKEN + "x"},
        {"Authorization": TOKEN},
    ):
        response = _calibrate(client, white, headers=headers)
        assert response.status_code == 403 and "X-Admin-Token" in response.json()["detail"]
        assert response.headers["x-content-type-options"] == "nosniff"
    assert _state(client) == before
    # 거절은 업로드를 읽기 전에 일어난다.
    header, head = _multipart_head("images")
    status, _, _, pulled = _asgi(client.app, "POST", "/calibrate", headers=header.items(), chunks=[head] * 3)
    assert status == 403 and pulled == []
    assert _calibrate(client, white).status_code == 200

    # 토큰은 환경 변수에서 올 수도 있다(컨테이너가 토큰을 받는 방식).
    monkeypatch.setenv("DEFECT_INSPECT_ADMIN_TOKEN", "from-env")
    env = TestClient(service.create_offline_app(), base_url=HOST)
    assert _calibrate(env, white, headers=ADMIN).status_code == 403
    assert _calibrate(env, white, headers={"X-Admin-Token": "from-env"}).status_code == 200


# 다른 출처의 페이지가 브라우저를 시켜 보낸 쓰기 요청(POST)을 거절하는지 확인한다. Sec-Fetch-Site가
# cross-site·same-site이거나, fetch 메타데이터가 없을 때 Origin이 Host와 다르거나 null이면 403이다.
# 읽기(GET)는 해가 없어 허용한다. 같은 출처, 직접 입력(none), Host를 바꾸는 프록시 뒤의 same-origin은
# 통과한다. docs/security.md에서 데모 폼의 Anti-CSRF 지적을 오탐으로 본 근거 가운데 하나다.
def test_requests_started_by_another_site_are_refused(client):
    before = _state(client)
    cross = [
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},  # 같은 기계의 다른 포트
        {"Origin": "http://evil.example"},  # fetch 메타데이터를 보내지 않는 브라우저
        {"Origin": "null"},
    ]
    for headers in cross:
        response = _calibrate(client, _bright(25), headers={**ADMIN, **headers})
        assert response.status_code == 403 and "cross-site" in response.json()["detail"]
        assert _inspect(client, _png(_normal()), headers=headers).status_code == 403
        assert client.get("/healthz", headers=headers).status_code == 200  # 읽기는 해가 없다
    assert _state(client) == before
    same = [
        {"Sec-Fetch-Site": "same-origin", "Origin": "http://127.0.0.1"},
        {"Sec-Fetch-Site": "none"},
        {"Origin": "http://127.0.0.1"},
        # Host를 바꿔 쓰는 프록시 뒤에서도 fetch 메타데이터는 실제 출처를 알려 준다.
        {"Sec-Fetch-Site": "same-origin", "Origin": "https://inspect.example"},
    ]
    for headers in same:
        assert _inspect(client, _png(_normal()), headers=headers).status_code == 200


# DELETE /calibrate는 토큰이 있어야 하고, 아티팩트를 읽었을 때의 임계값·alpha·보정 기록으로 정확히 되돌린다.
# 보정 기록이 없던 아티팩트는 되돌린 뒤에도 그 키가 없어야 한다(재보정한 값이 남지 않게).
def test_reset_restores_the_threshold_of_the_artifact(client):
    before = _state(client)
    changed = _calibrate(client, _bright(25), alpha="0.1").json()
    assert _state(client)["threshold"] == changed["threshold"] != before["threshold"]
    assert client.delete("/calibrate", params={"category": "demo"}).status_code == 403
    assert client.delete("/calibrate", params={"category": "nope"}, headers=ADMIN).status_code == 404
    reset = client.delete("/calibrate", params={"category": "demo"}, headers=ADMIN)
    assert reset.status_code == 200
    assert reset.json() == {
        "category": "demo",
        "previous_threshold": changed["threshold"],
        "threshold": before["threshold"],
        "alpha": before["alpha"],
        "calibration": before["calibration"],
    }
    assert _state(client) == before

    # 보정 기록이 없던 아티팩트는 되돌린 뒤에도 정확히 그 상태(기록 없음)로 돌아온다.
    inspector = service.offline_inspector()
    del inspector.meta["alpha"], inspector.meta["calibration"]
    bare = TestClient(service.create_app(inspectors={"demo": inspector}, admin_token=TOKEN), base_url=HOST)
    assert _calibrate(bare, _bright(25)).status_code == 200
    assert bare.delete("/calibrate", params={"category": "demo"}, headers=ADMIN).status_code == 200
    assert "alpha" not in inspector.meta and "calibration" not in inspector.meta
    assert _state(bare)["alpha"] is None and _state(bare)["calibration"] is None


# uvicorn이 로그 설정 없이 뜰 때 거는 기본 설정(uvicorn.config.LOGGING_CONFIG)을 이 테스트 동안만 적용한다.
# 왜: 서비스는 uvicorn --factory로 뜨고(Dockerfile의 기본 명령), 이 설정은 uvicorn.* 로거에만 핸들러를 단다.
# 그래서 다른 로거의 INFO 줄(logging.getLogger(__name__).info 등)은 어디에도 출력되지 않고, WARNING 이상만
# logging의 마지막 수단 핸들러로 시각 없이 표준 오류에 나간다. 감사 줄이 이 조건에서도 실제로 표준 출력에
# 나오는지 보려면 같은 설정을 건 뒤 capfd(파일 기술자 1·2를 잡는 캡처)로 읽어야 한다. caplog는 루트 로거에
# 자기 핸들러를 달아 '출력되지 않음'을 잡지 못한다. 감사 로거는 처음 만든 앱이 핸들러를 붙이므로, 새
# 프로세스처럼 핸들러가 없는 상태에서 시작한다. 주의: dictConfig는 프로세스 전체의 로깅 상태를 바꾼다(uvicorn
# 로거에 이 테스트의 캡처 스트림에 묶인 핸들러를 달고, 설정에 없는 로거의 disabled를 다시 쓴다). 그래서 끝나면
# 모든 로거의 핸들러·수준·propagate·disabled를 원래대로 돌려 다음 테스트에 영향이 없게 한다(이 테스트 중에
# 새로 생긴 로거는 기본 상태로).
@pytest.fixture
def uvicorn_logging():
    import copy
    import logging
    import logging.config

    uvicorn_config = pytest.importorskip("uvicorn.config")

    # 지금 있는 모든 로거(루트 포함). loggerDict에는 아직 로거가 없는 이름 자리(PlaceHolder)도 있어 뺀다.
    # list()로 한 번에 복사해, 도는 동안 다른 스레드가 로거를 만들어도 반복이 깨지지 않게 한다.
    def everyone() -> list:
        known = list(logging.root.manager.loggerDict.values())
        return [logging.root, *(log for log in known if isinstance(log, logging.Logger))]

    audit = logging.getLogger(AUDIT_LOGGER)
    saved = {log.name: (log.handlers[:], log.level, log.propagate, log.disabled) for log in everyone()}
    for handler in audit.handlers[:]:
        audit.removeHandler(handler)
    audit.setLevel(logging.NOTSET)
    audit.propagate = True
    logging.config.dictConfig(copy.deepcopy(uvicorn_config.LOGGING_CONFIG))
    try:
        yield
    finally:
        for log in everyone():
            handlers, level, propagate, disabled = saved.get(log.name, ([], logging.NOTSET, True, False))
            log.handlers[:] = handlers
            log.setLevel(level)
            log.propagate = propagate
            log.disabled = disabled


# capfd로 읽은 표준 출력을 감사 줄(JSON 객체)의 목록으로 바꾼다. 비어 있지 않은 줄은 모두 감사 줄이어야 한다:
# 다른 글이 섞이거나 한 기록이 여러 줄로 쪼개지면 json.loads가 실패해 테스트가 깨진다.
def _audit_lines(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.strip()]


# 재보정과 되돌리기는 임계값이 바뀔 때마다 표준 출력에 감사 줄(JSON 한 줄)을 남긴다: 언제(time), 무엇이(event:
# calibrate·reset), 어느 범주가 몇 장(n)·어떤 α로 보장 여부(guaranteed)와 함께 어떤 임계값에서 어떤
# 임계값으로(previous_threshold → threshold), 어디서(client: uvicorn 접근 로그와 같은 '주소:포트') 바꿨는지.
# 왜: 재보정한 임계값은 메모리에만 있어 다시 띄우면 사라지고, uvicorn 접근 로그에는 요청 줄과 상태 코드만 남아
# 판정 기준이 언제 무엇에서 무엇으로 바뀌었는지를 사후에 재구성할 수 없었다. reset 줄의 n·alpha·guaranteed는
# 되돌린 뒤(아티팩트)의 값이다. uvicorn 기본 로깅 설정에서 확인하고(uvicorn_logging), 앱을 둘 만들어도 줄이 두
# 번 나오지 않으며(핸들러는 없을 때만 한 번 붙임), 관리자 토큰과 업로드 파일 이름은 어느 출력에도 없고, 조회·
# 검사처럼 임계값을 바꾸지 않는 요청은 감사 줄을 남기지 않는다. 감사 줄은 표준 오류가 아니라 표준 출력에 쓴다.
def test_threshold_changes_are_audited_on_stdout_under_uvicorn_logging(uvicorn_logging, capfd):
    client = _client()
    _client()  # 두 번째 앱: 감사 핸들러를 또 붙이면 아래 줄이 두 번씩 나온다
    before = _state(client)
    capfd.readouterr()
    named = [("images", (f"line3-secret-{i}.png", _png(im), "image/png")) for i, im in enumerate(_bright(25))]
    changed = client.post("/calibrate", files=named, data={"category": "demo"}, headers=ADMIN)
    undone = client.delete("/calibrate", params={"category": "demo"}, headers=ADMIN)
    assert changed.status_code == 200 and undone.status_code == 200
    assert client.get("/categories").status_code == 200 and client.get("/healthz").status_code == 200
    assert _inspect(client, _png(_normal())).status_code == 200
    out, err = capfd.readouterr()

    lines = _audit_lines(out)
    assert [line["event"] for line in lines] == ["calibrate", "reset"]
    for line in lines:
        assert AUDIT_TIME.fullmatch(line["time"]) and line["level"] == "INFO"
    fields = [{k: v for k, v in line.items() if k not in ("time", "level")} for line in lines]
    calibrated = changed.json()
    assert fields[0] == {
        "event": "calibrate",
        "category": "demo",
        "n": 25,
        "alpha": 0.05,
        "guaranteed": True,
        "previous_threshold": before["threshold"],
        "threshold": calibrated["threshold"],
        "client": "testclient:50000",
    }
    assert fields[1] == {
        "event": "reset",
        "category": "demo",
        "n": before["calibration"]["n"],
        "alpha": before["alpha"],
        "guaranteed": before["calibration"]["guaranteed"],
        "previous_threshold": calibrated["threshold"],
        "threshold": before["threshold"],
        "client": "testclient:50000",
    }
    assert calibrated["previous_threshold"] == before["threshold"] != calibrated["threshold"]
    assert undone.json()["threshold"] == before["threshold"]
    # 감사 줄은 표준 출력에만 있고, 비밀값과 업로드 파일 이름은 어느 출력에도 남지 않는다.
    assert '"event"' not in err
    for secret in (TOKEN, "line3-secret"):
        assert secret not in out and secret not in err


# 관리자 토큰 때문에 403으로 거절한 요청은 사유를 담은 WARNING 감사 줄 하나를 남긴다. reason은 disabled(토큰을
# 설정하지 않아 /calibrate가 꺼져 있음), missing(X-Admin-Token 헤더가 없거나 빈 값), mismatch(값이 틀림)이고,
# method·path·client가 함께 남는다. 보낸 값이든 설정한 토큰이든 토큰 값은 남기지 않는다(틀린 값은 진짜 토큰과
# 한 글자만 다를 수 있다). 하위 경로(/calibrate/...)와 GET·DELETE도 같은 검사라 같은 줄이 남는다. path는
# 요청하는 쪽이 마음대로 고를 수 있는 필드라 uvicorn 접근 로그처럼 퍼센트 인코딩해 남긴다: 줄 바꿈과 JSON을
# 담은 경로로 가짜 감사 줄을 끼워 넣지 못하고(줄 수가 그대로), 디코딩하면 원래 경로가 된다. 라우트
# 수준의 같은 검사(require_admin, _Guard와 경로 판단이 어긋날 때의 이중 장치)가 거절해도 같은 줄이 남는다:
# uvicorn --root-path로 띄우면 scope의 path가 root_path로 시작해 _Guard는 관리자 경로로 보지 않고, 라우터가
# root_path를 떼어 /calibrate 라우트로 보낸 뒤 require_admin이 거절한다. 토큰이 맞는 요청과 조회는 거절 줄을
# 남기지 않는다.
def test_refused_admin_requests_are_audited_with_a_reason(uvicorn_logging, capfd, monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_ADMIN_TOKEN", raising=False)
    wrong = "not-the-admin-token-7f3a"
    closed = TestClient(service.create_offline_app(), base_url=HOST)  # 토큰을 설정하지 않음: 꺼져 있다
    client = _client()
    capfd.readouterr()
    white = [np.full((64, 64, 3), 255, np.uint8)] * 19
    assert _calibrate(closed, white).status_code == 403
    for headers in (None, {"X-Admin-Token": ""}, {"X-Admin-Token": wrong}):
        assert _calibrate(client, white, headers=headers).status_code == 403
    denied = client.delete("/calibrate", params={"category": "demo"}, headers={"X-Admin-Token": wrong})
    assert denied.status_code == 403
    assert client.get("/calibrate/anything").status_code == 403
    forged = '/calibrate/\n{"event": "calibrate", "threshold": 0}'
    assert _asgi(client.app, "POST", forged)[0] == 403
    rooted = _asgi(client.app, "POST", "/x/calibrate", root_path="/x", headers=[("X-Admin-Token", wrong)])
    assert rooted[0] == 403
    assert client.get("/healthz").status_code == 200
    assert _calibrate(client, white).status_code == 200
    out, err = capfd.readouterr()

    lines = _audit_lines(out)
    assert [line["event"] for line in lines] == ["admin_refused"] * 8 + ["calibrate"]
    refused = lines[:8]
    for line in refused:
        assert set(line) == {"time", "level", "event", "reason", "method", "path", "client"}
        assert AUDIT_TIME.fullmatch(line["time"]) and line["level"] == "WARNING"
    seen = [(line["reason"], line["method"], line["path"], line["client"]) for line in refused]
    assert seen[:6] == [
        ("disabled", "POST", "/calibrate", "testclient:50000"),
        ("missing", "POST", "/calibrate", "testclient:50000"),
        ("missing", "POST", "/calibrate", "testclient:50000"),
        ("mismatch", "POST", "/calibrate", "testclient:50000"),
        ("mismatch", "DELETE", "/calibrate", "testclient:50000"),
        ("missing", "GET", "/calibrate/anything", "testclient:50000"),
    ]
    # 가짜 줄을 끼우려던 경로: 한 줄로 남고, 인쇄 가능한 ASCII뿐이며, 디코딩하면 보낸 경로 그대로다.
    reason, method, path, address = seen[6]
    assert (reason, method, address) == ("missing", "POST", "127.0.0.1:50000")
    assert path.isascii() and path.isprintable() and urllib.parse.unquote(path) == forged
    assert seen[7] == ("mismatch", "POST", "/x/calibrate", "127.0.0.1:50000")
    for secret in (TOKEN, wrong):
        assert secret not in out and secret not in err


# 감사 기록 하나는 필드에 어떤 글자가 와도 한 줄이다. json.dumps는 따옴표·역슬래시와 U+0000~U+001F(줄 바꿈
# 포함)를 이스케이프하지만, ensure_ascii=False에서는 U+0085(NEL)·U+2028·U+2029를 그대로 둔다. 이 셋은
# str.splitlines()와 일부 로그 도구가 줄의 끝으로 보므로, 그대로 두면 한 기록이 두 줄로 쪼개지고 뒷부분이
# 따로 읽히는 가짜 줄이 된다. client 필드가 그런 글자를 담는 예: uvicorn은 믿는
# 프록시(--forwarded-allow-ips)에서 온 연결이면 X-Forwarded-For 값을 latin-1로 디코딩해 scope의 client로
# 쓰므로 U+0085가 들어올 수 있다. 감사 형식은 이 셋을 같은 뜻의 \u 이스케이프로 쓴다: 출력에 그 글자가 없고,
# JSON으로 읽으면 원래 값 그대로다. 한글 범주 이름처럼 줄을 끊지 않는 글자는 이스케이프하지 않고 그대로
# 읽히게 쓴다(ensure_ascii=False 유지).
def test_an_audit_record_stays_one_line_whatever_the_characters(uvicorn_logging, capfd):
    client = _client()
    korean = TestClient(
        service.create_app(inspectors={"캔들": service.offline_inspector()}, admin_token=TOKEN), base_url=HOST
    )
    capfd.readouterr()
    sneaky = '10.0.0.7\x85{"event": "calibrate", "threshold": 0}\u2028\u2029'
    assert _asgi(client.app, "POST", "/calibrate", peer=(sneaky, 0))[0] == 403
    assert korean.delete("/calibrate", params={"category": "캔들"}, headers=ADMIN).status_code == 200
    out, err = capfd.readouterr()

    lines = _audit_lines(out)
    assert [(line["event"], line.get("reason")) for line in lines] == [
        ("admin_refused", "missing"),
        ("reset", None),
    ]
    assert lines[0]["client"] == f"{sneaky}:0"
    assert not any(char in out for char in "\x85\u2028\u2029")
    assert lines[1]["category"] == "캔들" and '"category": "캔들"' in out
    assert '"event"' not in err


# 검사기 meta 대신 끼우는 dict. alpha·calibration에 접근할 때마다 그 순간 범주 잠금이 잡혀 있었는지 기록해,
# 임계값 상태를 잠금 밖에서 읽거나 쓰는 코드를 찾아낸다.
class _WatchedMeta(dict):
    """임계값 상태에 접근할 때마다 범주 잠금이 잡혀 있었는지 기록하는 검사기 meta."""

    watched = ("alpha", "calibration")

    def __init__(self, data, lock):
        super().__init__(data)
        self.lock = lock
        self.seen = []

    def _note(self, key):
        if key in self.watched:
            self.seen.append((key, self.lock.locked()))

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def __setitem__(self, key, value):
        self._note(key)
        super().__setitem__(key, value)


# 보정·조회·되돌리기의 모든 임계값 상태 접근이 범주 잠금 안에서 일어나는지 확인한다. 예전 처리기가 잠금을 푼
# 뒤 응답용으로 meta를 읽었던 문제(기존 주석)의 회귀를 막는다.
def test_threshold_state_is_only_touched_under_the_category_lock(client):
    inspector = client.app.state.inspectors["demo"]
    meta = _WatchedMeta(inspector.meta, client.app.state.locks["demo"])
    inspector.meta = meta
    assert _calibrate(client, _bright(25)).status_code == 200
    assert client.get("/categories").status_code == 200
    assert client.delete("/calibrate", params={"category": "demo"}, headers=ADMIN).status_code == 200
    # 예전 처리기는 잠금을 푼 뒤에 응답을 만들려고 meta["calibration"]을 읽었다.
    assert len(meta.seen) >= 6 and all(held for _, held in meta.seen)
    assert not client.app.state.locks["demo"].locked()


# 서로 다른 보정(25장 α 5%, 3장 α 10%)을 네 스레드에서 겹쳐 보내도, 각 응답과 그 직후 상태의 임계값·alpha·보정
# 기록이 한 보정에서 온 짝으로 맞아야 한다. 보정 결과의 일부만 보이는 상태(반쯤 바뀐 상태)가 없는지 확인한다.
# 서비스는 재보정을 한 번에 하나만 받으므로(다른 재보정이 진행 중이면 본문을 받기 전에 503 + Retry-After) 겹친
# 스레드는 503을 받으면 잠시 뒤 다시 보낸다. 그래도 한 스레드의 보정 쓰기는 다른 스레드의 /categories 읽기와
# 겹치므로, 범주 잠금 밖에서 상태를 읽거나 쓰는 코드는 여기서 드러난다.
def test_overlapping_calibrations_answer_with_their_own_record(client):
    sets = {25: ("0.05", _files(_bright(25))), 3: ("0.1", _files(_bright(3)))}
    expected = {}
    for n, (alpha, files) in sets.items():
        data = {"category": "demo", "alpha": alpha, "allow_unguaranteed": "true"}
        expected[n] = client.post("/calibrate", files=files, data=data, headers=ADMIN).json()
    assert expected[25]["threshold"] != pytest.approx(expected[3]["threshold"], rel=1e-3)

    def own(body, n) -> bool:
        """임계값, alpha, 보정 기록이 모두 `n`장으로 한 그 보정에서 온 것인지 본다."""
        return (
            body["threshold"] == pytest.approx(expected[n]["threshold"])
            and body["alpha"] == expected[n]["alpha"]
            and body["calibration"] == expected[n]["calibration"]
        )

    mixed = []

    def worker(n):
        alpha, files = sets[n]
        data = {"category": "demo", "alpha": alpha, "allow_unguaranteed": "true"}
        done = 0
        while done < 6:
            response = client.post("/calibrate", files=files, data=data, headers=ADMIN)
            if response.status_code == 503 and response.headers.get("retry-after") == "1":
                time.sleep(0.01)  # 다른 스레드의 재보정이 진행 중이다: 잠시 뒤 다시 보낸다
                continue
            if response.status_code != 200:
                mixed.append((n, response.status_code, response.text))
                break
            body, state = response.json(), _state(client)
            if not own(body, n) or not own(state, state["calibration"]["n"]):
                mixed.append((n, body, state))
            done += 1

    threads = [threading.Thread(target=worker, args=(n,)) for n in (25, 3, 25, 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert mixed == []


# 같은 범주의 검사 두 개가 모델 안에서 동시에 돌 수 있는지 확인한다. 모델 호출에 Barrier(2)를 걸어, 두 요청이
# 같이 들어오지 않으면 막혀 깨진다. 점수 계산은 잠금 밖에서 하므로(잠금은 임계값 상태만 보호) 범주 하나의
# 검사가 줄을 서지 않아야 한다.
def test_inspections_of_one_category_run_side_by_side(client):
    inspector = client.app.state.inspectors["demo"]
    real, together = inspector.session, threading.Barrier(2, timeout=10)

    class Meeting:
        def run(self, names, feeds):
            together.wait()  # 두 검사가 동시에 모델 안에 있지 않으면 깨진다
            return real.run(names, feeds)

    inspector.session = Meeting()
    codes = []
    threads = [
        threading.Thread(target=lambda: codes.append(_inspect(client, _png(_normal())).status_code))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert codes == [200, 200] and not together.broken


# 작업자 1, 대기 0으로 띄우고 첫 검사를 모델 안에 붙잡아 두면, 다음 검사와 보정은 503(Retry-After: 1)이고
# /healthz, /categories, 데모 페이지는 여전히 200이어야 한다. 이 처리기들은 async 함수라 작업자 스레드를
# 기다리지 않는다는 것도 확인한다. 첫 검사가 끝나면 자리가 다시 난다.
def test_a_full_queue_answers_503_and_the_health_check_stays_up():
    client = _client(workers=1, queue=0)
    inspector = client.app.state.inspectors["demo"]
    real, inside, release = inspector.session, threading.Event(), threading.Event()

    class Slow:
        def run(self, names, feeds):
            inside.set()
            assert release.wait(10)
            return real.run(names, feeds)

    inspector.session = Slow()
    first = []
    thread = threading.Thread(target=lambda: first.append(_inspect(client, _png(_normal()))))
    thread.start()
    try:
        assert inside.wait(10)
        busy = _inspect(client, _png(_normal()))
        assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
        assert _calibrate(client, _bright(25)).status_code == 503
        assert client.get("/healthz").status_code == 200
        assert client.get("/categories").status_code == 200 and client.get("/").status_code == 200
    finally:
        release.set()
        thread.join()
    assert first[0].status_code == 200
    assert _inspect(client, _png(_normal())).status_code == 200  # 자리가 다시 비었다

    # 상태 점검(probe)이 기대는 처리기는 작업자 스레드가 필요 없다.
    endpoints = {route.path: route.endpoint for route in client.app.routes if hasattr(route, "endpoint")}
    for path in ("/healthz", "/categories", "/"):
        assert pyinspect.iscoroutinefunction(endpoints[path])


# _Gate 단위 테스트: 수용량(작업자 + 대기)을 넘으면 503 HTTPException이고, 블록을 빠져나오면(안에서 예외가
# 나도) 자리가 돌아온다. run은 전용 스레드 풀에서 작업을 돌려 결과를 돌려준다. 작업자 0은 ValueError다.
def test_gate_admission():
    gate = service._Gate(workers=1, queue=1)
    with gate.admitted(), gate.admitted():
        with pytest.raises(service.HTTPException) as refused:
            with gate.admitted():
                pass
        assert refused.value.status_code == 503 and refused.value.headers == {"Retry-After": "1"}
    with gate.admitted():  # 블록을 빠져나오면 자리가 돌아온다(안에서 오류가 나도)
        with pytest.raises(KeyError):
            with gate.admitted():
                raise KeyError("x")
        with gate.admitted():
            assert asyncio.run(gate.run(lambda a, b: a + b, 1, 2)) == 3
    with pytest.raises(ValueError):
        service._Gate(workers=0, queue=1)


# 서버 없이 같은 이벤트 루프에서 요청을 겹치게 보낼 때, 요청마다 앱이 본문을 얼마나 받아 갔는지 기록하는
# ASGI 감싸개. TestClient는 본문을 한 번에 넘기지만 httpx의 ASGITransport는 앱이 receive()를 부를 때만 본문
# 조각(파일은 64KB씩)을 넘기므로, 거절된 요청의 본문이 앱에 한 조각도 들어가지 않았는지를 여기서 셀 수 있다.
# pulled에는 요청 순서대로 앱이 받아 간 http.request 조각들의 크기 목록이 쌓인다.
class _Counting:
    """요청마다 앱이 받아 간 본문 조각의 크기를 기록하는 ASGI 감싸개."""

    def __init__(self, app):
        self.app = app
        self.pulled: list[list[int]] = []

    async def __call__(self, scope, receive, send):
        chunks: list[int] = []
        self.pulled.append(chunks)

        async def counted():
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(len(message.get("body", b"")))
            return message

        await self.app(scope, counted, send)


# httpx.AsyncClient로 보낼 multipart 파일 목록. 파일을 BytesIO로 주면 httpx가 64KB 조각으로 보내, 앱이 본문을
# 어디까지 읽었는지가 조각 단위로 드러난다(bytes로 주면 파일 전체가 한 조각이 된다).
def _upload_files(field: str, images: list[bytes]) -> list:
    return [(field, (f"{i}.png", io.BytesIO(data), "image/png")) for i, data in enumerate(images)]


# 서버 없이 요청을 겹치게 보내는 비동기 클라이언트. 앱은 _Counting으로 감싸서 넘긴다.
def _async_client(counting: _Counting) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=counting), base_url=HOST)


# 작업자와 대기열이 모두 찼을 때 새 업로드가 본문을 한 조각도 받기 전에 503(Retry-After: 1)으로 거절되는지
# 확인한다. FastAPI는 엔드포인트가 돌기 전에 multipart 본문을 모두 받아 1MB가 넘는 파일을 /tmp에 스풀하므로,
# 입장 검사가 엔드포인트 안(_Gate)에만 있으면 거절될 요청의 20MB 업로드도 전부 받아 디스크에 쓴 뒤에야 503이
# 난다. 그래서 _Guard가 본문 전에 진행 중인 업로드 요청 수를 세어 거절해야 한다. 작업자 1·대기 0으로 띄우고
# 첫 검사를 모델 안에 붙잡은 채 2MB PNG를 보내면 503이고 앱이 받아 간 본문 조각이 없어야 하며, /calibrate도
# 같은 자리 수를 쓰므로 503이고, /healthz는 그대로 200이다. 첫 검사를 풀어 주면 같은 업로드가 끝까지 받아져
# 200이다.
def test_a_busy_service_refuses_an_upload_before_reading_its_body():
    app = service.create_offline_app(admin_token=TOKEN, workers=1, queue=0)
    inspector = app.state.inspectors["demo"]
    real, inside, release = inspector.session, threading.Event(), threading.Event()

    class Slow:
        def run(self, names, feeds):
            inside.set()
            assert release.wait(10)
            return real.run(names, feeds)

    inspector.session = Slow()
    counting = _Counting(app)
    big = _png(np.random.default_rng(0).integers(0, 256, (900, 900, 3), dtype=np.uint8))
    assert len(big) > 2 * 2**20
    calibration = [_png(im) for im in _bright(25)]

    async def scenario():
        async with _async_client(counting) as client:

            def inspect(data: bytes):
                files = _upload_files("image", [data])
                return client.post("/inspect", files=files, data={"category": "demo"})

            first = asyncio.create_task(inspect(_png(_normal())))
            assert await asyncio.to_thread(inside.wait, 10)
            busy = await inspect(big)
            files = _upload_files("images", calibration)
            full = await client.post("/calibrate", files=files, data={"category": "demo"}, headers=ADMIN)
            health = await client.get("/healthz")
            release.set()
            held = await first
            again = await inspect(big)
        return busy, full, health, held, again

    busy, full, health, held, again = asyncio.run(scenario())
    assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
    assert busy.json() == {"detail": "busy: try again shortly"}
    assert busy.headers["x-content-type-options"] == "nosniff"
    assert full.status_code == 503 and full.headers["retry-after"] == "1"
    assert health.status_code == 200
    # 거절된 두 요청(두 번째·세 번째)의 본문은 한 조각도 앱에 들어오지 않았다.
    assert counting.pulled[1] == [] and counting.pulled[2] == []
    # 자리가 나면 같은 업로드를 끝까지 받아 채점한다(multipart 틀까지 더해 파일보다 크다).
    assert held.status_code == 200 and again.status_code == 200
    assert sum(counting.pulled[4]) > len(big)


# 재보정은 한 번에 하나만 받는다: 첫 /calibrate가 모델 안에 붙잡혀 있는 동안 두 번째 /calibrate는 본문을
# 받기 전에 503(Retry-After: 1)이다. 재보정 본문은 256MB까지라 둘이 겹치면 /tmp가 가장 빨리 찬다. 그동안 검사
# 업로드는 자리가 남아 있으면(기본 작업자 2·대기 16) 그대로 들어가 모델 안에서 재보정과 나란히 돈다. 첫
# 재보정을 풀어 주면 둘 다 200이고 다음 재보정도 다시 받는다.
def test_only_one_calibration_runs_at_a_time_while_inspections_still_get_in():
    app = service.create_offline_app(admin_token=TOKEN)
    inspector = app.state.inspectors["demo"]
    real = inspector.session
    inside, release, overlap = threading.Event(), threading.Event(), threading.Event()
    calls = itertools.count()

    class Slow:
        def run(self, names, feeds):
            if next(calls) == 0:  # 첫 호출(첫 재보정의 첫 장)만 붙잡는다
                inside.set()
                assert release.wait(10)
            elif not release.is_set():
                overlap.set()  # 재보정이 붙잡혀 있는 동안 다른 작업이 모델 안에 들어왔다
            return real.run(names, feeds)

    inspector.session = Slow()
    counting = _Counting(app)
    images = [_png(im) for im in _bright(25)]

    async def scenario():
        async with _async_client(counting) as client:

            def calibrate():
                files = _upload_files("images", images)
                return client.post("/calibrate", files=files, data={"category": "demo"}, headers=ADMIN)

            first = asyncio.create_task(calibrate())
            assert await asyncio.to_thread(inside.wait, 10)
            second = await calibrate()
            inspected = await client.post(
                "/inspect", files=_upload_files("image", [_png(_normal())]), data={"category": "demo"}
            )
            release.set()
            held = await first
            again = await calibrate()
            state = await client.get("/categories")
        return second, inspected, held, again, state

    second, inspected, held, again, state = asyncio.run(scenario())
    assert second.status_code == 503 and second.headers["retry-after"] == "1"
    assert "calibration" in second.json()["detail"] and counting.pulled[1] == []
    # 검사는 재보정이 붙잡혀 있는 동안에도 들어가 모델 안에서 나란히 돌았다.
    assert inspected.status_code == 200 and overlap.is_set()
    assert held.status_code == 200 and again.status_code == 200
    assert state.json()[0]["calibration"] == {"strategy": "holdout", "n": 25, "guaranteed": True}


# 업로드를 /tmp에 스풀하는 동안 공간이 모자랐을 때(ENOSPC) 응답이 400이 아니라 503(Retry-After: 1)인지
# 확인한다. FastAPI는 multipart 해석 중의 예외를 400으로 바꾸는데, 400은 클라이언트 잘못이라 재시도 대상이
# 아니어서 정상 클라이언트가 결과를 잃는다. 공간 부족은 서버 쪽의 일시적 상태이므로 503으로 바꿔 재시도하게
# 하고, 운영자가 알 수 있게 경고 로그를 남긴다. 그 밖의 해석 오류(boundary 없는 multipart)는 그대로 400이고,
# 503 응답에 OS 오류 문구가 섞이지 않는다.
def test_running_out_of_scratch_space_during_an_upload_is_a_503_not_a_400(client, monkeypatch, caplog):
    from starlette.datastructures import UploadFile

    broken = client.post("/inspect", content=b"x", headers={"Content-Type": "multipart/form-data"})
    assert broken.status_code == 400

    async def no_space(self, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(UploadFile, "write", no_space)
    response = _inspect(client, _png(_normal()))
    assert response.status_code == 503 and response.headers["retry-after"] == "1"
    assert response.json() == {"detail": "busy: temporary storage is full, try again shortly"}
    assert "No space" not in response.text and response.headers["x-content-type-options"] == "nosniff"
    assert "ran out of space" in caplog.text
    assert _calibrate(client, _bright(25)).status_code == 503
    assert _state(client)["calibration"]["n"] == 24


# 허용된 Host 이름에만 답하는지 확인한다. 기본 목록은 포트와 대소문자를 무시하고 받으며, 접미사
# 위장(127.0.0.1.evil.example)이나 빈 Host는 400이다. 정적 파일 폴더의 리디렉션은 Location을 Host로 만들므로,
# 거절된 Host로는 리디렉션 자체가 나가지 않아야 한다. 환경 변수로 이름과 '*.도메인' 와일드카드를 줄 수 있고,
# '*'는 모두 허용한다.
def test_only_allowed_hosts_are_answered(monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_ALLOWED_HOSTS", raising=False)
    client = _client()
    for host in ("127.0.0.1", "127.0.0.1:8093", "localhost:8000", "LOCALHOST", "[::1]:8000"):
        assert client.get("/healthz", headers={"Host": host}).status_code == 200
    for host in ("evil.example", "127.0.0.1.evil.example", "evil.example:80", "[::2]", ""):
        response = client.get("/healthz", headers={"Host": host})
        assert response.status_code == 400 and response.json() == {"detail": "host not allowed"}
        assert response.headers["x-frame-options"] == "DENY"
    # 정적 파일 마운트의 리디렉션은 Host 헤더로 Location을 만든다.
    redirect = client.get("/static", headers={"Host": "evil.example"}, follow_redirects=False)
    assert redirect.status_code == 400 and "location" not in redirect.headers
    allowed = client.get("/static", headers={"Host": "localhost:8093"}, follow_redirects=False)
    assert allowed.status_code == 307 and allowed.headers["location"] == "http://localhost:8093/static/"

    monkeypatch.setenv("DEFECT_INSPECT_ALLOWED_HOSTS", "inspect.example, *.plant.example")
    named = TestClient(service.create_offline_app(), base_url="http://inspect.example")
    assert named.get("/healthz").status_code == 200
    assert named.get("/healthz", headers={"Host": "line3.plant.example:8000"}).status_code == 200
    assert named.get("/healthz", headers={"Host": "plant.example"}).status_code == 400
    assert named.get("/healthz", headers={"Host": "127.0.0.1"}).status_code == 400
    anyone = TestClient(service.create_offline_app(allowed_hosts=["*"]))
    assert anyone.get("/healthz").status_code == 200


# 데모 페이지는 인라인 스크립트 없이 /static/demo.js를 읽는다(CSP의 script-src 'self'와 맞춤). 페이지, 상태
# 점검, 404, 정적 파일 모두에 보안 헤더가 붙는다. CSP 아래에서 동작할 수 없는 대화형 문서(/docs, /redoc)는
# 내지 않지만, DAST 도구가 경로를 가져가는 /openapi.json은 남아 있어야 한다.
def test_demo_page_static_files_and_security_headers(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/static/demo.js" in page.text and "<script>" not in page.text  # 인라인 스크립트 없음
    assert client.get("/static/demo.js").status_code == 200
    assert client.get("/static/demo.css").status_code == 200
    for response in (page, client.get("/healthz"), client.get("/missing"), client.get("/static/demo.js")):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "default-src 'self'" in response.headers["content-security-policy"]
    # 대화형 문서는 이 정책 아래에서 동작할 수 없어(CDN의 스크립트를 쓴다) 내지 않는다.
    for path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
        assert client.get(path).status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/inspect", "/calibrate", "/categories", "/healthz"} <= set(paths)


# 정적 파일 경로로 패키지 밖 파일(service.py, pyproject.toml)을 읽을 수 없는지 확인한다. '..', 백슬래시, URL
# 인코딩된 점, 절대 경로 등 여러 탈출 시도를 _asgi로 정리 없이 보내 모두 404인지, 응답에 소스가 섞이지 않는지
# 본다.
def test_static_files_cannot_be_escaped(client):
    # 적은 그대로 앱에 보낸다: HTTP 클라이언트는 보내기 전에 점 경로 조각('.', '..')을 정리해 버린다.
    status, headers, body, _ = _asgi(client.app, "GET", "/static/demo.js")
    assert status == 200 and b"use strict" in body and headers["x-content-type-options"] == "nosniff"
    escapes = (
        "/static/../service.py",
        "/static/../../../pyproject.toml",
        "/static/..\\service.py",
        "/static/%2e%2e/service.py",
        "/static/./../service.py",
        "/static//" + str(Path(service.__file__).resolve()).replace("\\", "/"),
    )
    for path in escapes:
        status, headers, body, _ = _asgi(client.app, "GET", path)
        assert status == 404 and b"import" not in body, path
        assert headers["x-content-type-options"] == "nosniff"


# 아티팩트 경로가 없으면 환경 변수 이름을 알려 주는 RuntimeError, 경로에 검사기 아티팩트가 없으면 ValueError로
# 시작 단계에서 멈춘다.
def test_create_app_needs_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_ARTIFACTS", raising=False)
    with pytest.raises(RuntimeError, match="DEFECT_INSPECT_ARTIFACTS"):
        service.create_app()
    with pytest.raises(ValueError, match="no inspector artifacts"):
        service.create_app(tmp_path)


# 가짜 세션 감싸개: 모델 호출마다 받은 입력 텐서를 inputs에 남기고 실제 세션에 넘긴다. 시작 예열이 세션마다
# 몇 번, 어떤 입력으로 모델을 돌렸는지 세는 데 쓴다(위 _Counting은 ASGI 본문 조각을 세는 감싸개라 이름을
# 달리 한다).
class _RecordingSession:
    """모델 호출의 입력을 기록하는 세션 감싸개."""

    def __init__(self, real):
        self.real = real
        self.inputs: list[np.ndarray] = []

    def run(self, names, feeds):
        self.inputs.append(np.array(feeds["image"], copy=True))
        return self.real.run(names, feeds)


# 아티팩트에서 검사기를 읽는 경로(create_app에 경로를 줌)에서는 시작할 때 서로 다른 모델 세션마다
# 회색(128) 입력 크기 이미지로 검사 한 번을 돌려 둔다: 깨진 모델을 첫 요청 전에 잡고, onnxruntime이
# 첫 실행에서만 하는 일이 있으면 그 비용을 시작 단계에서 치른다(service.warm_up 주석).
# load_inspectors를 바꿔 끼워 세션 하나를 같이 쓰는 검사기 둘(a·b)과 자기 세션을 가진 검사기(c)를
# 돌려주면 공유 세션 1회·따로 1회여야 하고, 입력은 inspector.preprocess로 만든 회색 이미지와 같아야
# 한다. 예열은 임계값·보정 기록을 바꾸지 않고 예열 뒤 서비스는 그대로 답한다. 검사기를 직접
# 넘기면(테스트·오프라인 앱) 기본으로 예열하지 않고 warmup=True로 요구할 수 있으며, 아티팩트 경로라도
# warmup=False면 예열하지 않는다. warm_up 자체는 예열한 범주 이름을 이름 순으로 돌려준다(같은 세션의
# 두 번째 범주는 건너뜀).
def test_loading_artifacts_warms_up_each_model_session_once(tmp_path, monkeypatch):
    from defect_inspect.inspector import preprocess

    monkeypatch.delenv("DEFECT_INSPECT_PRECISION", raising=False)
    monkeypatch.delenv("DEFECT_INSPECT_THREADS", raising=False)
    shared, twin = service.offline_inspector(), service.offline_inspector()
    own = service.offline_inspector(seed=1)
    shared.session = twin.session = _RecordingSession(shared.session)
    own.session = _RecordingSession(own.session)
    asked = []

    def fake_load(artifacts, *, precision, threads):
        asked.append((Path(artifacts), precision, threads))
        return {"b": twin, "a": shared, "c": own}

    monkeypatch.setattr(service, "load_inspectors", fake_load)
    before = {name: (i.threshold, dict(i.meta)) for name, i in (("a", shared), ("c", own))}
    app = service.create_app(tmp_path, admin_token=TOKEN)
    assert asked == [(tmp_path, "fp32", None)]
    grey = preprocess(np.full((64, 64, 3), 128, np.uint8), 64)
    assert len(shared.session.inputs) == 1 and np.array_equal(shared.session.inputs[0], grey)
    assert len(own.session.inputs) == 1 and np.array_equal(own.session.inputs[0], grey)
    assert {name: (i.threshold, dict(i.meta)) for name, i in (("a", shared), ("c", own))} == before
    client = TestClient(app, base_url=HOST)
    assert client.get("/healthz").json() == {"status": "ok", "categories": ["a", "b", "c"]}
    assert _inspect(client, _png(_normal()), category="a").status_code == 200
    assert len(shared.session.inputs) == 2

    # 검사기를 직접 넘기면 기본으로 예열하지 않고, warmup으로 양쪽 모두 요구할 수 있다.
    direct = service.offline_inspector()
    direct.session = _RecordingSession(direct.session)
    service.create_app(inspectors={"demo": direct}, admin_token=TOKEN)
    assert direct.session.inputs == []
    service.create_app(inspectors={"demo": direct}, admin_token=TOKEN, warmup=True)
    assert len(direct.session.inputs) == 1
    shared.session.inputs.clear()
    own.session.inputs.clear()
    service.create_app(tmp_path, admin_token=TOKEN, warmup=False)
    assert shared.session.inputs == [] and own.session.inputs == []
    assert service.warm_up({"b": twin, "a": shared, "c": own}) == ["a", "c"]
    assert len(shared.session.inputs) == 1 and len(own.session.inputs) == 1


# 예열에서 모델이 실패하면(예상 밖 오류, 유한하지 않은 특징) 서비스는 뜨지 않고 그 범주 이름과 원인을 담은
# ValueError로 멈춘다. 깨진 모델을 첫 요청이 아니라 시작 단계에서 알리려는 의도된 동작이다
# (load_inspectors가 깨진 아티팩트를 알리는 방식과 같다).
def test_a_model_that_fails_the_warm_up_stops_the_service_from_starting(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_PRECISION", raising=False)
    monkeypatch.delenv("DEFECT_INSPECT_THREADS", raising=False)

    class Broken:
        def run(self, names, feeds):
            raise RuntimeError("the model fell over")

    class NotFinite:
        def run(self, names, feeds):
            out = service._GridMeanSession().run(names, feeds)[0]
            out[0, 0, 0, 0] = np.nan
            return [out]

    for session, message in ((Broken(), "fell over"), (NotFinite(), "non-finite features")):
        inspector = service.offline_inspector()
        inspector.session = session

        def fake_load(*args, found=inspector, **kwargs):
            return {"demo": found}

        monkeypatch.setattr(service, "load_inspectors", fake_load)
        with pytest.raises(ValueError, match=f"demo.*warm-up.*{message}"):
            service.create_app(tmp_path)


# load_inspectors가 같은 모델 파일을 쓰는 범주끼리 ONNX Runtime 세션 하나를 나눠 쓰는지 확인한다(범주마다
# 세션을 만들면 메모리와 시작 시간이 범주 수만큼 늘어난다는 소스 docstring의 이유). InferenceSession을 가짜로
# 바꿔, 상위 폴더의 공용 모델은 한 번, 자기 모델을 가진 pcb2는 따로 한 번 만들어지는지와 스레드 수·CPU 공급자
# 설정을 본다.
# 이어서 각 검사기의 모든 속성이 Inspector.load가 만든 것과 같은지(로더가 바뀌어도 이쪽이 뒤처지지 않게),
# 정밀도 검증, 그리고 아티팩트 문제(meta 불일치, 깨진 JSON, pickle이 든 뱅크, 모델 거부, 뱅크 없음)가 모두
# 폴더 이름을 담은 ValueError가 되는지 확인한다.
def test_load_inspectors_shares_one_session_per_model_file(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from defect_inspect.inspector import Inspector

    created = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = Path(path)
            created.append((self.path, sess_options.intra_op_num_threads, providers))

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    meta = {
        "version": 1,
        "name": "set",
        "backbone": "fake",
        "img_size": 64,
        "grid": [8, 8],
        "dim": 3,
        "reweight_k": 9,
        "sigma": 1.0,
        "alpha": 0.05,
    }
    (tmp_path / "model_fp32.onnx").write_bytes(b"shared")
    for index, name in enumerate(("candle", "pcb1", "pcb2")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "meta.json").write_text(
            json.dumps({**meta, "category": name, "threshold": 1.0 + index}), encoding="utf-8"
        )
        np.save(folder / "bank.npy", np.full((4 + index, 3), index, dtype=np.float16))
    (tmp_path / "pcb2" / "model_fp32.onnx").write_bytes(b"own")
    (tmp_path / "notes").mkdir()  # meta.json 없음: 범주가 아니다

    found = service.load_inspectors(tmp_path, threads=3)
    assert sorted(found) == ["candle", "pcb1", "pcb2"]
    assert found["candle"].session is found["pcb1"].session is not found["pcb2"].session
    assert [(p.parent.name, t, prov) for p, t, prov in created] == [
        (tmp_path.name, 3, ["CPUExecutionProvider"]),
        ("pcb2", 3, ["CPUExecutionProvider"]),
    ]
    # 각 검사기는 `Inspector.load`가 그 폴더로 만드는 것과 같아야 한다. 골라 낸 몇 개가 아니라 모든 속성을
    # 비교해, 로더가 바뀌었을 때 이쪽이 모르는 사이에 뒤처지지 않게 한다. 예외는 서비스 로더만 붙이는
    # model_sha256 하나(시작할 때 대조한 모델 파일 지문, /categories가 앞 12자를 보인다)이고, 이 세트의
    # meta에는 지문 기록이 없으므로 None이다. vars()는 인스턴스의 __dict__ 자체라 사본에서 뺀다.
    for name, inspector in found.items():
        alone = Inspector.load(tmp_path / name, threads=3)
        mine, theirs = dict(vars(inspector)), vars(alone)
        assert mine.pop("model_sha256") is None
        assert set(mine) == set(theirs)
        for key, value in mine.items():
            if key == "session":
                assert value.path == theirs[key].path
            elif isinstance(value, np.ndarray):
                assert value.dtype == theirs[key].dtype and np.array_equal(value, theirs[key])
            else:
                assert value == theirs[key], key
        assert (inspector.precision, inspector.threads) == (alone.precision, alone.threads) == ("fp32", 3)
    default = service.load_inspectors(tmp_path)["candle"]
    assert vars(default).get("threads") == vars(Inspector.load(tmp_path / "candle")).get("threads")

    with pytest.raises(ValueError, match="precision must be one of"):
        service.load_inspectors(tmp_path, precision="fp16")
    with pytest.raises(ValueError, match="model_int8.onnx is missing"):
        service.load_inspectors(tmp_path, precision="int8")
    # 아티팩트의 어떤 문제든 `Inspector.load`와 마찬가지로 폴더 이름을 담은 ValueError가 된다.
    good = (tmp_path / "pcb1" / "meta.json").read_text(encoding="utf-8")
    (tmp_path / "pcb1" / "meta.json").write_text(good.replace('"dim": 3', '"dim": 5'), encoding="utf-8")
    with pytest.raises(ValueError, match="pcb1"):
        service.load_inspectors(tmp_path)
    (tmp_path / "pcb1" / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="pcb1"):
        service.load_inspectors(tmp_path)
    (tmp_path / "pcb1" / "meta.json").write_text(good, encoding="utf-8")
    np.save(tmp_path / "pcb1" / "bank.npy", np.array([{"pickled": True}], dtype=object))
    with pytest.raises(ValueError, match="pcb1"):
        service.load_inspectors(tmp_path)
    np.save(tmp_path / "pcb1" / "bank.npy", np.zeros((4, 3), dtype=np.float16))

    def rejected(path, sess_options=None, providers=None):
        raise RuntimeError("INVALID_PROTOBUF")

    monkeypatch.setattr(ort, "InferenceSession", rejected)
    with pytest.raises(ValueError, match="candle.*INVALID_PROTOBUF"):
        service.load_inspectors(tmp_path)
    (tmp_path / "candle" / "bank.npy").unlink()
    with pytest.raises(ValueError, match="bank.npy is missing"):
        service.load_inspectors(tmp_path)


# 히트맵 PNG의 밝기 척도: 점수 0은 검정, 임계값의 1.5배 이상은 흰색(255), 그 사이는 선형(0.75 → 128).
def test_heatmap_png_scale():
    heat = np.array([[0.0, 0.75], [1.5, 3.0]], dtype=np.float32)
    grey = np.asarray(Image.open(io.BytesIO(base64.b64decode(service.heatmap_png(heat, 1.0)))))
    assert grey.tolist() == [[0, 128], [255, 255]]


# 유한하지 않은 값이 든 맵은 PNG로 그리지 않고 FloatingPointError를 낸다. NaN은 0~1로 자르고 uint8로 바꾸는
# 과정에서 0이 되어 '점수 0'(검정)처럼 조용히 그려지기 때문이다. 재구성 검사기는 run에서 먼저 막지만,
# PatchCore 경로의 맵은 여기가 마지막 방어선이다. 임계값이 0 이하여서 맵 최댓값으로 나누는 길(합성 범주)도
# 같다.
def test_heatmap_png_refuses_a_map_that_is_not_finite():
    for bad in (np.nan, np.inf, -np.inf):
        heat = np.array([[0.0, bad], [1.5, 3.0]], dtype=np.float32)
        for threshold in (1.0, 0.0):
            with pytest.raises(FloatingPointError):
                service.heatmap_png(heat, threshold)


# _DEMO_HARNESS: node에서 demo.js를 vm 샌드박스로 돌리는 JavaScript 하네스(demo.js용 브라우저 대역). 폼을
# 제출하고 페이지가 보여 주는 것을 살피는 데 필요한 만큼의 가짜 DOM·fetch·Image를 두고, 서버 응답 목록을
# 차례로 돌려주며 매 제출 뒤 화면 상태(결과 숨김, 판정, 점수, 임계값, 상태 문구, 버튼, 캔버스에 그린 것)를
# 기록한다. URL.createObjectURL을 부르면 실패하게 해 페이지가 로컬 파일을 직접 디코딩하지 않는지도 본다.
_DEMO_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const elements = {};
const drawn = [];
const ctx = {
  globalAlpha: 1,
  drawImage(image) { drawn.push(image.src ? image.src.slice(0, 22) : "buffer"); },
  clearRect() { drawn.push("clear"); },
  getImageData() { return { data: new Uint8ClampedArray(16) }; },
  putImageData() {},
};
function el(id) {
  if (!elements[id]) {
    elements[id] = {
      hidden: id === "result", textContent: "", className: "", value: id === "opacity" ? "60" : "demo",
      disabled: false, files: [{ name: "part.tiff" }], width: 384, height: 384, listeners: {},
      classList: { toggle() {} },
      addEventListener(type, handler) { this.listeners[type] = handler; },
      appendChild() {},
      getContext() { return ctx; },
    };
  }
  return elements[id];
}
const replies = JSON.parse(process.argv[3]);
const requests = [];
const sandbox = {
  document: {
    getElementById: el,
    createElement: (tag) => (tag === "canvas" ? { width: 0, height: 0, getContext: () => ctx } : {}),
  },
  fetch: async (url) => {
    if (url === "/categories") return { ok: true, status: 200, json: async () => [{ category: "demo" }] };
    requests.push(url);
    const reply = replies.shift();
    if (reply === "network") throw new Error("connection refused");
    return {
      ok: reply.status === 200,
      status: reply.status,
      json: async () => {
        if (reply.body === undefined) throw new SyntaxError("not JSON");
        return reply.body;
      },
    };
  },
  FormData: class { append() {} },
  Image: class {
    set src(value) {
      this.source = value;
      this.width = this.height = 256;
      setTimeout(() => (value.includes("BROKEN") ? this.onerror(new Error("decode")) : this.onload()), 0);
    }
    get src() { return this.source; }
  },
  URL: { createObjectURL() { throw new Error("the page must not decode the local file"); } },
  setTimeout,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
(async () => {
  const shown = [];
  const count = replies.length;
  for (let i = 0; i < count; i += 1) {
    drawn.length = 0;
    await el("form").listeners.submit({ preventDefault() {} });
    shown.push({
      hidden: el("result").hidden, verdict: el("verdict").textContent, score: el("score").textContent,
      threshold: el("threshold").textContent, status: el("status").textContent,
      disabled: el("submit").disabled, drawn: drawn.slice(),
    });
  }
  console.log(JSON.stringify({ shown, requests }));
})();
"""


# 데모 페이지가 새 업로드 옆에 이전 판정을 남기지 않는지 확인한다(node가 없으면 건너뛴다). 성공, 400 거절,
# 네트워크 오류, JSON이 아닌 502, 그릴 수 없는 미리보기, 422, Dinomaly 크기의 작은 점수 순서로 응답을 주고
# 화면을 본다. 오류면 결과를 숨기고 상태 문구를 보여야 하며, 미리보기를 그릴 수 없어도 이번 판정은 보이되 이전
# 그림은 지운다. 그림은 항상 서버의 preview(모델이 본 그림)에서 오고, 점수는 유효숫자 4자리로 보여 0.01 근처의
# 점수와 임계값이 같은 숫자로 보이지 않게 한다.
def test_demo_page_never_shows_the_previous_verdict_next_to_a_new_upload(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "harness.js"
    harness.write_text(_DEMO_HARNESS, encoding="utf-8", newline="\n")
    defect = {"is_defect": True, "score": 3.3421, "threshold": 0.625, "latency_ms": 12.5}
    good = {"is_defect": False, "score": 0.0018, "threshold": 0.625, "latency_ms": 9.0}
    # Dinomaly의 점수와 임계값은 0.01 근처다: 소수 셋째 자리까지만 보이면 둘 다 0.018로 보인다.
    close = {"is_defect": True, "score": 0.0181234, "threshold": 0.017962466925382614, "latency_ms": 150.0}
    replies = [
        {"status": 200, "body": {**defect, "heatmap_png": "HEAT", "input_png": "SEEN"}},
        {"status": 400, "body": {"detail": "the upload is not a readable image"}},
        {"status": 200, "body": {**defect, "heatmap_png": "HEAT", "input_png": "SEEN"}},
        "network",
        {"status": 502},  # JSON 대신 프록시 오류 페이지
        {"status": 200, "body": {**good, "heatmap_png": "HEAT", "input_png": "BROKEN"}},
        {
            "status": 422,
            "body": {"detail": [{"loc": ["body", "image"], "type": "missing", "msg": "Field required"}]},
        },
        {"status": 200, "body": {**close, "heatmap_png": "HEAT", "input_png": "SEEN"}},
    ]
    done = subprocess.run(
        [node, str(harness), str(service.STATIC_DIR / "demo.js"), json.dumps(replies)],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    # 그림은 서버(모델이 본 그림)에서 오므로 형식이나 EXIF 회전 때문에 달라질 수 없다.
    assert result["requests"] == ["/inspect?preview=true"] * len(replies)
    first, rejected, again, offline, proxy, undrawable, invalid, dinomaly = result["shown"]
    for shown in (first, again):
        assert (shown["hidden"], shown["verdict"], shown["score"], shown["threshold"], shown["status"]) == (
            False,
            "불량",
            "3.342",
            "0.625",
            "",
        )
        assert "data:image/png;base64," in shown["drawn"] and "buffer" in shown["drawn"]
    assert rejected["hidden"] is True and rejected["status"] == "the upload is not a readable image"
    for shown in (offline, proxy, invalid):
        assert shown["hidden"] is True and shown["status"] != ""
    # 그림을 그릴 수 없어도 이번 업로드의 판정은 안내 문구와 함께 보이고, 캔버스에 이전 그림이 남지 않는다.
    assert (undrawable["hidden"], undrawable["verdict"], undrawable["score"]) == (False, "양품", "0.0018")
    assert undrawable["status"] != "" and "clear" in undrawable["drawn"]
    assert not any(item.startswith("data:") for item in undrawable["drawn"])
    assert all(shown["disabled"] is False for shown in result["shown"])
    # 유효숫자 4자리면 소수 넷째 자리에서 갈리는 점수와 임계값이 서로 다르게 보인다.
    assert (dinomaly["verdict"], dinomaly["score"], dinomaly["threshold"]) == ("불량", "0.01812", "0.01796")

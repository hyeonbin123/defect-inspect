# [파일 역할] FastAPI 검사 서비스(`defect_inspect.service`)를 검증하는 테스트 묶음이다. 모델 파일 없이 도는
# 오프라인 앱(합성 범주 demo, 작은 PatchCore 흉내)으로 /inspect, /calibrate, DELETE /calibrate, /categories,
# /healthz, 데모 페이지와 정적 파일을 시험한다.
# [왜 필요한가] 이 서비스는 업로드를 받아 판정하고 관리자가 임계값을 바꿀 수 있으므로, 점수 정확성뿐 아니라
# 입력 검증(형식·크기·16비트·투명도), 보호 장치(Host 검사, 다른 출처의 쓰기 요청 거절, 관리자 토큰, 본문 크기
# 제한, 보안 헤더), 동시성(범주별 잠금, 작업자 수 제한과 503)이 모두 맞아야 한다. docs/security.md의 DAST
# 판정(예: Anti-CSRF 지적을 오탐으로 본 근거)도 이 파일의 테스트에 기대고 있다.
# [주요 구성] 요청 도우미와 ASGI 직접 호출 → 판정·히트맵·입력 형식 → 잘못된 요청과 크기 제한 →
# 손상·16비트·투명도·미리보기 → 재보정과 되돌리기 → 관리자 토큰·다른 출처 거절 → 잠금과 동시 요청·대기열 →
# Host 검사·보안 헤더·정적 파일 → 아티팩트 읽기(load_inspectors) → 데모 페이지 JS(node가 있을 때).

import asyncio
import base64
import inspect as pyinspect
import io
import itertools
import json
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

# 서비스는 serve 의존성 그룹(FastAPI, 업로드 파싱용 python-multipart)이 있어야 돈다. 기본 패키지만 설치한
# 환경에서는 이 파일 전체를 건너뛴다. torch는 필요 없다.
pytest.importorskip("fastapi")
pytest.importorskip("multipart")

# TestClient는 실제 서버를 띄우지 않고 앱을 같은 프로세스에서 HTTP처럼 부른다.
from fastapi.testclient import TestClient  # noqa: E402

from defect_inspect import service  # noqa: E402

# 기본 허용 Host(127.0.0.1)로 요청하는 주소, 테스트용 관리자 토큰과 그 헤더, 모든 응답에 붙어야 하는 보안 헤더
# 이름들.
HOST = "http://127.0.0.1"
TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}
SECURITY = ("x-content-type-options", "x-frame-options", "content-security-policy", "referrer-policy")


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
    """Normal images of a brighter line: all flagged by the threshold the offline inspector starts with."""
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
# 그대로 시험할 수 있다.
def _asgi(app, method: str, path: str, *, headers=(), chunks=()):
    """Call the ASGI app itself: the path is not normalised by a client and the body arrives in pieces.

    Returns (status, headers, body, sizes of the body pieces the app asked for).
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
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1"), *((k.lower().encode(), v.encode()) for k, v in headers)],
        "client": ("127.0.0.1", 50000),
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
    """Header and the opening of a multipart body whose file part never has to end."""
    head = (
        b'--B\r\nContent-Disposition: form-data; name="category"\r\n\r\ndemo\r\n'
        b'--B\r\nContent-Disposition: form-data; name="' + field.encode() + b'"; filename="x.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
    )
    return {"Content-Type": "multipart/form-data; boundary=B"}, head


# /healthz와 /categories의 기본 응답. 오프라인 검사기는 정상 24장으로 보정돼 있어 보정 기록이 holdout, n 24,
# 보장됨이다. 헤더만 묻는 HEAD 요청(상태 점검 도구가 쓰는 방식)도 200이어야 한다.
def test_health_and_categories(client):
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok", "categories": ["demo"]}
    items = client.get("/categories").json()
    assert [i["category"] for i in items] == ["demo"]
    assert items[0]["img_size"] == 64 and items[0]["threshold"] > 0
    assert items[0]["calibration"] == {"strategy": "holdout", "n": 24, "guaranteed": True}
    # Health checks that only ask for the headers work too.
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
    # The bright square covers rows/cols 80..144 on the 256 grid: hottest there, cold in the far corner.
    assert heat[112, 112] == heat.max() == 255
    assert heat[200:, 200:].max() < 128

    without = _inspect(client, _png(_defect(seed=3)), heatmap="false").json()
    assert "heatmap_png" not in without and without["score"] == pytest.approx(bad["score"])


# 크기·비율이 다른 회색조, JPEG, 팔레트, BMP, WebP 모두 받아 검사기 자체의 파이프라인과 같은 점수를 내는지
# 본다. 서비스는 디코딩 직후 모델 입력 크기로 줄이는데(큰 원본을 메모리에 두지 않으려고), 그 리사이즈가
# inspector.preprocess와 같은 BICUBIC 호출이라 점수가 바뀌지 않아야 한다.
def test_any_image_size_and_8_bit_mode_is_scored_like_the_inspector_itself(client):
    twin = service.offline_inspector()
    big = np.random.default_rng(0).integers(110, 146, (300, 480), dtype=np.uint8)  # greyscale, not square
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
        # The service resizes right after decoding; the score is the one of the plain pipeline.
        assert response.json()["score"] == pytest.approx(twin.inspect(Image.open(io.BytesIO(data))).score)
    decoded = service._decode(_png(big), 64)
    assert decoded.mode == "RGB" and decoded.size == (64, 64)
    expected = Image.fromarray(big).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
    assert np.array_equal(np.asarray(decoded), np.asarray(expected))


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
        client.post("/inspect", data={"category": "demo", "image": "hello"}),  # text where a file belongs
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
    # Announced: refused before a single byte of the body is handed to the app.
    status, out, _, pulled = _asgi(
        app, "POST", "/inspect", headers={**header, "Content-Length": "999999999"}.items(), chunks=[head]
    )
    assert status == 413 and pulled == [] and out["x-content-type-options"] == "nosniff"
    # Chunked (a generator body has no Content-Length): the old check did not look at these at all.
    chunked = client.post("/inspect", content=iter([whole]), headers=header)
    assert "content-length" not in chunked.request.headers and chunked.status_code == 413
    # Arriving in pieces: the service stops asking for more right after the limit instead of spooling it.
    pieces = itertools.chain([head], itertools.repeat(b"\0" * 500, 400))
    status, out, payload, pulled = _asgi(app, "POST", "/inspect", headers=header.items(), chunks=pieces)
    assert status == 413 and json.loads(payload) == {"detail": "request body too large"}
    assert sum(pulled) <= 3000 + 500 and out["connection"] == "close"
    assert out["x-content-type-options"] == "nosniff"

    # A calibration call has its own, larger limit for the whole body.
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
    # Halve the declared length of the data chunk: Pillow then reads a chunk header out of the compressed
    # bytes and raises SyntaxError, not OSError. (Flipping a byte of the length can also make it longer,
    # which still decodes; whether it grows depends on the zlib build, so the damage is written out.)
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

    # A white patch that no viewer shows: fully transparent pixels with white underneath.
    hidden = np.dstack([_defect(), alpha])
    hidden[20:36, 20:36, 3] = 0
    keyed = _encode(Image.fromarray(_normal()).quantize(8), transparency=0)  # palette entry 0 is transparent
    for data in (_png(hidden), keyed):
        response = _inspect(client, data)
        assert response.status_code == 400 and "transparency" in response.json()["detail"]


# preview=true의 input_png가 모델이 실제로 본 그림인지 확인한다. EXIF 회전 정보가 있는 JPEG도 회전하지 않은
# 저장 픽셀을 채점하고(모델과 뱅크를 만든 리사이즈 캐시와 같은 방식), 미리보기·점수·히트맵이 모두 같은 그림을
# 가리킨다. 브라우저가 그리지 못하는 TIFF도 미리보기가 오고, 요청하지 않으면 미리보기를 보내지 않는다.
def test_preview_is_the_picture_the_model_saw(client):
    stored = np.random.default_rng(2).integers(110, 146, (48, 80, 3), dtype=np.uint8)
    stored[2:14, 2:20] = 255  # top-left in the stored pixels
    exif = Image.Exif()
    exif[0x0112] = 6  # viewers show this file rotated by 90 degrees
    rotated = _encode(stored, "JPEG", quality=95, exif=exif)
    plain = _encode(stored, "JPEG", quality=95)
    assert ImageOps.exif_transpose(Image.open(io.BytesIO(rotated))).size == (48, 80)

    body = _inspect(client, rotated, preview="true").json()
    seen = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["input_png"]))))
    expected = Image.open(io.BytesIO(plain)).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
    # The stored pixels, not the rotated view: score, heatmap and preview all refer to the same picture.
    assert seen.shape == (64, 64, 3) and np.array_equal(seen, np.asarray(expected))
    assert body["score"] == pytest.approx(_inspect(client, plain).json()["score"])
    heat = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["heatmap_png"]))))
    assert heat[:128, :128].max() == heat.max() and seen[:16, :16].mean() > 200

    # A format browsers cannot draw still gets a preview, and no preview is sent unless asked for.
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
    # A brighter line: every image of the new condition is flagged with the old threshold.
    bright = _bright(30)
    assert _inspect(client, _png(bright[0])).json()["is_defect"] is True

    response = _calibrate(client, bright[:25], alpha="0.05")
    assert response.status_code == 200
    body = response.json()
    assert body["previous_threshold"] == pytest.approx(before) and body["threshold"] > before
    assert body["alpha"] == 0.05
    assert body["calibration"] == {"strategy": "holdout", "n": 25, "guaranteed": True}
    assert _state(client)["threshold"] == pytest.approx(body["threshold"])
    # Images of the new condition that were not used for calibration now mostly pass.
    flagged = [_inspect(client, _png(im)).json()["is_defect"] for im in bright[25:]]
    assert sum(flagged) <= 1

    # The service scores outside the lock and applies the rule itself: same result as the inspector's own.
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
    assert _calibrate(client, _bright(8), alpha="0.1").status_code == 400  # 9 are needed at 10%
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
    # A rejected call leaves the threshold as it was.
    assert _state(client)["calibration"]["n"] == 24


# 관리자 토큰: 토큰을 설정하지 않으면 /calibrate와 DELETE /calibrate가 아예 꺼져 있다(403 disabled). 토큰을
# 설정하면 틀린 토큰·빈 값·다른 헤더는 403이고, 거절은 업로드 본문을 읽기 전에 일어난다(pulled가 비어 있음).
# 토큰은 환경 변수(DEFECT_INSPECT_ADMIN_TOKEN, 컨테이너가 받는 방식)로도 줄 수 있다.
def test_calibration_needs_the_admin_token(monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_ADMIN_TOKEN", raising=False)
    white = [np.full((64, 64, 3), 255, np.uint8)] * 19

    # No token configured: thresholds cannot be changed over HTTP at all.
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
    # The refusal comes before the upload is read.
    header, head = _multipart_head("images")
    status, _, _, pulled = _asgi(client.app, "POST", "/calibrate", headers=header.items(), chunks=[head] * 3)
    assert status == 403 and pulled == []
    assert _calibrate(client, white).status_code == 200

    # The token can come from the environment (the way the container gets it).
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
        {"Sec-Fetch-Site": "same-site"},  # another port of the same machine
        {"Origin": "http://evil.example"},  # a browser without fetch metadata
        {"Origin": "null"},
    ]
    for headers in cross:
        response = _calibrate(client, _bright(25), headers={**ADMIN, **headers})
        assert response.status_code == 403 and "cross-site" in response.json()["detail"]
        assert _inspect(client, _png(_normal()), headers=headers).status_code == 403
        assert client.get("/healthz", headers=headers).status_code == 200  # reading is harmless
    assert _state(client) == before
    same = [
        {"Sec-Fetch-Site": "same-origin", "Origin": "http://127.0.0.1"},
        {"Sec-Fetch-Site": "none"},
        {"Origin": "http://127.0.0.1"},
        # Behind a proxy that rewrites Host, the fetch metadata still tells the truth.
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

    # An artifact without a calibration record gets back exactly that.
    inspector = service.offline_inspector()
    del inspector.meta["alpha"], inspector.meta["calibration"]
    bare = TestClient(service.create_app(inspectors={"demo": inspector}, admin_token=TOKEN), base_url=HOST)
    assert _calibrate(bare, _bright(25)).status_code == 200
    assert bare.delete("/calibrate", params={"category": "demo"}, headers=ADMIN).status_code == 200
    assert "alpha" not in inspector.meta and "calibration" not in inspector.meta
    assert _state(bare)["alpha"] is None and _state(bare)["calibration"] is None


# 검사기 meta 대신 끼우는 dict. alpha·calibration에 접근할 때마다 그 순간 범주 잠금이 잡혀 있었는지 기록해,
# 임계값 상태를 잠금 밖에서 읽거나 쓰는 코드를 찾아낸다.
class _WatchedMeta(dict):
    """Inspector meta that records whether the category lock was held at each threshold-state access."""

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
    # The old handler read meta["calibration"] for its response after the lock was released.
    assert len(meta.seen) >= 6 and all(held for _, held in meta.seen)
    assert not client.app.state.locks["demo"].locked()


# 서로 다른 보정(25장 α 5%, 3장 α 10%)을 네 스레드에서 겹쳐 보내도, 각 응답과 그 직후 상태의 임계값·alpha·보정
# 기록이 한 보정에서 온 짝으로 맞아야 한다. 보정 결과의 일부만 보이는 상태(반쯤 바뀐 상태)가 없는지 확인한다.
def test_overlapping_calibrations_answer_with_their_own_record(client):
    sets = {25: ("0.05", _files(_bright(25))), 3: ("0.1", _files(_bright(3)))}
    expected = {}
    for n, (alpha, files) in sets.items():
        data = {"category": "demo", "alpha": alpha, "allow_unguaranteed": "true"}
        expected[n] = client.post("/calibrate", files=files, data=data, headers=ADMIN).json()
    assert expected[25]["threshold"] != pytest.approx(expected[3]["threshold"], rel=1e-3)

    def own(body, n) -> bool:
        """Threshold, alpha and calibration record all belong to the calibration with `n` images."""
        return (
            body["threshold"] == pytest.approx(expected[n]["threshold"])
            and body["alpha"] == expected[n]["alpha"]
            and body["calibration"] == expected[n]["calibration"]
        )

    mixed = []

    def worker(n):
        alpha, files = sets[n]
        data = {"category": "demo", "alpha": alpha, "allow_unguaranteed": "true"}
        for _ in range(6):
            body = client.post("/calibrate", files=files, data=data, headers=ADMIN).json()
            state = _state(client)
            if not own(body, n) or not own(state, state["calibration"]["n"]):
                mixed.append((n, body, state))

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
            together.wait()  # breaks unless two inspections are inside the model at the same time
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
    assert _inspect(client, _png(_normal())).status_code == 200  # the slot is free again

    # Handlers a probe depends on do not need a worker thread.
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
    with gate.admitted():  # leaving frees the places, also after an error inside
        with pytest.raises(KeyError):
            with gate.admitted():
                raise KeyError("x")
        with gate.admitted():
            assert asyncio.run(gate.run(lambda a, b: a + b, 1, 2)) == 3
    with pytest.raises(ValueError):
        service._Gate(workers=0, queue=1)


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
    # The redirect of the static mount builds its Location from the Host header.
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
    assert "/static/demo.js" in page.text and "<script>" not in page.text  # no inline script
    assert client.get("/static/demo.js").status_code == 200
    assert client.get("/static/demo.css").status_code == 200
    for response in (page, client.get("/healthz"), client.get("/missing"), client.get("/static/demo.js")):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "default-src 'self'" in response.headers["content-security-policy"]
    # The interactive docs cannot work under this policy (scripts from a CDN), so they are not served.
    for path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
        assert client.get(path).status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/inspect", "/calibrate", "/categories", "/healthz"} <= set(paths)


# 정적 파일 경로로 패키지 밖 파일(service.py, pyproject.toml)을 읽을 수 없는지 확인한다. '..', 백슬래시, URL
# 인코딩된 점, 절대 경로 등 여러 탈출 시도를 _asgi로 정리 없이 보내 모두 404인지, 응답에 소스가 섞이지 않는지
# 본다.
def test_static_files_cannot_be_escaped(client):
    # Sent to the app as written: an HTTP client would collapse the dot segments before sending.
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
    (tmp_path / "notes").mkdir()  # no meta.json: not a category

    found = service.load_inspectors(tmp_path, threads=3)
    assert sorted(found) == ["candle", "pcb1", "pcb2"]
    assert found["candle"].session is found["pcb1"].session is not found["pcb2"].session
    assert [(p.parent.name, t, prov) for p, t, prov in created] == [
        (tmp_path.name, 3, ["CPUExecutionProvider"]),
        ("pcb2", 3, ["CPUExecutionProvider"]),
    ]
    # Each inspector is what `Inspector.load` builds for its folder: every attribute, not a chosen few, so
    # that a change to the loader cannot leave this one behind unnoticed.
    for name, inspector in found.items():
        alone = Inspector.load(tmp_path / name, threads=3)
        mine, theirs = vars(inspector), vars(alone)
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
    # Anything wrong with an artifact is a ValueError that names the folder, as with `Inspector.load`.
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


# _DEMO_HARNESS: node에서 demo.js를 vm 샌드박스로 돌리는 JavaScript 하네스. 폼 제출에 필요한 만큼의 가짜
# DOM·fetch·Image를 두고, 서버 응답 목록을 차례로 돌려주며 매 제출 뒤 화면 상태(결과 숨김, 판정, 점수, 임계값,
# 상태 문구, 버튼, 캔버스에 그린 것)를 기록한다. URL.createObjectURL을 부르면 실패하게 해 페이지가 로컬 파일을
# 직접 디코딩하지 않는지도 본다.
# A browser stand-in for demo.js: just enough DOM to submit the form and look at what the page shows.
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
    # Dinomaly scores and thresholds are around 0.01: three decimals would show both as 0.018.
    close = {"is_defect": True, "score": 0.0181234, "threshold": 0.017962466925382614, "latency_ms": 150.0}
    replies = [
        {"status": 200, "body": {**defect, "heatmap_png": "HEAT", "input_png": "SEEN"}},
        {"status": 400, "body": {"detail": "the upload is not a readable image"}},
        {"status": 200, "body": {**defect, "heatmap_png": "HEAT", "input_png": "SEEN"}},
        "network",
        {"status": 502},  # a proxy error page instead of JSON
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
    # The picture comes from the server (what the model saw), so formats and EXIF rotation cannot differ.
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
    # The verdict of this upload is shown even when its picture cannot be drawn, with a note, and the
    # canvases do not keep the previous picture.
    assert (undrawable["hidden"], undrawable["verdict"], undrawable["score"]) == (False, "양품", "0.0018")
    assert undrawable["status"] != "" and "clear" in undrawable["drawn"]
    assert not any(item.startswith("data:") for item in undrawable["drawn"])
    assert all(shown["disabled"] is False for shown in result["shown"])
    # Four significant digits keep a score and a threshold that differ in the fourth decimal apart.
    assert (dinomaly["verdict"], dinomaly["score"], dinomaly["threshold"]) == ("불량", "0.01812", "0.01796")

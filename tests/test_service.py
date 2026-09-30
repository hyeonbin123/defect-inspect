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

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from defect_inspect import service  # noqa: E402

HOST = "http://127.0.0.1"
TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}
SECURITY = ("x-content-type-options", "x-frame-options", "content-security-policy", "referrer-policy")


def _encode(array: np.ndarray | Image.Image, fmt: str = "PNG", **options) -> bytes:
    image = array if isinstance(array, Image.Image) else Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def _png(array: np.ndarray) -> bytes:
    return _encode(array)


def _normal(seed=1, size=64) -> np.ndarray:
    return np.random.default_rng(seed).integers(110, 146, (size, size, 3), dtype=np.uint8)


def _defect(seed=1) -> np.ndarray:
    image = _normal(seed)
    image[20:36, 20:36] = 255
    return image


def _bright(count: int) -> list[np.ndarray]:
    """Normal images of a brighter line: all flagged by the threshold the offline inspector starts with."""
    return [np.clip(_normal(seed=s).astype(np.int16) + 60, 0, 255).astype(np.uint8) for s in range(count)]


def _files(images) -> list:
    return [("images", (f"{i}.png", _png(im), "image/png")) for i, im in enumerate(images)]


def _client(**settings) -> TestClient:
    settings.setdefault("admin_token", TOKEN)
    return TestClient(service.create_offline_app(**settings), base_url=HOST)


@pytest.fixture
def client():
    return _client()


def _inspect(client, data: bytes, category="demo", headers=None, **params):
    return client.post(
        "/inspect",
        files={"image": ("x.png", data, "image/png")},
        data={"category": category},
        params=params,
        headers=headers,
    )


def _calibrate(client, images, headers=ADMIN, **fields):
    return client.post(
        "/calibrate", files=_files(images), data={"category": "demo", **fields}, headers=headers
    )


def _state(client) -> dict:
    return client.get("/categories").json()[0]


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


def _multipart_head(field: str = "image") -> tuple[dict, bytes]:
    """Header and the opening of a multipart body whose file part never has to end."""
    head = (
        b'--B\r\nContent-Disposition: form-data; name="category"\r\n\r\ndemo\r\n'
        b'--B\r\nContent-Disposition: form-data; name="' + field.encode() + b'"; filename="x.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
    )
    return {"Content-Type": "multipart/form-data; boundary=B"}, head


def test_health_and_categories(client):
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok", "categories": ["demo"]}
    items = client.get("/categories").json()
    assert [i["category"] for i in items] == ["demo"]
    assert items[0]["img_size"] == 64 and items[0]["threshold"] > 0
    assert items[0]["calibration"] == {"strategy": "holdout", "n": 24, "guaranteed": True}
    # Health checks that only ask for the headers work too.
    assert client.head("/healthz").status_code == 200 and client.head("/").status_code == 200


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


def test_damaged_and_unsupported_files_are_a_400_not_a_500(client, monkeypatch):
    rgba = np.dstack([_normal(), np.full((64, 64), 255, np.uint8)])
    damaged = bytearray(_png(rgba))
    damaged[36] ^= 0xFF  # the length of the first data chunk: Pillow raises SyntaxError here, not OSError
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


def test_16_bit_images_are_refused_instead_of_scored_as_white(client):
    grey = np.random.default_rng(3).integers(110, 146, (64, 64), dtype=np.uint8)
    assert _inspect(client, _png(grey)).json()["is_defect"] is not None
    deep = grey.astype(np.uint16) * 257
    for data in (_encode(deep, "PNG"), _encode(deep, "TIFF"), _encode(grey.astype(np.float32), "TIFF")):
        response = _inspect(client, data)
        assert response.status_code == 400 and "8 bits" in response.json()["detail"]
    assert _calibrate(client, [deep] * 19).status_code == 400
    assert _state(client)["calibration"]["n"] == 24


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


def test_create_app_needs_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_ARTIFACTS", raising=False)
    with pytest.raises(RuntimeError, match="DEFECT_INSPECT_ARTIFACTS"):
        service.create_app()
    with pytest.raises(ValueError, match="no inspector artifacts"):
        service.create_app(tmp_path)


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


def test_heatmap_png_scale():
    heat = np.array([[0.0, 0.75], [1.5, 3.0]], dtype=np.float32)
    grey = np.asarray(Image.open(io.BytesIO(base64.b64decode(service.heatmap_png(heat, 1.0)))))
    assert grey.tolist() == [[0, 128], [255, 255]]


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
      status: el("status").textContent, disabled: el("submit").disabled, drawn: drawn.slice(),
    });
  }
  console.log(JSON.stringify({ shown, requests }));
})();
"""


def test_demo_page_never_shows_the_previous_verdict_next_to_a_new_upload(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "harness.js"
    harness.write_text(_DEMO_HARNESS, encoding="utf-8", newline="\n")
    defect = {"is_defect": True, "score": 3.3421, "threshold": 0.625, "latency_ms": 12.5}
    good = {"is_defect": False, "score": 0.0018, "threshold": 0.625, "latency_ms": 9.0}
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
    first, rejected, again, offline, proxy, undrawable, invalid = result["shown"]
    for shown in (first, again):
        assert (shown["hidden"], shown["verdict"], shown["score"], shown["status"]) == (
            False,
            "불량",
            "3.342",
            "",
        )
        assert "data:image/png;base64," in shown["drawn"] and "buffer" in shown["drawn"]
    assert rejected["hidden"] is True and rejected["status"] == "the upload is not a readable image"
    for shown in (offline, proxy, invalid):
        assert shown["hidden"] is True and shown["status"] != ""
    # The verdict of this upload is shown even when its picture cannot be drawn, with a note, and the
    # canvases do not keep the previous picture.
    assert (undrawable["hidden"], undrawable["verdict"], undrawable["score"]) == (False, "양품", "0.002")
    assert undrawable["status"] != "" and "clear" in undrawable["drawn"]
    assert not any(item.startswith("data:") for item in undrawable["drawn"])
    assert all(shown["disabled"] is False for shown in result["shown"])

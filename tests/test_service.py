import base64
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from defect_inspect import service  # noqa: E402


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _normal(seed=1, size=64) -> np.ndarray:
    return np.random.default_rng(seed).integers(110, 146, (size, size, 3), dtype=np.uint8)


def _defect(seed=1) -> np.ndarray:
    image = _normal(seed)
    image[20:36, 20:36] = 255
    return image


@pytest.fixture
def client():
    return TestClient(service.create_offline_app())


def _inspect(client, data: bytes, category="demo", **params):
    return client.post(
        "/inspect", files={"image": ("x.png", data, "image/png")}, data={"category": category}, params=params
    )


def test_health_and_categories(client):
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok", "categories": ["demo"]}
    items = client.get("/categories").json()
    assert [i["category"] for i in items] == ["demo"]
    assert items[0]["img_size"] == 64 and items[0]["threshold"] > 0
    assert items[0]["calibration"] == {"strategy": "holdout", "n": 24, "guaranteed": True}


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


def test_any_image_size_and_mode_is_accepted(client):
    big = np.random.default_rng(0).integers(110, 146, (300, 480), dtype=np.uint8)  # greyscale, not square
    assert _inspect(client, _png(big)).status_code == 200
    buffer = io.BytesIO()
    Image.fromarray(_normal()).save(buffer, format="JPEG")
    assert _inspect(client, buffer.getvalue()).status_code == 200


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
    monkeypatch.setattr(service, "MAX_IMAGE_PIXELS", 40_000_000)
    monkeypatch.setattr(service, "MAX_REQUEST_BYTES", 10)
    assert _inspect(client, _png(_normal())).status_code == 413


def test_calibrate_replaces_the_threshold(client):
    before = client.get("/categories").json()[0]["threshold"]
    # A brighter line: every image of the new condition is flagged with the old threshold.
    bright = [np.clip(_normal(seed=s).astype(np.int16) + 60, 0, 255).astype(np.uint8) for s in range(30)]
    assert _inspect(client, _png(bright[0])).json()["is_defect"] is True

    files = [("images", (f"{i}.png", _png(im), "image/png")) for i, im in enumerate(bright[:25])]
    response = client.post("/calibrate", files=files, data={"category": "demo", "alpha": "0.05"})
    assert response.status_code == 200
    body = response.json()
    assert body["previous_threshold"] == pytest.approx(before) and body["threshold"] > before
    assert body["calibration"] == {"strategy": "holdout", "n": 25, "guaranteed": True}
    assert client.get("/categories").json()[0]["threshold"] == pytest.approx(body["threshold"])
    # Images of the new condition that were not used for calibration now mostly pass.
    flagged = [_inspect(client, _png(im)).json()["is_defect"] for im in bright[25:]]
    assert sum(flagged) <= 1

    few = client.post("/calibrate", files=files[:3], data={"category": "demo"}).json()
    assert few["calibration"]["guaranteed"] is False and few["alpha"] == 0.05


def test_calibrate_validation(client, monkeypatch):
    files = [("images", ("0.png", _png(_normal()), "image/png"))]
    assert client.post("/calibrate", files=files, data={"category": "nope"}).status_code == 404
    for alpha in ("0", "-0.1", "0.6", "abc"):
        assert (
            client.post("/calibrate", files=files, data={"category": "demo", "alpha": alpha}).status_code
            == 422
        )
    assert client.post("/calibrate", data={"category": "demo"}).status_code == 422
    bad = [("images", ("0.png", b"junk", "image/png"))]
    assert client.post("/calibrate", files=bad, data={"category": "demo"}).status_code == 400
    monkeypatch.setattr(service, "MAX_CALIBRATION_FILES", 1)
    assert client.post("/calibrate", files=files * 2, data={"category": "demo"}).status_code == 413
    # A rejected call leaves the threshold as it was.
    assert client.get("/categories").json()[0]["calibration"]["n"] == 24


def test_demo_page_static_files_and_security_headers(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/static/demo.js" in page.text and "<script>" not in page.text  # no inline script
    assert client.get("/static/demo.js").status_code == 200
    assert client.get("/static/demo.css").status_code == 200
    assert client.get("/static/../service.py").status_code == 404
    for response in (page, client.get("/healthz"), client.get("/missing")):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "default-src 'self'" in response.headers["content-security-policy"]


def test_create_app_needs_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFECT_INSPECT_ARTIFACTS", raising=False)
    with pytest.raises(RuntimeError, match="DEFECT_INSPECT_ARTIFACTS"):
        service.create_app()
    with pytest.raises(ValueError, match="no inspector artifacts"):
        service.create_app(tmp_path)


def test_heatmap_png_scale():
    heat = np.array([[0.0, 0.75], [1.5, 3.0]], dtype=np.float32)
    grey = np.asarray(Image.open(io.BytesIO(base64.b64decode(service.heatmap_png(heat, 1.0)))))
    assert grey.tolist() == [[0, 128], [255, 255]]

"""Inspection service: upload a product image, get the anomaly score, the verdict and a heatmap.

Runs on the CPU with onnxruntime only (no torch). Inspectors are loaded from an artifact set built by
`defect_inspect.export` (one sub-folder per category). `create_offline_app` serves a tiny synthetic
inspector without any model file, for tests and security scans.
"""

import base64
import io
import os
import threading
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .inspector import ARTIFACT_VERSION, Inspector

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Whole request body (a calibration call carries many images).
MAX_REQUEST_BYTES = 256 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_CALIBRATION_FILES = 200
STATIC_DIR = Path(__file__).parent / "static"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "Cache-Control": "no-store",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def load_inspectors(
    artifacts: Path, *, precision: str = "fp32", threads: int | None = None
) -> dict[str, Inspector]:
    """Every category folder of an artifact set that holds a `meta.json`."""
    artifacts = Path(artifacts)
    found = {}
    for meta in sorted(artifacts.glob("*/meta.json")):
        found[meta.parent.name] = Inspector.load(meta.parent, precision=precision, threads=threads)
    if not found:
        raise ValueError(f"no inspector artifacts under {artifacts}")
    return found


def _decode(data: bytes) -> Image.Image:
    """Bytes of an upload -> RGB image, or a 400 for anything that is not a reasonable image."""
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail="image has too many pixels")
        return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="the upload is not a readable image") from None


async def _read_limited(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload larger than {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    return data


def heatmap_png(heatmap: np.ndarray, threshold: float) -> str:
    """Base64 PNG (8-bit grey, 256x256): 0 at score 0, white at 1.5 x the threshold and above."""
    scale = 1.5 * threshold if threshold > 0 else float(heatmap.max()) or 1.0
    grey = np.clip(heatmap / scale, 0.0, 1.0)
    buffer = io.BytesIO()
    Image.fromarray((grey * 255.0 + 0.5).astype(np.uint8), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_app(
    artifacts: Path | str | None = None,
    *,
    inspectors: dict[str, Inspector] | None = None,
    precision: str | None = None,
    threads: int | None = None,
) -> FastAPI:
    """The service. Without arguments the artifact set comes from `DEFECT_INSPECT_ARTIFACTS`."""
    if inspectors is None:
        artifacts = artifacts or os.environ.get("DEFECT_INSPECT_ARTIFACTS")
        if not artifacts:
            raise RuntimeError("set DEFECT_INSPECT_ARTIFACTS to an artifact set directory")
        precision = precision or os.environ.get("DEFECT_INSPECT_PRECISION", "fp32")
        env_threads = os.environ.get("DEFECT_INSPECT_THREADS")
        threads = threads if threads is not None else (int(env_threads) if env_threads else None)
        inspectors = load_inspectors(Path(artifacts), precision=precision, threads=threads)

    app = FastAPI(title="defect-inspect", version=str(ARTIFACT_VERSION), docs_url="/docs", redoc_url=None)
    app.state.inspectors = inspectors
    # Calibration replaces a threshold: one lock per category keeps a request from seeing half of it.
    locks = {name: threading.Lock() for name in inspectors}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            response = JSONResponse({"detail": "request body too large"}, status_code=413)
        else:
            response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    def get(category: str) -> Inspector:
        inspector = inspectors.get(category)
        if inspector is None:
            raise HTTPException(status_code=404, detail="unknown category")
        return inspector

    def describe(name: str, inspector: Inspector) -> dict:
        meta = inspector.meta
        return {
            "category": name,
            "threshold": inspector.threshold,
            "img_size": inspector.size,
            "bank_rows": inspector.bank_rows,
            "backbone": meta.get("backbone"),
            "precision": inspector.precision,
            "alpha": meta.get("alpha"),
            "calibration": meta.get("calibration"),
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "categories": sorted(inspectors)}

    @app.get("/categories")
    def categories() -> list[dict]:
        return [describe(name, inspectors[name]) for name in sorted(inspectors)]

    @app.post("/inspect")
    async def inspect(
        image: Annotated[UploadFile, File()],
        category: Annotated[str, Form(min_length=1, max_length=64)],
        heatmap: Annotated[bool, Query()] = True,
    ) -> JSONResponse:
        inspector = get(category)
        data = await _read_limited(image)

        def work():
            picture = _decode(data)
            with locks[category]:
                return inspector.inspect(picture, heatmap=heatmap)

        started = time.perf_counter()
        result = await run_in_threadpool(work)
        body = {
            "category": category,
            "score": result.score,
            "threshold": result.threshold,
            "is_defect": result.is_defect,
            "latency_ms": round((time.perf_counter() - started) * 1e3, 1),
        }
        if result.heatmap is not None:
            body["heatmap_png"] = heatmap_png(result.heatmap, result.threshold)
        return JSONResponse(body)

    @app.post("/calibrate")
    async def calibrate(
        images: Annotated[list[UploadFile], File()],
        category: Annotated[str, Form(min_length=1, max_length=64)],
        alpha: Annotated[float, Form(gt=0.0, le=0.5)] = 0.05,
    ) -> dict:
        """Set the category's threshold from normal images of the current condition (kept in memory)."""
        inspector = get(category)
        if len(images) > MAX_CALIBRATION_FILES:
            raise HTTPException(status_code=413, detail=f"at most {MAX_CALIBRATION_FILES} images per call")
        blobs = [await _read_limited(upload) for upload in images]

        def work():
            pictures = [_decode(blob) for blob in blobs]
            with locks[category]:
                return inspector.threshold, inspector.calibrate(pictures, alpha=alpha)

        previous, threshold = await run_in_threadpool(work)
        return {
            "category": category,
            "threshold": threshold,
            "previous_threshold": previous,
            "alpha": alpha,
            "calibration": inspector.meta["calibration"],
        }

    @app.get("/", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


class _GridMeanSession:
    """Offline stand-in for an ONNX model: the mean colour of each cell of a 8x8 grid."""

    grid = 8

    def run(self, names, feeds):
        x = feeds["image"]
        b, c, h, _ = x.shape
        cell = h // self.grid
        out = x.reshape(b, c, self.grid, cell, self.grid, cell).mean(axis=(3, 5))
        return [np.ascontiguousarray(out.transpose(0, 2, 3, 1)).astype(np.float32)]


def offline_inspector(size: int = 64, seed: int = 0) -> Inspector:
    """A synthetic inspector: normal images are grey noise, anything else scores high."""
    from .inspector import preprocess

    rng = np.random.default_rng(seed)
    session = _GridMeanSession()
    normals = rng.integers(110, 146, (48, size, size, 3), dtype=np.uint8)
    feats = [
        session.run(["features"], {"image": preprocess(im, size)})[0].reshape(-1, 3) for im in normals[:24]
    ]
    meta = {
        "version": ARTIFACT_VERSION,
        "name": "offline",
        "category": "demo",
        "backbone": "gridmean",
        "img_size": size,
        "grid": [session.grid, session.grid],
        "dim": 3,
        "reweight_k": 9,
        "sigma": 1.0,
        "threshold": 0.0,
    }
    inspector = Inspector(session, np.concatenate(feats).astype(np.float16), meta)
    inspector.precision = "offline"
    inspector.calibrate(list(normals[24:]), alpha=0.05)
    return inspector


def create_offline_app() -> FastAPI:
    """The same service with one synthetic category `demo` and no model file."""
    return create_app(inspectors={"demo": offline_inspector()})

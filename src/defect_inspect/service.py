"""Inspection service: upload a product image, get the anomaly score, the verdict and a heatmap.

Runs on the CPU with onnxruntime only (no torch). Inspectors are loaded from an artifact set (one sub-folder
per category): a reconstruction model built by `defect_inspect.dinomaly_serving export` (Dinomaly, the
served model since stage 6: one ONNX model scores every category, each folder holds its threshold) or a
PatchCore set built by `defect_inspect.export` (a memory bank per category). `create_offline_app` serves a
tiny synthetic inspector without any model file, for tests and security scans.

Uploads: PNG, JPEG, BMP, TIFF or WebP with 8 bits per channel and no transparency. The stored pixels are
scored as they are (no EXIF rotation), like the resize cache the models and memory banks were built from.

Settings (environment variables, or the keyword arguments of `create_app`):
- `DEFECT_INSPECT_ALLOWED_HOSTS`: comma-separated Host names the service answers to (`*.example.com` and
  `*` are wildcards). Default: localhost, 127.0.0.1 and [::1].
- `DEFECT_INSPECT_ADMIN_TOKEN`: enables `/calibrate`; callers send it as the `X-Admin-Token` header.
  Without it the thresholds of the artifact set cannot be changed over HTTP.
- `DEFECT_INSPECT_WORKERS` / `DEFECT_INSPECT_QUEUE`: images decoded and scored at once (default 2) and
  requests allowed to wait for a worker (default 16); beyond that the answer is a 503 with `Retry-After`.
"""

import asyncio
import base64
import hmac
import io
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.datastructures import Headers, MutableHeaders

from .calibrate import conformal_rank, conformal_threshold
from .inspector import (
    ARTIFACT_VERSION,
    PATCHCORE,
    PRECISIONS,
    RECONSTRUCTION,
    RECONSTRUCTION_PRECISIONS,
    Inspector,
    ReconstructionInspector,
    artifact_kind,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Whole request body of a calibration call (it carries many images).
MAX_REQUEST_BYTES = 256 * 1024 * 1024
# Room for the multipart framing and the text fields around the single image of an inspection call.
MULTIPART_SLACK_BYTES = 64 * 1024
# 24 MP sensors fit. A decoded picture takes 4 bytes per pixel until it is resized to the model input.
MAX_IMAGE_PIXELS = 25_000_000
MAX_CALIBRATION_FILES = 200
IMAGE_FORMATS = ("PNG", "JPEG", "BMP", "TIFF", "WEBP")
# Pillow clips these to 255 when converting to RGB, which would score every such image as all white.
DEEP_MODES = frozenset({"I", "F", "I;16", "I;16L", "I;16B", "I;16N"})
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]")
DEFAULT_WORKERS = 2
DEFAULT_QUEUE = 16
ADMIN_HEADER = "X-Admin-Token"
ADMIN_PATH = "/calibrate"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
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
_MISSING = object()
# Model file of each precision, by artifact kind (the `kind` of a category folder's meta.json).
MODEL_FILES = {PATCHCORE: PRECISIONS, RECONSTRUCTION: RECONSTRUCTION_PRECISIONS}
AnyInspector = Inspector | ReconstructionInspector


def load_inspectors(
    artifacts: Path, *, precision: str = "fp32", threads: int | None = None
) -> dict[str, AnyInspector]:
    """Every category folder of an artifact set that holds a `meta.json`.

    The folder's `kind` picks the class: `ReconstructionInspector` (no bank) or the PatchCore `Inspector`.
    Each inspector is what that class's `load` builds for its folder (tests compare the two), except that
    categories using the same model file (an artifact set keeps one next to the category folders) share one
    ONNX Runtime session: `session.run` is thread-safe, and a session per category would multiply the
    memory and the start-up time by the number of categories. Anything wrong with an artifact is a
    `ValueError` that names its folder.
    """
    import onnxruntime as ort

    artifacts = Path(artifacts)
    known = sorted({name for files in MODEL_FILES.values() for name in files})
    if precision not in known:
        raise ValueError(f"precision must be one of {known}, got {precision!r}")
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = int(threads)
    sessions: dict[Path, Any] = {}
    found: dict[str, AnyInspector] = {}
    for meta_path in sorted(artifacts.glob("*/meta.json")):
        folder = meta_path.parent
        kind = artifact_kind(folder)
        files = MODEL_FILES.get(kind)
        if files is None:
            raise ValueError(f"{folder}: unknown artifact kind {kind!r}")
        if precision not in files:
            raise ValueError(
                f"{folder}: precision must be one of {sorted(files)} for a {kind} artifact, got {precision!r}"
            )
        model = folder / files[precision]
        if not model.exists():
            model = artifacts / files[precision]
        needed = (model,) if kind == RECONSTRUCTION else (folder / "bank.npy", model)
        for path in needed:
            if not path.exists():
                raise ValueError(f"not an inspector artifact: {path} is missing")
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            bank = None if kind == RECONSTRUCTION else np.load(folder / "bank.npy", allow_pickle=False)
            key = model.resolve()
            if key not in sessions:
                sessions[key] = ort.InferenceSession(
                    str(model), sess_options=options, providers=["CPUExecutionProvider"]
                )
            if bank is None:
                inspector = ReconstructionInspector(sessions[key], meta)
            else:
                inspector = Inspector(sessions[key], bank, meta)
        except Exception as err:  # unreadable files, a model onnxruntime rejects, a meta/bank/model mismatch
            raise ValueError(f"{folder} ({model.name}): {err}") from err
        inspector.precision = precision
        inspector.threads = None if threads is None else int(threads)
        found[folder.name] = inspector
    if not found:
        raise ValueError(f"no inspector artifacts under {artifacts}")
    return found


def _decode(data: bytes, size: int | None = None) -> Image.Image:
    """Bytes of an upload -> 8-bit RGB picture, or a 400 for anything that is not a reasonable image.

    With `size` the picture is resized to the model input right away, by the same call as
    `inspector.preprocess`, so the scores do not change and the full-size decode is not kept.
    """
    try:
        image = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail="image has too many pixels")
        if image.mode in DEEP_MODES:
            raise HTTPException(
                status_code=400, detail="16-bit and float images are not supported: send 8 bits per channel"
            )
        if image.has_transparency_data:
            # Pixels under a transparent area are invisible in a viewer but would be scored.
            image = image.convert("RGBA")
            if image.getchannel("A").getextrema()[0] < 255:
                raise HTTPException(status_code=400, detail="images with transparency are not supported")
        picture = image.convert("RGB")
        if size is not None:
            picture = picture.resize((size, size), Image.Resampling.BICUBIC)
        return picture
    except (HTTPException, MemoryError):
        raise
    except Exception:
        # Pillow's decoders raise more than OSError on damaged files (SyntaxError, RuntimeError, ...).
        raise HTTPException(status_code=400, detail="the upload is not a readable image") from None


def _read_limited(upload: UploadFile) -> bytes:
    """The bytes of one uploaded file. Blocking: call it in a worker, next to the decode."""
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload larger than {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    return data


def _png_base64(image: Image.Image, **options) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def heatmap_png(heatmap: np.ndarray, threshold: float) -> str:
    """Base64 PNG (8-bit grey, 256x256): 0 at score 0, white at 1.5 x the threshold and above."""
    scale = 1.5 * threshold if threshold > 0 else float(heatmap.max()) or 1.0
    grey = np.clip(heatmap / scale, 0.0, 1.0)
    return _png_base64(Image.fromarray((grey * 255.0 + 0.5).astype(np.uint8)))


def _min_calibration_images(alpha: float) -> int:
    """Smallest n whose conformal rank fits into n scores, i.e. `ceil(1 / alpha) - 1`."""
    n = max(1, math.ceil(1 / Fraction(format(alpha, ".12g"))) - 1)
    while conformal_rank(n, alpha) > n:
        n += 1
    return n


def _host_allowed(value: str, allowed: Sequence[str]) -> bool:
    """Whether a Host header (the port is ignored) matches one of the allowed names."""
    host = value.strip().lower()
    host = host.partition("]")[0] + "]" if host.startswith("[") else host.partition(":")[0]
    for pattern in allowed:
        if pattern == "*" or host == pattern or (pattern.startswith("*.") and host.endswith(pattern[1:])):
            return True
    return False


def _cross_site(headers: Headers) -> bool:
    """A request that a page of another origin made the browser send (Fetch metadata, else Origin)."""
    site = headers.get("sec-fetch-site")
    if site is not None:
        return site.strip().lower() not in ("same-origin", "none")
    origin = headers.get("origin")
    if origin is None:
        return False  # not a browser
    return urlsplit(origin).netloc.lower() != headers.get("host", "").strip().lower()


def _admin_problem(headers: Headers, token: str | None) -> str | None:
    """Why a request may not change thresholds, or None when it carries the admin token."""
    if not token:
        return "calibration is disabled: start the service with DEFECT_INSPECT_ADMIN_TOKEN"
    given = headers.get(ADMIN_HEADER, "")
    if not hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8")):
        return f"a valid {ADMIN_HEADER} header is required"
    return None


def _body_limit(path: str) -> int:
    if path == ADMIN_PATH:
        return MAX_REQUEST_BYTES
    return MAX_UPLOAD_BYTES + MULTIPART_SLACK_BYTES


def _refusal(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


class _Guard:
    """Pure ASGI wrapper around the whole app.

    Before anything is read from the body: the Host check, the cross-site check and the admin token.
    While the body arrives: a byte limit that also holds for chunked requests. On the way out: the security
    headers on every response, including the JSON 500 this class writes for an unhandled error.
    """

    def __init__(self, app, *, allowed_hosts: Sequence[str], admin_token: str | None):
        self.app = app
        self.allowed_hosts = tuple(allowed_hosts)
        self.admin_token = admin_token

    def _refuse(self, scope, headers: Headers) -> JSONResponse | None:
        path = scope["path"]
        # The body of a refused request never reaches the app: the server reads it off the socket and
        # drops it, so the client still gets to see the answer.
        if not _host_allowed(headers.get("host", ""), self.allowed_hosts):
            return _refusal(400, "host not allowed")
        if scope["method"] not in SAFE_METHODS and _cross_site(headers):
            return _refusal(403, "cross-site requests are not accepted")
        if path == ADMIN_PATH or path.startswith(ADMIN_PATH + "/"):
            problem = _admin_problem(headers, self.admin_token)
            if problem:
                return _refusal(403, problem)
        length = headers.get("content-length", "")
        if length.isdigit() and int(length) > _body_limit(path):
            return _refusal(413, "request body too large")
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False
        received = 0
        limit = _body_limit(scope["path"])

        async def send_with_headers(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(raw=message.setdefault("headers", []))
                for key, value in SECURITY_HEADERS.items():
                    headers.setdefault(key, value)
            await send(message)

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # More than announced, or a chunked body with no end in sight: answer and hang up
                    # ("Connection: close" makes the server drop the connection instead of reading on).
                    # FastAPI passes an HTTPException raised while it parses the body through as it is.
                    raise HTTPException(
                        status_code=413, detail="request body too large", headers={"Connection": "close"}
                    )
            return message

        refusal = self._refuse(scope, Headers(scope=scope))
        if refusal is not None:
            await refusal(scope, receive, send_with_headers)
            return
        try:
            await self.app(scope, limited_receive, send_with_headers)
        except Exception:
            if started:
                raise
            logger.exception("unhandled error in %s %s", scope["method"], scope["path"])
            await _refusal(500, "internal server error")(scope, receive, send_with_headers)


class _Gate:
    """At most `workers` jobs compute at once and at most `queue` more requests wait; the rest get a 503.

    The jobs run on threads of their own. The bound keeps decoded images from piling up in memory, a
    waiting request does not hold a thread of the shared pool (static files need those), and the big
    image buffers stay in the memory arenas of these few threads instead of spreading over forty.
    """

    def __init__(self, workers: int, queue: int):
        if workers < 1 or queue < 0:
            raise ValueError("workers must be at least 1 and queue at least 0")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inspect")
        self._lock = threading.Lock()
        self._admitted = 0
        self.capacity = workers + queue

    @contextmanager
    def admitted(self) -> Iterator[None]:
        with self._lock:
            if self._admitted >= self.capacity:
                raise HTTPException(
                    status_code=503, detail="busy: try again shortly", headers={"Retry-After": "1"}
                )
            self._admitted += 1
        try:
            yield
        finally:
            with self._lock:
                self._admitted -= 1

    async def run(self, job: Callable, *args):
        return await asyncio.get_running_loop().run_in_executor(self._pool, job, *args)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def create_app(
    artifacts: Path | str | None = None,
    *,
    inspectors: dict[str, AnyInspector] | None = None,
    precision: str | None = None,
    threads: int | None = None,
    admin_token: str | None = None,
    allowed_hosts: Sequence[str] | None = None,
    workers: int | None = None,
    queue: int | None = None,
) -> FastAPI:
    """The service. Arguments left out come from the `DEFECT_INSPECT_*` environment variables."""
    if inspectors is None:
        artifacts = artifacts or os.environ.get("DEFECT_INSPECT_ARTIFACTS")
        if not artifacts:
            raise RuntimeError("set DEFECT_INSPECT_ARTIFACTS to an artifact set directory")
        precision = precision or os.environ.get("DEFECT_INSPECT_PRECISION", "fp32")
        env_threads = os.environ.get("DEFECT_INSPECT_THREADS")
        threads = threads if threads is not None else (int(env_threads) if env_threads else None)
        inspectors = load_inspectors(Path(artifacts), precision=precision, threads=threads)
    if admin_token is None:
        admin_token = os.environ.get("DEFECT_INSPECT_ADMIN_TOKEN") or None
    if allowed_hosts is None:
        listed = os.environ.get("DEFECT_INSPECT_ALLOWED_HOSTS", "")
        allowed_hosts = [h for h in listed.split(",") if h.strip()] or DEFAULT_ALLOWED_HOSTS
    allowed_hosts = tuple(h.strip().lower() for h in allowed_hosts)
    gate = _Gate(
        workers if workers is not None else _env_int("DEFECT_INSPECT_WORKERS", DEFAULT_WORKERS),
        queue if queue is not None else _env_int("DEFECT_INSPECT_QUEUE", DEFAULT_QUEUE),
    )

    # The interactive docs need scripts from a CDN, which the content security policy forbids.
    app = FastAPI(title="defect-inspect", version=str(ARTIFACT_VERSION), docs_url=None, redoc_url=None)
    app.add_middleware(_Guard, allowed_hosts=allowed_hosts, admin_token=admin_token)
    app.state.inspectors = inspectors
    # Scoring runs outside these locks. They only cover reading and replacing a category's threshold
    # state (threshold, alpha, calibration), so that nobody sees half of a calibration.
    locks = {name: threading.Lock() for name in inspectors}
    app.state.locks = locks
    # What the artifact set came with, for DELETE /calibrate.
    originals = {
        name: (i.threshold, i.meta.get("alpha", _MISSING), i.meta.get("calibration", _MISSING))
        for name, i in inspectors.items()
    }
    index_html = (STATIC_DIR / "index.html").read_bytes()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Where and what, without echoing the input or Python type names.
        detail = [
            {
                "loc": list(error.get("loc", ())),
                "type": error.get("type"),
                "msg": "Invalid value" if error.get("type") == "value_error" else error.get("msg"),
            }
            for error in exc.errors()
        ]
        return JSONResponse({"detail": detail}, status_code=422)

    def require_admin(request: Request) -> None:
        # The guard already refused such a request before its body was read; this is the same rule at
        # the route, in case the two ever disagree about a path.
        problem = _admin_problem(request.headers, admin_token)
        if problem:
            raise HTTPException(status_code=403, detail=problem)

    def get(category: str) -> AnyInspector:
        inspector = inspectors.get(category)
        if inspector is None:
            raise HTTPException(status_code=404, detail="unknown category")
        return inspector

    def threshold_state(inspector: AnyInspector) -> dict:
        """Call with the category's lock held."""
        calibration = inspector.meta.get("calibration")
        return {
            "threshold": inspector.threshold,
            "alpha": inspector.meta.get("alpha"),
            "calibration": dict(calibration) if isinstance(calibration, dict) else calibration,
        }

    def describe(name: str, inspector: AnyInspector) -> dict:
        with locks[name]:
            state = threshold_state(inspector)
        return {
            "category": name,
            "kind": inspector.kind,
            "model": inspector.meta.get("name"),
            "threshold": state["threshold"],
            "img_size": inspector.size,
            "bank_rows": inspector.bank_rows,
            "backbone": inspector.meta.get("backbone"),
            "precision": inspector.precision,
            "alpha": state["alpha"],
            "calibration": state["calibration"],
        }

    # Async handlers: a health check must not queue behind inspections for a thread.
    @app.head("/healthz", include_in_schema=False)
    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "categories": sorted(inspectors)}

    @app.get("/categories")
    async def categories() -> list[dict]:
        return [describe(name, inspectors[name]) for name in sorted(inspectors)]

    @app.post("/inspect")
    async def inspect(
        image: Annotated[UploadFile, File()],
        category: Annotated[str, Form(min_length=1, max_length=64)],
        heatmap: Annotated[bool, Query()] = True,
        preview: Annotated[bool, Query()] = False,
    ) -> JSONResponse:
        """Score one image. `preview=true` adds `input_png`: the picture exactly as the model saw it."""
        inspector = get(category)

        def work() -> dict:
            picture = _decode(_read_limited(image), inspector.size)
            result = inspector.inspect(picture, heatmap=heatmap)
            with locks[category]:
                threshold = inspector.threshold  # read once: the verdict and the map scale agree
            body = {
                "category": category,
                "score": result.score,
                "threshold": threshold,
                "is_defect": result.score > threshold,
            }
            if result.heatmap is not None:
                body["heatmap_png"] = heatmap_png(result.heatmap, threshold)
            if preview:
                body["input_png"] = _png_base64(picture, compress_level=1)
            return body

        with gate.admitted():
            started = time.perf_counter()
            body = await gate.run(work)
        body["latency_ms"] = round((time.perf_counter() - started) * 1e3, 1)
        return JSONResponse(body)

    @app.post(ADMIN_PATH, dependencies=[Depends(require_admin)])
    async def calibrate(
        images: Annotated[list[UploadFile], File()],
        category: Annotated[str, Form(min_length=1, max_length=64)],
        alpha: Annotated[float, Form(gt=0.0, le=0.5)] = 0.05,
        allow_unguaranteed: Annotated[bool, Form()] = False,
    ) -> dict:
        """Set the category's threshold from normal images of the current condition (kept in memory).

        Only the threshold changes: the model (and a PatchCore memory bank) stays as loaded. Needs the admin
        token. Too few images for the conformal rank at `alpha` are refused unless `allow_unguaranteed` is
        set (the threshold is then the largest score).
        """
        inspector = get(category)
        if len(images) > MAX_CALIBRATION_FILES:
            raise HTTPException(status_code=413, detail=f"at most {MAX_CALIBRATION_FILES} images per call")
        if not allow_unguaranteed and conformal_rank(len(images), alpha) > len(images):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{len(images)} images cannot guarantee alpha {alpha}: send at least "
                    f"{_min_calibration_images(alpha)} or set allow_unguaranteed"
                ),
            )

        def score(upload: UploadFile) -> float:
            picture = _decode(_read_limited(upload), inspector.size)
            return inspector.inspect(picture, heatmap=False).score

        # One image at a time: decoded pictures are not kept, and inspections get a worker in between.
        scores = []
        with gate.admitted():
            for upload in images:
                scores.append(await gate.run(score, upload))

        # The same rule and bookkeeping as `Inspector.calibrate`, applied to the scores found above.
        found = conformal_threshold(np.array(scores, dtype=np.float32), alpha)
        with locks[category]:
            previous = inspector.threshold
            inspector.set_threshold(found.value)
            inspector.meta["alpha"] = float(alpha)
            inspector.meta["calibration"] = {
                "strategy": "holdout",
                "n": len(scores),
                "guaranteed": bool(found.guaranteed),
            }
            return {"category": category, "previous_threshold": previous, **threshold_state(inspector)}

    @app.delete(ADMIN_PATH, dependencies=[Depends(require_admin)])
    async def reset_calibration(category: Annotated[str, Query(min_length=1, max_length=64)]) -> dict:
        """Back to the threshold, alpha and calibration record the artifact set was loaded with."""
        inspector = get(category)
        threshold, alpha, calibration = originals[category]
        with locks[category]:
            previous = inspector.threshold
            inspector.set_threshold(threshold)
            for key, value in (("alpha", alpha), ("calibration", calibration)):
                if value is _MISSING:
                    inspector.meta.pop(key, None)
                else:
                    inspector.meta[key] = dict(value) if isinstance(value, dict) else value
            return {"category": category, "previous_threshold": previous, **threshold_state(inspector)}

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def demo() -> Response:
        return Response(index_html, media_type="text/html")

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


def create_offline_app(**settings) -> FastAPI:
    """The same service with one synthetic category `demo` and no model file."""
    return create_app(inspectors={"demo": offline_inspector()}, **settings)

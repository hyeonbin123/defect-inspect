"""검사 서비스: 제품 사진을 올리면 이상 점수, 판정, 히트맵을 돌려준다.

torch 없이 CPU의 onnxruntime만으로 돈다. 검사기는 아티팩트 묶음(범주마다 하위 폴더 하나)에서 읽는다:
`defect_inspect.dinomaly_serving export`가 만든 재구성 모델(Dinomaly, 6단계부터의 서빙 모델: ONNX 모델 하나가
모든 범주를 채점하고 폴더마다 임계값을 둔다) 또는 `defect_inspect.export`가 만든 PatchCore 묶음(범주마다
메모리 뱅크). `create_offline_app`은 모델 파일 없이 아주 작은 합성 검사기를 서빙한다(테스트와 보안 스캔용).

업로드: 채널당 8비트이고 투명도가 없는 PNG, JPEG, BMP, TIFF, WebP. 저장된 화소를 그대로 채점한다(EXIF 회전
없음). 모델과 메모리 뱅크를 만든 리사이즈 캐시도 그렇게 만들었다.

설정(환경 변수 또는 `create_app`의 키워드 인자):
- `DEFECT_INSPECT_ALLOWED_HOSTS`: 서비스가 응답할 Host 이름을 쉼표로 나열한다(`*.example.com`과 `*`는
  와일드카드). 기본값: localhost, 127.0.0.1, [::1].
- `DEFECT_INSPECT_ADMIN_TOKEN`: `/calibrate`를 켠다. 호출하는 쪽은 이 값을 `X-Admin-Token` 헤더로 보낸다.
  없으면 HTTP로 아티팩트 묶음의 임계값을 바꿀 수 없다.
- `DEFECT_INSPECT_WORKERS` / `DEFECT_INSPECT_QUEUE`: 동시에 디코딩·채점하는 이미지 수(기본 2)와 작업자를
  기다릴 수 있는 요청 수(기본 16). 그보다 많으면 `Retry-After`를 붙인 503으로 답한다.
"""

# [파일 역할] FastAPI 검사 서비스. 사진 한 장과 범주를 받아 이상 점수·임계값·양품/불량 판정·히트맵을
# 돌려주고(POST /inspect), 관리자 토큰이 있으면 현재 촬영 조건의 정상 사진으로 그 범주의 임계값만 다시
# 잡거나(POST /calibrate) 되돌린다(DELETE /calibrate). GET /은 데모 페이지다.
# [설계 이유] torch 없이 onnxruntime CPU만으로 돈다(Docker 이미지 약 650MB, README). 6단계 이후 서빙 모델은
# Dinomaly ViT-S(dms-280-car)이고 ONNX 하나를 12개 범주가 같이 쓴다. 범주 폴더 meta.json의 kind로 복원
# 방식(ReconstructionInspector)과 4단계 PatchCore(Inspector)를 골라 둘 다 읽는다. 임계값이 "촬영 조건이
# 그대로일 때만" 목표를 지킨다는 실험 결과(3·7단계) 때문에, 조건이 바뀌면 모델은 그대로 두고 임계값만 다시
# 잡는 /calibrate를 둔다.
# [보안 구성] 본문을 읽기 전에 Host 허용 목록, 다른 출처의 쓰기 요청 거절, 관리자 토큰을 검사하고(_Guard),
# 본문 크기를 스트리밍 중에도 제한하며, 모든 응답에 보안 헤더(CSP 등)를 붙인다. 이미지 디코딩은 형식·픽셀
# 수·비트 깊이·투명도를 제한하고, 동시에 계산하는 작업 수를 묶어 둔다(_Gate). DAST(HawkScan, ZAP) 기록은
# docs/security.md에 있다.
# [연결] inspector.py(전처리·추론·임계값 상태), calibrate.py(conformal 임계값). 실행은 uvicorn
# defect_inspect.service:create_app --factory(Dockerfile, compose.yaml) 또는 모델 파일 없는
# create_offline_app(테스트와 보안 스캔용).

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

# fastapi: 경로·폼·파일 업로드 검증과 응답. PIL: 업로드 디코딩과 PNG 인코딩. starlette의
# Headers·MutableHeaders: ASGI 수준(_Guard)에서 요청 헤더를 읽고 응답 헤더를 붙이는 데 쓴다. onnxruntime은
# load_inspectors 안에서 늦게 가져온다.
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.datastructures import Headers, MutableHeaders

# calibrate: /calibrate가 연구 코드와 같은 conformal 규칙(l = ceil((n+1)(1-α))번째 점수)으로 임계값을 정하게
# 한다. inspector: 두 종류의 검사기와 아티팩트 형식 상수.
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

# 요청 한계값.
# - MAX_UPLOAD_BYTES: 파일 하나 20MB(README). MAX_REQUEST_BYTES: 여러 장을 올리는 /calibrate 본문 전체 256MB
# - MULTIPART_SLACK_BYTES: /inspect 본문 한계 = 파일 20MB + multipart 틀과 텍스트 필드 여유 64KB
# - MAX_IMAGE_PIXELS: 2,500만 화소(2,400만 화소 센서의 사진이 들어간다). 디코딩한 그림이 모델 입력 크기로
#   리사이즈되기 전까지 화소당 4바이트를 차지하므로 메모리 상한을 두는 것이다
# - MAX_CALIBRATION_FILES: /calibrate 한 번에 최대 200장
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 256 * 1024 * 1024
MULTIPART_SLACK_BYTES = 64 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_CALIBRATION_FILES = 200
# 디코딩을 허용하는 형식 다섯 가지(README의 받는 형식). Pillow가 지원하는 다른 형식의 디코더를 열지 않으므로
# 공격 면이 줄어든다.
# DEEP_MODES: 16비트·실수 모드. RGB로 바꾸면 255에서 잘려 이미지 전체가 흰색처럼 채점되므로 조용히 자르지 않고
# 거절한다(README).
IMAGE_FORMATS = ("PNG", "JPEG", "BMP", "TIFF", "WEBP")
DEEP_MODES = frozenset({"I", "F", "I;16", "I;16L", "I;16B", "I;16N"})
# 기본으로 응답하는 Host 이름(로컬만). 다른 이름으로 열려면 DEFECT_INSPECT_ALLOWED_HOSTS에 적고, 그 값이 기본
# 목록을 대신한다. Host 허용 목록은 일반적으로 DNS 리바인딩처럼 엉뚱한 이름으로 들어온 요청을 막는 장치다.
# DEFAULT_WORKERS, DEFAULT_QUEUE: 동시에 계산하는 작업 2개, 기다릴 수 있는 요청 16개(넘으면 503).
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]")
DEFAULT_WORKERS = 2
DEFAULT_QUEUE = 16
# 관리자 토큰 헤더와 보호 경로. 다른 출처의 페이지는 이 사용자 정의 헤더를 붙일 수 없다(CORS를 열지 않으므로).
# SAFE_METHODS는 상태를 바꾸지 않는 메서드로, 다른 출처 검사에서 뺀다.
ADMIN_HEADER = "X-Admin-Token"
ADMIN_PATH = "/calibrate"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
STATIC_DIR = Path(__file__).parent / "static"
# 모든 응답에 붙이는 보안 헤더. MIME 추측 금지, 프레임 삽입 금지(클릭재킹), 리퍼러 안 보냄, 다른 출처의 리소스
# 읽기 금지, 자기 출처의 스크립트·스타일만 허용하고 인라인 스크립트를 막는 CSP(데모 페이지가 static/의 파일만
# 쓰는 이유), 캐시 금지, 카메라·마이크·위치 권한 끔. img-src의 data:는 데모 페이지(demo.js)가 응답의 base64
# PNG(모델 입력 미리보기, 히트맵)를 그리는 데 쓴다.
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
# _MISSING: meta에 키가 아예 없었다는 표시(None과 구별해 DELETE /calibrate가 원래 상태로 정확히 되돌리게).
# MODEL_FILES: 아티팩트 종류별 정밀도 → 모델 파일 이름. AnyInspector: 두 검사기 클래스 중 하나.
_MISSING = object()
# MODEL_FILES의 키는 아티팩트 종류(범주 폴더 meta.json의 `kind`)다.
MODEL_FILES = {PATCHCORE: PRECISIONS, RECONSTRUCTION: RECONSTRUCTION_PRECISIONS}
AnyInspector = Inspector | ReconstructionInspector


# 아티팩트 폴더에서 범주 검사기를 모두 읽는다. 범주 폴더의 meta.json kind로 클래스를 고르고, 같은 모델 파일을
# 쓰는 범주는 ONNX Runtime 세션 하나를 같이 쓴다(dms-280-car는 12개 범주가 세션 하나). 세션 run은 스레드
# 안전하고, 범주마다 세션을 만들면 메모리와 시작 시간이 범주 수만큼 늘기 때문이다. 문제가 있으면
# 폴더 이름을 담은 ValueError로 서비스 시작을 멈춘다.
# 호출 위치: create_app, tests/test_service.py.
def load_inspectors(
    artifacts: Path, *, precision: str = "fp32", threads: int | None = None
) -> dict[str, AnyInspector]:
    """아티팩트 묶음에서 `meta.json`이 있는 모든 범주 폴더의 검사기.

    폴더의 `kind`가 클래스를 고른다: `ReconstructionInspector`(뱅크 없음) 또는 PatchCore `Inspector`. 각
    검사기는 그 클래스의 `load`가 그 폴더로 만드는 것과 같다(테스트가 둘을 비교한다). 다만 같은 모델 파일을
    쓰는 범주들(아티팩트 묶음은 범주 폴더 옆에 모델 파일 하나를 둔다)은 ONNX Runtime 세션 하나를 같이 쓴다:
    `session.run`은 스레드 안전하고, 범주마다 세션을 만들면 메모리와 시작 시간이 범주 수만큼 늘기 때문이다.
    아티팩트에 문제가 있으면 그 폴더 이름을 담은 `ValueError`를 낸다.
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
        # 모델 파일은 범주 폴더에 없으면 아티팩트 최상위에서 찾는다(공유 모델). PatchCore는 범주별 메모리
        # 뱅크(bank.npy)도 있어야 한다.
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
            # allow_pickle=False로 뱅크 파일을 읽어, 조작된 .npy가 피클로 임의 코드를 실행하지 못하게 한다.
            bank = None if kind == RECONSTRUCTION else np.load(folder / "bank.npy", allow_pickle=False)
            # 실제 경로로 키를 만들어 같은 파일이면 세션을 다시 만들지 않는다. CPU
            # 전용(CPUExecutionProvider)으로 연다.
            key = model.resolve()
            if key not in sessions:
                sessions[key] = ort.InferenceSession(
                    str(model), sess_options=options, providers=["CPUExecutionProvider"]
                )
            if bank is None:
                inspector = ReconstructionInspector(sessions[key], meta)
            else:
                inspector = Inspector(sessions[key], bank, meta)
        except Exception as err:  # 읽을 수 없는 파일, onnxruntime이 거부한 모델, meta·뱅크·모델 불일치
            raise ValueError(f"{folder} ({model.name}): {err}") from err
        inspector.precision = precision
        inspector.threads = None if threads is None else int(threads)
        found[folder.name] = inspector
    if not found:
        raise ValueError(f"no inspector artifacts under {artifacts}")
    return found


# 업로드 바이트를 8비트 RGB 그림으로 바꾸고, 이미지가 아니거나 받지 않는 형식이면 400을 낸다. size를 주면 바로
# 모델 입력 크기로 줄여, 큰 원본을 메모리에 오래 두지 않는다. 리사이즈는 inspector.preprocess와 같은
# 호출(BICUBIC)이라 점수가 달라지지 않는다.
# EXIF 회전은 적용하지 않는다. 모델과 뱅크를 만든 리사이즈 캐시도 저장된 화소 그대로였기 때문이다(모듈
# docstring).
def _decode(data: bytes, size: int | None = None) -> Image.Image:
    """업로드 바이트 -> 8비트 RGB 그림. 적절한 이미지가 아니면 400을 낸다.

    `size`를 주면 `inspector.preprocess`와 같은 호출로 바로 모델 입력 크기로 줄인다. 그래서 점수는 그대로이고
    원본 크기로 디코딩한 그림을 계속 들고 있지 않는다.
    """
    try:
        image = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        # Image.open은 헤더만 읽으므로, 화소 수 검사를 실제 디코딩(convert) 전에 해서 "압축 폭탄"처럼 작은
        # 파일이 거대한 그림으로 풀리는 것을 막는다.
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail="image has too many pixels")
        if image.mode in DEEP_MODES:
            raise HTTPException(
                status_code=400, detail="16-bit and float images are not supported: send 8 bits per channel"
            )
        # 투명도가 있는 이미지는 알파가 모두 255(완전 불투명)일 때만 받는다. 투명한 영역 아래 화소는 화면에 안
        # 보이지만 채점되므로, 사람이 본 것과 모델이 본 것이 달라지는 것을 막는다.
        if image.has_transparency_data:
            image = image.convert("RGBA")
            if image.getchannel("A").getextrema()[0] < 255:
                raise HTTPException(status_code=400, detail="images with transparency are not supported")
        picture = image.convert("RGB")
        if size is not None:
            picture = picture.resize((size, size), Image.Resampling.BICUBIC)
        return picture
    # 의도한 HTTP 오류와 메모리 부족은 그대로 올리고, 그 밖의 디코더 예외(SyntaxError, RuntimeError 등 OSError
    # 아닌 것도 있음)는 내부 정보를 감춘 400 하나로 바꾼다(from None으로 원래 예외 연쇄도 숨김).
    except (HTTPException, MemoryError):
        raise
    except Exception:
        # Pillow 디코더는 손상된 파일에 OSError 말고도 여러 예외(SyntaxError, RuntimeError 등)를 낸다.
        raise HTTPException(status_code=400, detail="the upload is not a readable image") from None


# 업로드 하나를 한계 + 1바이트까지만 읽어, 한계를 넘으면 전부 읽지 않고 413을 낸다. 빈 파일은 400. 파일 읽기는
# 막히는(blocking) 작업이라 디코딩과 함께 작업 스레드에서 부른다.
def _read_limited(upload: UploadFile) -> bytes:
    """업로드한 파일 하나의 바이트. 막히는 호출이므로 디코딩과 함께 작업 스레드에서 부른다."""
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload larger than {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    return data


# 그림을 PNG로 인코딩해 base64 문자열로 돌려준다(JSON 응답에 넣기 위해).
def _png_base64(image: Image.Image, **options) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# 점수 맵을 회색조 PNG로 바꾼다. 0은 검정, 임계값의 1.5배 이상은 흰색으로 고정해 범주마다 같은 기준으로
# "임계값에 얼마나 가까운가"를 보이게 한다. 임계값이 0 이하이면(합성 범주 등) 맵 최댓값으로 나눈다. +0.5는
# 반올림이다.
def heatmap_png(heatmap: np.ndarray, threshold: float) -> str:
    """base64 PNG(8비트 회색조, 256x256): 점수 0은 검정, 임계값의 1.5배 이상은 흰색."""
    scale = 1.5 * threshold if threshold > 0 else float(heatmap.max()) or 1.0
    grey = np.clip(heatmap / scale, 0.0, 1.0)
    return _png_base64(Image.fromarray((grey * 255.0 + 0.5).astype(np.uint8)))


# 목표 α에서 conformal 순위가 표본 수 안에 들어오는 가장 작은 n. α = 5%면 19장이다. 이보다 적으면 임계값이
# "보장 불가"(최댓값)가 되므로 /calibrate가 거절할 때 필요한 장수를 알려 주는 데 쓴다. α를 12자리 십진
# 문자열로 바꿔 Fraction으로 계산해 부동소수 오차로 1장 차이가 나지 않게 하고, 마지막에 conformal_rank로
# 확인한다.
def _min_calibration_images(alpha: float) -> int:
    """conformal 순위가 점수 n개 안에 들어오는 가장 작은 n, 곧 `ceil(1 / alpha) - 1`."""
    n = max(1, math.ceil(1 / Fraction(format(alpha, ".12g"))) - 1)
    while conformal_rank(n, alpha) > n:
        n += 1
    return n


# Host 헤더(포트 무시)가 허용 목록과 맞는지 본다. IPv6 "[::1]:8093"은 대괄호까지를 이름으로 본다.
# "*.example.com"은 하위 도메인 와일드카드, "*"는 모두 허용이다.
def _host_allowed(value: str, allowed: Sequence[str]) -> bool:
    """Host 헤더(포트는 무시)가 허용한 이름 가운데 하나와 맞는지."""
    host = value.strip().lower()
    host = host.partition("]")[0] + "]" if host.startswith("[") else host.partition(":")[0]
    for pattern in allowed:
        if pattern == "*" or host == pattern or (pattern.startswith("*.") and host.endswith(pattern[1:])):
            return True
    return False


# 다른 출처의 페이지가 브라우저를 시켜 보낸 요청인지 판단한다. 최신 브라우저가 붙이는 Sec-Fetch-Site를 먼저
# 보고(same-origin이나 none이 아니면 다른 출처), 없으면 Origin 헤더의 호스트를 Host와 비교한다. 둘 다 없으면
# 브라우저가 아닌 클라이언트(curl 등)로 보고 통과시킨다. 서비스가 쿠키를 쓰지 않아도 CSRF식 쓰기 요청을 막는
# 장치다(docs/security.md의 Anti-CSRF 오탐 사유).
def _cross_site(headers: Headers) -> bool:
    """다른 출처의 페이지가 브라우저를 시켜 보낸 요청인지(Fetch 메타데이터로, 없으면 Origin으로)."""
    site = headers.get("sec-fetch-site")
    if site is not None:
        return site.strip().lower() not in ("same-origin", "none")
    origin = headers.get("origin")
    if origin is None:
        return False  # 브라우저가 아니다
    return urlsplit(origin).netloc.lower() != headers.get("host", "").strip().lower()


# 관리자 요청이 거절될 이유를 돌려준다(없으면 None). 토큰을 설정하지 않으면 /calibrate는 꺼져 있다. 비교는
# hmac.compare_digest(상수 시간 비교)로 해서 응답 시간 차이로 토큰을 한 글자씩 알아내는 공격을 막는다.
def _admin_problem(headers: Headers, token: str | None) -> str | None:
    """요청이 임계값을 바꿀 수 없는 이유. 관리자 토큰을 가졌으면 None."""
    if not token:
        return "calibration is disabled: start the service with DEFECT_INSPECT_ADMIN_TOKEN"
    given = headers.get(ADMIN_HEADER, "")
    if not hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8")):
        return f"a valid {ADMIN_HEADER} header is required"
    return None


# 경로별 본문 한계: /calibrate는 여러 장이라 256MB, 그 밖은 파일 하나 + 여유.
def _body_limit(path: str) -> int:
    if path == ADMIN_PATH:
        return MAX_REQUEST_BYTES
    return MAX_UPLOAD_BYTES + MULTIPART_SLACK_BYTES


# 거절 응답(JSON {"detail": ...}). FastAPI의 HTTPException 응답과 같은 모양으로 맞춘다.
def _refusal(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


# 앱 전체를 감싸는 순수 ASGI 미들웨어. FastAPI 라우트보다 먼저 돌아서, 본문을 하나도 읽기 전에 Host·다른
# 출처·관리자 토큰·Content-Length를 검사한다(거절된 요청의 큰 업로드를 파싱하지 않음). 본문이 들어오는
# 동안에도 바이트 수를 세어 chunked 요청까지 한계를 지킨다. 나가는 모든 응답(이 클래스가 쓰는 500 포함)에 보안
# 헤더를 붙인다.
class _Guard:
    """앱 전체를 감싸는 순수 ASGI 감싸개.

    본문을 하나라도 읽기 전에: Host 검사, 다른 출처 검사, 관리자 토큰. 본문이 들어오는 동안: chunked
    요청에도 지켜지는 바이트 한계. 나가는 길에: 모든 응답에 보안 헤더(처리되지 않은 오류에 이 클래스가 쓰는
    JSON 500 포함).
    """

    def __init__(self, app, *, allowed_hosts: Sequence[str], admin_token: str | None):
        self.app = app
        self.allowed_hosts = tuple(allowed_hosts)
        self.admin_token = admin_token

    # 검사 순서: Host 허용 목록(400) → 상태를 바꾸는 메서드의 다른 출처 요청(403) → /calibrate 경로의 관리자
    # 토큰(403, 하위 경로 포함) → 선언된 본문 크기(413). 하나라도 걸리면 그 응답을 돌려준다.
    def _refuse(self, scope, headers: Headers) -> JSONResponse | None:
        path = scope["path"]
        # 거절한 요청의 본문은 앱에 닿지 않는다: 서버가 소켓에서 읽어 버리므로 클라이언트는 그래도 응답을
        # 받아 볼 수 있다.
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

        # 응답 시작 메시지에 보안 헤더를 더한다. setdefault라 라우트가 이미 정한 헤더는 덮어쓰지 않는다.
        # started는 응답이 이미 나가기 시작했는지 기억해 아래 예외 처리에서 쓴다.
        async def send_with_headers(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(raw=message.setdefault("headers", []))
                for key, value in SECURITY_HEADERS.items():
                    headers.setdefault(key, value)
            await send(message)

        # 본문 조각을 받을 때마다 누적 크기를 세고, 한계를 넘으면 413과 Connection: close로 끊는다.
        # Content-Length를 속였거나 끝없는 chunked 본문도 이렇게 막는다.
        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # 선언보다 많거나, 끝이 보이지 않는 chunked 본문: 응답하고 연결을 끊는다("Connection:
                    # close"면 서버가 계속 읽지 않고 연결을 닫는다). 본문을 파싱하는 동안 난 HTTPException은
                    # FastAPI가 그대로 통과시킨다.
                    raise HTTPException(
                        status_code=413, detail="request body too large", headers={"Connection": "close"}
                    )
            return message

        refusal = self._refuse(scope, Headers(scope=scope))
        if refusal is not None:
            await refusal(scope, receive, send_with_headers)
            return
        # 처리되지 않은 예외는 응답이 아직 시작되지 않았다면 로그만 남기고 내부 정보 없는 JSON 500으로
        # 바꾼다(스택 추적을 응답에 노출하지 않음). 이미 응답이 나가기 시작했다면 서버에 맡긴다.
        try:
            await self.app(scope, limited_receive, send_with_headers)
        except Exception:
            if started:
                raise
            logger.exception("unhandled error in %s %s", scope["method"], scope["path"])
            await _refusal(500, "internal server error")(scope, receive, send_with_headers)


# 동시에 계산하는 작업을 workers개, 기다리는 요청을 queue개로 묶고 나머지는 Retry-After와 함께 503으로
# 돌려보낸다. 전용 스레드 풀에서 돌리므로 기다리는 요청이 공용 스레드 풀(정적 파일 응답 등이 씀)을 붙잡지
# 않고, 디코딩한 큰 이미지가 메모리에 쌓이지 않는다(docstring). DAST 결과에서 모델이 실제로 도는 서비스도
# 대기열이 넘친 503을 내지 않았다(docs/security.md).
class _Gate:
    """동시에 계산하는 작업은 최대 `workers`개, 더 기다리는 요청은 최대 `queue`개. 나머지는 503을 받는다.

    작업은 전용 스레드에서 돈다. 이 한도 덕에 디코딩한 이미지가 메모리에 쌓이지 않고, 기다리는 요청이 공용
    스레드 풀(정적 파일 응답에 필요)의 스레드를 붙잡지 않으며, 큰 이미지 버퍼가 마흔 개 스레드에 흩어지지 않고
    이 몇 개 스레드의 메모리 아레나에 머문다.
    """

    def __init__(self, workers: int, queue: int):
        if workers < 1 or queue < 0:
            raise ValueError("workers must be at least 1 and queue at least 0")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inspect")
        self._lock = threading.Lock()
        self._admitted = 0
        self.capacity = workers + queue

    # 입장 관리: 잠금 아래에서 자리 수를 세어 넘치면 즉시 503, 아니면 자리를 차지하고 작업이 끝나면(예외여도)
    # 돌려준다.
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

    # 막히는 작업(파일 읽기·디코딩·ONNX 추론)을 전용 스레드 풀에서 돌리고 이벤트 루프는 기다리기만 한다.
    # 그동안 /healthz 같은 비동기 경로는 계속 응답한다.
    async def run(self, job: Callable, *args):
        return await asyncio.get_running_loop().run_in_executor(self._pool, job, *args)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


# 서비스 앱을 만든다. 인자로 주지 않은 설정은 DEFECT_INSPECT_* 환경 변수에서 읽는다(아티팩트 경로, 정밀도,
# 스레드, 관리자 토큰, 허용 Host, 작업자·대기열 수). uvicorn --factory가 이 함수를 불러 앱을 만든다. 테스트는
# inspectors를 직접 넘긴다.
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
    """서비스 앱. 주지 않은 인자는 `DEFECT_INSPECT_*` 환경 변수에서 가져온다."""
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

    # Swagger·ReDoc 문서 화면은 CDN 스크립트가 필요해 CSP와 맞지 않으므로 끈다. /openapi.json 명세는 남아 있어
    # DAST가 경로를 가져오는 데 쓴다(docs/security.md).
    app = FastAPI(title="defect-inspect", version=str(ARTIFACT_VERSION), docs_url=None, redoc_url=None)
    app.add_middleware(_Guard, allowed_hosts=allowed_hosts, admin_token=admin_token)
    app.state.inspectors = inspectors
    # 범주별 잠금은 임계값 상태(임계값, α, 보정 기록)를 읽고 바꿀 때만 잡는다. 채점은 잠금 밖에서 하므로 보정
    # 중에도 검사가 막히지 않고, 누구도 반쯤 바뀐 상태를 보지 않는다.
    locks = {name: threading.Lock() for name in inspectors}
    app.state.locks = locks
    # 시작할 때의 임계값·α·보정 기록을 보관해 DELETE /calibrate가 정확히 그 상태로 되돌리게 한다. 재보정한
    # 값은 메모리에만 있고 다시 띄우면 아티팩트 값으로 돌아간다(README).
    originals = {
        name: (i.threshold, i.meta.get("alpha", _MISSING), i.meta.get("calibration", _MISSING))
        for name, i in inspectors.items()
    }
    index_html = (STATIC_DIR / "index.html").read_bytes()

    # 입력 검증 실패(422) 응답에서 입력값과 파이썬 타입 이름을 빼고 위치·종류·짧은 메시지만 돌려준다. 오류
    # 응답으로 내부 구조나 보낸 값이 그대로 되비치지 않게 한다.
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 어디서 무엇이 틀렸는지만 알리고, 입력값이나 파이썬 타입 이름은 되비치지 않는다.
        detail = [
            {
                "loc": list(error.get("loc", ())),
                "type": error.get("type"),
                "msg": "Invalid value" if error.get("type") == "value_error" else error.get("msg"),
            }
            for error in exc.errors()
        ]
        return JSONResponse({"detail": detail}, status_code=422)

    # 라우트 수준의 관리자 검사. _Guard가 이미 본문을 읽기 전에 같은 규칙으로 막지만, 둘의 경로 판단이
    # 어긋나는 경우에 대비해 라우트에도 같은 규칙을 두는 이중 장치다.
    def require_admin(request: Request) -> None:
        problem = _admin_problem(request.headers, admin_token)
        if problem:
            raise HTTPException(status_code=403, detail=problem)

    # 범주 이름으로 검사기를 찾고, 없으면 404. 이름을 오류 메시지에 되비치지 않는다.
    def get(category: str) -> AnyInspector:
        inspector = inspectors.get(category)
        if inspector is None:
            raise HTTPException(status_code=404, detail="unknown category")
        return inspector

    # 범주의 임계값 상태 사본(응답용). 호출하는 쪽이 범주 잠금을 잡고 있어야 한다.
    def threshold_state(inspector: AnyInspector) -> dict:
        """범주 잠금을 잡은 채로 부른다."""
        calibration = inspector.meta.get("calibration")
        return {
            "threshold": inspector.threshold,
            "alpha": inspector.meta.get("alpha"),
            "calibration": dict(calibration) if isinstance(calibration, dict) else calibration,
        }

    # /categories 응답 한 줄: 모델 종류(reconstruction/patchcore)와 이름, 임계값, 입력 크기, 뱅크 행 수(복원
    # 방식은 0), 백본, 정밀도, α, 보정 기록.
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

    # /healthz는 compose의 상태 점검이 부른다. 비동기 함수라 검사 작업 뒤에 줄 서서 스레드를 기다리지 않는다.
    @app.head("/healthz", include_in_schema=False)
    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "categories": sorted(inspectors)}

    @app.get("/categories")
    async def categories() -> list[dict]:
        return [describe(name, inspectors[name]) for name in sorted(inspectors)]

    # POST /inspect: multipart의 image(파일)와 category(1~64자)를 받는다. heatmap=false면 맵을 만들지 않고,
    # preview=true면 모델이 실제로 본 리사이즈된 그림(input_png)을 함께 돌려준다(데모 페이지가 보여 줌).
    @app.post("/inspect")
    async def inspect(
        image: Annotated[UploadFile, File()],
        category: Annotated[str, Form(min_length=1, max_length=64)],
        heatmap: Annotated[bool, Query()] = True,
        preview: Annotated[bool, Query()] = False,
    ) -> JSONResponse:
        """이미지 한 장을 채점한다. `preview=true`면 `input_png`(모델이 본 그대로의 그림)를 더한다."""
        inspector = get(category)

        # 작업 스레드에서 도는 부분: 읽기 → 디코딩·리사이즈 → 채점 → 판정. 임계값은 잠금 아래에서 한 번만
        # 읽어, 같은 응답 안의 판정과 히트맵 눈금이 서로 다른 임계값을 쓰지 않게 한다. 판정은 점수 >
        # 임계값이면 불량이다(연구 코드와 같은 규칙).
        def work() -> dict:
            picture = _decode(_read_limited(image), inspector.size)
            result = inspector.inspect(picture, heatmap=heatmap)
            with locks[category]:
                threshold = inspector.threshold  # 한 번만 읽는다: 판정과 맵 눈금이 같은 값을 쓴다
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

        # latency_ms는 입장한 뒤부터 작업이 끝날 때까지의 시간이다. 작업 스레드를 기다린 시간과 업로드
        # 읽기·디코딩·추론·PNG 인코딩이 들어가고, FastAPI가 multipart 본문을 받는 시간과 응답 전송은 빠진다.
        # README의 "모델 경로 150ms"보다 큰 이유다.
        with gate.admitted():
            started = time.perf_counter()
            body = await gate.run(work)
        body["latency_ms"] = round((time.perf_counter() - started) * 1e3, 1)
        return JSONResponse(body)

    # POST /calibrate(관리자 전용): 현재 촬영 조건의 정상 사진 여러 장으로 그 범주의 임계값만 다시 정한다.
    # 모델과 PatchCore 뱅크는 그대로다(README: 모델은 그대로). α는 0 초과 0.5 이하. 장수가 부족해 보장이 안
    # 되면 allow_unguaranteed 없이는 400으로 거절하고 필요한 장수를 알려 준다.
    @app.post(ADMIN_PATH, dependencies=[Depends(require_admin)])
    async def calibrate(
        images: Annotated[list[UploadFile], File()],
        category: Annotated[str, Form(min_length=1, max_length=64)],
        alpha: Annotated[float, Form(gt=0.0, le=0.5)] = 0.05,
        allow_unguaranteed: Annotated[bool, Form()] = False,
    ) -> dict:
        """현재 촬영 조건의 정상 사진으로 범주의 임계값을 정한다(메모리에만 보관).

        임계값만 바뀐다: 모델(과 PatchCore 메모리 뱅크)은 읽어 들인 그대로다. 관리자 토큰이 필요하다.
        `alpha`의 conformal 순위에 비해 사진이 너무 적으면 `allow_unguaranteed`를 주지 않는 한 거절한다(주면
        임계값은 가장 큰 점수가 된다).
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

        # 자리 하나를 얻은 뒤 사진을 한 장씩 작업 스레드에 보낸다. 디코딩한 그림을 모아 두지 않아 메모리가
        # 장수에 비례해 늘지 않고, 장과 장 사이에 다른 검사 요청도 작업자를 얻을 수 있다.
        scores = []
        with gate.admitted():
            for upload in images:
                scores.append(await gate.run(score, upload))

        # 점수로 conformal 임계값을 정한 뒤, 잠금 아래에서 임계값·α·보정 기록(holdout, 장수, 보장 여부)을 한
        # 번에 바꾸고 이전 임계값과 함께 돌려준다. 위에서 구한 점수에 `Inspector.calibrate`와 같은 규칙과 기록
        # 방식을 적용한 것이다.
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

    # DELETE /calibrate?category=...(관리자 전용): 재보정을 되돌려 시작할 때의 임계값·α·보정 기록으로
    # 돌아간다. 원래 없던 키는 _MISSING 표시로 알아보고 지운다.
    @app.delete(ADMIN_PATH, dependencies=[Depends(require_admin)])
    async def reset_calibration(category: Annotated[str, Query(min_length=1, max_length=64)]) -> dict:
        """아티팩트 묶음을 읽어 들였을 때의 임계값, alpha, 보정 기록으로 되돌린다."""
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

    # 데모 페이지(static/index.html, 시작할 때 한 번 읽어 둠)와 그 정적 파일(/static/demo.js, demo.css).
    # OpenAPI 명세에는 넣지 않는다.
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def demo() -> Response:
        return Response(index_html, media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


# 모델 파일 없이 도는 가짜 ONNX 세션: 입력을 8×8 칸으로 나눠 칸마다 평균 색을 "패치 특징"(3차원)으로 낸다.
# PatchCore Inspector가 기대하는 run(names, feeds) 모양만 맞춘다.
class _GridMeanSession:
    """ONNX 모델을 대신하는 오프라인용 가짜: 8x8 격자의 칸마다 평균 색."""

    grid = 8

    def run(self, names, feeds):
        x = feeds["image"]
        b, c, h, _ = x.shape
        cell = h // self.grid
        out = x.reshape(b, c, self.grid, cell, self.grid, cell).mean(axis=(3, 5))
        return [np.ascontiguousarray(out.transpose(0, 2, 3, 1)).astype(np.float32)]


# 합성 범주 demo의 검사기: 회색 잡음 정상 이미지 24장의 칸 평균으로 뱅크를 만들고, 다른 24장으로 목표 5%
# 임계값을 잡는다. 회색 잡음과 다른 사진(실제 사진 등)은 높은 점수가 나와 불량으로 판정된다. 시드를 고정해 늘
# 같은 검사기가 된다.
def offline_inspector(size: int = 64, seed: int = 0) -> Inspector:
    """합성 검사기: 정상 이미지는 회색 잡음이고, 그 밖의 것은 점수가 높게 나온다."""
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


# 모델 파일 없이 띄우는 같은 서비스(범주 demo 하나). 테스트, 데모 확인, DAST 스캔(docs/security.md의 오프라인
# 서비스)에 쓴다. 보호 장치는 create_app과 같다.
def create_offline_app(**settings) -> FastAPI:
    """모델 파일 없이 합성 범주 `demo` 하나로 띄우는 같은 서비스."""
    return create_app(inspectors={"demo": offline_inspector()}, **settings)

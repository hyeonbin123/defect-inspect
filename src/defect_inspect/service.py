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
  기다릴 수 있는 요청 수(기본 16). 둘의 합보다 많은 업로드 요청(`/inspect`·`/calibrate`)이 진행 중이면 새
  요청은 본문을 받기 전에 `Retry-After`를 붙인 503으로 돌려보낸다. `/calibrate`는 한 번에 하나만 받는다.
- `DEFECT_INSPECT_EXPECTED_MODEL_SHA256`: 배포가 승인한 모델 파일의 SHA-256(64자리 16진수)을 쉼표로
  나열한다(정밀도마다 하나씩 적어 둘 수 있다, 대소문자와 공백은 가리지 않는다). 설정하면 시작할 때 모든 모델
  파일의 지문을 계산해, 이 목록에 없는 파일이 하나라도 있으면 뜨지 않는다. 없으면 meta.json 대조만 한다.

감사 기록: 임계값이 바뀌거나(`POST /calibrate`, `DELETE /calibrate`) 관리자 토큰 때문에 403으로 거절할 때마다
`defect_inspect.audit` 로거로 표준 출력에 JSON 한 줄을 쓴다(토큰 값과 업로드 파일 이름은 쓰지 않는다).
"""

# [파일 역할] FastAPI 검사 서비스. 사진 한 장과 범주를 받아 이상 점수·임계값·양품/불량 판정·히트맵을
# 돌려주고(POST /inspect), 관리자 토큰이 있으면 현재 촬영 조건의 정상 사진으로 그 범주의 임계값만 다시
# 잡거나(POST /calibrate) 되돌린다(DELETE /calibrate). GET /은 데모 페이지다.
# [설계 이유] torch 없이 onnxruntime CPU만으로 돈다(Docker 이미지 약 650MB, README). 6단계 이후 서빙 모델은
# Dinomaly ViT-S(dms-280-car)이고 ONNX 하나를 12개 범주가 같이 쓴다. 범주 폴더 meta.json의 kind로 복원
# 방식(ReconstructionInspector)과 4단계 PatchCore(Inspector)를 골라 둘 다 읽는다. 임계값이 "촬영 조건이
# 그대로일 때만" 목표를 지킨다는 실험 결과(3·7단계) 때문에, 조건이 바뀌면 모델은 그대로 두고 임계값만 다시
# 잡는 /calibrate를 둔다. 아티팩트에서 검사기를 읽어 띄울 때는 모델 세션마다 회색 이미지로 한 번 추론해
# (warm_up) 그 추론조차 실패하는 깨진 모델은 첫 요청 전에 시작을 멈추고, onnxruntime이 첫 실행에서만 치르는
# 비용이 있으면 시작 단계로 옮긴다.
# [보안 구성] 본문을 읽기 전에 Host 허용 목록, 다른 출처의 쓰기 요청 거절, 관리자 토큰, 진행 중인 업로드 요청
# 수(작업자 + 대기열을 넘으면 503 + Retry-After)를 검사하고(_Guard), 본문 크기를 스트리밍 중에도 제한하며,
# 모든 응답에 보안 헤더(CSP 등)를 붙인다. 이미지 디코딩은 형식·픽셀 수·비트 깊이·투명도를 제한하고, 동시에
# 계산하는 작업 수를 묶어 둔다(_Gate, 같은 상한의 두 번째 장치). 업로드를 임시 저장소에 쓰다 공간이 모자라면
# 400이 아니라 503으로 답해 재시도하게 한다. DAST(HawkScan, ZAP) 기록은 docs/security.md에 있다.
# [운영 기록] 판정 기준(임계값)을 바꾼 요청과 관리자 토큰 거절은 전용 감사 로거(AUDIT_LOGGER)가 표준 출력에
# JSON 한 줄씩 남긴다(언제, 무엇이 무엇에서 무엇으로, 어디서). 재보정한 값은 메모리에만 있어 재시작하면
# 사라지고 uvicorn 접근 로그에는 요청 줄과 상태 코드만 남으므로, 바뀐 값과 거절 사유를 사후에 재구성하려는
# 것이다. /categories는 시작할 때 대조한 모델 파일 지문의 앞 12자(model_sha256)를 보여, 어떤 모델이 떠 있는지
# 밖에서 확인할 수 있다. 배포 설정에 승인한 지문 목록(DEFECT_INSPECT_EXPECTED_MODEL_SHA256)을 주면, 아티팩트
# 볼륨 안의 meta 기록이 아니라 볼륨 밖의 그 목록과도 대조해 다른 모델 파일로는 뜨지 않는다.
# [연결] inspector.py(전처리·추론·임계값 상태), calibrate.py(conformal 임계값). 실행은 uvicorn
# defect_inspect.service:create_app --factory(Dockerfile, compose.yaml) 또는 모델 파일 없는
# create_offline_app(테스트와 보안 스캔용).

import asyncio
import base64
import errno
import hmac
import io
import json
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

# fastapi: 경로·폼·파일 업로드 검증과 응답. http_exception_handler는 FastAPI의 기본 HTTPException 처리기로,
# create_app이 등록하는 처리기가 임시 저장소 부족(503)이 아닌 모든 경우를 그대로 넘긴다. PIL: 업로드 디코딩과
# PNG 인코딩. starlette의 Headers·MutableHeaders: ASGI 수준(_Guard)에서 요청 헤더를 읽고 응답 헤더를 붙이는 데
# 쓴다. starlette의 HTTPException은 FastAPI HTTPException의 부모 클래스라, 처리기를 이것으로 등록하면
# Starlette가 폼 해석 중에 직접 내는 400까지 한 처리기가 덮는다. onnxruntime은 load_inspectors 안에서 늦게
# 가져온다.
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException

# calibrate: /calibrate가 연구 코드와 같은 conformal 규칙(l = ceil((n+1)(1-α))번째 점수)으로 임계값을 정하게
# 한다. inspector: 두 종류의 검사기와 아티팩트 형식 상수, 모델 파일 지문(meta 대조, 지문 형식 검사, 파일
# 해시 캐시).
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
    check_model_sha256,
    is_sha256_hex,
    model_file_sha256,
)

logger = logging.getLogger(__name__)
# 감사 기록 로거(README 검사 서비스 절). 임계값이 바뀔 때(POST·DELETE /calibrate)와 관리자 토큰 때문에 403으로
# 거절할 때 JSON 한 줄씩 쓴다. 위 logger와 따로 두는 이유: 서비스를 띄우는 uvicorn의 기본 로깅 설정
# (uvicorn.config.LOGGING_CONFIG)은 uvicorn.* 로거에만 핸들러를 달아, 이 모듈 logger의 INFO 줄은 어디에도
# 출력되지 않고 WARNING 이상만 시각 없이 표준 오류로 나간다. 감사 로거는 create_app이 표준 출력 핸들러를 직접
# 붙이므로(_attach_audit_handler) 로그 설정 없이도 docker logs에 남고, 운영자는 이 이름으로 따로 설정할 수
# 있다.
AUDIT_LOGGER = "defect_inspect.audit"
audit_log = logging.getLogger(AUDIT_LOGGER)


# 감사 줄을 쓸 때마다 그 순간의 sys.stdout에 쓰는 핸들러. StreamHandler(sys.stdout)처럼 만들 때의 객체를
# 붙잡아 두면, 그 객체가 나중에 바뀌거나 닫힌 뒤(테스트 도구의 출력 캡처가 테스트마다 그렇다) 엉뚱한 곳에
# 쓰거나 쓰기에 실패한다. 표준 라이브러리의 마지막 수단 핸들러(logging.lastResort, sys.stderr용
# _StderrHandler)와 같은 방식이다: StreamHandler.__init__을 건너뛰어 stream을 저장하지 않고, stream 속성이 늘
# 지금의 sys.stdout을 돌려준다. StreamHandler는 기록마다 flush하므로 줄이 버퍼에 머물지 않는다(Dockerfile의
# PYTHONUNBUFFERED=1과 함께 docker logs에 바로 보인다).
class _StdoutHandler(logging.StreamHandler):
    """기록할 때마다 지금의 `sys.stdout`에 쓰는 StreamHandler."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stdout


# 감사 기록 하나를 JSON 한 줄로 만든다(_AuditFormatter와 _audit이 같이 쓴다). json.dumps는 따옴표·역슬래시와
# 제어 문자(U+0000~U+001F, 줄 바꿈 포함)를 이스케이프하지만, ensure_ascii=False(한글 범주 이름을
# 그대로 읽히게)에서는 U+0085(NEL)·U+2028·U+2029를 그대로 둔다. 이 셋은 str.splitlines()와 일부
# 로그 도구가 줄의 끝으로 보므로, 그대로 두면 한 기록이 쪼개져 뒷부분이 따로 읽히는 가짜 줄이
# 된다. 예: uvicorn은 믿는 프록시에서 온 연결이면 X-Forwarded-For 값을 latin-1로 디코딩해 scope의
# client로 쓰므로 U+0085가 들어올 수 있다. 그래서 이 셋을 같은 뜻의 \u 이스케이프로 바꾼다. json.dumps가
# 내는 글에서 이 글자들은 문자열 값 안에만 있고 역슬래시는 늘 \\로 나오므로, 바꾼 뒤에도 올바른 JSON이고
# 읽으면 원래 값 그대로다. default=str은 예상 밖 타입이 섞여도 감사 줄 때문에 이미 끝난 변경의 응답이
# 500이 되지 않게 하는 안전장치다.
_LINE_BREAKS = str.maketrans({"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"})


def _one_line_json(entry: dict) -> str:
    return json.dumps(entry, ensure_ascii=False, default=str).translate(_LINE_BREAKS)


# 감사 줄 하나의 모양: {"time": UTC ISO 8601 시각(밀리초), "level": "INFO"|"WARNING", "event": ..., 그 밖의
# 필드}를 담은 JSON 객체 한 줄. 사람이 docker logs로 읽기도 하고 jq 같은 도구로 바로 거를 수 있게 앞에 다른
# 글을 붙이지 않는다. 시각은 기록이 만들어진 때(record.created)이고, 컨테이너와 호스트의 시간대가 달라도
# 헷갈리지 않게 UTC로 적는다. 필드 값은 줄을 쪼개지 못한다(_one_line_json). 기록에 감사 필드(_audit이 붙이는
# audit)가 없으면(다른 코드가 이 로거에 보통 문자열을 쓴 경우) message 필드 하나로 담는다.
class _AuditFormatter(logging.Formatter):
    """감사 기록을 시각·수준과 사건 필드를 담은 JSON 한 줄로 만든다."""

    def format(self, record: logging.LogRecord) -> str:
        entry = getattr(record, "audit", None)
        if not isinstance(entry, dict):
            entry = {"message": record.getMessage()}
        when = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        return _one_line_json({"time": when, "level": record.levelname, **entry})


# 감사 로거에 핸들러가 하나도 없을 때만 표준 출력 핸들러(_StdoutHandler + _AuditFormatter)를 붙인다. 왜:
# uvicorn의 기본 로깅 설정은 이 로거를 다루지 않아, 붙이지 않으면 감사 줄이 어디에도 나가지 않는다. 핸들러가
# 이미 있으면(운영자가 uvicorn --log-config로 이 로거를 설정했거나, 같은 프로세스에서 앞서 만든 앱이 붙였다)
# 손대지 않아 같은 줄이 두 번 나오지 않는다. 로거 수준을 따로 정하지 않았으면(NOTSET이면 루트의 WARNING을
# 물려받아 INFO 줄이 버려진다) INFO로 두고, 운영자가 정한 수준은 그대로 둔다. propagate를 끄는 것은 루트
# 로거에 핸들러를 단 로그 설정에서 같은 줄이 그쪽으로 한 번 더 나가지 않게 하려는 것이다.
# 호출 위치: create_app.
def _attach_audit_handler() -> None:
    if audit_log.handlers:
        return
    handler = _StdoutHandler()
    handler.setFormatter(_AuditFormatter())
    audit_log.addHandler(handler)
    if audit_log.level == logging.NOTSET:
        audit_log.setLevel(logging.INFO)
    audit_log.propagate = False


# 감사 줄 하나를 쓴다. 입력: 로그 수준, 사건 이름(event), 필드들(문자열·정수·실수·참거짓·None). 같은 내용을
# 기록의 message에도 JSON으로 넣어, 운영자가 로그 설정으로 이 로거에 자기 핸들러·형식을 붙여도 내용이 그대로
# 보이게 한다(이 모듈의 핸들러는 _AuditFormatter가 audit 필드로 시각·수준과 함께 한 줄을 만든다). message도
# _one_line_json으로 만들어 그 자체로 한 줄이다.
def _audit(level: int, event: str, **fields: Any) -> None:
    entry = {"event": event, **fields}
    audit_log.log(level, _one_line_json(entry), extra={"audit": entry})


# ASGI scope의 client((주소, 포트) 또는 None)를 uvicorn 접근 로그와 같은 '주소:포트' 문자열로 바꾼다(uvicorn의
# get_client_addr와 같은 모양이라 두 로그의 줄을 맞춰 볼 수 있다). 리버스 프록시 뒤에서는 uvicorn이 믿는
# 주소(--forwarded-allow-ips, 기본 127.0.0.1과 ::1)에서 온 연결이면 X-Forwarded-For로 바꾼 주소이고, 아니면
# 프록시 자신의 주소다(uvicorn의 ProxyHeadersMiddleware가 scope를 바꾸므로 접근 로그와 같은 값이다). 주소를 알
# 수 없으면 None.
def _client_address(client: Sequence[Any] | None) -> str | None:
    if not client:
        return None
    return f"{client[0]}:{client[1]}"


# 관리자 토큰 때문에 403으로 거절한 요청 하나를 WARNING 감사 줄로 남긴다. 입력: 사유 코드(_admin_problem의
# disabled·missing·mismatch), 메서드, scope의 path(디코딩된 경로), scope의 client. 토큰 값(보낸 값·설정한
# 값)은 받지도 쓰지도 않는다: 틀린 값은 진짜 토큰과 한 글자만 다를 수 있다. path는 요청하는 쪽이 마음대로 고를
# 수 있는 필드라 uvicorn 접근 로그처럼 urllib.parse.quote로 퍼센트 인코딩한다. 인쇄 가능한 ASCII만 남으므로,
# 디코딩된 줄 바꿈이나 U+2028 같은 줄 구분 문자로 감사 줄을 위조하거나 터미널 제어 문자를 심지 못하고,
# unquote하면 원래 경로가 된다. method는 서버(uvicorn의 h11)가 토큰 문자(인쇄 가능한 ASCII)만 받는다.
def _audit_refusal(reason: str, method: str, path: str, client: Sequence[Any] | None) -> None:
    _audit(
        logging.WARNING,
        "admin_refused",
        reason=reason,
        method=method,
        path=quote(path),
        client=_client_address(client),
    )


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
# SAFE_METHODS는 상태를 바꾸지 않는 메서드로, 다른 출처 검사에서 뺀다. INSPECT_PATH와 ADMIN_PATH는 업로드
# 본문을 받는 두 경로로, _Guard가 진행 중인 요청 수를 세는 대상이다.
ADMIN_HEADER = "X-Admin-Token"
ADMIN_PATH = "/calibrate"
INSPECT_PATH = "/inspect"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
STATIC_DIR = Path(__file__).parent / "static"
# 503 거절의 문구와 헤더. BUSY_DETAIL: 작업자와 대기열이 모두 찼을 때(_Guard가 본문 전에, _Gate가 두 번째
# 장치로 같은 문구를 쓴다). CALIBRATING_DETAIL: 재보정이 이미 하나 진행 중일 때. SPOOL_FULL_DETAIL: 업로드를
# 임시 저장소에 쓰다 공간이 모자랐을 때. RETRY_AFTER: 세 경우 모두 1초 뒤 재시도를 권한다(한 검사가 수백 ms라
# 자리가 곧 난다).
BUSY_DETAIL = "busy: try again shortly"
CALIBRATING_DETAIL = "busy: another calibration is running, try again shortly"
SPOOL_FULL_DETAIL = "busy: temporary storage is full, try again shortly"
RETRY_AFTER = {"Retry-After": "1"}
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
# 배포 설정의 기대 모델 지문을 담는 환경 변수 이름(README 검사 서비스 절, compose.yaml). 오류 메시지에도 이
# 이름을 그대로 넣어, 시작이 멈췄을 때 운영자가 어느 설정을 고칠지 바로 알게 한다.
EXPECTED_SHA256_ENV = "DEFECT_INSPECT_EXPECTED_MODEL_SHA256"


# 배포 설정의 기대 모델 지문 목록을 읽는다.
# [왜 필요한가] load_inspectors의 meta 대조는 모델 파일과 같은 아티팩트 볼륨 안의 meta.json 기록과 비교한다.
# 그래서 폴더를 통째로(다른 학습본 + 그 학습본의 meta) 바꿔 끼우면 대조를 통과하고, 지문 기록이 없는 4단계
# PatchCore 세트는 대조 자체가 없다. 승인한 지문을 볼륨 밖의 배포 설정(환경 변수, compose)에 두면 믿을 기준이
# 아티팩트 내용에서 배포 설정으로 옮겨 간다(컨테이너 이미지를 digest로 고정하는 것과 같은 생각이다).
# [형식] SHA-256 16진수 64자리를 쉼표나 공백(줄 바꿈 포함)으로 나열한다. 정밀도마다(FP32, 동적 INT8) 또는 범주
# 폴더마다 다른 모델 파일이 있으면 그 지문을 모두 적는다. 쓰이지 않는 지문이 섞여 있어도 된다(같은 설정을 두
# 정밀도에 쓰려고). 대문자는 소문자로 바꾸고 같은 지문은 하나로 친다(대문자로 옮겨 적은 값도 그대로 받으려고).
# 지문에는 쉼표도 공백도 없으므로 이렇게 나눠도 뜻이 모호해지지 않고, 지문 가운데에 공백이 끼면 두 조각 모두
# 64자리가 아니어서 오류가 된다.
# [출력] 소문자 지문의 frozenset. 값이 없거나 공백뿐이면 None(설정 안 함: meta 대조만 하는 이전과 같은 시작).
# compose는 설정하지 않은 변수를 빈 문자열로 넘기므로(`${...:-}`) 빈 값은 설정 안 함이어야 한다.
# [오류] 구분자만 있어 지문이 하나도 없거나 64자리 16진수가 아닌 항목이 있으면 ValueError로 시작을 멈춘다.
# 고정을 의도한 설정이 오타로 조용히 꺼지지 않게 하려는 것이다. 메시지에는 항목의 순서, 길이, 앞 12자만
# 담는다(다른 비밀값을 잘못 넣었을 때 로그에 다 찍히지 않게, 그리고 /categories·지문 대조 오류와 같은 길이로
# 맞추려고).
# 호출 위치: create_app, tests/test_service.py, tests/test_service_reconstruction.py.
def parse_expected_sha256(value: str | None) -> frozenset[str] | None:
    """`DEFECT_INSPECT_EXPECTED_MODEL_SHA256` 값을 소문자 SHA-256 지문의 집합으로 바꾼다.

    값이 없거나 공백뿐이면 None(설정 안 함)이다. 지문이 하나도 없거나 64자리 16진수가 아닌 항목이 있으면
    ValueError를 낸다.
    """
    if value is None or not value.strip():
        return None
    items = value.replace(",", " ").split()
    if not items:
        raise ValueError(
            f"{EXPECTED_SHA256_ENV} lists no digest: give SHA-256 hex digests separated by commas"
        )
    digests = set()
    for position, item in enumerate(items, start=1):
        digest = item.lower()
        if not is_sha256_hex(digest):
            shown = item[:12] + ("..." if len(item) > 12 else "")
            raise ValueError(
                f"{EXPECTED_SHA256_ENV} item {position} is not a SHA-256 hex digest (64 hex digits): "
                f"{shown!r} ({len(item)} characters)"
            )
        digests.add(digest)
    return frozenset(digests)


# 아티팩트 폴더에서 범주 검사기를 모두 읽는다. 범주 폴더의 meta.json kind로 클래스를 고르고, 같은 모델 파일을
# 쓰는 범주는 ONNX Runtime 세션 하나를 같이 쓴다(dms-280-car는 12개 범주가 세션 하나). 세션 run은 스레드
# 안전하고, 범주마다 세션을 만들면 메모리와 시작 시간이 범주 수만큼 늘기 때문이다. 문제가 있으면
# 폴더 이름을 담은 ValueError로 서비스 시작을 멈춘다. expected_sha256(배포 설정의 기대 지문,
# parse_expected_sha256의 결과)을 주면 모든 모델 파일을 그 목록과도 대조한다.
# 호출 위치: create_app, tests/test_service.py, tests/test_service_reconstruction.py.
def load_inspectors(
    artifacts: Path,
    *,
    precision: str = "fp32",
    threads: int | None = None,
    expected_sha256: Collection[str] | None = None,
) -> dict[str, AnyInspector]:
    """아티팩트 묶음에서 `meta.json`이 있는 모든 범주 폴더의 검사기.

    폴더의 `kind`가 클래스를 고른다: `ReconstructionInspector`(뱅크 없음) 또는 PatchCore `Inspector`. 각
    검사기는 그 클래스의 `load`가 그 폴더로 만드는 것과 같다(테스트가 둘을 비교한다). 다만 같은 모델 파일을
    쓰는 범주들(아티팩트 묶음은 범주 폴더 옆에 모델 파일 하나를 둔다)은 ONNX Runtime 세션 하나를 같이 쓴다:
    `session.run`은 스레드 안전하고, 범주마다 세션을 만들면 메모리와 시작 시간이 범주 수만큼 늘기 때문이다.
    아티팩트에 문제가 있으면 그 폴더 이름을 담은 `ValueError`를 낸다. meta에 모델 파일의 SHA-256이 적혀 있으면
    (`source.onnx_sha256` 등, `inspector.check_model_sha256`) 실제 파일과 대조하고, 다르면 같은 오류로 멈춘다.
    `expected_sha256`(소문자 SHA-256 지문의 모음, `parse_expected_sha256`의 결과)을 주면 meta 기록과 상관없이
    모든 모델 파일의 지문을 계산해, 그 모음에 없는 파일이 있으면 그 파일로 세션을 만들기 전에 같은 오류로
    멈춘다. 대조를 통과한 지문은 검사기의 `model_sha256`에 남기고(어느 대조도 하지 않았으면 None), 클래스의
    `load`와 다른 점은 이 속성 하나다.
    """
    import onnxruntime as ort

    artifacts = Path(artifacts)
    known = sorted({name for files in MODEL_FILES.values() for name in files})
    if precision not in known:
        raise ValueError(f"precision must be one of {known}, got {precision!r}")
    # 기대 지문은 parse_expected_sha256이 만든 소문자 지문의 모음이어야 한다. 빈 모음이나 형식이 다른
    # 값(문자열을 그대로 넘겨 글자 하나하나가 항목이 된 경우 등)은 어떤 파일도 통과하지 못해 원인을 알기
    # 어려운 오류가 되므로, 파일을 읽기 전에 따로 알린다.
    expected = None if expected_sha256 is None else frozenset(expected_sha256)
    if expected is not None and (not expected or not all(is_sha256_hex(d) for d in expected)):
        raise ValueError(
            "expected_sha256 must hold lowercase SHA-256 hex digests (see parse_expected_sha256)"
        )
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = int(threads)
    sessions: dict[Path, Any] = {}
    # 모델 파일 지문 캐시(실제 경로 → SHA-256). 12개 범주가 한 파일을 같이 쓰므로 해시는 파일마다 한 번만
    # 낸다.
    digests: dict[Path, str] = {}
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
            # meta에 적힌 모델 지문(dinomaly_serving export가 적는 source.onnx_sha256 등)과 실제 파일을
            # 대조한다. 다르면 이 임계값은 다른 모델의 점수로 정한 것이라 조용히 틀린 판정을 내므로 시작을
            # 멈춘다. 같으면 그 지문을, 기록이 없어 건너뛰었으면 None을 돌려준다(아래 model_sha256).
            digest = check_model_sha256(model, meta, precision, digests)
            # 배포 설정의 기대 지문(EXPECTED_SHA256_ENV, parse_expected_sha256 주석): 주어지면 meta에 기록이
            # 없는 파일도 해시를 계산해(같은 캐시라 파일마다 한 번) 목록과 대조한다. meta 대조가 통과해도
            # 목록에 없으면 멈춘다(폴더를 통째로 바꿔 끼운 경우). 세션을 만들기 전에 하므로 승인하지 않은
            # 파일은 onnxruntime이 읽지도 않는다. 메시지에는 지문의 앞 12자만 담는다(check_model_sha256의
            # 메시지, /categories와 같음).
            if expected is not None:
                digest = digest or model_file_sha256(model, digests)
                if digest not in expected:
                    listed = ", ".join(f"{d[:12]}..." for d in sorted(expected))
                    raise ValueError(
                        f"{model.name} sha256 {digest[:12]}... is not one of the expected model digests in "
                        f"{EXPECTED_SHA256_ENV} ({listed}): this deployment did not approve this model file"
                    )
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
        # 대조를 통과한 모델 파일 지문(64자리 16진수). /categories가 앞 12자를 보여, 어떤 모델 파일이 떠
        # 있는지 밖에서 확인할 수 있다. meta에 이 정밀도 파일의 기록이 없고(4단계 PatchCore, INT8 지문이 없는
        # 옛 세트) 기대 지문도 주지 않았으면 None이다: 대조하지 않은 파일의 해시를 계산해 확인된 것처럼 보이지
        # 않는다. 기대 지문을 주면 모든 파일을 그 목록과 대조하므로 늘 그 지문이 남는다.
        inspector.model_sha256 = digest
        found[folder.name] = inspector
    if not found:
        raise ValueError(f"no inspector artifacts under {artifacts}")
    return found


# 시작 예열에 쓰는 입력 화소값. 모든 채널이 128인 회색은 ImageNet 정규화((x/255 - mean)/std) 뒤 채널마다 약
# 0.07·0.21·0.43으로 0 근처에 놓여, 어떤 범주의 모델에도 극단적인 입력이 되지 않는다(검정 0은 -1.8~-2.1).
WARMUP_GREY = 128


# 아티팩트에서 읽은 검사기들의 모델 세션마다 검사 한 번을 미리 돌린다.
# [왜 필요한가] 두 가지다. (1) 시작 점검: 평범한 회색 입력에도 유한한 점수·맵을 내지 못하는 모델(깨진 파일,
# 다른 입출력)은 첫 요청에서 500을 내는 대신 시작 때 알린다. (2) 일회성 비용 옮기기: onnxruntime이 첫
# run에서만 하는 일(메모리 아레나 할당 등)이 있으면 재시작 직후 첫 요청이 아니라 여기서 치른다. 다만 서빙
# 모델(dms-280-car)을 onnxruntime CPU로 조용한 개발 PC에서 확인했을 때 새 세션의 첫 추론(약 230~250 ms)은
# 그 뒤 추론과 잡음 안에서 같았으므로 큰 지연 이득을 기대하는 장치는 아니다(bench는 관례로 예열 5회를 뺀다).
# 비용은 시작 때 세션당 추론 한 번(dms-280-car 약 0.3초)이다.
# [방식] 같은 세션을 쓰는 범주(아티팩트 묶음의 12개 범주는 모델 하나·세션 하나)는 한 번만 돌린다. 입력은 그
# 검사기 입력 크기의 회색(WARMUP_GREY) uint8 이미지이고, inspect(heatmap=False)로 실제 요청과 같은 경로(전처리
# → 모델 → 유한성 검사 → 판정)를 거친다. 결과는 버리고 임계값·보정 기록은 건드리지 않는다.
# [실패] 모델 오류나 유한하지 않은 점수·맵(FloatingPointError)이면 그 범주 이름과 원인을 담은 ValueError를
# 내서 서비스 시작을 멈춘다. 평범한 회색 입력도 처리하지 못하는 모델은 깨진 것이므로, 첫 요청에서 500을 내는
# 대신 시작 때 알리는 것이 의도된 동작이다(load_inspectors가 지문 불일치를 알리는 방식과 같다).
# 호출 위치: create_app(아티팩트에서 읽은 경로, 또는 warmup=True), tests/test_service.py.
def warm_up(inspectors: dict[str, AnyInspector]) -> list[str]:
    """모델 세션마다 회색 이미지로 검사 한 번을 돌려 둔다. 예열한 범주 이름(이름 순)을 돌려준다.

    같은 세션을 함께 쓰는 범주는 한 번만 돌린다. 검사가 실패하면 그 범주 이름을 담은 ValueError를 낸다.
    """
    warmed: list[str] = []
    seen: set[int] = set()
    for name in sorted(inspectors):
        inspector = inspectors[name]
        if id(inspector.session) in seen:
            continue
        seen.add(id(inspector.session))
        grey = np.full((inspector.size, inspector.size, 3), WARMUP_GREY, dtype=np.uint8)
        started = time.perf_counter()
        try:
            inspector.inspect(grey, heatmap=False)
        except Exception as err:  # 모델 오류, 유한하지 않은 점수·맵, 출력 모양 불일치
            raise ValueError(f"{name}: the warm-up inspection of a grey image failed: {err}") from err
        logger.info(
            "warmed up the model session of %s in %.0f ms", name, (time.perf_counter() - started) * 1e3
        )
        warmed.append(name)
    return warmed


# 업로드 바이트를 8비트 RGB 그림으로 바꾸고, 이미지가 아니거나 받지 않는 형식이면 400을 낸다. size를 주면 바로
# 모델 입력 크기로 줄여, 큰 원본을 메모리에 오래 두지 않는다. 리사이즈는 inspector.preprocess와 같은
# 호출(BICUBIC)이라 점수가 달라지지 않는다. 이미 RGB인 그림은 convert("RGB")를 거치지 않는다(아래 주석).
# EXIF 회전은 적용하지 않는다. 모델과 뱅크를 만든 리사이즈 캐시도 저장된 화소 그대로였기 때문이다(모듈
# docstring).
def _decode(data: bytes, size: int | None = None) -> Image.Image:
    """업로드 바이트 -> 8비트 RGB 그림. 적절한 이미지가 아니면 400을 낸다.

    `size`를 주면 `inspector.preprocess`와 같은 호출로 바로 모델 입력 크기로 줄인다. 그래서 점수는 그대로이고
    원본 크기로 디코딩한 그림을 계속 들고 있지 않는다. 이미 RGB인 그림은 원본 크기 사본을 만들지 않는다.
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
        # 이미 RGB면 그대로 쓴다. Pillow의 convert는 같은 모드를 요구받으면 그림 전체의 사본(self.copy())을
        # 돌려주므로, 늘 convert("RGB")를 부르면 RGB 업로드(JPEG·PNG 대부분)는 원본 크기 사본이 하나 더 생겨
        # 디코딩한 원본과 나란히 메모리에 있게 된다(Pillow는 RGB도 화소당 4바이트라 2,400만 화소면 약 96MB).
        # 화소 값은 두 길이 같아 점수는 바뀌지 않는다(tests/test_service.py가 모드별로 예전 경로와 비교한다).
        picture = image if image.mode == "RGB" else image.convert("RGB")
        if size is not None:
            picture = picture.resize((size, size), Image.Resampling.BICUBIC)
        else:
            # 리사이즈가 없으면 여기서 화소 디코딩을 끝내, 손상된 파일의 디코더 예외가 이 try 밖(부른 쪽)에서
            # 나지 않게 한다(Image.open은 머리만 읽고 화소는 처음 쓸 때 디코딩한다).
            picture.load()
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
# 반올림이다. 유한하지 않은 값이 있으면 그리지 않고 FloatingPointError를 낸다: NaN은 아래 clip·uint8 변환에서
# 0(검정)이 되어 '점수 0'처럼 조용히 그려지기 때문이다. 재구성 검사기는 run에서 먼저 막고, 여기는 PatchCore
# 경로의 맵까지 덮는 마지막 방어선이다.
def heatmap_png(heatmap: np.ndarray, threshold: float) -> str:
    """base64 PNG(8비트 회색조, 256x256): 점수 0은 검정, 임계값의 1.5배 이상은 흰색.

    맵에 유한하지 않은 값이 있으면 `FloatingPointError`를 낸다.
    """
    heatmap = np.asarray(heatmap)
    if not np.isfinite(heatmap).all():
        raise FloatingPointError("the heatmap is not finite")
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


# 관리자 요청이 거절될 이유를 (사유 코드, 응답 문구)로 돌려준다(거절하지 않으면 None). 사유 코드:
# disabled(토큰을 설정하지 않아 /calibrate가 꺼져 있음), missing(X-Admin-Token 헤더가 없거나 빈 값),
# mismatch(값이 틀림). 사유 코드는 감사 줄(_audit_refusal)에만 쓰고 응답 문구는 예전 그대로다(missing과
# mismatch는 같은 문구). 값 비교는 hmac.compare_digest(상수 시간 비교)로 해서 응답 시간 차이로 토큰을 한
# 글자씩 알아내는 공격을 막는다. 빈 값을 먼저 가르는 것은 보낸 쪽이 이미 아는 사실(헤더를 보냈는가)만 보므로
# 토큰에 대해 새로 드러내는 것이 없다.
def _admin_problem(headers: Headers, token: str | None) -> tuple[str, str] | None:
    """요청이 임계값을 바꿀 수 없는 이유 (사유 코드, 응답 문구). 관리자 토큰을 가졌으면 None."""
    if not token:
        return "disabled", "calibration is disabled: start the service with DEFECT_INSPECT_ADMIN_TOKEN"
    given = headers.get(ADMIN_HEADER, "")
    if not given:
        return "missing", f"a valid {ADMIN_HEADER} header is required"
    if not hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8")):
        return "mismatch", f"a valid {ADMIN_HEADER} header is required"
    return None


# 경로별 본문 한계: /calibrate는 여러 장이라 256MB, 그 밖은 파일 하나 + 여유.
def _body_limit(path: str) -> int:
    if path == ADMIN_PATH:
        return MAX_REQUEST_BYTES
    return MAX_UPLOAD_BYTES + MULTIPART_SLACK_BYTES


# 거절 응답(JSON {"detail": ...}). FastAPI의 HTTPException 응답과 같은 모양으로 맞춘다. headers는 503의
# Retry-After처럼 거절에 덧붙일 헤더다.
def _refusal(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


# FastAPI가 400으로 바꾼 본문 해석 오류의 원인이 '임시 저장소 공간 부족'(ENOSPC)인지 본다. 1MB가 넘는 업로드
# 파트는 Starlette가 /tmp의 SpooledTemporaryFile에 쓰는데(compose에서는 메모리 한도에 포함되는 tmpfs), 거기가
# 차면 쓰기가 OSError(errno 28)를 내고 FastAPI는 이것을 다른 해석 오류와 똑같이 400 "There was an error
# parsing the body"로 바꾼다(원래 예외는 __cause__에 남는다). 400은 클라이언트 잘못으로 읽혀 재시도되지
# 않으므로 이 경우만 503 + Retry-After로 돌려 서버 쪽의 일시적 상태임을 알린다. 원인 연쇄(__cause__, 없으면
# __context__)를 몇 단계 따라가 OSError를 찾는다. 입력: 처리기에 들어온 예외. 출력: ENOSPC가 원인이면 True.
def _spool_full(exc: BaseException) -> bool:
    cause = exc.__cause__ or exc.__context__
    for _ in range(8):
        if cause is None:
            return False
        if isinstance(cause, OSError) and cause.errno == errno.ENOSPC:
            return True
        cause = cause.__cause__ or cause.__context__
    return False


# 앱 전체를 감싸는 순수 ASGI 미들웨어. FastAPI 라우트보다 먼저 돌아서, 본문을 하나도 읽기 전에 Host·다른
# 출처·관리자 토큰·Content-Length·진행 중인 업로드 요청 수를 검사한다(거절된 요청의 큰 업로드를 파싱하지
# 않음). 본문이 들어오는 동안에도 바이트 수를 세어 chunked 요청까지 한계를 지킨다. 나가는 모든 응답(이
# 클래스가 쓰는 500 포함)에 보안 헤더를 붙인다.
# [왜 여기서 동시 수를 세는가] FastAPI는 엔드포인트 함수가 돌기 전에 multipart 본문을 모두 받아 해석하고,
# Starlette는 1MB가 넘는 파일 파트를 /tmp의 SpooledTemporaryFile로 내린다(spool_max_size). 그래서 엔드포인트
# 안의 _Gate만으로는 거절될 요청도 20MB(재보정은 256MB) 본문을 전부 받아 디스크에 쓴 뒤에야 503을 받고, 본문을
# 천천히 보내는 연결 여러 개가 /tmp(compose의 tmpfs, 메모리 한도에 포함)를 채울 수 있다. 이 층에서 요청이
# 도착한 순간 세면 상한을 넘는 업로드는 본문을 한 조각도 받기 전에 돌려보내고, 동시에 디스크에 있을 수 있는
# 업로드의 합이 (작업자 + 대기열) × 본문 한계로 묶인다(compose.yaml의 tmpfs 크기 계산).
class _Guard:
    """앱 전체를 감싸는 순수 ASGI 감싸개.

    본문을 하나라도 읽기 전에: Host 검사, 다른 출처 검사, 관리자 토큰, 선언된 본문 크기, 진행 중인 업로드 요청
    수(`capacity`를 넘으면 503 + Retry-After, `/calibrate`는 한 번에 하나). 본문이 들어오는 동안: chunked
    요청에도 지켜지는 바이트 한계. 나가는 길에: 모든 응답에 보안 헤더(처리되지 않은 오류에 이 클래스가 쓰는
    JSON 500 포함).
    """

    def __init__(self, app, *, allowed_hosts: Sequence[str], admin_token: str | None, capacity: int):
        self.app = app
        self.allowed_hosts = tuple(allowed_hosts)
        self.admin_token = admin_token
        # capacity: 동시에 진행할 수 있는 업로드 요청(POST /inspect와 /calibrate를 합친 수)의 상한.
        # create_app이 _Gate의 작업자 + 대기열과 같은 값을 넘기므로, 여기를 지난 요청은 _Gate에서 다시
        # 거절되지 않는다(_Gate는 같은 상한의 두 번째 장치로 남는다). _uploads는 요청이 도착할 때 +1, 응답이
        # 끝나거나 실패하면 -1이고, _calibrations는 그중 /calibrate의 수다(한 번에 하나만 받는다).
        # 서버(uvicorn)는 프로세스마다 이벤트 루프 하나에서 이 코드를 부르고 작업자 스레드는 이 수를 건드리지
        # 않으므로 잠금 없이 세어도 어긋나지 않는다.
        self.capacity = capacity
        self._uploads = 0
        self._calibrations = 0

    # 검사 순서: Host 허용 목록(400) → 상태를 바꾸는 메서드의 다른 출처 요청(403) → /calibrate 경로의 관리자
    # 토큰(403, 하위 경로 포함) → 선언된 본문 크기(413). 하나라도 걸리면 그 응답을 돌려준다. 관리자 토큰
    # 거절은 사유(disabled·missing·mismatch)를 담은 감사 줄도 남긴다(uvicorn 접근 로그에는 403만 남고 사유가
    # 없다). 진행 중인 요청 수의 검사(503)는 자리를 차지하는 부작용이 있어 __call__에서 이 검사들을 모두 지난
    # 뒤에 한다(_admit): 거절 사유가 앞쪽 검사부터 보이고, 어차피 거절될 요청이 자리를 차지하지 않는다.
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
                reason, detail = problem
                _audit_refusal(reason, scope["method"], path, scope.get("client"))
                return _refusal(403, detail)
        length = headers.get("content-length", "")
        if length.isdigit() and int(length) > _body_limit(path):
            return _refusal(413, "request body too large")
        return None

    # 입장 제한 대상인지: 업로드 본문을 받는 두 경로(POST /inspect, POST /calibrate)만 센다. 다른 요청(GET과
    # HEAD, DELETE /calibrate, 모르는 경로의 POST)은 본문이 없거나 라우터가 본문을 읽지 않고 답하므로 세지
    # 않는다.
    @staticmethod
    def _upload(scope) -> bool:
        return scope["method"] == "POST" and scope["path"] in (INSPECT_PATH, ADMIN_PATH)

    # 자리 검사. 본문을 받기 전에 부른다: 진행 중인 업로드 요청이 capacity개면 503 + Retry-After(_Gate와 같은
    # 문구), 재보정이 이미 하나 진행 중인데 또 /calibrate면 503(다른 문구). 자리가 있으면 차지하고 None을
    # 돌려준다. 차지한 자리는 _leave가 돌려준다.
    def _admit(self, scope) -> JSONResponse | None:
        if self._uploads >= self.capacity:
            return _refusal(503, BUSY_DETAIL, headers=RETRY_AFTER)
        if scope["path"] == ADMIN_PATH:
            if self._calibrations >= 1:
                return _refusal(503, CALIBRATING_DETAIL, headers=RETRY_AFTER)
            self._calibrations += 1
        self._uploads += 1
        return None

    # _admit이 차지한 자리를 돌려준다(응답이 끝났거나 처리 중에 실패했거나 클라이언트가 끊었을 때 모두).
    def _leave(self, scope) -> None:
        self._uploads -= 1
        if scope["path"] == ADMIN_PATH:
            self._calibrations -= 1

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
        # 업로드 요청은 여기서 자리를 얻어야 앱에 닿는다. 자리가 없으면 본문을 한 조각도 받지 않고 503을
        # 보낸다(거절한 요청의 본문은 위 거절들과 마찬가지로 서버가 소켓에서 읽어 버린다).
        upload = self._upload(scope)
        if upload:
            refusal = self._admit(scope)
            if refusal is not None:
                await refusal(scope, receive, send_with_headers)
                return
        # 처리되지 않은 예외는 응답이 아직 시작되지 않았다면 로그만 남기고 내부 정보 없는 JSON 500으로
        # 바꾼다(스택 추적을 응답에 노출하지 않음). 이미 응답이 나가기 시작했다면 서버에 맡긴다. 어느 길로
        # 끝나든 finally가 자리를 돌려주므로 실패한 요청이 자리를 영영 차지하는 일은 없다.
        try:
            await self.app(scope, limited_receive, send_with_headers)
        except Exception:
            if started:
                raise
            logger.exception("unhandled error in %s %s", scope["method"], scope["path"])
            await _refusal(500, "internal server error")(scope, receive, send_with_headers)
        finally:
            if upload:
                self._leave(scope)


# 동시에 계산하는 작업을 workers개, 기다리는 요청을 queue개로 묶고 나머지는 Retry-After와 함께 503으로
# 돌려보낸다. 전용 스레드 풀에서 돌리므로 기다리는 요청이 공용 스레드 풀(정적 파일 응답 등이 씀)을 붙잡지
# 않고, 디코딩한 큰 이미지가 메모리에 쌓이지 않는다(docstring). 같은 상한(capacity)을 _Guard가 본문을 받기
# 전에 먼저 적용하므로 보통은 여기까지 넘치는 요청이 오지 않고, 이 503은 두 번째 장치다. DAST 결과에서 모델이
# 실제로 도는 서비스도 대기열이 넘친 503을 내지 않았다(docs/security.md).
class _Gate:
    """동시에 계산하는 작업은 최대 `workers`개, 더 기다리는 요청은 최대 `queue`개. 나머지는 503을 받는다.

    작업은 전용 스레드에서 돈다. 이 한도 덕에 디코딩한 이미지가 메모리에 쌓이지 않고, 기다리는 요청이 공용
    스레드 풀(정적 파일 응답에 필요)의 스레드를 붙잡지 않으며, 큰 이미지 버퍼가 마흔 개 스레드에 흩어지지 않고
    이 몇 개 스레드의 메모리 아레나에 머문다. 같은 수(`capacity`)를 `_Guard`가 본문을 받기 전에 먼저 적용한다.
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
                raise HTTPException(status_code=503, detail=BUSY_DETAIL, headers=dict(RETRY_AFTER))
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
# 스레드, 기대 모델 지문, 관리자 토큰, 허용 Host, 작업자·대기열 수). uvicorn --factory가 이 함수를 불러 앱을
# 만든다. 테스트는 inspectors를 직접 넘긴다.
def create_app(
    artifacts: Path | str | None = None,
    *,
    inspectors: dict[str, AnyInspector] | None = None,
    precision: str | None = None,
    threads: int | None = None,
    expected_model_sha256: str | None = None,
    admin_token: str | None = None,
    allowed_hosts: Sequence[str] | None = None,
    workers: int | None = None,
    queue: int | None = None,
    warmup: bool | None = None,
) -> FastAPI:
    """서비스 앱. 주지 않은 인자는 `DEFECT_INSPECT_*` 환경 변수에서 가져온다.

    `expected_model_sha256`은 `DEFECT_INSPECT_EXPECTED_MODEL_SHA256`과 같은 형식의 문자열이다(None이면 그
    환경 변수, 빈 문자열이면 설정 안 함). 아티팩트 경로, 정밀도, 스레드처럼 아티팩트에서 검사기를 읽을 때만
    쓰고, 직접 넘긴 `inspectors`에는 쓰지 않는다(대조할 모델 파일이 없다).
    `warmup`이 None이면 아티팩트에서 검사기를 읽었을 때만 시작 예열(`warm_up`)을 하고, 직접 넘긴 `inspectors`
    (테스트·오프라인 앱)는 예열하지 않는다. True/False를 주면 어느 경로든 그대로 따른다.
    """
    loaded = inspectors is None
    if loaded:
        artifacts = artifacts or os.environ.get("DEFECT_INSPECT_ARTIFACTS")
        if not artifacts:
            raise RuntimeError("set DEFECT_INSPECT_ARTIFACTS to an artifact set directory")
        precision = precision or os.environ.get("DEFECT_INSPECT_PRECISION", "fp32")
        env_threads = os.environ.get("DEFECT_INSPECT_THREADS")
        threads = threads if threads is not None else (int(env_threads) if env_threads else None)
        # 배포 설정의 기대 모델 지문(parse_expected_sha256 주석). 형식이 틀리면 모델 파일을 읽기 전에 여기서
        # 멈춘다. 설정하지 않았으면 None이 넘어가 meta 대조만 한다.
        if expected_model_sha256 is None:
            expected_model_sha256 = os.environ.get(EXPECTED_SHA256_ENV)
        expected = parse_expected_sha256(expected_model_sha256)
        inspectors = load_inspectors(
            Path(artifacts), precision=precision, threads=threads, expected_sha256=expected
        )
    # 시작 예열: 아티팩트에서 읽은 실제 모델은 첫 요청 전에 세션마다 한 번 돌려 둔다(warm_up 주석). 직접 넘긴
    # 검사기는 보통 테스트의 가짜 세션이라 기본으로 건너뛴다. 실패하면 ValueError로 여기서 시작이 멈춘다.
    if warmup is None:
        warmup = loaded
    if warmup:
        warm_up(inspectors)
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
    # 감사 줄이 uvicorn의 기본 로깅 설정에서도 표준 출력에 나가게 한다(이미 핸들러가 있으면 그대로 둔다).
    _attach_audit_handler()

    # Swagger·ReDoc 문서 화면은 CDN 스크립트가 필요해 CSP와 맞지 않으므로 끈다. /openapi.json 명세는 남아 있어
    # DAST가 경로를 가져오는 데 쓴다(docs/security.md).
    app = FastAPI(title="defect-inspect", version=str(ARTIFACT_VERSION), docs_url=None, redoc_url=None)
    # _Guard의 동시 업로드 상한은 _Gate와 같은 수(작업자 + 대기열)다: 본문을 받기 전에 같은 한도를 적용해
    # 거절될 요청의 업로드가 디스크에 쓰이지 않게 한다.
    app.add_middleware(_Guard, allowed_hosts=allowed_hosts, admin_token=admin_token, capacity=gate.capacity)
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

    # HTTPException 응답은 FastAPI 기본 처리기(`{"detail": ...}`, 상태 코드와 헤더 그대로)에 맡기되, 업로드를
    # /tmp에 스풀하다 공간이 모자라 난 400만 503 + Retry-After로 바꾼다(_spool_full). 운영자가 임시 저장소
    # (compose의 tmpfs)가 작다는 것을 알 수 있게 경고 로그를 남긴다. starlette의 HTTPException(FastAPI
    # HTTPException의 부모)으로 등록해 FastAPI가 기본으로 등록하는 것과 같은 범위를 덮는다.
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code == 400 and _spool_full(exc):
            logger.warning(
                "the upload spool ran out of space while receiving %s %s", request.method, request.url.path
            )
            return _refusal(503, SPOOL_FULL_DETAIL, headers=RETRY_AFTER)
        return await http_exception_handler(request, exc)

    # 라우트 수준의 관리자 검사. _Guard가 이미 본문을 읽기 전에 같은 규칙으로 막지만, 둘의 경로 판단이
    # 어긋나는 경우에 대비해 라우트에도 같은 규칙을 두는 이중 장치다. 예: uvicorn --root-path로 띄우면 scope의
    # path가 root_path로 시작해 _Guard는 관리자 경로로 보지 않고, 라우터가 root_path를 떼어 이 라우트로
    # 보낸다. 여기서 거절해도 _Guard와 같은 감사 줄을 남긴다(path는 서버가 넘긴 scope의 path 그대로).
    def require_admin(request: Request) -> None:
        problem = _admin_problem(request.headers, admin_token)
        if problem:
            reason, detail = problem
            _audit_refusal(reason, request.method, request.scope["path"], request.scope.get("client"))
            raise HTTPException(status_code=403, detail=detail)

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

    # 임계값을 바꾼 요청 하나를 INFO 감사 줄로 남긴다. 범주 잠금을 잡은 채로, 바꾼 직후의
    # 상태(threshold_state의 사본)로 부른다: 그래서 줄의 이전·새 임계값이 실제로 일어난 변경과 맞고, 같은
    # 범주를 바꾸는 요청이 겹쳐도 줄의 순서가 변경 순서와 같다. n·alpha·guaranteed는 바뀐 뒤의 보정
    # 기록이다(reset이면 아티팩트의 값이고, 기록이 없으면 None). 업로드 파일 이름·토큰 같은 요청 내용은 넣지
    # 않는다.
    def audit_change(event: str, category: str, previous: float, state: dict, request: Request) -> None:
        calibration = state["calibration"] if isinstance(state["calibration"], dict) else {}
        _audit(
            logging.INFO,
            event,
            category=category,
            n=calibration.get("n"),
            alpha=state["alpha"],
            guaranteed=calibration.get("guaranteed"),
            previous_threshold=previous,
            threshold=state["threshold"],
            client=_client_address(request.scope.get("client")),
        )

    # /categories 응답 한 줄: 모델 종류(reconstruction/patchcore)와 이름, 시작할 때 대조한 모델 파일 지문의 앞
    # 12자(model_sha256: load_inspectors가 붙인 값. meta 기록도 기대 지문 설정도 없거나 직접 넘긴 검사기면
    # null), 임계값, 입력 크기, 뱅크 행 수(복원 방식은 0), 백본, 정밀도, α, 보정 기록. 지문을 /healthz가
    # 아니라 여기에 두는 것은 /healthz 응답을 compose의 상태 점검과 CI가 정확히 비교하기 때문이다. 앞
    # 12자(48비트)면 기록과 맞춰 보기에 충분하고 시작 거부 오류 메시지(check_model_sha256, 기대 지문 대조)와
    # 같은 길이다.
    def describe(name: str, inspector: AnyInspector) -> dict:
        with locks[name]:
            state = threshold_state(inspector)
        digest = getattr(inspector, "model_sha256", None)
        return {
            "category": name,
            "kind": inspector.kind,
            "model": inspector.meta.get("name"),
            "model_sha256": digest[:12] if digest else None,
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
    @app.post(INSPECT_PATH)
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
    # 되면 allow_unguaranteed 없이는 400으로 거절하고 필요한 장수를 알려 준다. request는 감사 줄의 client에만
    # 쓴다(FastAPI가 넣어 주는 인자라 OpenAPI 명세의 입력 항목은 그대로다).
    @app.post(ADMIN_PATH, dependencies=[Depends(require_admin)])
    async def calibrate(
        request: Request,
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
        # 방식을 적용한 것이다. 같은 잠금 안에서 감사 줄(event calibrate)을 남긴다.
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
            state = threshold_state(inspector)
            audit_change("calibrate", category, previous, state, request)
            return {"category": category, "previous_threshold": previous, **state}

    # DELETE /calibrate?category=...(관리자 전용): 재보정을 되돌려 시작할 때의 임계값·α·보정 기록으로
    # 돌아간다. 원래 없던 키는 _MISSING 표시로 알아보고 지운다. 이미 아티팩트 값이어서 바뀌는 것이 없어도
    # 관리자가 한 일이므로 같은 잠금 안에서 감사 줄(event reset)을 남긴다.
    @app.delete(ADMIN_PATH, dependencies=[Depends(require_admin)])
    async def reset_calibration(
        request: Request, category: Annotated[str, Query(min_length=1, max_length=64)]
    ) -> dict:
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
            state = threshold_state(inspector)
            audit_change("reset", category, previous, state, request)
            return {"category": category, "previous_threshold": previous, **state}

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

# 검사 서비스의 CPU 이미지: onnxruntime + FastAPI만 넣고 torch는 넣지 않는다(이미지 약 650MB, README 검사 서비스 절).
# 서빙 모델은 ONNX로 내보낸 것이라 학습 의존성(torch, timm, anomalib)이 필요 없다.
FROM python:3.11-slim

# UV_LINK_MODE=copy: 캐시 마운트와 대상 폴더가 다른 파일 시스템이라 하드 링크 대신 복사한다.
# UV_COMPILE_BYTECODE=1: 설치할 때 .pyc를 미리 만들어, 읽기 전용 루트(compose의 read_only)에서도 시작이 느려지지 않게 한다.
# PYTHONUNBUFFERED=1: 로그를 버퍼 없이 바로 내보낸다(docker logs에서 바로 보이게).
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# uv 실행 파일과 그 내려받기 캐시는 빌드 단계에만 마운트하므로 이미지에 남지 않는다. uv 버전은 태그로 고정한다.
# 의존성을 먼저 설치해, 소스만 바뀌었을 때 이 층을 다시 만들지 않게 한다. --frozen은 uv.lock을 그대로 쓰고,
# --no-dev --group serve는 기본 패키지와 서비스 그룹만 넣는다(CI의 audit 작업이 같은 집합을 취약점 대조한다).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.12,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group serve --no-install-project

COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.12,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group serve

# root가 아닌 사용자로 돈다(uid 10001). 컨테이너가 뚫려도 이미지 안의 파일을 바꾸기 어렵게 한다.
RUN useradd --create-home --uid 10001 inspect
USER inspect

# glibc: 1MiB 이상의 버퍼(디코딩한 업로드)는 해제되면 시스템으로 돌려준다. 기본값(자라는 문턱)에서는 힙에 남아,
# 컨테이너가 한 번 찍은 최대 메모리에 계속 머문다.
ENV MALLOC_MMAP_THRESHOLD_=1048576

# 여기에 아티팩트 세트를 마운트한다: `python -m defect_inspect.dinomaly_serving export` + `calibrate`로 만든 서빙
# Dinomaly ViT-S 모델(artifacts/dms-280-car) 또는 `python -m defect_inspect.export`의 PatchCore 세트. 서비스는
# 시작할 때 meta.json에 적힌 모델 지문(SHA-256)을 실제 모델 파일과 대조하고, 다르면 뜨지 않는다.
ENV DEFECT_INSPECT_ARTIFACTS=/artifacts
EXPOSE 8000

# 프로세스 하나로 띄운다(--workers나 WEB_CONCURRENCY를 주지 않는다). /calibrate가 바꾼 임계값과 보정 기록은 그 프로세스의
# 메모리에만 있어서, 작업자 프로세스를 둘 이상 띄우면 재보정이 그 요청을 받은 프로세스에만 남아 프로세스마다 판정이
# 달라진다. 처리량은 DEFECT_INSPECT_WORKERS(디코딩·채점 스레드 수)와 DEFECT_INSPECT_THREADS(onnxruntime 연산 스레드 수)로
# 조절한다(compose.yaml, README 검사 서비스 절).
# --no-server-header: 응답에 `server: uvicorn` 헤더를 붙이지 않아 서버 소프트웨어를 드러내지 않는다.
# --limit-concurrency 32: 동시에 열린 연결(유휴 keep-alive 연결 포함)이나 처리 중인 요청이 32개에 이르면 uvicorn이
# 32번째부터 앱을 부르지 않고 자체 503(본문 "Service Unavailable", Connection: close)을 낸다. 앱 안의 입장 제한(진행
# 중인 업로드 요청을 작업자 + 대기열, 기본 2 + 16 = 18개로 묶고 넘으면 본문을 받기 전에 JSON 503 + Retry-After)
# 바깥의 비상 상한으로, 본문 없이 연결만 쥐고 있는 클라이언트가 수백 개 쌓이는 것을 막는다. 18보다 여유를 둔 것은
# 데모 페이지를 띄운 브라우저가 keep-alive 연결 몇 개를 쥐고 있어도 정상 요청이 거절되지 않게 하려는 것이다. 이보다
# 작게 잡으면 유휴 연결만으로도 503이 날 수 있다. --timeout-keep-alive는 기본값 5초라 적지 않는다.
CMD ["/app/.venv/bin/uvicorn", "defect_inspect.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--limit-concurrency", "32"]

# Inspection service on the CPU: onnxruntime + FastAPI, no torch.
FROM python:3.11-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# uv and its download cache are mounted for the build steps only, so neither ends up in the image.
# Dependencies first, so that source changes do not rebuild this layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.12,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group serve --no-install-project

COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.12,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group serve

RUN useradd --create-home --uid 10001 inspect
USER inspect

# glibc: buffers of 1 MiB and more (decoded uploads) go back to the system when they are freed. With the
# default, growing threshold they stay in the heap, and the container sits at its peak memory for good.
ENV MALLOC_MMAP_THRESHOLD_=1048576

# Mount an artifact set here: `python -m defect_inspect.dinomaly_serving export` + `calibrate` (the served
# Dinomaly ViT-S model, artifacts/dms-280-car) or a PatchCore set of `python -m defect_inspect.export`.
ENV DEFECT_INSPECT_ARTIFACTS=/artifacts
EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "defect_inspect.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]

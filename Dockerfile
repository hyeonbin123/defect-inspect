# Inspection service on the CPU: onnxruntime + FastAPI, no torch.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so that source changes do not rebuild this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group serve --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --group serve

RUN useradd --create-home --uid 10001 inspect
USER inspect

# Mount an artifact set built by `python -m defect_inspect.export` here.
ENV DEFECT_INSPECT_ARTIFACTS=/artifacts
EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "defect_inspect.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

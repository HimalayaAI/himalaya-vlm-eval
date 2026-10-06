# syntax=docker/dockerfile:1
# nepeval results API — the image the studio deploy pulls (ghcr.io/himalayaai/nepeval-api).
#
# Read-only and small: the core (pydantic, yaml) plus the `api` and `s3` extras. No torch,
# no datasets — it shares a 2 GB host with the studio. Build for the studio host with
#   docker buildx build --platform linux/arm64 -t ghcr.io/himalayaai/nepeval-api:<sha> .

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first: they change less often than the code.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra api --extra s3 --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --extra api --extra s3 --no-editable

FROM python:3.12-slim-bookworm AS runtime

RUN useradd --system --create-home --uid 10001 nepeval
WORKDIR /app
COPY --from=builder --chown=nepeval:nepeval /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEPEVAL_REFRESH_SECONDS=60

USER nepeval
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# NEPEVAL_STORE (s3://bucket/prefix or a mounted path) is required at runtime.
CMD ["nepeval", "serve", "--host", "0.0.0.0", "--port", "8000"]

# syntax=docker/dockerfile:1
# nepeval runner — runs benchmarks and publishes results. Not deployed with the studio;
# run it on a GPU box or anywhere with network access to the model endpoints:
#
#   docker build -f docker/runner.Dockerfile -t nepeval-runner .
#   docker run --rm --gpus all -e OPENROUTER_API_KEY -e NEPEVAL_STORE=s3://… \
#     -v $PWD/results:/work/results nepeval-runner \
#     nepeval run --model gpt-4o,glm-ocr-nepali --bench nepalipixel --limit 2000
#
# EXTRAS selects in-process OCR engines (each pulls torch or a native binary):
#   --build-arg EXTRAS="easyocr trocr"
# VLMEvalKit is not baked in (it pins its own stack); mount a checkout and its venv and
# set VLMEVALKIT_DIR / VLMEVALKIT_PYTHON.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tessdata/nep.traineddata /usr/share/tesseract-ocr/5/tessdata/nep.traineddata
RUN uv sync --locked --no-dev --no-editable --extra run --extra s3 --extra tesseract \
    $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)

ENV PATH="/app/.venv/bin:$PATH" NEPEVAL_WORK_DIR=/work/results
WORKDIR /work
ENTRYPOINT []
CMD ["nepeval", "--help"]

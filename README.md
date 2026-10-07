# himalaya-vlm-eval

Benchmark harness and results API for vision-language and OCR models — Nepali-first,
with the standard OCR, document, chart, math, chat and general suites alongside.

- **Run** any model on any benchmark: `himeval run --model gpt-4o,glm-ocr-nepali --bench nepalipixel,category:math`
- **Publish** automatically to a result store (local directory or S3) when a run finishes.
- **Serve** the results as a read-only leaderboard API that HimalayaAI Studio proxies.

```
 runner (GPU box, laptop, CI)           result store                 results API              studio
┌───────────────────────────┐  publish ┌──────────────────┐  read  ┌──────────────────┐ HTTP ┌─────────────┐
│ himeval run               │ ───────▶ │ runs/<id>/        │ ◀───── │ himeval serve    │ ◀─── │ backend     │
│  native engine (Nepali,   │          │   result.json     │        │ /v1/leaderboard  │      │ proxies /v1 │
│   docs, OCR engines)      │          │   samples.jsonl.gz│        │ /v1/runs/…       │      └─────────────┘
│  VLMEvalKit engine (std.  │          └──────────────────┘        └──────────────────┘
│   suites, OpenCompass-    │   himeval import ──▶ (numbers from other leaderboards)
│   comparable)             │
└───────────────────────────┘
```

## Renamed from nepeval-ocr

| | Before | Now |
|---|---|---|
| Distribution | `nepeval-ocr` | `himalaya-vlm-eval` |
| Python package | `nepeval_ocr` | `himalaya_vlm_eval` |
| Command | `nepeval` | `himeval` |
| API image | `ghcr.io/himalayaai/nepeval-api` | `ghcr.io/himalayaai/himalaya-vlm-eval-api` |
| Environment | `NEPEVAL_*` (`NEPEVAL_STORE`, …) | `HIMEVAL_*` (`HIMEVAL_STORE`, …) |
| Adapter entry points | `nepeval_ocr.adapters` | `himalaya_vlm_eval.adapters` |

Published results keep working: the store layout and `result.json` schema are unchanged,
and `himeval eval` re-scores run directories saved under the old name. A run started
before the rename does not resume under `himeval run` (its config hash names the harness),
so it starts fresh.

## Install

```bash
uv venv && uv pip install -e '.[run,api,s3,dev]'      # everything except OCR engines
uv pip install -e '.[tesseract]'                       # + an engine: tesseract | easyocr | paddle | surya | trocr
```

Extras keep each deployment small: the API needs only the core plus `api` (and `s3`);
`run` adds datasets, Pillow, httpx and rapidfuzz; OCR engines bring their own stacks.

## Run a benchmark

```bash
export HIMEVAL_STORE=s3://himalaya-vlm-eval-results          # or a directory; omit to keep results local
export OPENROUTER_API_KEY=… TARKA_API_KEY=…

himeval list benchmarks                             # 37 benchmarks, 7 categories
himeval list models                                 # presets; anything else works ad hoc

himeval run --model glm-ocr-nepali,gpt-4o,easyocr-ne --bench nepalipixel --limit 2000
himeval run --model openrouter:qwen/qwen3-vl-8b-instruct --bench category:document
himeval run --model vllm:Qwen/Qwen2.5-VL-7B-Instruct --bench nepalipixel   # VLLM_BASE_URL
```

What `run` does, per model × benchmark:

1. Loads the benchmark (HF dataset at a pinned revision, a local manifest, or a
   nepal-pixel-synthesis output dir) — `--limit N` takes a **seeded random** subset.
2. Runs one **pre-flight** sample synchronously; a bad key or model fails in seconds.
3. Fans out with bounded concurrency, appending every prediction to
   `results/runs/<model>__<bench>__<config-hash>/predictions.jsonl`. Transient HTTP
   errors retry with backoff; 25 consecutive failures abort; Ctrl-C keeps what finished.
4. **Re-running the same command resumes** — successes are kept, failures retried.
5. Scores into `result.json`: every metric, a bootstrap 95% CI on the headline metric,
   breakdowns (by level, font, document type …), latency, token use, truncations.
6. **Publishes** to `HIMEVAL_STORE` unless more than 5% of samples errored
   (`--max-error-rate`). Errors are scored as worst-case, never dropped.

One failing pair never stops the matrix; a summary table prints at the end.

| Command | |
|---|---|
| `himeval eval <run_dir>` | re-score saved predictions (after a scorer change) |
| `himeval publish <run_dir>` | publish a run evaluated earlier or elsewhere |
| `himeval import results.json` | publish numbers from another leaderboard (`--format-help`) |
| `himeval leaderboard <bench>` | print a board from the store |
| `himeval serve` | the results API |

## Benchmarks

| Category | Native (himeval) | VLMEvalKit |
|---|---|---|
| OCR | `nepalipixel`, `nepalipixel-docs-page` | OCRBench, OCRBench v2 (EN/ZH), CC-OCR, olmOCR-Bench, TextVQA |
| Document | `nepalipixel-docs-kv`, `-qa`, `-layout` | DocVQA, InfographicVQA, OmniDocBench |
| Chart / figure | `nepalipixel-docs-table` | ChartQA, AI2D, CharXiv (reasoning, descriptive), SEED-Bench-2-Plus |
| Math | | MathVista, MathVerse, MATH-Vision (+mini), LogicVista, We-Math, DynaMath |
| Chat | | MM-Vet, LLaVA-Bench |
| General | | MMMU, MMBench v1.1, MMStar, RealWorldQA, BLINK, MME |
| Hallucination | | HallusionBench, POPE |

**Native benchmarks** are YAML (`src/himalaya_vlm_eval/catalog/benchmarks/`). A new OCR or
short-answer VQA set is a file, no code:

```yaml
id: my-devanagari-set
engine: native
display_name: My Devanagari Set
category: ocr
primary_metric: char_accuracy
scale_max: 1.0
task: ocr
dataset: {source: hf, repo: org/dataset, revision: <sha>, split: test, image: image, references: text}
prompt: Transcribe all text in this image exactly as written.
metrics: [char_accuracy, cer, wer, exact_match]
```

Extra catalog directories load from `HIMEVAL_CATALOG=/path/a:/path/b` and override
built-ins by id.

**NepaliPixel document benchmarks** read a nepal-pixel-synthesis output directory
(`NEPALIPIXEL_DOCS_DIR`, rows with `split: benchmark`): key-value extraction (field F1),
QA with "ANSWER NOT PRESENT" negatives, page transcription with reading order (Kendall τ),
tables (TEDS on the generator's cell grid) and layout (role F1 at IoU 0.5). See
[docs/SCORING.md](docs/SCORING.md).

**VLMEvalKit benchmarks** run through a VLMEvalKit checkout so scores match the
OpenCompass leaderboard's method. Setup and the judge model: [docs/VLMEVALKIT.md](docs/VLMEVALKIT.md).

## Models

Presets live in `src/himalaya_vlm_eval/catalog/models/`: Himalaya models on Tarka
(`glm-ocr-nepali` via `/ocr`, `himalaya-gemma-4-*` via chat), GPT, Gemini, Claude, Qwen-VL,
Llama 4, Mistral, GLM-4.5V via OpenRouter, and Tesseract, EasyOCR, PaddleOCR, Surya,
TrOCR in-process. Anything else, ad hoc:

| Spec | Endpoint |
|---|---|
| `openrouter:<vendor>/<model>` | OpenRouter (`OPENROUTER_API_KEY`) |
| `openai:<model>` | OpenAI (`OPENAI_API_KEY`) |
| `tarka:<model>` | Tarka chat completions (`TARKA_API_KEY`) |
| `vllm:<model>` | any local OpenAI-compatible server at `VLLM_BASE_URL` (vLLM, SGLang, llama.cpp) |

Override any adapter parameter: `--model-arg max_tokens=8192 --model-arg max_image_side=2048`.
A new kind of backend is a `Model` subclass registered with `@register_adapter("name")` or
the `himalaya_vlm_eval.adapters` entry-point group.

## Results API

`himeval serve` (or the `ghcr.io/himalayaai/himalaya-vlm-eval-api` image) serves the store
read-only. Internal by design — the studio backend proxies it. Contract:
[docs/API.md](docs/API.md); studio wiring: [docs/STUDIO_INTEGRATION.md](docs/STUDIO_INTEGRATION.md).

```bash
HIMEVAL_STORE=s3://himalaya-vlm-eval-results himeval serve --port 8000
curl localhost:8000/v1/leaderboard/nepalipixel
```

Rankings follow LMArena: a model's rank is 1 + the number of models whose 95% CI is
entirely better, so models that cannot be told apart share a rank.

## Development

```bash
uv sync --extra run --extra api --extra s3 --extra dev
uv run ruff check src tests && uv run pytest
```

`docker build .` builds the API image; `docker/runner.Dockerfile` the runner.

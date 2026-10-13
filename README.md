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

### Logs and resuming

| Where | What |
|---|---|
| console | progress every 15 s (done/total, rate, ETA, errors, retried requests), the first 5 sample errors, an error summary by kind, the score with its CI |
| `results/runs/<run>/run.log` | everything for that run at DEBUG: the full config, every sample's outcome and latency, every error and adapter retry with its reason, truncations, timings; appended across resumes |
| `results/logs/run-<time>-<pid>.log` | the whole `himeval run` session, including tracebacks the console only summarises |
| `results/vlmevalkit/<model>__<dataset>.log` | VLMEvalKit's own output; the console gets a heartbeat with its latest line every minute |

`-v` shows the DEBUG lines on the console too.

A run can be stopped at any point — Ctrl-C, `kill`, `docker stop`, a preempted cloud GPU,
a dropped SSH session (SIGINT, SIGTERM and SIGHUP are all handled) — and **re-running the
same command continues where it stopped**:

- every answer is appended to `predictions.jsonl` as it arrives and synced to disk at
  each progress report; a line torn by a hard kill is skipped on read;
- on a stop, requests already in flight finish and are saved (a second Ctrl-C drops
  them); queued samples are not started;
- the run directory is keyed by a hash of model, parameters, benchmark and subset, so the
  same command finds it; earlier errors are retried (`--no-retry-errors` keeps them);
- in a matrix, finished pairs are skipped (their result is reused, publishing is
  idempotent) and the stopped pair resumes;
- VLMEvalKit runs use its `--reuse`, so its saved predictions are picked up the same way.

| Command | |
|---|---|
| `himeval eval <run_dir>` | re-score saved predictions (after a scorer change) |
| `himeval publish <run_dir>` | publish a run evaluated earlier or elsewhere |
| `himeval import results.json` | store numbers from another leaderboard (`--format-help`); kept for reference, never ranked |
| `himeval meta refresh` | refresh prices and context lengths (OpenRouter + overrides) |
| `himeval leaderboard <bench>` | print a board from the store |
| `himeval serve` | the results API |

## Benchmarks

| Category | Native (himeval) | VLMEvalKit |
|---|---|---|
| OCR | `nepalipixel`, `nepalipixel-docs-page` | OCRBench, OCRBench v2 (EN/ZH), CC-OCR, olmOCR-Bench, TextVQA |
| Document | `nepalipixel-docs-kv`, `-qa`, `-layout`; `nepalpixel-vlm-kie`, `-ocr`, `-abstain` | DocVQA, InfographicVQA, OmniDocBench |
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
primary_metric: akshara_accuracy   # char_accuracy for non-Devanagari text
scale_max: 1.0
task: ocr
dataset: {source: hf, repo: org/dataset, revision: <sha>, split: test, image: image, references: text}
prompt: Transcribe all text in this image exactly as written.
metrics: [akshara_accuracy, acer, cer, char_accuracy, wer, exact_match]
```

Extra catalog directories load from `HIMEVAL_CATALOG=/path/a:/path/b` and override
built-ins by id.

**NepaliPixel document benchmarks** read a nepal-pixel-synthesis output directory
(`NEPALIPIXEL_DOCS_DIR`, rows with `split: benchmark`): key-value extraction (field F1),
QA with "ANSWER NOT PRESENT" negatives, page transcription with reading order (Kendall τ),
tables (TEDS on the generator's cell grid) and layout (role F1 at IoU 0.5). See
[docs/SCORING.md](docs/SCORING.md).

**NepalPixel VLM Bench** ([`himalaya-ai/nepalpixel-vlm-bench`](https://huggingface.co/datasets/himalaya-ai/nepalpixel-vlm-bench),
gated, CC BY-NC 4.0) is three benchmarks over 2,130 synthetic Nepali government documents,
scored exactly as the dataset's own `score_vlm_bench.py` does (NFC + whitespace, no digit
folding; `src/himalaya_vlm_eval/vlm_bench.py`):

| Benchmark | Board | Headline | Also recorded |
|---|---|---|---|
| `nepalpixel-vlm-kie` | Document › Nepali Q&A | ANLS (τ = 0.5) per question | exact match |
| `nepalpixel-vlm-ocr` | Document › Nepali Page | CER, micro-averaged over pages (Σ edits ÷ Σ chars) | — |
| `nepalpixel-vlm-abstain` | Document › Nepali Q&A | abstention rate | leak rate |

`--limit N` draws N rows of a config: N *documents* for `kie`, about 15 questions each. A
question on a later sheet of a multi-page document is asked with that sheet (or sheets).

**VLMEvalKit benchmarks** run through a VLMEvalKit checkout so scores match the
OpenCompass leaderboard's method. Setup and the judge model: [docs/VLMEVALKIT.md](docs/VLMEVALKIT.md).

## What the board shows

**Only results this harness measured, of models in the catalog** (picked 2026-10-13).
Leaderboards worth trusting run the benchmarks themselves, so the results API ranks
`source.kind == "measured"` runs of `catalog.board_models()` and nothing else. An imported
number (`himeval import`) or an ad-hoc model (`openrouter:…`, `vllm:…`) stays in the store
and in `/v1/runs`, and never takes a rank. The Arena import that filled the board's first
categories until then was removed with this rule.

The catalog is therefore the comparison set: HimalayaAI's models and the open models of
the same size class they are measured against (`catalog/models/third_party.yaml`).

## Leaderboard layout and prices

`catalog/boards.yaml` lays the boards out the way the studio shows them: two top-bar types,
**Vision** and **Document**, each with a side list of categories, every one filled by suites
run here (Vision: Nepali OCR · OCR · Chart · Math · Chat · General · Hallucination; Document:
Nepali Fields · Nepali Q&A · Nepali Page · Nepali Table · Nepali Layout · Document Q&A ·
Parsing). A benchmark goes on its category's default
board unless its YAML says `board:`. Served by `GET /v1/boards`.

Every model carries `params_b`, its total parameter count (Hugging Face's safetensors total,
embeddings included), which the studio plots scores against. Price ($ per 1M input/output
tokens) and context length drive the board's filters where a model has them; self-hosted
models have no price. Results never carry them (prices change, results do not). `himeval meta refresh`
writes `meta/models.json` from OpenRouter's public models API plus `catalog/model_meta.yaml`
overrides, and the API joins it on at serve time. Unmatched models show N/A. Matching only crosses
reasoning-effort suffixes and snapshot dates, because a wrong price is worse than none.

## Models

Presets live in `src/himalaya_vlm_eval/catalog/models/`:

| File | Models | Reached through |
|---|---|---|
| `himalaya.yaml` | `glm-ocr-nepali` (1.33B), `himalaya-gemma-4-*` (Gemma 4 E2B, 5.12B), `glm-ocr` as the base control | Tarka |
| `third_party.yaml` | the 1–5B comparison set: Gemma 4 E2B (our base, untuned), Qwen3.5 2B and 4B, Ministral 3 3B, Gemma 3 4B | vLLM we run (`base_url_env: VLLM_BASE_URL`), or OpenRouter |
| `ocr_engines.yaml` | Tesseract, EasyOCR, PaddleOCR, Surya, TrOCR | in-process |

A self-hosted entry names its server through the environment, so one entry runs against
whichever vLLM the runner started:

```bash
vllm serve Qwen/Qwen3.5-2B --port 8001 &
VLLM_BASE_URL=http://localhost:8001/v1 himeval run --model qwen3.5-2b --bench nepalipixel
```

Anything else, ad hoc (stored, never ranked):

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

Rankings follow LMArena's convention: a model's rank is 1 + the number of models whose 95%
CI is entirely better, so models that cannot be told apart share a rank; the *rank spread*
runs from there to the worst rank the CIs allow.

## Development

```bash
uv sync --extra run --extra api --extra s3 --extra dev
uv run ruff check src tests && uv run pytest
```

`docker build .` builds the API image; `docker/runner.Dockerfile` the runner.

Going to production, releasing and publishing results: [docs/PRODUCTION.md](docs/PRODUCTION.md).

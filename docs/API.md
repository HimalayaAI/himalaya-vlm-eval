# Results API — `/v1`

Read-only view of the result store. Internal: the studio backend calls it over the
compose network and the browser never does. If `HIMEVAL_API_TOKEN` is set, every `/v1`
route requires `Authorization: Bearer <token>`; `/health` stays open for probes.

**Board routes show measured runs of catalog models only** (picked 2026-10-13):
`/v1/boards`, `/v1/benchmarks`, `/v1/models` and both `/v1/leaderboard` routes rank
`source.kind == "measured"` results of models listed in the catalog, and nothing else. An
imported number or an ad-hoc model is still stored and listed under `/v1/runs`.

Responses carry `ETag` and `Cache-Control: private, max-age=30`; send `If-None-Match` to
get `304`. The ETag changes only when the store's contents change.

| Env | |
|---|---|
| `HIMEVAL_STORE` | required — a path or `s3://bucket/prefix` (`AWS_*` credentials, `AWS_ENDPOINT_URL` for S3-compatible) |
| `HIMEVAL_REFRESH_SECONDS` | store re-list interval, default 60; new results appear within it |
| `HIMEVAL_API_TOKEN` | optional shared secret |

## `GET /health`

`200 {"ok": true, "version", "schema_version", "store", "runs", "invalid", "last_refresh", "last_error", "meta_models", "meta_generated_at"}`;
`503` while starting or when the last store listing failed. `invalid` counts result files
that failed validation — they are skipped, never served.

## `GET /v1/benchmarks?category=`

Every catalog benchmark.

```json
{"data": [{"id": "nepalipixel", "display_name": "NepaliPixel OCR", "category": "ocr",
           "description": "…", "url": "…", "language": "ne", "primary_metric": "char_accuracy",
           "higher_is_better": true, "scale_max": 1.0, "full_cases": 15000, "version": "1",
           "engine": "native", "models": 7, "updated_at": "2026-10-06T15:40:47Z"}]}
```

`category` ∈ `ocr · document · chart · math · chat · general · hallucination`.
`models: 0` means listed but not yet run — the UI can hide those.

BenchmarkInfo also carries `board` (`"vision/ocr"`, see `/v1/boards`), `style_control`
(`null` on every current board) and `cases_unit` (`cases`). `scale_max: null` would mark an
unbounded rating, which has no `score_100` and never enters the overview; no current
benchmark is one.

## `GET /v1/boards`

The leaderboard layout from `catalog/boards.yaml`: top-bar types → side-list categories →
the boards in each. A category with several boards shows a switcher. Every board the
catalog defines is listed, with or without results (`models: 0`).

```json
{"types": [{"id": "vision", "label": "Vision", "categories": [
  {"id": "ocr", "label": "OCR", "boards": [
    {"benchmark": "ocrbench", "engine": "vlmevalkit", "style_control": null,
     "scale_max": 1000, "cases_unit": "cases", "models": 0, "updated_at": null, …}]},
  {"id": "nepali-ocr", "label": "Nepali OCR", "boards": [{"benchmark": "nepalipixel", …}]}]},
 {"id": "document", "label": "Document", "categories": […]}]}
```

Types: `vision · document`. Vision categories: `nepali-ocr · ocr · chart · math · chat ·
general · hallucination`. Document: `nepali-fields · nepali-qa · nepali-page ·
nepali-table · nepali-layout · doc-qa · parsing`. Every category holds at least one suite
this harness runs. These ids are the studio's vocabulary; changing one is a change in both repos.

## `GET /v1/models`

Models with at least one result: `{"data": [{…ModelInfo, "benchmarks": ["mmstar", …], "pricing", "context_length"}]}`.
ModelInfo = `id, display_name, org, kind (vlm|ocr_engine), open_weights, params_b, license, url, endpoint`.
`pricing` = `{"input", "output", "blended", "source"}` in $ per 1M tokens, or `null` when
unknown (see *Prices* below).

## `GET /v1/leaderboard/{benchmark_id}`

Query filters: `kind`, `open_weights` (the studio's *License type*: `true` open,
`false` proprietary), `org` (comma list), `include_subsets` (default true), and ranges `score_min`/`score_max`,
`input_price_min`/`input_price_max`, `output_price_min`/`output_price_max` ($ per 1M
tokens), `context_min`/`context_max` (tokens). A row whose value is unknown (no price) is
dropped only when that range is set. Ranks are recomputed over the filtered rows.

```json
{"benchmark": {…BenchmarkInfo},
 "bounds": {"score": [0.41, 0.93], "input_price": [0.05, 0.1], "output_price": [0.1, 0.1],
            "context_length": [131072, 262144]},
 "attribution": null, "data_license": null,
 "meta_generated_at": "2026-10-13T08:34:33+00:00",
 "data": [{"rank": 1, "rank_worst": 2, "position": 1,
           "model": {"id": "himalaya-gemma-4-bf16", "org": "HimalayaAI", "params_b": 5.12, …},
           "pricing": null, "context_length": null, "pareto": false,
           "score": 0.912, "ci_low": 0.905, "ci_high": 0.919, "score_100": 91.2,
           "metrics": {"char_accuracy": 0.912, "cer": 0.101, "wer": 0.21, …},
           "cases": 2000, "errors": 3, "partial": true,
           "source": {"kind": "measured", "harness": "himalaya-vlm-eval", "harness_version": "0.2.0",
                      "judge": null, "url": null, "notes": null},
           "run_id": "…", "created_at": "…", "has_samples": true}]}
```

- `rank` — LMArena-style: 1 + models statistically better (non-overlapping 95% CI).
  Ties are expected; render "1", "1", "3".
- `rank_worst` — 1 + every model whose CI reaches above this one's: the worst rank the
  data allows. `rank`–`rank_worst` is the *rank spread*.
- `position` — plain ordinal by point score; the *Rank* column.
- `pricing` / `context_length` — joined from `meta/models.json` at serve time; `null` when
  unknown — always for a model we serve ourselves. `pricing.blended` = (3 × input + output) / 4.
- `pareto` — on the price/performance frontier: no cheaper-or-equal priced model scores
  higher. Unpriced models never are.
- `score` is on the benchmark's scale (`scale_max`, direction `higher_is_better`);
  `score_100` maps it to 0–100 higher-is-better for cross-suite display.
- `partial` — the run used a subset (`cases < full_cases`). Show it.
- `model.params_b` — total parameters (safetensors total), the studio chart's x-axis.
- `source.kind` — always `measured` on a board route.
- `source.judge` — LLM judge model when scoring used one.
- `cases` counts `benchmark.cases_unit`: evaluated samples.
- `bounds` — extents of the board under the non-range filters, for sliders; moving a
  range never moves its own ends.

`404` for an unknown benchmark; an empty `data` for a known one with no results.

## `GET /v1/leaderboard?benchmarks=a,b&category=…`

Cross-benchmark overview. Same filters.

```json
{"benchmarks": ["ocrbench", "nepalipixel"],
 "data": [{"rank": 1, "model": {…}, "avg_score_100": 84.1, "avg_position": 1.5,
           "covered": 2, "total": 2, "complete": true,
           "benchmarks": {"ocrbench": {"score": 806, "score_100": 80.6, "rank": 1, "position": 1}, …}}]}
```

Models with every selected benchmark rank first, then by coverage — a model is never
ranked above another on less evidence. Only bounded benchmarks take part: a rating cannot
be averaged with an accuracy.

## Prices

`meta/models.json` in the store, rewritten by `himeval meta refresh`: per model id, `input_price`/`output_price` ($ per 1M tokens),
`context_length`, and the `source` they came from — an OpenRouter model
(`openrouter:<id>`, from https://openrouter.ai/api/v1/models) or `catalog`
(`catalog/model_meta.yaml`). Names are matched only across reasoning-effort suffixes
(`-high`, `-max`, `-thinking` …) and snapshot dates (`-20250514`); a preview, a `-chat`
variant or a release tag (`-2506`) is a different model and stays unpriced unless an
override names it. A wrong price is worse than N/A.

## `GET /v1/runs?model=&benchmark=&source=&limit=&offset=`

Run summaries, newest first: `{"total", "data": [{"run_id", "model", "benchmark", "source", "score", "cases", "errors", "created_at"}]}`.

## `GET /v1/runs/{run_id}`

The full `RunResult` (`schema.py`): metrics, breakdowns
(`{"level": {"page": {"n": 120, "cer": 0.2, …}}}`), config (prompt hash, dataset
revision, model parameters, subset seed), stats (latency p50/p95, token usage,
truncations, top errors).

## `GET /v1/runs/{run_id}/samples`

Per-sample drill-down for measured runs: `sort=<metric>&order=asc|desc`, `ok=true|false`,
`group=<meta key>&value=<v>`, `limit` ≤ 200, `offset`.

```json
{"total": 2000, "data": [{"sample_id", "ok", "scores": {…}, "output", "error",
                          "references", "question", "meta": {…}, "latency_s", "finish_reason"}]}
```

`404` for imported runs (no per-sample data).

## `POST /v1/admin/refresh`

Re-list the store now instead of waiting for the interval.

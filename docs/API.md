# Results API — `/v1`

Read-only view of the result store. Internal: the studio backend calls it over the
compose network and the browser never does. If `NEPEVAL_API_TOKEN` is set, every `/v1`
route requires `Authorization: Bearer <token>`; `/health` stays open for probes.

Responses carry `ETag` and `Cache-Control: private, max-age=30`; send `If-None-Match` to
get `304`. The ETag changes only when the store's contents change.

| Env | |
|---|---|
| `NEPEVAL_STORE` | required — a path or `s3://bucket/prefix` (`AWS_*` credentials, `AWS_ENDPOINT_URL` for S3-compatible) |
| `NEPEVAL_REFRESH_SECONDS` | store re-list interval, default 60; new results appear within it |
| `NEPEVAL_API_TOKEN` | optional shared secret |

## `GET /health`

`200 {"ok": true, "version", "schema_version", "store", "runs", "invalid", "last_refresh", "last_error"}`;
`503` while starting or when the last store listing failed. `invalid` counts result files
that failed validation — they are skipped, never served.

## `GET /v1/benchmarks?category=`

Every catalog benchmark plus any only known from results.

```json
{"data": [{"id": "nepalipixel", "display_name": "NepaliPixel OCR", "category": "ocr",
           "description": "…", "url": "…", "language": "ne", "primary_metric": "char_accuracy",
           "higher_is_better": true, "scale_max": 1.0, "full_cases": 15000, "version": "1",
           "engine": "native", "models": 7, "updated_at": "2026-10-06T15:40:47Z"}]}
```

`category` ∈ `ocr · document · chart · math · chat · general · hallucination`.
`models: 0` means listed but not yet run — the UI can hide those.

## `GET /v1/models`

Models with at least one result: `{"data": [{…ModelInfo, "benchmarks": ["mmstar", …]}]}`.
ModelInfo = `id, display_name, org, kind (vlm|ocr_engine), open_weights, params_b, license, url, endpoint`.

## `GET /v1/leaderboard/{benchmark_id}`

Query filters: `kind`, `open_weights`, `org` (comma list), `include_imported` (default
true), `include_subsets` (default true).

```json
{"benchmark": {…BenchmarkInfo},
 "data": [{"rank": 1, "position": 1, "model": {…ModelInfo},
           "score": 0.912, "ci_low": 0.905, "ci_high": 0.919, "score_100": 91.2,
           "metrics": {"char_accuracy": 0.912, "cer": 0.101, "wer": 0.21, …},
           "cases": 2000, "errors": 3, "partial": true,
           "source": {"kind": "measured", "harness": "nepeval-ocr", "harness_version": "0.2.0",
                      "judge": null, "url": null, "notes": null},
           "run_id": "…", "created_at": "…", "has_samples": true}]}
```

- `rank` — LMArena-style: 1 + models statistically better (non-overlapping 95% CI).
  Ties are expected; render "1", "1", "3".
- `position` — plain ordinal by point score.
- `score` is on the benchmark's scale (`scale_max`, direction `higher_is_better`);
  `score_100` maps it to 0–100 higher-is-better for cross-suite display.
- `partial` — the run used a subset (`cases < full_cases`). Show it.
- `source.kind` — `measured` here or `imported` from another leaderboard (`source.url`).
- `source.judge` — LLM judge model when scoring used one.

`404` for an unknown benchmark; an empty `data` for a known one with no results.

## `GET /v1/leaderboard?benchmarks=a,b&category=…`

Cross-benchmark overview (Arena's default view). Same filters.

```json
{"benchmarks": ["ocrbench", "nepalipixel"],
 "data": [{"rank": 1, "model": {…}, "avg_score_100": 84.1, "avg_position": 1.5,
           "covered": 2, "total": 2, "complete": true,
           "benchmarks": {"ocrbench": {"score": 806, "score_100": 80.6, "rank": 1, "position": 1}, …}}]}
```

Models with every selected benchmark rank first, then by coverage — a model is never
ranked above another on less evidence.

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

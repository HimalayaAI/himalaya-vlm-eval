# How HimalayaAI Studio reads these results

The studio's **Models › Benchmarks** page is an Arena-style leaderboard over this repo's
results API. The studio stores no benchmark data; it proxies `/v1`. The studio's own docs
are authoritative for its side:

| Studio file | What it holds |
|---|---|
| `docs/contracts/BENCHMARKS.md` | the wire contract the studio validates against (fields it draws, fixed sets) |
| `MEMORIES.md` (2026-10-07) | the decisions: proxy, no table; third-party models on this board only; the vocabulary; chart colours; prices from OpenRouter; rank + spread |
| `docs/deployment/studio.md` § Benchmarks | bucket, IAM, compose profile, env |
| `docs/TESTING.md` § Benchmarks | the manual test matrix |

## The pieces

```
himeval run / import-arena ──publish──▶ s3://himalaya-vlm-eval-results ◀──read── himalaya-vlm-eval-api
                                         runs/<id>/result.json                  (/v1, compose profile
                                         meta/models.json (prices)              "benchmarks", no port)
                                                                                      ▲
                     browser ──▶ studio frontend ──▶ studio backend ─────────────────┘
                                 /models/benchmarks   GET /benchmarks/boards · /benchmarks/{id}
```

| Studio route | This API | Notes |
|---|---|---|
| `GET /benchmarks/boards` | `GET /v1/boards` | the layout from `catalog/boards.yaml` |
| `GET /benchmarks/{id}?…` | `GET /v1/leaderboard/{id}?…` | filters forwarded under the same names |

## The contract both repos keep

- **Vocabulary.** Types `vision · document` and the categories in `catalog/boards.yaml`
  are a contract enum in the studio (Pydantic `Literal`, TypeScript union). Adding, removing
  or renaming one is a change in both repos, studio first. Until the studio is updated, it
  answers `502` for the whole layout rather than draw a label nobody picked.
- **Fields.** The studio draws `rank`, `rank_worst`, `position`, `model.{id, org, license,
  open_weights}`, `score`, `ci_low`/`ci_high`, `cases` + `benchmark.cases_unit`, `pricing`,
  `context_length`, `pareto`, `created_at`, `source.kind`, `bounds`, `attribution`. Changing
  any of them is a contract revision there too.
- **Every row has its count and date.** The studio refuses a row without `cases` or
  `created_at` at its boundary, and this repo's schema already requires both.
- **Attribution.** Arena boards carry `attribution` / `data_license` (CC BY 4.0), and the
  studio renders the credit under the board, as the licence requires.

## Deploying

`ghcr.io/himalayaai/himalaya-vlm-eval-api` (multi-arch; `latest` and the short sha on
every merge to `main`) runs on the studio host under the compose profile `benchmarks`,
with read-only credentials on a dedicated bucket. The studio backend gets
`HIMEVAL_URL=http://himalaya-vlm-eval-api:8000`. Results are written from elsewhere: GPU
runners (`himeval run`) and this repo's weekly `arena-sync` workflow, each with write
credentials that never reach the studio host.

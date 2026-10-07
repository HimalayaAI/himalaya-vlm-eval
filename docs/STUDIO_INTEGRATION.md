# Wiring the results API into HimalayaAI Studio

A plan for the studio side — nothing here is applied to `himalaya-studio` yet. It follows
the studio's own rules (`CLAUDE.md`, `docs/contracts/`, `MEMORIES.md`): the backend is a thin
proxy, the browser only talks to the frontend, every new value set is a recorded decision.

## Decisions to record in the studio's MEMORIES.md first

1. **Benchmark result source** (closes the open question "JSON in docs/ or rows in
   Postgres?"): neither — the studio proxies `himalaya-vlm-eval-api`. No `benchmark_results` table,
   no migration; `db/CLAUDE.md`'s planned table is dropped.
2. **Third-party models on the Benchmarks board.** The board ranks Himalaya models against
   GPT, Gemini, Qwen, Tesseract …, because a leaderboard is a comparison. This is an explicit
   exception to "ours only" (MODEL_REGISTRY), scoped to `/benchmarks` the same way
   `glm-ocr` is a scoped exception on `/vision/models`. The Catalog stays ours-only.
3. **Category vocabulary**: `ocr · document · chart · math · chat · general · hallucination`
   (himeval's `schema.Category`). A new category is a contract revision in both repos.

## Deploy (`infra/`)

`infra/build-images.sh` — himeval builds its own image in CI
(`ghcr.io/himalayaai/himalaya-vlm-eval-api:<sha>`, multi-arch), so the studio only pins a tag:

```yaml
# infra/deploy/docker-compose.yaml
  himalaya-vlm-eval-api:
    image: ghcr.io/himalayaai/himalaya-vlm-eval-api:${HIMEVAL_TAG:?}
    environment:
      HIMEVAL_STORE: ${HIMEVAL_STORE:?}              # s3://himalaya-vlm-eval-results
      AWS_ACCESS_KEY_ID: ${HIMEVAL_S3_KEY:?}          # read-only on that bucket
      AWS_SECRET_ACCESS_KEY: ${HIMEVAL_S3_SECRET:?}
      AWS_DEFAULT_REGION: ap-south-1
      HIMEVAL_REFRESH_SECONDS: "60"
    expose: ["8000"]                                  # no ports: — internal only
    mem_limit: 256m
    restart: unless-stopped
```

The image has no torch or datasets — sized for the
t4g.small. Add `HIMEVAL_URL=http://himalaya-vlm-eval-api:8000` to the backend's environment.

**Storage**: a separate bucket `himalaya-vlm-eval-results`, not a prefix of the Himkosh bucket —
the studio rule "benchmarks never feed Himkosh" enforced by IAM. Two IAM users: the API
gets `s3:GetObject` + `s3:ListBucket`; runners get `s3:PutObject` + `s3:GetObject` +
`s3:ListBucket` (publishing checks for an existing result before writing).

## Backend (`backend/himalaya_studio/`)

Mirrors the Tarka/agents pattern:

| File | |
|---|---|
| `clients/himeval.py` | `httpx.AsyncClient(base_url=HIMEVAL_URL)`; `benchmarks()`, `leaderboard(id, **filters)`, `overview(**filters)`, `run(id)`, `samples(id, **q)`; maps transport errors to `Upstream` |
| `clients/fake_himeval.py` | fixture boards for `STUDIO_FAKE_HIMEVAL=1` (record a real response with `curl` and check it in) |
| `api/benchmarks.py` | `GET /benchmarks`, `/benchmarks/{id}`, `/benchmarks/overview`, `/benchmarks/runs/{run_id}`, `/benchmarks/runs/{run_id}/samples` — pass-through with the studio's auth; forward `If-None-Match`/`ETag` |
| `api/schemas/benchmarks.py` | Pydantic mirror of the `/v1` shapes the UI uses (contract-tested against a fixture) |
| `core/config.py` | `himeval_url`, `fake_himeval` |
| `tests/test_benchmarks.py` | fake client; 404 and upstream-down paths |

Health: add `himeval` to `GET /health`'s dependency report, non-fatal (the board shows a
stale/unavailable state, the rest of the studio is unaffected).

## Frontend — Models › Benchmarks (`frontend/app/(studio)/models/benchmarks/`)

Replaces the `Unbuilt` stub. Server component fetches; filters are URL params.

- **Tabs**: Overview · then one tab per category (OCR, Document, Chart, Math, Chat,
  General, Hallucination); inside a category, a benchmark switcher.
- **Board table** (Arena-style): Rank · Model (`font-mono` id, org beneath) · Score (on the
  benchmark's scale) · 95% CI (`± (hi−lo)/2`, or "—" for imported) · Cases (with a
  "subset" badge when `partial`) · Source (measured / imported ↗ `source.url`) · Date.
  Shared ranks render as repeated numbers. Himalaya rows get the brand accent.
- **Filters**: kind (VLM / OCR engine), open weights, include imported, include subsets.
- **Overview**: models × selected benchmarks matrix with `score_100`, avg, coverage;
  incomplete rows dimmed below complete ones.
- **Run drawer** (row click): breakdowns (e.g. NepaliPixel by level/font/intensity),
  latency, truncations, judge; "Worst samples" from `/samples?sort=<metric>` showing
  reference vs output in Devanagari with Noto Sans Devanagari (DESIGN.md §9).
- **Stat tiles** (existing design): benchmarks, models, last updated.

## Contract file

Add `docs/contracts/BENCHMARKS.md` in the studio pointing at
[docs/API.md](API.md) in this repo as the source of truth (versioned `/v1`,
`schema_version` in `/health`), plus the studio's own `/benchmarks/*` proxy routes.

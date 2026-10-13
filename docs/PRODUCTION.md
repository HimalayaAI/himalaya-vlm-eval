# Production checklist and push instructions

How himalaya-vlm-eval goes to production, and how code and results get there. Three
things ship, each on its own path:

| What | Where it runs | How it ships |
|---|---|---|
| **Code** (this repo) | GitHub | branch → PR → CI green → merge to `main` |
| **Results API** (`himalaya-vlm-eval-api` image) | the studio host, compose profile `benchmarks` | built and pushed to GHCR by CI on every merge to `main` |
| **Results** (`runs/<id>/result.json`, `meta/models.json`) | the S3 store | published by `himeval run` and `himeval meta refresh`, from a runner |

The board shows only what a runner measured, for the models in the catalog (picked
2026-10-13; README § What the board shows). There is no import step and no scheduled
workflow: a result exists because we ran it.

Nothing is ever built or published by hand on the studio host.

---

## 1. One-time setup

Do these once, in order. Tick each before moving on.

### 1.1 Result store (AWS, operator)

- [ ] **Bucket** `himalaya-vlm-eval-results` in `ap-south-1`, private, versioning on (results
      are immutable by convention; versioning makes an accidental overwrite recoverable).
- [ ] **Separate from Himkosh.** Never a prefix of the studio's Himkosh bucket: benchmarks
      never feed collection (studio MEMORIES.md 2026-09-11), and a separate bucket lets IAM
      enforce it.
- [ ] **Reader IAM user** (the API on the studio host): `s3:GetObject` + `s3:ListBucket` on
      that bucket only.
- [ ] **Writer IAM user** (runners): `s3:GetObject` + `s3:PutObject`
      + `s3:ListBucket` on that bucket only. Publishing reads before it writes (it refuses to
      overwrite a result with different content), hence `GetObject`.
- [ ] No `s3:DeleteObject` for either. Removing a result is an operator action.

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::himalaya-vlm-eval-results"},
  {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "arn:aws:s3:::himalaya-vlm-eval-results/*"}
]}
```
(writer: add `"s3:PutObject"` to the second statement)

An S3-compatible provider (Tarka Object Storage, MinIO, R2) works too: set
`AWS_ENDPOINT_URL` wherever `HIMEVAL_STORE` is set.

### 1.2 GHCR image

- [ ] Merges to `main` push `ghcr.io/himalayaai/himalaya-vlm-eval-api:<short-sha>` and
      `:latest` (`.github/workflows/images.yml`, linux/amd64 + linux/arm64). Confirm the
      package exists under the org's **Packages** and the studio host's GHCR token can read it
      (packages are private; the host already pulls the studio's private images the same way).
- [ ] Delete the stale `nepeval-api` package (pre-rename; nothing pulls it).

### 1.3 Run the comparison matrix

From a runner with the writer credentials and the model keys. The hosted models go first;
the self-hosted ones need a GPU (each fits on one 24 GB card) and one vLLM server at a time.

```bash
uv sync --extra run --extra s3
export HIMEVAL_STORE=s3://himalaya-vlm-eval-results AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_DEFAULT_REGION=ap-south-1
export TARKA_API_KEY=… OPENROUTER_API_KEY=…

# ours, the glm-ocr control, and the comparison models OpenRouter serves
uv run himeval run --bench all --model \
  glm-ocr-nepali,glm-ocr,himalaya-gemma-4-bf16,himalaya-gemma-4-q8,himalaya-gemma-4-q4,ministral-3-3b,gemma-3-4b-it

# the comparison models we serve ourselves
for pair in google/gemma-4-E2B-it=gemma-4-e2b-it Qwen/Qwen3.5-2B=qwen3.5-2b Qwen/Qwen3.5-4B=qwen3.5-4b; do
  vllm serve "${pair%%=*}" --port 8001 & pid=$!
  until curl -sf localhost:8001/v1/models >/dev/null; do sleep 5; done
  VLLM_BASE_URL=http://localhost:8001/v1 uv run himeval run --bench all --model "${pair##*=}"
  kill $pid; wait $pid
done
uv run himeval meta refresh          # prices for the models that have one
```

`--bench all` includes the VLMEvalKit suites, which need a VLMEvalKit checkout and a judge
(`docs/VLMEVALKIT.md`). `--limit N` takes a seeded subset for a first pass; the board marks
those rows partial.

- [ ] Every catalog model has a result on `nepalipixel` (`himeval leaderboard nepalipixel`).

### 1.4 Studio host

The studio side is documented in the studio repo, `docs/deployment/studio.md` § Benchmarks.
In short, in the host's `/opt/himalaya-studio/.env`:

- [ ] `HIMEVAL_TAG=latest` (or a pinned short sha)
- [ ] `HIMEVAL_STORE=s3://himalaya-vlm-eval-results`
- [ ] `HIMEVAL_S3_ACCESS_KEY_ID` / `HIMEVAL_S3_SECRET_ACCESS_KEY` = the **reader** user
- [ ] `HIMEVAL_URL=http://himalaya-vlm-eval-api:8000`
- [ ] `HIMEVAL_API_TOKEN` (optional): the same value on both services; the backend sends it,
      the API checks it. See the caution in §5.
- [ ] Deploy with the profile: `PROFILE=benchmarks` (or `voice,benchmarks`) on
      `infra/deploy.sh`.

---

## 2. Pushing code (every change)

1. **Branch from `main`.** Never commit to `main` directly.
   ```bash
   git switch main && git pull --ff-only
   git switch -c feat/<short-name>          # or fix/…, docs/…
   ```
2. **Check locally** — the same steps CI runs:
   ```bash
   uv sync --locked --extra run --extra api --extra s3 --extra dev
   uv run ruff check src tests
   uv run pytest -q
   ```
3. **Commit** in the repo's style (`Feat(): …`, `Fix(): …`, `Docs(): …`, `Refactor(): …`),
   then push and open a PR:
   ```bash
   git push -u origin feat/<short-name>
   gh pr create --base main
   ```
4. **CI must be green**: `test` (ruff + pytest) and `api-image` (builds the API image and
   smoke-tests it against an empty store).
5. **Merge to `main`.** The `Images` workflow pushes the new API image (`:<short-sha>`,
   `:latest`). Watch it: `gh run list --workflow Images --limit 1`.
6. **Roll the API on the studio host** (only if the API changed):
   `HOST=ubuntu@<host> PROFILE=benchmarks ./infra/deploy.sh` from the studio repo. With
   `HIMEVAL_TAG=latest` this pulls the new image; with a pinned sha, edit `HIMEVAL_TAG` on the
   host first.

### Changes that need the studio too

The studio validates this API against its `docs/contracts/BENCHMARKS.md`. These are
**contract changes**: land the studio side first (or in lock-step), never after:

- [ ] Adding, removing or renaming a board type or category in `catalog/boards.yaml`. Until
      the studio knows it, the studio answers `502` for the whole layout.
- [ ] Renaming or removing a field the studio draws (see `docs/STUDIO_INTEGRATION.md`).
- [ ] Bumping `SCHEMA_VERSION` in `schema.py`. The API refuses results of another version.

Safe without the studio: new benchmarks on existing boards, new models, new metrics, new
optional fields, scorer changes (they produce new run ids, old results stay).

---

## 3. Pushing results

Results go to the store, never into git. The API picks them up within
`HIMEVAL_REFRESH_SECONDS` (60), or at once via `POST /v1/admin/refresh`.

### 3.1 Measured runs (a GPU box or any runner)

```bash
docker build -f docker/runner.Dockerfile -t himalaya-vlm-eval-runner --build-arg EXTRAS="easyocr" .
docker run --rm --gpus all \
  -e HIMEVAL_STORE=s3://himalaya-vlm-eval-results -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  -e TARKA_API_KEY -e OPENROUTER_API_KEY \
  -v $PWD/results:/work/results himalaya-vlm-eval-runner \
  himeval run --model glm-ocr-nepali,gpt-4o --bench nepalipixel
```
Or without Docker: `uv sync --extra run --extra s3 [--extra easyocr …]` and the same
`himeval run …`.

- [ ] **Full set for the board.** `--limit N` publishes a *subset* result, which the board
      marks `subset`; once a full run of the same model exists, the board shows that instead.
      Use subsets for smoke tests.
- [ ] **Publish gate.** A run with more than 5% errored samples is evaluated but **not
      published** (`--max-error-rate`). Fix the cause and re-run the same command; it resumes.
- [ ] **Resumable.** Ctrl-C, `docker stop` or a preempted VM keeps every finished answer;
      re-run the same command. Logs: `results/runs/<run>/run.log` and `results/logs/`.
- [ ] **Document boards** (`nepalipixel-docs-*`) need `NEPALIPIXEL_DOCS_DIR` pointing at a
      nepal-pixel-synthesis output made with `--split benchmark`.
- [ ] **VLMEvalKit suites** need `VLMEVALKIT_DIR` / `VLMEVALKIT_PYTHON` and a judge
      (`HIMEVAL_JUDGE_API_KEY`, see `docs/VLMEVALKIT.md`).
- [ ] After publishing models with new names, refresh prices: `himeval meta refresh`.

Publish a run made earlier (or elsewhere) without re-running it:
`himeval publish results/runs/<run-dir> --store s3://himalaya-vlm-eval-results`.

### 3.2 Numbers from another leaderboard

`himeval import file.json` stores them for reference (`--format-help`, `--dry-run` first).
They are never ranked: board routes serve measured runs only.

### 3.3 Prices

`himeval meta refresh` rewrites `meta/models.json` from OpenRouter plus
`catalog/model_meta.yaml`. A model with the wrong or a missing price is fixed in that YAML
(by PR, §2), then `meta refresh` is run again. Never loosen the name matcher instead: a
wrong price is worse than N/A.

---

## 4. Go-live verification

- [ ] `curl https://<studio>/api/health` → `"benchmarks":"configured"`.
- [ ] On the host: `docker compose ps himalaya-vlm-eval-api` → `healthy`; its `/health`
      shows `"ok": true`, `"runs"` > 0, `"invalid": 0`, `"meta_models"` > 0.
- [ ] Studio → Models → Benchmarks opens on Vision → OCR; Vision → Nepali OCR shows the
      measured NepaliPixel results for every catalog model, each with its size on the chart.
- [ ] No row is imported, and no model outside the catalog appears.
- [ ] Licence and one slider each change the board.
- [ ] Both themes, and a phone-width check.

## 5. Operating notes

- **Rollback (API):** set `HIMEVAL_TAG=<previous short sha>` on the host and redeploy with the
  profile. Results are unaffected; the store is the source of truth.
- **A bad result:** results are immutable. Publish a corrected run (it gets a new run id;
  the board shows the newest full measured run per model) or have the operator delete the
  `runs/<id>/` prefix, then `POST /v1/admin/refresh`.
- **Store unreachable:** the API keeps serving its last in-memory copy and reports
  `"ok": false` with `last_error` on `/health`. The studio page shows an error callout with
  retry; the rest of the studio is unaffected.
- **Caution — `HIMEVAL_API_TOKEN` has two meanings.** It is the API's bearer token *and*,
  for runners, a fallback Tarka key when `TARKA_API_KEY` is unset (a holdover from before
  the rename). Never set it on a runner. If a runner ever sends it to Tarka, rotate it.
- **Who holds which credential:** the studio host holds only the reader key and
  `HIMEVAL_API_TOKEN`. Runners hold the writer key and model keys
  (`TARKA_API_KEY`, `OPENROUTER_API_KEY`, judge key). None of them is baked into an image.

"""Read-only results API: `nepeval serve` / `uvicorn nepeval_ocr.api:app`.

Internal service. The studio backend proxies it; browsers never call it directly. If
NEPEVAL_API_TOKEN is set every /v1 route requires `Authorization: Bearer <token>`.

Env:
  NEPEVAL_STORE          path or s3://bucket/prefix (required)
  NEPEVAL_REFRESH_SECONDS  how often to re-list the store (default 60)
  NEPEVAL_API_TOKEN      optional shared secret
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from . import __version__, catalog
from .leaderboard import Filters, benchmark_board, overview, representatives
from .schema import SCHEMA_VERSION, RunResult
from .store import RESULT_KEY, Store, StoreError, open_store, read_samples

log = logging.getLogger("nepeval.api")


class Index:
    """In-memory view of every result.json in the store. result.json is immutable per key,
    so a refresh only downloads keys it has not seen (or whose version changed)."""

    def __init__(self, store: Store):
        self.store = store
        self._results: dict[str, RunResult] = {}
        self._versions: dict[str, str] = {}
        self.invalid: dict[str, str] = {}
        self.generation = 0
        self.last_refresh: float | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._samples: OrderedDict[str, list[dict]] = OrderedDict()

    @property
    def results(self) -> list[RunResult]:
        return list(self._results.values())

    def get(self, run_id: str) -> RunResult | None:
        return self._results.get(run_id)

    def refresh(self) -> None:
        with self._lock:
            try:
                seen: dict[str, str] = {}
                for obj in self.store.list("runs/"):
                    if obj.key.endswith("/" + RESULT_KEY):
                        seen[obj.key] = obj.version
                results = dict(self._results)
                versions = dict(self._versions)
                changed = False
                for key in set(versions) - set(seen):
                    run_id = key.split("/")[1]
                    results.pop(run_id, None)
                    versions.pop(key)
                    self.invalid.pop(key, None)
                    changed = True
                for key, version in seen.items():
                    if versions.get(key) == version:
                        continue
                    versions[key] = version
                    try:
                        r = RunResult.model_validate_json(self.store.get(key))
                        if f"runs/{r.run_id}/{RESULT_KEY}" != key:
                            raise ValueError(f"run_id {r.run_id!r} does not match its key")
                        results[r.run_id] = r
                        self.invalid.pop(key, None)
                    except Exception as exc:  # one bad file must not take the board down
                        self.invalid[key] = str(exc)[:500]
                        log.warning("invalid result %s: %s", key, exc)
                    changed = True
                self._results, self._versions = results, versions
                if changed:
                    self.generation += 1
                    self._samples.clear()
                self.last_refresh = time.time()
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("store refresh failed")

    def samples(self, run_id: str) -> list[dict]:
        if run_id in self._samples:
            self._samples.move_to_end(run_id)
            return self._samples[run_id]
        rows = read_samples(self.store, run_id)
        self._samples[run_id] = rows
        while len(self._samples) > 16:
            self._samples.popitem(last=False)
        return rows

    def etag(self, *parts: Any) -> str:
        raw = f"{self.generation}:{':'.join(map(str, parts))}"
        return '"' + hashlib.sha1(raw.encode()).hexdigest()[:16] + '"'


def create_app(store: Store | None = None, refresh_seconds: float | None = None,
               token: str | None = None) -> FastAPI:
    refresh_every = refresh_seconds if refresh_seconds is not None else float(
        os.environ.get("NEPEVAL_REFRESH_SECONDS", "60"))
    api_token = token if token is not None else os.environ.get("NEPEVAL_API_TOKEN") or None
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        index = Index(store or open_store())
        index.refresh()
        state["index"] = index
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(refresh_every):
                index.refresh()

        thread = None
        if refresh_every > 0:
            thread = threading.Thread(target=loop, name="store-refresh", daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop.set()

    app = FastAPI(title="nepeval results API", version=__version__, lifespan=lifespan)

    def index() -> Index:
        return state["index"]

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if api_token is None:
            return
        expected = f"Bearer {api_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(401, "missing or invalid bearer token")

    def cached(request: Request, response: Response, payload: Any, *etag_parts: Any) -> Any:
        tag = index().etag(request.url.path, request.url.query, *etag_parts)
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers={"ETag": tag})
        response.headers["ETag"] = tag
        response.headers["Cache-Control"] = "private, max-age=30"
        return payload

    def filters(kind: str | None, open_weights: bool | None, org: str | None,
                include_imported: bool, include_subsets: bool, models: str | None = None
                ) -> Filters:
        return Filters(
            kind=kind, open_weights=open_weights,
            orgs={o.strip() for o in org.split(",")} if org else None,
            models={m.strip() for m in models.split(",")} if models else None,
            include_imported=include_imported, include_subsets=include_subsets,
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        idx = state.get("index")
        if idx is None:
            return JSONResponse({"ok": False, "reason": "starting"}, status_code=503)
        ok = idx.last_error is None and idx.last_refresh is not None
        body = {
            "ok": ok,
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "store": idx.store.url,
            "runs": len(idx.results),
            "invalid": len(idx.invalid),
            "last_refresh": idx.last_refresh,
            "last_error": idx.last_error,
        }
        return body if ok else JSONResponse(body, status_code=503)

    v1 = [Depends(auth)]

    @app.get("/v1/benchmarks", dependencies=v1)
    def list_benchmarks(request: Request, response: Response,
                        category: str | None = None) -> Any:
        reps = representatives(index().results)
        counts: dict[str, int] = {}
        latest: dict[str, RunResult] = {}
        for (b, _), r in reps.items():
            counts[b] = counts.get(b, 0) + 1
            if b not in latest or r.created_at > latest[b].created_at:
                latest[b] = r
        items: dict[str, dict] = {}
        for bid, entry in catalog.benchmarks().items():
            items[bid] = {**entry.info.model_dump(mode="json"), "engine": entry.engine}
        for bid, r in latest.items():  # benchmarks only known from results (imports)
            items.setdefault(bid, {**r.benchmark.model_dump(mode="json"), "engine": None})
        data = [
            {**v, "models": counts.get(k, 0),
             "updated_at": latest[k].created_at.isoformat() if k in latest else None}
            for k, v in sorted(items.items())
            if not category or v["category"] == category
        ]
        return cached(request, response, {"data": data})

    @app.get("/v1/models", dependencies=v1)
    def list_models(request: Request, response: Response) -> Any:
        seen: dict[str, dict] = {}
        for (b, mid), r in representatives(index().results).items():
            row = seen.setdefault(mid, {**r.model.model_dump(mode="json"), "benchmarks": []})
            row["benchmarks"].append(b)
        data = sorted(seen.values(), key=lambda m: m["id"])
        for m in data:
            m["benchmarks"].sort()
        return cached(request, response, {"data": data})

    @app.get("/v1/leaderboard", dependencies=v1)
    def leaderboard_overview(
        request: Request, response: Response,
        benchmarks: str | None = None, category: str | None = None,
        kind: str | None = None, open_weights: bool | None = None, org: str | None = None,
        include_imported: bool = True, include_subsets: bool = True,
    ) -> Any:
        ids = [b.strip() for b in benchmarks.split(",")] if benchmarks else None
        results = index().results
        if category:
            in_cat = {r.benchmark.id for r in results if r.benchmark.category == category}
            ids = [b for b in (ids or sorted(in_cat)) if b in in_cat]
            if not ids:
                return cached(request, response, {"benchmarks": [], "data": []})
        f = filters(kind, open_weights, org, include_imported, include_subsets)
        defs = {k: e.info for k, e in catalog.benchmarks().items()}
        selected, rows = overview(results, ids, f, defs)
        return cached(request, response,
                      {"benchmarks": selected, "data": [r.to_dict() for r in rows]})

    @app.get("/v1/leaderboard/{benchmark_id}", dependencies=v1)
    def leaderboard(
        benchmark_id: str, request: Request, response: Response,
        kind: str | None = None, open_weights: bool | None = None, org: str | None = None,
        include_imported: bool = True, include_subsets: bool = True,
    ) -> Any:
        f = filters(kind, open_weights, org, include_imported, include_subsets)
        entry = catalog.benchmarks().get(benchmark_id)
        rows = benchmark_board(index().results, benchmark_id, f,
                               entry.info if entry else None)
        if rows:
            info = rows[0].result.benchmark.model_dump(mode="json")
        elif entry is not None:
            info = entry.info.model_dump(mode="json")
        else:
            raise HTTPException(404, f"unknown benchmark {benchmark_id!r}")
        return cached(request, response,
                      {"benchmark": info, "data": [r.to_dict() for r in rows]})

    @app.get("/v1/runs", dependencies=v1)
    def list_runs(
        request: Request, response: Response,
        model: str | None = None, benchmark: str | None = None,
        source: Literal["measured", "imported"] | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Any:
        rs = [r for r in index().results
              if (not model or r.model.id == model)
              and (not benchmark or r.benchmark.id == benchmark)
              and (not source or r.source.kind == source)]
        rs.sort(key=lambda r: r.created_at, reverse=True)
        data = [{"run_id": r.run_id, "model": r.model.id, "benchmark": r.benchmark.id,
                 "source": r.source.kind, "score": r.primary.value, "cases": r.cases,
                 "errors": r.errors, "created_at": r.created_at.isoformat()}
                for r in rs[offset:offset + limit]]
        return cached(request, response, {"total": len(rs), "data": data})

    @app.get("/v1/runs/{run_id}", dependencies=v1)
    def get_run(run_id: str, request: Request, response: Response) -> Any:
        r = index().get(run_id)
        if r is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return cached(request, response, r.model_dump(mode="json"))

    @app.get("/v1/runs/{run_id}/samples", dependencies=v1)
    def get_samples(
        run_id: str, request: Request, response: Response,
        sort: str | None = None, order: Literal["asc", "desc"] = "desc",
        ok: bool | None = None, group: str | None = None, value: str | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Any:
        r = index().get(run_id)
        if r is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        if not r.has_samples:
            raise HTTPException(404, f"run {run_id!r} has no per-sample data (imported)")
        try:
            rows = index().samples(run_id)
        except (KeyError, StoreError) as exc:
            raise HTTPException(404, f"samples for {run_id!r} unavailable: {exc}") from exc
        if ok is not None:
            rows = [s for s in rows if s["ok"] is ok]
        if group and value is not None:
            rows = [s for s in rows if _matches((s.get("meta") or {}).get(group), value)]
        if sort:
            if sort not in r.metrics:
                raise HTTPException(422, f"sort must be one of {sorted(r.metrics)}")
            rows = sorted(rows, key=lambda s: s["scores"].get(sort, 0.0),
                          reverse=order == "desc")
        return cached(request, response,
                      {"total": len(rows), "data": rows[offset:offset + limit]})

    @app.post("/v1/admin/refresh", dependencies=v1)
    def refresh() -> dict[str, Any]:
        index().refresh()
        return {"runs": len(index().results), "invalid": index().invalid,
                "error": index().last_error}

    return app


def _matches(meta_value: Any, wanted: str) -> bool:
    if isinstance(meta_value, (list, tuple)):
        return wanted in [str(v) for v in meta_value] or (wanted == "none" and not meta_value)
    return str(meta_value) == wanted


def __getattr__(name: str) -> Any:  # `uvicorn nepeval_ocr.api:app` builds from env lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)

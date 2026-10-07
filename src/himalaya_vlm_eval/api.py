"""Read-only results API: `himeval serve` / `uvicorn himalaya_vlm_eval.api:app`.

Internal service. The studio backend proxies it; browsers never call it directly. If
HIMEVAL_API_TOKEN is set every /v1 route requires `Authorization: Bearer <token>`.

Env:
  HIMEVAL_STORE          path or s3://bucket/prefix (required)
  HIMEVAL_REFRESH_SECONDS  how often to re-list the store (default 60)
  HIMEVAL_API_TOKEN      optional shared secret
"""

from __future__ import annotations

import hashlib
import json
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
from .leaderboard import Filters, Range, benchmark_board, blended_price, overview, representatives
from .meta import META_KEY
from .schema import SCHEMA_VERSION, RunResult
from .store import RESULT_KEY, Store, StoreError, open_store, read_samples

log = logging.getLogger("himeval.api")


class Index:
    """In-memory view of every result.json in the store. result.json is immutable per key,
    so a refresh only downloads keys it has not seen (or whose version changed)."""

    def __init__(self, store: Store):
        self.store = store
        self._results: dict[str, RunResult] = {}
        self._versions: dict[str, str] = {}
        self.invalid: dict[str, str] = {}
        # meta/models.json: prices and context lengths, rewritten by `himeval meta refresh`
        self.meta: dict[str, dict[str, Any]] = {}
        self.meta_generated_at: str | None = None
        self._meta_version: str | None = None
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
                changed = self._refresh_meta() or changed
                if changed:
                    self.generation += 1
                    self._samples.clear()
                self.last_refresh = time.time()
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("store refresh failed")

    def _refresh_meta(self) -> bool:
        version = next((o.version for o in self.store.list("meta/") if o.key == META_KEY), None)
        if version == self._meta_version:
            return False
        self._meta_version = version
        if version is None:
            self.meta, self.meta_generated_at = {}, None
            return True
        try:
            doc = json.loads(self.store.get(META_KEY))
            self.meta = {str(k): dict(v) for k, v in (doc.get("models") or {}).items()}
            self.meta_generated_at = doc.get("generated_at")
            self.invalid.pop(META_KEY, None)
        except Exception as exc:  # keep serving the last good prices
            self.invalid[META_KEY] = str(exc)[:500]
            log.warning("invalid %s: %s", META_KEY, exc)
        return True

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
        os.environ.get("HIMEVAL_REFRESH_SECONDS", "60"))
    api_token = token if token is not None else os.environ.get("HIMEVAL_API_TOKEN") or None
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

    app = FastAPI(title="himeval results API", version=__version__, lifespan=lifespan)

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
                include_imported: bool, include_subsets: bool, models: str | None = None,
                ranges: dict[str, Range] | None = None) -> Filters:
        r = ranges or {}
        return Filters(
            kind=kind, open_weights=open_weights,
            orgs={o.strip() for o in org.split(",")} if org else None,
            models={m.strip() for m in models.split(",")} if models else None,
            include_imported=include_imported, include_subsets=include_subsets,
            score=r.get("score"), input_price=r.get("input_price"),
            output_price=r.get("output_price"), context_length=r.get("context_length"),
            meta=index().meta,
        )

    def with_meta(model: dict[str, Any]) -> dict[str, Any]:
        info = index().meta.get(model["id"], {})
        priced = blended_price(info) is not None
        return {**model,
                "pricing": ({"input": info["input_price"], "output": info["output_price"],
                             "blended": blended_price(info), "source": info.get("source")}
                            if priced else None),
                "context_length": info.get("context_length")}

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
            "meta_models": len(idx.meta),
            "meta_generated_at": idx.meta_generated_at,
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
        data = sorted((with_meta(m) for m in seen.values()), key=lambda m: m["id"])
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

    @app.get("/v1/boards", dependencies=v1)
    def boards(request: Request, response: Response) -> Any:
        """The leaderboard layout: top-bar types → side-list categories → the boards in
        each (a category with several shows a switcher; Arena boards pair by style
        control). Every board the catalog defines is listed, with or without results."""
        reps = representatives(index().results)
        counts: dict[str, int] = {}
        latest: dict[str, str] = {}
        for (b, _), r in reps.items():
            counts[b] = counts.get(b, 0) + 1
            when = r.created_at.isoformat()
            latest[b] = max(latest.get(b, when), when)
        by_board: dict[str, list[dict]] = {}
        entries = sorted(catalog.benchmarks().values(), key=lambda e: e.engine != "arena")
        for e in entries:
            i = e.info
            by_board.setdefault(i.board or "", []).append({
                "benchmark": i.id, "display_name": i.display_name, "engine": e.engine,
                "style_control": i.style_control, "primary_metric": i.primary_metric,
                "higher_is_better": i.higher_is_better, "scale_max": i.scale_max,
                "cases_unit": i.cases_unit, "models": counts.get(i.id, 0),
                "updated_at": latest.get(i.id),
            })
        types = [{
            "id": t.id, "label": t.label,
            "categories": [{"id": c.id, "label": c.label,
                            "boards": by_board.get(f"{t.id}/{c.id}", [])}
                           for c in t.categories],
        } for t in catalog.layout().types]
        return cached(request, response, {"types": types})

    @app.get("/v1/leaderboard/{benchmark_id}", dependencies=v1)
    def leaderboard(
        benchmark_id: str, request: Request, response: Response,
        kind: str | None = None, open_weights: bool | None = None, org: str | None = None,
        include_imported: bool = True, include_subsets: bool = True,
        score_min: float | None = None, score_max: float | None = None,
        input_price_min: float | None = None, input_price_max: float | None = None,
        output_price_min: float | None = None, output_price_max: float | None = None,
        context_min: float | None = None, context_max: float | None = None,
    ) -> Any:
        ranges: dict[str, Range] = {
            "score": (score_min, score_max),
            "input_price": (input_price_min, input_price_max),
            "output_price": (output_price_min, output_price_max),
            "context_length": (context_min, context_max),
        }
        f = filters(kind, open_weights, org, include_imported, include_subsets, None, ranges)
        entry = catalog.benchmarks().get(benchmark_id)
        definition = entry.info if entry else None
        results = index().results
        rows = benchmark_board(results, benchmark_id, f, definition)
        if rows:
            info = rows[0].result.benchmark.model_dump(mode="json")
        elif entry is not None:
            info = entry.info.model_dump(mode="json")
        else:
            raise HTTPException(404, f"unknown benchmark {benchmark_id!r}")
        # Slider extents: the whole board under the non-range filters, so moving a slider
        # never moves its own ends.
        whole = benchmark_board(results, benchmark_id, filters(
            kind, open_weights, org, include_imported, include_subsets), definition)
        source = rows[0].result.source if rows else (whole[0].result.source if whole else None)
        body = {
            "benchmark": info,
            "attribution": source.attribution if source else None,
            "data_license": source.data_license if source else None,
            "bounds": _bounds([r.to_dict() for r in whole]),
            "meta_generated_at": index().meta_generated_at,
            "data": [r.to_dict() for r in rows],
        }
        return cached(request, response, body)

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


def _bounds(rows: list[dict[str, Any]]) -> dict[str, list[float] | None]:
    def span(values: list[float]) -> list[float] | None:
        return [min(values), max(values)] if values else None

    priced = [r["pricing"] for r in rows if r["pricing"]]
    return {
        "score": span([r["score"] for r in rows]),
        "input_price": span([p["input"] for p in priced]),
        "output_price": span([p["output"] for p in priced]),
        "context_length": span([r["context_length"] for r in rows if r["context_length"]]),
    }


def _matches(meta_value: Any, wanted: str) -> bool:
    if isinstance(meta_value, (list, tuple)):
        return wanted in [str(v) for v in meta_value] or (wanted == "none" and not meta_value)
    return str(meta_value) == wanted


def __getattr__(name: str) -> Any:  # `uvicorn himalaya_vlm_eval.api:app` builds from env lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)

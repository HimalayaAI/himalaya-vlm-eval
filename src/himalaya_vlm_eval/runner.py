"""Two-stage native runner.

Stage 1, `infer`: run the model over the benchmark, appending one JSON line per sample to
`predictions.jsonl`. Re-running the same (model, params, benchmark, subset) resumes: the
run directory is keyed by a hash of exactly those, successes are kept and errors retried.

Stage 2, `evaluate`: score predictions.jsonl into `scores.jsonl` and `result.json`
(a schema.RunResult). It never touches the model or the dataset, so scoring changes can be
re-applied to old predictions for free.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from . import metrics as M
from .benchmarks import NativeBenchmark
from .catalog import BenchmarkEntry, ModelEntry
from .models.base import FatalModelError, Model, ModelError
from .schema import MetricValue, RunResult, SourceInfo, utcnow

log = logging.getLogger("himeval")

PREDICTIONS = "predictions.jsonl"
SCORES = "scores.jsonl"
RUN_META = "run.json"
RESULT = "result.json"


class RunAborted(RuntimeError):
    pass


class Unsupported(RuntimeError):
    """The model cannot do this benchmark's task (e.g. an OCR engine on VQA)."""


@dataclass
class RunOptions:
    limit: int | None = None
    seed: int = 0
    concurrency: int | None = None
    retry_errors: bool = True
    # Abort when this many samples in a row fail: the endpoint is down, not the sample.
    max_consecutive_errors: int = 25
    progress_every_s: float = 15.0


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def run_dir_for(work_dir: Path, model_id: str, bench_id: str, h: str) -> Path:
    return work_dir / "runs" / f"{model_id}__{bench_id}__{h}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line from a killed process; everything before it is good.
                    log.warning("%s:%d: skipping unreadable line", path, n)
    return rows


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    """Last record per sample id wins (a retried error supersedes the failure)."""
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        out[row["sample_id"]] = row
    return out


# --- stage 1 -----------------------------------------------------------------------------


def infer(
    model_entry: ModelEntry,
    bench_entry: BenchmarkEntry,
    work_dir: Path,
    *,
    model_overrides: dict[str, Any] | None = None,
    options: RunOptions | None = None,
) -> Path:
    opts = options or RunOptions()
    bench = NativeBenchmark(bench_entry)
    model: Model = model_entry.build(model_overrides)
    if bench.task not in model.tasks:
        raise Unsupported(
            f"{model_entry.info.id} ({model.kind}) cannot run {bench.info.id} "
            f"(task {bench.task!r}; model supports {sorted(model.tasks)})"
        )

    log.info("loading %s …", bench.info.id)
    loaded = bench.load(opts.limit, opts.seed)
    config = {
        "harness": "himalaya-vlm-eval",
        "model": model.describe(),
        "benchmark": bench.describe(),
        "subset": {"limit": opts.limit, "seed": opts.seed, "cases": len(loaded.samples)},
        "dataset": loaded.provenance,
    }
    h = config_hash(config)
    run_dir = run_dir_for(work_dir, model_entry.info.id, bench.info.id, h)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / RUN_META
    if not meta_path.exists():
        meta = {
            "config_hash": h,
            "config": config,
            "model": model_entry.info.model_dump(mode="json"),
            "benchmark": bench.info.model_dump(mode="json"),
            "started_at": utcnow().isoformat(),
            "environment": {
                "himalaya_vlm_eval": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        _write_json(meta_path, meta)

    pred_path = run_dir / PREDICTIONS
    existing = latest_records(pred_path)
    done = {sid for sid, r in existing.items() if r.get("ok") or not opts.retry_errors}
    todo = [s for s in loaded.samples if s.id not in done]
    # Pre-flight on a sample that has not failed before, so one permanently bad sample
    # (a corrupt image, say) cannot block every resume.
    fresh = next((i for i, s in enumerate(todo) if s.id not in existing), 0)
    if fresh:
        todo.insert(0, todo.pop(fresh))
    log.info(
        "%s × %s: %d cases, %d done, %d to run → %s",
        model_entry.info.id, bench.info.id, len(loaded.samples), len(done), len(todo), run_dir,
    )
    if not todo:
        return run_dir

    workers = max(1, opts.concurrency or model.max_concurrency)
    try:
        model.setup()
        with pred_path.open("a", encoding="utf-8") as out:
            _preflight(model, bench, todo[0], out)
            _run_pool(model, bench, todo[1:], out, workers, opts, total=len(loaded.samples),
                      already=len(done) + 1)
    finally:
        model.close()
    return run_dir


def _infer_one(model: Model, bench: NativeBenchmark, sample: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_id": sample.id,
        "references": sample.references,
        "question": sample.question,
        "meta": sample.meta,
    }
    if getattr(sample, "target", None) is not None:
        record["target"] = sample.target
    started = time.perf_counter()
    try:
        gen = model.generate(sample.load_image(), bench.prompt(sample))
    except FatalModelError:
        raise
    except ModelError as exc:
        return record | {"ok": False, "error": str(exc)[:1000],
                         "latency_s": time.perf_counter() - started}
    except Exception as exc:  # image decode failures, adapter bugs: per-sample, recorded
        return record | {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1000],
                         "latency_s": time.perf_counter() - started}
    return record | {
        "ok": True,
        "output": gen.text,
        "latency_s": round(gen.latency_s, 4),
        "finish_reason": gen.finish_reason,
        "usage": gen.usage,
        "extra": {k: v for k, v in gen.extra.items() if v is not None} or None,
    }


def _write_record(out: Any, record: dict[str, Any]) -> None:
    out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    out.flush()


def _preflight(model: Model, bench: NativeBenchmark, sample: Any, out: Any) -> None:
    """One sample, synchronously, before fanning out: a broken adapter or key fails in
    seconds instead of after thousands of identical errors."""
    record = _infer_one(model, bench, sample)
    if not record["ok"]:
        raise RunAborted(f"pre-flight failed on {sample.id}: {record['error']}")
    _write_record(out, record)
    log.info("pre-flight ok (%.1fs)", record["latency_s"])


def _run_pool(model: Model, bench: NativeBenchmark, samples: list[Any], out: Any,
              workers: int, opts: RunOptions, *, total: int, already: int) -> None:
    started = last_report = time.monotonic()
    completed, errors, consecutive = 0, 0, 0
    pending: set[cf.Future] = set()
    it = iter(samples)
    with cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="infer") as pool:
        try:
            while True:
                # Bounded window: memory stays flat and Ctrl-C stops quickly.
                while len(pending) < workers * 2:
                    sample = next(it, None)
                    if sample is None:
                        break
                    pending.add(pool.submit(_infer_one, model, bench, sample))
                if not pending:
                    break
                finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for fut in finished:
                    record = fut.result()  # FatalModelError propagates
                    _write_record(out, record)
                    completed += 1
                    if record["ok"]:
                        consecutive = 0
                    else:
                        errors += 1
                        consecutive += 1
                        log.debug("error on %s: %s", record["sample_id"], record["error"])
                        if consecutive >= opts.max_consecutive_errors:
                            raise RunAborted(
                                f"{consecutive} consecutive errors; last: {record['error']}"
                            )
                now = time.monotonic()
                if now - last_report >= opts.progress_every_s or not pending:
                    last_report = now
                    rate = completed / max(now - started, 1e-9)
                    left = len(samples) - completed
                    log.info(
                        "  %d/%d  %.2f/s  eta %s  errors %d",
                        already + completed, total, rate,
                        _fmt_eta(left / rate if rate else 0), errors,
                    )
        except BaseException:
            for fut in pending:
                fut.cancel()
            raise


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# --- stage 2 -----------------------------------------------------------------------------


def _harness_version(env: dict) -> str:
    # runs saved before the rename recorded the package as `nepeval_ocr`
    return env.get("himalaya_vlm_eval") or env.get("nepeval_ocr", "unknown")


def evaluate(run_dir: Path, bench_entry: BenchmarkEntry | None = None,
             *, n_resamples: int = 1000) -> RunResult:
    """Score a run directory and write result.json. `bench_entry` defaults to the catalog
    entry with the run's benchmark id, so improved scorers apply to old predictions."""
    from .catalog import resolve_benchmark
    from .schema import BenchmarkInfo, ModelInfo

    meta = json.loads((run_dir / RUN_META).read_text("utf-8"))
    entry = bench_entry or resolve_benchmark(meta["benchmark"]["id"])
    bench = NativeBenchmark(entry)
    records = latest_records(run_dir / PREDICTIONS)
    expected = meta["config"]["subset"]["cases"]
    if not records:
        raise RunAborted(f"{run_dir}: no predictions")
    if len(records) < expected:
        raise RunAborted(
            f"{run_dir}: only {len(records)}/{expected} samples have predictions; "
            "finish the run (re-run the same command) before evaluating"
        )

    # The run id is a digest of what was scored and how: re-evaluating unchanged
    # predictions with unchanged scoring returns the same result (idempotent publish);
    # a scorer change or new predictions yields a new, separately publishable run.
    digest = hashlib.sha256()
    digest.update(json.dumps(bench.describe(), sort_keys=True, default=str).encode())
    for sid in sorted(records):
        digest.update(json.dumps(records[sid], sort_keys=True, default=str).encode())
    run_id = f"{run_dir.name}__{digest.hexdigest()[:10]}".lower()
    existing = run_dir / RESULT
    if existing.exists():
        try:
            prior = load_result(run_dir)
            if prior.run_id == run_id:
                return prior
        except Exception:  # unreadable or old schema: just re-evaluate
            pass

    per_sample: list[dict[str, Any]] = []
    for sid in sorted(records):
        r = records[sid]
        scores = bench.score(r.get("output") if r.get("ok") else None,
                             r.get("references") or [], r.get("target"))
        per_sample.append({"sample_id": sid, "ok": bool(r.get("ok")), "scores": scores,
                           "meta": r.get("meta") or {}})
    with (run_dir / SCORES).open("w", encoding="utf-8") as fh:
        for row in per_sample:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    metric_names = list(dict.fromkeys(k for row in per_sample for k in row["scores"]))
    primary = bench.info.primary_metric
    metric_values: dict[str, MetricValue] = {}
    for name in metric_names:
        # None = not applicable to that sample (e.g. answer accuracy on an unanswerable
        # question); it is left out of the mean rather than counted as zero.
        vals = [row["scores"][name] for row in per_sample
                if row["scores"].get(name) is not None]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        if name == primary:
            lo, hi = M.bootstrap_ci(vals, n_resamples=n_resamples)
            metric_values[name] = MetricValue(value=mean, ci_low=lo, ci_high=hi)
        else:
            metric_values[name] = MetricValue(value=mean)

    errors = sum(1 for row in per_sample if not row["ok"])
    model_info = ModelInfo(**meta["model"])
    result = RunResult(
        run_id=run_id,
        created_at=utcnow(),
        model=model_info,
        benchmark=BenchmarkInfo(**{**meta["benchmark"], **entry.info.model_dump()}),
        source=SourceInfo(kind="measured", harness="himalaya-vlm-eval",
                          harness_version=_harness_version(meta["environment"])),
        cases=len(per_sample),
        errors=errors,
        metrics=metric_values,
        breakdowns=_breakdowns(per_sample, bench.breakdowns),
        config={**meta["config"], "config_hash": meta["config_hash"]},
        stats=_stats(records, meta),
        has_samples=True,
    )
    _write_json(run_dir / RESULT, json.loads(result.model_dump_json()))
    return result


def _group_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value] or ["none"]
    return ["unknown" if value is None or value == "" else str(value)]


def _breakdowns(rows: list[dict[str, Any]], keys: list[str]
                ) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key in keys:
        groups: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
        for row in rows:
            for v in _group_values(row["meta"].get(key)):
                groups[v].append(row["scores"])
        table: dict[str, dict[str, float]] = {}
        for g, scores in sorted(groups.items()):
            cell: dict[str, float] = {"n": float(len(scores))}
            for m in dict.fromkeys(k for x in scores for k in x):
                vals = [x[m] for x in scores if x.get(m) is not None]
                if vals:
                    cell[m] = statistics.fmean(vals)
            table[g] = cell
        out[key] = table
    return out


def _stats(records: dict[str, dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    ok = [r for r in records.values() if r.get("ok")]
    lat = sorted(r["latency_s"] for r in ok if r.get("latency_s") is not None)
    usage = collections.Counter()
    for r in ok:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)) and k.endswith("tokens"):
                usage[k] += v
    finish = collections.Counter(str(r.get("finish_reason")) for r in ok if r.get("finish_reason"))
    errs = collections.Counter(r.get("error", "")[:200] for r in records.values()
                               if not r.get("ok"))

    def pct(p: float) -> float | None:
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 3) if lat else None

    return {
        "latency_s": {"mean": round(statistics.fmean(lat), 3) if lat else None,
                      "p50": pct(0.5), "p95": pct(0.95)},
        "usage": dict(usage),
        "finish_reasons": dict(finish),
        # Truncated outputs score badly for a reason that is not OCR quality; surface it.
        "truncated": finish.get("length", 0),
        "top_errors": [{"error": e, "count": c} for e, c in errs.most_common(5)],
        "started_at": meta.get("started_at"),
        "environment": meta.get("environment"),
    }


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), "utf-8")
    os.replace(tmp, path)


def load_result(run_dir: Path) -> RunResult:
    return RunResult.model_validate_json((run_dir / RESULT).read_text("utf-8"))


def created(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")

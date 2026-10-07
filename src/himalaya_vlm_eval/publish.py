"""Publishing measured runs and importing published numbers from elsewhere.

Measured runs go through a gate: evaluated, complete, error rate under a threshold.
Imported results are validated against the catalog: the benchmark must exist here (so its
metric, direction and scale are known) and the score must sit inside that scale.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import resolve_benchmark, slug
from .runner import RESULT, SCORES, load_result
from .schema import MetricValue, ModelInfo, RunResult, SourceInfo
from .store import Store, StoreError, gzip_samples, publish_result

log = logging.getLogger("himeval")

DEFAULT_MAX_ERROR_RATE = 0.05


class PublishRefused(StoreError):
    pass


def publish_run(run_dir: Path, store: Store, *, max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
                force: bool = False) -> tuple[RunResult, str]:
    if not (run_dir / RESULT).exists():
        raise PublishRefused(f"{run_dir} has no {RESULT}; evaluate it first")
    result = load_result(run_dir)
    rate = result.errors / result.cases if result.cases else 1.0
    if rate > max_error_rate:
        raise PublishRefused(
            f"{result.run_id}: {result.errors}/{result.cases} samples errored ({rate:.1%}) "
            f"> {max_error_rate:.0%}. Re-run to retry the failures, or raise --max-error-rate "
            "if the errors are the model's fault (they are scored as worst-case)."
        )
    samples = gzip_samples(run_dir) if result.has_samples and (run_dir / SCORES).exists() else None
    status = publish_result(store, result, samples, force=force)
    log.info("%s %s → %s", status, result.run_id, store.url)
    return result, status


# --- imports -----------------------------------------------------------------------------

IMPORT_FORMAT = """\
A JSON list (or {"results": [...]}) of:
{
  "model": {"id": "gpt-4o", "display_name": "GPT-4o", "org": "OpenAI",
            "open_weights": false, "params_b": null},      # schema.ModelInfo
  "benchmark": "ocrbench",                                  # a catalog benchmark id
  "score": 80.5,                                            # primary metric, catalog scale
  "metrics": {"other_metric": 1.0},                         # optional extras
  "cases": 1000,                                            # optional, defaults to full_cases
  "date": "2025-09-17",                                     # when it was measured/published
  "source": {"harness": "opencompass-openvlm", "url": "https://…", "notes": "…"}
}"""


def _parse_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y%m%d%H%M%S"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_imported(entry: dict[str, Any],
                   default_source: dict[str, Any] | None = None) -> RunResult:
    bench = resolve_benchmark(entry["benchmark"]).info
    model = ModelInfo(**entry["model"])
    source = SourceInfo(kind="imported", **{**(default_source or {}), **entry.get("source", {})})
    score = float(entry["score"])
    if bench.scale_max is not None and not 0 <= score <= bench.scale_max:
        raise ValueError(
            f"{model.id} on {bench.id}: score {score} outside the benchmark scale "
            f"0–{bench.scale_max} (convert it first)"
        )
    metrics = {bench.primary_metric: MetricValue(value=score)}
    for k, v in (entry.get("metrics") or {}).items():
        if k != bench.primary_metric and v is not None:
            metrics[k] = MetricValue(value=float(v))
    when = _parse_date(entry.get("date") or datetime.now(timezone.utc))
    cases = int(entry.get("cases") or bench.full_cases or 0)
    run_id ="__".join(slug(p) for p in (model.id, bench.id, source.harness, f"{when:%Y%m%d}"))
    return RunResult(
        run_id=run_id,
        created_at=when,
        model=model,
        benchmark=bench,
        source=source,
        cases=cases,
        metrics=metrics,
        config={"imported": True},
    )


def import_file(path: Path, store: Store, *, default_source: dict[str, Any] | None = None,
                force: bool = False, dry_run: bool = False) -> dict[str, int]:
    data = json.loads(path.read_text("utf-8"))
    entries = data["results"] if isinstance(data, dict) else data
    counts = {"published": 0, "unchanged": 0, "failed": 0}
    for i, entry in enumerate(entries):
        try:
            result = build_imported(entry, default_source)
            status = "published" if dry_run else publish_result(store, result, None, force=force)
            counts[status] += 1
        except Exception as exc:
            counts["failed"] += 1
            log.error("entry %d (%s / %s): %s", i, entry.get("model", {}).get("id"),
                      entry.get("benchmark"), exc)
    return counts

"""`himeval` command line.

    himeval list models | benchmarks [--category math]
    himeval run --model glm-ocr-nepali,gpt-4o --bench nepalipixel,category:math [--limit 500]
    himeval eval results/runs/<dir>            re-score saved predictions
    himeval publish results/runs/<dir>         push to HIMEVAL_STORE
    himeval import results.json                publish numbers from another leaderboard
    himeval leaderboard nepalipixel            print a board from the store
    himeval serve                              results API

`run` publishes every finished result to HIMEVAL_STORE unless --no-publish.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__

log = logging.getLogger("himeval")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "datasets", "huggingface_hub", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _model_args(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--model-arg expects key=value, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def _store(url: str | None, required: bool = True):
    from .store import StoreError, open_store

    try:
        return open_store(url)
    except StoreError as exc:
        if required:
            raise SystemExit(str(exc)) from exc
        return None


# --- commands ------------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    from . import catalog

    if args.what == "models":
        rows = [
            {"id": e.info.id, "org": e.info.org, "kind": e.info.kind, "adapter": e.adapter,
             "endpoint": e.info.endpoint or ""}
            for e in catalog.models().values()
        ]
        cols = ["id", "org", "kind", "adapter", "endpoint"]
    else:
        rows = [
            {"id": b.info.id, "category": b.info.category, "engine": b.engine,
             "metric": f"{b.info.primary_metric} ({'↑' if b.info.higher_is_better else '↓'})",
             "name": b.info.display_name}
            for b in catalog.benchmarks().values()
            if not args.category or b.info.category == args.category
        ]
        cols = ["id", "category", "engine", "metric", "name"]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    _table(rows, cols)
    return 0


def _table(rows: list[dict[str, Any]], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c)
              for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def cmd_run(args: argparse.Namespace) -> int:
    from . import catalog
    from .publish import PublishRefused, publish_run
    from .runner import RunAborted, RunOptions, Unsupported, evaluate, infer
    from .types import DataUnavailable

    work = Path(args.work_dir)
    overrides = _model_args(args.model_arg)
    try:
        names = [m.strip() for m in args.model.split(",") if m.strip()]
        model_entries = [catalog.resolve_model(m) for m in names]
        bench_entries = catalog.select_benchmarks(args.bench)
    except KeyError as exc:
        raise SystemExit(str(exc.args[0])) from exc

    store = None
    if not args.no_publish:
        store = _store(args.store, required=False)
        if store is None:
            log.warning("HIMEVAL_STORE not set: results stay local (pass --store or --no-publish)")

    vl_settings = None
    vl_classes: dict[str, str] = {}
    vl_benches = [b for b in bench_entries if b.engine == "vlmevalkit"]
    if vl_benches:
        from .engines.vlmevalkit import VLMEvalError, VLMEvalSettings, resolve_dataset_classes

        try:
            vl_settings = VLMEvalSettings.from_env()
            vl_settings.check()
            vl_classes = resolve_dataset_classes(
                vl_settings, sorted({b.spec["vlmeval"]["dataset"] for b in vl_benches}))
        except VLMEvalError as exc:
            if len(vl_benches) == len(bench_entries):
                raise SystemExit(str(exc)) from exc
            log.error("skipping %d VLMEvalKit benchmarks: %s", len(vl_benches), exc)
            vl_settings = None

    summary: list[dict[str, Any]] = []
    failed = 0
    for model_entry in model_entries:
        for bench in bench_entries:
            row: dict[str, Any] = {"model": model_entry.info.id, "benchmark": bench.info.id}
            try:
                if bench.engine == "native":
                    run_dir = infer(model_entry, bench, work, model_overrides=overrides,
                                    options=RunOptions(limit=args.limit, seed=args.seed,
                                                       concurrency=args.concurrency,
                                                       retry_errors=not args.no_retry_errors))
                    result = evaluate(run_dir, bench)
                else:
                    if vl_settings is None:
                        row["status"] = "skipped (VLMEvalKit unavailable)"
                        summary.append(row)
                        continue
                    if args.limit:
                        log.warning("--limit does not apply to VLMEvalKit benchmarks "
                                    "(they run their full or MINI split)")
                    from .engines.vlmevalkit import run_benchmark
                    from .runner import RESULT, _write_json

                    result = run_benchmark(model_entry, bench, work, vl_settings,
                                           model_overrides=overrides,
                                           dataset_class=vl_classes.get(bench.spec["vlmeval"]["dataset"]))
                    run_dir = work / "runs" / result.run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    _write_json(run_dir / RESULT, json.loads(result.model_dump_json()))
                p = result.primary
                ci = f" [{p.ci_low:.4g}, {p.ci_high:.4g}]" if p.ci_low is not None else ""
                row.update(score=f"{p.value:.4g}{ci}", cases=result.cases, errors=result.errors)
                if store is not None:
                    try:
                        _, status = publish_run(run_dir, store, max_error_rate=args.max_error_rate)
                        row["status"] = status
                    except PublishRefused as exc:
                        row["status"] = "not published"
                        log.error("%s", exc)
                else:
                    row["status"] = "local"
                row["run"] = str(run_dir)
            except Unsupported as exc:
                row["status"] = "skipped"
                log.info("skip: %s", exc)
            except DataUnavailable as exc:
                row["status"] = "skipped (no data)"
                log.warning("skip %s: %s", bench.info.id, exc)
            except (RunAborted, KeyboardInterrupt) as exc:
                failed += 1
                row["status"] = "aborted"
                log.error("%s × %s aborted: %s", model_entry.info.id, bench.info.id, exc)
                if isinstance(exc, KeyboardInterrupt):
                    summary.append(row)
                    break
            except Exception as exc:  # one pair failing never stops the matrix
                failed += 1
                row["status"] = "failed"
                log.error("%s × %s failed: %s: %s", model_entry.info.id, bench.info.id,
                          type(exc).__name__, exc)
                log.debug("traceback", exc_info=True)
            summary.append(row)
    print()
    _table(summary, ["model", "benchmark", "score", "cases", "errors", "status"])
    return 1 if failed else 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .runner import evaluate

    for d in args.run_dirs:
        r = evaluate(Path(d))
        p = r.primary
        print(f"{r.run_id}: {r.benchmark.primary_metric}={p.value:.4f} "
              f"[{p.ci_low:.4f}, {p.ci_high:.4f}] cases={r.cases} errors={r.errors}")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from .publish import PublishRefused, publish_run

    store = _store(args.store)
    bad = 0
    for d in args.run_dirs:
        try:
            r, status = publish_run(Path(d), store, max_error_rate=args.max_error_rate,
                                    force=args.force)
            print(f"{status}: {r.run_id}")
        except PublishRefused as exc:
            bad += 1
            print(f"refused: {exc}", file=sys.stderr)
    return 1 if bad else 0


def cmd_import(args: argparse.Namespace) -> int:
    from .publish import IMPORT_FORMAT, import_file

    if args.format_help:
        print(IMPORT_FORMAT)
        return 0
    if not args.file:
        raise SystemExit("import needs a FILE (see --format-help)")
    store = None if args.dry_run else _store(args.store)
    source = {k: v for k, v in {"harness": args.harness, "url": args.url}.items() if v}
    counts = import_file(Path(args.file), store, default_source=source or None,
                         force=args.force, dry_run=args.dry_run)
    print(json.dumps(counts))
    return 1 if counts["failed"] else 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from .api import Index
    from .leaderboard import Filters, benchmark_board

    index = Index(_store(args.store))
    index.refresh()
    from . import catalog

    entry = catalog.benchmarks().get(args.benchmark)
    rows = benchmark_board(index.results, args.benchmark,
                           Filters(include_imported=not args.measured_only),
                           entry.info if entry else None)
    if not rows:
        print(f"no published results for {args.benchmark}")
        return 1
    table = []
    for r in rows:
        p = r.result.primary
        ci = f"{p.ci_low:.4g}–{p.ci_high:.4g}" if p.ci_low is not None else ""
        table.append({"rank": r.rank, "model": r.result.model.id, "org": r.result.model.org,
                      "score": f"{p.value:.4g}", "95% CI": ci, "cases": r.result.cases,
                      "source": r.result.source.kind,
                      "date": r.result.created_at.strftime("%Y-%m-%d")})
    _table(table, ["rank", "model", "org", "score", "95% CI", "cases", "source", "date"])
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    app = create_app(_store(args.store), refresh_seconds=args.refresh_seconds)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="himeval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"himalaya-vlm-eval {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list catalog models or benchmarks")
    s.add_argument("what", choices=["models", "benchmarks"])
    s.add_argument("--category")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("run", help="run models × benchmarks, evaluate and publish")
    s.add_argument("--model", required=True,
                   help="comma list of catalog ids, or provider:model "
                        "(openrouter:, openai:, tarka:, vllm:)")
    s.add_argument("--bench", required=True,
                   help="comma list of ids, category:<name>, engine:<name> or all")
    s.add_argument("--limit", type=int, help="seeded random subset per native benchmark")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--concurrency", type=int)
    s.add_argument("--model-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="override an adapter parameter (YAML value), repeatable")
    s.add_argument("--work-dir", default=os.environ.get("HIMEVAL_WORK_DIR", "results"))
    s.add_argument("--store", help="result store (default $HIMEVAL_STORE)")
    s.add_argument("--no-publish", action="store_true")
    s.add_argument("--max-error-rate", type=float, default=0.05)
    s.add_argument("--no-retry-errors", action="store_true",
                   help="on resume, keep failed samples instead of retrying them")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("eval", help="re-score run directories from saved predictions")
    s.add_argument("run_dirs", nargs="+")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("publish", help="publish evaluated run directories")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--store")
    s.add_argument("--max-error-rate", type=float, default=0.05)
    s.add_argument("--force", action="store_true", help="overwrite a conflicting result")
    s.set_defaults(func=cmd_publish)

    s = sub.add_parser("import", help="publish results measured elsewhere")
    s.add_argument("file", nargs="?")
    s.add_argument("--store")
    s.add_argument("--harness", help="default source.harness for entries")
    s.add_argument("--url", help="default source.url for entries")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--format-help", action="store_true")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("leaderboard", help="print a benchmark's board from the store")
    s.add_argument("benchmark")
    s.add_argument("--store")
    s.add_argument("--measured-only", action="store_true")
    s.set_defaults(func=cmd_leaderboard)

    s = sub.add_parser("serve", help="run the results API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.add_argument("--store")
    s.add_argument("--refresh-seconds", type=float,
                   default=float(os.environ.get("HIMEVAL_REFRESH_SECONDS", "60")))
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

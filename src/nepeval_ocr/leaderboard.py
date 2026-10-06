"""Leaderboard computation over published RunResults. Pure functions; the API is a thin
layer over these.

Ranking follows LMArena's convention: a model's `rank` is 1 + the number of models that
are *statistically* better — whose confidence interval lies entirely on the better side of
this model's interval. Models whose intervals overlap share a rank, so the board never
claims an ordering the data cannot support. `position` is the plain ordinal by point score.
Imported numbers have no CI; they are compared by point value.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .schema import BenchmarkInfo, ModelInfo, RunResult


@dataclass
class Filters:
    kind: str | None = None  # vlm | ocr_engine
    open_weights: bool | None = None
    orgs: set[str] | None = None
    models: set[str] | None = None
    include_imported: bool = True
    include_subsets: bool = True

    def keep(self, r: RunResult) -> bool:
        m = r.model
        if self.kind and m.kind != self.kind:
            return False
        if self.open_weights is not None and m.open_weights is not self.open_weights:
            return False
        if self.orgs and m.org.lower() not in {o.lower() for o in self.orgs}:
            return False
        if self.models and m.id not in self.models:
            return False
        if not self.include_imported and r.source.kind == "imported":
            return False
        return self.include_subsets or not r.is_subset


def _preference(r: RunResult) -> tuple:
    # Measured beats imported (we know exactly how it was produced); a full run beats a
    # subset; then the newest.
    return (r.source.kind == "measured", not r.is_subset, r.created_at)


def representatives(results: Iterable[RunResult], filters: Filters | None = None
                    ) -> dict[tuple[str, str], RunResult]:
    """One result per (benchmark, model): the one the board shows."""
    f = filters or Filters()
    best: dict[tuple[str, str], RunResult] = {}
    for r in results:
        if not f.keep(r):
            continue
        key = (r.benchmark.id, r.model.id)
        if key not in best or _preference(r) > _preference(best[key]):
            best[key] = r
    return best


@dataclass
class BoardRow:
    rank: int
    position: int
    result: RunResult
    score_100: float
    better_than: int = 0

    def to_dict(self) -> dict:
        r = self.result
        p = r.primary
        return {
            "rank": self.rank,
            "position": self.position,
            "model": r.model.model_dump(mode="json"),
            "score": p.value,
            "ci_low": p.ci_low,
            "ci_high": p.ci_high,
            "score_100": round(self.score_100, 4),
            "metrics": {k: v.value for k, v in r.metrics.items()},
            "cases": r.cases,
            "errors": r.errors,
            "partial": r.is_subset,
            "source": r.source.model_dump(mode="json"),
            "run_id": r.run_id,
            "created_at": r.created_at.isoformat(),
            "has_samples": r.has_samples,
        }


def _bounds(r: RunResult) -> tuple[float, float]:
    p = r.primary
    if p.ci_low is None or p.ci_high is None:
        return p.value, p.value
    return p.ci_low, p.ci_high


def rank_results(results: list[RunResult], bench: BenchmarkInfo) -> list[BoardRow]:
    hib = bench.higher_is_better
    sign = 1 if hib else -1
    ordered = sorted(results, key=lambda r: (-sign * r.primary.value, r.model.id))
    rows: list[BoardRow] = []
    for r in ordered:
        lo, hi = _bounds(r)
        stat_better = 0
        point_better = 0
        for o in ordered:
            if o is r:
                continue
            olo, ohi = _bounds(o)
            if (olo > hi) if hib else (ohi < lo):
                stat_better += 1
            if sign * o.primary.value > sign * r.primary.value:
                point_better += 1
        rows.append(BoardRow(rank=1 + stat_better, position=1 + point_better, result=r,
                             score_100=bench.score_100(r.primary.value)))
    return rows


def under_definition(r: RunResult, ref: BenchmarkInfo) -> RunResult | None:
    """Express a result under the benchmark's current definition, or None if it cannot be.

    A benchmark's headline metric or direction can change (e.g. CER → char accuracy);
    older results that also recorded the new headline metric stay comparable, others are
    left off the board rather than ranked on a different scale. A different `version`
    means prompt, data or scoring changed: never comparable.
    """
    b = r.benchmark
    if b.version != ref.version:
        return None
    if b.primary_metric == ref.primary_metric and b.higher_is_better == ref.higher_is_better:
        return r if b == ref else r.model_copy(update={"benchmark": ref})
    if ref.primary_metric in r.metrics:
        return r.model_copy(update={"benchmark": ref})
    return None


def benchmark_board(results: Iterable[RunResult], benchmark_id: str,
                    filters: Filters | None = None,
                    definition: BenchmarkInfo | None = None) -> list[BoardRow]:
    """Rank one benchmark. `definition` is the benchmark as currently defined (the
    catalog's); without it the newest result's definition is used."""
    mine = [r for r in results if r.benchmark.id == benchmark_id]
    if not mine:
        return []
    ref = definition or max(mine, key=lambda r: r.created_at).benchmark
    comparable = [c for c in (under_definition(r, ref) for r in mine) if c is not None]
    reps = list(representatives(comparable, filters).values())
    return rank_results(reps, ref) if reps else []


@dataclass
class OverviewRow:
    model: ModelInfo
    avg_score_100: float
    avg_position: float
    covered: int
    total: int
    per_benchmark: dict[str, dict] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "model": self.model.model_dump(mode="json"),
            "avg_score_100": round(self.avg_score_100, 4),
            "avg_position": round(self.avg_position, 3),
            "covered": self.covered,
            "total": self.total,
            "complete": self.covered == self.total,
            "benchmarks": self.per_benchmark,
        }


def overview(results: Iterable[RunResult], benchmark_ids: list[str] | None = None,
             filters: Filters | None = None,
             definitions: dict[str, BenchmarkInfo] | None = None
             ) -> tuple[list[str], list[OverviewRow]]:
    """Cross-benchmark view: mean 0–100 score and mean position per model.

    Models with every selected benchmark rank first; models with gaps follow, ordered by
    coverage, so a model is never ranked above another on a subset of the evidence.
    """
    results = list(results)
    defs = definitions or {}
    boards_all = {b: benchmark_board(results, b, filters, defs.get(b))
                  for b in sorted({r.benchmark.id for r in results})}
    present = [b for b, rows in boards_all.items() if rows]
    selected = [b for b in (benchmark_ids or present) if b in present]
    boards = {b: boards_all[b] for b in selected}

    per_model: dict[str, dict] = defaultdict(lambda: {"scores": [], "positions": [], "cells": {}})
    models: dict[str, ModelInfo] = {}
    for b, rows in boards.items():
        for row in rows:
            mid = row.result.model.id
            models.setdefault(mid, row.result.model)
            d = per_model[mid]
            d["scores"].append(row.score_100)
            d["positions"].append(row.position)
            d["cells"][b] = {"score": row.result.primary.value, "score_100": row.score_100,
                             "rank": row.rank, "position": row.position}

    out = [
        OverviewRow(model=models[mid], avg_score_100=statistics.fmean(d["scores"]),
                    avg_position=statistics.fmean(d["positions"]), covered=len(d["scores"]),
                    total=len(selected), per_benchmark=d["cells"])
        for mid, d in per_model.items()
    ]
    out.sort(key=lambda o: (-o.covered, -o.avg_score_100, o.model.id))
    for i, row in enumerate(out, 1):
        row.rank = i
    return selected, out

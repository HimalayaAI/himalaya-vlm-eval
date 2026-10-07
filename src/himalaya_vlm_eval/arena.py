"""Import Arena (arena.ai) leaderboards from the official Hugging Face dataset.

`lmarena-ai/leaderboard-dataset` (CC BY 4.0) is the only source. Arena's Terms of Use
forbid scraping and automated access to arena.ai, so the site is never read; prices and
context lengths, which the dataset lacks, come from `meta.py` instead.

Every row becomes an imported RunResult on one of the `arena-*` benchmarks that
`catalog/arena.yaml` defines: the rating with its 95% CI, the vote count as `cases`, the
leaderboard's publish date as `created_at`, and the dataset revision in `config`. Run ids
are derived from (board, model, date), so re-importing the same snapshot is a no-op and a
new snapshot adds results rather than rewriting old ones.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .catalog import ARENA_METRIC, BenchmarkEntry, benchmarks, slug
from .schema import MetricValue, ModelInfo, RunResult, SourceInfo
from .store import Store, StoreError, publish_result

log = logging.getLogger("himeval")

# The dataset's organisation keys, as Arena displays them.
ORG_NAMES = {
    "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "alibaba": "Alibaba",
    "meta": "Meta", "xai": "xAI", "mistral": "Mistral", "zai": "Z.ai", "moonshot": "Moonshot",
    "stepfun": "StepFun", "xiaomi": "Xiaomi", "tencent": "Tencent", "allenai": "Ai2",
    "minimax": "MiniMax", "amazon": "Amazon", "microsoft": "Microsoft",
    "bytedance": "ByteDance", "baidu": "Baidu", "cohere": "Cohere", "nvidia": "NVIDIA",
    "deepseek": "DeepSeek", "thinky": "Thinking Machines",
}
PROPRIETARY = "Proprietary"


def arena_entries(arenas: Iterable[str] | None = None) -> list[BenchmarkEntry]:
    wanted = set(arenas) if arenas else None
    return [e for e in benchmarks().values()
            if e.engine == "arena" and (wanted is None or e.spec["arena"] in wanted)]


def resolve_revision(dataset: str, revision: str | None) -> str:
    """Pin the import to a commit, so the result records exactly what was read."""
    if revision and len(revision) == 40:
        return revision
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset, revision=revision).sha


def load_subset(dataset: str, subset: str, revision: str, split: str = "latest"
                ) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(dataset, f"{subset}/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=revision)
    return pq.read_table(path).to_pylist()


def model_info(row: dict[str, Any]) -> ModelInfo:
    name = str(row["model_name"]).strip()
    org_key = str(row.get("organization") or "").strip().lower()
    license_ = str(row.get("license") or "").strip()
    known_license = license_ not in ("", "-")
    return ModelInfo(
        id=slug(name),
        display_name=name,
        org=ORG_NAMES.get(org_key, org_key.title() if org_key else "Unknown"),
        open_weights=(license_ != PROPRIETARY) if known_license else None,
        license=license_ if known_license else None,
    )


def _date(value: Any) -> datetime:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


def build_results(entry: BenchmarkEntry, rows: list[dict[str, Any]], revision: str
                  ) -> list[RunResult]:
    """Rows of one subset → results for one board (rows for other categories ignored)."""
    spec = entry.spec
    out = []
    for row in rows:
        if row.get("category") != spec["category_key"]:
            continue
        model = model_info(row)
        when = _date(row["leaderboard_publish_date"])
        rating = float(row["rating"])
        lo, hi = row.get("rating_lower"), row.get("rating_upper")
        ci = (float(lo), float(hi)) if lo is not None and hi is not None else (None, None)
        out.append(RunResult(
            run_id="__".join(["arena", entry.info.id, model.id, f"{when:%Y%m%d}"]),
            created_at=when,
            model=model,
            benchmark=entry.info,
            source=SourceInfo(
                kind="imported", harness="arena", url=entry.info.url,
                notes=f"{spec['dataset']} {spec['subset']}@{revision[:12]}",
                data_license=spec.get("license"), attribution=spec.get("attribution"),
            ),
            cases=int(row.get("vote_count") or 0),
            metrics={ARENA_METRIC: MetricValue(value=rating, ci_low=ci[0], ci_high=ci[1])},
            config={
                "dataset": spec["dataset"], "revision": revision, "subset": spec["subset"],
                "category": spec["category_key"], "arena_rank": row.get("rank"),
                "variance": row.get("variance"),
            },
        ))
    return out


def import_arena(store: Store | None, *, arenas: Iterable[str] | None = None,
                 revision: str | None = None, history: bool = False,
                 dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """Import the latest snapshot (or, with `history`, every published snapshot) of each
    Arena board in the catalog. A category whose newest snapshot is older than its arena's
    newest is retired on arena.ai (e.g. the old `creative_writing`), and is skipped."""
    entries = arena_entries(arenas)
    if not entries:
        raise ValueError(f"no arena boards match {sorted(arenas or [])}")
    dataset = entries[0].spec["dataset"]
    rev = resolve_revision(dataset, revision)
    log.info("arena: %s @ %s, %d boards", dataset, rev[:12], len(entries))
    counts: dict[str, Any] = defaultdict(int)
    counts["revision"] = rev
    rows_by_subset: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        subset = entry.spec["subset"]
        if subset not in rows_by_subset:
            rows_by_subset[subset] = load_subset(dataset, subset, rev,
                                                 "full" if history else "latest")
        rows = rows_by_subset[subset]
        newest = max((str(r["leaderboard_publish_date"]) for r in rows), default="")
        mine = [r for r in rows if r.get("category") == entry.spec["category_key"]]
        if not mine:
            log.warning("arena: %s has no rows for category %r", subset,
                        entry.spec["category_key"])
            counts["empty"] += 1
            continue
        if max(str(r["leaderboard_publish_date"]) for r in mine) < newest:
            log.warning("arena: %s/%s is not in the newest snapshot (%s); skipped as retired",
                        subset, entry.spec["category_key"], newest)
            counts["retired"] += 1
            continue
        for result in build_results(entry, mine, rev):
            if dry_run or store is None:
                counts["published"] += 1
                continue
            try:
                counts[publish_result(store, result, force=force)] += 1
            except StoreError as exc:
                counts["failed"] += 1
                log.error("arena: %s: %s", result.run_id, exc)
        log.info("arena: %s ← %d models", entry.info.id, len(mine))
    return dict(counts)

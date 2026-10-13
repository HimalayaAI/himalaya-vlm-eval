"""NepalPixel VLM Bench (`himalaya-ai/nepalpixel-vlm-bench`): Nepali government documents.

Three configs, each its own benchmark (catalog/benchmarks/nepalpixel_vlm_bench.yaml):

- `kie`        one row per document with a `qa` list. Every question becomes a sample; its
               image is the page or pages its `pages` field names (page 1 is `image`,
               page n is `extra_pages[n - 2]`).
- `ocr_layout` one row per page; transcribe it.
- `abstain`    the field's value is painted out; the right answer is ANSWER NOT PRESENT.

Scoring mirrors the dataset's own `score_vlm_bench.py` (pinned with the dataset revision)
exactly, so a number here is the number that script would print:

- normalisation is NFC plus whitespace collapsing, nothing else — Devanagari and European
  digits are *not* folded, because reading `२०४९` as `2049` is a reading error;
- `kie`: ANLS (τ = 0.5) and exact match per question, averaged over questions;
- `ocr_layout`: CER **micro-averaged over pages** — Σ edits ÷ Σ reference characters — not
  the mean of per-page CERs, so a long page weighs what its length says;
- `abstain`: abstention (the reply, uppercased, is exactly ANSWER NOT PRESENT) and leak (a
  non-empty reply within ANLS range of the erased value);
- a missing or failed prediction scores as an empty reply, as the script treats a missing id.

The scorer leaves layout (region IoU, role accuracy, reading order) to the evaluator, and so
does this module: only what the script defines is measured.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Sequence
from typing import Any

from .types import Sample

NOT_PRESENT = "ANSWER NOT PRESENT"
VIEWS = ("kie", "ocr_layout", "abstain")


# --- scoring: score_vlm_bench.py, function for function ---------------------------------


def norm(s: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gold: str, tau: float = 0.5) -> float:
    p, g = norm(pred), norm(gold)
    if not p and not g:
        return 1.0
    nl = levenshtein(p, g) / max(len(p), len(g))
    return 1.0 - nl if nl < tau else 0.0


def score_kie(pred: str, refs: Sequence[str], target: Any) -> dict[str, float | None]:
    gold = refs[0] if refs else ""
    return {"anls": anls(pred, gold), "exact_match": float(norm(pred) == norm(gold))}


def score_ocr(pred: str, refs: Sequence[str], target: Any) -> dict[str, float | None]:
    """Per page, the two counts the micro-average is built from (`RATIOS`)."""
    gold = norm(refs[0] if refs else "")
    return {"edits": float(levenshtein(norm(pred), gold)), "chars": float(len(gold))}


def score_abstain(pred: str, refs: Sequence[str], target: Any) -> dict[str, float | None]:
    guess = norm(pred)
    erased = (target or {}).get("erased_value", "")
    return {
        "abstention": float(guess.upper() == NOT_PRESENT),
        "leak": float(bool(guess) and anls(guess, erased) > 0),
    }


# --- loading ----------------------------------------------------------------------------


def _subset(n: int, limit: int | None, seed: int) -> list[int]:
    # Same rule as NativeBenchmark._subset: a seeded random draw over rows (documents for
    # `kie`), never the head — the set is ordered by category.
    if not limit or limit >= n:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), limit))


class _Pages:
    """Lazy page loader: the question's page(s) from one document row."""

    __slots__ = ("ds", "pages", "pos")

    def __init__(self, ds: Any, pos: int, pages: Sequence[int]):
        self.ds, self.pos, self.pages = ds, pos, list(pages) or [1]

    def __call__(self) -> Any:
        row = self.ds[self.pos]
        sheets = [row["image"], *(row.get("extra_pages") or [])]
        images = [sheets[p - 1] for p in self.pages if 0 < p <= len(sheets)]
        if not images:
            raise ValueError(f"pages {self.pages} not in a {len(sheets)}-page document")
        return images[0] if len(images) == 1 else images


class _Image:
    __slots__ = ("ds", "pos")

    def __init__(self, ds: Any, pos: int):
        self.ds, self.pos = ds, pos

    def __call__(self) -> Any:
        return self.ds[self.pos]["image"]


def load(spec: dict[str, Any], limit: int | None, seed: int
         ) -> tuple[list[Sample], int, dict[str, Any]]:
    """Samples, the config's row count, and provenance. `limit` counts rows of the config
    (documents for `kie`, whose samples are its questions)."""
    import logging

    from datasets import load_dataset

    for noisy in ("httpx", "httpcore", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    view = spec.get("view")
    if view not in VIEWS:
        raise ValueError(f"dataset.view must be one of {VIEWS}, got {view!r}")
    repo, revision = spec["repo"], spec.get("revision")
    ds = load_dataset(repo, view, split=spec.get("split", "test"), revision=revision)
    total = len(ds)
    indices = _subset(total, limit, seed)
    ds = ds.select(indices)
    light = ds.remove_columns([c for c in ("image", "extra_pages", "words", "regions")
                               if c in ds.column_names])

    samples: list[Sample] = []
    for pos, row in enumerate(light):
        meta = {"category": row["category"], "doc_type": row.get("doc_type")}
        if view == "kie":
            for i, qa in enumerate(row["qa"]):
                samples.append(Sample(
                    id=f"{row['id']}#{i:02d}",
                    references=[qa["answer"]],
                    image=_Pages(ds, pos, qa.get("pages") or [1]),
                    question=qa["question"],
                    meta=meta | {"doc_id": row["id"], "field": qa.get("field"),
                                 "pages": qa.get("pages")},
                ))
        elif view == "ocr_layout":
            samples.append(Sample(
                id=row["id"],
                references=[row["text"]],
                image=_Image(ds, pos),
                meta=meta | {"doc_id": row.get("doc_id") or row["id"],
                             "layout_graded": row.get("layout_graded")},
            ))
        else:
            samples.append(Sample(
                id=row["id"],
                references=[NOT_PRESENT],
                image=_Image(ds, pos),
                question=row["question"],
                meta=meta | {"doc_id": row.get("positive_id") or row["id"],
                             "field": row.get("field")},
                target={"erased_value": row.get("erased_value", "")},
            ))
    return samples, total, {"repo": repo, "revision": revision, "config": view,
                            "split": spec.get("split", "test"), "rows": total}

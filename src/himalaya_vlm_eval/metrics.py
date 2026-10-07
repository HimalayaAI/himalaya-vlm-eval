"""Per-sample metrics and their aggregation.

Every metric here is a function `(prediction, references) -> float` so a benchmark can
name its metrics in config. Aggregation is a mean with a percentile-bootstrap CI.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from . import text as T

try:  # rapidfuzz is C-backed; the fallback keeps the core importable without the run extra
    from rapidfuzz.distance import Levenshtein as _Lev

    def edit_distance(a: Sequence, b: Sequence) -> int:
        return int(_Lev.distance(a, b))

except ImportError:  # pragma: no cover - exercised only without the extra

    def edit_distance(a: Sequence, b: Sequence) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


Metric = Callable[[str, Sequence[str]], float]


def _ref(refs: Sequence[str]) -> str:
    return refs[0] if refs else ""


# --- transcription -----------------------------------------------------------------------


def _best(error: Callable[[str, str], float], pred: str, refs: Sequence[str]) -> float:
    """Lowest error against any reference (a prediction equal to any one of them is perfect)."""
    return min((error(pred, r) for r in refs), default=error(pred, ""))


def _cer(pred: str, ref: str) -> float:
    r, p = T.canonical(ref), T.canonical(pred)
    if not r:
        return 0.0 if not p else 1.0
    return edit_distance(r, p) / len(r)


def _wer(pred: str, ref: str) -> float:
    r, p = T.canonical(ref).split(), T.canonical(pred).split()
    if not r:
        return 0.0 if not p else 1.0
    return edit_distance(r, p) / len(r)


def _ned(pred: str, ref: str) -> float:
    r, p = T.canonical(ref), T.canonical(pred)
    longest = max(len(r), len(p))
    return edit_distance(r, p) / longest if longest else 0.0


def _acer(pred: str, ref: str) -> float:
    r, p = T.aksharas(T.canonical(ref)), T.aksharas(T.canonical(pred))
    if not r:
        return 0.0 if not p else 1.0
    return edit_distance(r, p) / len(r)


def _wer_loose(pred: str, ref: str) -> float:
    r, p = T.loose(ref).split(), T.loose(pred).split()
    if not r:
        return 0.0 if not p else 1.0
    return edit_distance(r, p) / len(r)


def cer(pred: str, refs: Sequence[str]) -> float:
    """Character error rate on canonical text (Unicode code points). Unbounded above
    (insertions count). Best over all references."""
    return _best(_cer, pred, refs)


def wer(pred: str, refs: Sequence[str]) -> float:
    """Word error rate; words split on whitespace only (see `wer_loose`). Best over references."""
    return _best(_wer, pred, refs)


def ned(pred: str, refs: Sequence[str]) -> float:
    """Normalized edit distance in [0, 1] (OmniDocBench-style): distance / max length."""
    return _best(_ned, pred, refs)


def acer(pred: str, refs: Sequence[str]) -> float:
    """Akshara error rate: edit distance over aksharas (what a reader sees as one character),
    zero-width characters ignored. One wrong vowel sign costs one unit, not a fraction of one."""
    return _best(_acer, pred, refs)


def wer_loose(pred: str, refs: Sequence[str]) -> float:
    """WER on the `loose` text: punctuation (danda included) and zero-width characters are
    ignored and case folded, so `ल्यायो।` and `ल्यायो` are the same word."""
    return _best(_wer_loose, pred, refs)


def char_accuracy(pred: str, refs: Sequence[str]) -> float:
    """1 - CER, clipped to [0, 1]: a higher-is-better transcription score."""
    return max(0.0, 1.0 - cer(pred, refs))


def akshara_accuracy(pred: str, refs: Sequence[str]) -> float:
    """1 - ACER, clipped to [0, 1]: the headline for Nepali transcription. Bounded like
    `char_accuracy`, but a wrong vowel sign or broken conjunct costs a whole character."""
    return max(0.0, 1.0 - acer(pred, refs))


def akshara_accuracy_digitfold(pred: str, refs: Sequence[str]) -> float:
    """`akshara_accuracy` with Devanagari digits read as ASCII on both sides (४२ = 42). The gap
    to `akshara_accuracy` is what writing numbers in the other script cost."""
    return akshara_accuracy(T.fold_digits(pred), [T.fold_digits(r) for r in refs])


def exact_match(pred: str, refs: Sequence[str]) -> float:
    p = T.canonical(pred)
    return float(any(p == T.canonical(r) for r in refs))


def loose_match(pred: str, refs: Sequence[str]) -> float:
    p = T.loose(pred)
    return float(any(p == T.loose(r) for r in refs))


def length_ratio(pred: str, refs: Sequence[str]) -> float:
    r = T.canonical(_ref(refs))
    return len(T.canonical(pred)) / len(r) if r else float(bool(pred))


# --- VQA ---------------------------------------------------------------------------------


def anls(pred: str, refs: Sequence[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA / InfographicVQA)."""
    p = " ".join(pred.strip().lower().split())
    best = 0.0
    for ref in refs:
        r = " ".join(ref.strip().lower().split())
        longest = max(len(r), len(p))
        sim = 1.0 - (edit_distance(r, p) / longest if longest else 0.0)
        best = max(best, sim if sim >= threshold else 0.0)
    return best


def relaxed_accuracy(pred: str, refs: Sequence[str], tolerance: float = 0.05) -> float:
    """ChartQA: numeric answers within 5% relative error, otherwise case-insensitive match."""

    def to_float(s: str) -> float | None:
        s = s.strip().rstrip("%").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    p = pred.strip()
    for ref in refs:
        pf, rf = to_float(p), to_float(ref)
        if pf is not None and rf is not None:
            if rf == 0:
                if pf == 0:
                    return 1.0
            elif abs(pf - rf) / abs(rf) <= tolerance:
                return 1.0
        elif p.lower() == ref.strip().lower():
            return 1.0
    return 0.0


def contains_match(pred: str, refs: Sequence[str]) -> float:
    """OCRBench-style: any reference appears in the normalized prediction."""
    p = T.loose(pred)
    return float(any(T.loose(r) and T.loose(r) in p for r in refs))


METRICS: dict[str, Metric] = {
    "cer": cer,
    "wer": wer,
    "acer": acer,
    "wer_loose": wer_loose,
    "ned": ned,
    "char_accuracy": char_accuracy,
    "akshara_accuracy": akshara_accuracy,
    "akshara_accuracy_digitfold": akshara_accuracy_digitfold,
    "exact_match": exact_match,
    "loose_match": loose_match,
    "length_ratio": length_ratio,
    "anls": anls,
    "relaxed_accuracy": relaxed_accuracy,
    "contains_match": contains_match,
}

# Value a sample gets when the model errored: the worst score, so failures are never
# silently dropped from the mean.
WORST: dict[str, float] = {
    "cer": 1.0,
    "wer": 1.0,
    "acer": 1.0,
    "wer_loose": 1.0,
    "ned": 1.0,
    "char_accuracy": 0.0,
    "akshara_accuracy": 0.0,
    "akshara_accuracy_digitfold": 0.0,
    "exact_match": 0.0,
    "loose_match": 0.0,
    "length_ratio": 0.0,
    "anls": 0.0,
    "relaxed_accuracy": 0.0,
    "contains_match": 0.0,
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(METRICS)}") from None


def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0,
    groups: Sequence[Any] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic for a given seed.

    With `groups` (one id per value, e.g. the document a page or question came from), whole
    groups are resampled (cluster bootstrap): samples of one document are correlated, and
    resampling them independently gives an interval that is too narrow."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    if groups is not None:
        if len(groups) != n:
            raise ValueError("groups and values differ in length")
        clusters: dict[Any, list[float]] = {}
        for g, v in zip(groups, values, strict=True):
            clusters.setdefault(g, []).append(v)
        sums = [(sum(c), len(c)) for c in clusters.values()]
        if len(sums) == 1:
            m = statistics.fmean(values)
            return m, m
        rng = random.Random(seed)
        means = []
        for _ in range(n_resamples):
            picked = rng.choices(sums, k=len(sums))
            means.append(sum(s for s, _ in picked) / sum(k for _, k in picked))
        means.sort()
        lo = means[int((alpha / 2) * n_resamples)]
        hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
        return lo, hi
    if n == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    vals = list(values)
    means = sorted(statistics.fmean(rng.choices(vals, k=n)) for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi

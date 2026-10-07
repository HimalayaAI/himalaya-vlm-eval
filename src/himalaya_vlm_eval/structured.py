"""Structured scorers for document understanding: key-value extraction, extractive QA with
unanswerable questions, reading order, table structure (TEDS) and layout regions.

Each scorer is `(prediction, references, target) -> {metric: value | None}`. `None` means
"not applicable to this sample" (e.g. answer accuracy on an unanswerable question) and is
excluded from that metric's mean, so per-subset rates stay honest.

Text comparison mirrors the nepal-pixel-synthesis generator: Unicode NFC + whitespace
collapse, nothing else folded (digits and orthography are compared as printed).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from html.parser import HTMLParser
from typing import Any

from . import text as T
from .metrics import anls, edit_distance

Score = dict[str, float | None]
Scorer = Callable[[str, Sequence[str], Any], Score]

ANSWER_NOT_PRESENT = "ANSWER NOT PRESENT"
LOGO_TOKEN = "[LOGO]"


def norm(value: Any) -> str:
    return T.canonical(str(value)) if value is not None else ""


def _sim(a: str, b: str) -> float:
    longest = max(len(a), len(b))
    return 1.0 - edit_distance(a, b) / longest if longest else 1.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


# --- JSON extraction ---------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(text: str) -> Any:
    """The first JSON object/array in a model reply (tolerates prose and code fences)."""
    text = T.clean_model_output(text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        candidate = match.group(1)
        for end in range(len(candidate), 0, -1):  # trim trailing junk after the JSON
            if candidate[end - 1] in "}]":
                try:
                    return json.loads(candidate[:end])
                except ValueError:
                    continue
    return None


# --- key-value fields --------------------------------------------------------------------


_WRAPPER_KEYS = {"fields", "data", "result", "results", "output", "extracted", "answer", "json"}


def _unwrap(pred: Any) -> Any:
    """{"fields": {...}} / {"data": {...}} → the inner mapping (models often wrap the answer)."""
    while isinstance(pred, dict) and len(pred) == 1:
        (k, v), = pred.items()
        if str(k).strip().lower() in _WRAPPER_KEYS and isinstance(v, (dict, list)):
            pred = v
        else:
            break
    return pred


def _split_multi(value: str, gold_values: Sequence[str]) -> list[str]:
    """Split a string holding several values of a repeated field.

    Split on commas and newlines, but first try to put pieces back together when the joined
    text is one of the gold values: `ठमेल, काठमाडौं, बागमती` for gold `ठमेल, काठमाडौं` +
    `बागमती` must not become three values. The longest rejoin that matches a gold value wins;
    anything else falls back to the plain split."""
    pieces = [x for x in re.split(r"\s*[,\n]\s*", value) if x]
    gold = set(gold_values)
    out: list[str] = []
    i = 0
    while i < len(pieces):
        for j in range(len(pieces), i + 1, -1):
            joined = next((c for c in (", ".join(pieces[i:j]), ",".join(pieces[i:j]))
                           if norm(c) in gold), None)
            if joined is not None:
                out.append(joined)
                i = j
                break
        else:
            out.append(pieces[i])
            i += 1
    return out


def _pred_pairs(pred: Any, multi: dict[str, list[str]]) -> list[tuple[str, str]]:
    """`multi` maps a repeated field id (case-folded) to its gold values."""
    pairs: list[tuple[str, str]] = []
    pred = _unwrap(pred)
    if isinstance(pred, dict):
        for k, v in pred.items():
            values = v if isinstance(v, list) else [v]
            if not isinstance(v, list) and str(k).casefold() in multi and isinstance(v, str):
                values = _split_multi(v, multi[str(k).casefold()])
            for item in values:
                if item is None or (isinstance(item, str) and not item.strip()):
                    continue
                pairs.append((str(k).strip(), norm(item)))
    elif isinstance(pred, list):  # [{"key": …, "value": …}] style
        for item in pred:
            if isinstance(item, dict):
                k = item.get("key", item.get("name", item.get("field")))
                if k is not None and item.get("value") not in (None, ""):
                    pairs.append((str(k).strip(), norm(item["value"])))
    return pairs


def score_kv(pred: str, refs: Sequence[str], target: Any) -> Score:
    """Field-level precision/recall/F1 over (field_id, value) pairs, as a multiset.

    target = {"fields": [[name, value], …]}. A pair counts only if the key matches exactly
    and the NFC/whitespace-normalised value matches exactly; `kv_value_sim` gives the
    softer per-field view (mean best similarity of each gold value under its key).
    """
    gold = [(str(k), norm(v)) for k, v in target["fields"] if norm(v)]
    multi = {k.casefold(): [v for g, v in gold if g == k]
             for k, _ in gold if sum(1 for g, _ in gold if g == k) > 1}
    parsed = extract_json(pred)
    # Field ids are matched case-insensitively ("Name" = "name"); values stay exact.
    canon = {k.casefold(): k for k, _ in gold}
    pairs = [(canon.get(k.casefold(), k), v) for k, v in _pred_pairs(parsed, multi)]
    remaining = list(pairs)
    tp = 0
    for g in gold:
        if g in remaining:
            remaining.remove(g)
            tp += 1
    p = tp / len(pairs) if pairs else 0.0
    r = tp / len(gold) if gold else 1.0
    sims = []
    for k, v in gold:
        cands = [pv for pk, pv in pairs if pk == k]
        sims.append(max((_sim(v, c) for c in cands), default=0.0))
    return {
        "kv_f1": _f1(p, r) if gold or pairs else 1.0,
        "kv_precision": p,
        "kv_recall": r,
        "kv_value_sim": sum(sims) / len(sims) if sims else 1.0,
        "kv_doc_exact": float(tp == len(gold) and len(pairs) == len(gold)),
        "kv_parse_ok": float(parsed is not None),
    }


# --- QA with unanswerable questions -----------------------------------------------------


def is_abstention(pred: str) -> bool:
    p = T.loose(T.clean_model_output(pred))
    return T.loose(ANSWER_NOT_PRESENT) in p


def score_qa(pred: str, refs: Sequence[str], target: Any) -> Score:
    """target = {"answerable": bool}. `qa_score` is ANLS on answerable questions and
    1/0 for correctly abstaining on unanswerable ones; the rest split the two failure modes
    (hallucinating an answer that is not on the page vs. refusing one that is)."""
    answerable = bool(target.get("answerable", True))
    abstained = is_abstention(pred)
    if answerable:
        acc = 0.0 if abstained else anls(T.canonical(T.clean_model_output(pred)),
                                         [T.canonical(r) for r in refs])
        return {"qa_score": acc, "anls_answerable": acc, "false_abstention": float(abstained),
                "abstention_accuracy": None}
    return {"qa_score": float(abstained), "anls_answerable": None, "false_abstention": None,
            "abstention_accuracy": float(abstained)}


# --- reading order -----------------------------------------------------------------------


def strip_logo(text: str) -> str:
    lines = [T.canonical(line.replace(LOGO_TOKEN, " ")) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _kendall_tau(seq: list[int]) -> float:
    n = len(seq)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                concordant += 1
            elif seq[i] > seq[j]:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def score_reading_order(pred: str, refs: Sequence[str], target: Any,
                        match_threshold: float = 0.5) -> Score:
    """Page transcription in reading order. target = {"blocks": [text, …]} in gold order.

    Each gold block is aligned to its most similar unused prediction line (similarity
    ≥ threshold). `reading_order` is Kendall's τ of the matched blocks' positions in the
    prediction, mapped to [0, 1]; `block_recall` is how many blocks were found at all.
    Page-level `cer` is on the full text, so order mistakes also cost CER.
    """
    gold_blocks = [T.canonical(b) for b in target["blocks"] if T.canonical(b)]
    out = T.clean_model_output(pred)
    lines = [T.canonical(line) for line in out.split("\n") if T.canonical(line)]
    gold_text = T.canonical(" ".join(gold_blocks))
    pred_text = T.canonical(" ".join(lines))
    if gold_text:
        cer = edit_distance(gold_text, pred_text) / len(gold_text)
    else:
        cer = float(bool(pred_text))

    # Align gold blocks to prediction lines by descending similarity (one-to-one), not greedily
    # in gold order: with near-identical blocks ("Total: 1000" / "Total: 2000") a garbled line
    # otherwise steals its neighbour's line and a correctly ordered page is scored as misordered.
    candidates = sorted(
        ((_sim(block, line), gi, li) for gi, block in enumerate(gold_blocks)
         for li, line in enumerate(lines)),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_lines: set[int] = set()
    matched: dict[int, int] = {}
    for sim, gi, li in candidates:
        if sim < match_threshold:
            break
        if gi in used_gold or li in used_lines:
            continue
        used_gold.add(gi)
        used_lines.add(li)
        matched[gi] = li
    positions = [matched[gi] for gi in sorted(matched)]
    recall = len(positions) / len(gold_blocks) if gold_blocks else 1.0
    tau = _kendall_tau(positions)
    return {
        "cer": cer,
        "char_accuracy": max(0.0, 1.0 - cer),
        "reading_order": (tau + 1) / 2 if len(positions) >= 2 else None,
        "block_recall": recall,
    }


# --- tables: TEDS ------------------------------------------------------------------------


class _Node:
    __slots__ = ("children", "colspan", "rowspan", "tag", "text")

    def __init__(self, tag: str, colspan: int = 1, rowspan: int = 1, text: str = ""):
        self.tag, self.colspan, self.rowspan, self.text = tag, colspan, rowspan, text
        self.children: list[_Node] = []


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[_Node] = []
        self._table: _Node | None = None
        self._row: _Node | None = None
        self._cell: _Node | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table" and self._table is None:
            self._table = _Node("table")
        elif self._table is None:
            return
        elif tag == "tr":
            self._close_cell()
            self._row = _Node("tr")
            self._table.children.append(self._row)
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:
                self._row = _Node("tr")
                self._table.children.append(self._row)

            def span(key: str) -> int:
                try:
                    return max(1, int(str(a.get(key) or 1).strip()))
                except ValueError:
                    return 1

            self._cell = _Node("td", span("colspan"), span("rowspan"))
            self._row.children.append(self._cell)
            self._buf = []
        elif tag == "br" and self._cell is not None:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._table is None:
            return
        if tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._close_cell()
            self._row = None
        elif tag == "table":
            self._close_cell()
            self.tables.append(self._table)
            self._table = self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buf.append(data)

    def _close_cell(self) -> None:
        if self._cell is not None:
            self._cell.text = T.canonical("".join(self._buf))
            self._cell = None
            self._buf = []

    def close(self) -> None:
        super().close()
        if self._table is not None:  # unterminated table: keep what we have
            self._close_cell()
            self.tables.append(self._table)
            self._table = None


def _markdown_table(text: str) -> _Node | None:
    rows = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if not rows:
        return None
    table = _Node("table")
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # header separator
        tr = _Node("tr")
        tr.children = [_Node("td", text=T.canonical(c)) for c in cells]
        table.children.append(tr)
    return table if table.children else None


def parse_table(text: str) -> _Node | None:
    """First HTML <table> in the text, else a markdown table, else None."""
    body = T.clean_model_output(text)
    parser = _TableParser()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        pass
    if parser.tables:
        return parser.tables[0]
    return _markdown_table(body)


def table_from_cells(cells: list[dict[str, Any]]) -> _Node:
    """Gold table from [{row, col, row_span, col_span, text}] (origin cells only)."""
    table = _Node("table")
    by_row: dict[int, list[dict[str, Any]]] = {}
    for c in cells:
        by_row.setdefault(int(c["row"]), []).append(c)
    for r in sorted(by_row):
        tr = _Node("tr")
        for c in sorted(by_row[r], key=lambda c: int(c["col"])):
            tr.children.append(_Node("td", int(c.get("col_span", 1)), int(c.get("row_span", 1)),
                                     T.canonical(c.get("text", ""))))
        table.children.append(tr)
    return table


def table_to_html(table: _Node) -> str:
    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = []
    for tr in table.children:
        cells = []
        for td in tr.children:
            attrs = (f' colspan="{td.colspan}"' if td.colspan > 1 else "") + \
                    (f' rowspan="{td.rowspan}"' if td.rowspan > 1 else "")
            cells.append(f"<td{attrs}>{esc(td.text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _postorder(root: _Node) -> tuple[list[_Node], list[int]]:
    """Post-order nodes and, for each, the index of its leftmost leaf descendant."""
    nodes: list[_Node] = []
    lml: list[int] = []

    def walk(n: _Node) -> int:
        first = -1
        for child in n.children:
            leaf = walk(child)
            if first < 0:
                first = leaf
        nodes.append(n)
        idx = len(nodes) - 1
        lml.append(first if first >= 0 else idx)
        return lml[idx]

    walk(root)
    return nodes, lml


def tree_edit_distance(a: _Node, b: _Node, rename: Callable[[_Node, _Node], float]) -> float:
    """Zhang–Shasha ordered tree edit distance; insert/delete cost 1."""
    an, al = _postorder(a)
    bn, bl = _postorder(b)

    def keyroots(lml: list[int]) -> list[int]:
        seen: dict[int, int] = {}
        for i, leaf in enumerate(lml):
            seen[leaf] = i
        return sorted(seen.values())

    td = [[0.0] * len(bn) for _ in an]
    for i in keyroots(al):
        for j in keyroots(bl):
            li, lj = al[i], bl[j]
            m, n = i - li + 2, j - lj + 2
            fd = [[0.0] * n for _ in range(m)]
            for x in range(1, m):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, n):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, m):
                for y in range(1, n):
                    i1, j1 = li + x - 1, lj + y - 1
                    if al[i1] == li and bl[j1] == lj:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[x - 1][y - 1] + rename(an[i1], bn[j1]))
                        td[i1][j1] = fd[x][y]
                    else:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[al[i1] - li][bl[j1] - lj] + td[i1][j1])
    return td[-1][-1]


def _count(n: _Node) -> int:
    return 1 + sum(_count(c) for c in n.children)


def teds(pred: _Node | None, gold: _Node, structure_only: bool = False) -> float:
    """Tree-Edit-Distance-based Similarity (Zhong et al., 2020). Cells with different spans
    cost 1 to rename; same-span cells cost their normalised text edit distance (0 when
    structure_only)."""
    if pred is None:
        return 0.0

    def rename(x: _Node, y: _Node) -> float:
        if x.tag != y.tag or x.colspan != y.colspan or x.rowspan != y.rowspan:
            return 1.0
        if x.tag != "td" or structure_only:
            return 0.0
        return 1.0 - _sim(x.text, y.text)

    dist = tree_edit_distance(pred, gold, rename)
    return 1.0 - dist / max(_count(pred), _count(gold))


def score_table(pred: str, refs: Sequence[str], target: Any) -> Score:
    """target = {"cells": [...]} (see table_from_cells)."""
    gold = table_from_cells(target["cells"])
    parsed = parse_table(pred)
    return {
        "teds": teds(parsed, gold),
        "teds_struct": teds(parsed, gold, structure_only=True),
        "table_parse_ok": float(parsed is not None),
    }


# --- layout regions ----------------------------------------------------------------------


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _box_xyxy(item: dict[str, Any], size: Sequence[float] | None) -> list[float] | None:
    """A predicted box as normalised [x0, y0, x1, y1], or None if unreadable.

    Accepted: `bbox`/`box` as [x0, y0, x1, y1]; `box_2d` as [y0, x0, y1, x1] (the Gemini
    convention); values on a 0-1000 grid (the prompt's format), 0-1 floats, or pixels when the
    page size is known and a value exceeds 1000."""
    yxyx = "bbox" not in item and "box" not in item and "box_2d" in item
    raw = item.get("bbox", item.get("box", item.get("box_2d")))
    try:
        v = [float(x) for x in raw]
    except (TypeError, ValueError):
        return None
    if len(v) != 4:
        return None
    if yxyx:
        v = [v[1], v[0], v[3], v[2]]
    top = max(abs(x) for x in v)
    if top <= 1.5:
        scale = (1.0, 1.0)
    elif top > 1000 and size:
        scale = (float(size[0]), float(size[1]))
    else:
        scale = (1000.0, 1000.0)
    x0, y0, x1, y1 = v[0] / scale[0], v[1] / scale[1], v[2] / scale[0], v[3] / scale[1]
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _pred_regions(
    parsed: Any, size: Sequence[float] | None = None
) -> list[tuple[str, list[float]]]:
    items = parsed.get("regions", parsed.get("elements")) if isinstance(parsed, dict) else parsed
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("type", item.get("label", "")))).strip().lower()
        box = _box_xyxy(item, size)
        if box is not None:
            out.append((role, box))
    return out


def _match(gold: list[tuple[str, list[float]]], pred: list[tuple[str, list[float]]],
           thr: float, same_role: bool) -> int:
    pairs = sorted(
        ((iou(g[1], p[1]), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)
         if not same_role or g[0] == p[0]),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    tp = 0
    for v, gi, pi in pairs:
        if v < thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1
    return tp


def score_layout(pred: str, refs: Sequence[str], target: Any, thr: float = 0.5) -> Score:
    """target = {"regions": [{"role", "box": [x0,y0,x1,y1] in 0–1]}]}. Predictions are
    `[{"role", "bbox": [x0,y0,x1,y1]}]` on a 0–1000 grid. 0–1 floats, pixels (when
    target["size"] = [w, h] is given) and Gemini-style `box_2d` = [y0,x0,y1,x1] are also read.
    A match needs IoU ≥ 0.5; `layout_f1` also needs the role to agree, `detection_f1` does not."""
    gold = [(r["role"], list(r["box"])) for r in target["regions"]]
    parsed = extract_json(pred)
    preds = _pred_regions(parsed, target.get("size"))
    out: Score = {"layout_parse_ok": float(parsed is not None)}
    for name, same_role in (("layout", True), ("detection", False)):
        tp = _match(gold, preds, thr, same_role)
        p = tp / len(preds) if preds else 0.0
        r = tp / len(gold) if gold else 1.0
        out[f"{name}_f1"] = _f1(p, r) if gold or preds else 1.0
        if same_role:
            out["layout_precision"], out["layout_recall"] = p, r
    return out


SCORERS: dict[str, Scorer] = {
    "kv": score_kv,
    "qa": score_qa,
    "reading_order": score_reading_order,
    "table": score_table,
    "layout": score_layout,
}

METRICS: dict[str, list[str]] = {
    "kv": ["kv_f1", "kv_precision", "kv_recall", "kv_value_sim", "kv_doc_exact", "kv_parse_ok"],
    "qa": ["qa_score", "anls_answerable", "false_abstention", "abstention_accuracy"],
    "reading_order": ["cer", "char_accuracy", "reading_order", "block_recall"],
    "table": ["teds", "teds_struct", "table_parse_ok"],
    "layout": ["layout_f1", "layout_precision", "layout_recall", "detection_f1",
               "layout_parse_ok"],
}

# Worst-case values for an errored sample, per scorer.
WORST: dict[str, Score] = {
    "kv": {"kv_f1": 0.0, "kv_precision": 0.0, "kv_recall": 0.0, "kv_value_sim": 0.0,
           "kv_doc_exact": 0.0, "kv_parse_ok": 0.0},
    "qa": {"qa_score": 0.0},
    "reading_order": {"cer": 1.0, "char_accuracy": 0.0, "reading_order": None,
                      "block_recall": 0.0},
    "table": {"teds": 0.0, "teds_struct": 0.0, "table_parse_ok": 0.0},
    "layout": {"layout_parse_ok": 0.0, "layout_f1": 0.0, "layout_precision": 0.0,
               "layout_recall": 0.0, "detection_f1": 0.0},
}

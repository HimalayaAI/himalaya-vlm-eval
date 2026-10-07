"""Read nepal-pixel-synthesis document output as benchmark samples.

The generator writes one directory per leaf (document type):

    <root>/<category>/<subcategory>/<doc_type>/
      images/<sample_id>.png | <sample_id>_pNN.png | <sample_id>_ebNN.png
      ground_truth/<sample_id>.gt.json        key-value fields
      sharegpt/<sample_id>[_ebNN].sharegpt.json  Q&A (positives / "ANSWER NOT PRESENT" negatives)
      metadata.jsonl                          one row per document; page IR in row["pages"]

These document-level labels are not in the published Hugging Face export (that one is
line-level), so these benchmarks read a generator output directory directly. One loader,
five views:

    kv            whole document → JSON of canonical field ids        (single-page docs)
    qa            one question per sample, incl. unanswerable negatives (single-image)
    page          one page → transcription in reading order
    table         one structured table region, cropped → HTML table
    layout        one page → regions with roles and boxes
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from . import structured as S
from .types import DataUnavailable, Sample

log = logging.getLogger("himeval")

VIEWS = ("kv", "qa", "page", "table", "layout")
_NEGATIVE_SUFFIX = re.compile(r"_eb\d+$")
_LAYOUT_EXCLUDED_ROLES = {"table_cell", "table_row"}
_LAYOUT_STATUSES = {"rendered", "non_text", "clipped"}


class _PageImage:
    __slots__ = ("crop", "path")

    def __init__(self, path: Path, crop: tuple[int, int, int, int] | None = None):
        self.path, self.crop = path, crop

    def __call__(self) -> Any:
        from PIL import Image

        with Image.open(self.path) as im:
            im.load()
            img = im.convert("RGB")
        if self.crop:
            img = img.crop(self.crop)
        return img


def _root(spec: dict[str, Any]) -> Path:
    raw = spec.get("path") or os.environ.get(spec.get("path_env", "NEPALIPIXEL_DOCS_DIR"), "")
    if not raw:
        raise DataUnavailable(
            "document benchmarks read nepal-pixel-synthesis output: set dataset.path or "
            f"${spec.get('path_env', 'NEPALIPIXEL_DOCS_DIR')} to the generator's output root"
        )
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise DataUnavailable(f"generator output root {root} does not exist")
    return root


def _resolve_image(leaf: Path, root: Path, ref: str) -> Path:
    p = Path(ref)
    for candidate in (p, leaf / p, leaf / "images" / p.name, root / p):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image {ref!r} not found near {leaf}")


def _rect(poly: Any) -> tuple[float, float, float, float]:
    """4-point polygon [[x,y]…] or flat [x0,y0,x1,y1] → axis-aligned rect."""
    if poly and isinstance(poly[0], (list, tuple)):
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        return min(xs), min(ys), max(xs), max(ys)
    x0, y0, x1, y1 = (float(v) for v in poly)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in ("doc_type", "domain", "leaf", "format_id", "intensity",
                                     "layout_profile", "font_name", "split")}


def load_generator_output(spec: dict[str, Any]) -> tuple[list[Sample], dict[str, Any]]:
    view = spec.get("view")
    if view not in VIEWS:
        raise ValueError(f"dataset.view must be one of {VIEWS}, got {view!r}")
    root = _root(spec)
    splits = set(spec["splits"]) if spec.get("splits") else None
    digest = hashlib.sha256()
    docs: list[tuple[Path, dict[str, Any]]] = []
    for meta_path in sorted(root.rglob("metadata.jsonl")):
        data = meta_path.read_bytes()
        digest.update(meta_path.relative_to(root).as_posix().encode() + b"\0" + data)
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if splits is not None and row.get("split") not in splits:
                continue
            if "pages" not in row:  # a line-level corpus row, not a document
                continue
            docs.append((meta_path.parent, row))
    if not docs:
        raise ValueError(f"no document rows under {root}"
                         + (f" with split in {sorted(splits)}" if splits else ""))

    loader = {"kv": _kv, "qa": _qa, "page": _pages, "table": _tables, "layout": _layout}[view]
    samples: list[Sample] = []
    skipped: dict[str, int] = {}
    for leaf, row in docs:
        try:
            samples.extend(loader(root, leaf, row, skipped))
        except FileNotFoundError as exc:
            skipped["missing_file"] = skipped.get("missing_file", 0) + 1
            log.warning("%s: %s", row.get("id"), exc)
    if skipped:
        log.info("%s view: skipped %s", view, skipped)
    provenance = {"generator_output": str(root), "metadata_sha256": digest.hexdigest()[:16],
                  "documents": len(docs), "view": view, "splits": sorted(splits or []),
                  "skipped": skipped}
    return samples, provenance


# --- views -------------------------------------------------------------------------------


def _kv(root: Path, leaf: Path, row: dict[str, Any], skipped: dict[str, int]) -> list[Sample]:
    if int(row.get("page_count") or len(row["pages"])) != 1:
        skipped["multi_page"] = skipped.get("multi_page", 0) + 1
        return []
    gt_ref = row.get("gt_path") or f"ground_truth/{row['id']}.gt.json"
    gt_path = Path(gt_ref) if Path(gt_ref).is_absolute() else leaf / gt_ref
    if not gt_path.is_file():
        gt_path = leaf / "ground_truth" / f"{row['id']}.gt.json"
    gt = json.loads(gt_path.read_text("utf-8"))
    pairs = [[f["name"], f["value"]] for f in gt.get("fields", [])
             if str(f.get("value", "")).strip()]
    if not pairs:
        skipped["no_fields"] = skipped.get("no_fields", 0) + 1
        return []
    keys = list(dict.fromkeys(k for k, _ in pairs))
    page = row["pages"][0]
    return [Sample(
        id=str(row["id"]),
        references=[],
        image=_PageImage(_resolve_image(leaf, root, page["image_path"])),
        question=", ".join(keys),
        meta=_meta(row) | {"n_fields": len(pairs)},
        target={"fields": pairs},
    )]


def _qa(root: Path, leaf: Path, row: dict[str, Any], skipped: dict[str, int]) -> list[Sample]:
    out: list[Sample] = []
    sharegpt = leaf / "sharegpt"
    if not sharegpt.is_dir():
        return out
    doc_id = str(row["id"])
    for path in sorted(sharegpt.glob(f"{doc_id}*.sharegpt.json")):
        rec_id = path.name.removesuffix(".sharegpt.json")
        if rec_id != doc_id and not _NEGATIVE_SUFFIX.search(rec_id[len(doc_id):]):
            continue  # a different doc whose id shares this prefix
        rec = json.loads(path.read_text("utf-8"))
        images = rec.get("images") or [rec.get("image")]
        if len(images) != 1:
            skipped["multi_image"] = skipped.get("multi_image", 0) + 1
            continue
        image = _PageImage(_resolve_image(leaf, root, images[0]))
        turns = rec.get("conversations") or []
        for k in range(0, len(turns) - 1, 2):
            human, gpt = turns[k], turns[k + 1]
            if human.get("from") != "human" or gpt.get("from") != "gpt":
                continue
            question = str(human["value"]).replace("<image>", "").strip()
            answer = str(gpt["value"])
            answerable = answer.strip() != S.ANSWER_NOT_PRESENT
            blocking = rec.get("extraction_blocking") or {}
            out.append(Sample(
                id=f"{rec_id}#q{k // 2}",
                references=[answer] if answerable else [],
                image=image,
                question=question,
                meta=_meta(row) | {"answerable": answerable,
                                   "field": blocking.get("field")},
                target={"answerable": answerable},
            ))
    return out


def _page_text(page: dict[str, Any]) -> str:
    return S.strip_logo(str(page.get("text", "")))


def _pages(root: Path, leaf: Path, row: dict[str, Any], skipped: dict[str, int]) -> list[Sample]:
    out = []
    for page in row["pages"]:
        text = _page_text(page)
        if not text:
            skipped["empty_page"] = skipped.get("empty_page", 0) + 1
            continue
        out.append(Sample(
            id=f"{row['id']}#p{page.get('page_number', 1)}",
            references=[text.replace("\n", " ")],
            image=_PageImage(_resolve_image(leaf, root, page["image_path"])),
            meta=_meta(row) | {"page_number": page.get("page_number", 1)},
            target={"blocks": text.split("\n")},
        ))
    return out


def _cell_text(words: list[dict[str, Any]], table_rid: str, fields: list[str]) -> str:
    if not fields:
        return ""
    parts = [w["text"] for w in words
             if w.get("region_id") == table_rid and w.get("field") in fields
             and w.get("text") != S.LOGO_TOKEN]
    return " ".join(parts)


def _tables(root: Path, leaf: Path, row: dict[str, Any], skipped: dict[str, int]) -> list[Sample]:
    out = []
    for page in row["pages"]:
        regions = page.get("regions") or []
        words = page.get("bboxes") or []
        w, h = page.get("image_w"), page.get("image_h")
        for region in regions:
            if region.get("role") != "table" or region.get("render_mode") != "structured_table":
                continue
            if region.get("status") not in (None, "rendered"):
                continue
            rid = region["region_id"]
            cells = [r for r in regions if r.get("role") == "table_cell" and r.get("parent") == rid]
            if not cells:
                skipped["table_without_cells"] = skipped.get("table_without_cells", 0) + 1
                continue
            gold = [{"row": c["row"], "col": c["col"], "row_span": c.get("row_span", 1),
                     "col_span": c.get("col_span", 1),
                     "text": _cell_text(words, rid, c.get("fields") or [])} for c in cells]
            x0, y0, x1, y1 = _rect(region["box_2d"])
            pad = 8
            crop = (max(0, int(x0) - pad), max(0, int(y0) - pad),
                    int(min(w or x1 + pad, x1 + pad)), int(min(h or y1 + pad, y1 + pad)))
            out.append(Sample(
                id=f"{row['id']}#p{page.get('page_number', 1)}#{region.get('id', rid)}",
                references=[S.table_to_html(S.table_from_cells(gold))],
                image=_PageImage(_resolve_image(leaf, root, page["image_path"]), crop),
                meta=_meta(row) | {"cells": len(cells),
                                   "rows": 1 + max(int(c["row"]) for c in cells)},
                target={"cells": gold},
            ))
    return out


def _layout(root: Path, leaf: Path, row: dict[str, Any], skipped: dict[str, int]) -> list[Sample]:
    out = []
    for page in row["pages"]:
        regions = page.get("regions")
        if not regions:
            skipped["no_regions"] = skipped.get("no_regions", 0) + 1
            continue
        norm = {r["region_id"]: r for r in page.get("regions_norm") or []}
        w, h = float(page.get("image_w") or 0), float(page.get("image_h") or 0)
        gold = []
        for r in regions:
            if r.get("role") in _LAYOUT_EXCLUDED_ROLES or r.get("status") not in _LAYOUT_STATUSES:
                continue
            if r["region_id"] in norm:
                box = _rect(norm[r["region_id"]]["box_2d"])
            elif w and h:
                x0, y0, x1, y1 = _rect(r["box_2d"])
                box = (x0 / w, y0 / h, x1 / w, y1 / h)
            else:
                continue
            gold.append({"role": r["role"], "box": [round(v, 6) for v in box]})
        if not gold:
            continue
        out.append(Sample(
            id=f"{row['id']}#p{page.get('page_number', 1)}",
            references=[],
            image=_PageImage(_resolve_image(leaf, root, page["image_path"])),
            meta=_meta(row) | {"regions": len(gold)},
            target={"regions": gold},
        ))
    return out

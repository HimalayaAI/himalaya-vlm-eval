"""Document views over a synthetic generator output that mirrors nepal-pixel-synthesis'
on-disk format (metadata.jsonl page IR, ground_truth/*.gt.json, sharegpt/*.json)."""

import json
import os
from pathlib import Path

import pytest

from himalaya_vlm_eval import catalog
from himalaya_vlm_eval.benchmarks import NativeBenchmark

from .conftest import make_image


def _poly(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def build_output(root: Path, doc_id="cert_format_01_000000", split="benchmark") -> Path:
    leaf = root / "identity" / "citizenship" / "cert"
    make_image(leaf / "images" / f"{doc_id}.png", size=(1000, 800))
    make_image(leaf / "images" / f"{doc_id}_eb00.png", size=(1000, 800))
    regions = [
        {"region_id": "r000", "id": "logo#0", "section": "logo", "role": "logo",
         "status": "non_text", "render_mode": "no_text", "order_rank": 0,
         "box_2d": _poly(10, 10, 110, 110)},
        {"region_id": "r001", "id": "title#0", "section": "title", "role": "title",
         "status": "rendered", "render_mode": "plain_text", "order_rank": 1,
         "box_2d": _poly(300, 20, 700, 80)},
        {"region_id": "r002", "id": "kv#0", "section": "kv", "role": "key_value",
         "status": "rendered", "render_mode": "kv_pairs", "order_rank": 2,
         "box_2d": _poly(50, 150, 600, 200), "fields": ["citizenship_number"]},
        {"region_id": "r003", "id": "tbl#0", "section": "tbl", "role": "table",
         "status": "rendered", "render_mode": "structured_table", "order_rank": 3,
         "box_2d": _poly(50, 300, 950, 500)},
        {"region_id": "r003c0_0", "id": "tbl#0/r0c0", "role": "table_cell", "parent": "r003",
         "status": "rendered", "row": 0, "col": 0, "row_span": 1, "col_span": 2,
         "is_header": True, "fields": ["hdr"], "box_2d": _poly(50, 300, 950, 400)},
        {"region_id": "r003c1_0", "id": "tbl#0/r1c0", "role": "table_cell", "parent": "r003",
         "status": "rendered", "row": 1, "col": 0, "row_span": 1, "col_span": 1,
         "fields": ["father"], "box_2d": _poly(50, 400, 500, 500)},
        {"region_id": "r003c1_1", "id": "tbl#0/r1c1", "role": "table_cell", "parent": "r003",
         "status": "blank_by_design", "row": 1, "col": 1, "row_span": 1, "col_span": 1,
         "box_2d": _poly(500, 400, 950, 500)},
    ]
    words = [
        {"text": "[LOGO]", "box_2d": _poly(10, 10, 110, 110), "region_id": "r000", "field": "logo"},
        {"text": "नेपाल", "box_2d": _poly(300, 20, 450, 80), "region_id": "r001"},
        {"text": "सरकार", "box_2d": _poly(460, 20, 700, 80), "region_id": "r001"},
        {"text": "ना.प्र.नं.:", "box_2d": _poly(50, 150, 200, 200), "region_id": "r002"},
        {"text": "४२-४२", "box_2d": _poly(210, 150, 600, 200), "region_id": "r002"},
        {"text": "विवरण", "box_2d": _poly(60, 310, 300, 390), "region_id": "r003", "field": "hdr"},
        {"text": "बाबु", "box_2d": _poly(60, 410, 200, 490), "region_id": "r003", "field": "father"},
        {"text": "हरि", "box_2d": _poly(210, 410, 300, 490), "region_id": "r003", "field": "father"},
    ]
    page = {"page_number": 1, "image_path": f"images/{doc_id}.png", "image_w": 1000,
            "image_h": 800, "text": "[LOGO]\nनेपाल सरकार\nना.प्र.नं.: ४२-४२\nविवरण बाबु हरि",
            "regions": regions, "bboxes": words}
    row = {"id": doc_id, "doc_type": "cert", "domain": "government", "split": split,
           "page_count": 1, "pages": [page], "gt_path": f"ground_truth/{doc_id}.gt.json",
           "intensity": "clean", "layout_profile": "canonical"}
    other = {**row, "id": "cert_format_01_000009", "split": "training"}
    (leaf / "metadata.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(other, ensure_ascii=False) + "\n",
        "utf-8")
    (leaf / "ground_truth").mkdir()
    (leaf / "ground_truth" / f"{doc_id}.gt.json").write_text(json.dumps({
        "id": doc_id, "fields": [
            {"name": "citizenship_number", "value": "४२-४२", "section": "kv"},
            {"name": "father", "value": "बाबु हरि", "section": "tbl", "field": "father"},
            {"name": "logo", "value": " ", "section": "logo"}]}, ensure_ascii=False), "utf-8")
    (leaf / "sharegpt").mkdir()
    (leaf / "sharegpt" / f"{doc_id}.sharegpt.json").write_text(json.dumps({
        "id": doc_id, "image": f"images/{doc_id}.png", "images": [f"images/{doc_id}.png"],
        "conversations": [
            {"from": "human", "value": "<image>\nनागरिकता नम्बर के हो?"},
            {"from": "gpt", "value": "४२-४२"},
            {"from": "human", "value": "बाबुको नाम के हो?"},
            {"from": "gpt", "value": "बाबु हरि"}]}, ensure_ascii=False), "utf-8")
    (leaf / "sharegpt" / f"{doc_id}_eb00.sharegpt.json").write_text(json.dumps({
        "id": f"{doc_id}_eb00", "image": f"images/{doc_id}_eb00.png",
        "images": [f"images/{doc_id}_eb00.png"],
        "conversations": [{"from": "human", "value": "<image>\nनागरिकता नम्बर के हो?"},
                          {"from": "gpt", "value": "ANSWER NOT PRESENT"}],
        "extraction_blocking": {"field": "citizenship_number"}}, ensure_ascii=False), "utf-8")
    return root


@pytest.fixture
def docs(tmp_path, monkeypatch):
    root = build_output(tmp_path / "gen")
    monkeypatch.setenv("NEPALIPIXEL_DOCS_DIR", str(root))
    return root


def load(view):
    b = NativeBenchmark(catalog.resolve_benchmark(f"nepalipixel-docs-{view}"))
    return b, b.load()


def test_kv_view(docs):
    b, ls = load("kv")
    assert len(ls.samples) == 1  # the training-split doc is excluded
    s = ls.samples[0]
    assert s.target["fields"] == [["citizenship_number", "४२-४२"], ["father", "बाबु हरि"]]
    assert "Field ids: citizenship_number, father" in b.prompt(s).text
    assert ls.provenance["documents"] == 1 and ls.provenance["metadata_sha256"]


def test_qa_view_includes_negatives(docs):
    b, ls = load("qa")
    by_id = {s.id: s for s in ls.samples}
    assert set(by_id) == {"cert_format_01_000000#q0", "cert_format_01_000000#q1",
                          "cert_format_01_000000_eb00#q0"}
    neg = by_id["cert_format_01_000000_eb00#q0"]
    assert neg.target == {"answerable": False} and neg.references == []
    assert neg.question == "नागरिकता नम्बर के हो?"
    assert neg.load_image().size == (1000, 800)
    assert "ANSWER NOT PRESENT" in b.prompt(neg).text
    # the blocked copy is the same document, so the CI resamples them together
    assert {s.meta["doc_id"] for s in ls.samples} == {"cert_format_01_000000"}


def test_page_view_strips_logo(docs):
    b, ls = load("page")
    s = ls.samples[0]
    assert s.target["blocks"] == ["नेपाल सरकार", "ना.प्र.नं.: ४२-४२", "विवरण बाबु हरि"]
    assert "[LOGO]" not in s.references[0]
    assert b.score("\n".join(s.target["blocks"]), s.references, s.target)["cer"] == 0.0


def test_table_view_rebuilds_cells_and_crops(docs):
    b, ls = load("table")
    s = ls.samples[0]
    assert s.references[0] == ('<table><tr><td colspan="2">विवरण</td></tr>'
                               "<tr><td>बाबु हरि</td><td></td></tr></table>")
    assert s.load_image().size == (916, 216)  # region + 8px padding
    assert b.score(s.references[0], [], s.target)["teds"] == 1.0


def test_layout_view(docs):
    _, ls = load("layout")
    roles = sorted(r["role"] for r in ls.samples[0].target["regions"])
    assert roles == ["key_value", "logo", "table", "title"]  # cells excluded
    title = next(r for r in ls.samples[0].target["regions"] if r["role"] == "title")
    assert title["box"] == [0.3, 0.025, 0.7, 0.1]


def test_errored_sample_scores_worst(docs):
    b, _ = load("qa")
    assert b.score(None, [], {"answerable": True})["qa_score"] == 0.0


def test_missing_root_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("NEPALIPIXEL_DOCS_DIR", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="does not exist"):
        load("kv")
    monkeypatch.delenv("NEPALIPIXEL_DOCS_DIR")
    with pytest.raises(ValueError, match="NEPALIPIXEL_DOCS_DIR"):
        load("kv")


@pytest.mark.skipif(not os.environ.get("HIMEVAL_REAL_DOCS"),
                    reason="set HIMEVAL_REAL_DOCS to a real generator output root")
def test_real_generator_output(monkeypatch):
    monkeypatch.setenv("NEPALIPIXEL_DOCS_DIR", os.environ["HIMEVAL_REAL_DOCS"])
    for view in ("kv", "qa", "page", "table", "layout"):
        b, ls = load(view)
        assert ls.samples, view
        s = ls.samples[0]
        s.load_image()
        if view == "page":
            assert b.score("\n".join(s.target["blocks"]), s.references, s.target)["cer"] == 0.0
        if view == "qa":
            assert any(not x.target["answerable"] for x in ls.samples)

import json

import pytest

from himalaya_vlm_eval import structured as S

# --- key-value ---------------------------------------------------------------------------

KV = {"fields": [["citizenship_number", "४२-४२-२०५८-००५६३"], ["full_name", "सीता राम पौडेल"],
                 ["child", "राम"], ["child", "श्याम"]]}


def test_kv_perfect_with_fence_and_list_values():
    pred = "```json\n" + json.dumps({"citizenship_number": "४२-४२-२०५८-००५६३",
                                       "full_name": "सीता  राम पौडेल",
                                       "child": ["राम", "श्याम"]}, ensure_ascii=False) + "\n```"
    s = S.score_kv(pred, [], KV)
    assert s["kv_f1"] == 1.0 and s["kv_doc_exact"] == 1.0 and s["kv_parse_ok"] == 1.0


def test_kv_multi_value_as_comma_string():
    pred = json.dumps({"child": "राम, श्याम"}, ensure_ascii=False)
    s = S.score_kv(pred, [], KV)
    assert s["kv_recall"] == 0.5 and s["kv_precision"] == 1.0


def test_kv_partial_wrong_and_extra():
    pred = json.dumps({"citizenship_number": "४२-४२-२०५८-००५६४", "full_name": "सीता राम पौडेल",
                       "invented": "x"}, ensure_ascii=False)
    s = S.score_kv(pred, [], KV)
    assert s["kv_precision"] == pytest.approx(1 / 3)
    assert s["kv_recall"] == pytest.approx(1 / 4)
    assert 0 < s["kv_value_sim"] < 1  # near-miss on the number gets partial similarity
    assert s["kv_doc_exact"] == 0.0


def test_kv_digits_are_not_folded():
    pred = json.dumps({"citizenship_number": "42-42-2058-00563"})
    assert S.score_kv(pred, [], {"fields": [["citizenship_number", "४२-४२-२०५८-००५६३"]]})[
        "kv_f1"] == 0.0


def test_kv_unparseable_and_list_of_pairs_form():
    assert S.score_kv("I cannot read this", [], KV)["kv_parse_ok"] == 0.0
    pred = json.dumps([{"key": "full_name", "value": "सीता राम पौडेल"}], ensure_ascii=False)
    assert S.score_kv(pred, [], KV)["kv_precision"] == 1.0


def test_extract_json_tolerates_prose_and_trailing_text():
    assert S.extract_json('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert S.extract_json("[1, 2] trailing ]") == [1, 2]
    assert S.extract_json("no json") is None


# --- QA ------------------------------------------------------------------------------------


def test_qa_answerable_and_unanswerable():
    pos = S.score_qa("४२-४२-२०५८-००५६३", ["४२-४२-२०५८-००५६३"], {"answerable": True})
    assert pos["qa_score"] == 1.0 and pos["abstention_accuracy"] is None
    hallucinated = S.score_qa("राम", [], {"answerable": False})
    assert hallucinated["qa_score"] == 0.0 and hallucinated["abstention_accuracy"] == 0.0
    abstained = S.score_qa("answer not present.", [], {"answerable": False})
    assert abstained["qa_score"] == 1.0
    refused = S.score_qa("ANSWER NOT PRESENT", ["राम"], {"answerable": True})
    assert refused["qa_score"] == 0.0 and refused["false_abstention"] == 1.0


# --- reading order ---------------------------------------------------------------------------

BLOCKS = {"blocks": ["नेपाल सरकार", "गृह मन्त्रालय", "जिल्ला प्रशासन कार्यालय", "नाम थर: सीता"]}


def test_reading_order_perfect_and_swapped():
    perfect = S.score_reading_order("\n".join(BLOCKS["blocks"]), [], BLOCKS)
    assert perfect["reading_order"] == 1.0 and perfect["cer"] == 0.0
    reversed_ = S.score_reading_order("\n".join(reversed(BLOCKS["blocks"])), [], BLOCKS)
    assert reversed_["reading_order"] == 0.0
    assert reversed_["block_recall"] == 1.0
    assert reversed_["cer"] > 0


def test_reading_order_missing_blocks():
    s = S.score_reading_order("नेपाल सरकार", [], BLOCKS)
    assert s["block_recall"] == 0.25 and s["reading_order"] is None


def test_strip_logo():
    assert S.strip_logo("[LOGO]\nनेपाल सरकार\n[LOGO] गृह") == "नेपाल सरकार\nगृह"


# --- TEDS ----------------------------------------------------------------------------------

CELLS = [
    {"row": 0, "col": 0, "row_span": 1, "col_span": 2, "text": "शीर्षक"},
    {"row": 1, "col": 0, "row_span": 1, "col_span": 1, "text": "क"},
    {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "ख"},
]


def test_teds_identical_html_and_markdown():
    gold_html = S.table_to_html(S.table_from_cells(CELLS))
    assert S.score_table(gold_html, [], {"cells": CELLS})["teds"] == 1.0
    md = "| a | b |\n|---|---|\n| क | ख |"
    flat = [{"row": 0, "col": 0, "text": "a"}, {"row": 0, "col": 1, "text": "b"},
            {"row": 1, "col": 0, "text": "क"}, {"row": 1, "col": 1, "text": "ख"}]
    assert S.score_table(md, [], {"cells": flat})["teds"] == 1.0


def test_teds_text_error_vs_structure_error():
    text_err = '<table><tr><td colspan="2">शीर्षक</td></tr><tr><td>क</td><td>ग</td></tr></table>'
    s = S.score_table(text_err, [], {"cells": CELLS})
    assert s["teds_struct"] == 1.0 and 0.8 < s["teds"] < 1.0
    struct_err = "<table><tr><td>शीर्षक</td></tr><tr><td>क</td><td>ख</td></tr></table>"
    s2 = S.score_table(struct_err, [], {"cells": CELLS})
    assert s2["teds_struct"] < 1.0


def test_teds_missing_row_and_garbage():
    missing = '<table><tr><td colspan="2">शीर्षक</td></tr></table>'
    assert S.score_table(missing, [], {"cells": CELLS})["teds"] == pytest.approx(1 - 3 / 6)
    assert S.score_table("no table", [], {"cells": CELLS}) == {
        "teds": 0.0, "teds_struct": 0.0, "table_parse_ok": 0.0}


def test_teds_tolerates_unclosed_tags_and_th():
    html = '<table><tr><th colspan="2">शीर्षक<tr><td>क<td>ख'
    assert S.score_table(html, [], {"cells": CELLS})["teds"] == 1.0


def test_tree_edit_distance_known_value():
    # Zhang–Shasha classic: f(d(a c(b)) e) vs f(c(d(a b)) e) has distance 2.
    def n(tag, *kids):
        node = S._Node(tag)
        node.children = list(kids)
        return node

    t1 = n("f", n("d", n("a"), n("c", n("b"))), n("e"))
    t2 = n("f", n("c", n("d", n("a"), n("b"))), n("e"))
    assert S.tree_edit_distance(t1, t2, lambda x, y: float(x.tag != y.tag)) == 2


# --- layout ----------------------------------------------------------------------------------

LAYOUT = {"regions": [{"role": "title", "box": [0.1, 0.05, 0.9, 0.1]},
                      {"role": "table", "box": [0.1, 0.5, 0.9, 0.9]}]}


def test_layout_scoring():
    perfect = json.dumps([{"role": "title", "bbox": [100, 50, 900, 100]},
                          {"role": "table", "bbox": [100, 500, 900, 900]}])
    assert S.score_layout(perfect, [], LAYOUT)["layout_f1"] == 1.0
    wrong_role = json.dumps([{"role": "header", "bbox": [100, 50, 900, 100]},
                             {"role": "table", "bbox": [100, 500, 900, 900]}])
    s = S.score_layout(wrong_role, [], LAYOUT)
    assert s["layout_f1"] == 0.5 and s["detection_f1"] == 1.0
    shifted = json.dumps([{"role": "table", "bbox": [100, 800, 900, 1000]}])
    assert S.score_layout(shifted, [], LAYOUT)["layout_f1"] == 0.0
    assert S.score_layout("{}", [], LAYOUT)["layout_recall"] == 0.0


def test_iou():
    assert S.iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert S.iou([0, 0, 1, 1], [1, 1, 2, 2]) == 0.0
    assert S.iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_every_scorer_declares_metrics_and_worst_case():
    assert set(S.SCORERS) == set(S.METRICS) == set(S.WORST)
    for name, worst in S.WORST.items():
        assert set(worst) <= set(S.METRICS[name])


# --- regressions found in review (7 Oct 2026) ---------------------------------------------


def test_reading_order_garbled_line_does_not_steal_neighbour():
    gold = ["कुल रकम: १०००", "कुल रकम: २०००", "हस्ताक्षर"]
    pred = "कुल रकम १०\nकुल रकम: २०००\nहस्ताक्षर"  # correct order, first line has OCR errors
    s = S.score_reading_order(pred, [], {"blocks": gold})
    assert s["reading_order"] == 1.0 and s["block_recall"] == 1.0


def test_reading_order_reversed_pair_still_scores_zero():
    gold = ["पहिलो खण्ड", "दोस्रो खण्ड"]
    s = S.score_reading_order("दोस्रो खण्ड\nपहिलो खण्ड", [], {"blocks": gold})
    assert s["reading_order"] == 0.0


GOLD_BOX = {"regions": [{"role": "title", "box": [0.1, 0.2, 0.5, 0.4]}]}


@pytest.mark.parametrize("key,box", [
    ("bbox", [100, 200, 500, 400]),        # 0-1000 grid, x0 y0 x1 y1 (the prompt's format)
    ("box_2d", [200, 100, 400, 500]),      # Gemini convention: y0 x0 y1 x1
    ("bbox", [0.1, 0.2, 0.5, 0.4]),        # 0-1 floats
])
def test_layout_accepts_common_box_formats(key, box):
    pred = json.dumps([{"role": "title", key: box}])
    assert S.score_layout(pred, [], GOLD_BOX)["layout_f1"] == 1.0


def test_layout_pixels_need_page_size():
    target = {**GOLD_BOX, "size": [2000, 2800]}
    pred = json.dumps([{"role": "title", "bbox": [200, 560, 1000, 1120]}])
    assert S.score_layout(pred, [], target)["layout_f1"] == 1.0


def test_kv_wrapper_and_key_case():
    pred = json.dumps({"fields": {"Citizenship_Number": "४२-४२-२०५८-००५६३",
                                  "FULL_NAME": "सीता राम पौडेल", "child": ["राम", "श्याम"]}},
                      ensure_ascii=False)
    assert S.score_kv(pred, [], KV)["kv_f1"] == 1.0


# --- repeated field returned as one comma string (issue #10) -------------------------------

ADDR = {"fields": [["name", "राम बहादुर"], ["address", "ठमेल, काठमाडौं"], ["address", "बागमती"]]}


def test_kv_comma_string_keeps_gold_values_that_contain_a_comma():
    pred = json.dumps({"name": "राम बहादुर", "address": "ठमेल, काठमाडौं, बागमती"}, ensure_ascii=False)
    assert S.score_kv(pred, [], ADDR)["kv_f1"] == 1.0


def test_kv_comma_string_newline_separated_and_no_space_after_comma():
    for value in ("ठमेल, काठमाडौं\nबागमती", "ठमेल,काठमाडौं,बागमती"):
        pred = json.dumps({"name": "राम बहादुर", "address": value}, ensure_ascii=False)
        assert S.score_kv(pred, [], ADDR)["kv_f1"] == 1.0


def test_kv_comma_string_plain_split_still_works_and_wrong_values_still_count():
    pred = json.dumps({"child": "राम, श्याम, गोपाल"}, ensure_ascii=False)
    s = S.score_kv(pred, [], KV)
    assert s["kv_recall"] == 0.5 and s["kv_precision"] == pytest.approx(2 / 3)


def test_split_multi_prefers_the_longest_gold_rejoin():
    assert S._split_multi("क, ख, ग", ["क, ख", "ग"]) == ["क, ख", "ग"]
    assert S._split_multi("क, ख, ग", ["क", "ख", "ग"]) == ["क", "ख", "ग"]
    assert S._split_multi("क, ख, ग", []) == ["क", "ख", "ग"]


@pytest.mark.parametrize("reply", [
    "ANSWER NOT PRESENT",
    "उत्तर उपलब्ध छैन।",
    "The answer is not present in the document.",
    "I cannot find this.",
    "I can't find that on the page",
    "यो कागजातमा उल्लेख गरिएको छैन",
    "<think>looking…</think>फेला परेन",
])
def test_qa_loose_abstention_counts_refusals_in_other_words(reply):
    s = S.score_qa(reply, [], {"answerable": False})
    assert s["abstained_loose"] == 1.0
    strict = reply == "ANSWER NOT PRESENT"
    assert s["abstention_accuracy"] == float(strict) and s["qa_score"] == float(strict)


@pytest.mark.parametrize("reply", ["राम", "छैन", "Notary public", "रु. ५,०००", ""])
def test_qa_loose_abstention_does_not_count_an_answer(reply):
    s = S.score_qa(reply, [], {"answerable": False})
    assert s["abstained_loose"] == 0.0 and s["abstention_accuracy"] == 0.0


def test_qa_loose_abstention_not_applicable_to_answerable_questions():
    s = S.score_qa("I cannot find this.", ["राम"], {"answerable": True})
    assert s["abstained_loose"] is None
    assert s["false_abstention"] == 0.0  # the strict string decides false abstention


def test_nested_table_does_not_close_the_outer_table():
    html = ("<table><tr><td>क<table><tr><td>x</td><td>y</td></tr></table></td><td>ख</td></tr>"
            "<tr><td>ग</td><td>घ</td></tr></table>")
    table = S.parse_table(html)
    assert [[td.text for td in tr.children] for tr in table.children] == [
        ["क x y", "ख"], ["ग", "घ"]]


def test_reading_order_reports_akshara_accuracy_on_the_page_text():
    perfect = S.score_reading_order("\n".join(BLOCKS["blocks"]), [], BLOCKS)
    assert perfect["akshara_accuracy"] == 1.0 and perfect["acer"] == 0.0
    gold = {"blocks": ["किताब", "विद्यालय"]}
    s = S.score_reading_order("कताब\nविदयालय", [], gold)
    assert s["acer"] > s["cer"]  # vowel sign and halant errors weigh more per akshara
    assert s["akshara_accuracy"] == pytest.approx(1 - s["acer"])

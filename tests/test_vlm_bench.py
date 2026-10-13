"""NepalPixel VLM Bench: loading, and scoring exactly as the dataset's score_vlm_bench.py.

Expected values are worked by hand from that script's rules (revision 274aa27): NFC +
whitespace normalisation, no digit folding, ANLS τ = 0.5, page CER micro-averaged, a missing
prediction scored as an empty one.
"""

import pytest
from PIL import Image

from himalaya_vlm_eval import catalog
from himalaya_vlm_eval import structured as S
from himalaya_vlm_eval import vlm_bench as V
from himalaya_vlm_eval.benchmarks import NativeBenchmark
from himalaya_vlm_eval.models.base import Model, ModelError
from himalaya_vlm_eval.registry import register_adapter
from himalaya_vlm_eval.runner import evaluate, infer
from himalaya_vlm_eval.schema import ModelInfo
from himalaya_vlm_eval.types import Generation


@register_adapter("by_page")
class ByPage(Model):
    """Answers by the page it is shown, so the result does not depend on call order."""

    tasks = frozenset({"ocr", "vqa"})
    max_concurrency = 2

    def __init__(self, answers: dict[int, str | None]):
        self.answers = {int(k): v for k, v in answers.items()}

    def describe(self):
        return {"adapter": "by_page"}

    def generate(self, image, prompt):
        answer = self.answers[image.getpixel((0, 0))]
        if answer is None:
            raise ModelError("boom")
        return Generation(text=answer, latency_s=0.01, finish_reason="stop")


def page(shade: int) -> Image.Image:
    return Image.new("L", (8, 8), shade)


# --- scoring ----------------------------------------------------------------------------


def test_kie_is_anls_and_exact_match_with_digits_not_folded():
    assert V.score_kie("गाउँपालिकाको कार्यालय", ["गाउँपालिकाको कार्यालय"], None) == {
        "anls": 1.0, "exact_match": 1.0}
    # whitespace and NFC are normalised away …
    assert V.score_kie("  गाउँपालिकाको   कार्यालय ", ["गाउँपालिकाको कार्यालय"], None)[
        "exact_match"] == 1.0
    # … European digits for Devanagari ones are not: 4 of 4 characters differ
    assert V.score_kie("2049", ["२०४९"], None) == {"anls": 0.0, "exact_match": 0.0}
    # one edit in ten characters: ANLS 0.9; past half wrong, 0
    assert V.score_kie("abcdefghiX", ["abcdefghij"], None)["anls"] == pytest.approx(0.9)
    assert V.score_kie("abcdefXXXX", ["abcdefghij"], None)["anls"] == pytest.approx(0.6)
    assert V.score_kie("abcdXXXXXX", ["abcdefghij"], None)["anls"] == 0.0
    assert V.score_kie("", [""], None)["anls"] == 1.0


def test_page_cer_is_micro_averaged_over_pages():
    """Σ edits ÷ Σ characters, not the mean of per-page CERs."""
    a = V.score_ocr("abcd", ["abcdefgh"], None)  # 4 edits / 8 chars
    b = V.score_ocr("", ["xy"], None)            # 2 edits / 2 chars
    assert (a, b) == ({"edits": 4.0, "chars": 8.0}, {"edits": 2.0, "chars": 2.0})
    micro = (a["edits"] + b["edits"]) / (a["chars"] + b["chars"])
    assert micro == pytest.approx(0.6)  # the mean of 0.5 and 1.0 would be 0.75
    assert S.RATIOS["vlm_ocr"] == {"cer": ("edits", "chars")}


def test_abstention_and_leak():
    erased = {"erased_value": "वडा नं. ७"}
    assert V.score_abstain("answer not present", [], erased) == {"abstention": 1.0, "leak": 0.0}
    assert V.score_abstain(" ANSWER  NOT PRESENT ", [], erased)["abstention"] == 1.0
    # reproducing the erased value is a leak, close enough counts (ANLS > 0)
    assert V.score_abstain("वडा नं. ७", [], erased) == {"abstention": 0.0, "leak": 1.0}
    assert V.score_abstain("वडा नं ७", [], erased)["leak"] == 1.0
    # a different wrong answer neither abstains nor leaks; an empty one neither
    assert V.score_abstain("काठमाडौं", [], erased) == {"abstention": 0.0, "leak": 0.0}
    assert V.score_abstain("", [], erased) == {"abstention": 0.0, "leak": 0.0}


def test_a_failed_prediction_scores_as_an_empty_reply(monkeypatch):
    bench = NativeBenchmark(catalog.resolve_benchmark("nepalpixel-vlm-ocr"))
    assert bench.score(None, ["abc"]) == {"edits": 3.0, "chars": 3.0}
    kie = NativeBenchmark(catalog.resolve_benchmark("nepalpixel-vlm-kie"))
    assert kie.score(None, ["उत्तर"]) == {"anls": 0.0, "exact_match": 0.0}


# --- the catalog ------------------------------------------------------------------------


def test_three_benchmarks_on_existing_document_boards():
    b = catalog.benchmarks()
    assert b["nepalpixel-vlm-kie"].info.board == "document/nepali-qa"
    assert b["nepalpixel-vlm-abstain"].info.board == "document/nepali-qa"
    assert b["nepalpixel-vlm-ocr"].info.board == "document/nepali-page"
    assert not b["nepalpixel-vlm-ocr"].info.higher_is_better
    for bid in ("nepalpixel-vlm-kie", "nepalpixel-vlm-ocr", "nepalpixel-vlm-abstain"):
        spec = b[bid].spec["dataset"]
        assert spec["revision"] == "274aa2780b2bfc74d773efdf6d92651fdde3d69c"
        NativeBenchmark(b[bid])  # validates scorer, metrics, prompt
    # kie and abstain ask the same way, so abstaining is equally available on both
    assert b["nepalpixel-vlm-kie"].spec["prompt"] == b["nepalpixel-vlm-abstain"].spec["prompt"]


def test_nepali_ocr_is_the_first_vision_category():
    vision = catalog.layout().types[0]
    assert vision.id == "vision" and vision.categories[0].id == "nepali-ocr"


# --- loading ----------------------------------------------------------------------------


@pytest.fixture
def fake_hub(monkeypatch):
    """Three small configs in the dataset's shape, served in place of the Hub."""
    import datasets

    rows = {
        "kie": [
            {"id": "doc-a", "category": "land_property", "doc_type": "lalpurja",
             "image": page(10), "extra_pages": [page(20)],
             "qa": [{"question": "कसको नाममा?", "answer": "राम", "field": "owner", "pages": [1]},
                    {"question": "कित्ता नं?", "answer": "१२", "field": "plot", "pages": [2]},
                    {"question": "दुवै?", "answer": "x", "field": "both", "pages": [1, 2]}]},
            {"id": "doc-b", "category": "electoral", "doc_type": "voter_card",
             "image": page(30), "extra_pages": [],
             "qa": [{"question": "नाम?", "answer": "सीता", "field": "name", "pages": [1]}]},
        ],
        "ocr_layout": [
            {"id": "p1", "doc_id": "doc-a", "category": "land_property", "doc_type": "lalpurja",
             "text": "abcdefgh", "layout_graded": True, "image": page(40)},
            {"id": "p2", "doc_id": "doc-b", "category": "electoral", "doc_type": "voter_card",
             "text": "xy", "layout_graded": False, "image": page(50)},
        ],
        "abstain": [
            {"id": "doc-a_eb00", "positive_id": "doc-a", "category": "land_property",
             "doc_type": "lalpurja", "question": "कसको नाममा?", "answer": V.NOT_PRESENT,
             "field": "owner", "erased_value": "राम", "image": page(60)},
        ],
    }
    seen = []

    def load_dataset(repo, config, split, revision):
        seen.append((repo, config, split, revision))
        return datasets.Dataset.from_list(rows[config])

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return seen


def test_kie_questions_become_samples_with_their_pages(fake_hub):
    loaded = NativeBenchmark(catalog.resolve_benchmark("nepalpixel-vlm-kie")).load()
    by_id = {s.id: s for s in loaded.samples}
    assert sorted(by_id) == ["doc-a#00", "doc-a#01", "doc-a#02", "doc-b#00"]
    assert loaded.total == 2  # rows of the config: documents
    assert by_id["doc-a#00"].question == "कसको नाममा?" and by_id["doc-a#00"].references == ["राम"]
    assert by_id["doc-a#00"].load_image().getpixel((0, 0)) == 10       # page 1
    assert by_id["doc-a#01"].load_image().getpixel((0, 0)) == 20       # page 2: extra_pages[0]
    both = by_id["doc-a#02"].load_image()
    assert [im.getpixel((0, 0)) for im in both] == [10, 20]           # several pages: a list
    assert by_id["doc-a#00"].meta["doc_id"] == "doc-a"                 # bootstrap by document
    assert fake_hub[0][1:] == ("kie", "test", "274aa2780b2bfc74d773efdf6d92651fdde3d69c")


def test_limit_draws_documents_not_questions(fake_hub):
    loaded = NativeBenchmark(catalog.resolve_benchmark("nepalpixel-vlm-kie")).load(limit=1, seed=3)
    assert len({s.meta["doc_id"] for s in loaded.samples}) == 1


def test_abstain_carries_the_erased_value(fake_hub):
    (s,) = NativeBenchmark(catalog.resolve_benchmark("nepalpixel-vlm-abstain")).load().samples
    assert s.references == [V.NOT_PRESENT] and s.target == {"erased_value": "राम"}
    assert s.meta["doc_id"] == "doc-a"


def test_page_ocr_run_reports_micro_cer_with_breakdown(fake_hub, tmp_path):
    """End to end through the runner: the reported CER is Σ edits ÷ Σ chars, per category
    too, and a page the model failed on counts as an empty transcription."""
    # p1 (shade 40): 4 edits of 8 characters; p2 (shade 50): the model errors
    model = catalog.ModelEntry(ModelInfo(id="by-page", display_name="by-page"), "by_page",
                               {"answers": {40: "abcd", 50: None}})
    run_dir = infer(model, catalog.resolve_benchmark("nepalpixel-vlm-ocr"), tmp_path / "w")
    r = evaluate(run_dir)
    assert r.primary.value == pytest.approx((4 + 2) / (8 + 2))
    assert r.primary.ci_low is not None and r.errors == 1
    assert "edits" not in r.metrics and "chars" not in r.metrics
    cat = r.breakdowns["category"]
    assert cat["land_property"]["cer"] == pytest.approx(0.5)
    assert cat["electoral"]["cer"] == pytest.approx(1.0)

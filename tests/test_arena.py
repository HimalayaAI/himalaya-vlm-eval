"""Arena import, price matching, rank spread / Pareto / range filters, and the API surface
the studio's Benchmarks page reads. The HF dataset and OpenRouter are faked: rows have the
real dataset's columns (verified against lmarena-ai/leaderboard-dataset @ be06b33)."""

import json

import pytest
from fastapi.testclient import TestClient

from himalaya_vlm_eval import arena, meta
from himalaya_vlm_eval.api import create_app
from himalaya_vlm_eval.leaderboard import Filters, benchmark_board
from himalaya_vlm_eval.store import LocalStore, iter_results

REV = "be06b33784531833af86ad96817e12ba2a965b30"


def row(name, org, rating, lo, hi, votes, category="ocr", license_="Proprietary",
        date="2026-10-02"):
    return {"model_name": name, "organization": org, "license": license_, "rating": rating,
            "rating_lower": lo, "rating_upper": hi, "variance": 1.0, "vote_count": votes,
            "rank": 1, "category": category, "leaderboard_publish_date": date}


ROWS = {
    "vision_style_control": [
        row("claude-fable-5-high", "anthropic", 1326, 1318, 1334, 9815),
        row("qwen3.8-max", "alibaba", 1317, 1309, 1325, 6968),
        row("glm-5.3-flash", "zai", 1291, 1280, 1302, 800, license_="MIT"),
        row("cheap-open", "", 1150, 1140, 1160, 500, license_="-"),
        row("claude-fable-5-high", "anthropic", 1325, 1317, 1333, 13709, category="overall"),
        # a catalogued category whose newest rows are older than the arena's newest:
        # retired on arena.ai, so not imported
        row("old-model", "openai", 1200, 1190, 1210, 99, category="creative_writing_vision",
            date="2026-01-09"),
    ],
    "vision": [row("claude-fable-5-high", "anthropic", 1336, 1328, 1344, 9815)],
    "document_style_control": [
        row("claude-opus-5-high", "anthropic", 1498, 1490, 1506, 8497, category="overall")],
    "document": [],
}

OPENROUTER = [
    {"id": "anthropic/claude-fable-5", "context_length": 1000000,
     "pricing": {"prompt": "0.00001", "completion": "0.00005"}},
    {"id": "anthropic/claude-fable-5:batch", "context_length": 1,
     "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "z-ai/glm-5.3-flash", "context_length": 200000,
     "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}},
    {"id": "mistralai/mistral-small-2603", "context_length": 1,
     "pricing": {"prompt": "0.0000001", "completion": "0.0000001"}},
]


@pytest.fixture
def fake_hf(monkeypatch):
    calls = []

    def load(dataset, subset, revision, split="latest"):
        calls.append((subset, revision, split))
        return list(ROWS[subset])

    monkeypatch.setattr(arena, "load_subset", load)
    monkeypatch.setattr(arena, "resolve_revision", lambda d, r: REV)
    return calls


@pytest.fixture
def store(tmp_path, fake_hf):
    s = LocalStore(tmp_path / "store")
    counts = arena.import_arena(s)
    assert counts["retired"] == 1  # style-controlled creative writing (stale)
    return s


def test_import_builds_results_with_votes_ci_licence_and_attribution(store, fake_hf):
    rs = {r.run_id: r for r in iter_results(store)}
    r = rs["arena__arena-vision-ocr__claude-fable-5-high__20261002"]
    assert r.cases == 9815 and r.benchmark.cases_unit == "votes"
    assert (r.primary.value, r.primary.ci_low, r.primary.ci_high) == (1326, 1318, 1334)
    assert r.model.org == "Anthropic" and r.model.open_weights is False
    assert r.source.kind == "imported" and r.source.harness == "arena"
    assert r.source.data_license == "CC-BY-4.0" and "CC BY 4.0" in r.source.attribution
    assert r.config["revision"] == REV and r.config["subset"] == "vision_style_control"
    open_ = rs["arena__arena-vision-ocr__glm-5.3-flash__20261002"].model
    assert open_.open_weights is True and open_.license == "MIT" and open_.org == "Z.ai"
    unknown = rs["arena__arena-vision-ocr__cheap-open__20261002"].model
    assert unknown.open_weights is None and unknown.org == "Unknown"
    # the raw (no style control) variant is its own board with its own numbers
    raw = rs["arena__arena-vision-ocr-no-style-control__claude-fable-5-high__20261002"]
    assert raw.primary.value == 1336 and raw.benchmark.style_control is False
    assert not any("old-model" in k for k in rs)  # retired category skipped
    # each subset is downloaded once, at the pinned revision
    assert sorted({c[0] for c in fake_hf}) == sorted(ROWS) and {c[1] for c in fake_hf} == {REV}


def test_reimport_is_idempotent(store):
    again = arena.import_arena(store)
    assert again.get("published", 0) == 0 and again["unchanged"] > 0


def test_name_matching_never_crosses_releases():
    idx = meta.OpenRouterIndex(OPENROUTER)
    assert idx.match("claude-fable-5-high") == "anthropic/claude-fable-5"  # effort suffix
    assert idx.match("claude-fable-5-20260101") == "anthropic/claude-fable-5"  # snapshot date
    assert idx.match("glm-5.3-flash") == "z-ai/glm-5.3-flash"
    assert idx.match("mistral-small-2506") is None  # a different release, not a snapshot
    assert idx.match("claude-fable-5-preview") is None  # a preview is not the GA model
    assert "anthropic/claude-fable-5:batch" not in idx.by_id
    e = idx.entry("anthropic/claude-fable-5")
    assert (e["input_price"], e["output_price"], e["context_length"]) == (10, 50, 1000000)


def test_overrides_win_and_can_forbid_a_price(store, monkeypatch):
    models = [r.model for r in iter_results(store)]
    monkeypatch.setattr(meta, "load_overrides", lambda dirs=None: {
        "qwen3.8-max": {"input_price": 2, "output_price": 6, "context_length": 1000000},
        "glm-5.3-flash": {"pricing": "none"},
    })
    summary = meta.refresh(store, models, openrouter_models=OPENROUTER)
    data = meta.read(store)
    assert data["qwen3.8-max"]["input_price"] == 2 and data["qwen3.8-max"]["source"] == "catalog"
    assert "glm-5.3-flash" not in data and "glm-5.3-flash" in summary["missing"]
    assert data["claude-fable-5-high"]["source"] == "openrouter:anthropic/claude-fable-5"


@pytest.fixture
def priced(store, monkeypatch):
    monkeypatch.setattr(meta, "load_overrides", lambda dirs=None: {
        "qwen3.8-max": {"input_price": 2, "output_price": 6, "context_length": 1000000}})
    meta.refresh(store, [r.model for r in iter_results(store)], openrouter_models=OPENROUTER)
    return store


def test_rank_spread_matches_arena(priced):
    rows = benchmark_board(list(iter_results(priced)), "arena-vision-ocr",
                           Filters(meta=meta.read(priced)))
    d = {r.result.model.id: r.to_dict() for r in rows}
    # fable and qwen overlap: both can be first, and either can be second
    assert (d["claude-fable-5-high"]["rank"], d["claude-fable-5-high"]["rank_worst"]) == (1, 2)
    assert (d["qwen3.8-max"]["rank"], d["qwen3.8-max"]["rank_worst"]) == (1, 2)
    assert (d["glm-5.3-flash"]["rank"], d["glm-5.3-flash"]["rank_worst"]) == (3, 3)
    assert d["claude-fable-5-high"]["score_100"] is None  # a rating has no 0–100 mapping


def test_pareto_frontier_and_blended_price(priced):
    rows = benchmark_board(list(iter_results(priced)), "arena-vision-ocr",
                           Filters(meta=meta.read(priced)))
    d = {r.result.model.id: r.to_dict() for r in rows}
    assert d["glm-5.3-flash"]["pricing"]["blended"] == pytest.approx((3 * 0.1 + 0.4) / 4)
    assert d["claude-fable-5-high"]["pricing"]["blended"] == pytest.approx(20)
    # cheapest → glm; qwen beats it for more; fable beats qwen for more still
    assert {k for k, v in d.items() if v["pareto"]} == {
        "glm-5.3-flash", "qwen3.8-max", "claude-fable-5-high"}
    assert d["cheap-open"]["pricing"] is None and not d["cheap-open"]["pareto"]


def test_range_filters_drop_unknowns_only_when_set(priced):
    results = list(iter_results(priced))
    m = meta.read(priced)
    all_rows = benchmark_board(results, "arena-vision-ocr", Filters(meta=m))
    assert len(all_rows) == 4
    cheap = benchmark_board(results, "arena-vision-ocr",
                            Filters(meta=m, input_price=(None, 3)))
    assert {r.result.model.id for r in cheap} == {"qwen3.8-max", "glm-5.3-flash"}
    top = benchmark_board(results, "arena-vision-ocr", Filters(meta=m, score=(1300, None)))
    assert {r.result.model.id for r in top} == {"claude-fable-5-high", "qwen3.8-max"}
    long_ctx = benchmark_board(results, "arena-vision-ocr",
                               Filters(meta=m, context_length=(500000, None)))
    assert {r.result.model.id for r in long_ctx} == {"claude-fable-5-high", "qwen3.8-max"}
    open_ = benchmark_board(results, "arena-vision-ocr", Filters(meta=m, open_weights=True))
    assert [r.result.model.id for r in open_] == ["glm-5.3-flash"]


def test_api_boards_layout_and_board_payload(priced):
    with TestClient(create_app(priced, refresh_seconds=0)) as c:
        types = c.get("/v1/boards").json()["types"]
        assert [t["id"] for t in types] == ["vision", "document"]
        vision = {cat["id"]: cat for cat in types[0]["categories"]}
        assert list(vision)[:10] == ["overall", "english", "chinese", "captioning",
                                     "creative-writing", "diagram", "entity-recognition",
                                     "homework", "humor", "ocr"]
        assert vision["nepali-ocr"]["label"] == "Nepali OCR"
        ocr = vision["ocr"]["boards"]
        assert ocr[0]["benchmark"] == "arena-vision-ocr" and ocr[0]["style_control"] is True
        assert ocr[1]["style_control"] is False and ocr[0]["models"] == 4
        assert "ocrbench" in {b["benchmark"] for b in ocr}  # measured suites share the category
        doc = {cat["id"]: cat for cat in types[1]["categories"]}
        assert doc["nepali-fields"]["boards"][0]["benchmark"] == "nepalipixel-docs-kv"

        body = c.get("/v1/leaderboard/arena-vision-ocr",
                     params={"input_price_max": 3}).json()
        assert {r["model"]["id"] for r in body["data"]} == {"qwen3.8-max", "glm-5.3-flash"}
        assert body["bounds"]["input_price"] == [0.1, 10]  # extents ignore the range filter
        assert body["bounds"]["score"] == [1150, 1326]
        assert body["data_license"] == "CC-BY-4.0" and "Arena" in body["attribution"]
        assert body["benchmark"]["cases_unit"] == "votes"
        assert body["meta_generated_at"]

        models = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
        assert models["claude-fable-5-high"]["pricing"]["input"] == 10
        assert models["cheap-open"]["pricing"] is None
        assert c.get("/health").json()["meta_models"] >= 3
        # arena ratings never enter the 0–100 overview
        assert c.get("/v1/leaderboard").json()["benchmarks"] == []


def test_api_survives_a_broken_meta_document(priced):
    priced.put(meta.META_KEY, b"{not json", "application/json")
    with TestClient(create_app(priced, refresh_seconds=0)) as c:
        h = c.get("/health").json()
        assert h["ok"] and h["meta_models"] == 0
        assert len(c.get("/v1/leaderboard/arena-vision-ocr").json()["data"]) == 4


def test_cli_import_arena_dry_run(fake_hf, capsys, tmp_path):
    from himalaya_vlm_eval.cli import main

    assert main(["import-arena", "--arena", "document", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["published"] == 1 and out["revision"] == REV

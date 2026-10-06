from nepeval_ocr.leaderboard import Filters, benchmark_board, overview, representatives

from .conftest import result


def test_arena_style_ranks_share_when_intervals_overlap():
    rs = [result("a", "b", 80, ci=(78, 82)), result("b", "b", 79, ci=(77, 81)),
          result("c", "b", 60, ci=(58, 62))]
    rows = benchmark_board(rs, "b")
    assert [(r.result.model.id, r.rank, r.position) for r in rows] == [
        ("a", 1, 1), ("b", 1, 2), ("c", 3, 3)]


def test_lower_is_better_ordering():
    rs = [result("a", "cer", 0.10, ci=(0.09, 0.11), higher=False, scale=1.0),
          result("b", "cer", 0.30, ci=(0.28, 0.32), higher=False, scale=1.0)]
    rows = benchmark_board(rs, "cer")
    assert [r.result.model.id for r in rows] == ["a", "b"]
    assert rows[1].rank == 2
    assert rows[0].score_100 == 90


def test_point_estimates_without_ci_and_ties():
    rs = [result("a", "b", 50), result("b", "b", 50), result("c", "b", 40)]
    rows = benchmark_board(rs, "b")
    assert [r.position for r in rows] == [1, 1, 3]
    assert [r.rank for r in rows] == [1, 1, 3]


def test_representative_prefers_measured_full_then_newest():
    imported = result("a", "b", 90, kind="imported")
    subset = result("a", "b", 70, cases=10, days_ago=0)
    full_old = result("a", "b", 60, days_ago=5)
    full_new = result("a", "b", 65, days_ago=1)
    reps = representatives([imported, subset, full_old, full_new])
    assert reps[("b", "a")] is full_new
    reps = representatives([imported, subset])
    assert reps[("b", "a")] is subset
    reps = representatives([imported, subset], Filters(include_subsets=False))
    assert reps[("b", "a")] is imported
    assert representatives([imported], Filters(include_imported=False)) == {}


def test_filters():
    rs = [result("a", "b", 1, model_kind="ocr_engine", open_weights=True, org="X"),
          result("c", "b", 2, open_weights=False, org="Y")]
    assert [r.result.model.id for r in benchmark_board(rs, "b", Filters(kind="ocr_engine"))] == ["a"]
    assert [r.result.model.id for r in benchmark_board(rs, "b", Filters(open_weights=False))] == ["c"]
    assert [r.result.model.id for r in benchmark_board(rs, "b", Filters(orgs={"y"}))] == ["c"]
    assert benchmark_board(rs, "missing") == []


def test_overview_ranks_complete_coverage_first():
    rs = [
        result("full", "x", 60), result("full", "y", 60),
        result("partial", "x", 99),
        result("lower", "cer", 0.2, higher=False, scale=1.0, category="ocr"),
        result("full", "cer", 0.1, higher=False, scale=1.0),
    ]
    selected, rows = overview(rs, ["x", "y", "cer"])
    assert selected == ["x", "y", "cer"]
    assert rows[0].model.id == "full" and rows[0].covered == 3
    assert rows[0].avg_score_100 == (60 + 60 + 90) / 3
    assert [r.model.id for r in rows[1:]] == ["partial", "lower"]  # 1 each, by score
    d = rows[0].to_dict()
    assert d["complete"] and set(d["benchmarks"]) == {"x", "y", "cer"}


def test_overview_ignores_unknown_selection():
    selected, rows = overview([result("a", "x", 1)], ["x", "nope"])
    assert selected == ["x"] and len(rows) == 1


def test_results_from_an_older_definition_are_reexpressed_or_dropped():
    from nepeval_ocr.schema import MetricValue

    old_cer = result("old", "b", 0.9, higher=False, scale=1.0)  # headline was CER (m)
    old_cer.metrics["acc"] = MetricValue(value=0.4)  # but it also recorded accuracy
    old_only_cer = result("older", "b", 0.2, higher=False, scale=1.0, days_ago=3)
    new = result("new", "b", 0.92, scale=1.0)
    new = new.model_copy(update={
        "benchmark": new.benchmark.model_copy(update={"primary_metric": "acc"}),
        "metrics": {"acc": MetricValue(value=0.92, ci_low=0.9, ci_high=0.94)}})
    rows = benchmark_board([old_cer, old_only_cer, new], "b", definition=new.benchmark)
    assert [(r.result.model.id, round(r.result.primary.value, 2)) for r in rows] == [
        ("new", 0.92), ("old", 0.4)]
    bumped = new.benchmark.model_copy(update={"version": "2"})
    assert benchmark_board([old_cer, new], "b", definition=bumped) == []

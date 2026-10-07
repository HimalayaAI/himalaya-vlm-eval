import pytest
from fastapi.testclient import TestClient

from himalaya_vlm_eval import catalog
from himalaya_vlm_eval.api import create_app
from himalaya_vlm_eval.publish import publish_run
from himalaya_vlm_eval.runner import evaluate, infer
from himalaya_vlm_eval.store import LocalStore, publish_result

from .conftest import result as _result
from .test_runner import _answers, entry


def result(model, bench, value, **kw):
    """A fixture result on a real catalog benchmark, under its real definition."""
    from himalaya_vlm_eval.schema import MetricValue

    r = _result(model, bench, value, **kw)
    info = catalog.benchmarks()[bench].info
    return r.model_copy(update={"benchmark": info,
                                "metrics": {info.primary_metric: MetricValue(value=value)}})


@pytest.fixture
def store(tmp_path, manifest_catalog):
    s = LocalStore(tmp_path / "store")
    for r in [result("gpt-4o", "ocrbench", 805, kind="imported", org="OpenAI",
                     open_weights=False),
              result("qwen", "ocrbench", 700, open_weights=True),
              result("qwen", "mmstar", 60),
              result("gpt-4o", "mmstar", 70, org="OpenAI")]:
        publish_result(s, r)
    _answers(manifest_catalog["texts"])
    run_dir = infer(entry(concurrency=1), catalog.resolve_benchmark("local-ocr"), tmp_path / "w")
    evaluate(run_dir)
    publish_run(run_dir, s)
    return s


def client(store, **kw):
    return TestClient(create_app(store, refresh_seconds=0, **kw))


def test_health_and_lists(store):
    with client(store) as c:
        h = c.get("/health").json()
        assert h["ok"] and h["runs"] == 5 and h["invalid"] == 0
        benches = {b["id"]: b for b in c.get("/v1/benchmarks").json()["data"]}
        assert benches["ocrbench"]["models"] == 2 and benches["ocrbench"]["engine"] == "vlmevalkit"
        assert benches["local-ocr"]["models"] == 1
        assert benches["mathvista-mini"]["models"] == 0  # catalog-only, no results yet
        assert all(b["category"] == "math" for b in
                   c.get("/v1/benchmarks", params={"category": "math"}).json()["data"])
        models = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
        assert models["gpt-4o"]["benchmarks"] == ["mmstar", "ocrbench"]


def test_benchmark_leaderboard_and_filters(store):
    with client(store) as c:
        body = c.get("/v1/leaderboard/ocrbench").json()
        assert body["benchmark"]["scale_max"] == 1000
        assert [(r["model"]["id"], r["rank"]) for r in body["data"]] == [("gpt-4o", 1), ("qwen", 2)]
        assert body["data"][0]["source"]["kind"] == "imported"
        only_measured = c.get("/v1/leaderboard/ocrbench", params={"include_imported": False}).json()
        assert [r["model"]["id"] for r in only_measured["data"]] == ["qwen"]
        open_only = c.get("/v1/leaderboard/ocrbench", params={"open_weights": True}).json()
        assert [r["model"]["id"] for r in open_only["data"]] == ["qwen"]
        assert c.get("/v1/leaderboard/mathvista-mini").json()["data"] == []
        assert c.get("/v1/leaderboard/nope").status_code == 404


def test_overview(store):
    with client(store) as c:
        body = c.get("/v1/leaderboard", params={"benchmarks": "ocrbench,mmstar"}).json()
        assert body["benchmarks"] == ["ocrbench", "mmstar"]
        assert body["data"][0]["model"]["id"] == "gpt-4o" and body["data"][0]["complete"]
        cat = c.get("/v1/leaderboard", params={"category": "general"}).json()
        assert cat["benchmarks"] == ["mmstar"]
        assert c.get("/v1/leaderboard", params={"category": "math"}).json()["data"] == []


def test_runs_and_samples(store):
    with client(store) as c:
        runs = c.get("/v1/runs", params={"benchmark": "local-ocr"}).json()
        assert runs["total"] == 1
        run_id = runs["data"][0]["run_id"]
        full = c.get(f"/v1/runs/{run_id}").json()
        assert full["breakdowns"]["level"]["line"]["n"] == 3
        page = c.get(f"/v1/runs/{run_id}/samples", params={"limit": 2, "sort": "cer"}).json()
        assert page["total"] == 5 and len(page["data"]) == 2
        grouped = c.get(f"/v1/runs/{run_id}/samples",
                        params={"group": "level", "value": "word"}).json()
        assert grouped["total"] == 2
        assert c.get(f"/v1/runs/{run_id}/samples", params={"sort": "nope"}).status_code == 422
        imported = c.get("/v1/runs", params={"source": "imported"}).json()["data"][0]["run_id"]
        assert c.get(f"/v1/runs/{imported}/samples").status_code == 404
        assert c.get("/v1/runs/missing").status_code == 404


def test_etag_conditional_get(store):
    with client(store) as c:
        r1 = c.get("/v1/leaderboard/ocrbench")
        tag = r1.headers["etag"]
        r2 = c.get("/v1/leaderboard/ocrbench", headers={"If-None-Match": tag})
        assert r2.status_code == 304
        publish_result(store, result("new", "ocrbench", 100))
        c.post("/v1/admin/refresh")
        r3 = c.get("/v1/leaderboard/ocrbench", headers={"If-None-Match": tag})
        assert r3.status_code == 200 and len(r3.json()["data"]) == 3


def test_bad_files_do_not_break_the_board(store):
    store.put("runs/broken/result.json", b"{not json", "application/json")
    store.put("runs/mismatch/result.json",
              result("x", "ocrbench", 1).model_dump_json().encode(), "application/json")
    with client(store) as c:
        h = c.get("/health").json()
        assert h["invalid"] == 2 and h["runs"] == 5
        assert len(c.get("/v1/leaderboard/ocrbench").json()["data"]) == 2


def test_deleted_results_disappear(store):
    with client(store) as c:
        assert len(c.get("/v1/leaderboard/mmstar").json()["data"]) == 2
        run_id = c.get("/v1/runs", params={"benchmark": "mmstar", "model": "qwen"}).json()[
            "data"][0]["run_id"]
        (store.root / "runs" / run_id / "result.json").unlink()
        c.post("/v1/admin/refresh")
        assert len(c.get("/v1/leaderboard/mmstar").json()["data"]) == 1


def test_token_auth(store):
    with client(store, token="s3cret") as c:
        assert c.get("/v1/benchmarks").status_code == 401
        assert c.get("/v1/benchmarks", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/benchmarks",
                     headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200  # health stays open for probes


def test_store_failure_reports_unhealthy(tmp_path):
    class Broken(LocalStore):
        def list(self, prefix):
            raise OSError("bucket gone")

    with client(Broken(tmp_path)) as c:
        h = c.get("/health")
        assert h.status_code == 503 and "bucket gone" in h.json()["last_error"]

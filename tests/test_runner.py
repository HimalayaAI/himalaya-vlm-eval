import json
import threading

import pytest

from nepeval_ocr import catalog
from nepeval_ocr.models.base import FatalModelError, Model, ModelError
from nepeval_ocr.registry import register_adapter
from nepeval_ocr.runner import (
    PREDICTIONS,
    RESULT,
    SCORES,
    RunAborted,
    RunOptions,
    Unsupported,
    evaluate,
    infer,
    latest_records,
    load_result,
)
from nepeval_ocr.schema import ModelInfo
from nepeval_ocr.types import Generation

BEHAVIOUR: dict = {}


@register_adapter("fake")
class FakeModel(Model):
    """Answers from a lookup keyed by prompt+call count; behaviour set per test."""

    def __init__(self, mode: str = "perfect", concurrency: int = 4):
        self.mode = mode
        self.max_concurrency = concurrency
        self.calls = 0
        self.lock = threading.Lock()

    def describe(self):
        return {"adapter": "fake", "mode": self.mode}

    def generate(self, image, prompt):
        with self.lock:
            self.calls += 1
            n = self.calls
        answer = BEHAVIOUR["answers"].pop(0) if BEHAVIOUR.get("answers") else "x"
        if self.mode == "fatal":
            raise FatalModelError("bad key")
        if self.mode == "always_error" or (self.mode == "error_second" and n == 2):
            raise ModelError("boom")
        if self.mode == "crash_after_2" and n > 2:
            raise KeyboardInterrupt
        return Generation(text=answer, latency_s=0.01, finish_reason="stop",
                          usage={"total_tokens": 7})


def entry(mode="perfect", **params):
    return catalog.ModelEntry(ModelInfo(id=f"fake-{mode}", display_name="Fake"), "fake",
                              {"mode": mode, **params})


@pytest.fixture
def bench(manifest_catalog):
    return catalog.resolve_benchmark("local-ocr")


def _answers(texts):
    BEHAVIOUR["answers"] = list(texts)


def test_full_run_perfect_model(tmp_path, bench, manifest_catalog):
    _answers(manifest_catalog["texts"])
    run_dir = infer(entry(concurrency=1), bench, tmp_path)
    r = evaluate(run_dir)
    assert r.cases == 5 and r.errors == 0
    assert r.metrics["cer"].value == 0.0 and r.metrics["exact_match"].value == 1.0
    assert r.metrics["cer"].ci_low == 0.0 == r.metrics["cer"].ci_high
    assert set(r.breakdowns["level"]) == {"word", "line"}
    assert r.breakdowns["level"]["line"]["n"] == 3
    assert r.stats["usage"]["total_tokens"] == 35 and r.stats["finish_reasons"] == {"stop": 5}
    assert r.config["dataset"]["rows"] == 5 and r.config["config_hash"]
    assert r.has_samples and (run_dir / SCORES).exists()
    assert load_result(run_dir) == r


def test_resume_retries_only_failures(tmp_path, bench):
    _answers([])
    run_dir = infer(entry("error_second", concurrency=1), bench, tmp_path)
    recs = latest_records(run_dir / PREDICTIONS)
    assert sum(not r["ok"] for r in recs.values()) == 1
    # Same config → same directory; only the failed sample is re-run.
    run_dir2 = infer(entry("error_second", concurrency=1), bench, tmp_path)
    assert run_dir2 == run_dir
    lines = (run_dir / PREDICTIONS).read_text().splitlines()
    assert len(lines) == 6
    assert all(r["ok"] for r in latest_records(run_dir / PREDICTIONS).values())


def test_errors_count_as_worst_case(tmp_path, bench, manifest_catalog):
    _answers(manifest_catalog["texts"])
    run_dir = infer(entry("error_second", concurrency=1), bench, tmp_path,
                    options=RunOptions(retry_errors=False))
    r = evaluate(run_dir)
    assert r.errors == 1
    assert r.metrics["cer"].value == pytest.approx(1 / 5)
    assert r.stats["top_errors"][0]["error"] == "boom"


def test_different_params_get_a_different_run_dir(tmp_path, bench):
    a = infer(entry(concurrency=1), bench, tmp_path, options=RunOptions(limit=3))
    b = infer(entry(concurrency=1), bench, tmp_path, options=RunOptions(limit=4))
    assert a != b
    r = evaluate(a)
    assert r.cases == 3 and r.is_subset


def test_subset_is_seeded_and_not_the_head(bench):
    from nepeval_ocr.benchmarks import NativeBenchmark

    nb = NativeBenchmark(bench)
    ids1 = [s.id for s in nb.load(3, seed=1).samples]
    assert ids1 == [s.id for s in nb.load(3, seed=1).samples]
    assert any(ids1 != [s.id for s in nb.load(3, seed=k).samples] for k in range(2, 6))


def test_preflight_failure_aborts_without_fanning_out(tmp_path, bench):
    with pytest.raises(RunAborted, match="pre-flight"):
        infer(entry("always_error"), bench, tmp_path)


def test_fatal_error_stops_run(tmp_path, bench):
    with pytest.raises(FatalModelError):
        infer(entry("fatal"), bench, tmp_path)


def test_consecutive_errors_abort(tmp_path, bench, monkeypatch):
    import nepeval_ocr.runner as R

    monkeypatch.setattr(R, "_preflight", lambda *a, **k: None)
    with pytest.raises(RunAborted, match="consecutive"):
        infer(entry("always_error", concurrency=1), bench, tmp_path,
              options=RunOptions(max_consecutive_errors=3))


def test_interrupt_keeps_completed_predictions(tmp_path, bench):
    with pytest.raises(KeyboardInterrupt):
        infer(entry("crash_after_2", concurrency=1), bench, tmp_path)
    run_dir = next((tmp_path / "runs").iterdir())
    assert len(latest_records(run_dir / PREDICTIONS)) == 2
    with pytest.raises(RunAborted, match="only 2/5"):
        evaluate(run_dir)


def test_torn_last_line_is_tolerated(tmp_path, bench):
    run_dir = infer(entry(concurrency=1), bench, tmp_path)
    with (run_dir / PREDICTIONS).open("a") as fh:
        fh.write('{"sample_id": "s1", "ok": tr')
    assert len(latest_records(run_dir / PREDICTIONS)) == 5


def test_ocr_engine_refuses_vqa(tmp_path):
    from nepeval_ocr.models.ocr_engines import TesseractModel  # noqa: F401

    tess = catalog.models()["tesseract-nep"]
    with pytest.raises(Unsupported):
        infer(tess, catalog.resolve_benchmark("nepalipixel-docs-qa"), tmp_path)


def test_rescoring_uses_current_catalog(tmp_path, bench, manifest_catalog):
    _answers(["नमस्कार।", *manifest_catalog["texts"][1:]])
    run_dir = infer(entry(concurrency=1), bench, tmp_path)
    first = evaluate(run_dir)
    assert first.metrics["loose_match"].value == 1.0
    assert first.metrics["exact_match"].value == pytest.approx(0.8)
    data = json.loads((run_dir / RESULT).read_text())
    assert data["benchmark"]["id"] == "local-ocr"


def test_evaluate_is_idempotent_and_tracks_scoring_changes(tmp_path, bench, manifest_catalog):
    _answers(manifest_catalog["texts"])
    run_dir = infer(entry(concurrency=1), bench, tmp_path)
    first = evaluate(run_dir)
    assert evaluate(run_dir) == first  # unchanged predictions + scoring → same result
    changed = catalog.BenchmarkEntry(
        bench.info, bench.engine, {**bench.spec, "metrics": ["cer", "wer"]}, bench.path)
    second = evaluate(run_dir, changed)
    assert second.run_id != first.run_id and "exact_match" not in second.metrics


def test_resume_preflights_on_a_fresh_sample(tmp_path, bench, monkeypatch):
    import nepeval_ocr.runner as R

    run_dir = infer(entry(concurrency=1), bench, tmp_path, options=RunOptions(limit=3))
    # Mark the first sample failed and drop the rest: resume must not preflight on s-failed.
    recs = list(latest_records(run_dir / PREDICTIONS).values())
    failed = dict(recs[0], ok=False, error="corrupt image")
    (run_dir / PREDICTIONS).write_text(json.dumps(failed) + "\n")
    seen = []
    real = R._preflight
    monkeypatch.setattr(R, "_preflight", lambda m, b, s, o: (seen.append(s.id), real(m, b, s, o)))
    infer(entry(concurrency=1), bench, tmp_path, options=RunOptions(limit=3))
    assert seen and seen[0] != failed["sample_id"]
    assert all(r["ok"] for r in latest_records(run_dir / PREDICTIONS).values())


def test_cli_concurrency_works_for_engines_without_that_param(tmp_path, manifest_catalog,
                                                             caplog, monkeypatch):
    from nepeval_ocr.cli import main
    from nepeval_ocr.models import ocr_engines

    monkeypatch.setattr(ocr_engines.TesseractModel, "setup", lambda self: None)
    monkeypatch.setattr(ocr_engines.TesseractModel, "_recognize", lambda self, img: "नमस्कार")
    rc = main(["run", "--model", "tesseract-nep", "--bench", "local-ocr", "--concurrency", "2",
               "--work-dir", str(tmp_path), "--no-publish"])
    assert rc == 0, caplog.text
    assert "bad params" not in caplog.text

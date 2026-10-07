from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from nepeval_ocr import catalog
from nepeval_ocr.benchmarks import NativeBenchmark
from nepeval_ocr.models.openai_compat import OpenAICompatModel, TarkaOCRModel
from nepeval_ocr.schema import BenchmarkInfo, MetricValue, ModelInfo, RunResult, SourceInfo

from .conftest import result


def _base(**over):
    kw = {"run_id": "a__b", "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
          "model": ModelInfo(id="m", display_name="M"),
          "benchmark": BenchmarkInfo(id="b", display_name="B", category="ocr",
                                     primary_metric="cer"),
          "source": SourceInfo(kind="measured", harness="t"), "cases": 10,
          "metrics": {"cer": MetricValue(value=0.1)}}
    kw.update(over)
    return kw


def test_runresult_validation():
    RunResult(**_base())
    with pytest.raises(ValidationError, match="primary metric"):
        RunResult(**_base(metrics={"wer": MetricValue(value=0.1)}))
    with pytest.raises(ValidationError, match="lowercase"):
        RunResult(**_base(run_id="Bad ID"))
    with pytest.raises(ValidationError, match="timezone"):
        RunResult(**_base(created_at=datetime(2026, 1, 1)))
    with pytest.raises(ValidationError, match="errors cannot exceed"):
        RunResult(**_base(errors=11))
    with pytest.raises(ValidationError, match="schema_version"):
        RunResult(**_base(schema_version=99))
    with pytest.raises(ValidationError):
        MetricValue(value=1, ci_low=2, ci_high=1)
    with pytest.raises(ValidationError):
        MetricValue(value=1, ci_low=0)


def test_score_100_and_subset():
    lower = BenchmarkInfo(id="b", display_name="B", category="ocr", primary_metric="cer",
                          higher_is_better=False, scale_max=1.0)
    assert lower.score_100(0.1) == pytest.approx(90)
    assert lower.score_100(3.0) == 0  # CER above 1 clamps
    assert result("m", "b", 1, cases=50, full=100).is_subset
    assert not result("m", "b", 1, cases=100, full=100).is_subset
    assert not result("m", "b", 1, cases=5, full=None).is_subset


def test_builtin_catalog_loads_and_is_consistent():
    models = catalog.models()
    benches = catalog.benchmarks()
    assert len(models) >= 30 and len(benches) >= 35
    for b in benches.values():
        if b.engine == "native":
            NativeBenchmark(b)  # validates metrics/scorer/prompt
        else:
            spec = b.spec["vlmeval"]
            assert spec["dataset"] and spec["files"] and spec["judge"] in {"none", "extract",
                                                                            "required"}
            assert ("json_key" in spec) or ("column" in spec) or ("sum" in spec)
    for e in models.values():
        if e.adapter in ("openai_compat", "tarka_ocr"):
            m = e.build()
            assert isinstance(m, OpenAICompatModel)
            assert "key" not in str(m.describe()).lower() or "api_key" not in m.describe()
    assert isinstance(models["glm-ocr-nepali"].build(), TarkaOCRModel)


def test_categories_cover_the_requested_families():
    cats = {b.info.category for b in catalog.benchmarks().values()}
    assert {"ocr", "document", "chart", "math", "chat", "general", "hallucination"} <= cats


def test_adhoc_models():
    e = catalog.resolve_model("openrouter:qwen/qwen3-vl-8b-instruct")
    assert e.info.id == "qwen-qwen3-vl-8b-instruct" and e.info.org == "Alibaba"
    assert e.params["base_url"] == "https://openrouter.ai/api/v1"
    v = catalog.resolve_model("vllm:Qwen/Qwen2.5-VL-3B-Instruct")
    assert v.params["base_url"].startswith("http://localhost")
    with pytest.raises(KeyError, match="unknown model"):
        catalog.resolve_model("gpt-4")
    with pytest.raises(KeyError):
        catalog.resolve_model("nosuch:thing")


def test_select_benchmarks():
    ids = {b.info.id for b in catalog.select_benchmarks("category:math,ocrbench")}
    assert "ocrbench" in ids and "mathvista-mini" in ids
    assert {b.info.id for b in catalog.select_benchmarks("NepaliPixel")} == {"nepalipixel"}
    assert len(catalog.select_benchmarks("all")) == len(catalog.benchmarks())
    with pytest.raises(KeyError):
        catalog.select_benchmarks("category:nope")


def test_user_catalog_overrides_and_bad_entries(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yaml").write_text(
        "- id: gpt-4o\n  display_name: Overridden\n  adapter: openai_compat\n"
        "  params: {model: m, base_url: http://x}\n")
    monkeypatch.setenv("NEPEVAL_CATALOG", str(tmp_path))
    catalog.reload()
    assert catalog.models()["gpt-4o"].info.display_name == "Overridden"
    (tmp_path / "models" / "bad.yaml").write_text("- id: Bad\n  display_name: x\n  adapter: a\n")
    catalog.reload()
    with pytest.raises(ValueError, match=r"bad\.yaml"):
        catalog.models()


def test_bad_model_params_are_reported():
    e = catalog.ModelEntry(ModelInfo(id="x", display_name="x"), "openai_compat",
                           {"model": "m", "base_url": "http://x", "nope": 1})
    with pytest.raises(ValueError, match="bad params"):
        e.build()

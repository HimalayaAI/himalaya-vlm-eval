"""The VLMEvalKit engine against a fake VLMEvalKit checkout that mimics its CLI and
output layout (status.json, TSV predictions, per-dataset score files)."""

import json
import logging
import os
import sys
import textwrap
import time

import pytest

from himalaya_vlm_eval import catalog
from himalaya_vlm_eval.engines import vlmevalkit as V

FAKE_RUN = textwrap.dedent(r'''
    import argparse, json, os, sys, time
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--work-dir"); p.add_argument("--mode")
    p.add_argument("--api-nproc"); p.add_argument("--reuse", action="store_true")
    p.add_argument("--judge"); p.add_argument("--judge-api-nproc")
    a = p.parse_args()
    cfg = json.load(open(a.config))
    alias = next(iter(cfg["model"])); ds = next(iter(cfg["data"]))
    run = os.path.join(a.work_dir, alias, "T20261006-%06d" % (time.time_ns() % 1000000))
    os.makedirs(run, exist_ok=True)
    json.dump({"argv": sys.argv, "env": {k: os.environ.get(k) for k in
               ["OPENAI_API_KEY", "OPENAI_API_BASE", "LOCAL_LLM", "PRED_FORMAT", "MMEVAL_ROOT"]},
               "config": cfg}, open(os.path.join(a.work_dir, "invocation.json"), "w"))
    if os.environ.get("FAKE_FAIL"):
        print("Traceback: kaboom"); sys.exit(3)
    P = f"{alias}_{ds}"
    with open(os.path.join(run, P + ".tsv"), "w") as f:
        f.write('"index"\t"prediction"\n"0"\t"A"\n"1"\t"Failed to obtain answer via API."\n"2"\t"B"\n"3"\t"C"\n')
    files = {
      "OCRBench": (P + "_score.json", json.dumps({"Final Score": 812, "Final Score Norm": 81.2})),
      "MMStar": (P + "_acc.csv", '"split","Overall","coarse perception"\n"none","0.6413","0.7"\n'),
      "MathVista_MINI": (P + "_gpt-4o-mini_score.csv",
          '"Task&Skill","tot","acc"\n"Overall","1000","63.2"\n"FQA","200","60"\n'),
      "MME": (P + "_score.csv", '"perception","reasoning"\n"1600.5","500.25"\n'),
      "WeMath": (P + "_gpt4o-mini_score.csv", '"Score (Strict)","Score (Loose)"\n"45.33%","60%"\n'),
      "CCOCR": (P + "_acc.csv", '"lan_ocr","total"\n"0.7","0.655"\n'),
      "MMMU_DEV_VAL": (P + "_acc.csv", '"split","Overall"\n"dev","0.5"\n"validation","0.552"\n'),
    }
    name, body = files[ds]
    open(os.path.join(run, name), "w").write(body)
    json.dump({"schema_version": "1.0", "eval_id": os.path.basename(run), "commit": "54a063c5",
               "datasets": {ds: {"status": "done", "metrics": {"Overall": 1.0, "x": "str"}}}},
              open(os.path.join(run, "status.json"), "w"))
''')

FAKE_DATASETS = textwrap.dedent('''
    class ImageMCQDataset:
        @classmethod
        def supported_datasets(cls): return ["MMStar", "MMMU_DEV_VAL"]
    class OCRBench:
        @classmethod
        def supported_datasets(cls): return ["OCRBench"]
    class Other:
        @classmethod
        def supported_datasets(cls): return ["MathVista_MINI", "MME", "WeMath", "CCOCR"]
    DATASET_CLASSES = [ImageMCQDataset, OCRBench, Other]
''')


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    repo = tmp_path / "VLMEvalKit"
    (repo / "vlmeval" / "dataset").mkdir(parents=True)
    (repo / "vlmeval" / "__init__.py").write_text("")
    (repo / "vlmeval" / "dataset" / "__init__.py").write_text(FAKE_DATASETS)
    (repo / "run.py").write_text(FAKE_RUN)
    monkeypatch.setenv("VLMEVALKIT_DIR", str(repo))
    monkeypatch.setenv("VLMEVALKIT_PYTHON", sys.executable)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-model")
    monkeypatch.setenv("MMEVAL_ROOT", "/should/be/removed")
    return repo


def _run(bench_id, tmp_path, **settings_over):
    settings = V.VLMEvalSettings.from_env()
    for k, v in settings_over.items():
        setattr(settings, k, v)
    settings.check()
    return V.run_benchmark(catalog.models()["qwen3-vl-8b-instruct"],
                           catalog.resolve_benchmark(bench_id), tmp_path / "work", settings)


@pytest.mark.parametrize("bench,score", [
    ("ocrbench", 812.0), ("mmstar", 64.13), ("mme", 2100.75), ("wemath", 45.33),
    ("ccocr", 65.5), ("mmmu-dev-val", 55.2),
])
def test_scores_parsed_onto_catalog_scale(fake_repo, tmp_path, bench, score):
    r = _run(bench, tmp_path)
    assert r.primary.value == pytest.approx(score)
    assert r.source.harness == "vlmevalkit" and r.source.harness_version == "54a063c5"
    assert r.cases == 4 and r.errors == 1
    assert r.stats["vlmevalkit_metrics"] == {"Overall": 1.0}


def test_judge_and_secrets_handling(fake_repo, tmp_path):
    r = _run("mathvista-mini", tmp_path, judge_key="sk-judge", judge_model="openai/gpt-4o-mini",
             judge_base_url="https://openrouter.ai/api/v1")
    assert r.primary.value == pytest.approx(63.2) and r.source.judge == "openai/gpt-4o-mini"
    inv = json.loads((tmp_path / "work" / "vlmevalkit" / "invocation.json").read_text())
    assert "sk-or-model" not in " ".join(inv["argv"]) and "sk-judge" not in " ".join(inv["argv"])
    assert inv["argv"][inv["argv"].index("--judge") + 1] == "gpt-4o-mini"
    assert inv["env"]["LOCAL_LLM"] == "openai/gpt-4o-mini"
    assert inv["env"]["OPENAI_API_KEY"] == "sk-judge"
    assert inv["env"]["OPENAI_API_BASE"] == "https://openrouter.ai/api/v1/chat/completions"
    assert inv["env"]["PRED_FORMAT"] == "tsv" and inv["env"]["MMEVAL_ROOT"] is None
    model_cfg = inv["config"]["model"]["qwen3-vl-8b-instruct"]
    assert model_cfg["class"] == "GPT4V" and model_cfg["model"] == "qwen/qwen3-vl-8b-instruct"
    assert model_cfg["api_base"] == "https://openrouter.ai/api/v1/chat/completions"
    assert inv["config"]["data"]["MathVista_MINI"]["class"] == "Other"
    leftovers = list((tmp_path / "work" / "vlmevalkit").glob(".config-*"))
    assert leftovers == []  # the keyed config file is deleted


def test_judge_rules():
    s = V.VLMEvalSettings(repo=None, python="x", judge_key=None)
    assert V._judge_plan(s, "none", "ocrbench") == ([], {}, None)
    assert V._judge_plan(s, "extract", "mmstar")[0] == ["--judge", "exact_matching"]
    with pytest.raises(V.VLMEvalError, match="needs an LLM judge"):
        V._judge_plan(s, "required", "mmvet")
    s.judge_key, s.judge_model = "k", "openai/gpt-4o-mini"
    with pytest.raises(V.VLMEvalError, match="CharXiv"):
        V._judge_plan(s, "required", "charxiv-reasoning-val")
    s.judge_model = "gpt-4o-mini"
    args, env, _ = V._judge_plan(s, "required", "charxiv-reasoning-val")
    assert args[:2] == ["--judge", "gpt-4o-mini"] and "LOCAL_LLM" not in env


def test_refuses_dotenv_and_missing_checkout(fake_repo, tmp_path, monkeypatch):
    (fake_repo / ".env").write_text("OPENAI_API_KEY=x")
    with pytest.raises(V.VLMEvalError, match=r"\.env"):
        V.VLMEvalSettings.from_env().check()
    monkeypatch.setenv("VLMEVALKIT_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(V.VLMEvalError, match="not a VLMEvalKit checkout"):
        V.VLMEvalSettings.from_env().check()
    monkeypatch.delenv("VLMEVALKIT_DIR")
    with pytest.raises(V.VLMEvalError, match="VLMEVALKIT_DIR"):
        V.VLMEvalSettings.from_env()


def test_unsupported_dataset_and_failure_surface(fake_repo, tmp_path, monkeypatch):
    s = V.VLMEvalSettings.from_env()
    with pytest.raises(V.VLMEvalError, match="does not support"):
        V.resolve_dataset_classes(s, ["NoSuchSet"])
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(V.VLMEvalError, match="kaboom"):
        _run("ocrbench", tmp_path)


def test_rejects_non_chat_adapters(fake_repo, tmp_path):
    from himalaya_vlm_eval.runner import Unsupported

    with pytest.raises(Unsupported, match="chat-completions"):
        V.run_benchmark(catalog.models()["glm-ocr-nepali"], catalog.resolve_benchmark("ocrbench"),
                        tmp_path, V.VLMEvalSettings.from_env(), dataset_class="OCRBench")


def test_parse_helpers(tmp_path):
    (tmp_path / "m_X_acc.csv").write_text('"split","Overall"\n"a","1"\n"Overall","0.9"\n')
    v, _ = V.parse_score(tmp_path, {"files": ["{P}_acc.csv"], "column": "overall"}, "m_X")
    assert v == 0.9  # case-insensitive column, falls back to the Overall row
    with pytest.raises(V.VLMEvalError, match="no result file"):
        V.parse_score(tmp_path, {"files": ["{P}_nope.csv"], "column": "x"}, "m_X")
    with pytest.raises(ValueError, match="no row"):
        V.parse_score(tmp_path, {"files": ["{P}_acc.csv"], "column": "Overall",
                                 "row": {"split": "zzz"}}, "m_X")
    # glob metacharacters in the model alias must not break file lookup
    (tmp_path / "m[1]_X_acc.csv").write_text('"Overall"\n"0.5"\n')
    assert V.parse_score(tmp_path, {"files": ["{P}_acc.csv"], "column": "Overall"}, "m[1]_X")[0] == 0.5


def test_streaming_logs_heartbeats_and_stops_the_child(tmp_path, caplog, monkeypatch):
    s = V.VLMEvalSettings(repo=tmp_path, python=sys.executable)
    log_path = tmp_path / "out.log"
    script = ("import time\nfor i in range(3):\n"
              "    print(f'Infer {i}/3', flush=True)\n    time.sleep(0.2)")
    with caplog.at_level(logging.INFO, logger="himeval"):
        rc = V._run_streaming([sys.executable, "-c", script], s, {}, log_path, label="m × d",
                              heartbeat_s=0.1)
    assert rc == 0 and "Infer 2/3" in log_path.read_text()
    assert "vlmevalkit m × d running" in caplog.text

    # An interrupt in the parent terminates the child instead of orphaning it.
    pid_file = tmp_path / "pid"
    child = f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\n" \
            "print('started', flush=True)\ntime.sleep(60)"
    real_log = V.log.info

    def interrupt_on_heartbeat(msg, *args):
        real_log(msg, *args)
        if "running" in msg:
            raise KeyboardInterrupt

    monkeypatch.setattr(V.log, "info", interrupt_on_heartbeat)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        V._run_streaming([sys.executable, "-c", child], s, {}, log_path, label="x",
                         heartbeat_s=0.0)
    assert time.monotonic() - started < 30
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

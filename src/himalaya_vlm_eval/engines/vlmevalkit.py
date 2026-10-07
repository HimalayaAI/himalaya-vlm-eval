"""Drive VLMEvalKit as a subprocess and turn its outputs into RunResults.

VLMEvalKit lives in its own checkout and environment (it pins torch and friends), so it
is never imported here:

    git clone https://github.com/open-compass/VLMEvalKit ~/src/VLMEvalKit
    cd ~/src/VLMEvalKit && python -m venv .venv && .venv/bin/pip install -e .
    export VLMEVALKIT_DIR=~/src/VLMEvalKit VLMEVALKIT_PYTHON=~/src/VLMEvalKit/.venv/bin/python

Judge (for math/chat/chart suites; MCQ suites fall back to exact matching without one):
    HIMEVAL_JUDGE_MODEL     default gpt-4o-mini  (OpenRouter-style ids with "/" work via LOCAL_LLM)
    HIMEVAL_JUDGE_BASE_URL  default https://api.openai.com/v1
    HIMEVAL_JUDGE_API_KEY   falls back to OPENAI_API_KEY

Guard rails, from reading VLMEvalKit's source (docs/VLMEVALKIT.md):
- a `.env` in the checkout silently overrides our environment → refused;
- keys passed on argv are persisted into status.json → keys go only into a 0600 config file
  that is deleted after the run, and into the child's environment;
- its "primary metric" chooser is wrong for several suites → scores are read from the
  result files named in the catalog, never from that chooser.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..catalog import BenchmarkEntry, ModelEntry
from ..models.openai_compat import OpenAICompatModel, TarkaOCRModel
from ..runner import Unsupported
from ..schema import MetricValue, RunResult, SourceInfo, utcnow

log = logging.getLogger("himeval")

FAIL_PREFIX = "Failed to obtain answer"


class VLMEvalError(RuntimeError):
    pass


@dataclass
class VLMEvalSettings:
    repo: Path
    python: str
    api_nproc: int = 16
    judge_model: str = "gpt-4o-mini"
    judge_base_url: str = "https://api.openai.com/v1"
    judge_key: str | None = None
    judge_nproc: int = 8
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> VLMEvalSettings:
        repo = os.environ.get("VLMEVALKIT_DIR")
        if not repo:
            raise VLMEvalError(
                "VLMEVALKIT_DIR is not set. Clone https://github.com/open-compass/VLMEvalKit, "
                "install it in its own venv, and set VLMEVALKIT_DIR / VLMEVALKIT_PYTHON."
            )
        return cls(
            repo=Path(repo).expanduser(),
            python=os.environ.get("VLMEVALKIT_PYTHON", sys.executable),
            api_nproc=int(os.environ.get("VLMEVALKIT_API_NPROC", "16")),
            judge_model=os.environ.get("HIMEVAL_JUDGE_MODEL", "gpt-4o-mini"),
            judge_base_url=os.environ.get("HIMEVAL_JUDGE_BASE_URL", "https://api.openai.com/v1"),
            judge_key=os.environ.get("HIMEVAL_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )

    def check(self) -> None:
        if not (self.repo / "run.py").is_file():
            raise VLMEvalError(f"{self.repo} is not a VLMEvalKit checkout (no run.py)")
        if (self.repo / ".env").exists():
            raise VLMEvalError(
                f"{self.repo / '.env'} exists. VLMEvalKit loads it over the process environment "
                "(keys, judge endpoint), which would silently change what is evaluated. Remove it."
            )


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _judge_plan(settings: VLMEvalSettings, judge_use: str, bench_id: str
                ) -> tuple[list[str], dict[str, str], str | None]:
    """CLI args, env and the recorded judge name for one benchmark."""
    if judge_use == "none":
        return [], {}, None
    if not settings.judge_key:
        if judge_use == "required":
            raise VLMEvalError(
                f"{bench_id} needs an LLM judge: set HIMEVAL_JUDGE_API_KEY (or OPENAI_API_KEY)"
            )
        return ["--judge", "exact_matching"], {}, "exact_matching"
    env = {"OPENAI_API_KEY": settings.judge_key,
           "OPENAI_API_BASE": _chat_url(settings.judge_base_url)}
    name = settings.judge_model
    if "/" in name:
        # Judge names end up in file paths; a slash breaks them. VLMEvalKit's documented
        # workaround: a slash-free alias on the CLI, the real id via LOCAL_LLM.
        if bench_id.startswith("charxiv"):
            raise VLMEvalError(
                "CharXiv names its files after the judge's real model id, so a judge id with '/' "
                "breaks it. Use a slash-free judge (e.g. gpt-4o-mini on api.openai.com)."
            )
        env["LOCAL_LLM"] = name
        alias = name.rsplit("/", 1)[-1]
        return ["--judge", alias, "--judge-api-nproc", str(settings.judge_nproc)], env, name
    return ["--judge", name, "--judge-api-nproc", str(settings.judge_nproc)], env, name


def resolve_dataset_classes(settings: VLMEvalSettings, names: list[str]) -> dict[str, str]:
    """Ask VLMEvalKit (in its own interpreter) which dataset class serves each name."""
    code = (
        "import json,sys\n"
        "from vlmeval.dataset import DATASET_CLASSES\n"
        "out={}\n"
        "for n in json.loads(sys.argv[1]):\n"
        "    for c in DATASET_CLASSES:\n"
        "        if n in c.supported_datasets():\n"
        "            out[n]=c.__name__; break\n"
        "print('HIMEVAL_CLASSES='+json.dumps(out))\n"
    )
    proc = subprocess.run([settings.python, "-c", code, json.dumps(names)], cwd=settings.repo,
                          capture_output=True, text=True, env=_child_env(settings, {}))
    for line in proc.stdout.splitlines():
        if line.startswith("HIMEVAL_CLASSES="):
            found = json.loads(line.split("=", 1)[1])
            missing = [n for n in names if n not in found]
            if missing:
                raise VLMEvalError(f"VLMEvalKit does not support datasets {missing}")
            return found
    raise VLMEvalError(f"could not import vlmeval with {settings.python}: "
                       f"{(proc.stderr or proc.stdout)[-1500:]}")


def _child_env(settings: VLMEvalSettings, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    # Variables that would redirect output or swap keys/endpoints behind our back.
    for var in ("MMEVAL_ROOT", "OPENAI_API_KEY", "OPENAI_API_BASE", "LOCAL_LLM",
                "LMDEPLOY_API_KEY", "LMDEPLOY_API_BASE", "PRED_FORMAT", "EVAL_FORMAT"):
        env.pop(var, None)
    env.update({"PRED_FORMAT": "tsv", "EVAL_FORMAT": "csv", "PYTHONUNBUFFERED": "1"})
    env.update(settings.extra_env)
    env.update(extra)
    return env


def _model_config(model: OpenAICompatModel) -> dict[str, Any]:
    key = model._api_key()
    if model.api_key_env and not key:
        raise VLMEvalError(f"no API key for {model.model}: set one of {model.api_key_env}")
    cfg: dict[str, Any] = {
        "class": "GPT4V",
        "model": model.model,
        "api_base": _chat_url(model.base_url),
        "key": key or "EMPTY",
        "max_tokens": model.max_tokens,
        "temperature": model.temperature if model.temperature is not None else 0,
        "img_detail": "high",
        "timeout": int(model.timeout),
        "retry": model.retries,
        "verbose": False,
    }
    cfg.update(model.extra_body)
    return cfg


# --- result parsing ------------------------------------------------------------------------


def _read_table(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delim = "\t" if path.suffix == ".tsv" or sample.count("\t") > sample.count(",") else ","
        return list(csv.DictReader(fh, delimiter=delim))


def _num(value: Any, percent: bool = False) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if percent or text.endswith("%"):
        return float(text.rstrip("%"))
    return float(text)


def _col(row: dict[str, str], name: str) -> str:
    if name in row:
        return row[name]
    lowered = {k.lower().strip(): k for k in row}
    if name.lower() in lowered:
        return row[lowered[name.lower()]]
    raise KeyError(f"column {name!r} not in {list(row)}")


def _pick_row(rows: list[dict[str, str]], want: dict[str, Any] | None) -> dict[str, str]:
    if not rows:
        raise ValueError("empty result table")
    if want:
        hits = [r for r in rows if all(str(_col(r, k)).strip() == str(v) for k, v in want.items())]
        if not hits:
            raise ValueError(f"no row matching {want} in {len(rows)} rows")
        return hits[0]
    if len(rows) == 1:
        return rows[0]
    for r in rows:  # MCQ tables: split = none / Overall
        if str(r.get("split", "")).strip() in ("none", "Overall", "overall"):
            return r
    return rows[0]


def find_result_file(run_dir: Path, patterns: list[str], prefix: str) -> Path:
    for pattern in patterns:
        hits = sorted(glob.glob(str(run_dir / pattern.replace("{P}", glob.escape(prefix)))),
                      key=os.path.getmtime, reverse=True)
        if hits:
            return Path(hits[0])
    raise VLMEvalError(f"no result file matching {patterns} in {run_dir}")


def parse_score(run_dir: Path, spec: dict[str, Any], prefix: str) -> tuple[float, Path]:
    path = find_result_file(run_dir, spec["files"], prefix)
    if path.suffix == ".json":
        data = json.loads(path.read_text("utf-8"))
        raw = _num(data[spec["json_key"]], spec.get("percent", False))
    else:
        row = _pick_row(_read_table(path), spec.get("row"))
        if "sum" in spec:
            raw = sum(_num(_col(row, c)) for c in spec["sum"])
        else:
            raw = _num(_col(row, spec["column"]), spec.get("percent", False))
    mult = spec.get("multiply", 1)
    if mult == "auto":
        mult = 100 if raw <= 1.0 else 1
    return raw * float(mult), path


def count_predictions(run_dir: Path, prefix: str) -> tuple[int, int]:
    """(cases, failed) from the prediction file VLMEvalKit wrote (PRED_FORMAT=tsv)."""
    pred = run_dir / f"{prefix}.tsv"
    if not pred.exists():
        return 0, 0
    rows = _read_table(pred)
    failed = sum(1 for r in rows if str(r.get("prediction", "")).startswith(FAIL_PREFIX))
    return len(rows), failed


def latest_run_dir(work_dir: Path, alias: str) -> Path:
    runs = sorted((work_dir / alias).glob("T*/status.json"), key=os.path.getmtime)
    if not runs:
        raise VLMEvalError(f"VLMEvalKit wrote no run under {work_dir / alias}")
    return runs[-1].parent


# --- running ------------------------------------------------------------------------------


def run_benchmark(model_entry: ModelEntry, bench: BenchmarkEntry, work_dir: Path,
                  settings: VLMEvalSettings, *, model_overrides: dict[str, Any] | None = None,
                  dataset_class: str | None = None) -> RunResult:
    spec = bench.spec["vlmeval"]
    model = model_entry.build(model_overrides)
    if not isinstance(model, OpenAICompatModel) or isinstance(model, TarkaOCRModel):
        raise Unsupported(
            f"{model_entry.info.id} uses the {model_entry.adapter!r} adapter; VLMEvalKit "
            "benchmarks need a chat-completions endpoint (openai_compat)"
        )
    dataset = spec["dataset"]
    alias = model_entry.info.id
    vl_work = (work_dir / "vlmevalkit").resolve()
    vl_work.mkdir(parents=True, exist_ok=True)
    cls = dataset_class or resolve_dataset_classes(settings, [dataset])[dataset]
    judge_args, judge_env, judge_name = _judge_plan(settings, spec.get("judge", "none"),
                                                    bench.info.id)

    cfg_path = vl_work / f".config-{alias}-{dataset}.json"
    cfg = {"model": {alias: _model_config(model)},
           "data": {dataset: {"class": cls, "dataset": dataset}}}
    fd = os.open(cfg_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    cmd = [settings.python, "run.py", "--config", str(cfg_path), "--work-dir", str(vl_work),
           "--mode", "all", "--api-nproc", str(settings.api_nproc), "--reuse", *judge_args]
    log_path = vl_work / f"{alias}__{dataset}.log"
    log.info("vlmevalkit: %s on %s (judge: %s) → %s", alias, dataset, judge_name, log_path)
    try:
        with log_path.open("a", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=settings.repo, env=_child_env(settings, judge_env),
                                  stdout=logf, stderr=subprocess.STDOUT)
    finally:
        cfg_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = log_path.read_text("utf-8", errors="replace")[-2000:]
        raise VLMEvalError(f"VLMEvalKit exited {proc.returncode} on {dataset}:\n{tail}")

    run_dir = latest_run_dir(vl_work, alias)
    status = json.loads((run_dir / "status.json").read_text("utf-8"))
    entry = (status.get("datasets") or {}).get(dataset, {})
    if entry.get("status") not in (None, "done"):
        raise VLMEvalError(f"{dataset}: VLMEvalKit status {entry.get('status')!r}: "
                           f"{entry.get('error_message') or entry.get('skip_reason')}")
    prefix = f"{alias}_{dataset}"
    score, score_file = parse_score(run_dir, spec, prefix)
    cases, failed = count_predictions(run_dir, prefix)
    info = bench.info
    if not 0 <= score <= info.scale_max * 1.0001:
        raise VLMEvalError(f"{bench.info.id}: parsed score {score} outside 0–{info.scale_max}; "
                           f"check the catalog spec against {score_file}")

    eval_id = str(status.get("eval_id", run_dir.name))
    return RunResult(
        run_id="__".join([alias, info.id, "vlmevalkit", re.sub(r"[^a-z0-9]+", "-",
                                                                eval_id.lower()).strip("-")]),
        created_at=utcnow(),
        model=model_entry.info,
        benchmark=info,
        source=SourceInfo(kind="measured", harness="vlmevalkit",
                          harness_version=str(status.get("commit") or "") or None,
                          judge=judge_name),
        cases=cases or (info.full_cases or 0),
        errors=min(failed, cases) if cases else 0,
        metrics={info.primary_metric: MetricValue(value=score)},
        config={"harness": "vlmevalkit", "dataset": dataset, "dataset_class": cls,
                "model": model.describe(), "judge": judge_name,
                "score_file": score_file.name, "eval_id": eval_id},
        stats={"vlmevalkit_metrics": {k: v for k, v in (entry.get("metrics") or {}).items()
                                      if isinstance(v, (int, float))}},
    )

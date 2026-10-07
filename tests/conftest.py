from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from himalaya_vlm_eval import catalog
from himalaya_vlm_eval.schema import BenchmarkInfo, MetricValue, ModelInfo, RunResult, SourceInfo

Handler = Callable[[str, dict[str, Any]], tuple[int, Any, dict[str, str]]]


class MockOpenAI:
    """A real HTTP server speaking enough of the OpenAI API for adapter and CLI tests."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.handler: Handler = self.default
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                path = self.path.removeprefix("/v1")
                outer.requests.append((path, body, dict(self.headers)))
                status, payload, headers = outer.handler(path, body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def chat(text: str, finish: str = "stop") -> dict[str, Any]:
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    def default(self, path: str, body: dict[str, Any]) -> tuple[int, Any, dict[str, str]]:
        return 200, self.chat("ok"), {}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def mock_openai():
    server = MockOpenAI()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    for var in ("HIMEVAL_STORE", "HIMEVAL_CATALOG", "HIMEVAL_API_TOKEN", "OPENAI_API_KEY",
                "OPENROUTER_API_KEY", "TARKA_API_KEY", "HIMEVAL_JUDGE_API_KEY",
                "VLMEVALKIT_DIR", "VLMEVALKIT_PYTHON", "NEPALIPIXEL_DOCS_DIR"):
        monkeypatch.delenv(var, raising=False)
    catalog.reload()
    yield
    catalog.reload()


def make_image(path: Path, text: str = "x", size: tuple[int, int] = (64, 32)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


@pytest.fixture
def manifest_catalog(tmp_path, monkeypatch):
    """A local OCR benchmark (5 images) registered through HIMEVAL_CATALOG."""
    data = tmp_path / "data"
    rows = []
    texts = ["नमस्कार", "नेपाल सरकार", "काठमाडौं", "क़लम", "धन्यवाद।"]
    for i, t in enumerate(texts):
        make_image(data / f"img/{i}.png")
        rows.append({"id": f"s{i}", "image": f"img/{i}.png", "text": t,
                     "level": "word" if i % 2 else "line"})
    (data / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    cat = tmp_path / "catalog"
    (cat / "benchmarks").mkdir(parents=True)
    (cat / "models").mkdir(parents=True)
    (cat / "benchmarks" / "local.yaml").write_text(
        f"""
id: local-ocr
engine: native
display_name: Local OCR
category: ocr
language: ne
primary_metric: cer
higher_is_better: false
scale_max: 1.0
full_cases: 5
task: ocr
dataset:
  source: local
  path: {data / 'manifest.jsonl'}
  meta: [level]
prompt: Transcribe.
metrics: [cer, wer, exact_match, loose_match]
breakdowns: [level]
""", "utf-8")
    monkeypatch.setenv("HIMEVAL_CATALOG", str(cat))
    catalog.reload()
    return {"texts": texts, "dir": data, "catalog": cat}


def result(model: str, bench: str, value: float, *, ci: tuple[float, float] | None = None,
           higher: bool = True, kind: str = "measured", cases: int = 100,
           full: int | None = 100, days_ago: int = 0, category: str = "ocr",
           scale: float = 100.0, org: str = "Org", model_kind: str = "vlm",
           open_weights: bool | None = None) -> RunResult:
    created = datetime(2026, 10, 1, tzinfo=timezone.utc) - timedelta(days=days_ago)
    return RunResult(
        run_id=f"{model}__{bench}__{kind}__{days_ago}",
        created_at=created,
        model=ModelInfo(id=model, display_name=model, org=org, kind=model_kind,
                        open_weights=open_weights),
        benchmark=BenchmarkInfo(id=bench, display_name=bench, category=category,
                                primary_metric="m", higher_is_better=higher,
                                scale_max=scale, full_cases=full),
        source=SourceInfo(kind=kind, harness="test"),
        cases=cases,
        metrics={"m": MetricValue(value=value, ci_low=ci[0] if ci else None,
                                  ci_high=ci[1] if ci else None)},
    )

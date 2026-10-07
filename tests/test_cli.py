"""End to end through the CLI: a real HTTP model endpoint (mock), a local benchmark,
auto-publish to a store, then the API reading what was published."""

import json

from fastapi.testclient import TestClient

from himalaya_vlm_eval.api import create_app
from himalaya_vlm_eval.cli import main
from himalaya_vlm_eval.store import LocalStore

from .conftest import MockOpenAI


def test_run_publishes_and_api_serves(tmp_path, manifest_catalog, mock_openai, monkeypatch,
                                      capsys):
    calls = iter(range(100))

    def handler(path, body):
        next(calls)
        return 200, MockOpenAI.chat("नमस्कार"), {}

    mock_openai.handler = handler
    monkeypatch.setenv("VLLM_BASE_URL", mock_openai.url)
    store_dir = tmp_path / "store"
    rc = main(["run", "--model", "vllm:local/vision", "--bench", "local-ocr",
               "--work-dir", str(tmp_path / "w"), "--store", str(store_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "published" in out and "local-vision" in out
    assert len(mock_openai.requests) == 5

    with TestClient(create_app(LocalStore(store_dir), refresh_seconds=0)) as c:
        board = c.get("/v1/leaderboard/local-ocr").json()["data"]
        assert board[0]["model"]["id"] == "local-vision"
        assert board[0]["cases"] == 5 and board[0]["metrics"]["exact_match"] == 0.2

    # Re-running is a no-op resume and an idempotent publish.
    rc = main(["run", "--model", "vllm:local/vision", "--bench", "local-ocr",
               "--work-dir", str(tmp_path / "w"), "--store", str(store_dir)])
    assert rc == 0 and len(mock_openai.requests) == 5


def test_run_matrix_continues_past_failures(tmp_path, manifest_catalog, mock_openai, monkeypatch,
                                            capsys):
    mock_openai.handler = lambda p, b: (401, {"error": "bad key"}, {})
    monkeypatch.setenv("VLLM_BASE_URL", mock_openai.url)
    rc = main(["run", "--model", "vllm:a,tesseract-nep", "--bench",
               "local-ocr,nepalipixel-docs-qa", "--work-dir", str(tmp_path / "w"), "--no-publish"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "failed" in out and "skipped" in out  # tesseract cannot do QA → skipped, not failed


def test_list_and_import_and_leaderboard(tmp_path, capsys):
    assert main(["list", "benchmarks", "--category", "math", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and all(r["category"] == "math" for r in rows)
    f = tmp_path / "imp.json"
    f.write_text(json.dumps([{"model": {"id": "gpt-4o", "display_name": "GPT-4o", "org": "OpenAI"},
                              "benchmark": "mmstar", "score": 64.7, "date": "2025-09-01"}]))
    store = str(tmp_path / "s")
    assert main(["import", str(f), "--store", store, "--harness", "opencompass-openvlm"]) == 0
    capsys.readouterr()
    assert main(["leaderboard", "mmstar", "--store", store]) == 0
    assert "gpt-4o" in capsys.readouterr().out
    assert main(["leaderboard", "mmstar", "--store", store, "--measured-only"]) == 1
    assert main(["import", "--format-help"]) == 0


def test_docs_without_data_are_skipped_not_failed(tmp_path, capsys, mock_openai, monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", mock_openai.url)
    rc = main(["run", "--model", "vllm:a", "--bench", "nepalipixel-docs-kv",
               "--work-dir", str(tmp_path), "--no-publish"])
    assert rc == 0 and "skipped (no data)" in capsys.readouterr().out

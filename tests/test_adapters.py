import pytest
from PIL import Image

from nepeval_ocr.models.base import FatalModelError, ModelError
from nepeval_ocr.models.openai_compat import OpenAICompatModel, TarkaOCRModel, encode_image
from nepeval_ocr.types import Prompt

from .conftest import MockOpenAI

IMG = Image.new("RGB", (40, 20), "white")
PROMPT = Prompt("read it", system="sys")


def _model(server, **kw):
    kw.setdefault("retries", 3)
    m = OpenAICompatModel("vision-1", server.url, "TEST_KEY", **kw)
    m._sleep = lambda attempt, retry_after: None  # no real backoff in tests
    return m


def test_request_shape_and_auth(mock_openai, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    m = _model(mock_openai, temperature=0.0, max_tokens=99, extra_body={"top_p": 1})
    gen = m.generate(IMG, PROMPT)
    assert gen.text == "ok" and gen.usage["total_tokens"] == 15 and gen.finish_reason == "stop"
    path, body, headers = mock_openai.requests[0]
    assert path == "/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "vision-1" and body["max_tokens"] == 99 and body["top_p"] == 1
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    content = body["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "read it"}
    assert "sk-test" not in str(m.describe())


def test_missing_key_is_fatal(mock_openai):
    m = _model(mock_openai)
    with pytest.raises(FatalModelError, match="TEST_KEY"):
        m.setup()


def test_no_auth_endpoint(mock_openai):
    m = OpenAICompatModel("v", mock_openai.url, None)
    assert m.generate(IMG, PROMPT).text == "ok"
    assert "Authorization" not in mock_openai.requests[0][2]


def test_retries_transient_failures(mock_openai, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    replies = iter([(429, {"error": "slow down"}, {"retry-after": "0"}),
                    (503, {"error": "busy"}, {}),
                    (200, {"error": {"message": "upstream"}}, {}),
                    (200, MockOpenAI.chat("नमस्कार"), {})])
    mock_openai.handler = lambda p, b: next(replies)
    gen = _model(mock_openai).generate(IMG, PROMPT)
    assert gen.text == "नमस्कार" and gen.extra["attempts"] == 4


def test_gives_up_after_retries(mock_openai, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    mock_openai.handler = lambda p, b: (500, {"error": "boom"}, {})
    with pytest.raises(ModelError, match="gave up after 4"):
        _model(mock_openai).generate(IMG, PROMPT)
    assert len(mock_openai.requests) == 4


@pytest.mark.parametrize("status,exc", [(401, FatalModelError), (404, FatalModelError),
                                        (400, ModelError)])
def test_client_errors_do_not_retry(mock_openai, monkeypatch, status, exc):
    monkeypatch.setenv("TEST_KEY", "k")
    mock_openai.handler = lambda p, b: (status, {"error": "no"}, {})
    with pytest.raises(exc):
        _model(mock_openai).generate(IMG, PROMPT)
    assert len(mock_openai.requests) == 1


def test_content_parts_and_null_content(mock_openai, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    mock_openai.handler = lambda p, b: (200, {"choices": [{"message": {"content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}, {})
    assert _model(mock_openai).generate(IMG, PROMPT).text == "ab"
    mock_openai.handler = lambda p, b: (200, {"choices": [{"message": {"content": None},
                                                           "finish_reason": "content_filter"}]}, {})
    g = _model(mock_openai).generate(IMG, PROMPT)
    assert g.text == "" and g.finish_reason == "content_filter"


def test_reasoning_model_params(mock_openai, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    _model(mock_openai, temperature=None, max_tokens_field="max_completion_tokens").generate(
        IMG, PROMPT)
    body = mock_openai.requests[0][1]
    assert "temperature" not in body and body["max_completion_tokens"] == 4096


def test_tarka_ocr_endpoint(mock_openai, monkeypatch):
    monkeypatch.setenv("TARKA_API_KEY", "tk")
    mock_openai.handler = lambda p, b: (200, {"text": "नेपाल<|endoftext|>",
                                              "usage": {"total_tokens": 3}}, {})
    m = TarkaOCRModel("glm-ocr-nepali", base_url=mock_openai.url)
    m._sleep = lambda *a: None
    gen = m.generate(IMG, Prompt("Transcribe."))
    path, body, _ = mock_openai.requests[0]
    assert path == "/ocr" and body["model"] == "glm-ocr-nepali" and body["prompt"] == "Transcribe."
    assert body["image"].startswith("data:image/png;base64,")
    assert gen.text == "नेपाल<|endoftext|>"  # stripped at scoring by clean_model_output
    mock_openai.handler = lambda p, b: (200, {"unexpected": 1}, {})
    with pytest.raises(ModelError, match="no text"):
        m.generate(IMG, Prompt("x"))


def test_encode_image_resizes_and_converts():
    big = Image.new("RGBA", (4000, 1000))
    mime, b64 = encode_image(big, "jpeg", max_side=1000)
    assert mime == "image/jpeg" and len(b64) > 0
    import base64
    import io

    out = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert out.size == (1000, 250)


def test_ocr_engines_fail_cleanly_without_dependencies(monkeypatch):
    import builtins

    from nepeval_ocr.models.ocr_engines import EasyOCRModel, TesseractModel

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("pytesseract", "easyocr"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(FatalModelError, match=r"\.\[tesseract\]"):
        TesseractModel().setup()
    with pytest.raises(FatalModelError, match=r"\.\[easyocr\]"):
        EasyOCRModel().setup()
    assert "vqa" not in TesseractModel.tasks

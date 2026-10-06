"""Any OpenAI-compatible /chat/completions endpoint: Tarka, OpenRouter, OpenAI, vLLM,
SGLang, llama.cpp server, Gemini's and Anthropic's compatibility endpoints."""

from __future__ import annotations

import base64
import io
import os
import random
import threading
import time
from typing import Any

from ..types import Generation, Prompt
from .base import FatalModelError, Model, ModelError

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def encode_image(image: Any, fmt: str = "png", max_side: int | None = None) -> tuple[str, str]:
    """PIL image -> (mime, base64). Converts to RGB and optionally caps the long side."""
    img = image
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return mime, base64.b64encode(buf.getvalue()).decode("ascii")


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


class OpenAICompatModel(Model):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str | list[str] | None = "OPENAI_API_KEY",
        *,
        temperature: float | None = 0.0,
        max_tokens: int = 4096,
        max_tokens_field: str = "max_tokens",
        timeout: float = 300.0,
        retries: int = 5,
        concurrency: int = 8,
        image_format: str = "png",
        max_image_side: int | None = None,
        image_detail: str | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = [api_key_env] if isinstance(api_key_env, str) else (api_key_env or [])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_field = max_tokens_field
        self.timeout = timeout
        self.retries = retries
        self.max_concurrency = concurrency
        self.image_format = image_format
        self.max_image_side = max_image_side
        self.image_detail = image_detail
        self.extra_body = extra_body or {}
        self.extra_headers = extra_headers or {}
        self._client: Any = None
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "openai_compat",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "image_format": self.image_format,
            "max_image_side": self.max_image_side,
            "image_detail": self.image_detail,
            "extra_body": self.extra_body,
        }

    def _api_key(self) -> str | None:
        for name in self.api_key_env:
            if os.environ.get(name):
                return os.environ[name]
        return None

    def setup(self) -> None:
        import httpx

        key = self._api_key()
        if self.api_key_env and not key:
            raise FatalModelError(
                f"no API key: set one of {', '.join(self.api_key_env)} "
                "(or pass api_key_env=null for an endpoint without auth)"
            )
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=30.0),
            limits=httpx.Limits(max_connections=max(4, self.max_concurrency * 2)),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _body(self, image: Any, prompt: Prompt) -> dict[str, Any]:
        mime, b64 = encode_image(image, self.image_format, self.max_image_side)
        image_url: dict[str, Any] = {"url": f"data:{mime};base64,{b64}"}
        if self.image_detail:
            image_url["detail"] = self.image_detail
        messages: list[dict[str, Any]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image_url},
                    {"type": "text", "text": prompt.text},
                ],
            }
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.max_tokens_field: self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    path = "/chat/completions"

    def generate(self, image: Any, prompt: Prompt) -> Generation:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self.setup()
        payload, latency, attempts = self._request(self._body(image, prompt))
        gen = self._parse(payload)
        gen.latency_s = latency
        gen.extra["attempts"] = attempts
        return gen

    def _parse(self, payload: dict[str, Any]) -> Generation:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        return Generation(
            text=_content_text(message.get("content")),
            latency_s=0.0,
            finish_reason=choice.get("finish_reason"),
            usage=payload.get("usage"),
            extra={"provider": payload.get("provider")},
        )

    def _valid(self, payload: dict[str, Any]) -> str | None:
        """Return why a 200 payload is unusable (and worth retrying), or None."""
        # OpenRouter and some proxies report upstream errors inside a 200.
        if payload.get("error") and not payload.get("choices"):
            return f"provider error: {str(payload['error'])[:300]}"
        if not payload.get("choices"):
            return "response had no choices"
        return None

    def _request(self, body: dict[str, Any]) -> tuple[dict[str, Any], float, int]:
        import httpx

        last = ""
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                resp = self._client.post(self.path, json=body)
            except httpx.TransportError as exc:  # timeouts, resets, DNS
                last = f"{type(exc).__name__}: {exc}"
                self._sleep(attempt, None)
                continue
            latency = time.perf_counter() - started

            if resp.status_code in (401, 403):
                raise FatalModelError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code == 404:
                raise FatalModelError(f"HTTP 404 (unknown model or path?): {resp.text[:300]}")
            if resp.status_code in RETRYABLE_STATUS:
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                self._sleep(attempt, resp.headers.get("retry-after"))
                continue
            if resp.status_code >= 400:
                raise ModelError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            try:
                payload = resp.json()
            except ValueError:
                last = f"non-JSON response: {resp.text[:200]}"
                self._sleep(attempt, None)
                continue
            problem = self._valid(payload) if isinstance(payload, dict) else "non-object JSON"
            if problem:
                last = problem
                self._sleep(attempt, None)
                continue
            return payload, latency, attempt + 1
        raise ModelError(f"gave up after {self.retries + 1} attempts; last: {last}")

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.retries:
            return
        delay = min(60.0, 2.0**attempt) + random.uniform(0, 1)
        if retry_after:
            try:
                delay = max(delay, min(120.0, float(retry_after)))
            except ValueError:
                pass
        time.sleep(delay)


class TarkaOCRModel(OpenAICompatModel):
    """Tarka's native OCR endpoint (`POST /ocr`), used by Himalaya's dedicated OCR models.
    Multimodal chat models on Tarka use the plain OpenAI-compatible adapter instead."""

    path = "/ocr"

    def __init__(self, model: str, base_url: str = "https://tarka.rest/v1",
                 api_key_env: str | list[str] | None = ("TARKA_API_KEY", "NEPEVAL_API_TOKEN"),
                 **kwargs: Any) -> None:
        kwargs.setdefault("max_tokens", 4096)
        super().__init__(model, base_url, list(api_key_env or []), **kwargs)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "adapter": "tarka_ocr"}

    def _body(self, image: Any, prompt: Prompt) -> dict[str, Any]:
        mime, b64 = encode_image(image, self.image_format, self.max_image_side)
        body: dict[str, Any] = {
            "model": self.model,
            "image": f"data:{mime};base64,{b64}",
            "prompt": prompt.text,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        return body

    def _valid(self, payload: dict[str, Any]) -> str | None:
        if "text" not in payload:
            return f"response had no text: {str(payload)[:200]}"
        return None

    def _parse(self, payload: dict[str, Any]) -> Generation:
        return Generation(text=str(payload.get("text", "")), latency_s=0.0,
                          usage=payload.get("usage"))

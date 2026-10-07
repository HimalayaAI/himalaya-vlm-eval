"""Model metadata the leaderboard filters on but results do not carry: price per million
input/output tokens and context length.

Prices change while results are immutable, so this is a separate, rewritable document in
the store (`meta/models.json`) that the API joins onto rows at serve time. It is built by
`himeval meta refresh` from:

1. `catalog/model_meta.yaml` — explicit values or an explicit OpenRouter id per model id
   (fixes for names the matcher gets wrong, and our own models);
2. an `openrouter:` endpoint on the model (catalog presets and ad-hoc specs);
3. otherwise a match of the model's name against OpenRouter's public model list
   (https://openrouter.ai/api/v1/models, an API meant for programmatic use).

A model nothing matches has no price: the board shows N/A, as Arena does. Arena's own site
is never read for prices (its terms forbid automated access).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from .catalog import _catalog_dirs
from .schema import ModelInfo, utcnow
from .store import Store

log = logging.getLogger("himeval")

META_KEY = "meta/models.json"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
OVERRIDES_FILE = "model_meta.yaml"

# Suffixes Arena (and others) put on a model name for a reasoning budget or serving mode
# of the same priced model. Deliberately not here: preview / exp / chat / instruct and
# release tags like -2506, which name a *different* model that may be priced differently.
_VARIANT_SUFFIXES = (
    "xhigh", "high", "medium", "low", "minimal", "max", "thinking", "reasoning",
    "nothinking", "no-thinking", "instant", "latest",
)
# Snapshot dates only (20250514, 2025-04-16): the same model pinned on a day.
_DATE_SUFFIX = re.compile(r"-(20\d{6}|20\d{2}-\d{2}-\d{2})$")


def normalise(name: str) -> str:
    """Comparable form: no vendor prefix, lowercase, `.`/`_`/space/parens → `-`."""
    name = name.split("/", 1)[-1].split(":", 1)[0].lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return re.sub(r"-{2,}", "-", name)


def candidates(name: str) -> list[str]:
    """`claude-opus-4-6-high` → [claude-opus-4-6-high, claude-opus-4-6]; dates dropped too."""
    base = normalise(name)
    out = [base]
    current = base
    while True:
        stripped = _DATE_SUFFIX.sub("", current)
        for suffix in _VARIANT_SUFFIXES:
            if stripped.endswith("-" + suffix):
                stripped = stripped[: -len(suffix) - 1]
                break
        if stripped == current or not stripped:
            break
        current = stripped
        out.append(current)
    return out


def _price_per_million(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v * 1_000_000, 6) if v >= 0 else None  # OpenRouter quotes $/token; -1 = n/a


class OpenRouterIndex:
    def __init__(self, models: list[dict[str, Any]]):
        self.by_id: dict[str, dict[str, Any]] = {}
        self.by_name: dict[str, str] = {}
        for m in models:
            mid = m.get("id", "")
            if not mid or ":" in mid or mid.startswith("~"):  # :free, :batch, aliases
                continue
            self.by_id[mid] = m
            for key in candidates(mid):
                # Prefer the undated id when both exist (the dated one is a snapshot).
                if key not in self.by_name or len(mid) < len(self.by_name[key]):
                    self.by_name[key] = mid

    def match(self, name: str) -> str | None:
        for key in candidates(name):
            if key in self.by_name:
                return self.by_name[key]
        return None

    def entry(self, openrouter_id: str) -> dict[str, Any] | None:
        m = self.by_id.get(openrouter_id)
        if m is None:
            return None
        pricing = m.get("pricing") or {}
        return {
            "input_price": _price_per_million(pricing.get("prompt")),
            "output_price": _price_per_million(pricing.get("completion")),
            "context_length": m.get("context_length")
            or (m.get("top_provider") or {}).get("context_length"),
            "source": f"openrouter:{openrouter_id}",
        }


def fetch_openrouter(timeout: float = 30.0) -> list[dict[str, Any]]:
    import httpx

    resp = httpx.get(OPENROUTER_MODELS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["data"]


def load_overrides(dirs: Iterable[Path] | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in dirs or _catalog_dirs():
        path = d / OVERRIDES_FILE
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for mid, value in (yaml.safe_load(fh) or {}).items():
                    out[str(mid)] = dict(value or {})
    return out


def build(models: Iterable[ModelInfo], index: OpenRouterIndex,
          overrides: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in models:
        o = overrides.get(m.id, {})
        if o.get("pricing") == "none":
            continue
        meta: dict[str, Any] | None = None
        or_id = o.get("openrouter")
        if or_id is None and m.endpoint and m.endpoint.startswith("openrouter:"):
            or_id = m.endpoint.split(":", 1)[1]
        if or_id is None and not {"input_price", "output_price"} & set(o):
            or_id = index.match(m.display_name) or index.match(m.id)
        if or_id is not None:
            meta = index.entry(or_id)
            if meta is None:
                log.warning("meta: %s → %s is not on OpenRouter", m.id, or_id)
        explicit = {k: o[k] for k in ("input_price", "output_price", "context_length")
                    if k in o}
        if explicit:
            meta = {**(meta or {}), **explicit, "source": (meta or {}).get("source", "catalog")}
        if meta:
            out[m.id] = meta
    return out


def refresh(store: Store, models: Iterable[ModelInfo], *,
            openrouter_models: list[dict[str, Any]] | None = None,
            dry_run: bool = False) -> dict[str, Any]:
    models = list({m.id: m for m in models}.values())
    index = OpenRouterIndex(openrouter_models if openrouter_models is not None
                            else fetch_openrouter())
    data = build(models, index, load_overrides())
    doc = {"generated_at": utcnow().isoformat(), "source": OPENROUTER_MODELS,
           "models": data}
    if not dry_run:
        store.put(META_KEY, json.dumps(doc, indent=1, sort_keys=True).encode(),
                  "application/json")
    missing = sorted(m.id for m in models if m.id not in data)
    return {"models": len(models), "priced": len(data), "missing": missing}


def read(store: Store) -> dict[str, dict[str, Any]]:
    if not store.exists(META_KEY):
        return {}
    return json.loads(store.get(META_KEY)).get("models", {})


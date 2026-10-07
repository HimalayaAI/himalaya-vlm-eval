"""Model and benchmark catalogs, loaded from YAML.

Built-in entries ship in `himalaya_vlm_eval/catalog/`. Extra directories can be added with
`HIMEVAL_CATALOG=/path/a:/path/b`; each may hold `models/*.yaml` and `benchmarks/*.yaml`.
A later entry with the same id overrides an earlier one, so a deployment can retune a
preset without forking.

Models can also be named ad hoc, without a preset:
  openrouter:<vendor>/<model>   openai:<model>   tarka:<model>   vllm:<model>
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from .schema import BenchmarkInfo, ModelInfo

BUILTIN_DIR = Path(__file__).parent / "catalog"

Engine = Literal["native", "vlmevalkit"]


@dataclass
class ModelEntry:
    info: ModelInfo
    adapter: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self, overrides: dict[str, Any] | None = None):
        from .registry import get_adapter

        params = {**self.params, **(overrides or {})}
        try:
            return get_adapter(self.adapter)(**params)
        except TypeError as exc:
            raise ValueError(f"model {self.info.id}: bad params for {self.adapter}: {exc}") from exc


@dataclass
class BenchmarkEntry:
    info: BenchmarkInfo
    engine: Engine
    spec: dict[str, Any]  # engine-specific: dataset/prompt/metrics, or the VLMEvalKit name
    path: Path | None = None


def _catalog_dirs() -> list[Path]:
    dirs = [BUILTIN_DIR]
    for part in os.environ.get("HIMEVAL_CATALOG", "").split(os.pathsep):
        if part.strip():
            dirs.append(Path(part).expanduser())
    return dirs


def _yaml_docs(directory: Path) -> list[tuple[Path, Any]]:
    out = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.y*ml")):
            with path.open(encoding="utf-8") as fh:
                for doc in yaml.safe_load_all(fh):
                    if doc is not None:
                        out.append((path, doc))
    return out


_INFO_FIELDS = set(ModelInfo.model_fields)


@lru_cache(maxsize=1)
def models() -> dict[str, ModelEntry]:
    found: dict[str, ModelEntry] = {}
    for d in _catalog_dirs():
        for path, doc in _yaml_docs(d / "models"):
            for raw in doc if isinstance(doc, list) else [doc]:
                raw = dict(raw)
                try:
                    adapter = raw.pop("adapter")
                    params = raw.pop("params", {}) or {}
                    unknown = set(raw) - _INFO_FIELDS
                    if unknown:
                        raise ValueError(f"unknown keys {sorted(unknown)}")
                    entry = ModelEntry(ModelInfo(**raw), adapter, params)
                except Exception as exc:
                    raise ValueError(f"{path}: model {raw.get('id', '?')}: {exc}") from exc
                found[entry.info.id] = entry
    return found


@lru_cache(maxsize=1)
def benchmarks() -> dict[str, BenchmarkEntry]:
    found: dict[str, BenchmarkEntry] = {}
    info_fields = set(BenchmarkInfo.model_fields)
    for d in _catalog_dirs():
        for path, doc in _yaml_docs(d / "benchmarks"):
            for raw in doc if isinstance(doc, list) else [doc]:
                raw = dict(raw)
                try:
                    engine = raw.pop("engine", "native")
                    info = BenchmarkInfo(**{k: v for k, v in raw.items() if k in info_fields})
                    spec = {k: v for k, v in raw.items() if k not in info_fields}
                    if engine not in ("native", "vlmevalkit"):
                        raise ValueError(f"unknown engine {engine!r}")
                except Exception as exc:
                    raise ValueError(f"{path}: benchmark {raw.get('id', '?')}: {exc}") from exc
                found[info.id] = BenchmarkEntry(info, engine, spec, path)
    return found


def reload() -> None:
    models.cache_clear()
    benchmarks.cache_clear()


# --- ad-hoc model specs ------------------------------------------------------------------

_PROVIDERS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": ["OPENROUTER_API_KEY"],
    },
    "openai": {"base_url": "https://api.openai.com/v1", "api_key_env": ["OPENAI_API_KEY"]},
    "tarka": {
        "base_url": "https://tarka.rest/v1",
        "api_key_env": ["TARKA_API_KEY", "HIMEVAL_API_TOKEN"],
    },
    "vllm": {"base_url": None, "api_key_env": None},
}
_ORGS = {
    "openai": "OpenAI", "google": "Google", "anthropic": "Anthropic", "qwen": "Alibaba",
    "mistralai": "Mistral AI", "meta-llama": "Meta", "x-ai": "xAI", "z-ai": "Zhipu AI",
    "moonshotai": "Moonshot AI", "deepseek": "DeepSeek", "bytedance": "ByteDance",
    "baidu": "Baidu", "opengvlab": "Shanghai AI Lab", "microsoft": "Microsoft",
    "nvidia": "NVIDIA",
}


def slug(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9.]+", "-", text.lower())).strip("-.")


def resolve_model(name: str) -> ModelEntry:
    """A catalog id, or `provider:model` for anything the catalog does not list."""
    catalog = models()
    if name in catalog:
        return catalog[name]
    provider, sep, model = name.partition(":")
    if not sep or provider not in _PROVIDERS or not model:
        close = [m for m in catalog if name.lower() in m][:8]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(
            f"unknown model {name!r}. Use a catalog id (`himeval list models`) or "
            f"provider:model with provider in {sorted(_PROVIDERS)}.{hint}"
        )
    params: dict[str, Any] = {"model": model, **_PROVIDERS[provider]}
    if provider == "vllm":
        params["base_url"] = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        params["concurrency"] = 16
    vendor = model.split("/", 1)[0] if "/" in model else provider
    info = ModelInfo(
        id=slug(model.replace("/", "-")),
        display_name=model.split("/", 1)[-1],
        org=_ORGS.get(vendor, vendor),
        endpoint=name,
    )
    return ModelEntry(info, "openai_compat", params)


def resolve_benchmark(name: str) -> BenchmarkEntry:
    catalog = benchmarks()
    if name in catalog:
        return catalog[name]
    lowered = {k.lower(): k for k in catalog}
    if name.lower() in lowered:
        return catalog[lowered[name.lower()]]
    raise KeyError(f"unknown benchmark {name!r}; see `himeval list benchmarks`")


def select_benchmarks(spec: str) -> list[BenchmarkEntry]:
    """Comma list of ids, `category:<name>`, `engine:<name>`, or `all`."""
    out: dict[str, BenchmarkEntry] = {}
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token == "all":
            out.update(benchmarks())
        elif token.startswith("category:"):
            cat = token.split(":", 1)[1]
            matched = {k: b for k, b in benchmarks().items() if b.info.category == cat}
            if not matched:
                raise KeyError(f"no benchmarks in category {cat!r}")
            out.update(matched)
        elif token.startswith("engine:"):
            eng = token.split(":", 1)[1]
            out.update({k: b for k, b in benchmarks().items() if b.engine == eng})
        else:
            b = resolve_benchmark(token)
            out[b.info.id] = b
    return list(out.values())

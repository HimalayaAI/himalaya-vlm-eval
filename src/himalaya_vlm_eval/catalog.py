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

# `arena` boards are imported (`himeval import-arena`), never run.
Engine = Literal["native", "vlmevalkit", "arena"]
RUNNABLE_ENGINES = ("native", "vlmevalkit")


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


@dataclass(frozen=True)
class BoardCategory:
    id: str
    label: str


@dataclass(frozen=True)
class BoardType:
    id: str
    label: str
    categories: tuple[BoardCategory, ...]


@dataclass(frozen=True)
class Layout:
    """catalog/boards.yaml: the top-bar types, their side-list categories, and the default
    board for each benchmark category."""

    types: tuple[BoardType, ...]
    defaults: dict[str, str]

    def boards(self) -> list[str]:
        return [f"{t.id}/{c.id}" for t in self.types for c in t.categories]


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def layout() -> Layout:
    """The last catalog directory holding a boards.yaml wins (a deployment can re-lay the
    board without forking)."""
    path = [d / "boards.yaml" for d in _catalog_dirs() if (d / "boards.yaml").exists()][-1]
    raw = _load_yaml(path)
    types = tuple(
        BoardType(t["id"], t["label"],
                  tuple(BoardCategory(c["id"], c["label"]) for c in t["categories"]))
        for t in raw["types"]
    )
    out = Layout(types, dict(raw.get("defaults") or {}))
    known = set(out.boards())
    bad = {k: v for k, v in out.defaults.items() if v not in known}
    if bad:
        raise ValueError(f"{path}: defaults point at unknown boards {bad}")
    return out


@lru_cache(maxsize=1)
def benchmarks() -> dict[str, BenchmarkEntry]:
    found: dict[str, BenchmarkEntry] = {}
    info_fields = set(BenchmarkInfo.model_fields)
    lay = layout()
    known_boards = set(lay.boards())
    for d in _catalog_dirs():
        for path, doc in _yaml_docs(d / "benchmarks"):
            for raw in doc if isinstance(doc, list) else [doc]:
                raw = dict(raw)
                try:
                    engine = raw.pop("engine", "native")
                    if engine not in RUNNABLE_ENGINES:
                        raise ValueError(f"unknown engine {engine!r}")
                    raw.setdefault("board", lay.defaults.get(raw.get("category", "")))
                    info = BenchmarkInfo(**{k: v for k, v in raw.items() if k in info_fields})
                    spec = {k: v for k, v in raw.items() if k not in info_fields}
                    if info.board not in known_boards:
                        raise ValueError(f"board {info.board!r} is not in boards.yaml")
                except Exception as exc:
                    raise ValueError(f"{path}: benchmark {raw.get('id', '?')}: {exc}") from exc
                found[info.id] = BenchmarkEntry(info, engine, spec, path)
        arena_path = d / "arena.yaml"
        if arena_path.exists():
            for entry in _arena_entries(arena_path, known_boards):
                found[entry.info.id] = entry
    return found


ARENA_METRIC = "arena_score"


def _arena_entries(path: Path, known_boards: set[str]) -> list[BenchmarkEntry]:
    raw = _load_yaml(path)
    out = []
    for arena in raw["arenas"]:
        for cat in arena["categories"]:
            board = f"{arena['type']}/{cat['board']}"
            if board not in known_boards:
                raise ValueError(f"{path}: arena {arena['id']}/{cat['key']}: board {board!r} "
                                 "is not in boards.yaml")
            for style_control in (True, False):
                bid = f"arena-{arena['id']}-{cat['board']}"
                name = f"{arena['display_name']} · {cat['label']}"
                if not style_control:
                    bid += "-no-style-control"
                    name += " (no style control)"
                info = BenchmarkInfo(
                    id=bid,
                    display_name=name,
                    category=cat["category"],
                    description=arena["description"].strip(),
                    url=arena["url"],
                    language=cat.get("language", "multi"),
                    primary_metric=ARENA_METRIC,
                    higher_is_better=True,
                    scale_max=None,
                    board=board,
                    style_control=style_control,
                    cases_unit="votes",
                )
                spec = {
                    "dataset": raw["dataset"],
                    "subset": arena["subsets"]["style_control" if style_control else "raw"],
                    "category_key": cat["key"],
                    "arena": arena["id"],
                    "license": raw.get("license"),
                    "attribution": " ".join(str(raw.get("attribution", "")).split()),
                }
                out.append(BenchmarkEntry(info, "arena", spec, path))
    return out


def reload() -> None:
    models.cache_clear()
    layout.cache_clear()
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
    """Comma list of ids, `category:<name>`, `engine:<name>`, or `all`. `all` and
    `category:` select runnable benchmarks only (Arena boards are imported, not run)."""
    out: dict[str, BenchmarkEntry] = {}
    runnable = {k: b for k, b in benchmarks().items() if b.engine in RUNNABLE_ENGINES}
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token == "all":
            out.update(runnable)
        elif token.startswith("category:"):
            cat = token.split(":", 1)[1]
            matched = {k: b for k, b in runnable.items() if b.info.category == cat}
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

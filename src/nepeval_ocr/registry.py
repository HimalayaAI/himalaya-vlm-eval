"""Adapter registry: maps the `adapter:` name in a catalog entry to a Model class.

Built-ins are registered lazily by import path, so listing the catalog never imports
torch. Third-party packages can add adapters through the `nepeval_ocr.adapters`
entry-point group: `myocr = "my_pkg.module:MyModel"`.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from .models.base import Model

_BUILTIN: dict[str, str] = {
    "openai_compat": "nepeval_ocr.models.openai_compat:OpenAICompatModel",
    "tarka_ocr": "nepeval_ocr.models.openai_compat:TarkaOCRModel",
    "tesseract": "nepeval_ocr.models.ocr_engines:TesseractModel",
    "easyocr": "nepeval_ocr.models.ocr_engines:EasyOCRModel",
    "paddle": "nepeval_ocr.models.ocr_engines:PaddleOCRModel",
    "surya": "nepeval_ocr.models.ocr_engines:SuryaModel",
    "trocr": "nepeval_ocr.models.ocr_engines:TrOCRModel",
}
_CUSTOM: dict[str, type[Model]] = {}


def register_adapter(name: str):
    """Decorator for adapters defined in user code: `@register_adapter("mine")`."""

    def deco(cls: type[Model]) -> type[Model]:
        _CUSTOM[name] = cls
        return cls

    return deco


def _plugin_paths() -> dict[str, str]:
    return {ep.name: ep.value for ep in entry_points(group="nepeval_ocr.adapters")}


def adapter_names() -> list[str]:
    return sorted({*_BUILTIN, *_CUSTOM, *_plugin_paths()})


def get_adapter(name: str) -> type[Model]:
    if name in _CUSTOM:
        return _CUSTOM[name]
    path = _BUILTIN.get(name) or _plugin_paths().get(name)
    if path is None:
        raise KeyError(f"unknown adapter {name!r}; known: {adapter_names()}")
    module, _, attr = path.partition(":")
    cls = getattr(importlib.import_module(module), attr)
    if not (isinstance(cls, type) and issubclass(cls, Model)):
        raise TypeError(f"adapter {name!r} ({path}) is not a nepeval_ocr Model")
    return cls

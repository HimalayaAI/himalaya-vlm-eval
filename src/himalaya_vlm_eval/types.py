"""In-process types shared by benchmarks, models and the runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Task = Literal["ocr", "vqa"]


class DataUnavailable(ValueError):
    """The benchmark's data is not present on this machine (e.g. an unset data dir)."""


@dataclass
class Sample:
    id: str
    references: list[str]
    # A PIL image, or a zero-arg callable returning one (lets datasets decode lazily).
    image: Any
    question: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Structured ground truth (key-value fields, tables, word boxes …) for scorers that
    # need more than reference strings. Must be JSON-serialisable: it is stored with
    # the prediction so evaluation never reloads the dataset.
    target: Any = None

    def load_image(self) -> Any:
        img = self.image() if callable(self.image) else self.image
        if img is None:
            raise ValueError(f"sample {self.id} has no image")
        return img


@dataclass(frozen=True)
class Prompt:
    text: str
    system: str | None = None


@dataclass
class Generation:
    text: str
    latency_s: float
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


ImageLoader = Callable[[], Any]

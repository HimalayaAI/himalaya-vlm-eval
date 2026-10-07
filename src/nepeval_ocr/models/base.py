from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..schema import ModelKind
from ..types import Generation, Prompt, Task


class ModelError(RuntimeError):
    """A failure on one sample. The runner records it and moves on."""


class FatalModelError(RuntimeError):
    """A failure no retry or next sample will fix (bad key, unknown model). Stops the run."""


class Model(ABC):
    """One inference backend. Instances are built from a catalog entry plus overrides.

    Implementations must be thread-safe when `max_concurrency > 1`.
    """

    kind: ClassVar[ModelKind] = "vlm"
    # Engines that only transcribe cannot answer questions; the runner refuses VQA for them.
    tasks: ClassVar[frozenset[Task]] = frozenset({"ocr", "vqa"})
    max_concurrency: int = 1

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Load weights / open clients. Called once before the first sample."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources. Always called, even after a failure."""

    @abstractmethod
    def generate(self, image: Any, prompt: Prompt) -> Generation: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Parameters that affect output, for the run config hash. Never include secrets."""

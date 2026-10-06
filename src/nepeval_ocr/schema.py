"""The published result contract.

A `RunResult` is the only thing that crosses from the runner to the store, and from the
store to the API. It is self-describing: model and benchmark metadata are embedded, so the
API never needs this repo's config files to render a row, and an imported result from
another leaderboard has the same shape as one measured here.

Bump SCHEMA_VERSION on any breaking change; the API refuses versions it does not know.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

Category = Literal[
    "ocr",  # plain transcription: CER/WER
    "document",  # document VQA and parsing
    "chart",  # charts, figures, diagrams, tables
    "math",  # visual math reasoning
    "chat",  # open-ended, judge-scored
    "general",  # broad multimodal understanding
    "hallucination",
]
ModelKind = Literal["vlm", "ocr_engine"]
SourceKind = Literal["measured", "imported"]


def _check_id(value: str, what: str) -> str:
    if not ID_PATTERN.match(value):
        raise ValueError(
            f"{what} {value!r} must be lowercase and match {ID_PATTERN.pattern} "
            "(it is used in paths and URLs)"
        )
    return value


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    org: str = "Unknown"
    kind: ModelKind = "vlm"
    open_weights: bool | None = None
    params_b: float | None = Field(default=None, description="Parameters, billions")
    license: str | None = None
    url: str | None = None
    # Free-form provenance of the weights or endpoint, e.g. "openrouter:qwen/qwen3-vl-8b".
    endpoint: str | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _check_id(v, "model id")


class BenchmarkInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    category: Category
    description: str = ""
    url: str | None = None
    language: str = "en"
    primary_metric: str
    higher_is_better: bool = True
    # Scale of the primary metric as stored: (0, 1), (0, 100), (0, 1000) …
    scale_max: float = 100.0
    # Size of the full evaluation set; a run with fewer cases is a subset.
    full_cases: int | None = None
    # Bumped when prompt, scoring or data change in a way that breaks comparability.
    version: str = "1"

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _check_id(v, "benchmark id")

    def score_100(self, value: float) -> float:
        """Map a primary-metric value onto 0–100, higher is better (for cross-suite averages)."""
        frac = max(0.0, min(1.0, value / self.scale_max)) if self.scale_max else 0.0
        return 100.0 * (frac if self.higher_is_better else 1.0 - frac)


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SourceKind
    harness: str  # "nepeval-ocr", "vlmevalkit", "opencompass-openvlm", …
    harness_version: str | None = None
    url: str | None = None  # where an imported number was taken from
    judge: str | None = None  # LLM judge model, when scoring used one
    notes: str | None = None


class MetricValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    ci_low: float | None = None
    ci_high: float | None = None

    @model_validator(mode="after")
    def _ci_order(self) -> MetricValue:
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("ci_low and ci_high must both be set or both be null")
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError("ci_low > ci_high")
        return self


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str
    created_at: datetime
    model: ModelInfo
    benchmark: BenchmarkInfo
    source: SourceInfo
    cases: int = Field(ge=0)
    errors: int = Field(default=0, ge=0)
    metrics: dict[str, MetricValue]
    # breakdowns[group_key][group_value][metric] = value, e.g. {"level": {"page": {"cer": .1}}}
    breakdowns: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    # Everything needed to reproduce: prompt hash, dataset revision, model params, subset seed.
    config: dict[str, Any] = Field(default_factory=dict)
    # Token usage, latency and other run statistics.
    stats: dict[str, Any] = Field(default_factory=dict)
    has_samples: bool = False

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, v: str) -> str:
        return _check_id(v, "run id")

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {v}; this build reads {SCHEMA_VERSION}")
        return v

    @model_validator(mode="after")
    def _primary_present(self) -> RunResult:
        if self.benchmark.primary_metric not in self.metrics:
            raise ValueError(
                f"primary metric {self.benchmark.primary_metric!r} missing from metrics "
                f"{sorted(self.metrics)}"
            )
        if self.errors > self.cases:
            raise ValueError("errors cannot exceed cases")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return self

    @property
    def primary(self) -> MetricValue:
        return self.metrics[self.benchmark.primary_metric]

    @property
    def is_subset(self) -> bool:
        full = self.benchmark.full_cases
        return full is not None and self.cases < full

"""Native benchmarks: transcription or short-answer VQA over a Hugging Face dataset or a
local manifest, fully described by a catalog YAML entry.

    id: nepalipixel
    engine: native
    task: ocr                       # ocr | vqa
    dataset:
      source: hf                    # hf | local
      repo: himalaya-ai/nepalipixel-synthetic-ocr-benchmark
      revision: <commit sha>        # pin it; unpinned runs record the resolved sha
      split: train
      image: image                  # column names
      references: text              # a string or list-of-strings column
      id: id                        # optional; row index otherwise
      question: null                # vqa only
      meta: [level, font_name]      # carried into predictions, used for breakdowns
    prompt: "Transcribe …"          # `{question}` is substituted for vqa
    metrics: [cer, wer, …]          # names from himalaya_vlm_eval.metrics
    breakdowns: [level, font_name]

A local dataset is a JSONL manifest; `image` paths resolve relative to it:
    dataset: {source: local, path: /data/himkosh-write/manifest.jsonl}
    {"id": "w-001", "image": "img/w-001.png", "text": "…", "writer": "…"}
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import metrics as M
from . import structured as S
from .catalog import BenchmarkEntry
from .text import clean_model_output
from .types import Prompt, Sample, Task


@dataclass
class LoadedSet:
    samples: list[Sample]
    total: int  # rows in the split before subsetting
    provenance: dict[str, Any]


class NativeBenchmark:
    def __init__(self, entry: BenchmarkEntry):
        if entry.engine != "native":
            raise ValueError(f"{entry.info.id} is a {entry.engine} benchmark")
        self.entry = entry
        self.info = entry.info
        spec = entry.spec
        self.task: Task = spec.get("task", "ocr")
        self.dataset: dict[str, Any] = dict(spec["dataset"])
        self.prompt_text: str = spec["prompt"]
        self.system: str | None = spec.get("system")
        self.scorer: str | None = spec.get("scorer")
        self.breakdowns: list[str] = list(spec.get("breakdowns") or [])
        self.postprocess: bool = spec.get("postprocess", True)
        if self.scorer:
            if self.scorer not in S.SCORERS:
                raise ValueError(f"{self.info.id}: unknown scorer {self.scorer!r}; "
                                 f"known: {sorted(S.SCORERS)}")
            self.metric_names = list(S.METRICS[self.scorer])
        else:
            self.metric_names = list(spec.get("metrics") or [self.info.primary_metric])
            for name in self.metric_names:
                M.get_metric(name)
        if self.info.primary_metric not in self.metric_names:
            raise ValueError(f"{self.info.id}: primary metric not in metrics")
        if (self.task == "vqa" and not self.dataset.get("question")
                and self.dataset.get("source") != "nepalipixel_docs"):
            raise ValueError(f"{self.info.id}: vqa benchmarks need dataset.question")

    # --- identity --------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.info.id,
            "version": self.info.version,
            "task": self.task,
            "dataset": self.dataset,
            "prompt_sha": hashlib.sha256(
                f"{self.system or ''}\x00{self.prompt_text}".encode()
            ).hexdigest()[:12],
            "metrics": self.metric_names,
            "scorer": self.scorer,
            "postprocess": "clean_model_output" if self.postprocess else None,
        }

    # --- data ------------------------------------------------------------------------

    def load(self, limit: int | None = None, seed: int = 0) -> LoadedSet:
        source = self.dataset.get("source", "hf")
        if source == "hf":
            return self._load_hf(limit, seed)
        if source == "local":
            return self._load_local(limit, seed)
        if source == "nepalipixel_docs":
            from .docs import load_generator_output

            samples, provenance = load_generator_output(self.dataset)
            total = len(samples)
            keep = self._subset(total, limit, seed)
            samples = [samples[i] for i in keep]
            _check_unique(samples, self.info.id)
            return LoadedSet(samples, total, provenance)
        raise ValueError(f"{self.info.id}: unknown dataset source {source!r}")

    @staticmethod
    def _subset(n: int, limit: int | None, seed: int) -> list[int]:
        if not limit or limit >= n:
            return list(range(n))
        # A seeded random subset, not the first N rows: datasets are often sorted by
        # difficulty or source, and the head is not representative.
        return sorted(random.Random(seed).sample(range(n), limit))

    def _refs(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)]

    def _load_hf(self, limit: int | None, seed: int) -> LoadedSet:
        import logging

        from datasets import load_dataset

        # datasets/huggingface_hub reset logger levels on import; keep request logs quiet.
        for noisy in ("httpx", "httpcore", "huggingface_hub", "datasets", "fsspec"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        d = self.dataset
        revision = d.get("revision")
        resolved = revision
        try:
            from huggingface_hub import HfApi

            resolved = HfApi().dataset_info(d["repo"], revision=revision).sha
        except Exception:  # offline with a warm cache is fine; record what we know
            pass
        ds = load_dataset(d["repo"], d.get("config"), split=d.get("split", "test"),
                          revision=revision)
        total = len(ds)
        indices = self._subset(total, limit, seed)
        ds = ds.select(indices)

        image_col, ref_col = d.get("image", "image"), d.get("references", "text")
        id_col, q_col = d.get("id"), d.get("question")
        meta_cols = list(d.get("meta") or [])
        light_cols = [c for c in {ref_col, id_col, q_col, *meta_cols} if c]
        missing = [c for c in [image_col, *light_cols] if c not in ds.column_names]
        if missing:
            raise ValueError(f"{self.info.id}: columns {missing} not in {ds.column_names}")
        light = ds.select_columns(light_cols)

        samples = []
        for pos, (row_idx, row) in enumerate(zip(indices, light, strict=True)):
            sid = str(row[id_col]) if id_col else f"{d.get('split', 'test')}-{row_idx}"
            samples.append(
                Sample(
                    id=sid,
                    references=self._refs(row[ref_col]),
                    image=_HFImage(ds, pos, image_col),
                    question=str(row[q_col]) if q_col else None,
                    meta={c: row[c] for c in meta_cols} | {"row": row_idx},
                )
            )
        _check_unique(samples, self.info.id)
        return LoadedSet(samples, total, {"repo": d["repo"], "revision": resolved,
                                          "split": d.get("split", "test"), "rows": total})

    def _load_local(self, limit: int | None, seed: int) -> LoadedSet:
        manifest = Path(self.dataset["path"]).expanduser()
        base = manifest.parent
        rows = [json.loads(line) for line in manifest.read_text("utf-8").splitlines()
                if line.strip()]
        total = len(rows)
        ref_key = self.dataset.get("references", "text")
        q_key = self.dataset.get("question")
        meta_keys = list(self.dataset.get("meta") or [])
        samples = []
        for i in self._subset(total, limit, seed):
            row = rows[i]
            path = base / row[self.dataset.get("image", "image")]
            samples.append(
                Sample(
                    id=str(row.get(self.dataset.get("id", "id"), i)),
                    references=self._refs(row.get(ref_key)),
                    image=_FileImage(path),
                    question=str(row[q_key]) if q_key else None,
                    meta={k: row.get(k) for k in meta_keys},
                )
            )
        _check_unique(samples, self.info.id)
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
        return LoadedSet(samples, total, {"manifest": str(manifest), "sha256": digest,
                                          "rows": total})

    # --- prompting and scoring -------------------------------------------------------

    def prompt(self, sample: Sample) -> Prompt:
        text = self.prompt_text
        if sample.question is not None:
            text = text.replace("{question}", sample.question)
        return Prompt(text=text, system=self.system)

    def score(self, output: str | None, references: list[str],
              target: Any = None) -> dict[str, float | None]:
        if self.scorer:
            if output is None:
                return dict(S.WORST[self.scorer])
            return S.SCORERS[self.scorer](output, references, target)
        if output is None:  # the model errored on this sample
            return {name: M.WORST[name] for name in self.metric_names}
        pred = clean_model_output(output) if self.postprocess else output
        return {name: M.get_metric(name)(pred, references) for name in self.metric_names}


class _HFImage:
    """Lazy image accessor: decodes only when the worker needs it."""

    __slots__ = ("col", "ds", "pos")

    def __init__(self, ds: Any, pos: int, col: str):
        self.ds, self.pos, self.col = ds, pos, col

    def __call__(self) -> Any:
        return self.ds[self.pos][self.col]


class _FileImage:
    __slots__ = ("path",)

    def __init__(self, path: Path):
        self.path = path

    def __call__(self) -> Any:
        from PIL import Image

        with Image.open(self.path) as im:
            im.load()
            return im.copy()


def _check_unique(samples: list[Sample], bench: str) -> None:
    seen: set[str] = set()
    for s in samples:
        if s.id in seen:
            raise ValueError(f"{bench}: duplicate sample id {s.id!r}; ids key resume and scoring")
        seen.add(s.id)

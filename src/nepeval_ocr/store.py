"""Where published results live, shared by the runner (writes) and the API (reads).

Layout, identical for a local directory and an S3 prefix:

    runs/<run_id>/result.json        schema.RunResult — written LAST, it is the commit marker
    runs/<run_id>/samples.jsonl.gz   per-sample prediction, references and scores (measured runs)

Results are immutable: a run id is never overwritten with different content. Select the
store with NEPEVAL_STORE: a path, file:///path, or s3://bucket/prefix (needs the `s3`
extra; S3-compatible endpoints via AWS_ENDPOINT_URL).
"""

from __future__ import annotations

import gzip
import io
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schema import RunResult

RESULT_KEY = "result.json"
SAMPLES_KEY = "samples.jsonl.gz"


class StoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredObject:
    key: str
    version: str  # mtime+size locally, ETag on S3: changes when content changes


class Store(Protocol):
    url: str

    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> Iterator[StoredObject]: ...


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.url = str(self.root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise StoreError(f"key escapes store root: {key}")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> Iterator[StoredObject]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                st = path.stat()
                yield StoredObject(path.relative_to(self.root).as_posix(),
                                   f"{st.st_mtime_ns}-{st.st_size}")


class S3Store:
    def __init__(self, bucket: str, prefix: str = ""):
        try:
            import boto3
        except ImportError as exc:
            raise StoreError("S3 store needs the s3 extra: pip install 'nepeval-ocr[s3]'") from exc
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.url = f"s3://{bucket}/{self.prefix}" if self.prefix else f"s3://{bucket}"
        self._s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data,
                            ContentType=content_type)

    def get(self, key: str) -> bytes:
        try:
            return self._s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> Iterator[StoredObject]:
        paginator = self._s3.get_paginator("list_objects_v2")
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for obj in page.get("Contents", []):
                yield StoredObject(obj["Key"][strip:], obj["ETag"].strip('"'))


def open_store(url: str | None = None) -> Store:
    url = url or os.environ.get("NEPEVAL_STORE")
    if not url:
        raise StoreError("no result store: set NEPEVAL_STORE (a path or s3://bucket/prefix)")
    if url.startswith("s3://"):
        bucket, _, prefix = url[5:].partition("/")
        return S3Store(bucket, prefix)
    url = url.removeprefix("file://")
    return LocalStore(url)


# --- publishing --------------------------------------------------------------------------


def _result_bytes(result: RunResult) -> bytes:
    return json.dumps(json.loads(result.model_dump_json()), ensure_ascii=False, indent=2,
                      sort_keys=True).encode("utf-8")


def publish_result(store: Store, result: RunResult, samples: bytes | None = None,
                   *, force: bool = False) -> str:
    """Write a result (and its samples) under runs/<run_id>/. Returns "published",
    "unchanged" (identical result already there) or raises on a conflicting rewrite."""
    base = f"runs/{result.run_id}"
    body = _result_bytes(result)
    if store.exists(f"{base}/{RESULT_KEY}"):
        current = store.get(f"{base}/{RESULT_KEY}")
        if json.loads(current) == json.loads(body):
            return "unchanged"
        if not force:
            raise StoreError(
                f"{base} already published with different content; results are immutable "
                "(re-evaluate to get a new run id, or pass force=True)"
            )
    if samples is not None:
        store.put(f"{base}/{SAMPLES_KEY}", samples, "application/gzip")
    store.put(f"{base}/{RESULT_KEY}", body, "application/json")  # commit marker, last
    return "published"


def gzip_samples(run_dir: Path) -> bytes:
    """Join predictions with scores into one gzipped JSONL for drill-down."""
    from .runner import PREDICTIONS, SCORES, latest_records

    preds = latest_records(run_dir / PREDICTIONS)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, \
            (run_dir / SCORES).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            score = json.loads(line)
            p = preds.get(score["sample_id"], {})
            row = {
                "sample_id": score["sample_id"],
                "ok": score["ok"],
                "scores": score["scores"],
                "output": p.get("output"),
                "error": p.get("error"),
                "references": p.get("references"),
                "question": p.get("question"),
                "meta": p.get("meta"),
                "latency_s": p.get("latency_s"),
                "finish_reason": p.get("finish_reason"),
            }
            gz.write((json.dumps(row, ensure_ascii=False, default=str) + "\n").encode())
    return buf.getvalue()


def read_samples(store: Store, run_id: str) -> list[dict]:
    raw = store.get(f"runs/{run_id}/{SAMPLES_KEY}")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        return [json.loads(line) for line in gz.read().decode("utf-8").splitlines() if line]

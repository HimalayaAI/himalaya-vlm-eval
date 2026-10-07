import json

import pytest

from himalaya_vlm_eval import catalog
from himalaya_vlm_eval.publish import PublishRefused, build_imported, import_file, publish_run
from himalaya_vlm_eval.runner import evaluate, infer
from himalaya_vlm_eval.schema import RunResult
from himalaya_vlm_eval.store import (
    LocalStore,
    S3Store,
    StoredObject,
    StoreError,
    open_store,
    publish_result,
    read_samples,
)

from .conftest import result
from .test_runner import _answers, entry


def test_local_store_roundtrip_and_listing(tmp_path):
    s = LocalStore(tmp_path)
    s.put("runs/a/result.json", b"{}", "application/json")
    assert s.exists("runs/a/result.json") and s.get("runs/a/result.json") == b"{}"
    assert [o.key for o in s.list("runs/")] == ["runs/a/result.json"]
    with pytest.raises(KeyError):
        s.get("runs/missing")
    with pytest.raises(StoreError):
        s.put("../escape", b"", "x")


def test_open_store_from_env(tmp_path, monkeypatch):
    with pytest.raises(StoreError, match="HIMEVAL_STORE"):
        open_store()
    monkeypatch.setenv("HIMEVAL_STORE", f"file://{tmp_path}")
    assert isinstance(open_store(), LocalStore)


def test_results_are_immutable(tmp_path):
    store = LocalStore(tmp_path)
    r = result("m", "b", 50.0)
    assert publish_result(store, r) == "published"
    assert publish_result(store, r) == "unchanged"
    changed = r.model_copy(update={"cases": 99})
    with pytest.raises(StoreError, match="immutable"):
        publish_result(store, changed)
    assert publish_result(store, changed, force=True) == "published"


def test_publish_run_with_samples_and_gate(tmp_path, manifest_catalog):
    bench = catalog.resolve_benchmark("local-ocr")
    _answers(manifest_catalog["texts"])
    run_dir = infer(entry(concurrency=1), bench, tmp_path / "w")
    with pytest.raises(PublishRefused, match="evaluate it first"):
        publish_run(run_dir, LocalStore(tmp_path / "s"))
    r = evaluate(run_dir)
    store = LocalStore(tmp_path / "s")
    _, status = publish_run(run_dir, store)
    assert status == "published"
    rows = read_samples(store, r.run_id)
    assert len(rows) == 5 and rows[0]["scores"]["cer"] == 0.0 and rows[0]["references"]
    stored = RunResult.model_validate_json(store.get(f"runs/{r.run_id}/result.json"))
    assert stored == r


def test_publish_refuses_high_error_rate(tmp_path, manifest_catalog):
    from himalaya_vlm_eval.runner import RunOptions

    bench = catalog.resolve_benchmark("local-ocr")
    run_dir = infer(entry("error_second", concurrency=1), bench, tmp_path / "w",
                    options=RunOptions(retry_errors=False))
    evaluate(run_dir)
    with pytest.raises(PublishRefused, match=r"20\.0%"):
        publish_run(run_dir, LocalStore(tmp_path / "s"))
    _, status = publish_run(run_dir, LocalStore(tmp_path / "s"), max_error_rate=0.25)
    assert status == "published"


def test_import_validates_against_catalog(tmp_path):
    good = {"model": {"id": "gpt-4o", "display_name": "GPT-4o", "org": "OpenAI"},
            "benchmark": "ocrbench", "score": 805, "date": "2025/09/17",
            "source": {"harness": "opencompass-openvlm", "url": "https://x"}}
    r = build_imported(good)
    assert r.source.kind == "imported" and r.benchmark.scale_max == 1000
    assert r.run_id == "gpt-4o__ocrbench__opencompass-openvlm__20250917"
    with pytest.raises(ValueError, match="outside the benchmark scale"):
        build_imported({**good, "benchmark": "mmstar", "score": 805})
    with pytest.raises(KeyError):
        build_imported({**good, "benchmark": "nope"})

    f = tmp_path / "imports.json"
    f.write_text(json.dumps({"results": [good, {**good, "benchmark": "nope"}]}))
    store = LocalStore(tmp_path / "s")
    assert import_file(f, store) == {"published": 1, "unchanged": 0, "failed": 1}
    assert import_file(f, store) == {"published": 0, "unchanged": 1, "failed": 1}


class FakeS3:
    """Just enough of the boto3 S3 client for S3Store."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()
        import io

        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def get_paginator(self, name):
        objs = self.objects

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "ETag": f'"{hash(v)}"'}
                                    for k, v in sorted(objs.items()) if k.startswith(Prefix)]}

        return P()


def test_s3_store_with_prefix():
    s = S3Store.__new__(S3Store)
    s.bucket, s.prefix, s.url, s._s3 = "b", "himeval", "s3://b/himeval", FakeS3()
    r = result("m", "b", 1.0)
    assert publish_result(s, r) == "published"
    assert "himeval/runs/" + r.run_id + "/result.json" in s._s3.objects
    listed = list(s.list("runs/"))
    assert listed[0].key == f"runs/{r.run_id}/result.json"
    assert isinstance(listed[0], StoredObject)
    assert publish_result(s, r) == "unchanged"
    with pytest.raises(KeyError):
        s.get("runs/none")

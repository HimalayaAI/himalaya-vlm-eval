#!/usr/bin/env python3
"""Run Nepali Pixel OCR benchmark against any supported OCR/VLM model.

This script provides a unified interface to benchmark multiple OCR models:
- API models (OpenAI-compatible endpoints like Scalabs Tarka)
- Surya (local, auto-spawns vllm/llama.cpp)
- Tesseract (local, requires Nepali language data)
- trOCR (local Hugging Face model)

Usage:
    # Run against Scalabs API (Tarka)
    export TARKA_API_KEY="tk_live_..."
    python scripts/run_unified_benchmark.py --model api --limit 1000

    # Run Surya locally
    python scripts/run_unified_benchmark.py --model surya --limit 1000

    # Run Tesseract locally
    python scripts/run_unified_benchmark.py --model tesseract-nep --limit 1000

    # Run trOCR locally (CPU)
    python scripts/run_unified_benchmark.py --model trocr --device cpu --limit 1000
"""

import argparse
import collections
import concurrent.futures
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from nepeval_ocr.dataset import load_nepali_pixel_dataset
from nepeval_ocr.scorers import compute_metrics

args = None  # Global for use in evaluate_one


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=["api", "surya", "tesseract-nep", "trocr", "easyocr", "paddle"],
        help="Model to benchmark. 'api' requires --api-base-url. "
             "'surya' runs locally with auto-spawned server. "
             "'tesseract-nep' runs locally with Nepali lang pack. "
             "'trocr' runs locally with Hugging Face model. "
             "'easyocr' runs locally with EasyOCR. "
             "'paddle' runs locally with PaddleOCR."
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="API base URL (for api model). Default: https://himalayagpt.api.scalabs.ai/v1"
    )
    parser.add_argument(
        "--api-token-env",
        default="TARKA_API_KEY",
        help="Environment variable for API token"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Number of images to benchmark (default: 1000)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrency level for local models. API uses different setting."
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for local models (tesseract doesn't use this)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="API model temperature"
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="API request timeout"
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="API request retries"
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_existing_samples(samples_path: Path) -> List[dict]:
    """Load existing samples from a completed or partial run."""
    samples = []
    if samples_path.exists():
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def load_adapter():
    """Load the appropriate adapter based on model type."""
    global args
    
    if args.model == "api":
        from nepeval_ocr.adapters.openai_api import OpenAIVisionAdapter
        
        if not args.api_base_url:
            args.api_base_url = "https://himalayagpt.api.scalabs.ai/v1"
        
        token = os.getenv(args.api_token_env)
        if not token:
            for fallback in ["NEPEVAL_API_TOKEN", "OPENAI_API_KEY"]:
                token = os.getenv(fallback)
                if token:
                    break
        
        if not token:
            raise SystemExit(
                f"Set {args.api_token_env} or a fallback env var for API token."
            )
        
        return OpenAIVisionAdapter(
            model="scalabs/himalaya-bf16",
            base_url=args.api_base_url,
            token=token,
            timeout=args.timeout,
            retries=args.retries,
            temperature=args.temperature
        )
    
    elif args.model == "surya":
        from nepeval_ocr.adapters.surya import SuryaAdapter
        return SuryaAdapter(language="ne")
    
    elif args.model == "tesseract-nep":
        from nepeval_ocr.adapters.tesseract import TesseractAdapter
        return TesseractAdapter(lang="nep")
    
    elif args.model == "trocr":
        from nepeval_ocr.adapters.trocr import TrOCRAdapter
        return TrOCRAdapter(device=args.device)
    
    elif args.model == "easyocr":
        from nepeval_ocr.adapters.easyocr import EasyOCRAdapter
        return EasyOCRAdapter(langs=["ne", "en"], device=args.device)
    
    elif args.model == "paddle":
        from nepeval_ocr.adapters.paddle import PaddleOCRAdapter
        return PaddleOCRAdapter(lang="ne", device=args.device)
    
    else:
        raise ValueError(f"Unknown model type: {args.model}")


def evaluate_one(item, adapter, start_time: float = None, total_items: int = 0, completed: int = 0):
    """Evaluate a single sample and return results."""
    sample_id, image, ground_truth, metadata = item
    result = {
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "metadata": metadata,
        "model": args.model
    }
    
    try:
        if args.model == "api":
            response_text, stats = adapter.evaluate_sample_with_stats(image)
            result.update({
                "ok": True,
                "latency_sec": stats.get("latency_sec"),
                "response": response_text,
                "metrics": compute_metrics(ground_truth, response_text),
                "finish_reason": stats.get("finish_reason"),
                "usage": stats.get("usage"),
            })
        else:
            import time
            started = time.perf_counter()
            response_text = adapter.evaluate_sample(image)
            latency = time.perf_counter() - started
            
            result.update({
                "ok": True,
                "latency_sec": latency,
                "response": response_text,
                "metrics": compute_metrics(ground_truth, response_text),
            })
            
            # Print progress with ETA for local models
            if start_time is not None and total_items > 0:
                elapsed = time.perf_counter() - start_time
                completed += 1
                samples_per_sec = completed / elapsed if elapsed > 0 else 0
                remaining = total_items - completed
                eta_sec = remaining / samples_per_sec if samples_per_sec > 0 else 0
                
                eta_min = int(eta_sec / 60)
                eta_sec_rem = int(eta_sec % 60)
                elapsed_min = int(elapsed / 60)
                
                level = metadata.get("level", "unknown")
                print(f"  [{completed}/{total_items}] [{level}] {latency:.2f}s | "
                      f"Avg: {elapsed/completed:.2f}s | "
                      f"ETA: {eta_min}m {eta_sec_rem}s (elapsed: {elapsed_min}m) | "
                      f"sample_id={sample_id[:16]}...", flush=True)
    except Exception as exc:
        result.update({
            "ok": False,
            "latency_sec": None,
            "response": "",
            "error": str(exc),
            "metrics": {"cer": 1.0, "wer": 1.0, "exact_match": False}
        })
    return result


def main():
    global args
    args = parse_args()
    
    # Set output directory
    output_dir = args.output_dir or Path(f"results/unified_{args.model}_{timestamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load adapter
    print(f"Loading adapter for model: {args.model}...")
    adapter = load_adapter()
    
    # Determine model name for output
    model_name = args.model
    model_dir = output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    samples_path = model_dir / "samples.jsonl"
    progress_path = model_dir / "progress.json"
    
    # Load dataset
    print("Loading dataset (himalaya-ai/nepalipixel-synthetic-ocr-benchmark)...")
    dataset_gen = load_nepali_pixel_dataset(split="train")
    
    items = []
    skipped_pages = 0
    level_counts = collections.Counter()
    
    for i, item in enumerate(dataset_gen):
        if args.limit and len(items) >= args.limit:
            break
        metadata = item[3]
        level = metadata.get("level", "unknown")
        level_counts[level] += 1
        
        # Skip page-level and exact-level samples for Surya (they cause timeout issues)
        if args.model == "surya" and level in ("page", "exact"):
            skipped_pages += 1
            continue
        items.append(item)
    
    print(f"Dataset loaded. Level distribution: {dict(level_counts)}", flush=True)
    if skipped_pages > 0:
        print(f"Skipped {skipped_pages} samples (page/exact) for Surya", flush=True)
    print(f"Total items to process: {len(items)}", flush=True)
    
    # Pre-flight check with warm-up for local models
    if items:
        print("Pre-flight: testing adapter on first sample...", flush=True)
        try:
            first_item = items[0]
            first_metadata = first_item[3]
            first_level = first_metadata.get("level", "unknown")
            print(f"  Sample level: {first_level}", flush=True)
            preflight = evaluate_one(first_item, adapter)
            if not preflight["ok"]:
                print(
                    f"\nPre-flight check FAILED for {args.model}.\n"
                    f"  Error: {preflight['error']}\n"
                    f"  The adapter appears to be broken — aborting before "
                    f"processing {len(items)} samples.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except Exception as e:
            print(f"\nPre-flight check exception: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Pre-flight: OK (took {preflight['latency_sec']:.1f}s)", flush=True)
        
        # Warm-up phase for local models (first inference is slow due to model loading)
        if args.model != "api":
            print("Warm-up: processing 1 additional sample to load model...", flush=True)
            print("  (This may take 1-3 minutes on first run as the model loads into memory)", flush=True)
            if len(items) > 1:
                warmup_item = items[1]
                warmup_metadata = warmup_item[3]
                warmup_level = warmup_metadata.get("level", "unknown")
                print(f"  Warm-up sample level: {warmup_level}", flush=True)
                warmup_start = time.perf_counter()
                warmup = evaluate_one(warmup_item, adapter)
                warmup_time = time.perf_counter() - warmup_start
                print(f"Warm-up: completed (took {warmup_time:.1f}s)", flush=True)
                # Remove warm-up sample from results
                items = items[1:]
            else:
                print("Warm-up: only 1 item available, skipping", flush=True)
    
    # Run benchmark
    samples = []
    started_at = datetime.now(timezone.utc).isoformat()
    
    # Check for existing samples and resume if available
    existing_samples = []
    if samples_path.exists():
        existing_samples = load_existing_samples(samples_path)
        if existing_samples:
            print(f"Found {len(existing_samples)} existing samples. Resuming from where we left off...", flush=True)
            # Remove processed samples from items
            processed_ids = {s["sample_id"] for s in existing_samples}
            items = [item for item in items if item[0] not in processed_ids]
            # Show info about first remaining item
            if items:
                next_item = items[0]
                next_level = next_item[3].get("level", "unknown")
                print(f"  Next sample ({next_item[0]}) will be level: {next_level}", flush=True)
    
    if args.model == "api":
        # API models use concurrent futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(evaluate_one, item, adapter) for item in items]
            with samples_path.open("a", encoding="utf-8") as samples_handle:
                for completed, future in enumerate(concurrent.futures.as_completed(futures), start=len(existing_samples)):
                    sample = future.result()
                    samples.append(sample)
                    append_jsonl(samples_handle, sample)
                    
                    if completed % 25 == 0 or completed == len(existing_samples) + len(futures):
                        progress = {
                            "model": model_name,
                            "completed_completions": completed,
                            "total_completions": len(existing_samples) + len(futures),
                            "errored_completions": sum(1 for item in samples if not item["ok"]),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        with progress_path.open("w") as f:
                            json.dump(progress, f)
                        print(f"  {model_name}: {completed}/{len(existing_samples) + len(futures)} completions", flush=True)
    else:
        # Local models - sequential processing
        start_time = time.perf_counter()
        total_items = len(existing_samples) + len(items)
        with samples_path.open("a", encoding="utf-8") as samples_handle:
            for completed, item in enumerate(items, start=len(existing_samples) + 1):
                # Show what sample is starting
                sample_id, image, ground_truth, metadata = item
                level = metadata.get("level", "unknown")
                print(f"  Processing [{completed}/{total_items}] [{level}] sample_id={sample_id[:16]}...", flush=True)
                
                sample = evaluate_one(item, adapter, start_time=start_time, total_items=total_items, completed=completed-1)
                samples.append(sample)
                append_jsonl(samples_handle, sample)
                
                if completed % 25 == 0 or completed == total_items:
                    progress = {
                        "model": model_name,
                        "completed_completions": completed,
                        "total_completions": total_items,
                        "errored_completions": sum(1 for s in samples if not s["ok"]),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    with progress_path.open("w") as f:
                        json.dump(progress, f)
                    print(f"  {model_name}: {completed}/{total_items} completions", flush=True)
    
    # Summarize results
    valid_samples = [s for s in samples if s["ok"]]
    errored_completions = len(samples) - len(valid_samples)
    
    # Post-run guard
    if samples and (errored_completions / len(samples)) > 0.5:
        error_counts = collections.Counter(s["error"] for s in samples if not s["ok"])
        print(
            f"\nRun ABORTED for {model_name}: "
            f"{errored_completions}/{len(samples)} samples errored "
            f"({errored_completions / len(samples):.0%}).\n"
            f"  Distinct errors ({len(error_counts)}):",
            file=sys.stderr,
        )
        for msg, count in error_counts.most_common():
            print(f"    [{count}x] {msg}", file=sys.stderr)
        print(
            f"\n  Not writing summary.json — fix the adapter and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)
    
    summary = {
        "model": model_name,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "examples": len(samples),
        "errored_completions": errored_completions,
        "avg_cer": statistics.fmean(s["metrics"]["cer"] for s in valid_samples) if valid_samples else None,
        "avg_wer": statistics.fmean(s["metrics"]["wer"] for s in valid_samples) if valid_samples else None,
        "exact_match_rate": statistics.fmean(float(s["metrics"]["exact_match"]) for s in valid_samples) if valid_samples else None,
        "avg_latency_sec": statistics.fmean(s["latency_sec"] for s in valid_samples) if valid_samples else None,
    }
    
    with (model_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    # Run config
    run_config = {
        "run_id": output_dir.name,
        "model": model_name,
        "examples": len(items),
        "started_at": started_at,
    }
    if args.model == "api":
        run_config["api_base_url"] = args.api_base_url
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)
    
    # Full summary
    run_summary = {
        **run_config,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_summary": summary,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(run_summary, f, indent=2)
    
    print(f"\nWrote benchmark artifacts -> {output_dir}")
    print(f"\nSummary:")
    print(f"  Model: {summary['model']}")
    print(f"  Examples: {summary['examples']}")
    if summary['avg_cer']:
        print(f"  Avg CER: {summary['avg_cer']:.4f}")
    if summary['avg_wer']:
        print(f"  Avg WER: {summary['avg_wer']:.4f}")
    if summary['exact_match_rate'] is not None:
        print(f"  Exact Match Rate: {summary['exact_match_rate']:.2%}")
    if summary['avg_latency_sec']:
        print(f"  Avg Latency: {summary['avg_latency_sec']:.3f}s")


if __name__ == "__main__":
    main()

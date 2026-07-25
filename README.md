# nepeval-ocr

Standalone project for benchmarking OCR and VLM models against the `himalaya-ai/nepalipixel-synthetic-ocr-benchmark` dataset.

This project is completely isolated from the main `nepeval` codebase. It evaluates image-in/text-out inference with edit distance metrics (Character Error Rate, Word Error Rate, and Exact-Match).

## Features

- **Multiple Model Support**: Run evaluations with API-based VLMs or locally-hosted OCR models
- **Unified Interface**: Single script (`run_unified_benchmark.py`) supports all model types
- **Flexible Hosting**: Choose between cloud APIs or local inference (CPU/GPU)
- **Nepali Language**: Optimized for Nepali OCR evaluation

## Supported Models

### API Models (Cloud)
- **Scalabs Tarka** (OpenAI-compatible endpoint)
- Any OpenAI-compatible API endpoint

### Local Models
- **Surya OCR**: Requires Docker (GPU) or llama.cpp server binary (CPU)
- **Tesseract**: Classic OCR with Nepali language support
- **trOCR**: Hugging Face's Transformer-based OCR model for Nepali

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
# Default setup installs base dependencies
python -m pip install -e .
```

### Install Model-Specific Dependencies

Choose the model(s) you want to use:

| Model | Command | Notes |
|-------|---------|-------|
| **Surya** (local, GPU/CPU) | `pip install -e '.[surya]'` | Requires Docker (vllm) or llama.cpp server binary |
| **Tesseract** (local) | `pip install -e '.[tesseract]'` | Requires system binary + Nepali lang |
| **trOCR** (local, HF) | `pip install -e '.[trocr]'` | Uses `syubraj/TrOCR_Nepali` model |
| **EasyOCR** (local) | `pip install -e '.[easyocr]'` | - |
| **PaddleOCR** (local) | `pip install -e '.[paddle]'` | Requires [paddlepaddle](https://www.paddlepaddle.org.cn/en/install/quick) (Python 3.9-3.13 only) |

## Model Setup Instructions

### For API Models (Scalabs Tarka)

No extra dependencies needed. Just set your API token:

```bash
export TARKA_API_KEY="your_api_key_here"
```

Or use the fallback environment variables:
- `NEPEVAL_API_TOKEN`
- `OPENAI_API_KEY`

### For Local Models

#### Surya OCR

```bash
pip install -e '.[surya]'
# Surya requires an inference backend:
# - For GPU: Docker with NVIDIA runtime (auto-spawns vllm)
# - For CPU: llama.cpp server binary (https://github.com/ggml-org/llama.cpp/releases)
# 
# If Docker is available, Surya will auto-spawn vllm.
# For CPU-only, install llama.cpp and run:
#   export SURYA_INFERENCE_BACKEND=llamacpp
#   export LLAMA_CPP_BINARY=/path/to/llama-server
#   # Then run the benchmark
```

Note: Surya requires either Docker (for GPU) or the `llama-server` binary (for CPU). If neither is available, the benchmark will fail with a clear error message indicating how to install the required dependency.

#### Tesseract
```bash
# 1. Install Python package
pip install -e '.[tesseract]'

# 2. Install system binary and Nepali language data
# Debian/Ubuntu:
sudo apt install tesseract-ocr tesseract-ocr-nep

# Arch Linux:
sudo pacman -S tesseract tesseract-data-nep

# 3. Verify installation
tesseract --list-langs
```

Alternatively, copy the bundled `nep.traineddata` from the repo:
```bash
# Option 1: Copy to system tessdata
sudo cp tessdata/nep.traineddata /usr/share/tessdata/

# Option 2: Use TESSDATA_PREFIX (no sudo needed)
export TESSDATA_PREFIX="$(pwd)"
```

#### PaddleOCR
```bash
# 1. Install PaddleOCR
pip install -e '.[paddle]'

# 2. Install paddlepaddle (required for PaddleOCR 3.x)
# PaddlePaddle is not available on PyPI. Install from official channel:
# CPU version:
pip install paddlepaddle

# GPU version (requires CUDA):
# pip install paddlepaddle-gpu

# See https://www.paddlepaddle.org.cn/en/install/quick for more options

# Note: PaddlePaddle supports Python 3.9-3.13. Python 3.14+ is not supported.
```

#### trOCR
```bash
pip install -e '.[trocr]'
# Model (syubraj/TrOCR_Nepali) downloads on first run from Hugging Face
```

#### PaddleOCR
```bash
# 1. Install PaddleOCR
pip install -e '.[paddle]'

# 2. Install paddlepaddle (required for PaddleOCR 3.x)
# CPU version:
pip install paddlepaddle

# GPU version (requires CUDA):
# pip install paddlepaddle-gpu

# See https://www.paddlepaddle.org.cn/en/install/quick for more options
```

## Run Evaluations

The unified benchmark script supports all model types with a single interface.

### API Evaluation

```bash
export TARKA_API_KEY="your_key"
python scripts/run_unified_benchmark.py \
  --model api \
  --limit 100 \
  --output-dir results/api-scalabs \
  --concurrency 8
```

Options:
- `--api-base-url`: Override API endpoint (default: `https://himalayagpt.api.scalabs.ai/v1`)
- `--api-token-env`: Custom environment variable name (default: `TARKA_API_KEY`)
- `--concurrency`: API request parallelism (default: 4)
- `--timeout`: Request timeout in seconds (default: 300)
- `--retries`: Number of retry attempts (default: 3)

### Local Model Evaluations

#### Surya (Local)
```bash
python scripts/run_unified_benchmark.py \
  --model surya \
  --limit 100 \
  --output-dir results/surya-local \
  --concurrency 1  # Sequential for local models
```

**Note**: Surya requires Docker to auto-spawn the inference server. Make sure Docker is installed and running.

#### Tesseract (Local)
```bash
python scripts/run_unified_benchmark.py \
  --model tesseract-nep \
  --limit 100 \
  --output-dir results/tesseract-local
```

#### PaddleOCR (Local)
```bash
python scripts/run_unified_benchmark.py \
  --model paddle \
  --limit 100 \
  --output-dir results/paddle-local \
  --concurrency 1
```

#### trOCR (Local)
```bash
# Auto-detect GPU or use CPU explicitly
python scripts/run_unified_benchmark.py \
  --model trocr \
  --device auto \  # or: cpu, cuda, mps
  --limit 100 \
  --output-dir results/trocr-local
```

### Common Options (All Models)

- `--limit N`: Number of images to evaluate (default: 1000)
- `--output-dir PATH`: Custom output directory
- `--temperature`: API model temperature (default: 0.0)

## Output Structure

Each run generates:

```
results/<run-name>/
├── run_config.json          # Run configuration
├── summary.json             # Full summary with metrics
├── summary.md               # Human-readable summary
└── models/
    └── <model>/
        ├── summary.json     # Model-specific metrics
        ├── samples.jsonl    # Individual predictions (streamed live)
        └── progress.json    # Current progress status
```

### Metrics

- **CER** (Character Error Rate): Lower is better
- **WER** (Word Error Rate): Lower is better  
- **Exact Match**: Percentage of perfect predictions
- **Avg Latency**: Time per prediction (seconds)

## Compare Results

Compare multiple runs across different models:

```bash
python scripts/summarize_ocr_results.py \
  --output-prefix results/ocr_comparison \
  results/api-scalabs \
  results/surya-local \
  results/tesseract-local
```

This generates:
- `results/ocr_comparison.md` - Human-readable comparison table
- `results/ocr_comparison.json` - Raw comparison data

## Development

### Adding New Models

1. Create an adapter in `src/nepeval_ocr/adapters/`
2. Extend `BaseOCRAdapter` and implement `evaluate_sample_with_stats()`
3. Add a case in `run_unified_benchmark.py::load_adapter()`

### Testing

```bash
# Run mock tests
python test_mock_tesseract.py
```

## Attribution

This project uses the Nepali Pixel Synthetic OCR Benchmark:

- Dataset: [himalaya-ai/nepalipixel-synthetic-ocr-benchmark](https://huggingface.co/datasets/himalaya-ai/nepalipixel-synthetic-ocr-benchmark)
- Paper: [Koshur Pixel (arXiv:2606.23144)](https://arxiv.org/abs/2606.23144)

Please cite the dataset and paper when using these evaluation data.

## License

This project is provided as-is for research and evaluation purposes.

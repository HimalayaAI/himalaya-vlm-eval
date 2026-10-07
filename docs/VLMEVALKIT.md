# VLMEvalKit engine

The standard suites (OCRBench, DocVQA, ChartQA, MathVista, MMMU, MM-Vet …) run through
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit), the harness behind the
OpenCompass OpenVLM leaderboard, so prompts, answer extraction and scoring match it.
Facts below were verified against VLMEvalKit `main` @ `54a063c5` (2026-09-30).

## Setup

VLMEvalKit pins its own stack (torch even for API-only use), so it lives in its own
checkout and venv and is driven as a subprocess — never imported.

```bash
git clone https://github.com/open-compass/VLMEvalKit ~/src/VLMEvalKit
cd ~/src/VLMEvalKit && python3.11 -m venv .venv && .venv/bin/pip install -e .
export VLMEVALKIT_DIR=~/src/VLMEvalKit VLMEVALKIT_PYTHON=~/src/VLMEvalKit/.venv/bin/python

# judge for math/chat/chart suites (MCQ suites fall back to exact matching without one)
export NEPEVAL_JUDGE_API_KEY=sk-…                   # default judge: gpt-4o-mini on api.openai.com
# or OpenRouter: NEPEVAL_JUDGE_BASE_URL=https://openrouter.ai/api/v1 NEPEVAL_JUDGE_MODEL=openai/gpt-4o-mini

nepeval run --model gpt-4o,qwen3-vl-8b-instruct --bench category:math,ocrbench
```

Datasets download to `$LMUData` (default `~/LMUData`). olmOCR-Bench additionally needs
`vlmeval/dataset/olmOCRBench/eval_req.txt` installed in VLMEvalKit's venv.

## What the engine does

Per model × benchmark, one `python run.py --config <cfg> --work-dir results/vlmevalkit
--mode all --reuse` invocation:

- The model is any `openai_compat` catalog entry, passed as VLMEvalKit's `GPT4V` class
  with `api_base` = `<base_url>/chat/completions`. Tarka's `/ocr` models and in-process
  OCR engines cannot run these suites (they need chat completions).
- Dataset classes are resolved by asking VLMEvalKit itself (`DATASET_CLASSES`).
- `PRED_FORMAT=tsv`, `EVAL_FORMAT=csv`, so no Excel parsing is needed.
- The headline score is read from the result file named in the catalog
  (`catalog/benchmarks/vlmevalkit.yaml`) and rescaled to the declared scale; VLMEvalKit's
  own "primary metric" chooser is wrong for LLaVABench, MME, CCOCR, OCRBench_v2,
  OmniDocBench and MathVerse, and mixes 0–1 and 0–100.
- Failed API calls are counted from the prediction TSV (`Failed to obtain answer…`) and
  gate publishing like native errors.
- `source.harness_version` = VLMEvalKit commit; `source.judge` = judge model.

## Guard rails

| VLMEvalKit behaviour | What the engine does |
|---|---|
| A `.env` in the checkout overrides the process environment unconditionally | refuses to run while one exists |
| `sys.argv` (incl. `--key`, `--judge-key`) is written into `status.json` | keys go only in a 0600 config file deleted after the run, and the child env |
| Judge names are used in output file paths; `/` breaks them | slash-free alias on `--judge`, real id via `LOCAL_LLM` |
| CharXiv names files after the real judge id | refuses a judge id with `/` for CharXiv |
| `MMEVAL_ROOT`, `OPENAI_API_BASE`, `LOCAL_LLM` change output dir / endpoints | stripped from the child env, then set explicitly |
| `--judge` applies to every dataset in one invocation | one invocation per benchmark |
| A model name containing `gemini` drops `max_tokens`; `gpt-5`/`o1`/`o3`/`o4` switch to `max_completion_tokens` | inherited (it is VLMEvalKit's method); recorded in the run config |

## Headline score per benchmark

| Benchmark | File | Value | Stored scale |
|---|---|---|---|
| OCRBench | `{P}_score.json` | `Final Score` | 0–1000 |
| OCRBench v2 EN / ZH | `{P}_score.json` | `English/Chinese Overall Score` ×100 | 0–100 |
| CC-OCR | `{P}_acc.csv` | `total` (×100 if ≤ 1) | 0–100 |
| olmOCR-Bench | `olmOCRBench_eval_summary.csv` | row `type=overall`, `score` | 0–100 |
| DocVQA / InfoVQA / ChartQA / TextVQA | `{P}_acc.csv` | `Overall` | 0–100 |
| OmniDocBench | `{P}_overall.tsv` | `overall_EN` (edit distance, lower better) | 0–1 |
| AI2D, MMStar, RealWorldQA, SEED-Bench-2-Plus, BLINK, MMBench | `{P}_acc.csv` | `Overall` ×100 | 0–100 |
| MMMU | `{P}_acc.csv` | row `split=validation`, `Overall` ×100 | 0–100 |
| CharXiv | `{P}_*_acc.csv` | `Overall` ×100 | 0–100 |
| MathVista, LogicVista | `{P}_*_score.csv` | row `Task&Skill=Overall`, `acc` | 0–100 |
| MathVerse (vision-only) | `{P}_*_score.csv` | `Overall` | 0–100 |
| MATH-Vision (+mini) | `{P}_*_score.csv` | row `Subject=Overall`, `acc` | 0–100 |
| We-Math | `{P}_*_score.csv` | `Score (Strict)` (percent string) | 0–100 |
| DynaMath | `{P}_*_score.csv` | row `Setting=Average`, `Overall` ×100 | 0–100 |
| MM-Vet | `{P}_*_score.csv` | row `Category=Overall`, `acc` | 0–100 |
| LLaVA-Bench | `{P}_score_*.csv` | row `split=overall`, `Relative Score (main)` | 0–100 |
| HallusionBench | `{P}_score.csv` | row `split=Overall`, `Avg` | 0–100 |
| POPE | `{P}_score.csv` | row `split=Overall`, `Overall` (F1) | 0–100 |
| MME | `{P}_score.csv` | `perception + reasoning` | 0–2800 |

`{P}` = `<model id>_<dataset>`. Unverified choices (from code reading, not a run): the
leaderboard may report DynaMath worst-case and We-Math loose — change the row/column in
the catalog if so; the catalog is the only place these live.

# LLM for astronomy — ZTF / ALeRCE stamp benchmark

Vision–language benchmark on **ZTF** alerts brokered by [ALeRCE](https://science.alerce.online/): tabular metadata plus a **single RGB stamp montage** per object (science, reference/template, difference). API reference: [api.alerce.online](https://api.alerce.online/ztf/v1).

## Benchmark data (Hugging Face)

The public **primary benchmark assets** (1 500-row manifest, stamp montages, second-rollout ablation files, and Hub dataset card) are released on the Hugging Face Hub:

[https://huggingface.co/datasets/AnonymousUser16384/AstroAlertBench](https://huggingface.co/datasets/AnonymousUser16384/AstroAlertBench)

Clone or download that dataset, then point this codebase at the files (e.g. place montages under **`stamps_llm_updated/`** at the repo root, or set **`ZTF_STAMPS_LLM_DIR`** to the directory that contains the class/`oid`/`montage.png` layout).

**This Git repository** is intended for **code** (runners, prompts, scorers): **`api_settings/`**, **`prompts/`**, **`evaluate/`**, **`run/`**, and small local copies of manifests if you keep them for development. Avoid re-hosting the full PNG tree on GitHub when the Hub dataset is the canonical source.

You can rerun models and scoring **without** downloading FITS again once montages are available locally. Montage paths come from the **`prompts`** package (default dirname **`stamps_llm_updated`**) unless you set **`ZTF_STAMPS_LLM_DIR`**.

## Setup

```bash
pip install -r requirements.txt
```

Copy **`.env.example`** → **`.env`** (the real `.env` is not committed) and add API keys for the backends you use.

Work **from the repository root** so relative paths like `data/...` match the scripts.

The benchmark manifest already includes broker fields the scorer expects (e.g. **`fid`**, **`isdiffpos`**, **`fid_band`**). For a custom CSV, match that schema; **`data/manifest_enriched.csv`** is a tracked example of the enriched column layout.

## Source layout (main Python)

| Directory | Role |
|-----------|------|
| **`api_settings/`** | `api_tinker.py`, `api_openai.py`, `api_anthropic.py`, `api_google.py` |
| **`prompts/`** | `prompts.py` (default task text + schema), `prompts_*.py` variants (same API surface) |
| **`evaluate/`** | `evaluate.py` (metrics / JSONL scoring), `evaluate_second_rollout_ablation.py` |
| **`run/`** | `run_tinker_benchmark.py`, `retry_failed.py`, `run_second_rollout_benchmark.py`, `_runmeta.py` (wall-clock sidecar) |

The batch scripts under **`run/`** prepend **`api_settings/`**, **`evaluate/`**, **`prompts/`**, and the repo root to `sys.path` so existing **`import api_tinker`**, **`from evaluate import …`**, **`import prompts`**, and **`importlib.import_module("prompts_agn_instruction")`**-style `--prompts` values keep working. You can invoke them as **`python run/run_tinker_benchmark.py …`** or **`python -m run.run_tinker_benchmark …`** from the root.

If **`viz/`** is missing from your tree, those runners fall back to **`run._runmeta`** instead of **`viz._runmeta`** for the runmeta sidecar.

## Zero-shot evaluation (structured JSON, Parts A–C)

**Backends** (env vars in **`.env`** / environment):

| Used by | Env / notes |
|---------|-------------|
| `api_settings/api_tinker.py` | `TINKER_API_KEY` — [Tinker console](https://tinker-console.thinkingmachines.ai/) |
| `api_settings/api_openai.py` | `OPENAI_API_KEY` |
| `api_settings/api_anthropic.py` | `ANTHROPIC_API_KEY` (install `anthropic` if needed) |
| `api_settings/api_google.py` | `GOOGLE_API_KEY` |

**Prompt modules:** default is the package **`prompts`** (implemented in **`prompts/prompts.py`**). Alternatives are the same filenames as before, e.g. **`prompts_agn_instruction`**, **`prompts_second_roll_out_ablation`**, **`prompts_expert_example_ablation`**, **`prompts_token_limit_instruction`** (passed to **`--prompts`** on the run scripts).

Example (Tinker batch + scorer):

```bash
# Windows PowerShell; use export on Unix.
$env:TINKER_API_KEY="your_key"
python run/run_tinker_benchmark.py --manifest data/manifest_benchmark_final.csv --out predictions_run1.jsonl --model moonshotai/Kimi-K2.5 --concurrency 32 --prompts prompts
python -m evaluate.evaluate --predictions predictions_run1.jsonl --manifest data/manifest_benchmark_final.csv
```

`--model` overrides **`TINKER_MODEL`** in `.env`; `--concurrency` parallelizes API calls (start at **32** for Tinker unless your rate limits are tight); **`--prompts`** names the module under **`prompts/`** (default **`prompts`**; try **`prompts_agn_instruction`** for the AGN vs VS guidance variant). Other backends add **`--backend`** and matching keys — see the header examples in **`run/run_tinker_benchmark.py`**.

- **Images:** one montage PNG per row; layout under **`stamps_llm_updated`** (or **`ZTF_STAMPS_LLM_DIR`**).
- **Part A gold:** manifest vs [ZTF Avro alert schema](https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html) semantics; **`fid`**, **`isdiffpos`**, and related fields must match what the runner expects.
- **Part C gold:** `target_class` (SN, AGN, VS, asteroid, bogus).

**Second-rollout ablation:** `python run/run_second_rollout_benchmark.py …`, then **`python -m evaluate.evaluate_second_rollout_ablation …`** — see **`data_second_roll_out_ablation/METRICS_SPEC.md`** and **`data_second_roll_out_ablation/README.md`**.

## Reports and figures (in-repo)

**`results_comparison/report/`** holds narrative markdown and **`charts/`** PNGs. Figures are checked in for reading and reuse in papers.

# Log — second-rollout **five** experiments (model conditions)

**What this is:** The low-confidence ablation is **five independent second-pass sweeps**—one per **closed** model. Each run draws **n = 35** gold-labelled alerts from **that** model’s own first-pass benchmark pool where **mean Part B self-score (3 dimensions) &lt; 4** and Part C parses to a 5-class label, with **stratified** class counts following Hamilton–largest-remainder quotas (`random_seed = 42` in the build that produced the manifests). The second pass uses the **same** **`prompts`** package system rubric plus **second-trial** instructions; the user prompt embeds the model’s **verbatim first-pass JSON** (gold **not** revealed). Prompt module: **`prompts.second_roll_out_ablation`**. Evaluation: **`python -m evaluate.second_rollout_ablation`**; definitions: `data_second_roll_out_ablation/METRICS_SPEC.md`.

**Source of per-model OIDs, quotas, and first-pass `jsonl` paths:** `data_second_roll_out_ablation/ablation_summary.json`.  
**Per-model n=35 row manifests (CSV):** `data_second_roll_out_ablation/metadata/<slug>_n35.csv` (see table).

**Per-model second-pass output pattern (convention; produced after a successful run):**

- Predictions: `results/second_rollout_<slug>_n35.jsonl`
- Metrics:     `results/second_rollout_<slug>_n35.metrics.json`

**Aggregated report and charts (2026-04-25):** `results_comparison/report/20260425_report_second_rollout_ablation.md` — figure PNGs from `python -m viz._make_charts_second_rollout_apr25` → `results_comparison/report/charts/second_rollout_apr25/`.

---

| # | Slug | Model (as in figures) | Low-conf pool size | Second-pass n | Per-model n=35 manifest |
|---|------|------------------------|--------------------:|----------------:|---------------------------|
| 1 | `gpt54_high` | GPT-5.4 high (reasoning) | 118 | 35 | `metadata/gpt54_high_n35.csv` |
| 2 | `gpt54_none` | GPT-5.4 none (no reasoning) | 36 | 35 | `metadata/gpt54_none_n35.csv` |
| 3 | `gemini25_flash_none` | Gemini 2.5 Flash (`thinking_budget=0`) | 83 | 35 | `metadata/gemini25_flash_none_n35.csv` |
| 4 | `opus47_think` | Claude Opus 4.7 (adaptive thinking) | 498 | 35 | `metadata/opus47_think_n35.csv` |
| 5 | `opus47_nothink` | Claude Opus 4.7 (thinking off) | 494 | 35 | `metadata/opus47_nothink_n35.csv` |

**First-pass full-benchmark JSONL used to define “who was in the low-confidence pool” (per `ablation_summary.json`):** `results/benchmark_<slug style>.jsonl` — the exact `jsonl` key per model is stored under each model in `ablation_summary.json` (e.g. `results/benchmark_gpt54_high.jsonl` for row 1).

**Important design note:** The **35 object IDs are not the same** across the five conditions; each line of the ablation is **paired** only **within** a model (first vs second pass on the same OIDs for that model).

---

## Repro checklist (per slug)

1. Build / confirm manifests and priors if your pipeline uses them (`metadata/`, `priors/` as applicable).
2. Run the second pass with default `--prompts prompts.second_roll_out_ablation` (or equivalent) and point at the n=35 manifest row list for that model.
3. **`python -m evaluate.second_rollout_ablation`** (or your wrapper) on the new JSONL → `*.metrics.json`.
4. Regenerate figures: `python -m viz._make_charts_second_rollout_apr25` from repository root.

---

## Related (reporting & visualization, same project)

- **Narrative report** — `results_comparison/report/20260425_report_second_rollout_ablation.md` (tables + embedded figures, integrated prose).
- **Chart pipeline** — `viz/_make_charts_second_rollout_apr25.py` → `results_comparison/report/charts/second_rollout_apr25/*` (numeric annotations on all panels; **Fig 5** is a 2D scatter of ΔMSRS vs P(MSRS ≥ 4) with marker size ∝ second-pass accuracy, plus horizontal bar details for the two metrics).

---

*Logged for traceability: 2026-04-18.*

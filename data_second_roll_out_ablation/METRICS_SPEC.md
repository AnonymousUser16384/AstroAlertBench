# Second-rollout ablation — metric specification (final)

Paired design: for each model, the same **n = 35** alerts are scored on **first pass** (full-benchmark JSONL) and **second pass** (re-prompt with prior JSON embedded; no gold label revealed).

Low-confidence pool (per model): rows where **mean of the three Part B self-scores** is **strictly &lt; 4**, and Part C parses to a valid 5-class prediction (same rule as `evaluate.scoring` Part B↔C linkage). Gold class = manifest `target_class`.

Stratified n = 35: proportional to the **class counts inside that model’s low-confidence pool** (Hamilton largest-remainder rounding to exactly 35). Within each class, random sample without replacement (`--random-seed`, default 42).

---

## 1. Correction flow (primary)

| Metric | Definition | Notes |
|--------|--------------|--------|
| **Correction rate** | Among first-pass **wrong** 5-class predictions, fraction second-pass **correct**: \((W \to C') / (W \to C' + W \to W)\). | Primary “reflection power”. |
| **Damage rate** | Among first-pass **correct**, fraction second-pass **wrong**: \((C \to W) / (C \to C + C \to W)\). | Should stay near 0 for stable models. |
| **Persistence rate** | Among first-pass **wrong**, fraction still **wrong**: \(W \to W\) over same denominator as correction rate. | \(= 1 - \) correction rate. |
| **Net accuracy delta (ΔA)** | \(\text{mean}(\mathbf{1}[\text{correct}_2]) - \text{mean}(\mathbf{1}[\text{correct}_1])\) on the **same 35 rows**. | Signed; can be negative if damage dominates. |
| **Contingency counts** | `n_cc`, `n_cw`, `n_wc`, `n_ww` | Full 2×2 for appendix tables. |

5-class correctness: `evaluate.scoring.stages_to_final_class` on normalized Part C stages vs manifest `target_class` (SN / VS / AGN / bogus / asteroid).

---

## 2. Calibration & self-awareness (Part B / C)

| Metric | Definition |
|--------|-------------|
| **MSRS shift (ΔMSRS)** | Mean(second self-mean − first self-mean) on the 35 rows; per-row self-mean = mean of the three integer self-scores when all present, else `null` excluded from mean. | “Louder after fixing?” |
| **Δ confidence_overall** | If both passes expose `confidence_overall` in Part B, report mean delta; else `null` with `n_linked`. | Optional; many current JSONLs omit this key. |
| **Second-pass MSRS ≥ 4 rate** | Among rows with a valid second-pass self-mean, fraction with mean ≥ 4. (The n=35 cohort is *defined* by first-pass self-mean &lt; 4, so this equals “high-confidence recovery” on the subset where the model re-emits all three self-scores.) | Descriptive; not a p-value. |
| **Token overhead (second / first)** | Mean ratio `n_output_tokens_2 / n_output_tokens_1` (and same for `n_answer_tokens` when present). | “Cost of correction”; guard against divide-by-zero. |
| **Prior-reference rate** | Fraction of second-pass Part B text (concatenated fields) matching `(?i)(previous|prior|earlier|first (pass|attempt|analysis)|last time|my earlier)` | Cheap proxy for explicit self-review language. |
| **Stage-3 label flip** | `stage3` string differs first vs second (after normalization). | Structural change, not logical “contradiction”. |
| **Reasoning contradiction (manual / LLM)** | **Not auto-scored in code.** Both Part B blobs are echoed in the metrics JSON under `qualitative.part_b_first` / `part_b_second` for appendix or blind human / judge-model coding. | Avoids brittle NLP on scientific prose. |

---

## 3. Statistical rigor (appendix)

| Metric | Definition |
|--------|------------|
| **McNemar exact two-sided p-value** | Discordant pairs only: `b = n_wc`, `c = n_cw`, `n = b+c`. Under H₀, `b ~ Binomial(n, 0.5)`. Two-sided exact p-value = `2 * min(P(X≤min(b,c)), P(X≥max(b,c)))` (capped at 1). | No `scipy` dependency. |
| **Sample class table** | Counts of `target_class` in the 35-row manifest (reported per model). | Preempts class-bias critique. |
| **Low-confidence pool table** | Total low-conf count and per-class counts **before** subsampling. | Shows how harsh the filter is per model. |

---

## 4. Outputs produced by tooling

- Maintainer checkouts may include scripts that rebuild `metadata/*.csv`, `priors/<slug>/*.json`, and `ablation_summary.json` (pool stats + chosen OIDs).
- **`python -m run.run_second_rollout_benchmark`** — runs second pass; writes JSONL with `ablation_prior_file`, `first_pass_*` echoes for evaluation.
- **`python -m evaluate.second_rollout_ablation`** — reads first snapshot from each JSONL row + second model output; writes metrics JSON (e.g. `second_rollout_<slug>.metrics.json`) and optional markdown snippet.

---

## 5. Explicitly out of scope (for this codebase pass)

- Automated **semantic contradiction** detection between two scientific rationales (deferred to human/LLM judge; see §2).
- Cross-model comparison of p-values (each model has its **own** 35-OID set; only within-model paired tests are defined here).

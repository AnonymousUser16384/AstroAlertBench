"""
Evaluation for AstroAlertBench-style JSON outputs (Parts A-C).

Implements the full metric suite from the AstroAlertBench appendix:
  Part A: per-question accuracy, macro accuracy, exact-match rate
  Part B: mean self reasoning score (MSRS), per-dimension scores, pass rate
  Part C: stagewise accuracy, conditional accuracy, end-to-end staged accuracy,
          final 5-class accuracy, precision/recall/F1, Stage-3 confusion matrix

External-judge (MERS) and human scores are not computed here; they require a
separate judge pipeline.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES_5 = ["SN", "AGN", "VS", "asteroid", "bogus"]
STAGE3_LABELS = ["supernova", "variable_star", "AGN"]
PART_A_QUESTIONS = [
    "filter_band",
    "subtraction_sign",
    "magpsf",
    "sigmapsf",
    "ndethist",
    "ncovhist",
]
PART_B_SELF_SCORE_KEYS = [
    "self_score_key_evidence",
    "self_score_leading_interpretation_and_support",
    "self_score_alternative_analysis",
]

GOLD_STAGES: dict[str, tuple[str, str, str]] = {
    "bogus":    ("artifact",    "N/A",           "N/A"),
    "asteroid": ("real_object", "solar_system",  "N/A"),
    "SN":       ("real_object", "astrophysical", "supernova"),
    "AGN":      ("real_object", "astrophysical", "AGN"),
    "VS":       ("real_object", "astrophysical", "variable_star"),
}

FLOAT_TOLERANCE = 0.1

# Required top-level keys in a well-formed answer JSON.
REQUIRED_TOP_KEYS = (("Part A", "part_a"), ("Part B", "part_b"), ("Part C", "part_c"))

# Allowed enum values for Part C stages.
STAGE1_VALUES = {"real_object", "artifact"}
STAGE2_VALUES = {"solar_system", "astrophysical", "N/A"}
STAGE3_VALUES = {"supernova", "variable_star", "AGN", "N/A"}

# Format error codes (parser layer, mutually exclusive).
FORMAT_OK = "ok"
FORMAT_TRUNCATED_NO_JSON = "truncated_no_json"
FORMAT_TRUNCATED_PARTIAL_JSON = "truncated_partial_json"
FORMAT_PARSE_FAILED = "parse_failed"
FORMAT_SCHEMA_MISSING_TOP_LEVEL = "schema_missing_top_level"
FORMAT_SCHEMA_WRONG_TYPE = "schema_wrong_type"
FORMAT_EXTRA_TEXT_AROUND_JSON = "extra_text_around_json"

# Common refusal phrases (case-insensitive substring match).
REFUSAL_PHRASES = (
    "i cannot determine",
    "i can't determine",
    "i am unable to",
    "i'm unable to",
    "cannot classify",
    "insufficient information to classify",
    "i refuse",
)

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse first JSON object from model output.

    Handles ```json fences, <think>...</think> blocks, and nested braces.
    """
    if not text or not text.strip():
        return None
    s = text.strip()
    s = re.sub(r"<think>[\s\S]*?</think>", "", s).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    depth = 0
    start = -1
    for idx, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(s[start : idx + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                start = -1
    return None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _norm_str(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def normalize_stage1(val: Any) -> str | None:
    s = _norm_str(val)
    if s is None:
        return None
    low = s.lower().replace(" ", "_")
    if low in ("artifact", "artefact"):
        return "artifact"
    if low in ("real_object", "real"):
        return "real_object"
    return None


def normalize_stage2(val: Any) -> str | None:
    s = _norm_str(val)
    if s is None:
        return None
    low = s.lower().replace(" ", "_")
    if low in ("n/a", "na", "none", "null"):
        return "N/A"
    if low in ("solar_system",):
        return "solar_system"
    if "solar" in low:
        return "solar_system"
    if low in ("astrophysical", "astro"):
        return "astrophysical"
    if "astrophys" in low:
        return "astrophysical"
    return None


def normalize_stage3_label(val: Any) -> str | None:
    """Normalize to prompt-level stage3 labels: supernova, variable_star, AGN, N/A."""
    s = _norm_str(val)
    if s is None:
        return None
    low = s.lower().replace(" ", "_")
    if low in ("n/a", "na", "none", "null"):
        return "N/A"
    if low in ("supernova", "sn"):
        return "supernova"
    if low in ("variable_star", "vs"):
        return "variable_star"
    if low in ("agn",):
        return "AGN"
    if "supernova" in low:
        return "supernova"
    if "variable" in low:
        return "variable_star"
    if "agn" in low or "active galactic" in low.replace("_", " "):
        return "AGN"
    return None


def stages_to_final_class(s1: str, s2: str, s3: str) -> str | None:
    """Map predicted stage tuple to 5-class label (manifest convention)."""
    if s1 == "artifact":
        return "bogus"
    if s1 == "real_object":
        if s2 == "solar_system":
            return "asteroid"
        if s2 == "astrophysical":
            if s3 == "supernova":
                return "SN"
            if s3 == "AGN":
                return "AGN"
            if s3 == "variable_star":
                return "VS"
    return None


# ---------------------------------------------------------------------------
# Part A evaluation helpers
# ---------------------------------------------------------------------------


def _parse_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    m = re.match(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _parse_int(val: Any) -> int | None:
    if val is None:
        return None
    s = str(val).strip()
    m = re.match(r"[-+]?\d+", s)
    if m:
        try:
            return int(m.group())
        except ValueError:
            return None
    return None


def check_part_a_question(q: str, pred_val: Any, gold_val: Any) -> bool:
    """Check one Part A metadata-grounding question."""
    if pred_val is None or gold_val is None:
        return False

    if q in ("filter_band", "subtraction_sign"):
        return str(pred_val).strip().lower() == str(gold_val).strip().lower()

    if q in ("magpsf", "sigmapsf"):
        pf = _parse_float(pred_val)
        gf = _parse_float(gold_val)
        if pf is None or gf is None:
            return False
        return abs(pf - gf) < FLOAT_TOLERANCE

    if q in ("ndethist", "ncovhist"):
        pi = _parse_int(pred_val)
        gi = _parse_int(gold_val)
        if pi is None or gi is None:
            return False
        return pi == gi

    return False


def get_gold_part_a(row: pd.Series) -> dict[str, Any]:
    """Extract Part A gold answers from an enriched manifest row."""
    isdiffpos_raw = row.get("isdiffpos")
    if pd.notna(isdiffpos_raw):
        isdiffpos = "positive" if str(isdiffpos_raw).lower() in ("t", "1", "true") else "negative"
    else:
        isdiffpos = None

    ndethist = row.get("alert_ndethist")
    if pd.isna(ndethist):
        ndethist = row.get("ndethist")
    ncovhist = row.get("alert_ncovhist")
    if pd.isna(ncovhist):
        ncovhist = row.get("ncovhist")

    return {
        "filter_band": row.get("fid_band") if pd.notna(row.get("fid_band")) else None,
        "subtraction_sign": isdiffpos,
        "magpsf": row.get("magpsf") if pd.notna(row.get("magpsf")) else None,
        "sigmapsf": row.get("sigmapsf") if pd.notna(row.get("sigmapsf")) else None,
        "ndethist": ndethist if pd.notna(ndethist) else None,
        "ncovhist": ncovhist if pd.notna(ncovhist) else None,
    }


# ---------------------------------------------------------------------------
# Part B evaluation helpers (self-scores; external judge is separate)
# ---------------------------------------------------------------------------


def extract_self_scores(part_b: dict) -> list[int | None]:
    """Extract the 3 self-score values from a Part B dict."""
    scores: list[int | None] = []
    for key in PART_B_SELF_SCORE_KEYS:
        val = part_b.get(key)
        if val is not None:
            try:
                s = int(val)
                scores.append(s if 0 <= s <= 5 else None)
            except (ValueError, TypeError):
                scores.append(None)
        else:
            scores.append(None)
    return scores


# ---------------------------------------------------------------------------
# Generic binary P/R/F1
# ---------------------------------------------------------------------------


def _binary_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Error categorization (format + value)
# ---------------------------------------------------------------------------


def _get_top_key(parsed: dict, key_pair: tuple[str, str]) -> Any:
    for k in key_pair:
        if k in parsed:
            return parsed[k]
    return None


def categorize_format_error(
    answer_text: str,
    raw_text: str,
    parsed: dict | None,
    truncated: bool,
) -> str:
    """Classify the parser-side outcome of one record.

    Returns one of FORMAT_* codes. Mutually exclusive — first matching condition
    wins. The order of checks reflects severity (truncation > parse fail > schema).
    """
    text_for_check = answer_text if answer_text else raw_text or ""
    has_open_brace = "{" in text_for_check

    if parsed is None:
        if truncated:
            return FORMAT_TRUNCATED_PARTIAL_JSON if has_open_brace else FORMAT_TRUNCATED_NO_JSON
        return FORMAT_PARSE_FAILED

    if not isinstance(parsed, dict):
        return FORMAT_SCHEMA_WRONG_TYPE

    for key_pair in REQUIRED_TOP_KEYS:
        val = _get_top_key(parsed, key_pair)
        if val is None:
            return FORMAT_SCHEMA_MISSING_TOP_LEVEL
        if not isinstance(val, dict):
            return FORMAT_SCHEMA_WRONG_TYPE

    # JSON is structurally valid. If significant text surrounds it, flag (warning,
    # not a hard fail — record still scores normally).
    stripped = (answer_text or "").strip()
    if stripped:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first > 50 or (last != -1 and len(stripped) - last - 1 > 50):
            return FORMAT_EXTRA_TEXT_AROUND_JSON

    return FORMAT_OK


def _is_na(val: Any) -> bool:
    return val is None or normalize_stage2(val) == "N/A" or normalize_stage3_label(val) == "N/A"


def find_value_errors(
    parsed: dict,
    gold_a: dict | None,
    raw_text: str = "",
) -> list[str]:
    """Return list of value-error codes for one record. Codes can co-occur.

    Includes:
      - part_a_wrong_<field> / part_a_missing_<field>
      - Part C cross-field consistency rules (c1_*, c2_*)
      - enum violations
      - Part B presence + score-range checks
      - refusal detection
    """
    errors: list[str] = []

    # ---- Part A field-level ----
    part_a = _get_top_key(parsed, ("Part A", "part_a"))
    if isinstance(part_a, dict) and gold_a is not None:
        for q in PART_A_QUESTIONS:
            gold_val = gold_a.get(q)
            if gold_val is None:
                continue
            pred_val = part_a.get(q)
            if pred_val is None:
                errors.append(f"part_a_missing_{q}")
            elif not check_part_a_question(q, pred_val, gold_val):
                errors.append(f"part_a_wrong_{q}")

    # ---- Part C consistency ----
    part_c = _get_top_key(parsed, ("Part C", "part_c"))
    if isinstance(part_c, dict):
        s1_raw = part_c.get("stage1")
        s2_raw = part_c.get("stage2")
        s3_raw = part_c.get("stage3")
        s1 = normalize_stage1(s1_raw)
        s2 = normalize_stage2(s2_raw)
        s3 = normalize_stage3_label(s3_raw)

        if s1_raw is not None and s1 is None:
            errors.append("enum_violation_stage1")
        if s2_raw is not None and s2 is None:
            errors.append("enum_violation_stage2")
        if s3_raw is not None and s3 is None:
            errors.append("enum_violation_stage3")

        if s1 == "artifact":
            if s2 not in (None, "N/A"):
                errors.append("c1_artifact_must_zero_others")
            elif s3 not in (None, "N/A"):
                errors.append("c1_artifact_must_zero_others")
        elif s1 == "real_object":
            if s2 == "N/A":
                errors.append("c1_real_requires_stage2")
            if s2 == "solar_system" and s3 not in (None, "N/A"):
                errors.append("c2_solar_must_zero_stage3")
            if s2 == "astrophysical" and s3 not in {"supernova", "AGN", "variable_star"}:
                errors.append("c2_astro_requires_subtype")

    # ---- Part B presence + ranges ----
    part_b = _get_top_key(parsed, ("Part B", "part_b"))
    if not isinstance(part_b, dict):
        errors.append("part_b_missing")
    else:
        for dim in ("key_evidence", "leading_interpretation_and_support", "alternative_analysis"):
            if part_b.get(dim) in (None, ""):
                errors.append(f"part_b_missing_{dim}")
        for k in PART_B_SELF_SCORE_KEYS:
            v = part_b.get(k)
            if v is None:
                continue
            try:
                iv = int(v)
                if not (1 <= iv <= 5):
                    errors.append(f"part_b_score_out_of_range_{k}")
            except (ValueError, TypeError):
                errors.append(f"part_b_score_out_of_range_{k}")
        conf = part_b.get("confidence_overall")
        if conf is not None:
            try:
                cf = float(conf)
                if not (1 <= cf <= 5):
                    errors.append("part_b_confidence_out_of_range")
            except (ValueError, TypeError):
                errors.append("part_b_confidence_out_of_range")

    # ---- Refusal detection ----
    low = (raw_text or "").lower()
    if any(p in low for p in REFUSAL_PHRASES):
        errors.append("refusal")

    return errors


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_jsonl(
    predictions_path: Path,
    manifest_path: Path,
    write_back_errors: bool = True,
) -> dict[str, Any]:
    """Compute full AstroAlertBench metrics from a JSONL predictions file.

    If `write_back_errors` is True (default), each row is augmented in-place with
    `error_category` (one FORMAT_* code) and `value_errors` (list of codes), and
    the JSONL file is rewritten. This lets downstream tools (notebooks, the
    log-experiment skill) inspect why specific rows failed without re-running.
    """
    manifest = pd.read_csv(manifest_path, low_memory=False)
    gold_map = manifest.set_index("oid")

    rows: list[dict] = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n = len(rows)
    n_errors = sum(1 for r in rows if r.get("error"))
    json_ok = 0

    # Error category accumulators
    format_counts: Counter[str] = Counter()
    value_error_counts: Counter[str] = Counter()

    # Part A accumulators
    a_correct: Counter[str] = Counter()
    a_total: Counter[str] = Counter()
    a_exact_match = 0
    a_exact_total = 0

    # Part B accumulators
    b_self_scores: list[list[int]] = []

    # Part B-C linkage: per-example confidence vs correctness
    bc_confidence: list[float] = []
    bc_correct: list[int] = []

    # Part C accumulators
    stage1_preds: list[str] = []
    stage1_golds: list[str] = []
    stage2_preds: list[str] = []
    stage2_golds: list[str] = []
    stage3_preds: list[str] = []
    stage3_golds: list[str] = []
    stage2_cond_correct = 0
    stage2_cond_total = 0
    stage3_cond_correct = 0
    stage3_cond_total = 0
    all_stages_correct = 0
    all_stages_total = 0
    final_correct = 0
    final_total = 0
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()

    for r in rows:
        if r.get("error"):
            r["error_category"] = "runtime_error"
            r["value_errors"] = []
            format_counts["runtime_error"] += 1
            continue
        oid = r["oid"]
        tc = r.get("target_class")
        if tc is None and oid in gold_map.index:
            tc = gold_map.loc[oid, "target_class"]
        raw = r.get("raw_text", "")
        answer = r.get("answer_text", raw)
        parsed = r.get("parsed")
        if parsed is None:
            parsed = extract_json_object(answer)
        truncated = bool(r.get("truncated"))

        # ---- Format error categorization (one code per row) ----
        fmt = categorize_format_error(answer, raw, parsed, truncated)
        r["error_category"] = fmt
        format_counts[fmt] += 1

        # ---- Gold Part A (used by both value-error finder and Part A scoring) ----
        gold_a = None
        if oid in gold_map.index:
            gold_a = get_gold_part_a(gold_map.loc[oid])

        # ---- Value error categorization (multi-label) ----
        if isinstance(parsed, dict):
            verrs = find_value_errors(parsed, gold_a, raw_text=raw)
            r["value_errors"] = verrs
            for code in verrs:
                value_error_counts[code] += 1
        else:
            r["value_errors"] = []

        if parsed is None:
            continue
        json_ok += 1

        # ---- Part A ----
        part_a = parsed.get("Part A") or parsed.get("part_a")
        if isinstance(part_a, dict) and gold_a is not None:
            all_q_correct = True
            for q in PART_A_QUESTIONS:
                pred_val = part_a.get(q)
                gold_val = gold_a.get(q)
                if gold_val is not None:
                    a_total[q] += 1
                    if check_part_a_question(q, pred_val, gold_val):
                        a_correct[q] += 1
                    else:
                        all_q_correct = False
                else:
                    all_q_correct = False
            a_exact_total += 1
            if all_q_correct:
                a_exact_match += 1

        # ---- Part B (self-scores) ----
        part_b = parsed.get("Part B") or parsed.get("part_b")
        example_self_mean: float | None = None
        if isinstance(part_b, dict):
            scores = extract_self_scores(part_b)
            if all(s is not None for s in scores):
                b_self_scores.append(scores)  # type: ignore[arg-type]
                example_self_mean = float(np.mean(scores))

        # ---- Part C ----
        part_c = parsed.get("Part C") or parsed.get("part_c")
        if not isinstance(part_c, dict) or tc is None:
            continue

        gold_s = GOLD_STAGES.get(str(tc))
        if gold_s is None:
            continue
        gs1, gs2, gs3 = gold_s

        ps1 = normalize_stage1(part_c.get("stage1"))
        ps2 = normalize_stage2(part_c.get("stage2"))
        ps3 = normalize_stage3_label(part_c.get("stage3"))

        if ps1 is None or ps2 is None or ps3 is None:
            continue

        stage1_preds.append(ps1)
        stage1_golds.append(gs1)
        stage2_preds.append(ps2)
        stage2_golds.append(gs2)
        stage3_preds.append(ps3)
        stage3_golds.append(gs3)

        c1 = ps1 == gs1
        c2 = ps2 == gs2
        c3 = ps3 == gs3

        # Conditional Stage-2 accuracy (I_2: gold s1 = real_object)
        if gs1 == "real_object":
            stage2_cond_total += 1
            stage2_cond_correct += int(c2)

        # Conditional Stage-3 accuracy (I_3: gold s2 = astrophysical)
        if gs2 == "astrophysical":
            stage3_cond_total += 1
            stage3_cond_correct += int(c3)

        # End-to-end staged accuracy
        all_stages_total += 1
        c_all = c1 and c2 and c3
        all_stages_correct += int(c_all)

        # Final 5-class accuracy via g()
        pred_final = stages_to_final_class(ps1, ps2, ps3)
        if pred_final is not None:
            final_total += 1
            per_class_total[str(tc)] += 1
            is_correct = pred_final == str(tc)
            if is_correct:
                final_correct += 1
                per_class_correct[str(tc)] += 1
            if example_self_mean is not None:
                bc_confidence.append(example_self_mean)
                bc_correct.append(int(is_correct))

    # -----------------------------------------------------------------------
    # Assemble metrics
    # -----------------------------------------------------------------------
    metrics: dict[str, Any] = {
        "n_examples": n,
        "n_errors": n_errors,
        "json_parseable": json_ok,
        "json_valid_rate": round(json_ok / n, 4) if n else 0.0,
    }

    # Output token statistics (recorded per row by api_tinker.sample_vlm).
    # n_output_tokens covers the FULL generation (incl. reasoning); n_answer_tokens
    # covers only the post-thinking text. Tracking both lets us see how much
    # budget is being burned on internal reasoning vs the JSON answer.
    out_tok = [r["n_output_tokens"] for r in rows if isinstance(r.get("n_output_tokens"), int)]
    ans_tok = [r["n_answer_tokens"] for r in rows if isinstance(r.get("n_answer_tokens"), int)]
    if out_tok:
        out_arr = np.array(out_tok, dtype=int)
        metrics["output_tokens"] = {
            "mean": round(float(out_arr.mean()), 1),
            "median": int(np.median(out_arr)),
            "min": int(out_arr.min()),
            "max": int(out_arr.max()),
            "p95": int(np.percentile(out_arr, 95)),
        }
    if ans_tok:
        ans_arr = np.array(ans_tok, dtype=int)
        metrics["answer_tokens"] = {
            "mean": round(float(ans_arr.mean()), 1),
            "median": int(np.median(ans_arr)),
            "min": int(ans_arr.min()),
            "max": int(ans_arr.max()),
        }
    n_truncated = sum(1 for r in rows if r.get("truncated"))
    if any("truncated" in r for r in rows):
        metrics["n_truncated"] = n_truncated
        metrics["truncated_rate"] = round(n_truncated / n, 4) if n else 0.0

    # ---- Error breakdown (format codes are mutually exclusive; value codes co-occur) ----
    metrics["error_breakdown"] = {
        "format": dict(format_counts.most_common()),
        "value_top10": dict(value_error_counts.most_common(10)),
        "n_with_value_errors": sum(1 for r in rows if r.get("value_errors")),
    }

    # ---- Persist per-row categorization back to the JSONL ----
    if write_back_errors:
        with open(predictions_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Part A metrics
    per_q_acc = {}
    for q in PART_A_QUESTIONS:
        if a_total[q] > 0:
            per_q_acc[q] = round(a_correct[q] / a_total[q], 4)
    metrics["part_a_per_question_accuracy"] = per_q_acc
    if per_q_acc:
        metrics["part_a_macro_accuracy"] = round(
            sum(per_q_acc.values()) / len(per_q_acc), 4
        )
    metrics["part_a_exact_match_rate"] = (
        round(a_exact_match / a_exact_total, 4) if a_exact_total else 0.0
    )

    # Part B metrics (self-scores only)
    if b_self_scores:
        arr = np.array(b_self_scores, dtype=float)
        per_dim_mean = arr.mean(axis=0)
        row_means = arr.mean(axis=1)
        metrics["part_b_msrs"] = round(float(row_means.mean()), 4)
        metrics["part_b_per_dim_mean_self"] = {
            "key_evidence": round(float(per_dim_mean[0]), 4),
            "leading_interpretation": round(float(per_dim_mean[1]), 4),
            "alternative_analysis": round(float(per_dim_mean[2]), 4),
        }
        metrics["part_b_self_pass_rate"] = round(float((row_means >= 4).mean()), 4)

    # Part B-C confidence-accuracy correlation
    if len(bc_confidence) >= 5:
        conf_arr = np.array(bc_confidence)
        corr_arr = np.array(bc_correct, dtype=float)
        correct_mask = corr_arr == 1
        incorrect_mask = corr_arr == 0
        mean_conf_correct = float(conf_arr[correct_mask].mean()) if correct_mask.sum() > 0 else None
        mean_conf_incorrect = float(conf_arr[incorrect_mask].mean()) if incorrect_mask.sum() > 0 else None

        bc_metrics: dict[str, Any] = {
            "n_linked": len(bc_confidence),
            "mean_confidence_correct": round(mean_conf_correct, 4) if mean_conf_correct is not None else None,
            "mean_confidence_incorrect": round(mean_conf_incorrect, 4) if mean_conf_incorrect is not None else None,
        }
        if mean_conf_correct is not None and mean_conf_incorrect is not None:
            bc_metrics["calibration_gap"] = round(mean_conf_correct - mean_conf_incorrect, 4)

        # Point-biserial (Pearson) correlation: confidence vs binary correctness
        if conf_arr.std() > 1e-9:
            pearson_r = float(np.corrcoef(conf_arr, corr_arr)[0, 1])
            bc_metrics["pearson_r"] = round(pearson_r, 4)
        else:
            bc_metrics["pearson_r"] = None

        # Accuracy at high confidence (MSRS >= 4) vs low confidence (MSRS < 4)
        high_mask = conf_arr >= 4.0
        low_mask = conf_arr < 4.0
        if high_mask.sum() > 0:
            bc_metrics["accuracy_high_confidence"] = round(float(corr_arr[high_mask].mean()), 4)
            bc_metrics["n_high_confidence"] = int(high_mask.sum())
        if low_mask.sum() > 0:
            bc_metrics["accuracy_low_confidence"] = round(float(corr_arr[low_mask].mean()), 4)
            bc_metrics["n_low_confidence"] = int(low_mask.sum())

        metrics["part_bc_confidence_accuracy"] = bc_metrics

    # Part C metrics
    nn = len(stage1_preds)
    if nn > 0:
        metrics["part_c_n_evaluable"] = nn
        metrics["part_c_stage1_accuracy"] = round(
            sum(p == g for p, g in zip(stage1_preds, stage1_golds)) / nn, 4
        )
        metrics["part_c_stage2_accuracy"] = round(
            sum(p == g for p, g in zip(stage2_preds, stage2_golds)) / nn, 4
        )
        metrics["part_c_stage3_accuracy"] = round(
            sum(p == g for p, g in zip(stage3_preds, stage3_golds)) / nn, 4
        )

        if stage2_cond_total > 0:
            metrics["part_c_stage2_conditional_accuracy"] = round(
                stage2_cond_correct / stage2_cond_total, 4
            )
        if stage3_cond_total > 0:
            metrics["part_c_stage3_conditional_accuracy"] = round(
                stage3_cond_correct / stage3_cond_total, 4
            )

        metrics["part_c_end_to_end_staged_accuracy"] = (
            round(all_stages_correct / all_stages_total, 4)
            if all_stages_total
            else 0.0
        )

        # Final 5-class accuracy
        if final_total > 0:
            metrics["part_c_final_5class_accuracy"] = round(
                final_correct / final_total, 4
            )
            per_class_acc = {}
            for c in CLASSES_5:
                if per_class_total[c] > 0:
                    per_class_acc[c] = round(
                        per_class_correct[c] / per_class_total[c], 4
                    )
            metrics["per_class_accuracy"] = per_class_acc
            metrics["per_class_total"] = dict(per_class_total)
            metrics["per_class_correct"] = dict(per_class_correct)

        # Stage-1 P/R/F1 (positive class = real_object, over all N evaluable)
        s1_tp = sum(
            1
            for p, g in zip(stage1_preds, stage1_golds)
            if p == "real_object" and g == "real_object"
        )
        s1_fp = sum(
            1
            for p, g in zip(stage1_preds, stage1_golds)
            if p == "real_object" and g != "real_object"
        )
        s1_fn = sum(
            1
            for p, g in zip(stage1_preds, stage1_golds)
            if p != "real_object" and g == "real_object"
        )
        metrics["part_c_stage1_prf_real_object"] = _binary_prf(s1_tp, s1_fp, s1_fn)

        # Stage-2 P/R/F1 (positive class = astrophysical, over all N evaluable)
        s2_tp = sum(
            1
            for p, g in zip(stage2_preds, stage2_golds)
            if p == "astrophysical" and g == "astrophysical"
        )
        s2_fp = sum(
            1
            for p, g in zip(stage2_preds, stage2_golds)
            if p == "astrophysical" and g != "astrophysical"
        )
        s2_fn = sum(
            1
            for p, g in zip(stage2_preds, stage2_golds)
            if p != "astrophysical" and g == "astrophysical"
        )
        metrics["part_c_stage2_prf_astrophysical"] = _binary_prf(s2_tp, s2_fp, s2_fn)

        # Stage-3 macro-F1 + confusion matrix (conditional on I_3)
        s3_preds_cond = []
        s3_golds_cond = []
        for p3, g2, g3 in zip(stage3_preds, stage2_golds, stage3_golds):
            if g2 == "astrophysical":
                s3_preds_cond.append(p3)
                s3_golds_cond.append(g3)

        if s3_preds_cond:
            s3_f1s = {}
            for cls in STAGE3_LABELS:
                tp = sum(
                    1
                    for p, g in zip(s3_preds_cond, s3_golds_cond)
                    if p == cls and g == cls
                )
                fp = sum(
                    1
                    for p, g in zip(s3_preds_cond, s3_golds_cond)
                    if p == cls and g != cls
                )
                fn = sum(
                    1
                    for p, g in zip(s3_preds_cond, s3_golds_cond)
                    if p != cls and g == cls
                )
                s3_f1s[cls] = _binary_prf(tp, fp, fn)

            macro_f1 = sum(v["f1"] for v in s3_f1s.values()) / len(STAGE3_LABELS)
            metrics["part_c_stage3_macro_f1"] = round(macro_f1, 4)
            metrics["part_c_stage3_per_class_prf"] = s3_f1s

            pred_labels = STAGE3_LABELS + ["N/A"]
            cm: dict[str, dict[str, int]] = {
                g: {p: 0 for p in pred_labels} for g in STAGE3_LABELS
            }
            for p, g in zip(s3_preds_cond, s3_golds_cond):
                if g in cm:
                    col = p if p in cm[g] else "N/A"
                    cm[g][col] += 1
            metrics["part_c_stage3_confusion_matrix"] = cm

    return metrics


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def print_report(metrics: dict[str, Any]) -> None:
    def _print(d: dict, indent: int = 0) -> None:
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                _print(v, indent + 1)
            else:
                print(f"{prefix}{k}: {v}")

    _print(metrics)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate JSONL predictions vs manifest")
    ap.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL from run_tinker_benchmark.py",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest_enriched.csv"),
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of human-readable report",
    )
    ap.add_argument(
        "--no-write-back",
        action="store_true",
        help="Do NOT write per-row error_category and value_errors back into the JSONL.",
    )
    args = ap.parse_args()
    m = evaluate_jsonl(
        args.predictions, args.manifest, write_back_errors=not args.no_write_back
    )
    if args.json:
        print(json.dumps(m, indent=2))
    else:
        print_report(m)

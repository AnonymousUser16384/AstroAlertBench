"""
Evaluate paired first-vs-second rollout JSONL from the low-confidence ablation.

Reads ``--second-jsonl`` (output of ``run_second_rollout_benchmark.py``), loads
each row's first pass from ``ablation_prior_file``, and writes metrics including
correction/damage/persistence rates, Δaccuracy, ΔMSRS, token overhead, McNemar
exact p-value, and class tables.  See ``data_second_roll_out_ablation/METRICS_SPEC.md``.

Usage::

  python evaluate_second_rollout.py \\
    --second-jsonl results/second_rollout_gpt54_high_n35.jsonl \\
    --manifest data_second_roll_out_ablation/metadata/gpt54_high_n35.csv \\
    --out-json results/second_rollout_gpt54_high_n35.metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import (  # noqa: E402
    extract_json_object,
    extract_self_scores,
    normalize_stage1,
    normalize_stage2,
    normalize_stage3_label,
    stages_to_final_class,
)

_PRIOR_REF_RE = re.compile(
    r"(previous|prior|earlier|first (pass|attempt|trial|analysis)|last time|my earlier)",
    re.I,
)


def _parsed_self_mean(parsed: dict | None) -> float | None:
    if not isinstance(parsed, dict):
        return None
    part_b = parsed.get("Part B") or parsed.get("part_b")
    if not isinstance(part_b, dict):
        return None
    scores = extract_self_scores(part_b)
    if len(scores) != 3 or any(s is None for s in scores):
        return None
    return float(sum(scores)) / 3.0


def _confidence_overall(parsed: dict | None) -> float | None:
    if not isinstance(parsed, dict):
        return None
    part_b = parsed.get("Part B") or parsed.get("part_b")
    if not isinstance(part_b, dict):
        return None
    v = part_b.get("confidence_overall")
    if v is None:
        return None
    try:
        x = float(v)
        return x if 0 <= x <= 5 else None
    except (TypeError, ValueError):
        return None


def _final_correct(parsed: dict | None, gold_tc: str) -> bool | None:
    if not isinstance(parsed, dict):
        return None
    part_c = parsed.get("Part C") or parsed.get("part_c")
    if not isinstance(part_c, dict):
        return None
    ps1 = normalize_stage1(part_c.get("stage1"))
    ps2 = normalize_stage2(part_c.get("stage2"))
    ps3 = normalize_stage3_label(part_c.get("stage3"))
    if ps1 is None or ps2 is None or ps3 is None:
        return None
    pred = stages_to_final_class(ps1, ps2, ps3)
    if pred is None:
        return None
    return pred == str(gold_tc)


def _part_b_concat(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    part_b = parsed.get("Part B") or parsed.get("part_b")
    if not isinstance(part_b, dict):
        return ""
    chunks = []
    for k in ("key_evidence", "leading_interpretation_and_support", "alternative_analysis"):
        v = part_b.get(k)
        if isinstance(v, str) and v.strip():
            chunks.append(v)
    return "\n".join(chunks)


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(_binom_pmf(i, n, p) for i in range(0, k + 1))


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    """b = wrong→correct, c = correct→wrong; exact two-sided McNemar, Bin(n,0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    lo, hi = min(b, c), max(b, c)
    tail_lo = _binom_cdf(lo, n)
    tail_hi = 1.0 - _binom_cdf(hi - 1, n) if hi > 0 else 1.0
    return min(1.0, 2 * min(tail_lo, tail_hi))


def _load_first_parsed(rec: dict, gold_tc: str) -> tuple[dict | None, dict | None]:
    rel = rec.get("ablation_prior_file")
    if not rel:
        return None, None
    path = (ROOT / str(rel).replace("\\", "/")).resolve()
    if not path.is_file():
        return None, None
    prior = json.loads(path.read_text(encoding="utf-8"))
    fp = prior.get("first_pass") or {}
    parsed = fp.get("parsed")
    if not isinstance(parsed, dict):
        raw = fp.get("answer_text") or fp.get("raw_text") or ""
        parsed = extract_json_object(raw) if raw else None
    return fp, parsed if isinstance(parsed, dict) else None


def _stage3_norm(parsed: dict | None) -> str | None:
    if not isinstance(parsed, dict):
        return None
    part_c = parsed.get("Part C") or parsed.get("part_c")
    if not isinstance(part_c, dict):
        return None
    return normalize_stage3_label(part_c.get("stage3"))


def evaluate_pairs(
    rows: list[dict],
    manifest: pd.DataFrame,
    *,
    include_qualitative: bool = False,
) -> dict[str, Any]:
    gold_map = {str(r["oid"]): str(r["target_class"]) for _, r in manifest.iterrows()}

    c1_list: list[int] = []
    c2_list: list[int] = []
    d_msrs: list[float] = []
    second_msrs_values: list[float] = []
    d_conf: list[float] = []
    tok_ratios: list[float] = []
    ans_ratios: list[float] = []
    prior_ref_flags: list[int] = []
    stage3_flip: list[int] = []

    n_cc = n_cw = n_wc = n_ww = 0

    qualitative: list[dict[str, Any]] | None = [] if include_qualitative else None

    for rec in rows:
        oid = rec.get("oid")
        if not oid or rec.get("error"):
            continue
        gold = gold_map.get(str(oid))
        if gold is None:
            continue
        fp, p1 = _load_first_parsed(rec, gold)
        if p1 is None:
            continue
        p2 = rec.get("parsed")
        if not isinstance(p2, dict):
            p2 = extract_json_object(rec.get("answer_text") or rec.get("raw_text") or "")

        ok1 = _final_correct(p1, gold)
        ok2 = _final_correct(p2, gold) if isinstance(p2, dict) else None
        if ok1 is None or ok2 is None:
            continue

        b1, b2 = int(ok1), int(ok2)
        c1_list.append(b1)
        c2_list.append(b2)

        if b1 and b2:
            n_cc += 1
        elif b1 and not b2:
            n_cw += 1
        elif not b1 and b2:
            n_wc += 1
        else:
            n_ww += 1

        m1 = _parsed_self_mean(p1)
        m2 = _parsed_self_mean(p2) if isinstance(p2, dict) else None
        if m1 is not None and m2 is not None:
            d_msrs.append(m2 - m1)
        if m2 is not None:
            second_msrs_values.append(m2)

        co1 = _confidence_overall(p1)
        co2 = _confidence_overall(p2) if isinstance(p2, dict) else None
        if co1 is not None and co2 is not None:
            d_conf.append(co2 - co1)

        t1 = fp.get("n_output_tokens") if fp else None
        t2 = rec.get("n_output_tokens")
        if isinstance(t1, int) and isinstance(t2, int) and t1 > 0:
            tok_ratios.append(t2 / t1)
        a1 = fp.get("n_answer_tokens") if fp else None
        a2 = rec.get("n_answer_tokens")
        if isinstance(a1, int) and isinstance(a2, int) and a1 > 0:
            ans_ratios.append(a2 / a1)

        s1 = _part_b_concat(p1)
        s2 = _part_b_concat(p2) if isinstance(p2, dict) else ""
        prior_ref_flags.append(1 if (s2 and _PRIOR_REF_RE.search(s2)) else 0)

        z1 = _stage3_norm(p1)
        z2 = _stage3_norm(p2) if isinstance(p2, dict) else None
        if z1 is not None and z2 is not None:
            stage3_flip.append(1 if z1 != z2 else 0)

        if qualitative is not None:
            qualitative.append(
                {
                    "oid": oid,
                    "gold_target_class": gold,
                    "part_b_first": s1[:4000],
                    "part_b_second": s2[:4000],
                }
            )

    n = len(c1_list)
    if n == 0:
        return {"error": "no_evaluable_pairs", "n_rows_in_jsonl": len(rows)}
    acc1 = sum(c1_list) / n if n else 0.0
    acc2 = sum(c2_list) / n if n else 0.0
    wrong_first = n_wc + n_ww
    correct_first = n_cc + n_cw

    correction_rate = (n_wc / wrong_first) if wrong_first else None
    damage_rate = (n_cw / correct_first) if correct_first else None
    persistence_rate = (n_ww / wrong_first) if wrong_first else None

    p_mcnemar = mcnemar_exact_two_sided(n_wc, n_cw)

    out: dict[str, Any] = {
        "n_evaluable_pairs": n,
        "contingency": {"n_cc": n_cc, "n_cw": n_cw, "n_wc": n_wc, "n_ww": n_ww},
        "accuracy_first": round(acc1, 4),
        "accuracy_second": round(acc2, 4),
        "net_accuracy_delta": round(acc2 - acc1, 4),
        "correction_rate": round(correction_rate, 4) if correction_rate is not None else None,
        "damage_rate": round(damage_rate, 4) if damage_rate is not None else None,
        "persistence_rate": round(persistence_rate, 4) if persistence_rate is not None else None,
        "mcnemar_exact_two_sided_p_value": round(p_mcnemar, 6),
        "mcnemar_discordant_pairs": {"wrong_to_correct": n_wc, "correct_to_wrong": n_cw},
        "msrs_mean_delta": round(float(sum(d_msrs) / len(d_msrs)), 4) if d_msrs else None,
        "msrs_delta_n": len(d_msrs),
        "confidence_overall_mean_delta": round(float(sum(d_conf) / len(d_conf)), 4) if d_conf else None,
        "confidence_overall_delta_n": len(d_conf),
        "token_ratio_output_mean": round(float(sum(tok_ratios) / len(tok_ratios)), 4) if tok_ratios else None,
        "token_ratio_output_n": len(tok_ratios),
        "token_ratio_answer_mean": round(float(sum(ans_ratios) / len(ans_ratios)), 4) if ans_ratios else None,
        "token_ratio_answer_n": len(ans_ratios),
        "prior_reference_rate": round(sum(prior_ref_flags) / len(prior_ref_flags), 4) if prior_ref_flags else None,
        "stage3_flip_rate": round(sum(stage3_flip) / len(stage3_flip), 4) if stage3_flip else None,
        "second_pass_msrs_ge4_rate": (
            round(sum(1 for v in second_msrs_values if v >= 4.0) / len(second_msrs_values), 4)
            if second_msrs_values
            else None
        ),
        "second_pass_msrs_ge4_n": len(second_msrs_values),
        "sample_class_counts": manifest["target_class"].value_counts().to_dict(),
    }
    if qualitative is not None:
        out["qualitative_part_b_snippets"] = qualitative
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--second-jsonl", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument(
        "--include-qualitative",
        action="store_true",
        help="Embed truncated Part B text pairs per OID (for appendix / manual coding).",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    for line in args.second_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))

    manifest = pd.read_csv(args.manifest, low_memory=False)
    metrics = evaluate_pairs(rows, manifest, include_qualitative=args.include_qualitative)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

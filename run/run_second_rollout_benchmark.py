"""
Run the second-rollout ablation on a model-specific n=35 manifest.

Same concurrency / backend contract as ``run_tinker_benchmark.py``, but:

- Defaults ``--prompts`` to ``prompts_second_roll_out_ablation``.
- Copies ``ablation_prior_file`` from the manifest row into each JSONL record
  and echoes a few first-pass token counters for downstream metrics.

Example (GPT-5.4 high)::

  python run_second_rollout_benchmark.py \\
    --manifest data_second_roll_out_ablation/metadata/gpt54_high_n35.csv \\
    --backend openai --model gpt-5.4 --reasoning-effort high \\
    --out results/second_rollout_gpt54_high_n35.jsonl --concurrency 8

Example (Claude Opus 4.7 nothink, concurrency 2)::

  python run_second_rollout_benchmark.py \\
    --manifest data_second_roll_out_ablation/metadata/opus47_nothink_n35.csv \\
    --backend anthropic --model claude-opus-4-7 --reasoning-effort none \\
    --out results/second_rollout_opus47_nothink_n35.jsonl --concurrency 2
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
for _subdir in ("api_settings", "evaluate", "prompts"):
    _p = _REPO / _subdir
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import importlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import api_tinker
from api_tinker import ROOT
from evaluate import extract_json_object
try:
    from viz._runmeta import current_command_line, write_pass_from_perf_counter
except ModuleNotFoundError:  # pragma: no cover
    from run._runmeta import current_command_line, write_pass_from_perf_counter


def _first_pass_self_mean(fp: dict) -> float | None:
    from evaluate import extract_self_scores

    parsed = fp.get("parsed")
    if not isinstance(parsed, dict):
        raw = fp.get("answer_text") or fp.get("raw_text")
        if raw:
            parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    part_b = parsed.get("Part B") or parsed.get("part_b")
    if not isinstance(part_b, dict):
        return None
    scores = extract_self_scores(part_b)
    if len(scores) != 3 or any(s is None for s in scores):
        return None
    return float(sum(scores)) / 3.0


def _enrich_second_rollout(rec: dict, row: pd.Series) -> dict:
    rel = row.get("ablation_prior_file")
    if rel is None or (isinstance(rel, float) and pd.isna(rel)):
        return rec
    rel = str(rel).strip().replace("\\", "/")
    p = (ROOT / rel).resolve()
    if not p.is_file():
        rec["ablation_prior_missing"] = str(p)
        return rec
    prior = json.loads(p.read_text(encoding="utf-8"))
    fp = prior.get("first_pass") or {}
    rec["ablation_prior_file"] = rel
    rec["first_pass_n_output_tokens"] = fp.get("n_output_tokens")
    rec["first_pass_n_answer_tokens"] = fp.get("n_answer_tokens")
    rec["first_pass_n_reasoning_tokens"] = fp.get("n_reasoning_tokens")
    rec["first_pass_self_mean"] = _first_pass_self_mean(fp)
    return rec


def _process_row(
    oid: str,
    tc: str,
    row: pd.Series,
    model: str,
    backend_module,
    extra_kwargs: dict,
) -> dict:
    rec = backend_module.run_one(oid, tc, row, model_name=model, **extra_kwargs)
    rec["parsed"] = extract_json_object(rec.get("answer_text") or rec["raw_text"])
    return _enrich_second_rollout(rec, row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument(
        "--prompts",
        type=str,
        default="prompts_second_roll_out_ablation",
        help="Prompt module (default: prompts_second_roll_out_ablation).",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["tinker", "openai", "google", "anthropic"],
    )
    ap.add_argument("--reasoning-effort", type=str, default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    ap.add_argument("--thinking", type=str, default="enabled",
                    choices=["enabled", "disabled"])
    args = ap.parse_args()

    if args.backend == "openai":
        import api_openai
        backend_module = api_openai
        default_model = api_openai.DEFAULT_MODEL
    elif args.backend == "google":
        import api_google
        backend_module = api_google
        default_model = api_google.DEFAULT_MODEL
    elif args.backend == "anthropic":
        import api_anthropic
        backend_module = api_anthropic
        default_model = api_anthropic.DEFAULT_MODEL
    else:
        backend_module = api_tinker
        default_model = api_tinker.DEFAULT_MODEL

    prompt_mod = importlib.import_module(args.prompts)
    if args.prompts != "prompts":
        backend_module.SYSTEM_PROMPT = prompt_mod.SYSTEM_PROMPT
        backend_module.build_user_prompt = prompt_mod.build_user_prompt
        backend_module.manifest_row_to_metadata = prompt_mod.manifest_row_to_metadata
        print(f"Using prompt module: {args.prompts}")

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.manifest, low_memory=False)
    req_cols = getattr(prompt_mod, "required_manifest_columns", None)
    if callable(req_cols):
        missing = req_cols() - frozenset(df.columns)
        if missing:
            print(f"Manifest missing columns: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)
    if args.start:
        df = df.iloc[args.start :]
    if args.limit is not None:
        df = df.head(args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = args.model or default_model
    total = len(df)

    extra_kwargs: dict = {}
    _EFFORT_BACKENDS = ("openai", "google", "anthropic")
    if args.backend in _EFFORT_BACKENDS and args.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.backend == "tinker":
        extra_kwargs["thinking"] = args.thinking == "enabled"

    n_ok = 0
    n_err = 0
    write_lock = threading.Lock()
    t0 = time.perf_counter()
    t0_wall = time.time()

    with open(args.out, "w", encoding="utf-8") as fout:
        if args.concurrency <= 1:
            for idx, (_, row) in enumerate(df.iterrows(), 1):
                oid = str(row["oid"])
                tc = str(row["target_class"])
                try:
                    rec = _process_row(oid, tc, row, model, backend_module, extra_kwargs)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_ok += 1
                    print(f"[{idx}/{total}] OK {oid} ({tc})")
                except Exception as e:
                    n_err += 1
                    err_rec = {"oid": oid, "target_class": tc, "error": str(e)}
                    fout.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"[{idx}/{total}] FAIL {oid}: {e}", file=sys.stderr)
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for _, row in df.iterrows():
                    oid = str(row["oid"])
                    tc = str(row["target_class"])
                    fut = pool.submit(
                        _process_row, oid, tc, row, model, backend_module, extra_kwargs
                    )
                    futures[fut] = (oid, tc)

                for done_idx, fut in enumerate(as_completed(futures), 1):
                    oid, tc = futures[fut]
                    try:
                        rec = fut.result()
                        with write_lock:
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fout.flush()
                            n_ok += 1
                        print(f"[{done_idx}/{total}] OK {oid} ({tc})")
                    except Exception as e:
                        with write_lock:
                            n_err += 1
                            err_rec = {"oid": oid, "target_class": tc, "error": str(e)}
                            fout.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                            fout.flush()
                        print(f"[{done_idx}/{total}] FAIL {oid}: {e}", file=sys.stderr)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"Wrote {args.out}  ok={n_ok}  fail={n_err}  elapsed={elapsed:.1f}s")
    try:
        sc = write_pass_from_perf_counter(
            args.out,
            kind="initial",
            t0_perf=t0,
            t1_perf=t1,
            t0_wall_unix=t0_wall,
            rows_attempted=total,
            rows_ok=n_ok,
            rows_fail=n_err,
            concurrency=args.concurrency,
            command=current_command_line(),
        )
        print(f"Runmeta sidecar updated: {sc}")
    except Exception as e:
        print(f"Warning: could not write runmeta sidecar: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

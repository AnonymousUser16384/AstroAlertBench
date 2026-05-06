"""
Run VLM evaluation on the manifest. Supports two backends:
  --backend tinker  (default)  -> api_tinker.py (open-source via Tinker SDK)
  --backend openai             -> api_openai.py (closed-source via OpenAI Responses API)

Example (Tinker, manifest_enriched.csv with full fields):
  set TINKER_API_KEY=...
  python run_tinker_benchmark.py --manifest data/manifest_enriched.csv --model moonshotai/Kimi-K2.5 --limit 10 --out results/kimi_zs.jsonl
  python evaluate.py --predictions results/kimi_zs.jsonl --manifest data/manifest_enriched.csv

Example (OpenAI, GPT-5.4 with thinking high):
  set OPENAI_API_KEY=...
  python run_tinker_benchmark.py --backend openai --manifest data/manifest_fewshot.csv --model gpt-5.4 --reasoning-effort high --out results/fewshot_gpt54_high.jsonl --concurrency 8

Example (OpenAI, GPT-5.4 with thinking disabled):
  python run_tinker_benchmark.py --backend openai --manifest data/manifest_fewshot.csv --model gpt-5.4 --reasoning-effort none --out results/fewshot_gpt54_none.jsonl --concurrency 8

Example (Google Gemini 2.5 Pro, dynamic thinking = reasoning high):
  set GOOGLE_API_KEY=...
  python run_tinker_benchmark.py --backend google --manifest data/manifest_benchmark_final.csv --model gemini-2.5-pro --reasoning-effort high --out results/benchmark_gemini25_pro_high.jsonl --concurrency 8

Example (Google Gemini 2.5 Flash, thinking disabled):
  python run_tinker_benchmark.py --backend google --manifest data/manifest_benchmark_final.csv --model gemini-2.5-flash --reasoning-effort none --out results/benchmark_gemini25_flash_none.jsonl --concurrency 8

Example (Anthropic Claude Opus 4.7, adaptive thinking):
  set ANTHROPIC_API_KEY=...
  python run_tinker_benchmark.py --backend anthropic --manifest data/manifest_benchmark_final.csv --model claude-opus-4-7 --reasoning-effort high --out results/benchmark_opus47_think.jsonl --concurrency 8

Example (Anthropic Claude Opus 4.7, thinking disabled):
  python run_tinker_benchmark.py --backend anthropic --manifest data/manifest_benchmark_final.csv --model claude-opus-4-7 --reasoning-effort none --out results/benchmark_opus47_nothink.jsonl --concurrency 8

Example (Tinker, Qwen3.5 with thinking disabled — use *DisableThinkingRenderer):
  python run_tinker_benchmark.py --manifest data/manifest_benchmark_final.csv --model Qwen/Qwen3.5-4B --thinking disabled --out results/benchmark_qwen35_4b_nothink.jsonl --concurrency 32

Parallel execution (default concurrency=1 for backward compat):
  python run_tinker_benchmark.py --manifest data/manifest_fewshot.csv --out results/fewshot.jsonl --concurrency 64

Full metadata + extra Part B/C guidance for AGN vs variable_star:
  python run_tinker_benchmark.py --manifest data/manifest_enriched.csv --out results/run_agn_prompt.jsonl --prompts prompts_agn_instruction
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
    from viz._runmeta import (
        current_command_line,
        write_pass_from_perf_counter,
    )
except ModuleNotFoundError:  # pragma: no cover
    from run._runmeta import (
        current_command_line,
        write_pass_from_perf_counter,
    )


def _process_row(
    oid: str,
    tc: str,
    row: "pd.Series",
    model: str,
    backend_module,
    extra_kwargs: dict,
) -> dict:
    rec = backend_module.run_one(oid, tc, row, model_name=model, **extra_kwargs)
    rec["parsed"] = extract_json_object(rec.get("answer_text") or rec["raw_text"])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Tinker VLM on ZTF benchmark montages")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--out", type=Path, required=True, help="JSONL output path")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to process")
    ap.add_argument("--model", type=str, default=None, help="Override TINKER_MODEL")
    ap.add_argument("--start", type=int, default=0, help="Row offset")
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of parallel API calls (default: 1, try 32-64 for speed)",
    )
    ap.add_argument(
        "--prompts", type=str, default="prompts",
        help="Prompt module name (default: prompts). Use 'prompts_agn_instruction' for full fields + AGN vs VS guidance.",
    )
    ap.add_argument(
        "--backend", type=str, default="tinker",
        choices=["tinker", "openai", "google", "anthropic"],
        help=(
            "API backend (default: tinker). "
            "'openai' calls OpenAI Responses API via api_openai.py. "
            "'google' calls Gemini generate_content via api_google.py. "
            "'anthropic' calls Claude Messages API via api_anthropic.py."
        ),
    )
    ap.add_argument(
        "--reasoning-effort", type=str, default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help=(
            "OpenAI / Google / Anthropic backends. "
            "OpenAI: GPT-5.x reasoning_effort ('none' disables, 'xhigh' ceiling). "
            "Google: on Gemini 2.5 maps effort -> integer thinking_budget "
            "(none=0, minimal=128, low=1024, medium=8192, high=-1/dynamic); "
            "on Gemini 3.x maps 1:1 to thinking_level {minimal, low, medium, high}. "
            "'none' is only valid on Flash variants (2.5-flash can disable; 2.5-pro and "
            "Gemini 3.x are thinking-only). 'minimal' is Flash-only on both families. "
            "'xhigh' is always rejected (Gemini caps at high / dynamic). "
            "Anthropic (Claude Opus 4.7): 'none' disables thinking; "
            "{minimal, low, medium, high} enable adaptive thinking "
            "(thinking.type=adaptive + output_config.effort=<level>); "
            "'high' is the recommended adaptive thinking setting. "
            "'xhigh' is rejected (Anthropic enum tops out at 'high')."
        ),
    )
    ap.add_argument(
        "--thinking", type=str, default="enabled",
        choices=["enabled", "disabled"],
        help=(
            "Tinker backend only. For Kimi K2.5 and Qwen3.5 family, selects between "
            "the thinking-enabled renderer (default) and the *DisableThinkingRenderer. "
            "No-op for Qwen3-VL and other non-reasoning models. Ignored on --backend openai "
            "(use --reasoning-effort instead)."
        ),
    )
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

    df = pd.read_csv(args.manifest)
    req_cols = getattr(prompt_mod, "required_manifest_columns", None)
    if callable(req_cols):
        missing = req_cols() - frozenset(df.columns)
        if missing:
            print(
                f"Manifest missing required columns for --prompts {args.prompts!r}: "
                f"{sorted(missing)}. Use manifest_enriched.csv or run enrich_manifest_alerce.py.",
                file=sys.stderr,
            )
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
    if args.reasoning_effort is not None and args.backend not in _EFFORT_BACKENDS:
        print(
            "Warning: --reasoning-effort is only used with --backend "
            "openai/google/anthropic; ignoring.",
            file=sys.stderr,
        )
    if args.backend == "tinker":
        thinking_bool = args.thinking == "enabled"
        extra_kwargs["thinking"] = thinking_bool
        if not api_tinker._model_supports_thinking_toggle(model) and args.thinking == "disabled":
            print(
                f"Warning: --thinking disabled has no effect for model {model!r} "
                "(only Kimi K2.5 and Qwen3.5 have a *DisableThinkingRenderer). "
                "Proceeding with the default renderer.",
                file=sys.stderr,
            )
        print(f"Tinker renderer thinking mode: {'enabled' if thinking_bool else 'disabled'}")
    elif args.thinking != "enabled":
        print(
            "Warning: --thinking only affects --backend tinker; use --reasoning-effort "
            "with --backend openai. Ignoring.",
            file=sys.stderr,
        )

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

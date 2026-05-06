"""
Retry / resume an existing JSONL benchmark result file in-place.

What this script does
---------------------
Given an existing JSONL written by run_tinker_benchmark.py and the same
manifest CSV that produced it, this script will:

  1. Re-run every row whose record looks like a failure (error key present,
     no raw_text/parsed). These are typically rows that hit a transient API
     error (429 / 503 / quota / timeout) the first time.
  2. ALSO run every manifest OID that is NOT in the JSONL at all. This is
     the case when the original run was killed before completion (e.g.
     stopped on quota exhaustion), so part of the manifest never got an
     attempt at all. Successful rows already in the JSONL are kept as-is.

The final file is rewritten in MANIFEST ORDER, with:
  - newly retried rows replacing the old failed entries,
  - newly attempted rows inserted at their correct manifest position,
  - previously successful rows preserved exactly.

Use --dry-run to see how many rows would be retried / newly attempted before
hitting the API. Use --only failed | missing | both to scope the work.

Examples
--------
Resume a half-finished Gemini 2.5 Pro run after a quota-exhaustion stop:

  python retry_failed.py --results results/benchmark_gemini25_pro_high.jsonl \
      --manifest data/manifest_benchmark_final.csv \
      --backend google --model gemini-2.5-pro --reasoning-effort high --concurrency 4

Resume a Gemini 2.5 Flash run (non-reasoning):

  python retry_failed.py --results results/benchmark_gemini25_flash_none.jsonl \
      --manifest data/manifest_benchmark_final.csv \
      --backend google --model gemini-2.5-flash --reasoning-effort none --concurrency 8

Retry only the failed rows (skip missing OIDs):

  python retry_failed.py --results results/benchmark_gpt54_high.jsonl \
      --manifest data/manifest_benchmark_final.csv \
      --backend openai --model gpt-5.4 --reasoning-effort high \
      --only failed --concurrency 8

Claude Opus 4.7 (adaptive thinking):

  python retry_failed.py --results results/benchmark_opus47_think.jsonl \
      --manifest data/manifest_benchmark_final.csv \
      --backend anthropic --model claude-opus-4-7 --reasoning-effort high --concurrency 2

Audit:
  - <results>.bak   : copy of the original file before modification
  - <results>.retry.jsonl : just the rows fetched in this pass (audit log)
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
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import api_tinker
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


def _is_failed(rec: dict) -> bool:
    """Return True if this record needs to be re-fetched.

    A record is failed if it carries an "error" key with no usable output, or
    if it has neither raw_text nor parsed nor a model field (i.e. it's a stub
    error placeholder written by run_tinker_benchmark.py on exception).
    """
    if "error" in rec and rec.get("raw_text") is None:
        return True
    if rec.get("raw_text") is None and rec.get("parsed") is None and rec.get("model") is None:
        return True
    return False


def _process_row(oid, tc, row, model, backend_module, extra_kwargs):
    rec = backend_module.run_one(oid, tc, row, model_name=model, **extra_kwargs)
    rec["parsed"] = extract_json_object(rec.get("answer_text") or rec["raw_text"])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retry failed and/or fill in missing JSONL rows; merge in-place."
    )
    ap.add_argument("--results", type=Path, required=True,
                    help="Existing JSONL file produced by run_tinker_benchmark.py")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Manifest CSV matching the original run "
                         "(e.g. data/manifest_benchmark_final.csv)")
    ap.add_argument("--backend",
                    choices=["tinker", "openai", "google", "anthropic"],
                    default="openai")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--reasoning-effort", type=str, default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                    help="Forwarded to OpenAI / Google / Anthropic backends "
                         "(see api_*.py for per-model validity).")
    ap.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled",
                    help="Tinker backend only.")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--only", choices=["failed", "missing", "both"], default="both",
        help=(
            "Which rows to (re-)run. 'failed' = only rows already in the JSONL "
            "with an error; 'missing' = only manifest OIDs not in the JSONL "
            "at all; 'both' (default) = failed + missing."
        ),
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report counts; do not call the API.")
    args = ap.parse_args()

    if not args.results.is_file():
        print(f"Results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)
    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    existing: list[dict] = []
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        existing.append(json.loads(line))
    existing_by_oid: dict[str, dict] = {r.get("oid"): r for r in existing if r.get("oid")}

    df = pd.read_csv(args.manifest)
    manifest_oids = [str(r["oid"]) for _, r in df.iterrows()]
    df_by_oid = {str(r["oid"]): r for _, r in df.iterrows()}

    failed_oids = [oid for oid, r in existing_by_oid.items() if _is_failed(r)]
    missing_oids = [oid for oid in manifest_oids if oid not in existing_by_oid]
    extra_in_results = [oid for oid in existing_by_oid.keys() if oid not in df_by_oid]

    print(f"Manifest rows  : {len(manifest_oids)}")
    print(f"Result rows    : {len(existing)}  (file: {args.results})")
    print(f"  ok           : {len(existing) - len(failed_oids)}")
    print(f"  failed       : {len(failed_oids)}")
    print(f"Missing rows   : {len(missing_oids)}  (in manifest, not in result file)")
    if extra_in_results:
        print(f"WARNING: {len(extra_in_results)} OIDs in result file are NOT in manifest "
              f"(will be preserved in output): examples {extra_in_results[:3]}")

    if args.only == "failed":
        target_oids = list(failed_oids)
    elif args.only == "missing":
        target_oids = list(missing_oids)
    else:
        target_oids = list(failed_oids) + list(missing_oids)

    if not target_oids:
        print("Nothing to do.")
        return
    print(f"Will (re-)run {len(target_oids)} rows  [--only={args.only}]")
    if args.dry_run:
        sample = ", ".join(target_oids[:10])
        more = f" (+{len(target_oids) - 10} more)" if len(target_oids) > 10 else ""
        print(f"First targets: {sample}{more}")
        return

    not_in_manifest = [oid for oid in target_oids if oid not in df_by_oid]
    if not_in_manifest:
        print(f"ERROR: {len(not_in_manifest)} target OIDs not present in manifest; aborting.",
              file=sys.stderr)
        print(f"  examples: {not_in_manifest[:5]}", file=sys.stderr)
        sys.exit(2)

    if args.backend == "openai":
        import api_openai
        backend_module = api_openai
    elif args.backend == "google":
        import api_google
        backend_module = api_google
    elif args.backend == "anthropic":
        import api_anthropic
        backend_module = api_anthropic
    else:
        backend_module = api_tinker

    extra_kwargs: dict = {}
    if args.backend in ("openai", "google", "anthropic") and args.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.backend == "tinker":
        extra_kwargs["thinking"] = args.thinking == "enabled"

    bak = args.results.with_suffix(args.results.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(args.results, bak)
        print(f"Backup written: {bak}")
    else:
        print(f"Backup already exists (not overwriting): {bak}")

    retry_out = args.results.with_suffix(".retry.jsonl")
    new_by_oid: dict[str, dict] = {}
    still_failed: list[tuple[str, str]] = []
    total = len(target_oids)
    write_lock = threading.Lock()
    t0 = time.perf_counter()
    t0_wall = time.time()

    with open(retry_out, "w", encoding="utf-8") as fout:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for oid in target_oids:
                row = df_by_oid[oid]
                tc = str(row["target_class"])
                fut = pool.submit(
                    _process_row, oid, tc, row, args.model, backend_module, extra_kwargs
                )
                futures[fut] = (oid, tc)

            for done_idx, fut in enumerate(as_completed(futures), 1):
                oid, tc = futures[fut]
                try:
                    rec = fut.result()
                    with write_lock:
                        new_by_oid[oid] = rec
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                    print(f"[{done_idx}/{total}] OK {oid} ({tc})")
                except Exception as e:
                    err_rec = {"oid": oid, "target_class": tc, "error": str(e)}
                    with write_lock:
                        still_failed.append((oid, str(e)))
                        fout.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                        fout.flush()
                    print(f"[{done_idx}/{total}] FAIL {oid}: {e}", file=sys.stderr)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"\nRetry pass finished: ok={len(new_by_oid)}  fail={len(still_failed)}  "
          f"elapsed={elapsed:.1f}s")
    print(f"Retry-only log: {retry_out}")
    try:
        sc = write_pass_from_perf_counter(
            args.results,
            kind="retry",
            t0_perf=t0,
            t1_perf=t1,
            t0_wall_unix=t0_wall,
            rows_attempted=total,
            rows_ok=len(new_by_oid),
            rows_fail=len(still_failed),
            concurrency=args.concurrency,
            command=current_command_line(),
        )
        print(f"Runmeta sidecar updated: {sc}")
    except Exception as e:
        print(f"Warning: could not write runmeta sidecar: {e}", file=sys.stderr)

    # Rebuild merged output in MANIFEST ORDER. Priority for each OID:
    #   1) record produced in this retry pass (success or fresh error stub),
    #   2) existing record from the prior file,
    #   3) untouched OIDs that were skipped (e.g. --only failed) and never attempted
    #      get a placeholder error row so downstream loaders see a consistent length.
    fresh_failed_by_oid = {oid: msg for oid, msg in still_failed}
    merged: list[dict] = []
    for oid in manifest_oids:
        tc = str(df_by_oid[oid]["target_class"])
        if oid in new_by_oid:
            merged.append(new_by_oid[oid])
        elif oid in existing_by_oid:
            merged.append(existing_by_oid[oid])
        elif oid in fresh_failed_by_oid:
            merged.append({"oid": oid, "target_class": tc,
                           "error": fresh_failed_by_oid[oid]})
        else:
            merged.append({"oid": oid, "target_class": tc,
                           "error": "not_attempted (skipped by --only filter)"})

    # Preserve any extra rows that were in the file but not in the manifest,
    # so we never silently drop data.
    if extra_in_results:
        for oid in extra_in_results:
            merged.append(existing_by_oid[oid])

    tmp = args.results.with_suffix(args.results.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fout:
        for rec in merged:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(args.results)

    remaining_fail = sum(1 for r in merged if _is_failed(r))
    print(f"Merged results written to: {args.results}")
    print(f"  total rows  = {len(merged)}")
    print(f"  ok rows     = {len(merged) - remaining_fail}")
    print(f"  failed rows = {remaining_fail}")
    if still_failed:
        print("\nOIDs that still failed after retry:")
        for oid, msg in still_failed[:20]:
            print(f"  - {oid}: {msg[:120]}")
        if len(still_failed) > 20:
            print(f"  ... (+{len(still_failed) - 20} more)")


if __name__ == "__main__":
    main()

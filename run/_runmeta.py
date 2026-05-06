"""Wall-clock runtime sidecar for benchmark runs.

Each call to ``python -m run.run_tinker_benchmark`` (initial sweep) or ``python -m run.retry_failed``
(resume / fix pass) appends one entry to a sidecar JSON file living next to the
predictions JSONL: ``<results>.runmeta.json``.

This lets ``viz.build_run_folder`` surface a "Wall-clock runtime" line in
``report.md`` and a ``wall_clock`` block in ``metrics.json`` without having to
re-derive the numbers from terminal scrollback after the fact.

Schema
------
::

    {
      "passes": [
        {
          "kind": "initial" | "retry" | "manual",
          "started_at": "2026-04-23T09:43:02-07:00",  // ISO8601 with offset
          "finished_at": "2026-04-23T12:42:53-07:00",
          "elapsed_seconds": 10790.7,
          "rows_attempted": 1500,
          "rows_ok": 1500,
          "rows_fail": 0,
          "concurrency": 2,
          "command": "python -m run.run_tinker_benchmark ...",
          "estimated": false   // true only when back-filled from terminal logs
        },
        ...
      ]
    }

Aggregation rules
-----------------
- ``wall_clock_seconds_total`` = sum of ``elapsed_seconds`` across passes.
- ``wall_clock_human`` = ``Xh Ym Zs`` formatted from total.
- ``estimated`` at the top level is true iff any pass is estimated.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any


def _sidecar_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(jsonl_path.suffix + ".runmeta.json")


def _isoformat_local(ts: float) -> str:
    """Return a tz-aware ISO8601 string for a unix timestamp in local time."""
    return _dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def format_wallclock_human(seconds: float) -> str:
    """Render a duration in seconds as ``Xh Ym Zs`` (skipping leading zeros)."""
    if seconds is None:
        return "?"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h} h")
    if m or h:
        parts.append(f"{m} m")
    parts.append(f"{s} s")
    return " ".join(parts)


def append_pass(
    jsonl_path: str | Path,
    *,
    kind: str,
    started_at_unix: float,
    finished_at_unix: float,
    elapsed_seconds: float,
    rows_attempted: int,
    rows_ok: int,
    rows_fail: int,
    concurrency: int | None = None,
    command: str | None = None,
    estimated: bool = False,
) -> Path:
    """Append one pass to ``<jsonl>.runmeta.json`` and return the sidecar path.

    Creates the sidecar if it does not exist. Never deletes / rewrites prior
    passes — additive only.
    """
    sc = _sidecar_path(Path(jsonl_path))
    if sc.exists():
        try:
            data = json.loads(sc.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"passes": []}
    else:
        data = {"passes": []}
    data.setdefault("passes", []).append(
        {
            "kind": kind,
            "started_at": _isoformat_local(started_at_unix),
            "finished_at": _isoformat_local(finished_at_unix),
            "elapsed_seconds": round(float(elapsed_seconds), 1),
            "rows_attempted": int(rows_attempted),
            "rows_ok": int(rows_ok),
            "rows_fail": int(rows_fail),
            "concurrency": concurrency,
            "command": command,
            "estimated": bool(estimated),
        }
    )
    sc.parent.mkdir(parents=True, exist_ok=True)
    sc.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return sc


def read_runmeta(jsonl_path: str | Path) -> dict[str, Any] | None:
    """Return the parsed sidecar dict, or ``None`` if no sidecar exists."""
    sc = _sidecar_path(Path(jsonl_path))
    if not sc.exists():
        return None
    try:
        return json.loads(sc.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def summarize(runmeta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Aggregate a runmeta dict for reporting. Returns ``None`` if empty."""
    if not runmeta:
        return None
    passes = runmeta.get("passes") or []
    if not passes:
        return None
    total = sum(float(p.get("elapsed_seconds") or 0.0) for p in passes)
    estimated_any = any(bool(p.get("estimated")) for p in passes)
    per_pass = [
        {
            "kind": p.get("kind"),
            "elapsed_seconds": p.get("elapsed_seconds"),
            "rows_attempted": p.get("rows_attempted"),
            "rows_ok": p.get("rows_ok"),
            "rows_fail": p.get("rows_fail"),
            "concurrency": p.get("concurrency"),
            "started_at": p.get("started_at"),
            "finished_at": p.get("finished_at"),
            "estimated": bool(p.get("estimated")),
        }
        for p in passes
    ]
    return {
        "wall_clock_seconds_total": round(total, 1),
        "wall_clock_human": format_wallclock_human(total),
        "n_passes": len(passes),
        "estimated": estimated_any,
        "passes": per_pass,
    }


def format_wallclock_line(summary: dict[str, Any] | None) -> str | None:
    """Return the one-line "Wall-clock runtime" string used in ``report.md``,
    or ``None`` when no usable data is available."""
    if not summary:
        return None
    total = summary["wall_clock_seconds_total"]
    human = summary["wall_clock_human"]
    n = summary["n_passes"]
    est = " (estimated from terminal logs)" if summary["estimated"] else ""
    if n == 1:
        p = summary["passes"][0]
        conc = f", concurrency {p['concurrency']}" if p.get("concurrency") else ""
        return (
            f"{human} ({total:.1f} s, 1 pass: {p['kind']}, "
            f"{p['rows_ok']}/{p['rows_attempted']} ok{conc}){est}"
        )
    pieces = []
    for p in summary["passes"]:
        kind = p["kind"]
        secs = p["elapsed_seconds"]
        rows = p["rows_ok"]
        att = p["rows_attempted"]
        pieces.append(f"{kind} {secs:.0f} s ({rows}/{att} ok)")
    return f"{human} total ({total:.1f} s, {n} passes: " + " + ".join(pieces) + f"){est}"


def write_pass_from_perf_counter(
    jsonl_path: str | Path,
    *,
    kind: str,
    t0_perf: float,
    t1_perf: float,
    t0_wall_unix: float | None,
    rows_attempted: int,
    rows_ok: int,
    rows_fail: int,
    concurrency: int | None = None,
    command: str | None = None,
) -> Path:
    """Convenience wrapper for callers that measure with ``time.perf_counter``.

    ``t0_wall_unix`` should be ``time.time()`` captured at the same moment as
    ``t0_perf`` (so we can also record human-readable ISO timestamps).  If
    omitted we fall back to ``finished_at - elapsed`` for ``started_at``.
    """
    elapsed = max(0.0, float(t1_perf) - float(t0_perf))
    finished_unix = (t0_wall_unix + elapsed) if t0_wall_unix is not None else _dt.datetime.now().timestamp()
    started_unix = (t0_wall_unix if t0_wall_unix is not None else (finished_unix - elapsed))
    return append_pass(
        jsonl_path,
        kind=kind,
        started_at_unix=started_unix,
        finished_at_unix=finished_unix,
        elapsed_seconds=elapsed,
        rows_attempted=rows_attempted,
        rows_ok=rows_ok,
        rows_fail=rows_fail,
        concurrency=concurrency,
        command=command,
    )


def current_command_line() -> str:
    """Best-effort reconstruction of the invoking command line."""
    return "python " + " ".join(sys.argv) if sys.argv else "?"

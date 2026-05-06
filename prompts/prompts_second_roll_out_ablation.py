"""
Second-rollout ablation prompts: same task as `prompts.py`, plus the model's
own **first-pass JSON** (verbatim) for self-review — no gold label is revealed.

`run_tinker_benchmark.py` / `run_second_rollout_benchmark.py` must patch the
backend module to import *this* module's `SYSTEM_PROMPT`, `build_user_prompt`,
and `manifest_row_to_metadata` (same pattern as switching `--prompts`).

The manifest CSV must include a column `ablation_prior_file` with a
repo-relative path to the JSON file written under
`data_second_roll_out_ablation/priors/<slug>/<oid>.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import prompts as _base

ROOT = Path(__file__).resolve().parent

SECOND_ROLL_ADDENDUM = """

---

## Second trial (ablation)

You are seeing the **same** alert again (same images and metadata as below).

Below is the **exact JSON object you returned on your first attempt** for this
alert (Parts A–C and all fields). That earlier output is **not** labeled as
correct or incorrect; treat it only as your own prior work product.

**Your task now**

1. Carefully re-read the images and metadata.
2. Critically review your **previous** JSON: evidence, staged logic (artifact vs
   real, solar vs astrophysical, subclass), and self-scores.
3. Produce a **fresh** single JSON object with the **same top-level schema**
   (`Part A`, `Part B`, `Part C`) and the **same field names and constraints**
   as in the original benchmark instructions in the system message.

You may revise any part of your answer if the evidence warrants it, or keep
the same conclusion if re-analysis supports it. If you change your mind,
briefly explain why in the Part B prose (without claiming any external
verification of right/wrong).

**Important:** Output **only** one JSON object — no markdown fences, no
preamble or postscript outside the JSON.
"""


SYSTEM_PROMPT = _base.SYSTEM_PROMPT.strip() + SECOND_ROLL_ADDENDUM


def manifest_row_to_metadata(row: Any) -> dict[str, Any]:
    """Same enriched metadata as `prompts.py`, plus embedded first-pass JSON text."""
    meta = _base.manifest_row_to_metadata(row)
    rel = None
    if hasattr(row, "get"):
        rel = row.get("ablation_prior_file")
    elif isinstance(row, dict):
        rel = row.get("ablation_prior_file")
    if rel is None or (isinstance(rel, float) and pd.isna(rel)):
        raise ValueError(
            "Missing ablation_prior_file on manifest row — regenerate manifests "
            "with viz/build_second_rollout_ablation.py."
        )
    rel = str(rel).strip().replace("\\", "/")
    path = (ROOT / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ablation prior JSON not found: {path}")
    text = path.read_text(encoding="utf-8")
    # Keep under a private key consumed only by build_user_prompt.
    meta["__ablation_prior_json_text__"] = text
    return meta


def build_user_prompt(oid: str, metadata: dict[str, Any]) -> str:
    prior = metadata.get("__ablation_prior_json_text__")
    if not prior:
        raise ValueError("build_user_prompt: prior JSON not loaded (manifest_row_to_metadata).")
    base_user = _base.build_user_prompt(oid, {k: v for k, v in metadata.items() if k != "__ablation_prior_json_text__"})
    return (
        base_user
        + "\n\n---\n\n[YOUR FIRST ATTEMPT JSON — FOR SELF-REVIEW ONLY]\n\n"
        + prior.strip()
        + "\n\n---\n\nReturn your **second-trial** JSON object now."
    )


def required_manifest_columns() -> frozenset[str]:
    return _base.required_manifest_columns() | frozenset({"ablation_prior_file"})

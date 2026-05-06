"""
OpenAI API runner for ZTF/ALeRCE benchmark (vision-language).

Mirrors the public surface of api_tinker.py so run_tinker_benchmark.py can
dispatch to either backend without touching evaluate.py or prompts.py.

Pipeline:
  (1) First-alert inputs: image triplet (one PNG montage = Science|Reference|Difference)
      + ZTF metadata.
  (2) Prompt construction: SYSTEM_PROMPT + user message from prompts.py.
  (3) OpenAI Responses API call (vision + reasoning) -> JSON string -> evaluate.py parser.

Requires:
  - export OPENAI_API_KEY=...
  - pip install openai>=2.0  (already in requirements.txt)

Default model: gpt-5.4. Reasoning effort: 'high' by default (override via env or CLI).
Thinking can be disabled by passing reasoning_effort='none'.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from openai import OpenAI

from prompts import STAMPS_LLM_DIRNAME, SYSTEM_PROMPT, build_user_prompt, manifest_row_to_metadata

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
DEFAULT_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "high")

# Valid reasoning_effort values for GPT-5.x reasoning models (see OpenAI docs).
_VALID_EFFORT = {"none", "low", "medium", "high", "xhigh"}


def montage_path(target_class: str, oid: str, root: Path | None = None) -> Path:
    base = root or ROOT
    return base / STAMPS_LLM_DIRNAME / target_class / oid / "montage.png"


def _encode_image(image_path: Path) -> str:
    """Read PNG and return a data URI suitable for OpenAI vision input."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_messages(
    oid: str,
    target_class: str,
    metadata: dict[str, Any],
    image_path: Path,
) -> list[dict[str, Any]]:
    """Construct an OpenAI Responses API `input` list with one image + text user msg."""
    user_text = build_user_prompt(oid, metadata)
    image_url = _encode_image(image_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_url},
                {"type": "input_text", "text": user_text},
            ],
        },
    ]


import threading

_client_cache: dict[str, OpenAI] = {}
_client_lock = threading.Lock()


def _get_client() -> OpenAI:
    """Cached OpenAI client (per-process). Reads OPENAI_API_KEY from env."""
    with _client_lock:
        client = _client_cache.get("default")
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Add it to .env or export it before running."
                )
            client = OpenAI(api_key=api_key)
            _client_cache["default"] = client
        return client


def _normalize_effort(effort: str | None) -> str:
    """Resolve and validate reasoning_effort."""
    e = (effort or DEFAULT_REASONING_EFFORT or "none").lower()
    if e not in _VALID_EFFORT:
        raise ValueError(
            f"Invalid reasoning_effort {effort!r}. Must be one of {sorted(_VALID_EFFORT)}."
        )
    return e


def sample_vlm(
    messages: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 20000,
    temperature: float = 0.2,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """
    Run one VLM completion via OpenAI Responses API.

    Returns the same dict shape as api_tinker.sample_vlm (so evaluate.py is unchanged):
      - "raw_text": full visible text response (OpenAI hides the CoT itself; this is
        the post-thinking content)
      - "answer_text": same as raw_text — for OpenAI reasoning models the thinking
        is not exposed, so there is nothing to strip
      - "n_output_tokens": total output tokens from usage (visible + reasoning)
      - "n_answer_tokens": visible-only tokens (output_tokens - reasoning_tokens)
      - "n_reasoning_tokens": reasoning tokens spent (0 if reasoning disabled)
      - "n_prompt_tokens": input tokens consumed
      - "max_tokens": effective cap used (max_output_tokens)
      - "truncated": True if response.status == "incomplete" with reason max_output_tokens
      - "finish_reason": short status string for diagnostics
      - "reasoning_effort": effort level actually used
    """
    client = _get_client()
    effort = _normalize_effort(reasoning_effort)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "input": messages,
        "max_output_tokens": max_tokens,
    }
    if effort != "none":
        kwargs["reasoning"] = {"effort": effort}
    # GPT-5 reasoning family ignores temperature, but it is harmless to pass for
    # non-reasoning calls; gate it to avoid 400s on strict server-side validation.
    if effort == "none":
        kwargs["temperature"] = temperature

    resp = client.responses.create(**kwargs)

    raw_text = (resp.output_text or "").strip()

    usage = getattr(resp, "usage", None)
    n_output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    n_prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    n_reasoning_tokens = 0
    details = getattr(usage, "output_tokens_details", None) if usage else None
    if details is not None:
        n_reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    n_answer_tokens = max(0, n_output_tokens - n_reasoning_tokens)

    status = getattr(resp, "status", "completed")
    incomplete = getattr(resp, "incomplete_details", None)
    incomplete_reason = getattr(incomplete, "reason", None) if incomplete else None
    truncated = status == "incomplete" and incomplete_reason == "max_output_tokens"
    finish_reason = incomplete_reason if status == "incomplete" else status

    return {
        "raw_text": raw_text,
        "answer_text": raw_text,
        "n_output_tokens": n_output_tokens,
        "n_answer_tokens": n_answer_tokens,
        "n_reasoning_tokens": n_reasoning_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "max_tokens": max_tokens,
        "truncated": truncated,
        "finish_reason": finish_reason or "completed",
        "reasoning_effort": effort,
    }


def run_one(
    oid: str,
    target_class: str,
    row: Any,
    stamps_root: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build metadata from manifest row, load montage, call OpenAI vision model."""
    meta = manifest_row_to_metadata(row)
    img = montage_path(target_class, oid, stamps_root)
    if not img.is_file():
        raise FileNotFoundError(f"Missing montage: {img}")
    messages = build_messages(oid, target_class, meta, img)
    vlm_out = sample_vlm(messages, model_name=model_name, reasoning_effort=reasoning_effort)
    return {
        "oid": oid,
        "target_class": target_class,
        "montage_path": str(img),
        "raw_text": vlm_out["raw_text"],
        "answer_text": vlm_out["answer_text"],
        "n_output_tokens": vlm_out["n_output_tokens"],
        "n_answer_tokens": vlm_out["n_answer_tokens"],
        "n_reasoning_tokens": vlm_out["n_reasoning_tokens"],
        "n_prompt_tokens": vlm_out["n_prompt_tokens"],
        "max_tokens": vlm_out["max_tokens"],
        "truncated": vlm_out["truncated"],
        "finish_reason": vlm_out["finish_reason"],
        "reasoning_effort": vlm_out["reasoning_effort"],
        "model": model_name,
    }

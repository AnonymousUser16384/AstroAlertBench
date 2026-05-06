"""
Anthropic API runner for ZTF/ALeRCE benchmark (vision-language).

Mirrors the public surface of api_tinker.py / api_openai.py / api_google.py so
run_tinker_benchmark.py can dispatch to any of the four backends without
touching evaluate.py or prompts.py.

Pipeline:
  (1) First-alert inputs: image triplet (one PNG montage = Science|Reference|Difference)
      + ZTF metadata.
  (2) Prompt construction: SYSTEM_PROMPT + user message from prompts.py.
  (3) Anthropic `messages.create` call (vision + optional extended thinking)
      -> JSON string -> evaluate.py parser.

Requires:
  - export ANTHROPIC_API_KEY=...    (or put ANTHROPIC_API_KEY=... in .env)
  - pip install anthropic

Default model: claude-opus-4-7. Override via env ANTHROPIC_MODEL or CLI.

Reasoning effort maps to Claude Opus 4.7's adaptive-thinking controls:
  - 'none'    -> thinking={"type": "disabled"}                          (no thinking)
  - 'minimal' -> thinking={"type": "adaptive"}, output_config.effort="minimal"
  - 'low'     -> thinking={"type": "adaptive"}, output_config.effort="low"
  - 'medium'  -> thinking={"type": "adaptive"}, output_config.effort="medium"
  - 'high'    -> thinking={"type": "adaptive"}, output_config.effort="high"   (adaptive default)

Claude Opus 4.7 uses "adaptive" thinking: the model decides how much to think
based on the effort level and problem difficulty (no explicit token budget).
'high' is the recommended "adaptive thinking" setting for this benchmark and
'none' is the "no thinking" baseline. 'xhigh' is rejected (the Anthropic
effort enum tops out at 'high').

Legacy Claude 4.0/4.1 models that only accept `thinking.type=enabled` with a
fixed `budget_tokens` are NOT supported by this module — if you need them, add
a model-sniff branch similar to api_google's `_allowed_levels_for_model`.
"""
from __future__ import annotations

import base64
import os
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

try:
    import anthropic
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "api_anthropic.py requires the anthropic package. "
        "Install it with:  pip install anthropic"
    ) from exc

from prompts import STAMPS_LLM_DIRNAME, SYSTEM_PROMPT, build_user_prompt, manifest_row_to_metadata

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
DEFAULT_REASONING_EFFORT = os.environ.get("ANTHROPIC_REASONING_EFFORT", "high")

# How many times the Anthropic SDK should auto-retry transient errors before
# surfacing the exception. The SDK retries with exponential backoff + jitter
# on 408/409/429/5xx (honoring the `retry-after` header for 429), but does
# NOT retry on 400 — which is what credit-balance / insufficient-funds
# errors return ("Your credit balance is too low...") — so pushing this
# higher is safe: 429s get smoothed out while a real balance failure still
# surfaces immediately.
#
# At --concurrency 2 on the 30k-ITPM tier, ~every 60-70 s of sustained
# traffic the token bucket briefly runs dry; 8 retries with backoff absorb
# those stalls across a multi-hour 1500-row run.
DEFAULT_MAX_RETRIES = int(os.environ.get("ANTHROPIC_MAX_RETRIES", "8"))

# Valid reasoning_effort levels for the adaptive-thinking API (Claude Opus 4.7+).
# 'none' -> thinking disabled; everything else -> adaptive + output_config.effort.
# 'xhigh' is rejected (the Anthropic effort enum tops out at 'high').
_VALID_EFFORT = {"none", "minimal", "low", "medium", "high"}
_ADAPTIVE_EFFORTS = {"minimal", "low", "medium", "high"}


def montage_path(target_class: str, oid: str, root: Path | None = None) -> Path:
    base = root or ROOT
    return base / STAMPS_LLM_DIRNAME / target_class / oid / "montage.png"


def _encode_image(image_path: Path) -> str:
    """Read PNG and return a base64 string (no data: URI prefix — Anthropic wants raw b64)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


_client_cache: dict[str, "anthropic.Anthropic"] = {}
_client_lock = threading.Lock()


def _get_client() -> "anthropic.Anthropic":
    """Cached `anthropic.Anthropic` (per-process). Reads ANTHROPIC_API_KEY from env.

    `max_retries` is set from `ANTHROPIC_MAX_RETRIES` (default 8) so the SDK
    auto-retries 429 rate-limit errors with exponential backoff (honoring the
    `retry-after` header). 400-level errors (including credit-balance / billing
    failures) are NOT retried — they surface immediately, which is what we
    want for a long-running benchmark.
    """
    with _client_lock:
        client = _client_cache.get("default")
        if client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Add it to .env or export it before running."
                )
            client = anthropic.Anthropic(
                api_key=api_key,
                max_retries=DEFAULT_MAX_RETRIES,
            )
            _client_cache["default"] = client
        return client


def _normalize_effort(effort: str | None) -> str:
    """Resolve and validate reasoning_effort for Claude Opus 4.7.

    Accepts {none, minimal, low, medium, high}. 'xhigh' is rejected (the
    Anthropic effort enum tops out at 'high') with a clear error instead of
    an opaque server-side 400.
    """
    e = (effort or DEFAULT_REASONING_EFFORT or "none").lower()
    if e == "xhigh":
        raise ValueError(
            "reasoning_effort='xhigh' is not supported by the Anthropic backend: "
            "the Claude Opus 4.7 effort enum tops out at 'high'. "
            "Use --reasoning-effort high."
        )
    if e not in _VALID_EFFORT:
        raise ValueError(
            f"Invalid reasoning_effort {effort!r}. Must be one of {sorted(_VALID_EFFORT)}."
        )
    return e


def build_messages(
    oid: str,
    target_class: str,
    metadata: dict[str, Any],
    image_path: Path,
) -> list[dict[str, Any]]:
    """Construct an Anthropic Messages API `messages` list with one image + text user msg."""
    user_text = build_user_prompt(oid, metadata)
    image_b64 = _encode_image(image_path)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]


def _reasoning_is_effectively_on(effort: str) -> bool:
    """Return True iff `effort` enables Claude's adaptive thinking."""
    return effort in _ADAPTIVE_EFFORTS


def sample_vlm(
    messages: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """
    Run one VLM completion via Anthropic Messages API (Claude Opus 4.7 shape).

    max_tokens policy (matches api_tinker / api_google):
      - Caller default is 2048 (ample for the JSON answer when thinking is off).
      - When `reasoning_effort` enables thinking AND the caller did not raise
        the cap themselves (<= 4096), we auto-bump the effective cap to 20000.
      - Callers that pass a larger explicit cap are respected.

    Thinking controls:
      - 'none'     -> thinking={"type": "disabled"}
      - otherwise  -> thinking={"type": "adaptive"} + output_config.effort=<level>

    Returns the same dict shape as api_openai / api_google / api_tinker so
    evaluate.py is unchanged:
      - "raw_text": visible response text (post-thinking; the 'text' content block)
      - "answer_text": same as raw_text
      - "n_output_tokens": total output tokens (visible + thinking)
      - "n_answer_tokens": visible-only tokens (output - thinking, if available)
      - "n_reasoning_tokens": thinking tokens reported by usage (0 if disabled
        or not exposed by the server)
      - "n_prompt_tokens": input tokens
      - "max_tokens": effective cap actually applied
      - "truncated": True if stop_reason == "max_tokens"
      - "finish_reason": Anthropic stop_reason ("end_turn", "max_tokens",
        "stop_sequence", "tool_use", ...)
      - "reasoning_effort": effort level actually used
    """
    client = _get_client()
    effort = _normalize_effort(reasoning_effort)

    effective_max = max_tokens
    thinking_on = _reasoning_is_effectively_on(effort)
    if thinking_on and effective_max <= 4096:
        effective_max = 20000

    kwargs: dict[str, Any] = {
        "model": model_name,
        "max_tokens": effective_max,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }

    if thinking_on:
        kwargs["thinking"] = {"type": "adaptive"}
        # Opus 4.7 surfaces a separate output_config block that carries the
        # effort knob. Passed through `extra_body` so older anthropic SDK
        # versions (pre-Opus-4.7) tolerate the field on the way to the server.
        kwargs["extra_body"] = {"output_config": {"effort": effort}}
    else:
        kwargs["thinking"] = {"type": "disabled"}
    # Note: Claude Opus 4.7 rejects `temperature` ("deprecated for this model"),
    # so we never forward the caller's `temperature` argument. It remains part
    # of the signature for interface symmetry with the other backends.
    _ = temperature

    resp = client.messages.create(**kwargs)

    # Extract visible text (sum all `text` content blocks). Thinking blocks
    # live alongside as type=="thinking"; we ignore them for `raw_text` to
    # match the behavior of the other backends (they hide the CoT too).
    visible_parts: list[str] = []
    for block in resp.content or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            visible_parts.append(getattr(block, "text", "") or "")
    raw_text = "".join(visible_parts).strip()

    # Usage accounting. `output_tokens` is total (thinking + visible). Some SDK
    # versions surface a separate `cache_creation_input_tokens` / thinking
    # breakdown; probe defensively.
    usage = getattr(resp, "usage", None)
    n_prompt = int(getattr(usage, "input_tokens", 0) or 0)
    n_output = int(getattr(usage, "output_tokens", 0) or 0)
    n_reasoning = 0
    for cand in ("thinking_tokens", "cache_creation_input_tokens"):
        v = getattr(usage, cand, None) if usage else None
        if v:
            # Only 'thinking_tokens' is really reasoning; cache_creation is input-side.
            if cand == "thinking_tokens":
                n_reasoning = int(v)
            break
    n_answer = max(0, n_output - n_reasoning)

    stop_reason = getattr(resp, "stop_reason", None) or "end_turn"
    truncated = stop_reason == "max_tokens"

    return {
        "raw_text": raw_text,
        "answer_text": raw_text,
        "n_output_tokens": n_output,
        "n_answer_tokens": n_answer,
        "n_reasoning_tokens": n_reasoning,
        "n_prompt_tokens": n_prompt,
        "max_tokens": effective_max,
        "truncated": truncated,
        "finish_reason": stop_reason,
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
    """Build metadata from manifest row, load montage, call Anthropic Claude VLM."""
    meta = manifest_row_to_metadata(row)
    img = montage_path(target_class, oid, stamps_root)
    if not img.is_file():
        raise FileNotFoundError(f"Missing montage: {img}")
    messages = build_messages(oid, target_class, meta, img)
    vlm_out = sample_vlm(
        messages,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
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

"""
Google Gemini API runner for ZTF/ALeRCE benchmark (vision-language).

Mirrors the public surface of api_tinker.py / api_openai.py so
run_tinker_benchmark.py can dispatch to any of the three backends
without touching evaluate.py or prompts.py.

Pipeline:
  (1) First-alert inputs: image triplet (one PNG montage = Science|Reference|Difference)
      + ZTF metadata.
  (2) Prompt construction: SYSTEM_PROMPT + user message from prompts.py.
  (3) Gemini `generate_content` call (vision + optional thinking) -> JSON string
      -> evaluate.py parser.

Requires:
  - export GOOGLE_API_KEY=...       (or put GOOGLE_API_KEY=... in .env)
  - pip install google-genai        (the modern unified SDK for Gemini 2.5 / 3.x)

Default model: gemini-2.5-pro. Override via env GOOGLE_MODEL or CLI.

Reasoning effort is mapped to the appropriate `ThinkingConfig` field for each
Gemini family. Gemini 2.5 takes an integer `thinking_budget` (token cap /
sentinel); Gemini 3.x takes a string `thinking_level`.

Gemini 2.5 (integer `thinking_budget`):
    gemini-2.5-pro          : budget in {128..32768} or -1 (dynamic).
                              Cannot disable (no budget=0). 'none'/'minimal'
                              are rejected.
    gemini-2.5-flash        : budget in {0..24576} or -1 (dynamic).
                              Can disable with 'none' (budget=0).
    gemini-2.5-flash-lite   : same as Flash; 'none' allowed.

    Effort -> budget mapping on 2.5:
      none    -> 0            (Flash family only)
      minimal -> 128          (Flash family only)
      low     -> 1024
      medium  -> 8192
      high    -> -1           (dynamic: let Gemini allocate as needed)
      xhigh   -> rejected (use 'high').

Gemini 3.x (string `thinking_level`):
    gemini-3.1-pro*          : {low, medium, high}     (no 'minimal', no 'none')
    gemini-3.1-flash*        : {minimal, low, medium, high}
    gemini-3.1-flash-lite*   : {minimal, low, medium, high}

    'none' is rejected for Gemini 3.x (thinking-only). 'xhigh' is rejected
    (enum tops out at 'high').

A caller passing an unsupported effort (e.g. 'none' on Pro, 'xhigh' anywhere)
gets a clear, actionable error instead of an opaque server-side 400.
"""
from __future__ import annotations

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
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "api_google.py requires the google-genai package. "
        "Install it with:  pip install google-genai"
    ) from exc

from prompts import STAMPS_LLM_DIRNAME, SYSTEM_PROMPT, build_user_prompt, manifest_row_to_metadata

DEFAULT_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-pro")
DEFAULT_REASONING_EFFORT = os.environ.get("GOOGLE_REASONING_EFFORT", "high")

# Automatic retry config for the Gemini HTTP client. Gemini 3.1 Pro is in
# high demand and returns transient 503 ("model overloaded") / 429 ("resource
# exhausted") errors during peak hours; the SDK will retry these with
# exponential backoff + jitter, honoring 4xx client errors (billing, auth,
# invalid request) by failing them fast.
#
# Retryable status codes: 429, 500, 502, 503, 504
# NOT retried: 400 (bad request / safety block), 401 (auth), 403 (permission
# / billing), 404 (model not found) — these won't resolve by retrying.
DEFAULT_MAX_RETRIES = int(os.environ.get("GOOGLE_MAX_RETRIES", "10"))
DEFAULT_RETRY_INITIAL_DELAY = float(os.environ.get("GOOGLE_RETRY_INITIAL_DELAY", "2.0"))
DEFAULT_RETRY_MAX_DELAY = float(os.environ.get("GOOGLE_RETRY_MAX_DELAY", "60.0"))

# -------- Gemini 3.x: string `thinking_level` --------
# 'minimal' is only available on Flash / Flash-Lite; Pro starts at 'low'.
_V3_PRO_LEVELS = {"low", "medium", "high"}
_V3_FLASH_LEVELS = {"minimal", "low", "medium", "high"}

# -------- Gemini 2.5: integer `thinking_budget` --------
# Effort -> budget (shared map; per-model validity is enforced separately).
#   high -> -1 means "dynamic": let Gemini choose how much to think (up to
#   the model's internal max). This is the recommended "max thinking" setting
#   for 2.5 Pro. For 2.5 Flash, 'none' (budget=0) disables thinking entirely.
_V25_EFFORT_BUDGETS: dict[str, int] = {
    "none": 0,
    "minimal": 128,
    "low": 1024,
    "medium": 8192,
    "high": -1,
}
_V25_PRO_EFFORTS = {"low", "medium", "high"}       # Pro cannot disable thinking
_V25_FLASH_EFFORTS = {"none", "minimal", "low", "medium", "high"}


def _model_family(model_name: str) -> str:
    """Return 'v25' for Gemini 2.5, 'v3' for Gemini 3.x, else 'v3' as a safe default.

    The distinction matters because 2.5 takes an integer `thinking_budget` and
    3.x takes a string `thinking_level`; we validate and dispatch accordingly.
    """
    m = (model_name or "").lower()
    if "gemini-2.5" in m or "gemini-2-5" in m:
        return "v25"
    return "v3"


def _is_flash_variant(model_name: str) -> bool:
    m = (model_name or "").lower()
    return "flash" in m  # covers flash and flash-lite


def _allowed_levels_for_model(model_name: str) -> set[str]:
    """Set of valid `reasoning_effort` values for `model_name` (any family)."""
    family = _model_family(model_name)
    flash = _is_flash_variant(model_name)
    if family == "v25":
        return _V25_FLASH_EFFORTS if flash else _V25_PRO_EFFORTS
    # v3
    return _V3_FLASH_LEVELS if flash else _V3_PRO_LEVELS


def montage_path(target_class: str, oid: str, root: Path | None = None) -> Path:
    base = root or ROOT
    return base / STAMPS_LLM_DIRNAME / target_class / oid / "montage.png"


_client_cache: dict[str, Any] = {}
_client_lock = threading.Lock()


def _get_client():
    """Cached `genai.Client` (per-process). Reads GOOGLE_API_KEY from env.

    The client is configured with `HttpRetryOptions` so transient 429/5xx
    errors (including Gemini 3.1 Pro's frequent 503 "model overloaded" under
    high demand) are retried with exponential backoff + jitter. 4xx errors
    like 400 (bad request), 401/403 (auth/billing), 404 (model not found) are
    NOT retried — they surface immediately so a real problem doesn't spin.
    """
    with _client_lock:
        client = _client_cache.get("default")
        if client is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY not set. Add it to .env or export it before running."
                )
            retry_opts = genai_types.HttpRetryOptions(
                attempts=DEFAULT_MAX_RETRIES,
                initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
                max_delay=DEFAULT_RETRY_MAX_DELAY,
                exp_base=2.0,
                jitter=0.3,
                http_status_codes=[429, 500, 502, 503, 504],
            )
            http_opts = genai_types.HttpOptions(retry_options=retry_opts)
            client = genai.Client(api_key=api_key, http_options=http_opts)
            _client_cache["default"] = client
        return client


def _normalize_effort(effort: str | None, model_name: str) -> str:
    """Resolve and validate reasoning_effort for `model_name`.

    - 'xhigh' is always rejected (Gemini's max is 'high' / dynamic).
    - Gemini 3.x rejects 'none' (thinking-only family).
    - Gemini 2.5 Pro rejects 'none' and 'minimal' (Pro cannot disable thinking).
    - Gemini 2.5 Flash accepts the full set {none, minimal, low, medium, high}.
    """
    e = (effort or DEFAULT_REASONING_EFFORT or "high").lower()
    family = _model_family(model_name)
    flash = _is_flash_variant(model_name)
    allowed = _allowed_levels_for_model(model_name)

    if e == "xhigh":
        raise ValueError(
            "reasoning_effort='xhigh' is not supported by the Google backend: "
            "Gemini caps out at 'high' (dynamic thinking on 2.5, thinking_level=high on 3.x). "
            "Use --reasoning-effort high."
        )

    if e == "none" and "none" not in allowed:
        if family == "v3":
            raise ValueError(
                "reasoning_effort='none' is not supported on Gemini 3.x: "
                "these models are thinking-only. "
                f"Use one of {sorted(allowed)} (e.g. 'low' for minimum)."
            )
        # v25 Pro
        raise ValueError(
            "reasoning_effort='none' is not supported on gemini-2.5-pro: "
            "Pro cannot disable thinking. Use a Flash variant "
            "(e.g. gemini-2.5-flash) for non-reasoning runs."
        )

    if e not in allowed:
        hint = ""
        if e == "minimal" and not flash:
            if family == "v25":
                hint = (
                    " 'minimal' is only available on gemini-2.5-flash*; "
                    "gemini-2.5-pro starts at 'low'."
                )
            else:
                hint = (
                    " 'minimal' is only available on gemini-3.1-flash* "
                    "(Flash / Flash-Lite); gemini-3.1-pro* starts at 'low'."
                )
        raise ValueError(
            f"Invalid reasoning_effort {effort!r} for model {model_name!r}. "
            f"Must be one of {sorted(allowed)}.{hint}"
        )
    return e


def build_contents(
    oid: str,
    target_class: str,
    metadata: dict[str, Any],
    image_path: Path,
) -> list[Any]:
    """Build the Gemini `contents` list with one user turn (image + text)."""
    user_text = build_user_prompt(oid, metadata)
    img_bytes = image_path.read_bytes()
    image_part = genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    text_part = genai_types.Part.from_text(text=user_text)
    return [
        genai_types.Content(role="user", parts=[image_part, text_part]),
    ]


def _reasoning_is_effectively_on(effort: str) -> bool:
    """Return True iff `effort` triggers internal reasoning.

    - 'none' (Gemini 2.5 Flash only) disables thinking entirely -> False.
    - Every other supported level ({minimal, low, medium, high}) performs at
      least some internal thinking, so the effective cap should be raised to
      give the model room for CoT + visible answer.
    """
    return effort in {"minimal", "low", "medium", "high"}


def _build_thinking_config(effort: str, model_name: str):
    """Build a `ThinkingConfig` appropriate for the model family.

    - Gemini 2.5: integer `thinking_budget` (0 = off, -1 = dynamic, else token cap).
    - Gemini 3.x: string `thinking_level` ({minimal, low, medium, high}).

    `effort` has already been validated against the model by `_normalize_effort`.
    """
    if _model_family(model_name) == "v25":
        budget = _V25_EFFORT_BUDGETS[effort]
        return genai_types.ThinkingConfig(thinking_budget=budget)
    return genai_types.ThinkingConfig(thinking_level=effort)


def sample_vlm(
    contents: list[Any],
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """
    Run one VLM completion via Gemini `generate_content`.

    max_tokens policy (matches api_tinker.sample_vlm):
      - Caller default is 2048 (ample for the JSON answer when reasoning is off).
      - When `reasoning_effort` triggers thinking AND the caller did not raise
        the cap themselves (<= 4096), we auto-bump the effective cap to 20000
        so that Gemini has room for thinking + visible answer.
      - Callers that pass a larger explicit cap (> 4096) are respected as-is.

    NOTE: 'none' (Gemini 2.5 Flash only) disables thinking — the 2048 default
    applies. Gemini 3.x is thinking-only and 2.5 Pro always thinks, so for
    those the 20000 branch fires whenever the caller hasn't raised max_tokens.

    Returns the same dict shape as api_openai.sample_vlm / api_tinker.sample_vlm
    so evaluate.py is unchanged:
      - "raw_text": visible response text (Gemini hides the thinking trace;
        this is the post-thinking answer)
      - "answer_text": same as raw_text — Gemini does not expose the CoT
      - "n_output_tokens": total output tokens (visible + thinking)
      - "n_answer_tokens": visible-only tokens
      - "n_reasoning_tokens": `thoughts_token_count` from the Gemini usage
      - "n_prompt_tokens": `prompt_token_count`
      - "max_tokens": effective cap actually applied (after any auto-bump)
      - "truncated": True if candidate finish_reason == MAX_TOKENS
      - "finish_reason": enum name from the first candidate ("STOP", "MAX_TOKENS",
        "SAFETY", ...) or "completed" if none returned
      - "reasoning_effort": effort level actually used
    """
    client = _get_client()
    effort = _normalize_effort(reasoning_effort, model_name)

    effective_max = max_tokens
    if _reasoning_is_effectively_on(effort) and max_tokens <= 4096:
        effective_max = 20000

    config_kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM_PROMPT,
        "max_output_tokens": effective_max,
        "temperature": temperature,
        # Gemini 2.5 expects integer `thinking_budget` (0=off, -1=dynamic);
        # Gemini 3.x expects string `thinking_level`. _build_thinking_config
        # dispatches on the model family.
        "thinking_config": _build_thinking_config(effort, model_name),
    }
    config = genai_types.GenerateContentConfig(**config_kwargs)

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    raw_text = (getattr(resp, "text", None) or "").strip()

    # Token accounting via usage_metadata.
    usage = getattr(resp, "usage_metadata", None)
    n_prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    n_answer = int(getattr(usage, "candidates_token_count", 0) or 0)
    n_thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
    # Some Gemini responses populate only total_token_count; derive visible+thinking
    # from it as a fallback.
    n_output = n_answer + n_thoughts
    if n_output == 0:
        total = int(getattr(usage, "total_token_count", 0) or 0)
        n_output = max(0, total - n_prompt)

    # Finish reason / truncation from the first candidate.
    finish_reason = "completed"
    truncated = False
    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        if fr is not None:
            fr_str = getattr(fr, "name", None) or str(fr).split(".")[-1]
            finish_reason = fr_str
            truncated = fr_str == "MAX_TOKENS"

    return {
        "raw_text": raw_text,
        "answer_text": raw_text,
        "n_output_tokens": n_output,
        "n_answer_tokens": n_answer,
        "n_reasoning_tokens": n_thoughts,
        "n_prompt_tokens": n_prompt,
        "max_tokens": effective_max,
        "truncated": truncated,
        "finish_reason": finish_reason,
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
    """Build metadata from manifest row, load montage, call Gemini VLM."""
    meta = manifest_row_to_metadata(row)
    img = montage_path(target_class, oid, stamps_root)
    if not img.is_file():
        raise FileNotFoundError(f"Missing montage: {img}")
    contents = build_contents(oid, target_class, meta, img)
    vlm_out = sample_vlm(
        contents,
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

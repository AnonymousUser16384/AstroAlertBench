"""
Few-shot ablation prompts: one human-expert vignette per benchmark class (5 montages).

Imports the zero-shot contract from `prompts.py`, then prepends expert-style examples
suited to multi-image APIs (one PNG montage per example before the target alert).

Reasoning texts:
- Supernova and AGN: external expert review (Matthew) on the listed oids.
- Variable Star, bogus, asteroid: single short notes from Zooniverse "Expert Example (Final)"
  (reviewers theodlz / aschig)—same informal style, not Parts A/B/C JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompts import (
    SYSTEM_PROMPT,
    STAMPS_LLM_DIRNAME,
    build_user_prompt,
    manifest_row_to_metadata,
)

ROOT = Path(__file__).resolve().parent.parent

# One demo per class; image attachment order matches this tuple.
ABLATION_CLASS_ORDER: tuple[str, ...] = ("VS", "SN", "AGN", "bogus", "asteroid")

_EXPERT_STYLE_NOTE = """
About the expert examples below:
- They are short, free-text astronomer notes (Zooniverse-style captions or a few sentences),
  not the structured AstroAlertBench answer (Part A/B/C + stage JSON) you must still produce
  for the new alert.
- Use them only as loose intuition; follow the system message for the final JSON.
""".strip()


@dataclass(frozen=True)
class ExpertAblationShot:
    """One demo montage + short expert reasoning for the ablation block."""

    target_class: str  # VS | SN | AGN | bogus | asteroid (benchmark folder + manifest)
    oid: str
    expert_username: str
    expert_reasoning: str


def _shots() -> tuple[ExpertAblationShot, ...]:
    return (
        ExpertAblationShot(
            "VS",
            "ZTF17aadkrvp",
            "aschig",
            "sgscore1 and classtar high. also ndethist",
        ),
        ExpertAblationShot(
            "SN",
            "ZTF24aaiulbk",
            "Matthew",
            (
                "Although there is a nearby persistent source in the reference image, it has a sgscore of 0 "
                "indicating a galaxy. The transient is point-like (classtar = 0.9770) and it only has "
                "ndethist = 5 whereas ncovhist = 3548. The transient has been going on for only 21 days "
                "so it is recent. This suggests a supernova. PS1 colors (g-r, r-i) put on the edge of the "
                "AGN space so a SN is more likely."
            ),
        ),
        ExpertAblationShot(
            "AGN",
            "ZTF19abkexhe",
            "Matthew",
            (
                "There is a persistent source in the reference image. Ndethist = 48 with ncovhist = 327 and "
                "deltajd = 2326 so this is a variable source. Nearest source is galaxy-like and the PS1 "
                "colors are consistent with AGN. It is also a negative flux change so variable."
            ),
        ),
        ExpertAblationShot(
            "bogus",
            "ZTF23abqmngv",
            "theodlz",
            "definitely a really bad difference image. just dipoles. looks like an astrometry issue",
        ),
        ExpertAblationShot(
            "asteroid",
            "ZTF26aapnqlu",
            "theodlz",
            "just one detection, no visible host, most likely sso",
        ),
    )


EXPERT_ABLATION_SHOTS: tuple[ExpertAblationShot, ...] = _shots()


def montage_path(target_class: str, oid: str, stamps_root: Path | None = None) -> Path:
    """Same layout as api_* runners: ``<root>/<STAMPS_LLM_DIRNAME>/<class>/<oid>/montage.png``."""
    base = stamps_root or ROOT
    return base / STAMPS_LLM_DIRNAME / target_class / oid / "montage.png"


def expert_ablation_montage_paths(stamps_root: Path | None = None) -> list[Path]:
    """Five demo montages in prompt order (one per class: VS, SN, AGN, bogus, asteroid)."""
    return [montage_path(s.target_class, s.oid, stamps_root) for s in EXPERT_ABLATION_SHOTS]


def build_expert_ablation_preamble_text() -> str:
    """Human-readable block listing demos; parallel to `expert_ablation_montage_paths()`."""
    if len(EXPERT_ABLATION_SHOTS) != len(ABLATION_CLASS_ORDER):
        raise RuntimeError("EXPERT_ABLATION_SHOTS must align with ABLATION_CLASS_ORDER")
    lines: list[str] = [
        "[EXPERT-STYLE FEW-SHOT EXAMPLES — READ BEFORE YOUR TASK]",
        "",
        "You will receive 5 separate image attachments before this text block's subject alert:",
        "one Science|Reference|Difference montage per benchmark class, in this order: "
        + ", ".join(ABLATION_CLASS_ORDER)
        + ".",
        "Each line below gives the benchmark gold class, ZTF object id, which expert wrote the note,",
        "and their reasoning (informal text, not an LLM JSON answer).",
        "",
        _EXPERT_STYLE_NOTE,
        "",
    ]
    for i, shot in enumerate(EXPERT_ABLATION_SHOTS, start=1):
        if shot.target_class != ABLATION_CLASS_ORDER[i - 1]:
            raise RuntimeError(
                f"Shot order mismatch at index {i}: {shot.target_class} vs {ABLATION_CLASS_ORDER[i - 1]}"
            )
        reasoning = shot.expert_reasoning.strip()
        lines.append(
            f"Image {i} — class={shot.target_class} | oid={shot.oid} | expert={shot.expert_username} | reasoning: {reasoning}"
        )
    lines.append("")
    lines.append(
        "Next, classify the NEW alert described below (metadata + one more montage after the 5 examples)."
    )
    lines.append("")
    return "\n".join(lines)


def build_user_prompt_expert_ablation(
    oid: str,
    metadata: dict[str, Any],
    *,
    include_preamble: bool = True,
) -> str:
    """User message: optional 5-shot expert preamble + standard zero-shot task from `prompts.py`."""
    tail = build_user_prompt(oid, metadata)
    if not include_preamble:
        return tail
    return build_expert_ablation_preamble_text() + tail


def ablation_message_image_paths(
    target_class: str,
    target_oid: str,
    stamps_root: Path | None = None,
) -> list[Path]:
    """Five expert demo paths then the held-out alert montage (6 images total)."""
    demos = expert_ablation_montage_paths(stamps_root)
    test = montage_path(target_class, target_oid, stamps_root)
    return demos + [test]


__all__ = [
    "ABLATION_CLASS_ORDER",
    "EXPERT_ABLATION_SHOTS",
    "ExpertAblationShot",
    "SYSTEM_PROMPT",
    "STAMPS_LLM_DIRNAME",
    "ablation_message_image_paths",
    "build_expert_ablation_preamble_text",
    "build_user_prompt_expert_ablation",
    "expert_ablation_montage_paths",
    "manifest_row_to_metadata",
    "montage_path",
]

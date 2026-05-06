"""
Same user message as prompts.py, with extra system guidance to discourage
runaway re-checking loops in reasoning models (e.g. Qwen3.5, Kimi-K2.5 thinking).

Motivation
----------
Reasoning models occasionally enter "let me re-check that..." spirals,
re-evaluating the same evidence (often the difference image) until they
exhaust the token budget without ever emitting JSON. When that happens the
prediction is unparseable and counts as a failure. This module instructs
the model to time-box deliberation, commit to a label after a small number
of considerations, and surface remaining uncertainty inside Part B
(`alternative_analysis`) instead of in further internal monologue.

Usage
-----
  python run_tinker_benchmark.py --manifest data/manifest_enriched.csv \\
      --out results/run_token_limit.jsonl --prompts prompts_token_limit_instruction
"""

from __future__ import annotations

import prompts

build_user_prompt = prompts.build_user_prompt
manifest_row_to_metadata = prompts.manifest_row_to_metadata
required_manifest_columns = prompts.required_manifest_columns

REASONING_BUDGET_GUIDANCE = """
Reasoning budget and termination rules (these override any tendency to keep deliberating):

1. Soft thinking budget: aim to keep your internal reasoning under ~8000 tokens. The hard limit is a few thousand higher; if you hit the hard limit you produce no JSON and the answer is wasted. A short, decisive analysis that emits valid JSON is strictly better than a long, thorough analysis that gets cut off.

2. Two-pass rule: examine each piece of evidence (the three image panels and the metadata) at most twice. After the second pass, do not revisit it. Move to the next required field.

3. Anti-spiral triggers — if you catch yourself starting any of the following, STOP that line of thought immediately and commit to the most probable label so far:
   - "Wait, let me re-evaluate ..."
   - "Actually, on second thought ..."
   - "Let me look at the difference image again ..."
   - "Hmm, but what if ..."
   - Any third re-reading of the same panel or the same metadata field.

4. Decision forcing: if after two passes you remain uncertain between two classes, pick the one supported by the stronger single piece of evidence. Record the genuine uncertainty in Part B `alternative_analysis` (and lower `confidence_overall` accordingly), NOT by continuing to deliberate.

5. Output-first commitment: as soon as you have a tentative classification you can defend in one or two sentences, draft the JSON object in your head and proceed to write it. Do not refine the prose further once the structure is decided.

6. Hard requirement: the final assistant message MUST contain a complete, valid JSON object with all required Part A / Part B / Part C fields. An incomplete JSON or a response that is only reasoning text counts as a failure regardless of how good the reasoning was.

These rules apply to all classes (SN, AGN, variable_star, asteroid, bogus). They do not change the schema, the metadata interpretation, or the per-class evidence rules — they only constrain how long you may deliberate before committing.
"""

SYSTEM_PROMPT = prompts.SYSTEM_PROMPT + "\n\n" + REASONING_BUDGET_GUIDANCE

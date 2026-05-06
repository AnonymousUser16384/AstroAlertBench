"""
Full ZTF metadata prompt (same user message as prompts.py) with extra system guidance
for Part C stage3: AGN vs variable_star when stage2 = astrophysical.

Usage:
  python run_tinker_benchmark.py --manifest data/manifest_enriched.csv \\
      --out results/run_agn_prompt.jsonl --prompts prompts_agn_instruction
"""

from __future__ import annotations

import prompts

build_user_prompt = prompts.build_user_prompt
manifest_row_to_metadata = prompts.manifest_row_to_metadata
required_manifest_columns = prompts.required_manifest_columns

AGN_VS_VARIABLE_STAR_GUIDANCE = """
Additional guidance for Part B and Part C (single JSON response; write Part B before Part C in the output order):

When stage1 = real_object and stage2 = astrophysical, you must choose stage3 as either supernova, variable_star, or AGN. For the specific fork AGN vs variable_star, you must ground the choice in the provided evidence as follows.

Part B requirement (before you finalize Part C):
- In key_evidence and/or leading_interpretation_and_support, briefly cite relevant items when they are available and not sentinels: (i) cutout morphology (e.g. clear galaxy host or nuclear source vs isolated stellar profile), (ii) sgscore1 and distpsnr1 for the closest Pan-STARRS1 match, (iii) PS1 optical colors derived from sgmag1, srmag1, simag1 when those magnitudes are valid, (iv) ndethist, ncovhist, and deltajd only as weak, survey-definition-dependent context—never as a sole decisive rule.
- Part C stage3 must be consistent with the reasoning you wrote in Part B.

Colors from sgmag1/srmag1/simag1 are properties of the matched PS1 catalog source (see main field reference), not necessarily the transient alone.

PS1 colors (AB magnitudes as in the alert; compute only when values are real, not -999 or N/A):
- g_minus_r = sgmag1 - srmag1
- r_minus_i = srmag1 - simag1
Smaller (more negative) g_minus_r corresponds to a bluer g-r in the usual sense; larger g_minus_r and r_minus_i correspond to redder optical colors.

Interpreting sgscore1 and distpsnr1 (ZTF schema: closest PS1 source within 30 arcsec):
- sgscore1 is between 0 and 1; values closer to 1 imply a higher likelihood that the matched PS1 source is star-like; values closer to 0 are more galaxy-like.
- distpsnr1 is the angular separation in arcseconds to that closest PS1 source. When distpsnr1 is large, or PS1 magnitudes/scores are missing or sentinel, treat sgscore1 and derived colors as weak or ambiguous—the PS1 object may not be the physical counterpart of the transient.

Qualitative heuristics (not fixed thresholds—survey selection and dust matter):
- Many Galactic variable stars are plausibly described by a stellar-like PS1 neighbor (often higher sgscore1 when distpsnr1 is small) and optical colors consistent with the main stellar locus (commonly bluer to solar-type in g-r and r-i; very red giants and other classes exist, so combine color with cutout morphology).
- AGN are often associated with a galaxy host in the stamps; when the PS1 match is galaxy-like (sgscore1 toward 0) with small distpsnr1, and optical colors are redder or more early-type-like than a typical thin-disk star, AGN can be favored if variability and morphology support an active nucleus rather than a single isolated star.
- Blue, compact nuclei can be ambiguous: some quasars/blazars vs hot or unusual stellar variables. Explicitly weigh host extent, sgscore1, colors, and difference-image behavior together; state the ambiguity in Part B when appropriate.

Do not invent WISE or other bands not present in the metadata. Do not use rigid numeric color cuts (e.g. a single g-r threshold) as rules; reasoning should be qualitative and tied to the actual numbers shown.

When stage3 = supernova, the AGN vs variable_star discussion above is not required beyond what the images and metadata justify.
"""

SYSTEM_PROMPT = prompts.SYSTEM_PROMPT + "\n\n" + AGN_VS_VARIABLE_STAR_GUIDANCE

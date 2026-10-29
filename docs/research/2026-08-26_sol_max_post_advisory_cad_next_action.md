# Sol Max post-advisory CAD next-action consultation

> AI research consultation only. This is not manufacturing authority and does
> not satisfy either required human review.

DECISION: `FORMAL_STOP`

Exact STOP: `EXTERNAL_CAPABILITY_OPENING_REQUIRED / AWAITING_AUTHORITY_REVIEW`. Do not preregister or run `EXP_CAD_0008`. Keep `EXP_CAD_0002_NEUTRAL_HALF_TURN` frozen; runtime budget is zero and no new visual/scoring audit is authorized.

Evidence:

- The AI file explicitly says `ADVISORY_DRAFT_NOT_AUTHORITY`: 60/63 events are advisory `QC_BEND/NEUTRAL_HALF_TURN`, while all three C005 events remain unresolved because process intent is required. It also says the draft must not authorize another ontology-tuning experiment (`AI_SOL_MAX_ADVISORY_REPORT.md`, advisory JSON).
- Broader primitive-to-event contraction is invalid without signed ownership semantics. The frozen Candidate C plan permits contraction “only under the signed CIBEC mapping” and is explicitly “not execution authorization” (`2026-08-25_f1_post_cibec_dual_lane_preregistration_v1.md`).
- The source pipeline already contracts paired coaxial inner/outer cylindrical skins into one neutral event (`compile_cad_bend_template_v2.py`) and already applies scan-line support plus a two-surface normal-turn verifier (`localize_cad_bends_on_scan_v1.py`, `apply_f1_step_topology_targeted_verifier.py`). Its 63 retained predictions passed geometric research visual review; the failure remained event ontology and absolute line convention (`VISUAL_ADJUDICATION.json`).
- EXP_CAD_0007 is the decisive falsifier: 50 relocations passed geometric visual review, yet matched-11 F1 fell from `0.3721` to `0.2791`, losing eight matches and gaining two. Thus another geometric verifier or relocation cannot resolve the unsigned target (`EXP_CAD_0007_RESULT.json`).
- The frozen stop requires two distinct named human authorities, complete independent 63-event decisions, and disagreement adjudication before the CAD loop resumes (`F1_CAPABILITY_OPENING_STOP_20260825.json`).

Permitted now: complete and validate the two human reviews. Forbidden: use the AI advisory as a signature, contract C005, add another fused-PLY selector/veto, tune acceptance thresholds, relocate/interpolate lines, or make promotion claims. After the human gate opens, the highest-value candidate remains signed-CIBEC primitive-to-manufacturing-event contraction—not before.

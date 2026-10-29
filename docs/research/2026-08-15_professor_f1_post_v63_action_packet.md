# Sherman QC F1 — post-V63 action packet

Date: 2026-08-15

## Required ruling

Return exactly one of:

1. `RULING: EXECUTE_ONE` and one fully preregisterable, bounded, source-only intervention that can be implemented and run immediately; or
2. `RULING: CAPABILITY_OPENING_REQUIRED` and the smallest scientifically necessary new input capability; or
3. `RULING: SOURCE_ONLY_STOP` if no materially different source-only mechanism remains.

Do not propose a menu. Do not propose threshold tuning, rescue, or another paper-only stage. Ground the mechanism in relevant research where useful, but finish with an executable specification.

## Objective and user mandate

The objective remains materially better unseen-part bend-line F1/generalization. The user explicitly wants rapid prospective tests, early retirement of failures, and native-resolution visual critique. The next step must attack a demonstrated failure, not merely add theoretical machinery.

Gold, labels, CAD, expected counts, filename/identity, benchmark feedback, runtime/main, and production remain closed. Private synthetic truth may be opened exactly once only after a new intervention and its outputs are frozen. No tuning or rerun rescue.

## Retained lineage

V52 remains frozen and retained as the research baseline. V53–V63 are terminal and unavailable for rescue. V63 may not be relaxed, retuned, rerun, or imported as a decision rule.

## Prospective V5R3B lane

The V5R3B batch was visually accepted before inference and then sealed. It contains six source-only organized-range triangle meshes with calibrated noise/dropout and 12 views each:

- O001: four-sided tray with ears, window, and short creases;
- O003: windowed U-guard with a broad rolled transition;
- O005: continuous finite local stamped hat;
- O007: vented guard with a broad roll;
- O009: deep box with integral hem/returns;
- O011: asymmetric instrument bracket with small creases and a rounded negative.

The batch is intentionally imperfect synthetic engineering data, not a substitute for the physical corpus. Its capture density/spacing and scan dropout were conditioned on the observed LiDAR corpus. It is materially closer than the rejected first V5R3 attempt, but remains cleaner and more topologically regular than many real parts. The six native review overlays were inspected before truth. The sealed batch is now consumed and may not be regenerated from its outcomes.

## Baseline activation and V63 result

The one-time truth opening showed a valid mixed baseline:

| system | TP | FP | FN | precision | recall | F1 | exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| V62/V52 lineage baseline | 9 | 9 | 25 | 0.5000 | 0.2647 | 0.3462 | 0 |
| V63 LTCO | 7 | 8 | 27 | 0.4667 | 0.2059 | 0.2857 | 0 |

V63 deleted three V52-origin predictions: one false positive and two true positives. It lowered precision, recall, and F1 and is terminally retired.

Per case:

- O001: unchanged at 2 TP / 1 FP / 4 FN.
- O003: 2/2/2 -> 1/2/3; one legitimate long guard/channel transition was deleted.
- O005: 2/2/4 -> 2/1/4; one long oblique false line was correctly deleted.
- O007: unchanged at 0/3/4; the conspicuous parallel broad-roll false-positive doublet survived.
- O009: 3/0/7 -> 2/0/8; one legitimate long return/hem bend was deleted.
- O011: unchanged at 0/1/4.

## Native visual finding

All six scored overlays and all three transient profile sheets were inspected at native resolution.

The decisive residual is O007: two nearly parallel, long predictions run along a smooth rolled surface and both are false. They survived V63 unchanged. Conversely, real long structural transitions on O003 and O009 were removed. The next mechanism must distinguish a smooth-surface parallel roll doublet from real long wall/return/channel bends.

V63's profile sheets expose its causal flaw. O005 and O009 both returned `INSUFFICIENT_REQUESTED_PROFILE_EXTENT` at all nine stations, yet the O005 deletion was false and the O009 deletion was true. O003 was dominated by missing-support, insufficient-coverage, and output-cap states. V63 therefore deleted by missingness rather than discriminative bend geometry.

## Non-negotiable exclusions

The next intervention must not:

- treat missing profile coverage, failed profile construction, sparse support, or output-cap exhaustion as negative evidence;
- reuse or relax V63 LTCO predicates;
- tune thresholds against V5R3B truth;
- regenerate V5R3B or replace any consumed case;
- suppress candidates merely because they are long, parallel, cylindrical, rounded, or V52-origin;
- use identity, CAD, labels, gold, expected counts, or benchmark outcomes during inference;
- change geometry of retained V52 lines unless the ruling explicitly selects a replacement architecture and preregisters its invariant;
- authorize a rescue rerun after truth.

## Desired causal direction

Prefer positive evidence that a pair of detections belongs to the same smooth roll rather than negative evidence that one lacks local data. A credible intervention should exploit source-observable distinctions such as continuous normal transport between parallel candidates, shared smooth surface ownership, absence/presence of a localized tangent discontinuity, endpoint/junction attachment, or multi-view phase consistency. It must explain why that evidence should reject O007-like doublets while protecting O003/O009-like structural transitions.

## Required executable specification

For `EXECUTE_ONE`, provide all of the following:

1. exact architecture name, short name, and lock token;
2. whether it is deletion-only, replacement, or append-only;
3. exact eligible input population;
4. allowed source fields and explicitly forbidden fields;
5. deterministic geometry/graph construction;
6. every fixed constant and tie-break;
7. fail-closed behavior for missing/insufficient evidence;
8. precise mutation rule;
9. primary/replay and resource bounds;
10. prospective activation gate;
11. hard retain/retire gate preserving TP/recall and materially reducing FP;
12. required native-resolution visual checks;
13. causal explanation tied to O007 versus O003/O009;
14. research grounding or closest established geometric precedent;
15. a strict no-rerun/no-rescue clause.

The intervention must be small enough to implement, freeze, execute once, score once, visually audit, and retire or retain immediately.

## Frozen evidence paths

- terminal V63 report: `docs/research/2026-08-15_f1_v63_ltco_v1_terminal_retirement.md`
- frozen score: `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/score.json`
- scored overlays: `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/`
- transient profile sheets: `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-ltco/transient-profile-visuals/`
- V63 execution manifest SHA-256: `65d3418e106601a823a2648d8a53eb14abbb8ea4987bd48c3aadcdbd948af886`
- score SHA-256: `26acda117b77fac8dc9e550d6c29515623f0408d0b4d83492a2fb7b1355b2625`

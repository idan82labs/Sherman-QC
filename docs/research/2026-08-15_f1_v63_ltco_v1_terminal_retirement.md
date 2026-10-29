# V63 LTCO V1 terminal retirement

Date: 2026-08-15

## Decision

`V63_LOCALIZED_TURN_CORRIDOR_OWNERSHIP_V1` is **terminally retired**. It is not retained, may not be tuned, rerun, rescued, or moved toward runtime/main/production. V52 remains the retained frozen lineage.

This is a valid negative result from the sealed V5R3B source-only prospective batch. The baseline activation gate passed before the one-time private truth opening (`TP=9`, `FP=9`), so the intervention was tested on a lane containing both real recoveries and false positives.

## Frozen aggregate result

| system | TP | FP | FN | precision | recall | F1 | exact cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| V62 baseline | 9 | 9 | 25 | 0.5000 | 0.2647 | 0.3462 | 0 |
| V63 LTCO | 7 | 8 | 27 | 0.4667 | 0.2059 | 0.2857 | 0 |

V63 deleted three V52-origin predictions: one false positive and two true positives. It therefore reduced both precision and recall and lowered F1 by 0.0604 absolute (17.45% relative).

## Hard-gate result

The activation gates passed. The retention gates failed:

- TP preservation: **fail** (`9 -> 7`)
- FP quartering: **fail** (`9 -> 8`)
- survivor precision >= 0.90: **fail** (`7/15 = 0.4667`)
- identical recall: **fail** (`0.2647 -> 0.2059`)
- overall precision improvement: **fail** (`0.5000 -> 0.4667`)
- at least one exact-case gain: **fail** (`0 -> 0`)
- all deleted predictions false positive: **fail** (two of three were true positive)
- no exact-case loss: pass, but vacuous because neither system had an exact case

## Native-resolution visual audit

All six scored overlays and all three transient profile sheets were manually inspected at native resolution.

1. **O003 — harmful deletion.** V63 removed a visually legitimate long wall/roll transition on the windowed U-guard. The deleted line follows a declared truth line and was a baseline true positive. This is a channel/guard suppression failure.
2. **O005 — useful deletion.** V63 removed one long oblique false line crossing the local stamped-hat part while preserving the two baseline true positives. This is the only intended success.
3. **O009 — harmful deletion.** V63 removed a visually legitimate long return/hem bend on the deep box. The baseline had no false positives on this case, so the intervention could only do harm and did.
4. **O007 — broad-roll false positives survive.** The conspicuous parallel pair of long false detections remains unchanged. The method did not act on the primary visual failure it was meant to control.
5. **O001 and O011 — unchanged.** Their existing false positives and misses remain.

Manual gates therefore fail:

- no broad-roll survivor: **fail** (O007)
- no hat/channel suppression: **fail** (O003; structurally legitimate guard/channel transition removed)
- native visual agreement: **fail** (O003 and O009 deletions contradict the visible geometry and private truth)

## Failure mechanism

The transient profile sheets show that the deletion decisions were dominated by data-availability/error states rather than discriminative turn geometry:

- O003 stations report `INSUFFICIENT_LEFT_SEGMENTS`, `INSUFFICIENT_TWO_SIDED_COVERAGE`, `MISSING_STATION_SOURCE_SUPPORT`, and `PROFILE_OUTPUT_CAP`.
- O005 and O009 report `INSUFFICIENT_REQUESTED_PROFILE_EXTENT` at every station.

The same missing/insufficient-profile condition occurs for both the one correct deletion (O005) and a harmful true-positive deletion (O009). Therefore missing profile extent is not a valid ownership signal. Treating profile-construction failure as evidence against a candidate created deletion-by-missingness, not bend/roll discrimination.

## Evidence and reproducibility

- Frozen score: `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/score.json`
- Score SHA-256: `26acda117b77fac8dc9e550d6c29515623f0408d0b4d83492a2fb7b1355b2625`
- V63 execution manifest SHA-256: `65d3418e106601a823a2648d8a53eb14abbb8ea4987bd48c3aadcdbd948af886`
- Scored overlays: `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/`
- Transient profile sheets: `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-ltco/transient-profile-visuals/`
- Primary/replay execution was identical.
- The sealed V5R3B batch is consumed and terminal. It will not be regenerated or reshaped from this result.

## Constraint for the next cycle

Do not fund another deletion rule whose negative evidence is missing profile coverage, failed profile construction, or output-cap exhaustion. The next bounded architecture must positively identify the surviving broad-roll failure (especially the O007 parallel pair) while protecting long structural return/channel bends like O003 and O009. It must again be prospective, source-only, frozen before truth, and visually audited at native resolution.

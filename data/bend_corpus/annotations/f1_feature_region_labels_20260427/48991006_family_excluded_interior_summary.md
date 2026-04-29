# 48991006 Family-Excluded Interior Recovery

This pass masks atoms near existing raw owned bend-line families before running the interior ridge search. It is intended to avoid rediscovering the lower/right `F1-F5` alias family and to target the manually reported unmarked middle bend.

## Results

Family-excluded interior search:

- Output: `48991006_family_excluded_interior_bend_gaps.json`
- Status: `family_excluded_interior_candidates_found`
- Raw-family excluded atoms: `11`
- Selected atoms after exclusion: `23`
- Admissible candidates: `2`
- Merged hypotheses: `1`

Local contact recovery:

- Output: `48991006_family_excluded_interior_flange_contact_recovery.json`
- Status: `fragment_contact_recovered`
- Validated pair: `FEIG2 / F1-F5`
- Contact score: `4.030378`
- Axis delta: `0.176258°`
- Mean normal-arc cost: `0.150998`
- Nearby flange counts: `F1=149`, `F5=39`

Missing-bend family exclusion:

- Output: `48991006_family_excluded_missing_bend_gate.json`
- Status: `separable_missing_bend_candidates_found`
- Passing candidate: `FEIG2`
- Same-pair relation counts for `FEIG2`: `separable=5`

Arrangement check:

- Output: `48991006_junction_bend_arrangement_with_family_excluded_contact.json`
- Recovered candidate: `RCB1`, source `FEIG2`
- Duplicate status: `offset_parallel_same_pair_candidate`
- Unconstrained arrangement status: `overfit_arrangement`
- Best unconstrained hypothesis: `H4_locked_1_plus_3_new`

## Visual Verdict

The 1080p overlays show `FEIG2/RCB1` as an offset parallel support above the raw `F1-F5` lower-edge family. It is no longer the same failure as the previous `IBG1/RCB1` alias path.

However, this is not promotion-ready. Once `RCB1` is admitted, the arrangement solver still chooses too many additional candidates unless constrained, and a one-new-bend constraint chooses a different ridge candidate. That means the local support recovery improved, but the global local-arrangement selection is still unstable on this poor-quality scan.

## Decision

Keep this as a positive diagnostic candidate, not a runtime/count promotion.

This is the best current `48991006` missing-middle recovery signal:

- it survives raw-family masking,
- it validates local flange contact,
- it passes same-pair family exclusion,
- it is visually offset from the raw same-pair line family.

The remaining blocker is arrangement selection, not local support detection.

## Next Work

- Add a local arrangement mode that can prefer recovered-contact candidates when they pass family exclusion, without allowing unrelated ridge candidates to overfit.
- Add a render focused on `FEIG2/RCB1` versus `JBL1` so the competing one-bend choices can be compared visually.
- Keep `48991006` as a poor-scan stress test; do not use this case alone for promotion.

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
- Unconstrained arrangement status: `underfit_arrangement`
- Best unconstrained hypothesis: `H2_locked_1_plus_1_new`
- Best selected candidates: `JBL11`, `RCB1`
- Target-one arrangement status: `single_bend_arrangement_candidate`
- Target-one selected candidates: `JBL11`, `RCB1`
- Suppressed generic raw-family candidates: `JBL1`, `JBL6`, `JBL7`

## Visual Verdict

The 1080p overlays show `FEIG2/RCB1` as an offset parallel support above the raw `F1-F5` lower-edge family. It is no longer the same failure as the previous `IBG1/RCB1` alias path.

The candidate-competition renders showed that `JBL1`, `JBL6`, and `JBL7` ride already-owned raw edge families rather than the visually reported missing middle support. The arrangement solver now suppresses comparable-score generic ridge candidates when they are raw-family covered and a clean recovered-contact candidate exists. With that guard, the target-one arrangement selects `RCB1`.

This is still not promotion-ready. The scan/process quality remains poor, and the unconstrained local arrangement is now intentionally conservative (`underfit_arrangement`) rather than allowed to overfit raw-family edge candidates.

## Decision

Keep this as a positive diagnostic candidate, not a runtime/count promotion.

This is the best current `48991006` missing-middle recovery signal:

- it survives raw-family masking,
- it validates local flange contact,
- it passes same-pair family exclusion,
- it is visually offset from the raw same-pair line family.

The remaining blocker is arrangement selection, not local support detection.

## Next Work

- Add a broader validation pass to make sure raw-family suppression does not hide true direct transitions on cleaner scans.
- Keep the candidate-comparison render path for future disputes (`JBL1` vs `RCB1`, `JBL6` vs `RCB1`).
- Keep `48991006` as a poor-scan stress test; do not use this case alone for promotion.

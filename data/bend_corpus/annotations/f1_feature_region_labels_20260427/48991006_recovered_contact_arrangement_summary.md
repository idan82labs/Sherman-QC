# 48991006 Recovered Contact Arrangement

This pass feeds the validated `IBG1/F1-F5` contact-recovery result into the junction-aware arrangement as `RCB1`.

## Result

- Output: `48991006_junction_bend_arrangement_with_recovered_contact.json`
- Render: `48991006_recovered_contact_visual_check/recovered_contact_birth_ambiguity_1080p.png`
- Recovered candidates: `1`
- Candidate: `RCB1`
- Pair: `F1-F5`
- Source: `IBG1`
- Raw candidate score: `4.191579`
- Admissible before recovered-contact gate: `true`
- Final admissible: `false`
- Duplicate status: `ambiguous_near_existing_owned_region`

Nearest same-pair owned regions:

- `OB2`: atom IoU `0.0`, centroid distance `12.563241 mm`
- `OB1`: atom IoU `0.0`, centroid distance `20.970649 mm`
- `OB3`: atom IoU `0.0`, centroid distance `25.668592 mm`
- `OB6`: atom IoU `0.0`, centroid distance `50.546375 mm`
- `OB7`: atom IoU `0.0`, centroid distance `61.006022 mm`

## Visual Verdict

The recovered red `RCB1` support sits above the orange raw same-pair band in the render and does not overlap atom ownership with the existing raw `F1-F5` regions. This is stronger than the previous broad-corridor failure and is plausibly the manually reported missing middle bend.

However, it is still close to raw `OB2` and uses the same `F1-F5` pair, so it cannot be safely counted until the solver can decide whether `RCB1` and the nearby raw `F1-F5` owned regions are distinct same-pair bend lines or aliases/fragments of one physical bend.

## Decision

Keep `RCB1` as a diagnostic recovered-contact birth candidate. Do not promote it to count logic.

Next implementation step: add same-pair line deconfliction for recovered-contact candidates, using atom overlap, centroid distance, axis/endpoint separation, and visual line-band separation.

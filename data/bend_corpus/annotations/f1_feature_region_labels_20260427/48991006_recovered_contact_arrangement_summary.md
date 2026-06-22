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
- Duplicate status: `same_pair_line_alias`

Nearest same-pair owned regions:

- `OB2`: atom IoU `0.0`, centroid distance `12.563241 mm`, fitted-line distance `1.027793 mm`, candidate-axis span overlap `0.48646`
- `OB6`: atom IoU `0.0`, centroid distance `50.546375 mm`, fitted-line distance `0.401037 mm`, candidate-axis span overlap `0.0`
- `OB7`: atom IoU `0.0`, centroid distance `61.006022 mm`, fitted-line distance `5.325207 mm`, candidate-axis span overlap `0.0`
- `OB3`: atom IoU `0.0`, centroid distance `25.668592 mm`, fitted-line distance `5.705925 mm`, candidate-axis span overlap `0.0`
- `OB1`: atom IoU `0.0`, centroid distance `20.970649 mm`, fitted-line distance `13.859056 mm`, candidate-axis span overlap `1.0`

## Visual Verdict

The recovered red `RCB1` support sits above the orange raw same-pair band in the render and does not overlap atom ownership with the existing raw `F1-F5` regions. This is stronger than the previous broad-corridor failure and is plausibly related to the manually reported missing middle bend.

However, line deconfliction shows `RCB1` is very close to raw `OB2` in model space: the fitted-line distance is only about `1.03 mm`, and the projected span overlaps by about `0.49`. That makes it a same-pair line alias under the current safety rule, not a countable separate bend.

## Decision

Keep `RCB1` as a diagnostic recovered-contact birth candidate. Do not promote it to count logic.

Next implementation step: recover/repair the raw `F1-F5` same-pair family itself. The current evidence suggests `OB2` and `RCB1` may be two fragments or aliases of one same-pair bend-line family, so the solver needs same-pair support consolidation before this case can improve count safely.

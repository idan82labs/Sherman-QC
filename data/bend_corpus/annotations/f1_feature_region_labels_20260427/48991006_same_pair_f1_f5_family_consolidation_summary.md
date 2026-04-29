# 48991006 Same-Pair F1-F5 Family Consolidation

This diagnostic checks whether the recovered-contact birth candidate `RCB1`, or the underlying interior fragments `IBG1`/`IBG2`/`IBH1`, are separate missing middle bends or aliases/fragments of the existing raw `F1-F5` owned-region family.

## Result

- Output: `48991006_same_pair_f1_f5_family_consolidation.json`
- Pair: `F1-F5`
- Status: `recovered_contact_aliases_existing_family`
- Same-pair line count: `9`
- Alias-family count: `3`
- Relation counts: `alias=9`, `ambiguous=2`, `separable=25`
- Interior separate components: `0`

Alias-connected components:

- `IBG1`, `IBG2`, `IBH1`, `OB1`, `OB2`, `OB3`, `RCB1`
- `OB6`
- `OB7`

Key alias relations:

- `OB2` to `RCB1`: atom IoU `0.0`, centroid distance `12.563241 mm`, fitted-line distance `1.027788 mm`, span overlap `0.74496`, axis delta `19.779538°`
- `OB1` to `IBG2`: atom IoU `0.0`, centroid distance `34.13284 mm`, fitted-line distance `0.867458 mm`, span overlap `1.0`, axis delta `21.656764°`
- `IBG1` to `IBG2`: atom IoU `0.0`, centroid distance `21.128709 mm`, fitted-line distance `1.114957 mm`, span overlap `0.698696`, axis delta `22.726125°`
- `OB1` to `OB2`: atom IoU `0.0`, fitted-line distance `0.622107 mm`, span overlap `1.0`
- `OB1` to `OB3`: atom IoU `0.0`, fitted-line distance `1.056589 mm`, span overlap `1.0`

## Decision

Do not count `RCB1`, `IBG1`, `IBG2`, or `IBH1` as new bends. They are geometrically tied to the existing raw `F1-F5` same-pair line family, even when atom overlap is zero.

This narrows the `48991006` blocker: the missed middle bend is not solved by accepting the current `F1-F5` interior fragments as additional countable objects. The next implementation target should search for a different missing-middle line family or improve interior birth generation so it can produce a support region that is separable from the existing `F1-F5` raw family.

## Next Work

- Keep the recovered-contact path offline-only.
- Add a scan-level missing-bend search that avoids same-pair families already covered by raw owned regions unless the new support has clear separability in line distance, span, axis, and support topology.
- Use `48991006` as a negative-control case for same-pair alias suppression: accepting every validated local contact would overcount this poor-quality scan.

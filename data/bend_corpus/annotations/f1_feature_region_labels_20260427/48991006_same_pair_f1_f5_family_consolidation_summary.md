# 48991006 Same-Pair F1-F5 Family Consolidation

This diagnostic checks whether the recovered-contact birth candidate `RCB1` is a separate missing middle bend or an alias/fragment of the existing raw `F1-F5` owned-region family.

## Result

- Output: `48991006_same_pair_f1_f5_family_consolidation.json`
- Pair: `F1-F5`
- Status: `recovered_contact_aliases_existing_family`
- Same-pair line count: `6`
- Alias-family count: `3`
- Relation counts: `alias=3`, `separable=12`

Alias-connected components:

- `OB1`, `OB2`, `OB3`, `RCB1`
- `OB6`
- `OB7`

Key alias relations:

- `OB2` to `RCB1`: atom IoU `0.0`, centroid distance `12.563241 mm`, fitted-line distance `1.027788 mm`, span overlap `0.74496`, axis delta `19.779538°`
- `OB1` to `OB2`: atom IoU `0.0`, fitted-line distance `0.622107 mm`, span overlap `1.0`
- `OB1` to `OB3`: atom IoU `0.0`, fitted-line distance `1.056589 mm`, span overlap `1.0`

## Decision

Do not count `RCB1` as a new bend. It is geometrically tied to the existing raw `F1-F5` same-pair line family, even though it has zero atom overlap with `OB2`.

This narrows the `48991006` blocker: the missed middle bend is not solved by accepting `IBG1/F1-F5` as an additional countable object. The next implementation target should search for a different missing-middle line family or improve interior birth generation so it can produce a support region that is separable from the existing `F1-F5` raw family.

## Next Work

- Keep the recovered-contact path offline-only.
- Add a scan-level missing-bend search that avoids same-pair families already covered by raw owned regions unless the new support has clear separability in line distance, span, and axis.
- Use `48991006` as a negative-control case for same-pair alias suppression: accepting every validated local contact would overcount this poor-quality scan.

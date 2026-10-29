# EXP_CAD_0002 — CAD neutral-surface bend lines

## Decision

Retain the neutral-surface half-turn line construction inside the CAD-first research path. Do not promote it as the global CAD-assisted accuracy incumbent or to autonomous QC.

GPI: **GPI-C, positive development direction**. This is consumed physical evidence, not unseen-family generalization proof.

## Result

V1 reported each paired cylinder's center-of-curvature axis. V2 preserves the exact CAD event inventory but evaluates the line at the mean material radius and half STEP angular turn.

On the matched 11-case diagnostic:

- F1: 0.3538 → **0.3721**;
- precision: 0.3898 → **0.4138**;
- recall: 0.3239 → **0.3380**;
- exact-count accuracy: 0.0000 → **0.3636**;
- exact localized-set accuracy: 0.0000 → **0.1818**;
- minimum-family recall: unchanged at 0.0000.

The older retained CAD diagnostic remains higher at F1 0.4580, so V2 is not the overall accuracy incumbent.

## Visual and operational gates

All 11 completed cases received native-resolution six-view inspection. No gross off-surface regression was found. Two registration cases correctly abstained. The remaining visual errors concern which real formed features count as inspection events, particularly raised slot lips and short side tabs.

The independent replay was byte-identical for all 13 semantic results and templates. Median inference was 0.647 seconds, P95 was 0.847 seconds, maximum RSS was 1,281,720,320 bytes, and all 13 jobs completed without process failure.

## Next attack

The next experiment should use STEP face adjacency and continuous sheet-shell ownership to classify event context. It must not use absolute feature size, part identity, expected counts, or gold. The falsifiable target is removal of isolated embossed-slot event false positives without deleting perimeter bends, return lips, or valid short tab bends.

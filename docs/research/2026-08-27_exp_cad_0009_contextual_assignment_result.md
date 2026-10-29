# EXP_CAD_0009 — finite-span contextual assignment result

## Ruling

`COMPLETE_REJECTED / GPI-C_NEGATIVE_DEVELOPMENT_VISUAL_REJECTED`.

Joint finite-span localization improved several visibly overextended endpoints, but the complete intervention did not improve accuracy. Hard one-to-one CAD/scan assignment suppressed a legitimate close parallel bend and reduced recall.

| Gold11 | Retained CAD | EXP_CAD_0009 | Delta |
|---|---:|---:|---:|
| Precision | 0.5000 | 0.5179 | +0.0179 |
| Recall | 0.4225 | 0.4085 | -0.0141 |
| F1 | 0.4580 | 0.4567 | -0.0013 |
| Exact count | 1/11 | 4/11 | +3 cases |
| Minimum-family recall | 0.2000 | 0.2222 | +0.0222 |
| TP / FP / FN | 30 / 30 / 41 | 29 / 27 / 42 | -1 / -3 / +1 |

The candidate is deterministic across the frozen primary and replay executions, but it fails the F1, recall, effect-size, and visual gates. The retained CAD incumbent remains unchanged.

## Causal finding

The experiment separates two mechanisms that should not be bundled again:

1. **Finite observed-span localization helped.** It visibly repaired gross endpoint overextension on cases including `CG191DEC7F81` and `CG1E0BBA2933`.
2. **Hard one-to-one ownership hurt.** On `CG4FAF273093`, the contextual assignment collapsed two distinct nearby parallel physical bends into one event. That case lost one true positive, explaining the aggregate recall regression.

The next architecture must allow event capacity greater than one when CAD topology contains distinct parallel bend primitives. Localization and event cardinality must be solved separately.

## Native visual adjudication

All 13 exact candidate payloads were inspected at native resolution. Five passed and eight failed. The dominant remaining failures are:

- legitimate close parallel bends are over-suppressed by hard exclusivity;
- conservative null assignment leaves severe undercoverage and one zero-output case;
- local slot/lip and short-return event identity remains ambiguous;
- finite endpoint cropping does not repair missing event discovery.

## Evidence

- Preregistration: `output/research/autoresearch-loop/EXP_CAD_0009_PREREGISTRATION.json`
- Result: `output/research/autoresearch-loop/EXP_CAD_0009_RESULT.json`
- Score: `output/research/autoresearch-loop/exp-cad-0009-contextual-assignment-v1/SCORE.json`
- Candidate-bound visual manifest: `output/research/autoresearch-loop/exp-cad-0009-contextual-assignment-v1/VISUAL_MANIFEST.json`
- Visual adjudication: `output/research/autoresearch-loop/exp-cad-0009-contextual-assignment-v1/VISUAL_ADJUDICATION.json`

## Next bounded experiment

Do not rescue EXP_CAD_0009 by loosening thresholds. The next candidate should:

1. retain source-only finite span localization;
2. group CAD primitives into topology-aware manufacturing events before matching;
3. use capacity-aware assignment so distinct parallel bends can each own evidence;
4. preserve an explicit null/abstention path;
5. bind scoring and all native-resolution visuals to the exact candidate payload.

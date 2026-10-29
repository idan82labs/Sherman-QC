# EXP_CAD_0021 — bend-protected fixed-overlap registration selector

## Outcome

EXP21 implemented and executed a deterministic six-candidate registration selector on the seven authenticated scans with source-native bend events. It is a source-only registration mechanism result, not an F1 result.

Generalization Progress Indicator:

`GPI-C_POSITIVE_SOURCE_ONLY_REGISTRATION_GUARD_NOT_F1_PROGRESS`

The selector uses density-capped fixed point identities, deterministic visible CAD samples frozen under the EXP15 incumbent, bounded proper-rigid ICP deltas, per-event annulus-bin non-regression, curvature-vs-plane qualification, and native-resolution before/after review. No label, expected count, gold line, or evaluator was opened.

## Results

| Case | Numeric selection | Visual ruling | Key result |
|---|---|---|---|
| N003 | baseline | retain baseline | Large forming departure was not registered away. |
| N004 | baseline | retain baseline | Globally attractive alternatives lost bend support or exceeded the motion envelope. |
| N011 | baseline only | fail closed | EXP15 absolute transform has determinant -1; visually plausible reflection is not physical SE(3). |
| N012 | P2PL_T1P5 | pass candidate | 0.570 mm / 0.106°; frozen-visible median 0.3573→0.3131 mm, p90 0.4966→0.4410 mm; 4/4 spans preserved. |
| N014 | baseline | retain baseline | Every global improvement sacrificed at least one curvature-qualified bin among 20 native events. |
| N015 | P2PL_T2P5 | hypothesis only | 0.207 mm / 0.308°; 5/5 spans preserved, but incomplete tessellation prevents replacement of the baseline. |
| N016 | P2PL_T2P5 | hypothesis only | 0.435 mm / 0.110°; 5/5 spans preserved, but incomplete tessellation prevents replacement of the baseline. |

The source-only selection changed three cases numerically. Native review retained one as an actual development refinement, held two as hypotheses, retained three baselines, and quarantined one reflected incumbent.

## Important falsifications

- A scalar global alignment score is unsafe. N014 refinements substantially improved frozen-visible residuals but were rejected because a single bend neighborhood lost curvature-qualified support.
- EXP18-style annulus occupancy is insufficient by itself. EXP21 adds fixed-point curvature-vs-plane diagnostics; annulus-only support remains locality evidence, not bend conformance.
- EXP15's flip pool admitted improper transformations. N011's `det(R)=-1` result is now explicitly fail closed even though the overlay looks good.
- Incomplete CAD tessellation can support a bounded registration hypothesis but cannot support a global promotion claim.

## Reproducibility and visual gate

- Focused tests: 8 passed.
- Independent execution replay: exact candidate rows, selected transforms, selected PLY bytes, and every native render for 7/7 cases.
- Native review: all seven whole-body artifacts plus all changed-case global-delta and event sheets inspected at original resolution.

Primary evidence:

- `output/research/autoresearch-loop/exp-cad-0021-protected-registration-selector-v1/SOURCE_ONLY_SUMMARY.json`
- `output/research/autoresearch-loop/exp-cad-0021-protected-registration-selector-v1/EXP21_MANUAL_NATIVE_VISUAL_REVIEW.json`
- `output/research/autoresearch-loop/exp-cad-0021-protected-registration-selector-v1/EXP21_REPLAY_AUDIT.json`

## Next bounded action

Replace the legacy flip search with all 24 orientation-preserving signed PCA-axis permutations and a proper-rigid candidate pool. N011 is the immediate recovery challenge; N012 is the same-family proper-transform control. Reflections, fitted scale, EXP20 local shifts, labels, and evaluator truth remain excluded.

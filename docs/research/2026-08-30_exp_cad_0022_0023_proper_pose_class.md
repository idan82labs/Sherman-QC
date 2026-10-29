# EXP_CAD_0022–0023 — proper-rigid recovery and CAD symmetry

## Executive result

The reflection-capable EXP15 pose was replaced by a source-only search over all 24 orientation-preserving signed PCA-axis permutations. A proper SO(3) pose class was recovered for N011, but its absolute feature ownership is not identifiable from the current incomplete, symmetric CAD evidence.

EXP22 is terminally rejected:

`GPI-C_NEGATIVE_SOURCE_ONLY_FIXED_THRESHOLD_REJECTED`

Its absolute harmonic-overlap threshold rejected both N011 and the proper N012 control because missing CAD tessellation caps scan-to-CAD coverage. The threshold measured reference completeness, not registration validity.

EXP23 is retained:

`GPI-C_POSITIVE_PROPER_RIGID_POSE_CLASS_RECOVERY_NOT_F1_PROGRESS`

## Source-only measurements

| Case | H2 incumbent | H2 best proper | L5 incumbent | L5 best proper | Ruling |
|---|---:|---:|---:|---:|---|
| N011 | 0.61326 | **0.62170** | 0.32434 | **0.31463** | Proper pose class recovered; reflected incumbent quarantined; absolute ownership abstains. |
| N012 | 0.61152 | **0.62048** | 0.32415 | **0.30992** | Same-family control exposes a symmetry-equivalent pose about 180° from the incumbent. |

Both selected proper candidates support all eight finite bins on all four native bend events, with median radial residual below half the source-scaled tolerance and p90 below the tolerance.

## Native visual ruling

The best N011 proper pose aligns the whole part and all four bend carriers, but moves the asymmetric tab from the legacy side to the symmetry-equivalent side. N012 shows the same phenomenon even though its incumbent is already proper: the alternative is approximately 180° and 45.7 mm away yet has an equally coherent source-only fit.

Therefore:

- reflections are forbidden;
- the recovered SO(3) equivalence class is retained;
- unique left/right or feature ownership is not emitted;
- an asymmetric fixture datum, scanner-pose prior, or complete CAD face coverage is required to break the symmetry;
- no F1, defect, or production claim follows.

## Verification

- EXP21/22/23 focused tests: 15 passed.
- EXP23 primary and replay semantic payloads are exact for 2/2 cases.
- All four EXP22 native legacy/proper and delta images were manually inspected at original resolution and hash-bound into the EXP23 visual review.
- Labels, expected counts, gold, and evaluator data remained closed.

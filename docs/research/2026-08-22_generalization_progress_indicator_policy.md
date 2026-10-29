# Generalization Progress Indicator policy

Date: 2026-08-22

Every new F1 version must publish a Generalization Progress Indicator (GPI) record with its terminal result. The grade measures evidence for physical unseen-family generalization, not engineering effort or architectural novelty.

## Grades

- **A — verified physical progress:** a clean untouched-family physical comparison at 30 scans/15 families or larger beats the frozen baseline in F1, does not regress exact-count accuracy or minimum-family recall, passes every native-resolution visual audit, deterministic replay, resource gate, and all frozen target gates.
- **B — positive physical indicator:** the same directionally clean physical win exists, but the sample is smaller than 30/15 or the full promotion package is incomplete.
- **C — no generalization evidence:** fixture, source-free, synthetic, consumed-development, or otherwise incomparable evidence; or a clean comparison with no strict F1 gain. This can still represent useful engineering progress.
- **D — negative physical indicator:** a valid, leakage-free, matched physical comparison regresses F1, exact-count accuracy, minimum-family recall, or fails native-resolution visual adjudication.
- **F — non-evaluable:** lock/protocol/integrity failure or leakage prevents scientific interpretation. F is not the same as a negative physical result.

## Required fields after every version

1. Protocol validity and leakage status.
2. Number of physical scans and externally defined families.
3. Whether families were untouched and truth was independently annotated.
4. Whether the same frozen baseline ran on the same cases.
5. Deltas for F1, exact-count accuracy, and minimum-family recall.
6. Native-resolution visual ruling.
7. Deterministic replay, resource, and target-gate outcomes.
8. Grade, direction, and machine-readable reasons.

## Current calibration

- V52 synthetic success: **GPI-C** — useful engineering evidence, no physical generalization comparison.
- Aug-15 U01-U04 affine-resampling result: **GPI-D** — valid development-physical robustness evidence showing instability and visual failures, though not an untouched promotion result.
- V73 source-free lock: **GPI-F** — 31/32 fixture cases passed, but the frozen protocol failed before replay or physical execution.

No version currently holds GPI-A or GPI-B.

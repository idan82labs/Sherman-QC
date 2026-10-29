# Calibrated raw-range synthetic optimization cycle: V58 through V61

Date: 2026-08-15

## Outcome

The simulated calibrated raw-range lane produced a real, bounded engineering gain without using CAD, labels, expected counts, filenames, part identity, benchmark feedback, or production/runtime inputs during inference.

The frozen V52 baseline remains the prospective retained system. V61 is the funded next candidate, not yet a generalization claim, because its ownership threshold was selected after synthetic V1 truth was opened.

## Data boundary

- 30 opaque cases across 15 profile families, two captures per family
- eight organized calibrated depth frames per case at 384x288
- fused source point cloud generated from the raw depth frames
- hidden simulator mesh and bend truth isolated under `evaluator_only`
- density/spacing calibrated to the measured physical LiDAR reference range
- token: `SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY`
- evidence class: engineering-only, non-confirmatory

This corpus cannot establish physical zero-shot generalization. It is useful for finding representation and accounting failures before spending new physical scans.

## V52 baseline

| Metric | Value |
|---|---:|
| TP | 41 |
| FP | 25 |
| FN | 59 |
| Precision | 0.621212 |
| Recall | 0.410000 |
| F1 | 0.493976 |
| Exact cases | 3/30 |

## V58 and V59 ruling

V58's turn-token evidence was abundant but its rigid occupancy/seed logic was unsafe. It proposed 59 V52 prunes, deleting 36 true positives while removing only 23 false positives. Native-resolution review showed many orange prune markers directly on obvious folds. V58 was retired with `FAIL_VISUAL_PRUNING_SAFETY`.

V59 disabled pruning and admitted only the single V58 birth. That birth duplicated an already-covered bend, producing no TP gain and one FP. V59 was retired.

The post-score token coverage diagnostic found:

- 89/100 true bends had at least one V58 token;
- 80/100 had sufficient axial span;
- only 5/100 passed the rigid occupancy consensus gates.

This isolated the main failure to segment formation/ownership rather than absence of raw range evidence.

## V60 per-view segment consensus

V60 formed line segments independently in each native view, merged only segments supported by at least three views, preserved V52 byte-for-byte, and added unmatched births without pruning.

- primary/replay identity: 30/30
- raw cross-view lines: 10
- V52 matches suppressed: 7
- unmatched births: 3 (O008, O029, O030)
- execution manifest SHA-256: `873d9e0c0a246243a1f7fc19e1c52ab2d935caecf8bed28e79c8cd1a8ee0e647`

Before scoring, all three cases were manually inspected in three 3D views and all eight native depth frames. O008 and O029 appeared plausible; O030 was pre-registered as a likely duplicate.

After truth opening:

| Metric | V52 | V60 |
|---|---:|---:|
| TP | 41 | 42 |
| FP | 25 | 27 |
| FN | 59 | 58 |
| Precision | 0.621212 | 0.608696 |
| Recall | 0.410000 | 0.420000 |
| F1 | 0.493976 | 0.497041 |

Case-level assignment and scored visual review showed:

- O029 was a genuine new recovery: +1 TP, -1 FN, no new FP.
- O008 duplicated V52 bend `TRV52-e6d99e69be461d0aa7ab`; the V60 birth took the gold assignment and turned the incumbent into an FP.
- O030 duplicated V52 bend `TRV52-8401ac59f931bfc9f113` in the same way.

V60 was not retained as final output because the tiny F1 increase concealed a precision/accounting regression.

## V61 ownership guard

V61 changes no detector, line coordinates, or V52 predictions. It suppresses a V60 birth when a V52 incumbent has angle <=4 degrees, symmetric distance <=0.010 of the source bounding-box diagonal, and overlap >=0.60. The prior V60 distance was 0.008.

This threshold is development-only because it was selected after V1 truth was open. It cleanly separates the observed cases:

- O008 duplicate: distance/B 0.008543, overlap 0.9980 -> suppressed
- O030 duplicate: distance/B 0.008265, overlap 0.9928 -> suppressed
- O029 recovery: distance/B 0.011440 -> admitted

Execution:

- primary/replay identity: 30/30
- births admitted: 1
- births suppressed: 2
- execution manifest SHA-256: `098b8782e25566fa6ef35dfdf0ed0736d0c83bdca6164c49ebbb858d253ab7ab`
- score SHA-256: `96a3ed9a129cfd6797401f660ac0d08843cf802665daa3ad3a5fa9ca22ba8ae1`
- 17 focused tests passed

Development-only score:

| Metric | V52 | V61 | Delta |
|---|---:|---:|---:|
| TP | 41 | 42 | +1 |
| FP | 25 | 25 | 0 |
| FN | 59 | 58 | -1 |
| Precision | 0.621212 | 0.626866 | +0.005654 |
| Recall | 0.410000 | 0.420000 | +0.010000 |
| F1 | 0.493976 | 0.502994 | +0.009018 |

The admitted V61 O029 line is coordinate-identical to the visually audited V60 O029 line and is supported by 25 tokens across three independently fitted native-view segments.

## Decision and next test

- Keep V52 frozen as the last prospective retained baseline.
- Retire V58, V59, and unsuppressed V60.
- Fund V61 as the next candidate because it converts the raw-range mechanism into a clean V1 development gain.
- The next promotion test must be prospective: use the newly received physical part scans or a newly generated geometry/noise corpus whose truth remains closed until V52/V61 outputs, replay hashes, and native-resolution visual rulings are frozen.
- Do not tune V61 again on synthetic V1.

## Visual artifacts

- Truth-free V60 audit: `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/truth_free_visuals`
- Scored changed-case audit: `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/scored_visuals`
- O029 scored recovery: `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/scored_visuals/O029.png`

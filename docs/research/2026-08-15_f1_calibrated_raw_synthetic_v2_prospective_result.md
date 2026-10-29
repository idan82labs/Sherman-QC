# Calibrated raw-range synthetic V2 prospective result

Date: 2026-08-15

## Decision

Promote V61, the source-only birth-ownership guard, as the retained **synthetic research candidate** for the next independent stress corpus. It is not yet a physical-scanner or production claim.

V61 improved the frozen V52 baseline on the independent V2 captures while preserving false-positive count. The result agrees with both the pre-truth visual ruling and the post-score overlay audit.

## Frozen aggregate result

| Route | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| V52 | 34 | 39 | 66 | 0.465753 | 0.340000 | 0.393064 |
| V60 | 37 | 40 | 63 | 0.480519 | 0.370000 | 0.418079 |
| V61 | 37 | 39 | 63 | 0.486842 | 0.370000 | 0.420455 |

Relative to V52, V61 adds three true positives, adds no false positives, removes three false negatives, and raises F1 by 0.027391 absolute.

## Changed-case accounting

| Case | V61 effect versus V52 | Truth-free visual ruling | Post-score result |
|---|---|---|---|
| O007 | +1 TP, -1 FN | distinct observable fold | confirmed TP |
| O019 | +1 TP, -1 FN | distinct observable fold | confirmed TP |
| O023 | +1 TP, -1 FN | distinct observable fold | confirmed TP |
| O024 | no aggregate metric change | duplicate of V52 incumbent | V60 birth matched truth, but V52 already owned the same truth; V61 suppression preserved recall and removed the duplicate prediction |

The changed families were SF04, SF10, and SF12. No family lost a true positive under the intervention.

## Visual audit

Before truth was opened, every V60 birth was inspected in three 3D projections and all eight native 384x288 calibrated frames. The frozen ruling admitted O007/O019/O023 and suppressed O024.

After scoring, truth overlays confirmed:

- O007, O019, and O023: the cyan V61 segment lies on a previously missed magenta truth line and becomes a true positive.
- O024: the V60 cyan segment is visibly coincident with an existing green true-positive prediction. The V61 overlay removes that duplicate while retaining the matched incumbent.

Artifacts:

- truth-free audit: `output/research/f1-calibrated-raw-range-synthetic-v2-20260815/v61-truth-free-audit`
- scored V61 changes: `output/research/f1-calibrated-raw-range-synthetic-v2-20260815/v61-scored-changed-case-audit`
- scored V60 O024 comparison: `output/research/f1-calibrated-raw-range-synthetic-v2-20260815/v60-scored-o024-audit`

## Evidence boundaries

- V2 is independent of the V1 captures but uses the same 15 profile families.
- V61 thresholds were selected on V1 and frozen before V2 execution.
- V2 is synthetic, source-only, and non-confirmatory for real scanner deployment.
- The newly received physical fused PLY scans do not contain the calibrated raw depth frames, intrinsics, and poses required by V60/V61, so they cannot honestly close the physical validation gap.
- The next test is a separately locked V3 geometry stress corpus with short/local creases, smaller bend signatures, compound parts, and rounded facets, visually validated at scanner-like density before inference.

## Integrity hashes

- V2 public manifest: `d9bdccf85b19276a213888ffd4b2dcfaa6d4aab4059e6779eaef36fdc180eeb3`
- V52 manifest: `4f31278c4cae9f4c127dd1e99c6b5b79665c9eab3a176fc9e15d0526d8bcc230`
- V60 manifest: `04c861cefb1a2fb3f6b92764cc61bce6073e1405ef4f5c1786fe508eb1ab45eb`
- V61 manifest: `92831fad30448488293e393ed122f0139881988617741a18887018677123a650`
- V52/V60 score: `c7cdbd13dc47996db966b45237b6c24bab4024caa6c775cf71d07031b4d5a370`
- V52/V61 score: `eafd11c544b69c7e903a9e0a89ed5794dd08294c59360df413e25cf21381431e`

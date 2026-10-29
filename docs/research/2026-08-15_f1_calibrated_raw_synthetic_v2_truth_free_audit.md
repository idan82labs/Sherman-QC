# Calibrated raw-range synthetic V2 truth-free visual audit

Date: 2026-08-15

## Frozen execution

- corpus: `f1-calibrated-raw-range-synthetic-v2-20260815`
- public manifest SHA-256: `d9bdccf85b19276a213888ffd4b2dcfaa6d4aab4059e6779eaef36fdc180eeb3`
- preflight summary SHA-256: `fe722bb003a413b3e2dd680824f62b0a3369be85662d3144723b2e2d912a57a7`
- V51 manifest SHA-256: `f628f2c557a20b945d334792fdf22f9abdd4a06bc15723619bd81b5106244006`
- V52 manifest SHA-256: `4f31278c4cae9f4c127dd1e99c6b5b79665c9eab3a176fc9e15d0526d8bcc230`
- V60 manifest SHA-256: `04c861cefb1a2fb3f6b92764cc61bce6073e1405ef4f5c1786fe508eb1ab45eb`
- V61 manifest SHA-256: `92831fad30448488293e393ed122f0139881988617741a18887018677123a650`
- primary/replay semantic identity: 30/30 at every stage
- evaluator truth opened before this audit: no

## Changed-case audit

All four V60 births were inspected in three 3D source-cloud projections and all eight native 384x288 calibrated depth frames. The audit includes V61 admissions and suppressions.

| Case | V61 action | Final truth-free ruling | Evidence |
|---|---|---|---|
| O007 | admit | plausible distinct fold | Cyan follows a coherent discontinuity in all eight views. Apparent overlap in one 3D projection disappears in the other projections; nearest incumbent separation is 0.114298B. |
| O019 | admit | plausible distinct fold | Cyan consistently tracks the upper long discontinuity and is separated from the nearest incumbent by 0.060407B. |
| O023 | admit | plausible distinct fold | Cyan tracks the central/upper fold across the native views; nearest incumbent separation is 0.089327B. |
| O024 | suppress | correct visual duplicate suppression | Cyan and the V52 incumbent are visually coincident; angle 0.06045 degrees, distance 0.009895B, overlap 1.0. |

## Pre-score ruling

`PASS_TRUTH_FREE_VISUAL_OWNERSHIP`

All three V61 admissions appear to be distinct observable folds, and the only V61 suppression appears to be a duplicate. This ruling was frozen before opening V2 evaluator truth. No detector, ownership threshold, case membership, or capture setting was changed after viewing these artifacts.

## Artifacts

The complete changed-case audit is under:

`output/research/f1-calibrated-raw-range-synthetic-v2-20260815/v61-truth-free-audit`

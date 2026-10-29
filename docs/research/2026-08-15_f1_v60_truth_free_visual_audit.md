# V60 truth-free visual audit

Date: 2026-08-15

## Locked context

- Corpus: `f1-calibrated-raw-range-synthetic-v1-20260815`
- Corpus status: synthetic, non-confirmatory, source-density matched diagnostic only
- Frozen parent: V52
- V60 execution manifest SHA-256: `873d9e0c0a246243a1f7fc19e1c52ab2d935caecf8bed28e79c8cd1a8ee0e647`
- Primary/replay semantic identity: 30/30
- V60 unmatched births: 3
- Synthetic evaluator truth was not consulted for this audit.

## Native-resolution review

Every changed case was inspected in three 3D source-cloud views and all eight native 384x288 calibrated depth frames.

| Case | Truth-free visual ruling | Observation |
|---|---|---|
| O008 | plausible new fold | Cyan line follows a coherent long range discontinuity in all eight views and is visibly separate from retained V52 lines. |
| O029 | plausible new fold | Cyan line follows the lower long fold/discontinuity consistently in all eight views and is visibly separate from retained V52 lines. |
| O030 | likely duplicate | Cyan line lies on the same upper fold as a retained green V52 line. The matcher failed to suppress it because the fitted extents/offset differ enough numerically, despite visual equivalence. |

## Pre-score ruling

`PARTIAL_PASS_WITH_VISIBLE_DUPLICATE`

V60 improves the qualitative selectivity of the raw-range lane relative to V58, but its final V52-match suppression is not yet visually safe. No threshold or code was changed after this observation. The frozen V60 outputs may now be scored to determine whether O008/O029 are genuine recoveries and to quantify the O030 duplicate.

## Artifacts

- `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/truth_free_visuals/O008.png`
- `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/truth_free_visuals/O008_native_views.png`
- `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/truth_free_visuals/O029.png`
- `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/truth_free_visuals/O029_native_views.png`
- `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/truth_free_visuals/O030.png`
- `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v60-per-view-segment-consensus/truth_free_visuals/O030_native_views.png`

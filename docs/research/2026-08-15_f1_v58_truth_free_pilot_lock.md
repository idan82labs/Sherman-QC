# V58 calibrated multiview turn-consensus truth-free pilot lock

Date: 2026-08-15  
Corpus: `output/research/f1-calibrated-raw-range-synthetic-v1-20260815`  
Pilot case: `O001`  
Synthetic truth opened: no  
Family map opened: no

## Purpose

This is an engineering-only test of whether organized calibrated raw-range
observations provide a useful causal channel beyond the fused source cloud.
It is not confirmatory evidence for physical scanner generalization.

## Preserved pilot sequence

1. Attempt 1 used the preregistered global image-lattice stride inside the
   physical normal banks. It emitted zero tokens because a three-pixel jump
   exceeded the `0.006B` connectivity radius.
2. Attempt 2 retained the global stride for profile centres but computed
   physical banks at native organized resolution. It emitted zero tokens;
   native banks existed but failed the fixed concentration/roughness gates.
3. Attempt 3 budgeted actual valid range samples instead of empty image
   background. It still emitted zero tokens under the exact bank gates.
4. The final bounded engineering correction uses native valid samples, a
   12-degree median / 20-degree P90 bank-residual gate, and concentration floor
   0.50. These values were selected from unscored O001 range geometry only.
   It emitted 11 thinned tokens but no three-view consensus line.

The final correction is now frozen before any synthetic truth is opened. No
further threshold rescue is allowed within V58. Full-corpus execution may
either demonstrate cross-view consensus on other geometries or retire the
mechanism.

## Frozen breadth preflight

The unchanged final implementation was then run on O003, O007, O013, and O019
with primary/replay identity. It emitted 29, 44, 40, and 51 thinned tokens,
respectively, but zero accepted raw-view lines in all four cases. It proposed
0, 3, 2, and 2 V52 prunes. Together with O001, V58 therefore has zero raw births
across five unscored cases. The full 30-case execution is funded only to obtain
a terminal mechanism measurement, not to justify further V58 adjustment.

## Baseline compatibility ruling

The retained V52 chain assumes triangle-bearing source PLYs. Scanner-like
`source.ply` files are point clouds. The source-only compatibility adapter does
not reconstruct a hidden or calibrated-view mesh; it preserves V16 incumbents
and records `V43_TRIANGLE_MESH_CAPABILITY_UNAVAILABLE`. This keeps the baseline
from receiving the new organized-view capability.

## Evidence paths

- `v51-mesh-baseline-failed-triangle-assumption-attempt1`
- `v58-cmtcc-pilot-O001-attempt1-empty-strided-banks`
- `v58-cmtcc-pilot-O001-attempt2-native-banks-strided-centres`
- `v58-cmtcc-pilot-O001-attempt3-valid-lattice-exact-gates`
- `v58-cmtcc-pilot-O001`

## No-claim boundary

The corpus is synthetic, the raw-depth sensor model is incomplete, and V58's
final bank tolerances differ from the stricter professor-proposed CMTCC values.
Any score is diagnostic engineering evidence only.

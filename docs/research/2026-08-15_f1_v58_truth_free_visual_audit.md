# V58 truth-free visual audit

Date: 2026-08-15  
Evidence: `output/research/f1-calibrated-raw-range-synthetic-v1-20260815/v58-cmtcc-full/truth_free_visuals`  
Truth opened: no

## Ruling

`FAIL_VISUAL_PRUNING_SAFETY`

All 25 changed primary cases were inspected in seven review sheets. The fused
source cloud was shown in gray, retained V52 lines in green, proposed V52
prunes in dashed orange, and raw V58 births in cyan.

V58 proposed only one raw-view birth, on O024. That cyan line is visually
plausible: it lies continuously on the upper longitudinal fold in oblique,
top, and side views.

The O024 birth was additionally projected into all eight native 384x288 depth
frames. It tracks the same continuous range-image fold especially clearly in
views 1, 2, 5, 6, and 7; it does not float across a silhouette or hole. The
preserved artifact is `truth_free_visuals/O024_raw_birth_native_views.png`.

The pruning behavior is not admissible. In many cases the dashed orange line
lies directly on an obvious continuous longitudinal fold rather than a flat
surface or distributed-curvature artifact. Clear examples include O004,
O005, O006, O007, O008, O011, O012, O014, O016, O017, O020, O021, O022,
O023, O024, O026, and O030. Shorter or more ambiguous proposed removals also
occur in O009, O010, O013, O015, O018, O019, O025, and O029.

This is a terminal visual falsification of V58's negative-evidence pruning
rule. A missing raw token is not reliable evidence that an incumbent line is
not a bend. V58 may still establish that calibrated views can birth at least
one plausible line, but its complete output cannot be retained or promoted.

No threshold or station-count rescue is authorized within V58. Scoring, after
primary/replay freeze, is diagnostic only.

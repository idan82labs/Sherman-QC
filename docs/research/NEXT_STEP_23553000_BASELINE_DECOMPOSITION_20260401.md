# Next Step: 23553000 Baseline Decomposition

## Current kept benchmark state
Reference clean benchmark to keep:
- /Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260401_line_aware_locality_v6/summary.json

Current kept structured metrics:
- structured total MAE: 1.455
- structured completed MAE: 0.190
- structured partial total MAE: 3.133
- structured full exact completion rate: 0.895
- failed scans: 0

This is better than:
- /Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260330_partial_to_full_retention_v1/summary.json

Current best mixed validation state after the kept `probe_region_completion_v1` patch and the skip-existing continuity fix:
- /Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260403_probe_region_completion_v1/projected_summary_from_reused_tail_v2.json

Projected structured metrics relative to the clean 20260402 baseline:
- structured total MAE: `1.394 -> 1.364`
- structured completed MAE: `0.190 -> 0.143`
- structured partial total MAE: `3.000 -> 2.933`
- structured partial completed MAE: `1.000 -> 0.667`
- structured full exact completion rate: `0.895 -> 0.944`

Important interpretation:
- the `38172000` regression was not a real QC regression
- it was a `--skip-existing` evaluator bug that dropped sequence continuity during mixed reruns
- that issue is now fixed and is no longer the next target

## What was tried and rejected
Rejected because not MAE-positive:
1. exact bend-id lookup inside selected-CAD local refinement
- helped target visibility of missing bends
- regressed `23553000_B_SCAN_2.ply` from 7 to 6 completed bends

2. short-bend locality relaxation for non-shallow bends
- improved raw full-scan count on `23553000_B_FULL_SCAN 1.ply` from 13 to 14
- regressed `23553000_B_SCAN_2.ply` from 7 to 6
- would likely worsen aggregate MAE

## Remaining high-value misses
Main remaining MAE-driving family:
- 23553000

Current outcomes in the best mixed validation state:
- `23553000_B_SCAN_1.ply`: structured median 5, expected completed 4
- `23553000_B_SCAN_2.ply`: structured median 9, expected completed 9
- `23553000_B_FULL_SCAN 1.ply`: structured median 15, expected completed 16

The biggest remaining structured completed-bend misses are:
- `23553000_B_FULL_SCAN 1.ply`: short by 1
- `23553000_B_SCAN_1.ply`: over by 1

## Root-cause diagnosis
The evidence now points away from generic local metrology thresholds.

For `23553000_B_FULL_SCAN 1.ply`:
- dense local refinement aligns well
- CAD baseline selected count is 16
- extracted mesh baseline during local refinement collapses to 6 bends
- selected-bend local measurement is therefore being driven through an under-resolved extracted surface decomposition

Observed repeated misses:
- CAD_B4
- CAD_B14
- CAD_B15

These bends are not primarily failing because of scan completeness.
They are failing because the selected CAD baseline is being measured against an extracted surface graph that is too collapsed to support clean local face routing.

Typical failure pattern:
- surface classification: one side has zero local CAD faces
- profile section: often succeeds numerically but lands at implausible angle
- probe region: sometimes finds points but not a trustworthy angle

## Why continuity is not the next win
`23553000_B_SCAN_2.ply` is now on target in the best mixed validation state.
The remaining residual on this family is no longer a carry problem.
It is a CAD-baseline decomposition and stage-semantics problem on:
- `23553000_B_FULL_SCAN 1.ply`
- `23553000_B_SCAN_1.ply`

## Next step to execute
Do not add more broad heuristics.

Implement a target-debug decomposition for `23553000` that records, per CAD bend:
- selected CAD bend id
- extracted-surface ids used for local measurement
- local CAD face counts for each side
- aligned scan point count in bend tube
- local support / side balance
- whether the bend fails at:
  - extracted surface availability
  - profile strip isolation
  - probe-region angle plausibility
  - assignment / exclusivity

Required focus targets:
- likely early over-promotion:
  - `CAD_B8`
- CAD_B4
- CAD_B14
- CAD_B15
- compare them against nearby successful short bends:
  - CAD_B5
  - CAD_B9
  - CAD_B10
  - CAD_B11
  - CAD_B16

## Intended patch after decomposition
Only after the above debug card exists, choose one of:
1. improve selected-CAD to extracted-surface mapping for collapsed legacy baselines
2. bypass extracted surface ids for those bends and derive local side regions directly from selected CAD normals / bend tubes
3. add a narrow per-family override for 23553000 if and only if geometry proves it is a CAD-baseline decomposition defect rather than a detector defect

## Acceptance rules for the next patch
- `23553000_B_SCAN_1.ply` must not worsen beyond structured median `5`
- `23553000_B_SCAN_2.ply` must stay at structured median `9`
- `23553000_B_FULL_SCAN 1.ply` must improve or stay at structured median `15`
- `28870000_A_FULL_SCAN.ply` must remain 2/2
- `28870000_A_SCAN_1.ply` must not gain false discrete bends
- no regression against the best mixed-validation metrics above

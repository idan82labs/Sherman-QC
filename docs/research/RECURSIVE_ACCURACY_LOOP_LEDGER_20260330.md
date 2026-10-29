# Recursive Accuracy Loop Ledger


## iteration_20260330_baseline_v2

- output: `/Users/idant/82Labs/Sherman QC/Full Project/output/family_accuracy_loop/iteration_20260330_baseline_v2`
- baseline structured total MAE: `1.485`
- baseline structured completed MAE: `0.238`
- 28870000 / full_reference: detected `1` of `2`, countable completed `1`
- 28870000 / part1_candidate: detected `1` of `2`, countable completed `1`
- 23553000 / partial_scan_2: detected `6` of `16`, countable completed `6`
- 23553000 / full_scan: detected `12` of `16`, countable completed `12`

## iteration_20260402_baseline_decomposition_v1

- output: `/Users/idant/82Labs/Sherman QC/Full Project/output/family_accuracy_loop/iteration_20260402_baseline_decomposition_v1`
- baseline structured total MAE: `1.485`
- baseline structured completed MAE: `0.238`
- 28870000 / full_reference: detected `2` of `2`, countable completed `2`
- 28870000 / part1_candidate: detected `2` of `2`, countable completed `2`
- 23553000 / partial_scan_2: detected `6` of `16`, countable completed `6`
- 23553000 / full_scan: detected `13` of `16`, countable completed `13`

## iteration_20260402_stage_semantics_v1

- output: `/Users/idant/82Labs/Sherman QC/Full Project/output/family_accuracy_loop/iteration_20260402_stage_semantics_v1`
- baseline structured total MAE: `1.485`
- baseline structured completed MAE: `0.238`
- 28870000 / full_reference: detected `2` of `2`, countable completed `2`
- 28870000 / part1_candidate: detected `2` of `2`, countable completed `2`
- 23553000 / partial_scan_1: detected `6` of `16`, countable completed `6`
- 23553000 / partial_scan_2: detected `7` of `16`, countable completed `7`
- 23553000 / full_scan: detected `13` of `16`, countable completed `13`

## iteration_20260402_stage_semantics_23553000_only

- output: `/Users/idant/82Labs/Sherman QC/Full Project/output/family_accuracy_loop/iteration_20260402_stage_semantics_23553000_only`
- baseline structured total MAE: `1.485`
- baseline structured completed MAE: `0.238`
- 23553000 / partial_scan_1: detected `6` of `16`, countable completed `6`
- 23553000 / partial_scan_2: detected `7` of `16`, countable completed `7`
- 23553000 / full_scan: detected `13` of `16`, countable completed `13`

## iteration_20260403_probe_completion_and_skip_existing_fix

- mixed validation dir: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260403_probe_region_completion_v1`
- corrected projected summary: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260403_probe_region_completion_v1/projected_summary_from_reused_tail_v2.json`
- projected structured total MAE: `1.364`
- projected structured completed MAE: `0.143`
- projected structured partial total MAE: `2.933`
- projected structured partial completed MAE: `0.667`
- projected structured full exact completion rate: `0.944`
- key retained gain:
  - `23553000_B_SCAN_2.ply`: structured completed `8 -> 9`
- evaluator consistency fix:
  - `38172000_01_SCAN_2 1.ply`: `CAD_B2` restored via `SEQUENCE_CONTINUITY` during `--skip-existing` reruns
- 28870000 guard:
  - no rolled/process feature promoted into a discrete bend during this tranche
- next target:
  - `23553000` residual CAD-baseline decomposition on `CAD_B4`, `CAD_B14`, `CAD_B15`

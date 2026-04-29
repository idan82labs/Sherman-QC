# F1 Merge Readiness Status

Generated: `2026-04-29T17:07:36`

Runtime contract: `No runtime zero-shot or product API mutation is approved by this report.`

## Decision

- Main-branch runtime promotion: `not_ready`.
- Offline PR merge readiness: `safe_as_diagnostic_artifacts`.

Blockers:
- `manual_validation_has_truth_outside_range_cases`
- `raw_crease_lane_has_hold_or_abstain_cases`
- `runtime_path_unchanged_and_f1_still_offline_only`

Allowed next steps:
- `sparse_arrangement_candidate_exists_for_offline_review`
- `expand_sparse_arrangement_validation_on_clean_known_cases`
- `keep_scan_quality_gate_for_partial_or_poor_coverage_scans`
- `do_not_replace_existing_exact_f1_results_when_raw_crease_lane_abstains`

## Manual Validation Metrics

- Scored cases: `15` from `16` records.
- Range hit rate: `0.6` (`9/15`).
- Exact hit rate where an exact/prediction exists: `0.1667` (`1/6`).
- Average accepted range width: `3.5333`.

Range misses:
- `49001000` / `49001000_A-PRELIM.ply`: truth `5`, range `[7, 22]`, status `truth_outside_range`.
- `10839000` / `10839000_01_scan_1.ply`: truth `4`, range `[2, 2]`, status `truth_outside_range`.
- `15651004` / `15651004_01_3RD_SCAN.stl`: truth `2`, range `[3, 3]`, status `truth_outside_range`.
- `38172000` / `38172000_01_SCAN_1.ply`: truth `6`, range `[4, 4]`, status `truth_outside_range`.
- `40500000` / `40500000_0_SCAN2 1.ply`: truth `7`, range `[1, 1]`, status `truth_outside_range`.
- `10839000` / `10839000_01_full_scan.ply`: truth `8`, range `[9, 10]`, status `truth_outside_range`.

## Raw Crease Review Gates

- `candidate_for_sparse_arrangement_review`: 1
- `hold_exact_count_scan_quality_limited`: 2
- `preserve_existing_f1_exact_raw_lane_abstains`: 1

Sparse-arrangement review candidates:
- `37361005`: selected `4` / target `4`, safe candidates `6`.

Held or abstained cases:
- `47959001`: `hold_exact_count_scan_quality_limited` (`limited_observed_support`, `sparse_arrangement_underfit`).
- `49001000`: `hold_exact_count_scan_quality_limited` (`limited_observed_support`, `sparse_arrangement_underfit`).
- `48963008`: `preserve_existing_f1_exact_raw_lane_abstains` (`insufficient_support_abstain`, `no_safe_support`).

## Practical Merge Path

- Merge the current branch only as an offline research/diagnostic artifact lane.
- Do not route production counts through sparse raw creases yet.
- Next promotion candidate is not all F1; it is the narrow sparse-arrangement gate on clean known scans like `37361005` after broader validation.
- Poor/partial scans such as `47959001` and `49001000` should stay behind scan-quality/observed-support gates.

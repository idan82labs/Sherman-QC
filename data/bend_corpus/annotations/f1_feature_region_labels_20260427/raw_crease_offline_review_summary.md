# Raw Crease Offline Review Summary

Generated: `2026-04-29T17:17:16`

This is an offline-only diagnostic gate. It summarizes raw-crease support quality and sparse-arrangement evidence; it does not promote any result into runtime.

## Case Table

| Part | Target | Track | Quality | Sparse | Selected | Safe | Duplicates | Recommendation |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 47959001 | 5 | partial | limited_observed_support | sparse_arrangement_underfit | 1 | 2 | 1 | hold_exact_count_scan_quality_limited |
| 49001000 | 5 | partial | limited_observed_support | sparse_arrangement_underfit | 1 | 2 | 1 | hold_exact_count_scan_quality_limited |
| 37361005 | 4 | known | arrangement_needed | sparse_arrangement_exact_candidate | 4 | 6 | 2 | candidate_for_sparse_arrangement_review |
| 48963008 | 3 | known | insufficient_support_abstain | no_safe_support | 0 | 0 | 0 | preserve_existing_f1_exact_raw_lane_abstains |
| 49125000 | 2 | known | promotion_needs_review | sparse_arrangement_underfit | 1 | 2 | 1 | hold_underfit_support_creation_needed |
| 44211000 | 1 | known | insufficient_support_abstain | no_safe_support | 0 | 0 | 0 | preserve_existing_f1_exact_raw_lane_abstains |
| 15651004 | 1 | known | insufficient_support_abstain | no_safe_support | 0 | 0 | 0 | preserve_existing_f1_exact_raw_lane_abstains |

## Recommendation Counts

- `candidate_for_sparse_arrangement_review`: 1
- `hold_exact_count_scan_quality_limited`: 2
- `hold_underfit_support_creation_needed`: 1
- `preserve_existing_f1_exact_raw_lane_abstains`: 3

## Case Notes

### 47959001

- Target: `5`; track: `partial`; existing F1 exact: `None`.
- Quality gate: `limited_observed_support` / `scan_quality_gate`; reasons: bridge_safe_below_human_target, partial_or_observed_scan_support_incomplete, low_bridge_safe_fraction.
- Sparse arrangement: `sparse_arrangement_underfit`; selected `1` from `2` bridge-safe candidates; suppressed duplicates `1`.
- Final recommendation: `hold_exact_count_scan_quality_limited`; reasons: scan_support_below_target.

### 49001000

- Target: `5`; track: `partial`; existing F1 exact: `None`.
- Quality gate: `limited_observed_support` / `scan_quality_gate`; reasons: bridge_safe_below_human_target, partial_or_observed_scan_support_incomplete, low_bridge_safe_fraction.
- Sparse arrangement: `sparse_arrangement_underfit`; selected `1` from `2` bridge-safe candidates; suppressed duplicates `1`.
- Final recommendation: `hold_exact_count_scan_quality_limited`; reasons: scan_support_below_target.

### 37361005

- Target: `4`; track: `known`; existing F1 exact: `None`.
- Quality gate: `arrangement_needed` / `sparse_arrangement_gate`; reasons: bridge_safe_exceeds_human_target, duplicate_flange_pair_support.
- Sparse arrangement: `sparse_arrangement_exact_candidate`; selected `4` from `6` bridge-safe candidates; suppressed duplicates `2`.
- Final recommendation: `candidate_for_sparse_arrangement_review`; reasons: sparse_arrangement_count_matches_target.

### 48963008

- Target: `3`; track: `known`; existing F1 exact: `3`.
- Quality gate: `insufficient_support_abstain` / `observe_only`; reasons: no_bridge_safe_support.
- Sparse arrangement: `no_safe_support`; selected `0` from `0` bridge-safe candidates; suppressed duplicates `0`.
- Final recommendation: `preserve_existing_f1_exact_raw_lane_abstains`; reasons: existing_f1_exact_matches_target, raw_crease_lane_abstains.

### 49125000

- Target: `2`; track: `known`; existing F1 exact: `None`.
- Quality gate: `promotion_needs_review` / `manual_review`; reasons: bridge_safe_matches_human_target, low_bridge_safe_fraction.
- Sparse arrangement: `sparse_arrangement_underfit`; selected `1` from `2` bridge-safe candidates; suppressed duplicates `1`.
- Final recommendation: `hold_underfit_support_creation_needed`; reasons: sparse_arrangement_below_target.

### 44211000

- Target: `1`; track: `known`; existing F1 exact: `1`.
- Quality gate: `insufficient_support_abstain` / `observe_only`; reasons: no_bridge_safe_support.
- Sparse arrangement: `no_safe_support`; selected `0` from `0` bridge-safe candidates; suppressed duplicates `0`.
- Final recommendation: `preserve_existing_f1_exact_raw_lane_abstains`; reasons: existing_f1_exact_matches_target, raw_crease_lane_abstains.

### 15651004

- Target: `1`; track: `known`; existing F1 exact: `1`.
- Quality gate: `insufficient_support_abstain` / `observe_only`; reasons: no_bridge_safe_support.
- Sparse arrangement: `no_safe_support`; selected `0` from `0` bridge-safe candidates; suppressed duplicates `0`.
- Final recommendation: `preserve_existing_f1_exact_raw_lane_abstains`; reasons: existing_f1_exact_matches_target, raw_crease_lane_abstains.

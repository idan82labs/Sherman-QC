# F1 Next Solver Recovery Plan

Generated: `2026-04-29T17:33:49`

Primary next step: `T2_hard_known_latent_support_creation`

Promotion rule: No runtime promotion until a tranche wins on visual support coverage and count accuracy without breaking preserve-existing and scan-quality gates.

## Tranches

### T1_clean_sparse_arrangement_candidate

- Purpose: Validate the only current sparse-arrangement exact candidate before considering any runtime wiring.
- Cases: `37361005`
- Exit rule: Requires more clean-known cases with visually correct owned bend-line support; current evidence is insufficient for runtime.

### T2_hard_known_latent_support_creation

- Purpose: Recover hard repeated/raised-feature known scans where raw sparse creases under-cover the target.
- Cases: `10839000`, `ysherman`
- Exit rule: Do not promote raw sparse lane; prototype typed support creation / latent decomposition and require visual support coverage.

### T3_scan_quality_observed_support_gate

- Purpose: Keep poor/partial scans behind observed-support gates rather than forcing exact full-count promotion.
- Cases: `47959001`, `49001000`
- Exit rule: Report observed support limits and escalate manual validation when scan coverage is insufficient.

### T4_rolled_mixed_transition_semantics

- Purpose: Separate rounded/rolled mixed transitions from conventional bend-line counts.
- Cases: `49125000`
- Exit rule: Requires an explicit rolled/mixed-transition class before any count promotion.

### T5_preserve_existing_f1_exact

- Purpose: Use raw-crease abstain cases as negative controls and preserve existing exact F1/control results.
- Cases: `48963008`, `44211000`, `15651004`
- Exit rule: Raw lane should not replace exact accepted results when it abstains.

## Case Routing

| Part | Target | Selected | Safe | Gap | Category | Next Experiment |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 47959001 | 5 | 1 | 2 | 4 | `scan_quality_or_observed_support_limited` | `scan_quality_observed_support_gate` |
| 49001000 | 5 | 1 | 2 | 4 | `scan_quality_or_observed_support_limited` | `scan_quality_observed_support_gate` |
| 37361005 | 4 | 4 | 6 | 0 | `sparse_arrangement_review_candidate` | `visual_review_then_expand_clean_known_validation` |
| 48963008 | 3 | 0 | 0 | 3 | `preserve_existing_f1_raw_lane_abstains` | `preserve_existing_result_and_use_raw_lane_as_negative_control` |
| 49125000 | 2 | 1 | 2 | 1 | `rolled_mixed_transition_semantics` | `rolled_mixed_transition_router_and_metadata_class` |
| 44211000 | 1 | 0 | 0 | 1 | `preserve_existing_f1_raw_lane_abstains` | `preserve_existing_result_and_use_raw_lane_as_negative_control` |
| 15651004 | 1 | 0 | 0 | 1 | `preserve_existing_f1_raw_lane_abstains` | `preserve_existing_result_and_use_raw_lane_as_negative_control` |
| 10839000 | 8 | 4 | 4 | 4 | `hard_known_latent_support_creation` | `typed_latent_support_creation_or_junction_decomposition` |
| ysherman | 12 | 2 | 2 | 10 | `hard_known_latent_support_creation` | `typed_latent_support_creation_or_junction_decomposition` |

## Engineering Conclusion

- Stop expanding raw-crease sparse promotion for hard cases; it under-covers `10839000` and `ysherman`.
- The next solver implementation should target typed latent support creation / decomposition for hard known cases.
- Keep `37361005` as the only sparse-arrangement review candidate and use the other cases as gates/blockers.

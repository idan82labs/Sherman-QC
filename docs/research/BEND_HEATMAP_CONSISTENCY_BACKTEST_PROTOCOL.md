# Bend Heatmap Consistency Backtest Protocol

## Purpose
Validate whether the new advisory validator is useful enough to keep and later promote.

## Required Slices
- full vs partial scans
- observed vs partially observed/unobserved bends
- measurement method
  - `signed_distance_gradient`
  - `surface_classification`
  - `profile_section`
  - `probe_region`
- part family
  - standard folded
  - rolled/hybrid

## Primary Metrics
- contradiction precision on bends believed out of spec
- false-contradiction rate on bends believed in spec
- abstention rate
- supported-rate on observed folded bends
- full-scan summary agreement rate

## Guardrails
- report `NON_INDEPENDENT` separately
- exclude rolled/process features from v1 per-bend scorecards
- do not use validator output to relabel bend truth in the same backtest

## Minimum Acceptance Targets For v1
- full scans, observed folded bends:
  - false-contradiction rate <= 10%
  - contradiction precision >= 75%
- partial scans, observed folded bends:
  - false-contradiction rate <= 20%
  - contradiction precision >= 65%
- >= 70% of observed folded bends on full scans should end as `SUPPORTED` or `CONTRADICTED`

## First Backtest Result
- Run:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/heatmap_consistency_backtest_20260326_v2`
- Outcome:
  - the validator produced non-trivial results on all `34` scans
  - `SUPPORTED`: `74`
  - `CONTRADICTED`: `89`
  - `INSUFFICIENT_EVIDENCE`: `38`
  - supported rate: `0.368`
  - contradicted rate: `0.443`
- Decision:
  - v1 fails the intended contradiction posture and remains advisory only
  - the next backtest must include stricter slices by:
    - `measurement_method`
    - scan state
    - observability state
  - threshold tuning must target contradiction reduction first, not support-rate maximization

## Second Backtest Result
- Run:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/heatmap_consistency_backtest_20260326_v3`
- Outcome:
  - contradicted rate dropped from `0.443` to `0.088`
  - `NOT_DETECTED` contradicted dropped to `0`
  - full observed `PASS` contradiction rate was `0.087`
  - supported rate dropped from `0.368` to `0.259`
  - full observed folded bend evaluable rate was `0.581`
- Acceptance check:
  - passed:
    - `heatmap_not_detected_contradicted = 0`
    - `PASS` contradiction rate <= `20%`
    - full-scan `PASS` contradiction rate <= `15%`
    - contradicted rate dropped materially from `0.443`
  - failed:
    - supported rate stayed below `0.30`
    - full observed folded bend evaluable rate stayed below `0.60`
- Decision:
  - the contradiction fix is successful
  - next tuning target is support-rate recovery, not more contradiction suppression

## Third Backtest Result
- Run:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/heatmap_consistency_backtest_20260327_v4`
- Change under test:
  - relaxed only the `FAIL` / `WARNING` support gate while preserving all v2 contradiction restrictions
- Outcome:
  - supported bends increased from `50` to `57`
  - supported rate increased from `0.259` to `0.285`
  - contradicted rate stayed effectively flat: `0.088 -> 0.090`
  - active-fail (`FAIL` / `WARNING`) support increased on the intended slice
  - `NOT_DETECTED` contradicted remained `0`
- Acceptance check:
  - passed:
    - `heatmap_not_detected_contradicted = 0`
    - contradiction stayed low
    - support recovered in the intended `FAIL` / `WARNING` slice
  - failed:
    - supported rate stayed below `0.30`
    - evaluator is still too abstention-heavy to treat as a strong corroboration layer
- Decision:
  - keep the current support change
  - keep the feature advisory-only
  - next iteration should target observed full-scan folded bends specifically, without loosening contradiction gates

## Fourth Backtest Result
- Run:
  - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/heatmap_consistency_backtest_20260327_v5`
- Change under test:
  - added two narrow full-scan-only support paths:
    - quiet `PARTIAL`-independence `PASS`
    - moderate-signal active-fail bends with strong coverage
- Outcome:
  - supported bends increased from `57` to `61`
  - supported rate increased from `0.285` to `0.307`
  - contradicted rate stayed flat at `0.090`
  - full-scan supported rate increased from `0.329` to `0.369`
  - full observed folded bend evaluable rate increased to `0.800`
  - `NOT_DETECTED` contradicted remained `0`
- Acceptance check:
  - passed:
    - supported rate >= `0.30`
    - contradiction stayed low
    - `NOT_DETECTED` contradiction stayed at `0`
    - full observed folded bend evaluable rate is now well above the prior floor
  - remaining caution:
    - two heavy corpus cases still timed out
- Decision:
  - accept v5 as the new advisory baseline
  - keep contradiction gates unchanged
  - keep the validator advisory-only pending family-specific tuning and better runtime stability

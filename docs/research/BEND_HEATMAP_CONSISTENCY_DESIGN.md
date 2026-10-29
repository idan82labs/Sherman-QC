# Bend Heatmap Consistency Design

## Goal
Add an advisory validator that checks whether each discrete bend verdict is supported by the aligned-scan deviation field used for the whole-part heatmap.

## Scope
- Advisory only in v1
- Countable discrete bends only
- Rolled/process features return `NOT_APPLICABLE`
- Whole-part summary only for full scans

## Data Flow
1. Bend worker aligns scan to CAD with `ScanQCEngine`
2. Bend worker persists `aligned_scan.ply`, `deviations.npy`, `deviation_stats.json`
3. Bend worker runs `domains/bend/services/heatmap_consistency_service.py`
4. Per-bend `heatmap_consistency` and report-level `heatmap_consistency_summary` are merged into the bend report JSON
5. Bend Inspection UI renders the results as advisory chips and summary strip
6. Corpus evaluator aggregates the validator outputs for later backtesting

## Current Interfaces
- Per bend:
  - `status`
  - `consistency_score`
  - `independence_class`
  - `roi_point_count`
  - `roi_coverage_ratio`
  - `local_abs_mean_mm`
  - `local_abs_p95_mm`
  - `local_out_of_tol_ratio`
  - `side_asymmetry_score`
  - `signal_extent_along_flange`
  - `notes[]`
- Summary:
  - `status`
  - `inconsistent_bend_count`
  - `insufficient_evidence_bend_count`
  - `evaluable_bend_count`
  - `bend_roi_energy_share`
  - `global_in_tolerance_rate`
  - `notes[]`

## Decision Rules
- Weak ROI coverage -> `INSUFFICIENT_EVIDENCE`
- Quiet local heatmap supporting `PASS` -> `SUPPORTED`
- Hot local heatmap contradicting `PASS` -> `CONTRADICTED`
- Hot local heatmap supporting `FAIL`/`WARNING` -> `SUPPORTED`
- Quiet local heatmap contradicting `FAIL`/`WARNING` -> `CONTRADICTED`
- `NOT_DETECTED` usually -> `INSUFFICIENT_EVIDENCE`

## Independence Classes
- `INDEPENDENT`: validator uses different evidence than the bend metrology path
- `PARTIAL`: validator and metrology share some geometry assumptions but not the same exact aggregation
- `NON_INDEPENDENT`: measurement already depends directly on signed-distance evidence, so validator is corroboration only

## Non-Goals
- No override of bend pass/fail
- No override of structured count
- No rolled-feature validator in v1

## Current Risk After Backtest v1
- The current ROI/policy combination is too broad and too aggressive.
- First corpus results show a contradiction-heavy regime:
  - supported rate: `0.368`
  - contradicted rate: `0.443`
- The main observed failure mode is:
  - local ROI covers enough globally hot residual structure that otherwise plausible `PASS` and `NOT_DETECTED` bends are marked `CONTRADICTED`
- Practical implication:
  - keep the validator advisory
  - treat `CONTRADICTED` as a review hint, not a decision signal
  - next iteration should shrink or reshape bend-local ROIs and tighten contradiction thresholds before any promotion

## Current Risk After Backtest v2
- The main contradiction problem is now substantially reduced.
- The new main risk is over-abstention:
  - supported rate dropped to `0.259`
  - insufficient-evidence bends rose to `126`
- Practical implication:
  - keep the v2 contradiction policy
  - next iteration should recover support on clearly observed full-scan folded bends
  - do not reopen `NOT_DETECTED` contradiction while doing that

## Current Risk After Support-Recovery Pass
- The validator recovered some useful support without materially reopening contradiction:
  - supported rate increased to `0.285`
  - contradicted rate stayed effectively flat at `0.090`
  - `NOT_DETECTED` contradicted remained `0`
- The current main risk is still over-abstention:
  - `INSUFFICIENT_EVIDENCE` remains the dominant outcome
  - support is still below the intended `0.30` floor
- Practical implication:
  - keep the current contradiction posture
  - keep the feature advisory-only
  - next iteration should recover support by targeting clearly observed full-scan folded bends, not by weakening contradiction gates

## Current Risk After Support-Recovery Pass v2
- The validator now clears the minimum support-rate floor while keeping contradiction flat:
  - supported rate increased to `0.307`
  - contradicted rate remained `0.090`
  - `NOT_DETECTED` contradicted remained `0`
  - full observed folded bend evaluable rate increased to `0.800`
- The main remaining risks are no longer validator-policy safety:
  - corpus runtime instability on a small set of heavy scans
  - family-specific behavior, especially rolled/hybrid parts
- Practical implication:
  - keep the feature advisory-only
  - treat this as the first acceptable default advisory baseline
  - future work should be family-aware tuning and runtime robustness, not broader contradiction loosening

# Mathematical Specification — Labels and Benchmarks

Date: 2026-03-29
Status: draft
Priority: high
Dependencies:
- `MATH_SPEC_FEATURE_CLASSIFICATION.md`
- `MATH_SPEC_OBSERVABILITY_MODEL.md`
- `MATH_SPEC_EXPECTED_ANGLE_TRUTH.md`
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define the minimum label set and benchmark semantics required to evaluate the redesigned pipeline without collapsing target ontology, observability, and decision policy into a single count metric.

## Core principle
Count MAE is necessary but insufficient.

A scan can have:
- correct count and wrong target set
- good global overlay and wrong local frame
- wrong binary call because the target was actually unobservable

The benchmark must therefore separate:
1. feature ontology truth
2. measurement truth
3. observability truth
4. decision truth

## Target schema
Each target in the corpus should eventually carry:
- `target_id`
- `feature_type`
  - `COUNTABLE_DISCRETE_BEND`
  - `ROLLED_PROCESS_FEATURE`
  - `NON_BEND_TRANSITION`
- `parent_island_id`
- `measurement_primitive`
  - `ANGLE_RADIUS`
  - `ROLLED_PROFILE`
  - `PROFILE`
  - `GAP`
  - `OFFSET`
- `nominal_parameters`
- `tolerance_parameters`
- `observability_truth`
  - `FORMED`
  - `UNFORMED`
  - `UNKNOWN`
- `decision_confidence_truth` if available

## Minimum new label
Highest-value immediate label:

`per_bend_state_with_unknown`

For every scan and every CAD target:

`{ state: FORMED | UNFORMED | UNKNOWN }`

Optional fields:
- `visible_length_fraction`
- `reviewer_confidence`
- `notes`

Reason:
- teaches abstention explicitly
- upgrades evaluation from count-only to target-set correctness
- attacks the partial-scan error concentration directly

## Benchmark layers
### Layer 1 — Count metrics
Keep:
- heuristic MAE vs expected total completed bends
- structured MAE vs expected total completed bends
- heuristic MAE vs expected completed bends
- structured MAE vs expected completed bends

These remain useful for continuity with the current benchmark history.

### Layer 2 — Exact target-set metrics
For scans with target-level labels:
- exact completed-target-set accuracy on observable targets
- target-level precision / recall on `FORMED`
- target-level false positive rate on `UNFORMED`
- target-level abstention rate on `UNKNOWN`

### Layer 3 — Registration metrics
Per scan:
- winning anchor source
- normalized residual
- hold-out residual
- transform covariance proxy
- alignment abstention rate

### Layer 4 — Primitive-specific metrology
Discrete bends:
- angle MAE on observable targets
- radius MAE on observable targets
- confidence-interval coverage

Rolled features:
- centerline placement error
- section-profile error
- rolled pass/fail accuracy

## Primary industrial metric
Primary deployment metric:

`OSR_α = max_{τ : FAR(τ) <= α} OSR(τ)`

where:
- `OSR`: operator-review save rate
- `FAR`: false-accept rate
- `α`: fixed safety ceiling

Definitions:

`a_i(τ) = 1[ r̂_i <= τ and no decisive target is UNKNOWN ]`

`OSR(τ) = (1 / N) Σ_i a_i(τ) c_i`

`FAR(τ) = (Σ_i 1[a_i(τ)=1, ŷ_i=ACCEPT, y_i=REJECT]) / (Σ_i 1[y_i=REJECT])`

This is the metric that best matches industrial usefulness once calibration exists.

## Benchmark slices
Mandatory slices:
- full vs partial
- standard folded vs rolled / hybrid
- strong datum vs weak datum
- observable vs unobservable targets
- high-confidence vs abstained alignment

Priority proving families for the next redesign tranche:
- `23553000`
- `10839000`
- `38172000`
- `40500000`
- `11979003`
- `28870000` as hybrid semantic guard

## Benchmark rules
### Unknown handling
`UNKNOWN` must not be scored as a false positive or false negative for target-formation state.

Instead, track:
- abstention rate
- abstention on full scans
- abstention on partial scans

### Ambiguous nominal handling
If released nominal sources disagree materially and no master exists:
- exclude the target from pass/fail angle scoring
- count it in a separate `spec_ambiguity` bucket

### Rolled-feature handling
Rolled features:
- do not contribute to discrete bend-count MAE
- do contribute to rolled-feature validation metrics

## End-of-wave MAE recomputation
Always recompute against the stable benchmark baseline:

`/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`

For continuity, retain:
- `scans_with_expected_total = 33`
- `scans_with_expected_completed = 21`

Current baseline:
- structured total absolute error: `1.727 * 33 ≈ 57`
- structured partial absolute error: `3.6 * 15 = 54`

This implies nearly all current structured error mass is in partial scans.

## Proof criteria for the redesign
Minimum acceptable first tranche:
- structured partial total error reduced by at least `6`
- structured total error reduced by at least `6`
- no regression in structured full exact completion rate
- no increase in benchmark failures/timeouts

Main target after the first redesign tranche:
- structured partial total error reduced by `12`
- structured total error reduced by `12`
- structured completed-bends total error reduced from about `7` to about `5`

## Non-goals
This spec does not define:
- the registration objective itself
- the local bend estimator itself
- the rolled-section validator internals

It defines how those components must be labeled and judged.

## Short conclusion
The benchmark must stop collapsing ontology, observability, and decision quality into one count number. Count MAE stays, but target-set correctness and abstention-aware evaluation must become first-class.

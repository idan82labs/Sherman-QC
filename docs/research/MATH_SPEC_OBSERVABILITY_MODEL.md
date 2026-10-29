# Mathematical Specification — Observability Model

Date: 2026-03-29
Status: draft
Priority: high
Dependencies:
- `MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
- `MATH_SPEC_LOCAL_BEND_METROLOGY.md`
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define when a bend target on a scan is:
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

This model is primarily for partial scans, but the same formal rule applies to full scans.

## Core principle
Missing evidence is not negative evidence.

`UNKNOWN` is not a fallback error state. It is the correct state whenever the scan cannot support a confident geometric decision in the trusted parent-island frame.

## Scope
This specification applies to countable discrete bend targets.

It does not apply to:
- rolled/process features
- non-bend transitions
- families whose primary measurement primitive is not angle / localized reorientation

## State semantics
### FORMED
The bend operation has happened and the target exhibits non-flat discrete reorientation.

This does **not** necessarily mean the final angle is already within tolerance.
Final QC remains a separate check.

### UNFORMED
The expected bend zone is sufficiently visible and the geometry supports the claim that the bend has not yet formed.

This is the hardest claim and should require stronger evidence than `FORMED`.

### UNKNOWN
The scan does not provide enough evidence to call either `FORMED` or `UNFORMED` safely.

## Evidence variables
All variables are evaluated in the trusted parent-island frame.

### Frame confidence
`c_F = exp(- (σ_rot / 0.5°)^2 - (σ_pos / 0.25 mm)^2 )`

### Two-side visibility
`v_2 = min(1, K_2side / K_min)`

where `K_2side` is the number of valid slices with both adjacent sides visible.

### Curvature-band visibility
`v_C = min(1, K_curv / K_min)`

where `K_curv` is the number of slices with visible localized curvature band.

### Slice consistency
`c_S = exp(- (MAD_β / 1.5°)^2 )`

### Contextual localization confidence
Let `d_perp` be the perpendicular distance from observed curvature-band centroid to the expected nominal bend line in the parent frame.

`c_L = exp(- (d_perp / τ_d)^2 )`

with:
- `τ_d = max(2h, 0.2 t_m)`

## Observability score
Use a hybrid score plus hard gates.

`O = 0.30 c_F + 0.25 v_2 + 0.20 v_C + 0.15 c_S + 0.10 c_L`

The score is necessary but not sufficient. State mapping must still satisfy hard evidence conditions.

## State decision rules
### FORMED
Declare `FORMED` only if all hold:
1. `O >= 0.70`
2. `K_eff >= 5`
3. bend model is preferred in at least `70%` of valid slices
4. lower 95% bound on reorientation exceeds flat threshold:
   - `β^- > 5°`

Optional stricter family rule:
- use `8°` instead of `5°` for shallow-angle-prone families if calibration supports it

### UNFORMED
Declare `UNFORMED` only if all hold:
1. `O >= 0.80`
2. expected bend zone visible over at least `60%` of nominal target length
3. flat / non-bend model preferred in at least `80%` of valid slices
4. no localized curvature band of meaningful width:
   - `P90(w_k) < max(3h, t_m)`
5. upper 95% bound on reorientation remains below flat threshold:
   - `β^+ < 5°`

### UNKNOWN
Return `UNKNOWN` in all other cases.

## Full vs partial scans
Use the same formal state rules for full and partial scans.

Operational interpretation differs:
- on full scans, `UNKNOWN` should be rare and treated as a system/debugging issue
- on partial scans, `UNKNOWN` is expected whenever visibility gates fail

## Strong negative policy
The system must not infer `UNFORMED` from:
- one visible side only
- global overlay quality only
- heatmap residual only
- process prior only

The system must not infer `FORMED` from:
- one lucky patch with no consistent multi-slice support
- generic local residual without localized bend evidence

## Output contract for observability
Per target:
- `observability_score`
- `frame_confidence`
- `two_side_visibility`
- `curvature_band_visibility`
- `slice_consistency`
- `contextual_localization_confidence`
- `effective_slice_count`
- `state`
- `state_confidence_interval` if available
- `gate_failures[]`
- `notes[]`

## Relationship to structured count
Structured bend count must consume the ternary state explicitly.

Guiding rule:
- `FORMED` contributes positive completion evidence
- `UNFORMED` contributes explicit negative completion evidence
- `UNKNOWN` contributes neither positive nor negative evidence unless a later probabilistic model justifies partial mass

## Benchmark proof slice
First proving slices:
1. partial-stage scans with strong invariant base-panel coverage
2. partial-stage scans with weak visibility where current system overclaims
3. clean full scans where `UNKNOWN` should be very rare

Success criteria:
- lower false-formed rate on partial scans
- fewer overcounts on partial completed-bend counts
- `UNKNOWN` used honestly on weak-observability cases
- full-scan `UNKNOWN` remains rare

## Non-goals
This spec does not define:
- datum-island selection itself
- local angle/radius estimator internals
- rolled-feature validation semantics
- scan-level auto-disposition metric

## Short conclusion
The observability model is the formal mechanism that prevents partial scans from being forced into false binary decisions.

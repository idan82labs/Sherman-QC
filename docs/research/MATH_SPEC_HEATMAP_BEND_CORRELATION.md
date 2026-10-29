# Mathematical Specification — Heatmap vs Bend Correlation

Date: 2026-03-29
Status: draft
Priority: medium-high
Dependencies:
- `MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
- `MATH_SPEC_LOCAL_BEND_METROLOGY.md`
- `MATH_SPEC_EXPECTED_ANGLE_TRUTH.md`
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define when whole-part or local residual heatmaps should correlate with bend metrology, and replace generic heatmap disagreement with a target-specific projected residual model.

## Core principle
Heatmap residual is not a bend metric.

It mixes:
- residual pose error
- local frame error
- true angle error
- true radius error
- correspondence noise

Therefore:
- heatmap may remain a triage / visualization layer
- bend validation should prefer mode-projected local residuals over raw heatmap energy

## Variables and objects
- `r(x)`: residual vector at nominal point `x` after parent-island registration
- `ROI`: local bend region of interest
- `δξ`: residual rigid-pose error
- `δλ`: local frame error
- `δθ`: bend-angle error
- `δr`: bend-radius error
- `ε_corr(x)`: correspondence / ROI mismatch term
- `ε_noise(x)`: scan noise term
- `p_b`: bend-axis point
- `b̂`: bend-axis unit direction

## Residual decomposition
After parent-island registration, linearize residuals in the bend ROI:

`r(x) ≈ J_ξ(x) δξ + J_λ(x) δλ + J_θ(x) δθ + J_r(x) δr + ε_corr(x) + ε_noise(x)`

### Pose Jacobian
`J_ξ(x) = [ I  -[x]_x ]`

where `[x]_x` is the cross-product matrix.

### Angle-mode Jacobian
On the moved child side:

`J_θ(x) = b̂ × (x - p_b)`

Interpretation:
- angle sensitivity grows with lever arm away from the bend axis
- very short flanges produce weak heatmap energy even when angle is wrong

### Radius-mode Jacobian
Inside the bend zone:

`J_r(x)` is approximated by the local arc normal direction

## Generic heatmap energy
The generic local heatmap statistic is:

`E_H = (1 / |ROI|) Σ_{x∈ROI} ||r(x)||^2`

This metric is useful for triage only.

It is not bend-specific because it conflates all residual sources.

## Expected correlation table
### Strong correlation cases
- full scans
- parent-island frame accepted
- released nominal source is unambiguous
- discrete bend with sufficient flange lever arm
- both adjacent sides observed

### Weak or misleading correlation cases
- short flanges
- partial scans
- wrong local frame
- wrong nominal source
- rolled or profile-dominant targets
- generic correspondence noise

## Mode-projected residual estimators
### Angle-mode projection
Estimate the local angle discrepancy by:

`δθ̂_mode = (Σ_i w_i J_θ(x_i)^T r(x_i)) / (Σ_i w_i ||J_θ(x_i)||^2)`

with:
- `x_i ∈ ROI`
- `w_i`: local point weights based on visibility, confidence, and correspondence quality

### Radius-mode projection
Estimate local radius discrepancy by:

`δr̂_mode = (Σ_i w_i J_r(x_i)^T r(x_i)) / (Σ_i w_i ||J_r(x_i)||^2)`

with the bend-zone subset `BZ ⊂ ROI`.

## Recommended use policy
### Allowed
- use heatmap or local residual maps for:
  - visual review
  - operator prioritization
  - confidence downgrading
  - identifying likely frame/nominal mismatches

### Not allowed
- do not let raw heatmap agreement automatically override a local bend failure
- do not let raw heatmap contradiction automatically invalidate a local bend pass

### Preferred comparison
When correlating metrology to residual evidence, compare:
- measured bend-angle/radius error
against:
- `δθ̂_mode`
- `δr̂_mode`

not against raw `E_H`

## Advisory validator semantics
The bend-vs-heatmap validator must remain advisory unless later corpus evidence shows:
- strong calibration of projected residuals
- low false contradiction rate under fixed benchmark slices
- robustness on partial scans and hybrids

## Output contract
Per bend:
- `generic_heatmap_energy`
- `mode_projected_angle_residual`
- `mode_projected_radius_residual`
- `heatmap_correlation_class`
- `correlation_confidence`
- `correlation_notes[]`

Summary:
- `heatmap_support_count`
- `heatmap_contradiction_count`
- `heatmap_insufficient_count`
- `projected_residual_calibration_version`

## Benchmark proof slice
Use:
- clean full discrete-bend parts
- short-flange edge cases
- partial-stage parts
- hybrid rolled parts as negative controls

Success criteria:
- projected residual correlates better with local bend error than raw heatmap energy
- fewer false contradictions than generic heatmap ROI rules
- no use of heatmap as a silent pass/fail override

## Non-goals
This spec does not define:
- datum-island registration
- section-based bend metrology
- observability gating
- rolled-section pass/fail semantics

## Short conclusion
Generic heatmap residual is a mixed signal. Bend validation should use target-specific projected residual modes and keep heatmap-based judgments advisory.

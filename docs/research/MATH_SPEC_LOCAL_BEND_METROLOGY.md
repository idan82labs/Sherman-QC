# Mathematical Specification — Local Bend Metrology

Date: 2026-03-29
Status: draft
Priority: high
Dependency: `MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define how a discrete bend's:
- local frame
- angle
- radius
- confidence

are measured once a trusted parent datum island transform exists.

## Core principle
A local bend is measured in the frame of its parent rigid datum island, not in the whole-part frame.

## Inputs
Required:
- accepted parent-island transform `T_I(x) = R_I x + p_I`
- parent-island frame in CAD coordinates: `(o_I, u_I, v_I, n_I)`
- nominal bend line `γ(s)` in CAD coordinates
- nominal bend-line tangent `b(s)`
- scan point cloud `X`
- scan spacing / noise scale `h`
- material thickness `t_m`

Optional:
- nominal radius `r_nom`
- family-specific angle conversion `θ = g_f(β)`

## Local frame definition
Transport nominal quantities through the accepted parent-island transform.

In scan coordinates:
- `γ_scan(s) = R_I^T (γ(s) - p_I)`
- `b_scan(s) = R_I^T b(s)`
- `nI_scan = R_I^T n_I`

Define section plane at station `s`:
- `Π_s = {x : b_scan(s)^T (x - γ_scan(s)) = 0}`

Define section basis:
- `e1_s = normalize(b_scan(s) × nI_scan)`
- `e2_s = nI_scan`

Section-plane coordinates for point `x`:
- `u = (x - γ_scan(s)) · e1_s`
- `v = (x - γ_scan(s)) · e2_s`

## Local-frame validity gates
The local bend frame is valid only if all of the following hold:
1. accepted parent-island registration exists
2. unique in-plane datum direction exists on the parent island
3. visible nominal bend-line extent:
   - `L_vis >= max(5 t_m, 10 h)`
4. bend line tangent lies in parent plane:
   - `|b(s) · n_I| <= sin(3°)`
5. scan-implied local ridge / curvature tangent `b̂` agrees with transported nominal tangent:
   - `angle(b̂, b_scan) <= 5°`

Reject the local frame if:
- section-plane orientation uncertainty > `0.5°`
- bend-line location uncertainty > `0.25 mm`
- fewer than half the planned slices contain enough points
- parent-island transform is itself abstained

## Slice construction
Let visible bend-line interval be `[s_min, s_max]` with length `L_vis`.

Choose slice spacing:
- if `r_nom` known:
  - `Δs = max(2h, t_m, 0.25 r_nom)`
- otherwise:
  - `Δs = max(2h, t_m)`

Set:
- `K = min(25, floor(L_vis / Δs) + 1)`
- require `K >= 5`

For each station `s_k`, define thin slab:
- `X_k = {x in X : |b_k^T (x - γ_k)| <= δ_s}`
- with `δ_s = max(1.5h, 0.2 t_m)`

## Per-slice fitting
For each slice:
1. project points into the section plane
2. fit side lines by total least squares
3. fit transition arc by Taubin or Pratt circle fit
4. enforce left-to-right ordering and tangency consistency
5. reassign and iterate 2–3 times

Slice is valid for angle/radius only if either:
- both side lines are recovered, or
- one side line and a sufficiently long arc are recovered with consistent tangency

If only one side line is visible and arc support is short, the slice is invalid for angle estimation.

## Per-slice outputs
Let `ℓ1_k`, `ℓ2_k` be fitted unit directions of the two side traces.

Face-normal angle primitive:
- `β_k = arccos(|ℓ1_k^T ℓ2_k|)`

Family-specific reported angle:
- `θ_k = g_f(β_k)`

Radius:
- `r_k = radius of fitted arc`

Per-slice quality score:
- `q_k = 1[slice valid] * exp(-E_B,k / (c1 h^2)) * exp(-(σ_β,k^2 / c2^2))`

where:
- `E_B,k` is slice bend-model fit error
- `σ_β,k` is internal slice angle uncertainty
- `c1`, `c2` are calibration constants frozen once

## Aggregation across slices
Preferred robust aggregation:
- weighted Huber M-estimator, or
- weighted median if a simpler path is needed first

Angle estimate:
- `β̂ = argmin_β Σ_k q_k ρ((β_k - β) / σ_β,0)`

Radius estimate:
- `r̂ = argmin_r Σ_k q_k ρ((r_k - r) / σ_r,0)`

Reported angle:
- `θ̂ = g_f(β̂)`

## Effective evidence and consistency
Define effective slice count:
- `K_eff = (Σ_k q_k)^2 / Σ_k q_k^2`

Define consistency metrics:
- `MAD_β = median_k |β_k - β̂|`
- `MAD_r = median_k |r_k - r̂|`

Estimate confidence intervals by bootstrap over slices.

## Abstention conditions
Return `ABSTAIN_METROLOGY` if any hold:
- `K_eff < 5`
- `MAD_β > max(1.5°, 0.25 * τ_θ,spec)`
- angle bootstrap 95% CI width > `max(2°, τ_θ,spec)`
- fewer than `60%` of planned slices are valid
- parent-island frame invalid

## Output contract for local bend metrology
Per target:
- `parent_island_id`
- `local_frame_valid`
- `slice_count_planned`
- `slice_count_valid`
- `effective_slice_count`
- `per_slice_angles`
- `per_slice_radii`
- `reported_angle`
- `face_normal_angle`
- `reported_radius`
- `angle_confidence_interval`
- `radius_confidence_interval`
- `angle_consistency_mad`
- `radius_consistency_mad`
- `abstain_flag`
- `abstain_reason`

## Benchmark proof slice
First proving slice:
- clean folded parts without rolled hybrids
- full scans with trusted base-panel datums

Success criteria:
- lower internal inconsistency across slices than current one-shot local methods
- better agreement between bend target placement and measured local geometry
- improvement in exact completed-bend-set accuracy, not only raw count MAE

## Non-goals
This spec does not define:
- whether the target is truly a discrete bend
- whether the target is observable enough to call formed/unformed
- how rolled sections are validated

Those are separate specs.

## Short conclusion
Local bend metrology should become a section-first, parent-island-anchored estimator with explicit confidence and abstention.

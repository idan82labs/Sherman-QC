# Mathematical Specification — Rolled Section Validation

Date: 2026-03-29
Status: draft
Priority: high
Dependencies:
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define how rolled sections are:
- represented
- localized
- measured
- validated against tolerance

without forcing discrete-bend semantics onto them.

## Core principle
A rolled section is not a countable discrete bend.

It is a separate feature family with its own:
- parameterization
- observability logic
- tolerance semantics
- pass/fail rule

## Representation
### Minimal QC representation
Represent a rolled feature by:
- `c(u)`: centerline, `u in [0,1]`
- `r(u)`: radius field along centerline
- `α(u)`: arc-extent field along centerline
- `b1(u), b2(u)`: transition boundary curves to adjacent flats/flanges

Minimal feature state:
`R_min = { c(u), r(u), α(u), b1(u), b2(u) }`

### Richer representation when needed
For mating-critical or profile-critical cases, represent the full section field:
- `q(u,η)`, `η in [0,1]`

where `q(u,η)` is the section-profile curve in the local section frame at station `u`.

## Centerline definition
Let `b1(u)` and `b2(u)` be the two transition boundaries traced along the roll.

Default centerline:
- `c(u) = 0.5 * (b1(u) + b2(u))`

If the section is strongly asymmetric, replace midpoint centerline with the principal-curvature ridge or a fitted medial curve.

## Local section frame
Use a rotation-minimizing / Bishop frame along the centerline.

Define tangent:
- `t(u) = c'(u) / ||c'(u)||`

Construct orthonormal normal directions `n1(u), n2(u)` with minimum twist subject to:
- `n1(u) ⟂ t(u)`
- `n2(u) = t(u) × n1(u)`

Section plane at station `u`:
- `Π_u = { x : t(u)^T (x - c(u)) = 0 }`

## Section extraction
At station `u_k`, collect a slab of scan points around `Π_u` with half-thickness:
- `δ_roll = max(1.5h, 0.2 t_m)`

Project points into the section plane and fit either:
1. the simplified circular-arc surrogate, or
2. the full section-profile curve

depending on feature family and required precision.

## Radius and arc extent
### Radius field
For the circular surrogate at station `u_k`, fit circle radius `r_k`.

Default representation is a field:
- `r = r(u)`

If the family is known constant-radius and calibration supports it, allow simplified model:
- `r(u) = r0`

Do not assume constancy by default.

### Arc extent
Let the section intersect the two transition boundaries at two points. Relative to the fitted circle center, define:
- `α(u) = angle(boundary ray 1, boundary ray 2)`

## Transition boundary definition
In each section, transition points are change points where both hold over a contiguous interval of length at least `max(3h, t_m)`:
1. curvature falls below the flat threshold
2. outside points fit a line or nominal adjacent face with RMS <= `τ_line`

Trace those change points across `u` to obtain `b1(u), b2(u)`.

Default:
- `τ_line = 2h`

## Measurement outputs
For each sampled station `u_k`, compute:
- `ĉ(u_k)`: measured centerline point
- `d_c(u_k)`: centerline placement deviation
- `d_p(u_k)`: section-profile deviation
- optional `d_r(u_k) = |r̂(u_k) - r(u_k)|`
- optional `d_α(u_k) = |α̂(u_k) - α(u_k)|`

## Centerline deviation metric
Ignore tangential drift along the centerline. Measure normal displacement only:
- `d_c(u) = || (I - t(u)t(u)^T) (ĉ(u) - c(u)) ||`

This prevents reparameterization drift along the feature from dominating the QC signal.

## Section-profile deviation metric
After anchoring at the corresponding centerline point, define profile normal deviation:
- `d_p(u) = P95_{η in [0,1]} | (q̂(u,η) - q(u,η)) · n_q(u,η) |`

where `n_q(u,η)` is the local normal of the nominal section profile.

If using the simplified circular surrogate, keep:
- `d_r(u) = |r̂(u) - r(u)|`
- `d_α(u) = |α̂(u) - α(u)|`

as auxiliary diagnostics or explicit toleranced metrics when called out by the print.

## Pass/fail semantics
Rolled-section validation uses a dual criterion:
1. placement
2. profile shape

### Placement pass
Placement passes if both hold:
- `P95(d_c) <= τ_c`
- `max_u d_c(u) <= τ_c_max`

### Profile pass
Profile passes if both hold:
- `P95(d_p) <= τ_p`
- `max_u d_p(u) <= τ_p_max`

### Optional explicit radius / extent checks
If radius or arc extent are explicitly toleranced, also require:
- `P95(d_r) <= τ_r`
- `P95(d_α) <= τ_α`

### Overall rolled-feature pass rule
A rolled feature passes if and only if:
1. centerline placement passes
2. section profile passes
3. all explicitly called-out radius / extent metrics pass

This is an AND rule, not a weighted average.

## Observability requirements
A rolled feature is evaluable only if:
- enough stations are available
- centerline fit is stable
- section extraction is stable at a sufficient fraction of stations

Default minimums:
- sampled stations `K >= 5`
- valid stations / planned stations >= `0.60`
- no large station-to-station centerline discontinuity inconsistent with nominal

If these fail, return `ABSTAIN_ROLLED_VALIDATION`.

## Relationship to global heatmap
Global heatmap is acceptable only for coarse triage, and only when all of the following hold:
- full scan
- simple constant-radius roll
- centerline already validated within tolerance
- the family has already shown strong empirical correlation between local heatmap ROI and section-profile deviation

Otherwise, global heatmap is not a valid rolled-feature validator.

It may remain a visualization layer only.

## Output contract for rolled validator
Per rolled feature:
- `feature_type = ROLLED_PROCESS_FEATURE`
- `centerline_samples`
- `radius_samples`
- `arc_extent_samples`
- `centerline_p95_deviation`
- `centerline_max_deviation`
- `profile_p95_deviation`
- `profile_max_deviation`
- optional `radius_p95_deviation`
- optional `arc_extent_p95_deviation`
- `placement_pass`
- `profile_pass`
- `overall_pass`
- `valid_station_count`
- `planned_station_count`
- `abstain_flag`
- `abstain_reason`

## Benchmark proof slice
First proving slice:
- rolled/hybrid parts with released final scans judged good by human review
- a smaller subset with known-bad rolled geometry if available

Success criteria:
- rolled geometry is no longer forced into bend-angle semantics
- good rolled parts pass on centerline + profile metrics
- bad rolled parts fail for local profile or placement for interpretable reasons
- disagreement with global heatmap becomes diagnostic, not a blocker

## Non-goals
This spec does not define:
- countable discrete bend formation
- local bend angle/radius metrology on non-rolled features
- scan-level operator-disposition policy

## Short conclusion
Rolled sections need a dedicated validator based on centerline placement and local section-profile agreement, not a repurposed bend-angle check.

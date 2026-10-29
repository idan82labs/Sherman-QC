# Mathematical Specification — Feature Classification

Date: 2026-03-29
Status: draft
Priority: high
Dependencies:
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define a mathematically testable discriminator between:
- `COUNTABLE_DISCRETE_BEND`
- `ROLLED_PROCESS_FEATURE`
- `NON_BEND_TRANSITION`

This classification is required before counting, observability, or local metrology can be trusted.

## Core principle
Feature type must be decided by competing section models plus side-rigidity evidence.

Line heuristics alone are insufficient.

## Inputs
For a candidate feature with nominal line or ridge `γ(s)`:
- scan point cloud `X`
- nominal candidate line / ridge `γ(s)`
- local nominal sheet normal `n(s)`
- scan spacing / noise scale `h`
- material thickness `t_m`

## Section construction
At station `s_k`, define:
- tangent `t_k = γ'(s_k) / ||γ'(s_k)||`
- section plane `Π_k = { x : t_k^T (x - γ(s_k)) = 0 }`
- in-plane basis:
  - `e1_k = normalize(t_k × n_k)`
  - `e2_k = n_k`

Collect section points:
- `X_k = { x in X : | t_k^T (x - γ_k) | <= δ_s }`
- `δ_s = max(1.5h, 0.2 t_m)`

Project each point into 2D section coordinates `(u,v)`.

## Competing section models
### Model M_B — discrete bend
Use a line-arc-line model with tangency continuity:
- two side lines
- one central circular arc
- tangent continuity at side/arc transitions

Per-slice fit error:
- `E_B^(k) = min_{θ_B} Σ_{z_i∈X_k} ρ( d_perp(z_i, g_B)^2 / h^2 )`

### Model M_R — rolled / distributed-curvature feature
Use a smooth profile spline with curvature regularization:
- no requirement for long flat side plateaus
- broad curvature support allowed

Per-slice fit error:
- `E_R^(k) = min_{θ_R} Σ_{z_i∈X_k} ρ( d_perp(z_i, g_R)^2 / h^2 ) + λ_R ∫ |κ'(η)|^2 dη`

### Model M_N — non-bend transition
Use the better of:
1. single-line model
2. short local blend / quadratic / fillet surrogate with small-angle prior

Per-slice fit error:
- `E_N^(k) = min(E_L^(k), E_Q^(k))`

## Additional discriminative quantities
Fit error alone is not sufficient.

### Curvature support localization
Estimate a smoothed section curve `ĝ_k(u)` and its curvature `κ_k(u)`.
Let `σ_κ` be curvature noise estimated from nearby flat or low-curvature regions.

Define curvature support:
- `C_k = { u : |κ_k(u)| > 3 σ_κ }`

Define support width:
- `w_k = |C_k|`

Define left/right plateau intervals `A1_k`, `A2_k` as maximal side intervals adjacent to `C_k` such that:
- line-fit RMS <= `τ_line`
- `|κ_k(u)| <= 3 σ_κ`

Define localization ratio:
- `λ_k = w_k / (w_k + |A1_k| + |A2_k|)`

### Total turning angle
Define:
- `Φ_k = ∫_{C_k} |κ_k(u)| du`

This is the local reorientation captured by the curvature band.

### Side rigidity / planarity
For side patches `P1_k`, `P2_k`, fit planes `π1_k`, `π2_k`.

Plane RMS:
- `ε_π,j,k = sqrt( (1/|Pj_k|) Σ_{x∈Pj_k} d(x, πj_k)^2 )`

Side-normal dispersion:
- `σ_n,j,k = sqrt( (1/|Pj_k|) Σ_{x∈Pj_k} angle(n_x, n_{πj_k})^2 )`

If literal planarity is too strict for a family, replace plane fit with rigid-fit residual to nominal side geometry.

## Per-slice model scoring
Use robust BIC-like score:
- `BIC~_m^(k) = n_k log(E_m^(k) / n_k) + d_m log(n_k)`

for `m in {B, R, N}`.

Aggregate by median across valid slices:
- `BIC~_m = median_k BIC~_m^(k)`
- `λ~ = median_k λ_k`
- `Φ~ = median_k Φ_k`

Aggregate side rigidity metrics by median across slices.

## Default thresholds
- `Δ_BIC = 10`
- `τ_line = 2h`
- `τ_π = 3h`
- `τ_n = 3°`
- minimum real bend turning angle: `Φ~ >= 10°`
- non-bend small-angle trigger: `Φ~ < 5°`
- discrete bend localization ratio: `λ~ <= 0.35`
- rolled localization ratio: `λ~ >= 0.60`
- minimum valid slices: `K_min = 5`

The interval `0.35 < λ~ < 0.60` is intentionally ambiguous unless the model winner is overwhelming.

## Decision rules
### Classify as COUNTABLE_DISCRETE_BEND if all hold
1. `BIC~_B + Δ_BIC < min(BIC~_R, BIC~_N)`
2. `λ~ <= 0.35`
3. `Φ~ >= 10°`
4. both side regions are rigid-like:
   - `median ε_π,j,k <= τ_π`
   - `median σ_n,j,k <= τ_n`

### Classify as ROLLED_PROCESS_FEATURE if all hold
1. `BIC~_R + Δ_BIC < min(BIC~_B, BIC~_N)`
2. and either:
   - `λ~ >= 0.60`, or
   - one or both side plateaus are missing / non-rigid

### Classify as NON_BEND_TRANSITION if either holds
1. `BIC~_N + Δ_BIC < min(BIC~_B, BIC~_R)`
2. `Φ~ < 5°`

## Abstention
Return `UNKNOWN_FEATURE_TYPE` instead of forcing a class if any hold:
- fewer than `K_min` valid slices
- no model wins by `Δ_BIC`
- slice-level votes disagree by more than `30%`
- one or both side patches are too occluded to evaluate rigidity
- candidate line itself is unstable under small perturbations

## Feature-family caveats
### Faceted CAD approximations of rolls
Collapse nearby candidate lines before section fitting, otherwise one rolled feature may look like many micro-bends.

### Coarse scans
Very coarse scans can broaden curvature support numerically and make a real bend look rolled. Calibrate thresholds against `h`.

### Very shallow features
Features under `10°` total turning angle are family-sensitive and should not be forced into ordinary bend semantics without an explicit family rule.

### Embossed or decorated faces
If side planarity is violated by intentional emboss/decor, use rigid-fit-to-nominal side residual instead of raw plane RMS.

## Output contract for classifier
Per candidate target:
- `feature_type`
- `feature_type_confidence`
- `winning_model`
- `model_scores`
- `localization_ratio`
- `total_turning_angle`
- `side_rigidity_metrics`
- `valid_slice_count`
- `abstain_flag`
- `abstain_reason`

## Benchmark proof slice
First proving slices:
1. clean folded bends
2. rolled/hybrid parts with known rolled bodies and local flange bends
3. non-bend transition examples where current system over-detects

Success criteria:
- fewer rolled-process features miscounted as discrete bends
- fewer shell-profile surrogates selected as bend targets
- cleaner separation between countable targets and process-only features

## Non-goals
This spec does not define:
- parent-island transform estimation
- final bend angle/radius measurement
- observability state mapping
- scan-level auto-disposition metric

## Short conclusion
Feature classification must become a model-selection problem in normal sections, not a heuristic ranking over candidate lines.

# Mathematical Specification — Datum-Island Alignment

Date: 2026-03-29
Status: draft
Priority: highest
Source: derived from the professor's mathematical response on 2026-03-29

## Purpose
Define the authoritative local metrology frame for:
- full finished parts
- partial-stage parts
- rolled hybrids

This specification replaces the current implicit assumption that one whole-part transform is sufficient for local bend metrology.

## Core principle
Whole-part overlay may remain useful for visualization.

It is **not** the authoritative local measurement frame.
The authoritative frame is the accepted rigid transform of a trusted parent datum island.

## Variables and objects
- `X = {x_i}`: scan point cloud in scan coordinates
- `S_CAD`: released nominal CAD surface / mesh in CAD coordinates
- `T(x) = R x + p`, where `R ∈ SO(3)`, `p ∈ R^3`: rigid transform from scan to CAD coordinates
- `I`: candidate parent datum island
- `H`: hole features on `I`
- `S`: slot features on `I`
- `P`: plane / invariant flat features on `I`
- `E`: edge / notch features on `I`
- `U`: invariant surface patches on `I`
- `q_i ∈ [0,1]`: feature quality score
- `σ_i`: feature uncertainty
- `ρ(·)`: Huber loss with scale initialized from scan spacing `h`

## Candidate transform
For each candidate datum island `I`, solve:

`T*_I = argmin_{T in SE(3)} L_I(T)`

where `L_I(T)` is a weighted sum over allowed datum residuals on that island.

## Objective function
`L_I(T) = Σ_{i∈H} w_i^H ρ(e_i^H(T)) + Σ_{i∈S} w_i^S ρ(e_i^S(T)) + Σ_{j∈P} w_j^P ρ(e_j^P(T)) + Σ_{k∈E} w_k^E ρ(e_k^E(T)) + Σ_{ℓ∈U} w_ℓ^U ρ(e_ℓ^U(T))`

with default weights:

`w = q / σ^2`

### Hole residual
For fitted scan hole center `ĉ_i`, optional axis `â_i`, nominal center `c_i`, nominal axis `a_i`:

`e_i^H(T) = ||T ĉ_i - c_i||^2 / σ_{c,i}^2 + λ_a * angle(R â_i, a_i)^2 / σ_{a,i}^2`

### Slot residual
For slot center `ĉ_i`, major-axis direction `û_i`, optional length `ℓ̂_i`:

`e_i^S(T) = ||T ĉ_i - c_i||^2 / σ_{c,i}^2 + λ_u * angle(R û_i, u_i)^2 / σ_{u,i}^2 + λ_ℓ * (ℓ̂_i - ℓ_i)^2 / σ_{ℓ,i}^2`

### Plane residual
For scan patch `Q_j` and nominal plane `(n_j, d_j)`:

`e_j^P(T) = (1 / |Q_j|) Σ_{x∈Q_j} (n_j^T (T x) + d_j)^2 / σ_{p,j}^2`

Optional scan-normal term:

`+ λ_n * angle(R n̂_j, n_j)^2 / σ_{n,j}^2`

### Edge / notch residual
For scan edge points `Q_k` and nominal edge curve `E_k`:

`e_k^E(T) = (1 / |Q_k|) Σ_{x∈Q_k} min_{y∈E_k} ||T x - y||^2 / σ_{e,k}^2`

### Invariant surface residual
For trusted invariant patch `Q_ℓ`:

`e_ℓ^U(T) = (1 / |Q_ℓ|) Σ_{x∈Q_ℓ} dist(T x, S_ℓ_CAD)^2 / σ_{u,ℓ}^2`

## Feature activation by family
### Full finished parts
Active terms:
- holes / slots on the correct rigid panel
- island plane / invariant flats
- edges / notches on that island
- invariant surfaces on that island

Weight policy:
- holes / slots dominate if quality and conditioning are good

### Partial-stage parts
Active terms:
- only features on invariant geometry shared between the current stage and the nominal target stage

Hard rule:
- zero weight on active bend zones, future bend influence zones, rolled zones, and moving free flanges

### Rolled hybrids
Active terms:
- non-rolled flange datums and base-panel datums only

Hard rule:
- rolled patch and rolled transition geometry must not anchor the transform in first-pass metrology alignment

## Conditioning and trustworthiness
A low residual alone is insufficient.

After solving for `T*_I`, linearize and form:

`F = J^T W J`

Use `Σ_T ≈ F^-1` as the transform covariance proxy.

Accept a candidate datum-island alignment only if all of the following hold:
1. normalized mean residual is low:
   - `L_I(T*_I) / N_active <= τ_residual`
   - initialize `τ_residual ≈ 1` in normalized units
2. 95% pose uncertainty at the island centroid is small:
   - translation < `0.25 mm`
   - rotation < `0.25°`
3. hold-out residual remains stable:
   - `e_holdout <= 1.5 * e_infit`

## Hole-pattern conditioning rule
Project accepted hole/slot centers to the parent-island plane. Let weighted covariance be `M` with eigenvalues `λ1 >= λ2`.

Require:
- non-collinearity: `λ2 / λ1 >= 0.1`
- spatial spread: `sqrt(λ1 + λ2) >= 0.3 * D_I`

where `D_I` is the parent-island diagonal.

## Hole/slot feature quality thresholds
Accept a hole/slot feature only if all relevant criteria hold:
- visible coverage fraction >= `0.60`
- center uncertainty <= `max(2h, 0.05d_i)`
- diameter sanity: `|d̂_i - d_i| <= max(2h, 0.03d_i)`
- if axis used: `angle(â_i, a_i) <= 3°`

## Source trust hierarchy
Define source scores:

`S_H = log det(F_H + εI) - β ē_H - γ a_H`
`S_P = log det(F_P + εI) - β ē_P`
`S_U = log det(F_U + εI) - β ē_U - η d_deform`

Interpretation:
- `F_*`: source-specific information matrix
- `ē_*`: mean normalized residual
- `a_H`: ambiguity penalty for repeated / symmetric hole patterns
- `d_deform`: deformation risk on surfaces near bends / rolls / moving flanges

Decision rule:
1. if `S_H >= S_P + 1` and hole conditioning passes, trust holes/slots over planes
2. if plane solution is trusted, planes outrank generic surface refinement
3. surface refinement is subordinate only and must not move pose by more than:
   - `0.25 mm`
   - `0.25°`

## Material disagreement rule
If two trusted sources disagree materially:
- translation disagreement > `0.25 mm`, or
- rotation disagreement > `0.25°`

then do **not** average them.

Return `ABSTAIN_ALIGNMENT` unless one source wins by trust margin.

## Output contract for alignment module
Each solved parent-island alignment should expose:
- `parent_island_id`
- `transform`
- `transform_covariance_proxy`
- `normalized_residual`
- `holdout_residual`
- `anchor_source_winner`
- `source_scores`
- `feature_residual_breakdown`
- `conditioning_metrics`
- `abstain_flag`
- `abstain_reason`

## Benchmark proof slice
First proving slice for this spec:
- scans with preserved base panel and either:
  - 3 quality holes/slots on the same rigid panel, or
  - 2 holes/slots plus 1 stable plane

Success criteria on that slice:
- fewer false local bend deltas caused by bad whole-part pose
- reduced disagreement between local metrology and visual overlay on clean full parts
- lower rate of spurious bend failure on globally well-aligned finished parts

## Non-goals
This spec does not define:
- bend vs rolled feature classification
- local bend angle/radius estimation
- rolled-section validation
- scan-level auto-disposition policy

Those depend on this spec but are separate.

## Short conclusion
Registration is the first mathematical model to freeze because every downstream local decision depends on it.

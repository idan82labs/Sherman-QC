# Bend State Factor Model Architecture

## Purpose
This document converts the current open research question into an implementation direction for the bend-inspection engine.

The main conclusion is:

- the model must stop treating `FORMED`, `NOT_FORMED`, `PARTIALLY_OBSERVED`, and `UNOBSERVED` as one flat latent state
- instead, it should separate:
  - physical bend completion
  - observability / visibility
  - local evidence assignment

This is the next serious mathematical step for lowering bend-count MAE on hard partial scans.

## Current problem
The current runtime still mixes three different concepts:

1. `physical completion`
- is the bend actually formed relative to the CAD target?

2. `observability`
- was the bend sufficiently visible in this scan to support a decision?

3. `assignment`
- which local scan evidence belongs to which CAD bend?

This is why several mathematically plausible changes have failed the regression gate:
- threshold tightening
- geometry-weighted local assignment
- global optimal assignment

They improve one piece of the logic but leave the state space itself underspecified.

## Proposed latent variables
For each CAD bend `i`, define:

- `C_i`: physical completion state
  - initial implementation: binary `FORMED` / `NOT_FORMED`
  - future extension: continuous angle progress or hinge variable

- `rho_i`: calibrated continuous visibility score
  - initial implementation should use a continuous observed/estimated nuisance variable rather than a discrete latent visibility class
  - sources:
    - CAD ray-casting / expected visible fraction
    - realized local support
    - local density / coverage
    - alignment stability
  - discrete observability labels can still be reported, but should be derived from `rho_i` and posterior confidence

- `A_i`: assignment state
  - which local candidate patch supports bend `i`
  - includes explicit `null` assignment

Optional global variable:

- `S`: process stage
  - only used when stage information or scan ordering exists

Reported user-facing label `R_i` is then derived, not primary:

- `UNOBSERVED`
  - `rho_i < τ_0`
- `PARTIALLY_OBSERVED`
  - `rho_i ≥ τ_0`, but posterior over completion still not decisive
- `FORMED`
  - observability sufficient, posterior favors formed state
- `NOT_FORMED`
  - observability sufficient, posterior favors unformed/flat state

## Evidence decomposition
For each CAD bend, build two evidence blocks.

### 1. Observability features `u_i`
These estimate whether the bend could have been judged at all, and are used to build/calibrate `rho_i`.

Candidates:
- local point support
- side1 / side2 support
- expected visible fraction from CAD-facing visibility
- realized coverage ratio
- local neighborhood occupancy
- scan density near bend line
- local occlusion heuristics
- alignment stability in the local patch

### 2. Geometry / fit features `z_i`
These estimate physical completion given that the bend is observable.

Candidates:
- fitted local bend angle
- fitted radius
- flange-normal consistency
- bend-line direction consistency
- local residual / fit error
- planarity metrics
- line-center deviation
- line length / arc length deviation

Key modeling rule:
- negative evidence must be visibility-weighted
- low `rho_i` must suppress confidence in geometric conclusions
- `UNOBSERVED` is not evidence of absence

## Factorization
The current recommended starting model is a discriminative latent-variable factor graph / CRF-style conditional model:

`p(C, A | x) ∝ ∏_i ψ_i(C_i, A_i ; rho_i, g_i) ∏_m ψ_excl_m(A) ∏_(i,j) ψ_adj_ij(C_i, C_j ; rho_i, rho_j)`

Where:

- `C_i`
  - binary physical completion state for bend `i` in v1: `FORMED` / `NOT_FORMED`
- `A_i`
  - explicit assignment of bend `i` to a local evidence candidate or `null`
- `rho_i`
  - calibrated continuous observability / informativeness score
- `ψ_i`
  - local conditional potential using candidate-local geometric features `g_i`
- `ψ_excl`
  - hard exclusivity over confusable evidence atoms / candidate cliques
- `ψ_adj`
  - weak adjacency only when later needed, and only when visibility is sufficient

## Why CRF / factor graph style
This project already uses overlapping engineered geometric features.
That makes a CRF-style conditional model a better fit than a naive generative factorization.

Reason:
- angle fit, radius fit, normal consistency, residuals, and planarity are not conditionally independent
- multiplying them naively double-counts evidence
- CRF-style potentials let us use overlapping features directly and calibrate them empirically

Practical recommendation:
- fit learned conditional unary terms from the corpus evidence vectors immediately
- use separate learned heads rather than a flat multiclass label model:
  - one observability head for `rho_i`
  - one formed / not-formed state head conditioned on candidate evidence
- do not hand-tune all unaries first
- keep pairwise / higher-order factors hand-specified in the first implementation
- postpone end-to-end structured learning until the local unary model is stable

## First learned model recommendation
The recommended first learned unary stack is:

- calibrated gradient-boosted trees for observability
- calibrated gradient-boosted trees for state conditioned on candidate evidence
- penalized logistic regression kept as an ablation / sanity-check baseline

Reason:
- current evidence is tabular and heterogeneous
- corpus size is meaningful but not large enough to justify a neural-first approach
- calibration and debug visibility matter more than representational capacity in v1

## Sparse coupling only
Do not couple everything.
Only add structure that has physical meaning.

Keep:
- hard exclusivity between confusable bends / evidence atoms first
- weak shared-flange adjacency only later if exclusivity plateaus
- known or plausible precedence constraints only when data supports them
- count monotonicity across ordered scans through posterior regularization, not bend-wise monotonicity in v1

Do not add:
- dense all-to-all similarity coupling
- coupling based only on parallel bend lines or similar target angles
- a primary stage latent variable in the first serious model

## Initial implementation scope
This should not begin as a full research-grade inference engine.
The first implementation should be intentionally small.

### Phase 1 runtime model
- keep current CAD extraction and local measurement pipeline
- introduce internal fields for:
  - `physical_completion_state`
  - `observability_state`
  - `assignment_source`
  - `assignment_confidence`
- add per-bend observability features to persisted outputs
- derive current UI/report labels from these states

### Phase 2 local candidate model
- generate explicit local candidate patches per CAD bend neighborhood
- allow `null` assignment
- score candidates separately from completion decision
- make confusable-bend exclusivity the first essential coupling

### Phase 3 structured decoding
- add hard exclusivity for confusable bends / evidence atoms
- add weak adjacency only if needed after exclusivity ablations
- keep temporal structure to soft count monotonicity first
- decode completed-bend count from a posterior over total formed bends, not only by independent per-bend argmax or raw joint MAP

## Decoding objective
Current reporting ultimately needs:
- per-bend label
- total completed bends
- in-spec / out-of-spec status

For count accuracy, do not rely only on independent bend argmax or raw joint MAP.
The decoded completed-bend count should be taken from a posterior over total formed bends.

Practical implication:
- keep per-bend probabilities
- also track posterior or score over total completed count `K = Σ_i 1[C_i = FORMED]`
- report count from the posterior over `K`
- use the posterior median of `K` as the MAE-aligned point estimate in v1
- then recover a per-bend explanation that is reconciled with that count

## Immediate target families
This model direction should first be tested on:

- `38172000`
- `40500000`
- `10839000`
- `11979003` partial

Reason:
- these are the informative staged/partial failures
- they cover repeated tray bends, weak early-stage evidence, and ambiguous local observability

## Acceptance criteria for promotion
No model change should be promoted unless:

1. it passes backend tests
2. it improves or preserves the staged target slice
3. it does not regress the promoted full-corpus baseline
4. it improves reproducibility on sensitive families such as `49024000_A.ply`

## Non-goals for the first pass
Do not attempt all of these immediately:
- full Bayesian calibration
- full temporal graphical model across all scans
- full continuous hinge-state inference
- family-specific specialized inference engines

These are later extensions.

The first real win is simply this:
- split completion from observability and assignment
- make those separations explicit in runtime outputs
- evaluate them with the current corpus

## Monotonicity recommendation
For staged-process scans:

- do not start by pushing monotonicity harder into the state prior
- start with posterior regularization with slack instead
- start with count monotonicity only
- add bend-wise monotonicity only later, visibility-gated, for bends that were strongly observed and strongly formed in earlier scans

Reason:
- it preserves the distinction between weak observation and true physical progression
- it avoids letting process assumptions masquerade as measurement evidence

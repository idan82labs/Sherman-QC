# Professor Response: Implementation Translation Note

## Status
This note converts the professor guidance into concrete implementation choices for the bend-state model.

## New authoritative decisions

### 1. Core model family
Use a conditional latent-variable factor graph / CRF-style model with three explicit components per bend:
- physical completion state `C_i`
- observability score `rho_i`
- candidate-or-null assignment `A_i`

Do not use:
- a flat multiclass state model over `FORMED / NOT_FORMED / PARTIALLY_OBSERVED / UNOBSERVED`
- a primary stage latent variable in v1
- a pure heuristic local decision system as the main path forward

### 2. First learned model
Use a two-head calibrated gradient-boosted-tree stack:
- observability head for `rho_i`
- formed / not-formed state head conditioned on candidate evidence

Keep penalized logistic regression as an ablation baseline.
Do not start with a small neural model.

### 3. Visibility formulation
Use one global scalar `rho_i` first.
Include family-aware features.
If needed later, allow family-specific calibration rather than separate family-specific models from day one.

### 4. Assignment structure
Use explicit candidate-or-null assignment.
Use hard exclusivity first, but over confusable evidence atoms / cliques rather than raw overlapping windows.

Shared-flange adjacency is secondary.
Add it only later, weakly, and visibility-gated.

### 5. Count decoding
Do not use raw joint MAP as the final count decoder.
Do not add a separate count head in v1.

Instead:
- derive a posterior over total formed bends `K`
- decode count using the posterior median of `K`
- then reconcile the bend-wise explanation with the reported count

### 6. Monotonicity
Use count monotonicity first, only as soft posterior regularization.
Do not impose bend-wise monotonicity in v1 except possibly later for strongly observed bends.

## Immediate implications for implementation order

### Priority 1
Fit learned unaries on top of the current candidate surface:
- observability head
- formed/not-formed head

### Priority 2
Define confusable evidence atoms / cliques for hard exclusivity.

### Priority 3
Add count-posterior decoding aligned with MAE.

### Priority 4
Add soft count monotonicity across staged scans.

### Priority 5
Only then consider weak adjacency.

## Practical interpretation
The professor's main diagnosis is that the hardest unresolved issue is not generic unary quality alone.
It is wrong assignment structure under partial observability.

That means the next model should not be “more threshold tuning” or “a better local classifier in isolation.”
It should be:
- visibility-aware,
- candidate-based,
- exclusivity-aware,
- and count-decoded in a way aligned with MAE.

## Most important consequence
The current repository direction was broadly correct, but this response sharpens the next steps materially:
- GBDT first, not neural-first
- hard exclusivity first, not soft-first
- count posterior median, not raw MAP count
- count-only monotonicity first, not bend-wise first

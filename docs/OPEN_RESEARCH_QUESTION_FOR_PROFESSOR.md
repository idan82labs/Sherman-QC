# Open Technical Question

## Main unresolved mathematical question

How should we formulate bend-state inference on partial industrial point-cloud scans so that the system can distinguish, in a statistically and geometrically principled way, between:

1. a bend that is already formed,
2. a bend that is still flat / not formed,
3. a bend that is only partially observed,
4. and a bend that is simply unobserved,

when several candidate bends in the CAD have very similar local geometry, the scan has strong occlusion / anisotropic coverage, and the alignment is only globally good but not necessarily locally informative?

## Why this is the real blocker

Our current bottleneck is not CAD extraction on easy parts, and not UI/reporting. It is the mathematical problem of local state inference under partial observability.

In the hard tray-style families, several bends:
- have similar target angles,
- are spatially near one another,
- may have nearly parallel bend directions,
- and may expose only one flange or one side of the local neighborhood in the scan.

This creates a failure mode where a locally weak measurement can be assigned to the wrong CAD bend, or can be interpreted as evidence of formation when it is really only weak/partial coverage.

## The specific question I would ask a math professor / senior theoretical engineer

If you had to design this from first principles, would you model this as:

- a latent-state estimation problem on a graph of CAD bends,
- a Bayesian inference problem with observability-conditioned likelihoods,
- a structured optimal transport / assignment problem with hidden observation states,
- or a sequential filtering problem with monotonic process priors,

and what would be the most defensible mathematical formulation for the likelihood term that connects local scan evidence to CAD bend state?

More concretely:

Given, for each CAD bend `b_i`, a local evidence vector

- local point support,
- visibility / coverage score,
- fitted local angle / radius,
- flange-normal consistency,
- bend-line direction consistency,
- local planarity / residual,
- and global alignment quality,

how would you define a model that estimates

`P(state_i | local_evidence_i, neighboring_states, process_stage)`

where

`state_i ∈ {FORMED, NOT_FORMED, PARTIALLY_OBSERVED, UNOBSERVED}`

in a way that is:

- geometrically faithful,
- robust to occlusion,
- monotone across production stages when appropriate,
- and identifiable enough to reduce systematic bend-count MAE on hard partial scans?

## Why I think this matters

This is the point where further heuristic threshold tuning is no longer enough.

We already tested and rejected:
- stricter one-sided evidence thresholds,
- geometry-weighted local assignment tweaks,
- global optimal assignment for dense local refinement,

because they either did not solve the hard staged cases or regressed the broader corpus.

So the real open problem is:

What is the right mathematical state model for bend completion under partial observability?

That is the question most likely to move the system materially toward lower MAE and real enterprise-grade reliability.

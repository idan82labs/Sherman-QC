# One-Page Research Memo: Partial Bend-State Inference Under Occlusion

## Context
We are building an industrial bend-inspection system that compares CAD geometry against LiDAR / point-cloud scans of sheet-metal parts during production. The system must infer progression on partially bent parts: how many bends are complete, which bends are still missing, and whether completed bends are within tolerance.

The practical target is low bend-count MAE on real staged production scans, not only good performance on fully completed parts.

## Core problem
The main unsolved technical problem is not CAD extraction on easy parts, and not reporting/UI. It is this:

How should we infer the state of each CAD bend on a partial point-cloud scan when local evidence is incomplete, ambiguous, and unevenly observed?

For each CAD bend, the true latent state is one of:
- `FORMED`
- `NOT_FORMED`
- `PARTIALLY_OBSERVED`
- `UNOBSERVED`

The difficulty is that on real tray/channel-style parts:
- several bends are geometrically similar
- multiple bends may have similar target angles
- bend lines may be parallel or nearly parallel
- scan coverage is anisotropic and occluded
- alignment can be globally good while still being locally uninformative

This causes a systematic ambiguity:
- weak local evidence can be assigned to the wrong bend
- partial coverage can be mistaken for bend completion
- missing evidence can be mistaken for negative evidence

## What has already been established experimentally
We now have a real industrial regression corpus with CAD, scans, drawings, and partial truth labels. On this corpus:

1. A useful principle was confirmed:
- local `UNOBSERVED` must be treated as missing evidence, not evidence of absence
- promoting this rule improved hard staged families without harming the broader corpus

2. Several plausible fixes were tested and rejected:
- stricter one-sided local evidence thresholds
- geometry-weighted local assignment using bend direction and flange-normal consistency
- global optimal assignment across dense local bend candidates

Reason for rejection:
- they either failed to improve the hard staged target families
- or regressed broader corpus metrics

This suggests the remaining error is not just bad thresholds or a weak assignment objective. The deeper issue is the lack of a correct state model for local bend completion under partial observability.

## Current mathematical question
If this problem were formulated from first principles, what is the most defensible model for estimating bend state from partial point-cloud evidence?

More concretely, for each CAD bend `b_i`, we can compute a local evidence vector such as:
- local point support
- visibility / coverage score
- fitted local bend angle / radius
- flange-normal consistency
- bend-line direction consistency
- local residual / planarity / fit quality
- global alignment quality
- sequence position in a staged process, if available

The target is to estimate:

`P(state_i | local_evidence_i, neighboring_states, process_stage)`

where:
- `state_i ∈ {FORMED, NOT_FORMED, PARTIALLY_OBSERVED, UNOBSERVED}`

The key requirements are:
- geometrically faithful
- robust to occlusion and one-sided visibility
- monotone across process stages when appropriate
- identifiable enough to reduce systematic bend-count MAE on hard partial scans

## Candidate formulation families
The question is which family is the right starting point:

1. **Bayesian latent-state model**
- Each bend has a hidden state.
- Local scan evidence provides an observability-conditioned likelihood.
- Process stage and adjacent bends provide priors or coupling terms.
- Attractive because it explicitly separates uncertainty, visibility, and completion.

2. **Graphical / structured state model**
- Represent CAD bends as nodes in a graph.
- Add pairwise constraints for spatial adjacency, shared flanges, and progression monotonicity.
- Infer the joint bend-state configuration rather than independent bend labels.
- Attractive because repeated similar bends may be distinguishable only in a joint configuration.

3. **Sequential filtering / monotone progression model**
- Treat the manufacturing process as an ordered latent progression.
- Later scans update the posterior over bend states subject to monotonic constraints.
- Attractive if stage order is known or partially known.

4. **Structured assignment with hidden observability states**
- Keep a geometric assignment layer, but explicitly model “formed vs partially observed vs unobserved” as latent variables.
- Attractive if the core issue is still mostly one of matching plus visibility.

## What would be most valuable guidance
The most useful answer is not a generic “try Bayesian methods.” The valuable guidance would be:

1. What is the right formal state space?
- Should `PARTIALLY_OBSERVED` and `UNOBSERVED` be modeled separately from physical bend completion?
- Should observability be a separate latent variable from bend state?

2. What should the likelihood look like?
- How should local geometric evidence be converted into a principled likelihood term?
- What terms should be coupled, and which should remain conditionally independent?

3. What structure should couple bends together?
- Independent bend posteriors may be too weak.
- What graph/process prior would be mathematically justified without over-constraining the system?

4. How should sequence monotonicity be imposed?
- Hard constraint, soft prior, or posterior regularization?

5. What is the simplest defensible model that is likely to outperform heuristic threshold systems on real industrial staged scans?

## Why this matters
This is the point where additional heuristic tuning is unlikely to produce the required step change. The project now likely needs a principled latent-state model for bend completion under partial observability.

If solved well, this would likely improve:
- partial-scan progression accuracy
- stability / reproducibility
- bend-count MAE on hard families
- the scientific defensibility of the system architecture

## Bottom-line question
What is the mathematically right way to model per-bend completion state from partial, occluded point-cloud evidence against CAD, such that the model distinguishes “not formed” from “not observed” and remains stable on repeated similar bends?

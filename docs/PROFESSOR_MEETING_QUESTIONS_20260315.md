# Meeting Brief: Open Research Questions for Partial Bend-State Inference

## Purpose
This note is meant to stand on its own.
It summarizes the problem, what has already been established experimentally, what has failed so far, and the specific questions where guidance would materially affect the next implementation.

## Short context
We are building an industrial bend-inspection system for sheet-metal parts.

Inputs per case can include:
- a CAD model of the fully bent part,
- one or more point-cloud scans of the same part during production,
- technical drawings / spec sheets,
- and sometimes multiple partial scans from successive production stages.

The practical goal is not only final-part QA. It is also to infer progression on partially bent parts:
- how many bends are already completed,
- which bends are still missing,
- which completed bends are in or out of tolerance,
- and how to do this robustly when the scan is partial, occluded, or anisotropic.

The operating metric that matters most right now is bend-count MAE on real industrial scans, especially hard partial scans.

## Current empirical state
We already have a real regression corpus with:
- CAD files,
- partial and full point-cloud scans,
- technical drawings,
- some spec sheets,
- and partial truth labels.

We also already know the following experimentally:

1. `UNOBSERVED` must be treated as missing evidence, not evidence of absence.
This improved difficult staged families without harming the broader regression corpus.

2. Several intuitive fixes were tested and rejected because they regressed the broader benchmark or failed to solve the hard staged families:
- stricter one-sided evidence thresholds,
- geometry-weighted local assignment,
- global optimal assignment across dense local candidates.

3. The real failure mode is not “easy CAD extraction.”
The main remaining technical gap is local bend-state inference under partial observability.

## The central modeling issue
The current problem seems to come from conflating three different things:

- the **physical completion state** of a bend,
- the **observability / visibility** of that bend in the scan,
- and the **assignment** of local scan evidence to a specific CAD bend.

The current direction is therefore to separate these as:
- `C_i`: physical completion state,
- `rho_i`: continuous visibility / observability score,
- `A_i`: assignment of local evidence candidate or `null`,
- `S`: optional process-stage variable.

Then the user-facing labels are derived outputs rather than primary latent states:
- `FORMED`
- `NOT_FORMED`
- `PARTIALLY_OBSERVED`
- `UNOBSERVED`

## Why this is hard
On the difficult tray/channel families:
- several bends are geometrically similar,
- multiple bends can have nearly parallel bend lines,
- local scan support may only cover one flange or one side,
- global alignment can be good while still being locally uninformative,
- and weak local evidence can be claimed by the wrong bend.

So the challenge is not only detection.
It is to decide, for each CAD bend, whether the scan implies:
- physically formed,
- physically not formed,
- partially observed,
- or simply unobserved.

## Questions where guidance is most valuable

### 0. Foundational formulation question
Before choosing implementation details, there is one higher-level question:

If this problem were designed from first principles, which mathematical formulation family is actually the right one?

Candidate families we have considered:
- latent-state inference on a sparse graph of CAD bends,
- Bayesian inference with observability-conditioned likelihoods,
- structured assignment with hidden observation states,
- sequential filtering with monotone process priors,
- or a hybrid of the above.

Question:
Which family would you choose as the cleanest and most defensible starting point for this specific industrial problem, and why?

The practical issue is that several simpler heuristic variants have already failed, so the next model needs to be structurally right, not just locally improved.

### 1. Unary model choice for the first learned stage
For the first learned conditional unary model on the current corpus, what is the best starting point?

Options we are considering:
- calibrated logistic / ordinal model on engineered evidence features,
- gradient-boosted trees with post-hoc calibration,
- or a small neural model.

The constraints are:
- the current corpus is real but not enormous,
- calibration matters because these scores will drive structured inference,
- and interpretability is useful for debugging industrial failure cases.

Question:
What would you choose as the best first unary model, and why?

### 2. Visibility modeling: one global `rho_i` or family-specific?
We are currently moving toward a continuous visibility score `rho_i ∈ [0,1]` rather than a discrete latent visibility class.

That score will likely combine things such as:
- expected visible fraction from CAD-facing geometry,
- realized local point support,
- local density,
- side balance,
- and alignment stability.

Question:
Should the first calibrated visibility model be:
- one global model across all part families,
- or separate models for families such as trays/channels, rounded parts, and simpler brackets?

What would be the mathematically cleaner starting point given likely family-dependent visibility regimes?

### 3. Assignment structure: hard or soft exclusivity first?
For repeated similar bends, the core failure mode is that the same weak local evidence may be claimed by several candidate CAD bends.

We expect exclusivity to be the first essential structural coupling.

Question:
Would you start with:
- hard one-to-one candidate usage constraints,
- or soft exclusivity penalties from day one,
when local candidate patches may themselves be ambiguous or partially overlapping?

Related question:
At what point would you introduce shared-flange adjacency, and how weak should it be initially so it does not propagate local mistakes?

### 4. Count decoding: induced posterior over `K` or separate count model?
The practical metric we care about is bend-count MAE.
That means the total reported count matters as much as, or more than, the single best bend-wise configuration.

We are considering two directions:
- infer bend states and derive the posterior over total formed bends `K = Σ_i 1[C_i = FORMED]`,
- or train a separate count model / count head on top of the evidence.

Question:
For a principled first implementation, would you recommend:
- deriving `p(K | E)` from the bend-state model and decoding count from that posterior,
- or introducing a separate count model earlier?

If the answer is “derive it,” what approximation would you consider acceptable in the first version?

### 5. Monotonicity: count-only first, or bend-wise too?
For staged scans of the same part over time, progression should usually be monotone, but scans can also be noisy or under-observed.

The current thought is to add monotonicity first through posterior regularization rather than baking it directly into the state prior.

Question:
Would you start with:
- count monotonicity only,
- or bend-wise monotonicity as well for bends with high posterior mass in earlier scans?

What is the safer first step if the main risk is under-observation rather than true physical reversal?

### 6. Is a scalar `rho_i` enough at first?
The current recommendation is to start with a continuous calibrated visibility score rather than a discrete latent visibility state.

Question:
Do you think a scalar `rho_i` is enough for the first serious model,
or is it likely that visibility will be too multimodal and family-dependent for a single scalar to remain identifiable and useful?

If a scalar is acceptable initially, what failure signs would tell us it is time to promote visibility into a richer latent variable?

### 7. What is the right local likelihood structure?
For each CAD bend we can compute an engineered local evidence vector containing things like:
- local point support,
- side support,
- fitted local angle / radius,
- flange-normal consistency,
- bend-line direction consistency,
- residual / planarity,
- alignment quality,
- and visibility features.

Question:
What is the most defensible likelihood structure for the first model?

More concretely:
- should we treat the unary as a purely conditional model over engineered features,
- or is there a better structured likelihood for “visible surfaces under a bend hypothesis” that should be approximated even in v1?

The main concern is to avoid double-counting correlated geometric features while still using the evidence effectively.

### 8. What is the simplest model that is likely to beat the current heuristic regime?
If you had to recommend the smallest defensible model likely to outperform the current heuristic pipeline on hard partial scans, would it be something like:
- learned unaries over `(z_i, rho_i)`,
- explicit candidate-or-null assignment,
- confusable-bend exclusivity,
- weak adjacency,
- posterior-based count decoding,
- and posterior regularization for staged monotonicity?

Question:
Would you change that stack materially, or is that the right first serious model to build?

## Main meta-question
If you had to choose only one modeling mistake to fix first, which of these is the true bottleneck?

- weak local unary modeling,
- poor observability modeling,
- wrong assignment structure,
- count decoding mismatch,
- or monotonicity / temporal structure?

That answer would directly determine implementation order.

## Research-positioning question
There is also one research-framing question that matters for how we write this work up.

The project is not “point clouds + CAD” in the generic sense; those ingredients already exist in prior work.
The possible novelty seems to be the combination of:
- partial-process bend progression against finished CAD,
- observability-aware bend-state inference,
- structured candidate-or-null assignment under occlusion,
- and a real staged industrial regression corpus.

Question:
Is that a defensible and sufficiently narrow novelty claim for thesis-grade or publishable applied research, or is the claim still too broad / too close to existing structured-perception work?

Related question:
If you had to phrase one crisp research contribution that could be defended academically, what would you make it?

## Desired outcome from the meeting
The most useful result is not a full solution.
The most useful result is a clear recommendation on:

1. the best first learned unary model,
2. the right first visibility formulation,
3. whether exclusivity should be hard or soft in v1,
4. how to decode count in a way aligned with MAE,
5. and the safest first use of monotonicity in staged scans.

That would be enough to set the next research and implementation phase correctly.

# Professor Response — Mathematical Gap Analysis

Date: 2026-03-29
Status: active

## Purpose
This document records the professor's follow-up critique of the earlier response to the bend-system memo.

The correct interpretation is:
- the earlier response identified the right architectural fault lines
- but it did not answer the memo at the mathematical level the project now requires

That distinction matters. We should keep the system-level doctrine, but we should not pretend it is already a formal technical specification.

## Direct verdict
The criticism is valid.

The prior response was:
- directionally correct on ontology, datum policy, observability, and rolled-feature separation
- too weak on equations, loss functions, discriminators, and measurable thresholds

The missing layer is not "more detail" in the general sense. It is the mathematical layer that turns policy into implementable geometry and QC logic.

## What the prior response got right
1. Split the system into three truths:
   - target inventory truth
   - inspection nominal truth
   - observability truth
2. Retire whole-part global alignment as the metrology authority.
3. Separate rolled/process features from countable discrete bends.
4. Replace binary partial-state decisions with ternary observability-aware state.

These remain valid and should stay in the roadmap.

## What the prior response failed to provide
### 1. No formal discriminators for bend vs rolled vs transition
The answer used qualitative language such as:
- localized curvature band
- quasi-rigid adjacent regions
- distributed curvature

But it did not define:
- how curvature is estimated
- how localization is measured
- how rigidity / planarity is tested
- what threshold separates a discrete bend from a rolled/profile feature

### 2. No explicit observability model
The answer correctly introduced `FORMED / UNFORMED / UNKNOWN`, but it did not define:
- what evidence is necessary and sufficient for each state
- an observability score or decision rule
- how partial visibility translates into confidence

### 3. No registration objective
The answer recommended datum-island alignment over global ICP, but it did not specify:
- what transform is solved for
- the loss function
- the weighting of holes, planes, edges, or invariant surfaces
- the conditioning constraints that decide whether a transform is trustworthy

### 4. No metric decomposition for heatmap vs bend metrology
The answer treated disagreement between heatmap and bend verdicts as informative, which is correct, but did not decompose the geometry into:
- rigid pose error
- local frame error
- angle error
- radius error
- correspondence error

Without that decomposition, correlation expectations stay vague.

### 5. No quantitative tolerance semantics for rolled sections
The answer recommended centerline and section-profile validation, but it did not define:
- how the centerline is fit
- what section-profile mismatch is measured
- whether pass/fail is based on max, RMS, percentile, or worst-station criteria

## The corrected interpretation of the professor guidance
The project now needs a two-layer doctrine.

### Layer 1 — System doctrine
Keep the following as architectural decisions:
- split truths
- datum-island metrology
- rolled-feature separation
- ternary observability state

### Layer 2 — Mathematical specification
For each doctrine item, write the formal operational layer:
- variables
- transforms
- objective functions
- measurable thresholds
- abstention conditions
- pass/fail rules

Without Layer 2, the system remains definitionally unstable even if the architecture is correct.

## Immediate mathematical workstreams implied by the critique
### A. Formal target classification
Need a measurable discriminator between:
- discrete bend
- rolled/process feature
- non-bend transition

Minimum required mathematical content:
- normal-section model candidates
- curvature-support definition
- model-selection criterion
- confidence / abstention rule

### B. Formal datum-island registration
Need a registration objective for:
- holes / slots
- planes / invariant flats
- edges / notches

Minimum required mathematical content:
- transform family
- weighted objective
- conditioning score
- datum trust policy

### C. Formal local bend metrology
Need a section-based estimator for:
- included angle
- bend radius
- bend line location

Minimum required mathematical content:
- section-plane definition
- per-section fitting rules
- robust aggregation across sections
- observability threshold

### D. Formal rolled-feature validation
Need a separate rolled validator with:
- centerline representation
- section-profile deviation metric
- placement metric
- tolerance rule

### E. Formal observability state
Need explicit rules for:
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

Minimum required mathematical content:
- observability score or hard gating criteria
- visibility / coverage requirements
- multi-slice consistency rule

## Practical consequence for the project
The next project phase should not be framed as "more tuning".

It should be framed as:
- formalizing the geometry model
- formalizing the datum model
- formalizing observability

Only after that should we expect substantial progress on:
- hard partial parts
- rolled hybrids
- trustworthy local bend deltas

## Short conclusion
The earlier answer was good as a systems diagnosis and weak as a mathematical specification.

The professor's follow-up does not overturn the roadmap. It sharpens it:
- keep the architectural split
- add the formal geometry, loss, threshold, and abstention definitions that make the split operational

# Professor Response — Mathematical Specification

Date: 2026-03-29
Status: governing draft
Purpose: capture the professor's mathematically specified response as the current highest-quality external guidance for bend-system redesign.

## Executive assessment
This response is materially better than the earlier one.

Why:
- it defines variables
- it specifies competing models
- it proposes objective functions
- it provides thresholds
- it defines abstention conditions
- it distinguishes measurement primitives from semantic feature types

This is the first response that is close to an implementable mathematical specification rather than architecture-level advice.

## Core upgrade over prior guidance
The previous guidance correctly diagnosed the system-level problem.
This response provides the mathematical layer that was missing.

It resolves or materially improves these previously underdefined areas:
1. formal bend-vs-roll-vs-transition discrimination
2. datum-island registration objective and trust hierarchy
3. local bend frame definition
4. multi-slice angle/radius estimator
5. ternary observability with hard evidence gates
6. rolled-section representation and tolerance semantics
7. heatmap-vs-bend decomposition
8. industrial metric definition beyond count MAE

## Most important new principle
A nominal is not measurable until it is attached to:
1. a parent datum island
2. a measurement primitive

This is the most important mathematical clarification in the response.

## Key mathematical decisions to adopt

### 1. Feature classification must be model-based
Do not classify by line heuristics alone.

Use competing section models:
- discrete bend: line-arc-line
- rolled/profile: smooth distributed-curvature model
- non-bend: single-line / short local blend model

Use:
- robust per-slice fit error
- BIC-like model comparison
- curvature-support localization
- side rigidity / planarity tests

Important default thresholds proposed:
- strong model-selection margin: `Δ_BIC = 10`
- discrete bend localization ratio: `λ <= 0.35`
- rolled localization ratio: `λ >= 0.60`
- non-bend turning angle: `Φ < 5°`
- minimum real bend turning angle: `Φ >= 10°`

### 2. Datum-island registration must become explicit and scored
Use one weighted objective over allowed datum features on the chosen parent island:
- holes
- slots
- planes
- edges
- invariant surfaces

Important design consequences:
- solve one accepted island-local rigid transform `T*`
- compute conditioning from the information matrix
- reject or abstain when sources disagree materially
- surface refinement is subordinate, not primary

Key recommendation:
- trust a good hole/slot pattern on the correct rigid panel over generic surface fit
- do not fall back to unconstrained whole-part ICP and pretend local metrology is valid

### 3. Local bend metrology must be section-first
The local frame should be derived from:
- parent island frame
- transported nominal bend-line direction
- section planes orthogonal to the bend line

Angle and radius should be estimated from multiple section slices with robust aggregation.

Key consequence:
- the current system should move away from one-shot bend-line/global overlay logic for final local metrology
- multi-slice estimation becomes the canonical path

### 4. Partial scans require explicit abstention
The ternary state is now mathematically justified, not just policy justified:
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

Important point:
- `UNFORMED` is the harder claim
- missing evidence is not negative evidence
- partial scans should default to `UNKNOWN` unless visibility gates are satisfied

### 5. Rolled sections need their own representation and validator
Rolled features should be represented by:
- centerline `c(u)`
- radius field `r(u)`
- arc extent field `α(u)`
- transition boundaries `b1(u), b2(u)`

Validation should use:
- centerline deviation
- section-profile deviation
- percentile + max guard pass/fail semantics

Important point:
- global heatmap is only coarse triage, not authoritative rolled validation

### 6. Heatmap should not be treated as a generic bend validator
The response makes this precise.

Residuals should be decomposed into:
- rigid pose error
- local frame error
- angle error
- radius error
- correspondence error

Most important recommendation:
- replace generic bend-vs-heatmap comparison with a bend-mode projected residual where possible
- keep whole-part heatmap as triage and visualization, not as a primary local metrology authority

### 7. Count MAE is not enough as a success metric
The mathematically preferred industrial metric is:
- operator-review save rate at a fixed false-accept ceiling

This should become a top-line metric once the target-level states and calibrated scan-level risk exist.

## What this means for the current system
The current system is now clearly under-specified in five places:
1. feature ontology is only partially mathematical
2. datum selection/alignment is not yet formalized as the metrology authority
3. local bend measurement is not yet fully section-first and confidence-driven
4. observability is not yet a formal gated estimator
5. rolled sections are not yet handled by their own mathematical validator path

## Immediate implications for implementation order
The professor explicitly chose the first mathematical model to write:
1. **datum-island registration**

This should now drive the order of implementation.

Recommended immediate order:
1. write datum-island registration mathematical spec into project docs
2. implement a registration module that exposes:
   - winning transform
   - feature-wise residual breakdown
   - transform covariance / confidence
   - anchor-source winner
   - abstain flag
3. use that as the parent frame for:
   - multi-slice local bend metrology
   - observability gating
4. only after that push harder on rolled validator and target-label upgrades

## Relationship to current benchmark baseline
Reference benchmark:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`

Current stable baseline:
- heuristic MAE vs expected total completed bends: `1.909`
- structured MAE vs expected total completed bends: `1.727`
- heuristic MAE vs expected completed bends: `0.571`
- structured MAE vs expected completed bends: `0.333`

Interpretation:
- these numbers are now trustworthy enough to serve as a baseline
- but the next improvements should be judged against mathematically specified modules, not free-form heuristic tuning

## Highest-value immediate actions
### Action 1
Write `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`
using the professor's registration objective, conditioning rules, and trust hierarchy.

### Action 2
Write `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_LOCAL_BEND_METROLOGY.md`
using the professor's parent-island frame and multi-slice estimator.

### Action 3
Write `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_OBSERVABILITY_MODEL.md`
using the professor's explicit `FORMED / UNFORMED / UNKNOWN` gating and evidence model.

### Action 4
Write `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MATH_SPEC_ROLLED_SECTION_VALIDATION.md`
using the centerline/profile semantics and percentile + max guard pass/fail rule.

## Short conclusion
This response should be treated as the current best external technical guidance.

It is not just useful feedback.
It materially changes how the next phase should be executed:
- specification first
- implementation second
- benchmarked calibration third

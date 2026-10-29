# Professor Follow-Up — Mathematical Questions

Date: 2026-03-29
Purpose: obtain the mathematical layer that is still missing after the first response.

## Why this follow-up matters
The first response was useful at the architectural level, but not yet at the mathematical specification level.

The project now needs explicit answers for:
- definitions
- transforms
- loss functions
- thresholds
- abstention rules
- pass/fail semantics

This follow-up should not revisit the high-level doctrine unless the doctrine itself is mathematically wrong.

## Requested answer style
For each question, please answer in this structure when possible:
1. **Variables / objects**
2. **Formal definition or objective**
3. **Decision rule / threshold**
4. **Abstention condition**
5. **Failure modes / counterexamples**

Equations or pseudo-equations are preferred over prose where possible.

---

## Question 1 — Formal discriminator: discrete bend vs rolled feature vs non-bend transition
We need a mathematically testable discriminator.

Please specify:
- how to define the local section geometry around a candidate
- what candidate models should compete
  - e.g. line-arc-line vs distributed-curvature spline vs local blend model
- what fit-error criterion should choose between them
- how to define curvature support localization quantitatively
- how to define the planarity/rigidity of the adjacent regions quantitatively
- what threshold or margin should decide the class
- when the classifier should abstain instead of forcing one of the classes

Expected output:
- candidate models
- fit objective for each model
- model-selection rule
- thresholded decision rule

## Question 2 — Datum-island registration objective
We need a formal objective for local metrology alignment.

Please specify:
- what transform family should be solved for in the normal case
- how holes, slots, planes, edges, and invariant surfaces should each contribute to the loss
- a recommended weighted loss function
- what conditioning constraints make an alignment trustworthy
- what trust rule should decide when holes outrank planes, when planes outrank surface refinement, and when to abstain
- whether the same objective should be used for:
  - full finished parts
  - partial-stage parts
  - rolled hybrids

Expected output:
- explicit objective function `L(T)`
- feature-specific residual terms
- conditioning score / acceptability rule
- family-specific routing notes

## Question 3 — Local bend frame definition
We need the formal local frame in which bend angle and radius are measured.

Please specify:
- how the parent rigid island frame is defined
- how the bend-line direction is transported into scan space
- how section planes are defined from that frame
- what minimum geometry is required for the frame to be valid
- when the local frame should be rejected as unreliable

Expected output:
- frame construction equations or pseudo-equations
- validity conditions
- rejection criteria

## Question 4 — Multi-slice bend angle estimator
We need a formal estimator, not just “measure in multiple sections.”

Please specify:
- how section slices should be spaced and selected
- how each slice should be fit
  - line fits
  - arc fit
  - or other
- how included angle should be defined from a slice
- how radius should be defined from a slice
- what robust aggregation should combine the slices
- what variance / consistency rule should cause abstention

Expected output:
- per-slice estimator
- robust aggregation rule
- confidence / abstention rule

## Question 5 — Observability model for partial scans
We need a mathematically grounded rule for `FORMED / UNFORMED / UNKNOWN`.

Please specify:
- what evidence variables should contribute
  - e.g. local frame confidence, two-side visibility, curvature-band visibility, slice consistency, contextual localization
- whether you recommend:
  - a scalar observability score, or
  - hard gating rules
- what the minimum sufficient evidence is for `FORMED`
- what the minimum sufficient evidence is for `UNFORMED`
- when the correct output must be `UNKNOWN`
- whether different rules should apply for full vs partial scans

Expected output:
- formal observability score or hard rule set
- thresholded state mapping
- explicit abstention rule

## Question 6 — Rolled-section representation
We need a mathematical representation suitable for QC, not only a semantic description.

Please specify:
- how to define the rolled feature centerline
- how to define local section frames along the centerline
- whether radius should be constant, piecewise constant, or a field `r(s)`
- how arc extent should be represented
- how transition boundaries to adjacent flats/flanges should be defined
- whether a simpler surrogate exists that is good enough for industrial use

Expected output:
- rolled feature parameterization
- recommended minimal representation for QC
- recommended richer representation if needed

## Question 7 — Rolled-section tolerance semantics
We need a formal pass/fail rule.

Please specify:
- how centerline deviation should be measured
- how section-profile deviation should be measured
- whether pass/fail should use:
  - max
  - RMS
  - median
  - percentile
  - worst-station
- what combination rule should define overall rolled-feature pass/fail
- when global heatmap is acceptable as a proxy and when it is not

Expected output:
- formal rolled validation metrics
- station aggregation rule
- pass/fail criterion

## Question 8 — Expected-angle truth source when CAD, drawing, and process data disagree
We need an operational rule, not just “use released nominal.”

Please specify:
- a precedence rule between CAD nominal, drawing nominal, and process/tooling angle
- how to detect an ambiguity case automatically
- what to do when CAD and drawing disagree materially
- whether there are families where angle should not be the truth primitive at all

Expected output:
- precedence table
- ambiguity trigger
- abstention / escalation rule

## Question 9 — Heatmap vs bend-metric decomposition
We need a mathematical explanation of when these should and should not correlate.

Please specify:
- a decomposition of local discrepancy into at least:
  - rigid pose error
  - local frame error
  - angle error
  - radius error
  - correspondence error
- which terms should produce strong heatmap correlation
- which terms can produce local bend error with low heatmap energy
- whether there is a useful local residual statistic that should correlate with true bend error better than the current heatmap ROI

Expected output:
- decomposition
- expected correlation table by error source
- recommended local residual metric

## Question 10 — Hole/slot anchors as precision datums
We need a mathematically grounded trust rule.

Please specify:
- minimum anchor count
- non-collinearity / conditioning requirement
- minimum spatial spread requirement
- feature quality threshold for accepting a hole/slot fit
- how to score trust when hole-based and surface-based registration disagree

Expected output:
- conditioning rule
- trust score or trust hierarchy
- conflict-resolution rule

## Question 11 — Partial-stage invariant geometry selection
We need a formal method, not only examples.

Please specify:
- how to determine whether a region is invariant across stages
- what geometric tests should disqualify a region as a datum for partial alignment
- whether there is a simple graph or segmentation model you recommend for “rigid islands” on partially formed parts

Expected output:
- invariant-region criteria
- disqualification criteria
- suggested graph/segmentation formulation if useful

## Question 12 — One-level-better metric than count MAE
We need a mathematically defined industrial metric.

Please specify:
- a formal definition of operator-review save rate at a fixed false-accept ceiling
- what should count as an auto-dispositioned correct case
- what calibration or confidence requirement must hold before a case is counted as saved
- whether this should be defined per target, per scan, or per work order

Expected output:
- metric definition
- denominator / numerator definition
- decision thresholding rule

## Question 13 — If you had to choose only one mathematical model to write first
If implementation time is constrained, which single mathematical specification should be written first because it will unlock the most progress?

Please choose exactly one from:
- feature classification
- datum-island registration
- local bend metrology
- rolled-section validation
- observability model

Expected output:
- choice
- short justification
- what downstream problems it would unlock first

## Question 14 — If our current framing is still mathematically wrong
If you think the whole split into:
- target inventory truth
- inspection nominal truth
- observability truth

is still insufficient or mathematically misguided, please state the corrected decomposition explicitly.

Expected output:
- corrected decomposition
- why the current one is insufficient
- what implementation consequence changes immediately

## Question 15 — Minimal viable mathematical spec for the next 2 weeks
If you were constraining us to the smallest mathematically serious next step, what should we implement in the next two weeks?

Expected output:
- one short prioritized list
- each item with:
  - mathematical object to define
  - engineering artifact to produce
  - what benchmark slice should prove it worked

## Closing note
The goal of this follow-up is not broader advice. It is to get formal decision rules that can be implemented directly and benchmarked against:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`

Current baseline from that benchmark:
- heuristic MAE vs expected total completed bends: `1.909`
- structured MAE vs expected total completed bends: `1.727`
- heuristic MAE vs expected completed bends: `0.571`
- structured MAE vs expected completed bends: `0.333`

# Implementation Plan From Professor Response — 2026-03-29

## Goal
Convert the professor's response into concrete project changes.

This is not a brainstorming list. It is the next architectural and algorithmic roadmap.

## Priority order

### Priority 0 — Freeze doctrine in writing
Create or update ADR-style documentation for:
1. split-truth architecture
2. datum-island alignment policy
3. ternary observability state
4. rolled-feature family policy
5. expected-angle truth-source policy

## Priority 1 — Change the ontology and contracts
### 1.1 Add explicit feature families
Every CAD target should be classified as one of:
- `COUNTABLE_DISCRETE_BEND`
- `ROLLED_PROCESS_FEATURE`
- `NON_BEND_TRANSITION`
- optionally later:
  - `HEM_FAMILY`
  - `JOGGLE_FAMILY`
  - `BEAD_FAMILY`

### 1.2 Split truth in the data model
Add explicit separation between:
- `target_inventory_truth`
- `inspection_nominal_truth`
- `observability_truth`

### 1.3 Replace binary partial-state with ternary state
Per target, per scan:
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

This change should hit:
- backend result contracts
- evaluator state handling
- structured count model inputs
- UI chips / tables / overlays

## Priority 2 — Change alignment architecture
### 2.1 Introduce datum-island alignment
Stop treating one whole-part transform as the metrology truth.

New architecture:
1. choose a master rigid island
2. solve alignment on that island
3. define local bend frames relative to that island
4. use whole-part overlay only as a visualization layer

### 2.2 Add alignment hierarchy by scan family
Implement routing by:
- full finished
- partial stage
- rolled hybrid

With explicit precedence for:
- holes/slots on correct rigid island
- invariant flats
- restricted surface refinement

### 2.3 De-emphasize global ICP
Keep global ICP for:
- visualization
- coarse initialization

Do not let it remain the authoritative local metrology frame.

## Priority 3 — Replace local bend comparison method on hard families
### 3.1 Section-first local measurement
For discrete bends, especially hard ones:
- anchor on parent rigid panel
- build section planes orthogonal to bend line
- fit lines/arcs in several slices
- aggregate by robust statistic

### 3.2 Use hybrid selector rules from section topology
Primary discriminator:
- if local section does not decompose into two rigid sides plus localized transition band, do not call it a discrete bend

### 3.3 Stop allowing shell-profile surrogates to stand in for flange bends
This is especially important on rolled-shell + flange parts.

## Priority 4 — Split rolled sections into their own pipeline branch
### 4.1 Representation
Model rolled features by:
- centerline
- local section frames
- curvature/radius field
- arc extent field
- transition boundaries

### 4.2 Validation
Use:
- primary: section-profile deviation
- secondary: centerline deviation
- not single bend-angle semantics

### 4.3 Progress/counting
Do not count rolled sections as discrete bend targets in bend-count MAE.
Keep them in a separate process-feature channel.

## Priority 5 — Fix expected-angle truth source
### 5.1 Policy
Expected bend angle should come from released inspection nominal:
- CAD+PMI or drawing master
- not tooling/process-sheet angle

### 5.2 Exception handling
For families like:
- hems
- rolled sections
- closure-driven features

Do not force ordinary bend-angle semantics.

## Priority 6 — Data collection policy
### 6.1 Highest-value new label
Add per-target labels:
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

### 6.2 Second-best label
Invariant datum regions for partial parts.

### 6.3 Evaluation upgrade
Move from count-only emphasis toward:
- exact completed-bend-set accuracy on observable targets
- local angle/radius error on high-observability targets
- operator-review save rate at a fixed false-accept ceiling

## Suggested work packages

### Work package A — Contracts and evaluator
- add ternary state
- add target family types
- update evaluator for observable-target metrics
- preserve old MAE for continuity

### Work package B — Datum-island alignment MVP
- add hole/slot + invariant-flat alignment mode
- full-part transform becomes visualization-only
- compare against current baseline on stable families first

### Work package C — Section-first bend metrology MVP
- implement multi-slice local measurement
- first on standard folded parts
- then on hybrid flange bends

### Work package D — Rolled feature branch
- represent and validate rolls separately
- remove them from discrete bend target competition

### Work package E — New label schema and intake
- add per-bend `FORMED/UNFORMED/UNKNOWN`
- add optional invariant datum masks
- backfill on the highest-value existing parts first

## Immediate next action recommendation
If starting tomorrow, the correct first implementation sequence is:
1. freeze docs/ADRs for the new doctrine
2. add ternary observability state to contracts and evaluator
3. build datum-island alignment MVP
4. only then revise local bend metrology

## What not to do next
- do not spend the next cycle polishing global ICP
- do not keep treating rolled sections as just hard bends
- do not keep forcing partial scans into formed/not formed when evidence is weak
- do not optimize only count MAE and call that sufficient


## Mathematical-specification correction
The follow-up critique on 2026-03-29 is accepted.

This plan remains directionally correct, but it is not sufficient by itself. Before major algorithm rewrites, the project now needs a formal mathematical specification for the same decisions.

That means adding explicit workstreams for:
1. bend-vs-rolled model selection in normal sections
2. datum-island registration objectives and conditioning rules
3. multi-slice local bend angle / radius estimation
4. rolled centerline + section-profile validation
5. observability scoring or hard-gated `FORMED / UNFORMED / UNKNOWN`

The correct interpretation is:
- this implementation plan defines *what* the system should become
- the mathematical specification defines *how* each decision is computed and when it must abstain

## Revised first implementation order from mathematical response
1. datum-island registration specification and MVP
2. local bend-frame + multi-slice estimator specification and MVP
3. observability model specification and ternary-state integration
4. rolled-section representation and validator specification
5. only then broader algorithm retuning against the stable benchmark

Reason:
- the professor explicitly ranked datum-island registration as the single highest-leverage mathematical model to write first
- the downstream metrology, observability, and heatmap interpretation all depend on it

## New immediate priority before major algorithm tuning
### Priority 0.5 — Write mathematical specification documents
Create and keep separate, traceable specs for:
1. bend / rolled / transition discriminator
2. datum-island alignment objective
3. local bend metrology estimator
4. rolled-section validator
5. observability model and abstention rules

Without these, tuning will continue to mix ontology, datum, and measurement errors together.

## Execution status — tranche 1 completed
Completed on 2026-03-29:
- ternary observability semantics introduced:
  - `FORMED`
  - `UNFORMED`
  - `UNKNOWN`
- legacy detail observability retained separately:
  - `OBSERVED_FORMED`
  - `OBSERVED_NOT_FORMED`
  - `PARTIALLY_OBSERVED`
  - `UNOBSERVED`
- explicit runtime metadata added:
  - `feature_family`
  - `measurement_primitive`
  - `observability_detail_state`
- structured count logic changed so `UNKNOWN` is neutral rather than implicit negative evidence

Trace file:
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/MAE_EXECUTION_PROGRESS_20260329.md`

Reference benchmark before tranche 1:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`

Reference benchmark after tranche 1:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260329_observability_v1b/summary.json`

Observed result:
- structured total MAE vs expected total completed bends:
  - `1.727 -> 1.636`
- structured MAE vs expected completed bends:
  - `0.333 -> 0.286`
- structured partial total MAE:
  - `3.6 -> 3.4`
- structured full exact completion rate:
  - unchanged at `0.895`

Interpretation:
- tranche 1 is worth keeping
- it delivered the minimum targeted ROI by reducing partial error without hurting full exact completion
- next ROI should come from datum-island alignment and parent-frame local metrology, not from relaxing the new `UNKNOWN` behavior

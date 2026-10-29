# Professor Response Synthesis — 2026-03-29

## Status
This document captures the substantive decisions implied by the professor's response to the bend-system memo.

The answer is not just feedback. It changes the system architecture.

## Highest-level diagnosis
The current system is trying to do three different jobs with one ontology and one transform.

The professor's recommendation is to separate:
1. **Target inventory truth**
   - what features exist
   - which of them are countable bend targets
2. **Inspection nominal truth**
   - what geometry the scan is supposed to match
3. **Observability truth**
   - what this particular scan can support confidently right now

This is now the most important framing decision in the project.

## Three assumptions to retire
The response explicitly rejects these assumptions:
1. a globally good overlay is sufficient for local bend metrology
2. final geometry alone can define what counts as a bend target
3. partial scans should force a binary formed / not formed decision

## Core adopted decisions

### 1. Canonical bend ontology
- `countable discrete bend`
  - localized reorientation between two quasi-rigid adjacent regions
  - section looks like two line segments joined by one arc
  - corresponds to one intentional discrete forming action
- `rolled/process feature`
  - distributed curvature without one uniquely localizable hinge line or one included angle
  - better represented by centerline + section/profile field
- `non-bend transition`
  - blends, fillets, shell offsets, reliefs, tangent changes, modeling/detail transitions

### 2. Truth basis split
- for **target inventory and progress counting**:
  - manufacturing operation wins
- for **final acceptance geometry**:
  - released inspection nominal wins
- process-sheet/tooling angles are not inspection truth

### 3. Alignment doctrine
Use **datum-island alignment**, not one whole-part transform as the metrology truth.

Hierarchy:
- full finished parts:
  - primary: explicit datum features on the correct rigid panel
  - secondary: base panel plane and stable trimmed edges
  - tertiary: restricted surface refinement on invariant regions only
- partial-stage parts:
  - primary: largest invariant flat region plus holes/slots on it
  - secondary: stable edges/notches on that same region
  - tertiary: already completed rigid panels tied back to the invariant region
- rolled hybrids:
  - primary: discrete datums on non-rolled flanges/base panels
  - secondary: stable adjacent flange planes and edges
  - tertiary: roll centerline / sampled roll sections for validation or light refinement only

Explicit recommendation:
- remove unconstrained global ICP from the metrology critical path first

### 4. Hole/slot policy
For finished parts, hole/slot-based anchoring is valid and often preferable if:
- anchors are on the correct rigid panel
- non-collinear and sufficiently spread
- clean enough to fit robustly

Rule of thumb:
- one flat region plus one in-plane anchor is enough
- one flat region alone is usually not enough

### 5. Partial-state policy
Adopt a ternary state:
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

Explicit policy:
- missing evidence is not negative evidence
- unknown is the default on partial scans unless observability clears a threshold

### 6. Rolled-feature policy
Rolled sections are:
- not countable discrete bends
- both a process family and a geometry family
- best represented by:
  - centerline
  - radius/curvature field along centerline
  - arc extent field
  - local section shape
  - transition boundaries

Recommended validation:
- primary metric: local section-profile deviation after centerline registration
- secondary metric: centerline deviation
- global heatmap is not enough except for coarse envelope screening

### 7. Bend-angle truth source
Default rule:
- inspect against the released final nominal
- use CAD+PMI or drawing, whichever is the released inspection master
- do not inspect against tooling angle

If final CAD and scan agree globally but a local bend angle appears wrong, suspect first:
1. wrong local frame
2. wrong target definition
3. wrong expected-angle source
4. actual manufacturing defect

### 8. Local bend measurement doctrine
Preferred local frame:
- anchor on the stable parent panel
- transport CAD bend-line direction through that anchored transform
- measure in multiple section slices orthogonal to the bend line
- aggregate per-section angle/radius robustly, usually by median

### 9. Hybrid selector doctrine
Primary discriminator on rolled hybrids:
- **local section topology first**
- line heuristics second

Decisive rule:
- if a normal section does not decompose into two rigid sides plus one localized transition band, it is not a countable discrete bend

### 10. Heatmap doctrine
Heatmap and bend metrology are not the same measurement.

Heatmap agreement may:
- support a review decision
- indicate local-frame or nominal-truth problems
- flag contradictions worth investigation

Heatmap should not automatically overturn a bend failure.

### 11. Highest-value new label
The professor's highest-value label recommendation is effectively:
- **per-bend state with unknown**

For each scan and each CAD bend target:
- `FORMED`
- `UNFORMED`
- `UNKNOWN`

This is higher-value than more count-only labels.

### 12. Preferred industrial metric
The response recommends a metric closer to:
- operator review save rate at a fixed false-accept ceiling

Not raw count MAE alone.

### 13. Forced ranking of blockers
The ranking implied by the response is:
1. target definition
2. alignment / datum policy
3. partial-scan observability
4. local bend-frame definition
5. rolled-section representation
6. lack of labeled truth
7. benchmark design

## Direct consequences for the project
These are no longer optional ideas. They are the correct new project direction.

1. Freeze a split-truth architecture.
2. Replace whole-part metrology dependence with datum-island metrology.
3. Split rolls into a separate feature family.
4. Replace binary partial-state decisions with ternary observability-aware state.
5. Promote section-based local measurement over one-shot 3D bend-line fitting on hard families.

## One-sentence synthesis
The authoritative local comparison should be the released inspection nominal, expressed in the frame of a trusted rigid datum island, and any target that cannot be observed in that frame should be marked unknown instead of guessed.

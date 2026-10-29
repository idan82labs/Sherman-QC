# Professor Response — Mathematical Worklist

Date: 2026-03-29
Status: active
Purpose: translate the professor's architectural guidance and mathematical critique into formal specification tasks.

## Principle
The prior response answered mostly at the policy layer.

The project now needs the computation layer for the same decisions:
- variables
- transforms
- objectives
- thresholds
- abstention rules
- pass/fail semantics

## Workstream 1 — Formal feature ontology
### Question to formalize
When is a candidate a discrete bend, a rolled/process feature, or a non-bend transition?

### Required mathematical content
1. Section extraction model
   - define section planes normal to candidate line direction
2. Competing local models
   - discrete bend: line-arc-line or two-line-plus-transition model
   - rolled/profile: smooth distributed-curvature model
   - non-bend transition: small blend / fillet / offset transition model
3. Model-selection criterion
   - compare fit errors with explicit margin rule
4. Curvature-support localization metric
   - define band width / support length and threshold
5. Planarity or rigidity metric on adjacent regions
6. Abstention rule
   - if evidence is weak or models are not separable, return `UNKNOWN_FEATURE_TYPE`

### Deliverable
- `docs/research/MATH_SPEC_FEATURE_CLASSIFICATION.md`

## Workstream 2 — Datum-island registration objective
### Question to formalize
What transform is the authoritative local frame for metrology on full, partial, and rolled-hybrid scans?

### Required mathematical content
1. Candidate transform family
   - rigid SE(3) minimum
   - optional constrained rigid solve per datum island
2. Feature losses
   - hole/slot center residual
   - plane normal and offset residual
   - edge/notch residual
3. Weighted objective
   - one explicit loss function combining allowed datum terms
4. Conditioning score
   - rank / spread / non-collinearity / coverage metrics
5. Datum trust rule
   - when holes outrank planes, when invariant flats are sufficient, when transform must abstain
6. Family-specific routing
   - finished
   - partial
   - rolled hybrid

### Deliverable
- `docs/research/MATH_SPEC_DATUM_ISLAND_ALIGNMENT.md`

## Workstream 3 — Local discrete-bend metrology
### Question to formalize
How is bend angle, radius, and location measured in a trusted local frame?

### Required mathematical content
1. Parent-island anchored local frame definition
2. Section-plane construction orthogonal to bend line
3. Per-section fitting
   - side line fits
   - transition arc fit
4. Per-section outputs
   - angle
   - radius
   - line location / section quality
5. Robust aggregation
   - median / trimmed statistic across sections
6. Observability gates
   - minimum section count
   - minimum face visibility
   - consistency threshold across sections
7. Abstention rule for weak or one-sided evidence

### Deliverable
- `docs/research/MATH_SPEC_LOCAL_BEND_METROLOGY.md`

## Workstream 4 — Rolled-feature representation and validation
### Question to formalize
How should a rolled section be represented and inspected without forcing bend-angle semantics onto it?

### Required mathematical content
1. Centerline definition
2. Local section frame along centerline parameter `s`
3. Section-profile mismatch metric
4. Centerline placement metric
5. Aggregation rule along stations
   - max
   - percentile
   - worst-station
6. Pass/fail semantics
   - both placement and profile must pass
7. Relation to global heatmap
   - coarse envelope only vs local validator authority

### Deliverable
- `docs/research/MATH_SPEC_ROLLED_SECTION_VALIDATION.md`

## Workstream 5 — Observability model
### Question to formalize
When is a target `FORMED`, `UNFORMED`, or `UNKNOWN` on a partial or noisy scan?

### Required mathematical content
1. Evidence variables
   - local frame confidence
   - section consistency
   - two-side visibility
   - curvature-band visibility
   - contextual localization confidence
2. Either:
   - an explicit observability score, or
   - hard gating criteria with mandatory conditions
3. State decision rule
   - `FORMED`
   - `UNFORMED`
   - `UNKNOWN`
4. Special rules for partial scans
   - missing evidence is not negative evidence
5. Calibration targets
   - false-formed ceiling on partial scans

### Deliverable
- `docs/research/MATH_SPEC_OBSERVABILITY_MODEL.md`

## Workstream 6 — Expected-angle truth-source doctrine
### Question to formalize
What is the inspection truth when CAD, drawing, and process-sheet/tooling angles differ?

### Required mathematical content
1. Truth-source precedence table
2. Explicit exclusion of tooling angle from inspection nominal
3. Exception families
   - rolled sections
   - hems
   - closure-driven features
4. Consistency checks between CAD nominal and drawing nominal
5. Fallback / abstention when inspection nominal is ambiguous

### Deliverable
- `docs/research/MATH_SPEC_EXPECTED_ANGLE_TRUTH.md`

## Workstream 7 — Heatmap vs bend-metric decomposition
### Question to formalize
When should local bend deviation and heatmap residual correlate, and what failure mode does disagreement imply?

### Required mathematical content
1. Error decomposition
   - rigid pose error
   - local frame error
   - angle error
   - radius error
   - correspondence error
2. Local ROI definition in bend frame
3. Expected correlation statements by error type
4. Advisory-only consistency rules
5. Cases where disagreement should lower trust rather than override verdict

### Deliverable
- `docs/research/MATH_SPEC_HEATMAP_BEND_CORRELATION.md`

## Workstream 8 — Label and benchmark upgrade
### Question to formalize
What new truth is most valuable for learning and evaluation?

### Required mathematical content
1. Per-target labels
   - `FORMED`
   - `UNFORMED`
   - `UNKNOWN`
2. Observable-target exact-set metrics
3. Local angle/radius error metrics on high-observability targets
4. Industrial metric
   - operator-review save rate at fixed false-accept ceiling
5. Benchmark slices
   - full vs partial
   - family type
   - observability level

### Deliverable
- `docs/research/MATH_SPEC_LABELS_AND_BENCHMARKS.md`

## Immediate project sequence implied by this worklist
1. Freeze the system doctrine already accepted.
2. Write the mathematical specs above.
3. Only then revise the live algorithm against those specs.
4. Use the stabilized core benchmark as the reference baseline while doing so.

## Current baseline to preserve during this phase
Reference benchmark:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`

Current core metrics:
- heuristic MAE vs expected total completed bends: `1.909`
- structured MAE vs expected total completed bends: `1.727`
- heuristic MAE vs expected completed bends: `0.571`
- structured MAE vs expected completed bends: `0.333`

## Short conclusion
The roadmap is still correct, but it is incomplete until these mathematical specs exist.

The immediate job is no longer just "improve the algorithm". It is:
- define the geometry mathematically
- define the datum mathematically
- define observability mathematically
- then implement against those definitions

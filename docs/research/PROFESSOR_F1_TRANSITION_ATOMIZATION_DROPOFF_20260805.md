# Sherman QC F1 transition-side atomization follow-up

You are the GPT-5.6 Pro extended-thinking research reviewer for Sherman QC F1.
Continue from `02_PRIOR_PROFESSOR_RESPONSE.md`; do not restart the architecture
discussion and do not treat this as an email. Inspect every uploaded file.

This is a research-only, scan-only zero-shot bend-inference lane. Runtime/main
promotion remains prohibited.

## What your previous plan established

The implementation now has a deterministic typed union, compact atom and plane
tables, completed incidence, unsupported-path prohibition, normalized atom-mass
ownership, two-sided bend minima, duplicate and panel-boundary constraints,
proof-carrying continuations, exact deterministic CP-SAT stages, per-atom
owners, and bounded `BIRTH_V1` integration. The detector uses one deterministic
working cloud capped at 200,000 points; caches serialize no source points; all
measured jobs remain below 16 GiB peak RSS.

Those mechanics work. The representation does not.

## Frozen A0 evidence

- Gold37: 37 scans, 27 part-family keys, 182 gold bend lines.
- Final output: 75 lines, 42 TP, 33 FP, 140 FN; precision 0.560, recall
  0.231, line F1 0.327, exact count 1/37.
- Authoritative 13: 15 TP, 17 FP, 56 FN; precision 0.469, recall 0.211,
  line F1 0.291, exact count 1/13.
- All incidence-complete eligible paths: 112/182 gold lines represented
  (61.5% candidate recall) but 2,078/2,190 are false candidates.
- Therefore 70 final false negatives are representation failures and another
  70 represented gold objects lose semantic ownership/selection.
- `BIRTH_V1`: zero eligible, negative-lower-bound columns on all 37 scans.
- The instrumented `BIRTH_V1` stage census localizes the collapse before MDL:
  861,688 atom/plane-pair trials produce 8,571 atoms after transverse gating,
  2,960 after normal gating, and 390 after atom-axis gating. Those form 312
  connected components; 311 fail the six-atom minimum. The only survivor
  reaches the axial-span stage and then fails occupied-bin coverage. Zero
  components reach the MDL lower-bound stage.
- Post-hoc two-sided occupied-bin minima do not improve F1. At three bins FP
  falls 45.5%, below the prior 50% GO threshold, while recall falls from 23.1%
  to 15.9%, far beyond the five-point allowance.
- All 37 six-view sheets were inspected. Alignment is correct. False outputs
  are chiefly scan-supported exterior/frame/perimeter aliases; misses are
  chiefly repeated parallel lanes, interior transitions, short returns, and
  localized bends. No gross floating or hole-centred hallucination explains
  the aggregate failure.
- 37/37 exact jobs are OPTIMAL, 10/10 replay byte-identical, solver P95 is
  14.78 s, and maximum reported child-solver RSS is 1.66 GB.

The previous GO/STOP rules therefore falsify the current straight two-sided
birth and side-minimum representation. Do not recommend threshold sweeps,
expected-count logic, part-specific routing, filename/identity features, or a
larger learned selector on top of the unchanged atoms.

## Current atomization that must be audited

`13_patch_graph_latent_decomposition.py::build_surface_atom_graph()` creates a
single cubic voxel grid at:

```text
voxel_size_mm = max(3 * local_spacing_mm, 2.5 mm)
```

Each occupied voxel becomes one atom containing one centroid, one mean normal,
one dominant covariance axis, one scalar curvature proxy, weight, and area
proxy. Its point indices never leave the transient builder. The compact pool
serializes only those statistics and sparse voxel-neighbour edges.

The stage census in `07_BIRTH_V1_STAGE_CENSUS.json` shows that this is not an
MDL-cost failure: almost all geometrically surviving atoms become isolated or
sub-six-atom components, and the only size-qualified component fails axial-bin
coverage. This may erase the exact distinction the model needs: two panel-side normal
modes can occupy one voxel near a sharp transition and collapse to one averaged
normal. Conversely, panel-interior and exposed-boundary support can be
geometrically nearby yet semantically different. `BIRTH_V1` subsequently
assigns every residual atom exclusively to the nearer of two planes and then
requires at least three axial bins containing atoms from both sides. The zero
proposal result may therefore be caused by atom construction, plane coverage,
residual ownership, birth eligibility, or an inconsistent combination.

## Required response

Give an implementation-level continuation, not a broad survey. Clearly label
claims as OBSERVED, INFERENCE, or UNKNOWN and answer these questions in order.

### 1. Localize the zero-birth failure before redesign

From `13` through `17`, give a deterministic diagnostic cascade that assigns
every gold miss, for research analysis only, to the first failed stage:

1. neither incident plane mode exists;
2. no transition-side atoms survive;
3. side modes exist but are merged into one atom;
4. atom-graph connectivity breaks the axial run;
5. support is non-residual after A0 ownership;
6. BIRTH_V1 geometry or lower-bound eligibility rejects it;
7. an admissible candidate exists but MAP selection rejects it.

Specify the exact counters and bounded intermediate fields to record without
serializing source points. Say which diagnostics may use gold only after the
blind prediction is frozen, and which must remain inference-safe.

### 2. Specify `TRANSITION_ATOM_V2`

Provide replacement pseudocode for
`build_surface_atom_graph()` and the compact serializer. Decide whether the
smallest defensible representation is:

- multiple normal-mode children per occupied voxel;
- separate panel-interior, transition-strip, and exposed-boundary atom kinds;
- an adaptive or two-resolution surfel graph;
- or another bounded construction.

For the chosen design, specify deterministic clustering, normal-sign handling,
minimum support, spatial and angular adjacency, parent/child ownership,
transition-side membership, boundary evidence, canonical IDs/hashes, and exact
per-case caps. Preserve the global 200,000-point input bound, the 20,000 compact
atom cap unless you justify a replacement, deterministic CPU execution, and no
source-point serialization. Explain how the design avoids multiplying one
physical surface into duplicate bend evidence.

### 3. Make panel-side ownership coherent

Define the schema and constraints for assigning V2 atoms to zero, one, or more
plane-side modes. In particular, answer whether a transition parent may have
two mutually exclusive normal-mode children, whether ownership is on the
parent or child, and how two-sided occupied-bin minima should be computed
without double-counting area. Give solver constraints or an equivalent exact
formulation for bend, perimeter/free-edge, corner, junction, and residual
objects.

### 4. Redesign the first birth slice only as far as evidence permits

Audit `15_f1_structured_residual_birth.py` for internal contradictions with the
V1 atom schema. Provide exact V2 eligibility and a safe reduced-cost lower
bound. Cover missing/fragmented plane extraction, short returns, repeated
parallel bends, and one-sided occlusion. If these require separate birth
families, order them and define a falsification test for each; do not combine
them into an unconstrained generic line proposer.

### 5. Separate true folds from outlines at representation level

Give inference-safe features or hard certificates that distinguish:

- a material transition with two incident panel modes;
- an exposed perimeter/free edge;
- an inner/outer skin duplicate of one physical bend;
- a corner or junction trace.

Use local topology, normal modes, occupancy, and boundary structure—not global
span or expected counts. State which distinctions are impossible from the
present PLY evidence and therefore must remain UNKNOWN/review-required.

### 6. Define a family-held-out preregistered experiment

The 37 scans contain 27 part-family keys, with repeated scans for some parts.
Specify how to group near-duplicate scans and related geometry families before
training, using metadata only for evaluation partitioning and never inference.
Define A0/A1/A2 where:

- A0 is the frozen V1 result;
- A1 changes atomization/side representation only, with the existing solver;
- A2 adds the first justified V2 birth family.

Give per-family metrics, bootstrap or paired uncertainty, exact GO/STOP
thresholds, and the minimum number of represented training families required
before fitting any low-capacity potentials. Explain how to keep the current
37-case corpus useful for development while preserving a later locked
confirmation set of at least 30 scans across at least 15 families.

### 7. Provide the smallest code tranche

End with a file-by-file ordered patch plan for `13` through `17`, including
unit fixtures, property tests, memory/schema regressions, deterministic replay,
and post-lock gold diagnostics. Name every new schema field and invariant.
Identify one implementation milestone after which we should stop and rerun the
full Gold37 batch before adding learning or another birth family.

Do not authorize runtime promotion. If the evidence is insufficient to choose
among atom representations, give the smallest scan-only probe that would make
the choice falsifiable instead of guessing.

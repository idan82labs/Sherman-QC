# Sherman QC F1 — post-V65 fixture-contradiction architecture request

You are the professor/architect for a zero-shot, source-only bend-line detector for sheet-metal 3D scan geometry. Select exactly one bounded action-oriented next architecture, or issue a formal capability-opening STOP. Ground your ruling in relevant primary research where possible, but make the implementation contract self-contained and executable.

Return exactly one leading token:

- `EXECUTE_ONE` — followed by one fully frozen architecture and one source-free fixture suite; or
- `CAPABILITY_OPENING_STOP` — followed by the smallest scientifically necessary new information/capability and why source geometry alone is now insufficient.

Do not propose several alternatives. Do not ask for more analysis. We will implement and execute your single selection immediately and rule it out quickly if its preregistered fixtures fail.

## Objective

Improve zero-shot bend-line generalization and ultimately F1 on genuinely unseen physical parts, while preventing duplicate bend responses and false flat-face/perimeter lines. Results must receive native-resolution visual inspection before pseudo-truth scoring or retain/retire decisions.

## Retained baseline and dominant failure

- Retained frozen baseline: V52.
- On the consumed R001–R008 real-scan affine/resampling lane, V52 was primary/replay byte-identical but visually poor on 7/8 cases.
- Pseudo score after visual precommit: TP 18, FP 22, FN 14; precision 0.45, recall 0.5625, F1 0.50; exact cases 0/8.
- Dominant defect: multiple close, parallel candidate lines occupying what visually appears to be one physical fold, often with junction/perimeter clutter.
- A distinct R004-type failure is a long flat-face crossing and is not necessarily solved by pairwise ownership.
- Broad distance-only NMS is unsafe because nearby real bends exist.

## V65 ruling that was implemented

Your prior ruling selected `V65_SHARED_FOLD_BASIN_OWNERSHIP_CERTIFICATE_V1` (`SFBOC_V1`) with lock `V65_SFBOC_V1_MECHANISM_LOCKED`:

- pre-NMS, positive-evidence, pairwise ownership consolidator;
- mesh vertex projector is the area-weighted mean of incident face-normal projectors;
- path profile sample is valid only when largest eigenvalue of averaged projector is at least 0.85;
- positive pair requires at least 7/9 `SAME_BASIN` and zero `SPLIT_BASIN`;
- first locked fixture SF01 is one 70-degree dihedral fold with three routes at transverse positions -3h,+1h,+4h and requires all three to cluster, two suppressions, and at least 8/9 SAME_BASIN for every positive pair;
- failure of any one source-free fixture terminally retires V65 before real execution.

## What was executed

The V65 primitive was implemented without changing its thresholds. SF01 was instantiated as an equal-resolution, equal-area triangulated 70-degree dihedral extruded 80h along the bend axis. Ordinary NMS was forced false. All three candidate pairs were eligible and evaluated at the nine prescribed stations.

Observed result:

- eligible pairs: 3
- positive edges: 0/3
- SAME_BASIN: 0/27 stations
- SPLIT_BASIN: 0/27 stations
- INDETERMINATE: 27/27 stations
- station reason: `AMBIGUOUS_NORMAL` at every station
- mutation: none
- deterministic replay behavior: yes

## Exact mathematical contradiction

For an equal-area sharp dihedral with face unit normals n1 and n2 separated by angle theta, the locked crease-vertex projector is:

`P = 0.5 * (n1 n1^T + n2 n2^T)`

Its largest eigenvalue is:

`lambda_max = (1 + |cos(theta)|) / 2`

At theta = 70 degrees:

`lambda_max = 0.6710100717... < 0.85`

A material path crossing the sharp crease necessarily samples this crease neighborhood, so SF01 cannot simultaneously be an equal-area 70-degree dihedral and pass the locked per-sample confidence threshold. A deterministic unit diagnostic independently reproduced the eigenvalue. We did not round the crease, bias face density, lower 0.85, or otherwise rescue the consumed architecture. Per your lock, V65 is terminally retired and no real V65 lane was opened.

## Required ruling

Select the single smallest new source-only architecture that directly addresses the close-parallel ownership failure without inheriting this contradiction. It may be a corrected replacement, but it must receive a new architecture identity and lock token; do not amend or rescue V65.

The new contract must specify:

1. exact insertion point relative to route-local gates, V52 composition and ordinary final NMS;
2. exact input fields, source population, normalization and deterministic scale;
3. exact pair/cluster eligibility;
4. an orientation- and ordering-invariant representation that remains valid at sharp creases (for example face-local rather than crease-averaged evidence, if scientifically justified);
5. exact positive and negative evidence, fail-closed behavior and missing-support behavior;
6. exact mutation and representative ranking;
7. deterministic compact audit and primary/replay semantic identity;
8. CPU, memory, graph, pair, candidate and wall-time caps with atomic no-op;
9. a source-free fixture suite covering at minimum: sharp single fold with duplicate routes, rounded single fold, two nearby same-sign folds, U-channel/opposing folds, return/hem double skin, flat plate/perimeter distractor, true fold plus adjacent flat false line, missing support, exact positive-station boundary, replay/winding/face-order/vertex-order invariance;
10. one fresh, truth-unopened real-scan protocol using deterministic variants of exposed U01–U04 but not consumed R001–R008;
11. native-resolution visual audit requirements before pseudo-truth scoring;
12. hard retain/retire gates, including protection against false duplicate suppression;
13. whether the newly received physical part is required now or only for later unseen-part evidence.

## Frozen scientific constraints

- Source-only decisions: scan coordinates, scan mesh adjacency when present, candidate segment geometry, and opaque hashes only.
- Closed: gold, labels, CAD, drawings, expected counts, filenames, part/parent/family identity, good/bad status, transformed pseudo-truth, benchmark feedback, production outcomes and post-hoc tuning.
- R001–R008 are consumed and unavailable for selection or threshold fitting.
- V52 remains retained and frozen.
- V65 is terminal and may not be rescued.
- No synthetic-data optimization masquerading as real evidence.
- Every changed real case must be inspected visually at native resolution before pseudo-truth is opened.
- No rerun rescue or replacement input after a prospective lane begins.
- Gold and runtime/main/production remain closed unless you explicitly prove the smallest scientifically necessary opening.

Please be mathematically explicit and check internal consistency between every mandatory positive fixture and every confidence/validity threshold before issuing `EXECUTE_ONE`.

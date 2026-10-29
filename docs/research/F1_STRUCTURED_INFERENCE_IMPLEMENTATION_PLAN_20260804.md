# F1 structured-inference implementation plan

Status: research-only; runtime/main promotion remains prohibited.

Source review: `output/professor_package/Sherman_QC_F1_Structured_Inference_Dropoff_20260804/04_professor_response.json`

## Architectural diagnosis

The locked 45796000 failure is not a scalar-threshold problem. Its scan-only
candidate union already contains the eight STEP-adjudicated bends, plus four
locally plausible wall-end/corner paths. The v23 global router discards the
complete routed hypothesis and commits to a six-object mesh lane. Candidate
local scores cannot repair that loss because they do not model panel-boundary
ownership or the higher-order relationship among the twelve paths.

The replacement must therefore:

1. Treat routed, mesh, plane-lattice, cylindrical, and strict-owned proposals
   as measurements of common latent paths, never as mutually exclusive lanes.
2. Preserve one explicit semantic label per latent path. Only
   `BEND_STRAIGHT_COUNTED` contributes to bend count; corner, perimeter, hole,
   junction, curve, and residual paths remain visible but count zero.
3. Assign transition/boundary support exclusively, with explicit junction
   sharing only where the graph permits it.
4. Join compatible fragments into maximal paths before counting so one
   physical bend is counted once.
5. Use chord/corner and panel-incidence factors to distinguish true bends from
   locally strong outline paths. Do not add a part-specific cutoff, expected
   count, filename cue, or CAD feature.
6. Keep preprocessing bounded to 200,000 points and preserve deterministic CPU
   execution and byte-identical replay.

## Typed graph contract

The research graph uses the following semantic labels:

- `BEND_STRAIGHT_COUNTED`
- `CORNER_NONCOUNTED`
- `PERIMETER_NONCOUNTED`
- `HOLE_NONCOUNTED`
- `JUNCTION_NONCOUNTED`
- `CURVE_NONCOUNTED`
- `RESIDUAL_NONCOUNTED`

Every canonical path records geometry, incident panels, independent source
groups, support features, endpoint junctions, duplicate relations,
continuation relations, and an audit trail back to raw candidates. Human-readable
case IDs and scan paths are excluded from features, ordering, and tie breaking.

The reported bend count is the number of connected components of counted
fragments under selected continuation links. Equivalently, for an acyclic
continuation graph:

`bend_count = counted_fragments - selected_continuations`.

## Delivery sequence

### Tranche A: deterministic intermediate representation

- Canonicalize the union of existing candidate sources into latent paths.
- Recover incident panel pairs from structured fields or source provenance.
- Collapse correlated generator variants into independent source groups.
- Emit stable geometry-derived IDs and a stable JSON graph hash.
- Add invariance tests for candidate order, source order, filename, and part ID.

### Tranche B: typed relational baseline

- Add one-of-seven labels and explicit counted/non-counted output.
- Add duplicate and continuation relations.
- Add generic panel-pair chord/corner factors. Length may contribute only as
  part of a relational signature; it is never a standalone rejection rule.
- Preserve second-best energy/margin and all rejected alternatives for audit.

### Tranche C: exact deterministic solver

- Integerize dimensionless costs at `1e6` precision.
- Solve disconnected factor components independently.
- Use a pinned single-worker exact solver with canonical decision order and a
  proof of optimality for promotable research results.
- Apply sequential exact tie breaks: primary energy, owned non-residual support,
  continuation quality, birth count, then canonical geometry tuple.

### Tranche D: residual birth and learning

- Add deterministic residual column generation only when at least two
  independent scan modalities support the same unowned transition.
- Learn only low-capacity unary/structured potentials with nested
  leave-one-family-out evaluation. Weight by family, not by candidate or point.
- Freeze architecture, tolerances, weights, hashes, and environment before any
  new untouched-family confirmation.

## 45796000 regression contract

The case is now a post-design regression, not independent validation.

- S1-S8: `BEND_STRAIGHT_COUNTED`.
- S9-S12: explicit non-counted wall-end/corner paths; primary label
  `CORNER_NONCOUNTED`, with manufacturing subtype left auditable/uncertain.
- The output count is eight.
- The result must not read CAD, drawing, expected count, manual labels, part ID,
  or filename during inference.
- Removing union fusion must reproduce the v23 candidate-recall loss.
- Removing the chord/corner factor must make the wall-wall aliases harder to
  reject on this case or an equivalent held-out topology.
- Repeated runs must produce byte-identical graph, labels, axes, and objective.

## Evidence boundary

Gold-27 remains development evidence, and 45796000 is no longer untouched for
the replacement architecture. STEP pairing is strong engineering adjudication,
not independent human truth. Runtime promotion requires frozen evaluation on
multiple new manufacturing families with independent bend and non-bend labels,
plus the existing memory, determinism, visual-integrity, and release gates.

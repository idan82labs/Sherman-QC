# Geometry-aware scan-plane seeding — prospective successor

Status: standalone source implemented and independently reviewed; tests are
being independently authored. No imports, execution or integration yet.
This is an engine proposal mechanism, not a scoring change or a retry of any
terminal connected packet. Frozen V5/V7/V12 and support-cover V2 remain intact.

## Retained failure and hypothesis

The complete `support_cover_v2_connected-zlspiuqf` packet has V2 measurement
positives on U-R/U-A, but L-R/L-A each produce zero scan-plane proposals. All
13 declared topology-only BFS tile-plus-halo cores were evaluated. Twelve have
no fixed-normal all-row witness; one has a witness but a ~0.251 rad conditional
normal cone, above the unchanged pi/18 policy. No closure or registration start
exists for these two cases. The clean L-A case also fails, so noise alone does
not explain the failure. The evidence is not proof that no plane exists.

Source mechanism: arbitrary fixed-size topology tiles can cross welded
surface junctions; the unconditional one-ring halo can also cross them. Whole
support qualification correctly rejects mixed supports, but the proposal
generator has no geometry-aware way to propose a coherent broad planar support.
This is the next mechanism to test, not a reason to relax final qualification.

## Research grounding and distinctions

CGAL's [Shape Detection manual](https://doc.cgal.org/latest/Shape_detection/index.html)
describes fixed seed-face planes, edge-adjacent region growth, area-based seed
ordering, and distance/normal gates for polygon meshes. That supports using
geometric seeds rather than topology-only chunks; it does not establish our
uncertainty model or predict success on this corpus.

[Schnabel, Wahl and Klein (2007), sections4.4 and4.6](https://www.hinkali.com/Education/PointCloud.pdf)
separate candidate distance/normal/connectivity support from subsequent fitting.
We do not adopt its random sampling, largest-component score, point removal,
or expanded refit distance. No statistical confidence or RANSAC success claim
will be attached to a deterministic bounded seed schedule.

## Source contract and limitations

- Seed solely from immutable raw mesh triangle geometry. CAD, Gold, part names,
  constructed roles, poses and measured truth are absent from this API.
- Freeze a bounded, deterministic seed schedule before evaluating regions.
  Retain every scheduled seed and its disposition, including duplicate supports,
  rejected fits and numerical failures. Report omitted seeds and nonexhaustiveness.
- A seed's observed normal is a proposal direction, not a calibrated normal
  bound. Use one fixed seed plane to propose distance- and normal-compatible
  triangles. Do not chain locally similar normals across a curved surface.
- Retain only the component containing that seed in the original shared-edge
  graph, with all three original vertices required for each selected triangle.
  Disconnected coplanar pieces are not silently merged; holes are not filled.
- Refit every unique proposed support using every distinct original raw ID in
  that triangle union. Use unchanged V5 all-row offset-witness and conditional
  normal-cone mathematics and thresholds. No residual-based peeling, coordinate
  welding, error-bound enlargement or nominal-plane substitution.
- Seed membership thresholds are disclosed engineering proposal assumptions,
  not uncertainty coverage or surface absence proofs. Final qualified supports
  are conditional plane proposals, not physical face identities or bend counts.
- The actual seed slab is epsilon for the candidate row plus the largest
  epsilon on the seed triangle, plus arithmetic guard: uniform0.05mm means a
  0.10mm proposal slab. The observed triangle-normal gate uses10degrees as an
  engineering proposal heuristic. Final V5 qualification still uses0.05mm and
  its unchanged conditional normal-cone bound; neither proposal threshold is
  presented as calibrated uncertainty. Seed angular error can remove genuine
  broad support by approximately L*sin(theta); success is not guaranteed.
- Budgets must cover seed selection, every raw/triangle/edge visit, full fits,
  memberships, arrays and encoded evidence. Late fatal failure clears all
  publication prefixes and retains attempted-work/discarded-evidence counts.

Source caps:32seeds,64Mselection-triangle/32Mproposal-raw/64Mproposal-triangle/
96Mproposal-edge evaluations,32Mfull-fit rows,20Mtriangle/32Mraw memberships,
32fits,256MiB retained arrays and64MiB encoded metadata. Logical evaluation
counters are not literal primitive-operation or array-pass counts. Fixed V5
input caps govern initial owned input validation; preparation has separate
reported row/triangle/edge counters. A1GiB declared working-array reservation is
checked after owned inputs, not before their allocation and not as an OS RSS
guarantee. The isolated run supervisor supplies the actual sampled RSS limit.

Before integration, independently constructed tests must cover proper poses,
noise, row permutations, irregular tessellation, welded orthogonal faces, thin
caps, smooth curvature, holes, disconnected coplanar surfaces, duplicate raw
coordinates, nonmanifold input, ambiguous/degenerate seeds, and late exhaustion.
The existing complete curved packet keeps its original geometry and output
oracles. Its two negative localized cases must not be replaced by easier cases.

## Not authorized by this document

This note does not freeze a runnable experiment, promote a candidate, authorize
native or physical execution, change Gold/scoring, or claim the old F1 was beaten.
The next source and execution gates need exact versioned code/configuration,
proportionate budgets and independent PRE review.

## Conditional integration plan (source review, not implemented)

After the standalone source gate, use one new owned composite preparation
helper rather than copy the registration stack. Preserve complete V7 and
seeded-result schemas and failures under an explicit composite schema. Assign
deterministic combined plane IDs with lane/original-kind/original-record and
support-hash provenance. Seeded search observations use V6's supported DECLARED
kind; their truthful SEEDED_REGION origins remain in the manifest.

An explicit validated mechanism option threads through V12 configuration to
the V8 preparation seam. The default remains the original V7 path. The V12
registration API schema may remain because its numerical/context semantics do
not change, but both requested and returned mechanism/digests must be bound.
Do not install caller callbacks, monkeypatch module globals or relabel seeded
regions as V7 cores. Preserve every terminal source archive before later edits.

A strict composite surface-role admission helper can construct the same owned
token-bound role context. The existing selector must see both lanes together:
combining separate exclusion masks could lose competing noncap evidence. Bind
the composite metadata SHA in the existing four-field role-source binding.
Keep omitted seed counts separate from unscheduled qualified V7 cores. Require
both lanes complete and enforce aggregate membership/metadata/work limits;
never keep an apparently successful V7 prefix after a fatal seeded-lane failure.

Independent integration checks must include original-path equivalence, origin
ID collisions, missing qualified records, cross-lane role conflicts, late lane
failure, aggregate cap exhaustion and requested/returned mechanism mismatch.
Then rerun the original full curved construction under a new frozen protocol;
the current RED packet remains untouched. Native-file and physical advancement
remain downstream of those gates.

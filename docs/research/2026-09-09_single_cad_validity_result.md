# Single-CAD native validity: retained negative and context-coverage gap

Date: 2026-09-09. Independent post-run reviewer: OpenAI Codex AI agent
`/root/cad_support_lineage`. Scope: the one completed frozen diagnostic and
explicitly released earlier invented outputs; no original CAD, scan, Gold,
held-out or email access, native execution, healing or retry.

## Independent disposition

**Custody GREEN: a valid, terminal negative native-validity observation.**
The imported solid is reported `native_valid=false`; that finding is retained.
No engine output, count/F1 improvement or physical part defect follows from it.

**Complete causal diagnosis and clean-local-patch admission are NOT GREEN.**
Although the frozen exporter reports `DIAGNOSTIC_COMPLETE` and
`evidence_complete=true`, its context records omit 296 of 592 edge-to-face
incidences and 172 of 592 vertex-to-face incidences represented by its own
occurrence tree. The current completeness flag establishes successful traversal
of the returned context iterator, not complete incident-context coverage.
This is a P1 blocker for consuming this report as a local-admission certificate,
not a reason to erase the run or its global negative.

The same edge-context deficit is already present in retained invented box
outputs. The next justified step is a separately versioned context-completeness
validator and collector tested on invented geometry, before considering local
feature admission. A blanket validity bypass is unsupported.

## Exact execution and custody

The [frozen bundle](v2_native_validity_diagnostic_attempt/FROZEN.json) is
`4b91d6c7db4e8709a609eb6f9fc845271d83eebc316378875866f4a2b0f1fd5c`.
The [pre-run audit](v2_native_validity_diagnostic_attempt/PRE_RUN_AUDIT.json)
was bound into the exclusive STARTED and worker/coordinator capabilities before
the sole original descriptor read. Its canonical JSON digest is
`3a3eb2f0e57d8eb854c71c3191f48ce7b6d0d7724532a8d3ea4fab97d7a0e222`.

Read and cross-checked the retained STARTED, coordinator dispatch/claim,
verified context, worker dispatch/claim, coordinator completion, final completion,
result and output. Independently recomputed their canonical binding hashes and
the result/output byte hashes. Every one of the seven current and frozen source
pairs still matches FROZEN. No frozen source or control was edited.

| Retained observation | Value |
| --- | --- |
| Final status | `COMPLETE_DEVELOPMENT_DIAGNOSTIC`, phase `COMPLETE` |
| Approved input | Sole consumed family `49001000`, provenance token `CG191DEC7F81`; 260402 bytes |
| Descriptor hash | `d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5` |
| Descriptor observations | Regular, metadata stable before/after; exact bytes/hash match |
| Parent elapsed | 4.405096000060439 s |
| Worker CPU | 4.161675 s |
| Sampled complete group plus outer peak | 394919936 bytes |
| Worker high-water RSS | 323207168 bytes |
| Output | 296 bytes, observed and retained counts equal |
| Structured result | 266614 bytes |
| Budget/source outcome | No budget stop, no error, sources consistent |
| Terminal cleanup receipt | Coordinator group gone, no group attention, private cache cleanup confirmed |
| Native actions | One STEP import, one analyzer |

The [result](v2_native_validity_diagnostic_attempt/RESULT.json) byte hash is
`157cc870a6b8ca529875dfbb95b84e1b570c288a5133193b65ad7d99c7d06915`.
The [original output](v2_native_validity_diagnostic_attempt/OUTPUT.txt) byte hash is
`9f55360c3eb58749c9f7adb2281009648dfd4dff94611a69d74f398a4ba31f9b`.
The [completion receipt](v2_native_validity_diagnostic_attempt/COMPLETED.json)
retains those hashes, the worker-claim digest and the nested coordinator report.
All parent/worker IDs agree across the claim chain; no historical PID was signalled
or rechecked as if it remained owned.

The fixed 120 s total / 110 s work plus 10 s reserve, 90 s worker CPU, 1 GiB sampled
complete-tree-plus-outer RSS, 1 MiB output and 768 KiB result limits were not changed.
Version metadata is not binary attestation; Python-level I/O guards and sampled
process accounting are not an adversarial OS sandbox or continuous RSS proof.

## What the retained native observations establish

One transferred root, source length declaration `MILLIMETRE`, target transfer
unit `mm`; physical calibration is explicitly not established. The native result
contains 730 unique entities and 2026 occurrences: 1 solid, 1 shell, 114 faces,
134 wires, 296 edges and 184 vertices. All 730 detail IDs are unique and match the
entity IDs; no property/status read error or declared truncation was recorded.

Not all curved faces fail:

| Native surface type | Total faces | Self `[27]`, subtree false | Self `[0]`, subtree true |
| --- | ---: | ---: | ---: |
| Plane | 44 | 0 | 44 |
| Cylinder | 60 | 37 | 23 |
| BSplineSurface | 10 | 7 | 3 |
| Total | 114 | 44 | 70 |

Thus 26 of 70 nonplanar faces have the latter reported status. These are native
status classifications, not qualified bend roots, hole classifications or
admitted local measurement patches.

All 44 self-error entities are faces with the sole self code 27,
`BRepCheck_UnorientableShape`. The 46 false subtree results are those 44 faces,
the shell and the solid. The shell and solid have self `[0]`; their subtree
failure must not be relabelled as 46 independent underlying defects.

The exported context errors are 22 distinct BSplineCurve edges, each in one of
14 Cylinder face contexts, with the pair `[8,11]`:
`BRepCheck_InvalidCurveOnSurface` and `BRepCheck_InvalidSameParameterFlag`.
Every one of these edges nevertheless has self `[0]` and subtree true. All 296
stored `SameParameter` and `SameRange` flags are true. An edge-only `IsValid`
check or checking those flags alone would therefore miss the reported failures.

All exported native tolerance values are finite/nonnegative; maxima are
0.0013532160452609844 mm for edges and vertices, and 1e-7 mm for faces. Those
stored maxima are not measured curve/surface discrepancies, a scanner error
budget, or evidence that increasing tolerance is safe.

## Why the completeness claim is too strong

The independent aggregation used only the retained occurrence tree: follow each
occurrence's parent chain, identify relevant owning face/edge/solid entities,
and deduplicate `(entity_id, owner_id)` pairs. No coordinate inference or native
query was used. Compare these pairs with the explicit owner IDs in the exported
context lists:

| Context relation | Occurrence-derived pairs | Exported pairs | Missing |
| --- | ---: | ---: | ---: |
| Edge to face | 592 | 296 | 296 |
| Vertex to face | 592 | 420 | 172 |
| Vertex to edge | 592 | 592 | 0 |
| Wire to face | 134 | 134 | 0 |
| Shell to solid | 1 | 1 | 0 |
| Total | 1911 | 1443 | 468 |

The report's 1443 observed context records agree with its own counter and use
known entity IDs. Every edge has two distinct incident faces in the occurrence
tree, but exactly one exported face-context record. There are no extraneous
edge-to-face pairs. Absence is not a favorable status.

The 22 observed bad contexts cover 14 of the 44 self-error faces. The other 30
self-error faces (23 Cylinder, 7 BSplineSurface) have no exported incoming error
context. The current output therefore cannot support an exhaustive cause map.
It does not establish whether missing records reflect cached native-state
coverage, the bound iterator/API behavior, or the collection sequence.

Re-aggregation of the already retained
[invented native matrix output](cad_engine_v1_synthetic_checks/native_validity_v2_constructed-thbnotxp/OUTPUT.txt),
hash `2d092fc3bcecc18fcae257bb65c71304a7f0120189a9055908b63a56f65e70a7`,
finds the same issue without running anything:

| Previously passed invented case | Edge/face incidence pairs | Reported pairs |
| --- | ---: | ---: |
| Valid box | 24 | 12 |
| Open-shell invalid solid | 20 | 12 |
| Unit-bound STEP box | 24 | 12 |

Those cases still satisfy their frozen native-validity dispositions. Their old
oracle did not test incident-context coverage; a passing old oracle cannot
close this newly identified gap. The earlier failed fixture and failed smoke
attempt remain retained and unchanged.

## Primary-source interpretation, not an asserted physical cause

OCCT's 7.9.3 analyzer checks edge/face representation compatibility using a 3-D
edge curve `C(t)`, a face surface `S`, and its 2-D pcurve `P(t)`. Conceptually the
same-parameter requirement is agreement of `C(t)` and `S(P(t))` along the finite
edge interval, within the edge's tolerance model. True stored flags are a
declaration, not proof of that compatibility. An edge outside a face context
does not receive the same test. [Analyzer API contract](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Analyzer.hxx)

The edge implementation emits the observed code pair when curve-on-surface
validation either does not finish or fails its tolerance check. Hence this
report does not distinguish numerical noncompletion from an excessive
representation discrepancy. It contains neither maximum discrepancy nor its
parameter location. [Edge validation branch](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Edge.cxx)

The validator's exact-method path still uses numerical curve-on-surface work,
and its tolerance comparison includes curve/surface precision terms. This is
not certified exact arithmetic or a measured physical deformation.
[Validation implementation](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepLib/BRepLib_ValidateEdge.cxx)

`UnorientableShape` can be a downstream analyzer status: certain bad edge/face
statuses prevent wire-orientation checking and cause that face status to be
set. Subshape failures can also make a containing shell/solid invalid. The
14 observed bad-context faces are consistent with this propagation; the
incomplete context coverage prevents extending that explanation to all 44.
It does not prove that reversing normals fixes the input.
[Analyzer propagation and subtree logic](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Analyzer.cxx)

OCCT exposes both an iterator and keyed cached-context access through
`IsStatusOnShape(shape)` / `StatusOnShape(shape)`; the latter can distinguish a
missing cached context from a cached favorable result without invoking
`InContext`. [Result interface](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Result.hxx)

These are upstream source explanations for the 7.9.x API, not proof that the
installed `cadquery-ocp 7.9.3.1.1` binaries exactly equal an upstream source tag.
The result cannot isolate original export quality versus transfer/numerical
behavior. It does not establish that the manufactured part is defective.

## Bounded next engineering step

1. Preserve the existing global validity refusal and every terminal artifact.
   Add a separately versioned, pure incidence/status-coverage validator with a
   complete expected-context denominator, unique pair checks and explicit
   `MISSING_CONTEXT` / `CONTEXT_ERROR` / `CONTEXT_CLEAR` dispositions. Missing,
   malformed or over-budget evidence clears any complete-admission claim.
2. On separately audited invented geometry, compare iterator enumeration with
   keyed cached access for every required incident context, before and after
   subtree queries and in reversed entity traversal order. Include a shared-edge
   box, open shell, finite cylinder/flange patch, one corrupted incident pcurve,
   a stale same-parameter declaration and a deliberately missing-context double.
   Preserve both observations when they disagree. No `InContext`, healing or
   tolerance change should manufacture a favorable cache entry for this check.
3. Only after that seam is proven, investigate a separate local geometric
   admission mode. A candidate needs explicit same-import root/flange identities,
   complete finite trims, all consumed wires/edges/vertices and relevant incident
   context witnesses. Any bad/unknown dependency excludes that candidate. A
   global-invalid solid must remain global-invalid; orientation/material-side
   claims need their own local witness and cannot inherit an invalid owner's
   presumed outward direction. Status-clean cylinders may be holes and are not
   automatically bend candidates.
4. Require a declared finite geometry/parameterization check where extraction
   relies on that representation, with numerical uncertainty and unsupported
   cases explicit. A same-parameter discrepancy can arise even when differently
   parameterized curves share a geometric image; toggling a flag or widening
   tolerance is not a substitute. Preserve partial candidate inventory and
   unknown observed bend count. Qualification requires separate scan evidence.

This recommends source/invented validation work, not a new physical command or
permission to reopen this attempt. A later physical successor requires its own
documented mechanism, freeze and independent gate. No production/default path,
Gold, geometry gate, tolerance or frozen diagnostic was changed by this review.

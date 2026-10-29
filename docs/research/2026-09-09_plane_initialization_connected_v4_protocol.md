# Plane-only finite-face initialization V4 — prospective protocol

Date: 2026-09-09. **Source-only implementation; no execution or promotion is
reported here.** The earlier V1, dense-V1, V2 and fine-only V3 negative receipts
remain unchanged. This is a separately versioned initialization mechanism, not a
retry, broader noisy-scanner claim or permission to open physical/held-out data.

## Hypothesis and scope

The previous finite-surface coarse basin can be poor even when local pose rank
is adequate. Three independent scan/CAD planes can generate a rigid start without
PCA or a supplied pose. Finite assigned-face support can discard many wrong
infinite-plane permutations before spending the unchanged V2 refinement budget.

This is a bounded **nonexhaustive search heuristic**. A rejected start might have
refined successfully. Partial observations, deformation, fragmented noisy mesh
planes, genuine planar cylinder facets and repeated geometry can defeat it. It
does not certify that every plane was found, that rejected starts cannot converge,
that a physical pose is unique, that a bend is absent, or that CAD conformance is
known. Near-roundoff mesh compression is not general noisy-plane extraction.

## Source entrypoints and owned input

New source: [cad_guided_plane_initialization_v4.py](../../backend/cad_guided_plane_initialization_v4.py).
Both entrypoints are pure array seams with no native/file/app/physical input IO:

```python
prefilter_plane_initializations_v4(
    scan_points, scan_triangle_raw_rows, cad_triangles,
    cad_face_ids, cad_source_triangle_ids, *, registration_policy,
    patch_policy=..., patch_budgets=...,
    hypothesis_policy=..., hypothesis_budgets=...,
    budgets=PlaneInitializationBudgetsV4(), surface_budgets=None)

register_whole_part_planes_v4(
    scan_points, scan_triangle_raw_rows, cad_triangles,
    cad_face_ids, cad_source_triangle_ids, *, cad_initialization_points,
    registration_policy, unit_assumption, registration_raw_rows=None,
    patch_policy=..., patch_budgets=...,
    hypothesis_policy=..., hypothesis_budgets=...,
    budgets=PlaneInitializationBudgetsV4(), surface_budgets=None,
    registration_budgets=None)
```

Geometry/indices are exact numeric ndarrays, copied to byte-owned immutable
little-endian arrays. Policy and budget fields are revalidated and reconstructed,
including frozen instances modified by bypass. Public callers cannot supply a
pose, plane-equation dictionary, selected support dictionary, part identity,
nominal bend answer or labels. Import-local face/triangle IDs serve lineage only.

The public function itself calls the unchanged scan component extractor and CAD
complete-face fitter, then the unchanged plane-triplet helper. It verifies each
used scan patch's triangle rows against its complete component label and its raw
support against the union of those original triangle indices. Each used CAD face
is checked against **every** input triangle carrying that face ID and the matching
local triangle IDs. These checks occur once per used patch/face, not per pairing.
Hashes bind byte snapshots; they do not authenticate native ownership or files.

The private `_prefilter_prepared(context, result, policy, budgets,
surface_budgets)` seam consumes internally constructed context. Tests that inject
planes or callbacks there establish only that explicitly mocked helper contract,
not whole-public-input lineage or autonomous registration.

The outer workflow must bind the exact byte-owned STEP/native tessellation to
CAD triangles and CAD initialization samples, and the new raw triangle parser to
the scan. This source does not make those native/file claims on arbitrary arrays.
The existing mesh parser, native shape ownership and unit-conversion safeguards
remain necessary; this module does not reopen or reinterpret them.

## Mathematical decision rule

The helper enumerates bounded independent triplets, CAD permutations and normal
signs. It uses proper Wahba normal alignment and intrinsic plane intersections
`t = x_CAD - R x_scan`. Direct offset solving is equivalent only for exactly
aligned normals; its origin-dependent diagnostic is not a decision gate.

For each hypothesis `h` and each of its three assigned pairs `(i,j)`, let `S_i`
contain every original raw vertex ID of scan patch `i`, and `F_j` the complete
finite triangle union of CAD face `j`. The prefilter calculates

```text
c(h,i,j) = count[p in S_i:
                  distance(R_h p + t_h, F_j) <= max_correspondence_distance_mm
                  and V2 correspondence is unambiguous] / len(S_i)
```

All three coverages must reach the **unchanged registration policy's**
`min_scan_coverage`. The threshold is not selected from a CAD bend count or pose
oracle. The denominator is raw IDs, not coordinate-deduplicated positions,
subsampled support, retained ICP rows, pooled area or another patch's support.
All three patches are fully evaluated even if the first is a geometric rejection.
The plane helper's support admission applies; no extra V2 global 36-point minimum
is imposed on an individual small patch.

Each distance is to the finite assigned face's original triangle union. No filled
polygon, convex hull, infinite plane or normal-only approval is substituted.
Holes, disconnected pieces and finite boundaries remain. EDGE/VERTEX distances
are allowed. V2's `normal_ambiguous | closest_point_ambiguous` mask is reused;
different face tags alone are not a geometric ambiguity. There is no per-face
reverse-coverage requirement: a genuinely partially observed face may pass.
The unchanged final V2 global CAD-sample reverse-coverage gate still applies.

Every hypothesis and signed assignment provenance remains in the report. Exact
byte-identical validated `(R,t)` starts, including signed-zero distinctions, may
share subsequent V2 refinement only **after each assignment is independently
screened**. Rejected assignments are never rescued by a passing duplicate.
No tolerance/epsilon clustering, source-ID ranking or top-prefix selection occurs.
The initial face assignments remain initialization provenance after refinement;
they are not published as final face matches.

## Unchanged V2 refinement and final gates

For each surviving exact start group the wrapper calls the unchanged V2 `_refine`
on the V1 spatial coarse subset, then on the V1 spatial fine subset. It preserves
iteration limits, global trim, proper weighted Kabsch, Huber weights, center-based
step caps and fixed-correspondence backtracking. It does not invoke V3, add
iterations, force an oracle pose or refill informative rows.

The unchanged V2 `_diagnose` supplies every final metric. Finite Euclidean P95 is
tested against the **unchanged numeric legacy plane-P95 ceiling**; it is not
relabeled a plane residual. All-support, actual-trim and Huber-weighted FACE-normal
observability, forward coverage, reverse CAD-sample coverage and residual gates
remain unchanged. `_v2_select` preserves V2 eligibility, sorting, final refined
pose-class threshold and competing-class criteria verbatim in mathematical action.

No PCA fallback is mixed into this challenger. Frozen V2 remains a separate
control. Original initialized rotations/translations and their hypothesis IDs
are retained alongside refined transformations and V2 diagnostics.

Without `registration_raw_rows`, the wrapper preserves V2's maximum 200,000 raw
registration-input rows and canonical coordinate deduplication. A supplied array
must contain 36–200,000 unique bounded **original raw IDs**, not coordinates or a
fabricated centroid cloud. It changes only the global registration input; the
prefilter still checks every original patch support row. Reports explicitly say
selected-row global coverage, record full/selected/unique row counts and digest
the selection plus its positions. No hidden adaptive sampling is performed.
Future physical selection must be frozen/bound separately. The original invented
connected fixture uses `None`, so its actual raw input is unchanged.

The wrapper requires `PROVISIONAL_COMMON_MM_NOT_CALIBRATED`; the standalone helper
labels `PROVISIONAL_ARRAY_MM_UNVERIFIED`. No fitted scale, stable datum, calibrated
uncertainty, material side, production use, observed bend count or conformance
approval follows. A global best fit may absorb real deformation.

## Frozen internal resource caps

These are prospective engineering scope limits, not measured capacity claims.
Callers may lower exact integer caps, never raise them or pass booleans/floats.
`chunk_rows` must be positive. Each underlying triangle-query cap remains in force.

| New initialization resource | Hard maximum |
| --- | ---: |
| Hypothesis records | 4,096 |
| Cached complete CAD face indices | 24 |
| Total CAD triangles admitted/indexed for face screening | 500,000 |
| Prefilter query calls | 32,768 |
| Prefilter query rows | 32,000,000 |
| Rows per query chunk | 65,536 |
| Cumulative node visits | 512,000,000 |
| Cumulative triangle pairs | 256,000,000 |
| Cumulative normal pairs | 32,000,000 |
| Cumulative logical query work units | 4,000,000,000 |
| Cumulative face-build nodes | 131,071 |
| Cumulative face-build triangle visits | 10,000,000 |
| Surviving exact start groups | 24 |
| Detailed JSON report | 16 MiB |

The entire all-three-patches row and chunk-call obligation is checked before any
face BVH/query. Face caches use each complete imported face once. Per-call budgets
are the minimum of the unchanged triangle-kernel budget and the remaining new
cumulative budget. No `hypotheses × raw-points` distance tensor is allocated.
The existing patch and plane-assignment input/work caps apply independently.
Fixed input caps also bound ownership rechecks; they do not claim constant-time
processing of arbitrary mesh topology.

The V2 whole-surface index and V2 cumulative refinement/query budget are a separate,
explicitly reported phase, unchanged across all surviving starts. Resource
statistics are not CPU instruction counts. An independent serial process-tree
watchdog must enforce wall time, CPU, RSS and stdout; source counters are not a
replacement. The planned small pure-array test tier is 60 s wall / 45 s CPU /
1 GiB process-tree RSS with 1 MiB stdout. A later connected native gate must be
separately frozen by the integrating agent before execution; this document does
not raise an existing native budget or authorize a run.

## Failure, completeness and report semantics

Patch/plane proposal, invalid input, finite-query/index, cumulative budget or
surviving-start overflow makes initialization unavailable and clears **all usable
starts**. Completed, rejected, incomplete and not-attempted hypothesis records
remain explicitly attempted evidence, with no usable group index. More than 24
distinct surviving byte-exact starts refuses the whole arm; no first/best24 escape
exists. Fatal global registration errors clear the final transform and retain all
attempted/unattempted candidate records and query counters.

V2's per-candidate numerical refusal is preserved, rather than silently changing
its acceptance rule. `candidate_numerical_failure_count` is explicit and
`candidate_evaluation_complete` is false if any candidate failed numerically or
was unattempted. Completing all attempted refinements is not global search proof.
Any later **positive connected-mechanism claim requires zero numerical-failed or
unattempted candidates**, in addition to its fixed pose/measurement criteria.

The public report is JSON-native and forbids NaN/Infinity. It retains all compact
patch/component metadata and all hypothesis/assignment/candidate records. Large
internal raw/triangle maps are represented in output by exact counts and
little-endian SHA-256 digests, explicitly not embedded maps or authenticated native
lineage. Each used support query still uses the complete internal arrays.

If the detailed-report cap itself is exceeded, the **entire** detailed report is
discarded, not truncated to a successful prefix. Only a fixed small refusal
envelope, examined byte count and discarded record counts are returned. This
refusal envelope is outside the detailed-report allowance; no pose or starts
remain. Such an attempt is an incomplete negative, never positive proof.

## Independently authored prospective checks

The integrating agent owns independent tests and all execution. No source control
may be called a public registration success when it supplies mocked proposals or
diagnostics. Fixed test groups, before execution:

1. Independently posed asymmetric finite plane triads: correct action retained,
   every full-support denominator exact, wrong infinite assignments rejected;
   raw-row/triangle order and import-local face-ID relabeling do not supply pose.
2. Wide flange assigned to a narrow cap, and three large/small supports: reject
   finite mismatch and do not pool a failed small patch into passing flanges.
3. A true CAD hole and disconnected face union versus its filled plane/hull;
   finite boundary behavior must follow actual triangle distances.
4. Correct partial face observation without a new per-face reverse gate, plus
   recoverable/deformed near misses honestly labeled heuristic rejections.
5. Repeated faces and equal poses with different assignments: every assignment
   independently screened; only bitwise-identical passing poses share refinement;
   nearby poses do not merge and excess surviving groups refuse all.
6. Stale/forged raw union, CAD face union, ID/digest mutation and mutable caller
   input; public internals recompute ownership, not trust supplied dictionaries.
7. Every exact typed input/budget boundary, preflight row/call cap, late finite
   failure, build/work/start/report cap: no usable partial prefix, all attempted
   evidence/counters truthful, strict full JSON or explicit whole-report refusal.
8. Unchanged V2 wrapper math: real finite-kernel/public geometry checks where
   bounded; clearly mocked seam tests for candidate bookkeeping, numerical
   failures, selection, local-rank and symmetry refusal. Pose oracle is only
   post-inference. No test may equate an abstention with a required good-case pass.

Only after independent source review and focused controls should a separately
frozen successor reuse all three original invented native cases: nominal V1
feature arm must measure one finite instance within every original radius, sweep,
endpoint and pose ceiling; missing-arc must emit zero without fatal/incomplete
execution; R0-extractor nominal arm must also measure one. Same independently
constructed geometry, raw points and 32,768 CAD sample count as the dense/V2
successor; same registration/measurement gates, no supplied true pose. Numerical
and unattempted candidate counts must both be zero. Retain every case, candidate,
failure and source/resource receipt. A connected improvement is not Gold-F1,
unseen-family accuracy, noisy capacity, factory readiness or product promotion.

## Frozen dependencies for source-drift review

These SHA-256 values were read from source only during this implementation.
They do not authenticate a future executed closure. The integrating runner must
bind the new source/tests/protocol and check all exact dependency bytes before
and after any separately authorized run.

| Source | SHA-256 |
| --- | --- |
| `cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `cad_guided_global_registration_v1.py` | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `cad_guided_mesh_plane_proposals_v4.py` | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `cad_guided_plane_hypotheses_v4.py` | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `cad_guided_mesh_topology_v4.py` (outer adapter) | `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734` |

The relevant V2 copied selection block is `register_whole_part_v2` from
`eligible = ...` through its selected candidate publication; compare it to new
`_v2_select` in mathematical action. V2 `_refine`, `_diagnose`, `_Queries`, V1
canonicalization/spatial sampling and all original files remain unchanged.

## Handoff state

Only the new source and this new protocol were authored by this subagent.
No tests, Python imports, native operations, capacity sweep or physical inputs
were executed/accessed by it. Independent tests/source audit and the root's frozen
serial harness are the next gates. No default product route or UI was changed.

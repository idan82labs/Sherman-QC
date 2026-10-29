# Connected invented mesh/STEP plane initialization V4

Date: 2026-09-09. **Prospective source-only protocol. No native execution result is
reported here.** Root must complete independent source/test review and freeze the
exact dependency closure before running the single serial three-case job.

## New artifacts and unchanged control

New implementation: `backend/cad_guided_mesh_offline_v4.py`.
New connected/source checks: `tests/test_cad_guided_mesh_connected_planes_v4.py`.

The new workflow calls the reviewed [plane-only finite initializer](2026-09-09_plane_initialization_connected_v4_protocol.md)
instead of V2 PCA starts. It preserves V2 finite refinement, final global gates,
native feature branches, measurement, finite-instance construction and rigid
inverse publication. No product default route, UI, V1/V2/V3 source, original test,
failed receipt, held-out or physical input is changed.

Retained predecessors are `mesh_connected_native_v1-jkz08n45`,
`mesh_connected_native_dense_v1-j5p3bnlr` and
`mesh_connected_surface_v2-lqfsnygb`. The last V2 matrix was 1/3 with both nominal
cases refusing. This protocol cites those fixed identities without reading or
rerunning their output. The new r0 branch is an **r0 feature extractor in the new
common wrapper**, not the frozen historical pipeline or an empirical historical
baseline rerun.

## Exact connected data path

`inspect_cad_guided_mesh_v4` is a separate opt-in byte API. Its four existing policy
arguments retain their meaning; optional patch/hypothesis/initialization and
finite-query budgets expose the new bounded mechanism. `MeshWorkflowPolicyV4`
inherits the exact V2 policy fields and constructor admission, with a distinct
required public type. All policies/budgets are copied and revalidated, including
bypass-mutated frozen instances. Native work requires exact boolean `allow_ocp=True`.

The workflow reuses V2 `_read_step_owned_snapshot`, `_scan_binding`,
`_surface_binding`, `_rigid_transform` and `_publish_measurements` directly. It does
not monkeypatch an existing module, transform source AST at runtime or copy their
geometry calculations. Its new orchestration adds these bindings:

1. Unchanged V1 scan preparation and new `parse_triangle_mesh_ply_v4` read the same
   immutable PLY bytes. Their source hash, full raw-position arrays/digests and
   face cardinalities must agree. Triangle rows are exact original PLY indices;
   no coordinate welding, hole filling, mesh repair or silent triangulation occurs.
2. One owned STEP snapshot is read with the unchanged unit-transfer sequence.
   The same returned native shape is passed to both feature extraction and V2
   CAD triangle/sample export. No second native import or independently rescaled
   shape substitutes for either call.
3. Existing V2 surface binding verifies immutable complete triangles, import-local
   face/triangle IDs, barycentric sample incidence and the exact sample arithmetic.
4. V4 registration receives full raw positions and original triangles for plane
   extraction/prefiltering. If unchanged V1 preparation retained identity rows,
   `registration_raw_rows=None`; otherwise its already-bound explicit raw IDs
   select only the global registration cloud. No new sampling policy is invented.
5. Before using a successful pose, the workflow checks V4's input digests against
   the actual full raw points, triangles and CAD IDs. It separately binds selected
   registration positions/indices and canonicalized CAD initialization samples.
6. All original raw points are transformed for the unchanged local measurement.
   The existing V2 publisher keeps raw support IDs/digests and maps each measured
   finite endpoint back by `(point_CAD - t) @ R`. CAD feature proposals cannot
   substitute for scan-measured radius, sweep or finite carrier evidence.

These are in-process source/byte/array bindings, not authenticated native face
identity, scan calibration, material topology or manufacturing event identity.
Registration remains provisional SE(3) best fit, no fitted scale or stable datum.
Its deformation-absorption caveat, unknown observed bend count, indeterminate
conformance and production-use prohibition remain explicit.

Fatal input/native/sampling/registration/publication failures retain negative stage
evidence but no successful prediction prefix. A malformed successful registration
binding or transform is explicitly downgraded to unavailable before measurement.
Every full workflow payload must serialize as strict JSON within 16 MiB; otherwise
the entire detailed payload is replaced with a small explicit refusal envelope,
with no pose, prediction or measurement. This is not a passing abstention.

## Frozen cases, geometry and policies

The original `_asymmetric_native_strip`, `_owned_step_bytes`, `_analytic_mesh_scan`,
`_ply_bytes` and `_nominal_assertions` are called directly, unchanged. The fixture
is the independently constructed unequal-flange extruded bend, an analytic
partial inner scan plus remote cap patches, and a nonidentity quaternion pose.
CAD tessellation never supplies scan observations. The known pose is used only
to construct the scan and evaluate returned hypotheses/candidates **after**
inference; it is not a workflow argument or injected optimizer result.

| Fixed case | Required outcome |
| --- | --- |
| `v1_nominal_oblique` | One actual measured finite instance; all original error ceilings |
| `v1_missing_arc_oblique` | Zero predictions from a completed geometric abstention, not fatal/incomplete execution |
| `r0_common_wrapper_nominal_oblique` | One actual measured finite instance; all original error ceilings |

An abstention in either nominal case fails. Every case is retained even if an
earlier case, fixture or artifact fails. No true pose, case identity, reference
line or expected count enters the engine. Branch choice is the explicit declared
native-extractor control, not an inferred case category.

All four existing inference policies match the dense/V2 predecessor values
exactly: 32,768 CAD samples, registration row limit3,000, coarse/fine budgets and
iterations, trim/Huber/coverage/residual/rank/ambiguity gates, CAD deflection,
measurement windows/tangency/model and finite-instance support policy. The new
plane/extractor/budget defaults are the independently reviewed V4 source contract;
there is no result-dependent override. This is not the physical-development policy.

The original nominal assertions preserve maximum errors of:

- Radius: `2e-5 mm`.
- Absolute sweep: `2e-5 degrees`.
- Measured generator **and axis** endpoints: `.02 mm`.
- Recovered rigid-pose action on every original scan row: `.20 mm`.

These are invented-fixture algorithm ceilings, not manufacturing tolerances.
The common predecessor assertions are preserved except its initialization-specific
`candidate_count == 24` PCA requirement. An AST equality control proves that sole
change; no result is relabeled or padded to pretend24 starts occurred. Other
source controls prove unchanged policy values/globals and direct original helper
calls, with no native import during ordinary collection.

## Completion and the missing-arc negative control

For nonempty surviving V4 start groups, every group must be attempted and have
the complete original V2 geometric diagnostics. The matrix requires zero candidate
numerical failures, no unattempted group and `candidate_evaluation_complete=True`.
This is a prospective positive-evidence requirement; runtime V2 numerical-refusal
and acceptance rules themselves are unchanged. There may be1–24 exact start
groups; the test does not demand24 or merge nearby starts.

Every retained prefilter hypothesis must be fully evaluated, with all three
support denominators and successful/attempted query counts equal to the complete
original patch supports. No kernel/input/budget/report failure or incomplete
hypothesis can satisfy the sentinel.

Root froze a separate allowed negative-control disposition **before execution**:
if initialization is `COMPLETE`, every hypothesis has a completed geometric
rejection, and there are zero surviving starts, the missing-arc case may pass as
`PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE`. It must carry the exact geometric
no-start reason and zero attempted V2 candidates/numerical failures. This is not
a claim that a missing bend was detected or that absence/coverage is known. A
completed global-registration/local-support refusal is retained distinctly from
this prefilter disposition. Nominal cases cannot use this exception.

The actual native worker calls `_missing_arc_assertions`, which whitelists exact
geometric outcomes rather than accepting any nonempty reason. The registration
branch allows only completed no-supported-observable-pose or generated-pose
ambiguity refusals, plus the separately declared completed no-start prefilter.
It requires the exact matching workflow reason and no published pose. In
particular, `INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT`,
`WORKFLOW_EXCEPTION:*`, unknown/mixed reasons, and input/kernel/numerical/resource/
report failures cannot pass even if every candidate was previously completed.

The local-support branch requires a still-valid conditional pose/binding, exact
`COMPLETE` / `NO_FINITE_SCAN_SUPPORTED_PREDICTIONS` workflow state, complete
measurement computation, every nonempty native feature/window ID and denominator,
consistent measured-window counts, and no finite instances. Feature and finite-
support reasons have **separate explicit geometric allowlists** in the test.
Missing, unknown or mixed reasons fail closed. Solver-incomplete, linear-algebra,
floating-point, support-budget, invalid-input and undefined-carrier outcomes are
not geometric negative controls; `computation_status=COMPLETE` alone is inadequate.
This is a pre-run evidence-audit correction, not a change to geometry, engine
acceptance thresholds, any fixture, or resource budgets. Pure injected-report
regressions exercise the same helper called by the native worker.

Consequently a possible3/3 result means two required connected positive measurements
plus one safe negative-control abstention; it does not mean perfect missing-bend
detection, physical F1 or factory readiness.

## Frozen external and artifact budgets

One serial heavy job, unchanged from the V2 native successor:

| Resource | Frozen cap |
| --- | ---: |
| Wall time | 180 seconds |
| Aggregate CPU | 120 seconds |
| Process-tree RSS | 2 GiB |
| Stdout/compact manifest | 1 MiB |
| Full per-case JSON artifact | 16 MiB |
| Three full case artifacts combined | 48 MiB |

The artifact allowance is separately prospective: thousands of full hypothesis
and assignment records must not be discarded merely to fit stdout. It does not
increase CPU/RSS/time budgets. All V4 internal preflight/build/query/start/report
caps and V2 cumulative registration/query budgets remain unchanged.

The root supervisor supplies a **fresh, nonexisting absolute `ARTIFACTS` child**
under its newly retained job directory to
`run_invented_mesh_workflow_planes_v4_checks(artifact_directory=...)`. The worker
does not accept a CLI path or choose another output/raw directory. It creates
only that new child and exclusively writes `case_00.json`, `case_01.json` and
`case_02.json`. Each complete record is size-checked before file creation,
flushed/fsynced and verified against its SHA-256 and exact byte length. Existing
paths are never overwritten. Failed/partial files remain explicit negatives.

Each case artifact retains the full workflow payload, all initial hypotheses,
all refined candidates, post-inference action errors, fixed expected denominator,
failure/phase and original nominal error results. Files contain case-assertion
status only; matrix qualification additionally requires the final source/artifact
manifest. No scan/STEP fixture bytes or physical data are saved by this writer.

Only after all artifact writes/checks does the worker return a compact manifest,
including relative artifact names, hashes/sizes, every case result and a separate
`matrix_passed` flag. A case/report/artifact/source-consistency failure cannot
produce a positive matrix flag. No successful prefix substitutes for an incomplete
case record. Report overflow is a retained refusal, never a passing missing arc.

Native pytest requires both root-controlled environment entries:
`SHERMAN_RUN_INVENTED_MESH_CONNECTED_PLANES_V4=1` and
`SHERMAN_MESH_CONNECTED_PLANES_V4_ARTIFACT_DIRECTORY` naming that fresh child.
Ordinary collection skips native execution. The small source/helper controls may
use their existing60 s wall /45 s CPU /1 GiB tier; they do not qualify native work.
The root's external process-tree supervisor—not these source counters—enforces
all execution limits and records before/after source hashes and terminal outcome.

## Source closure and stop conditions

Source-only dependency hashes read while authoring:

| Preserved source | SHA-256 |
| --- | --- |
| `cad_guided_mesh_offline_v2.py` | `fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf` |
| `cad_guided_mesh_offline_v1.py` | `55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba` |
| `cad_guided_mesh_topology_v4.py` | `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734` |
| `cad_guided_plane_initialization_v4.py` | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| Original connected native fixture | `11455ceda7283bcd1b84dbd037b7bd16bd2517bd8d44985ef8f774d2e5c100e8` |
| Dense successor fixture | `396bd23361fc38e3b7871f9b8bfdb4ccb00a9ec6fe180b43cf1891ab200a76ea` |
| V2 finite successor fixture | `e28ed83f8a6a61975e8ac183922fc5a08c00ff9cc5e23f5f99d202ea49f6881c` |

The integrating runner must also pin the new three files and the entire reviewed
native/parser/sampler/plane/triangle/registration/measurement/instance dependency
closure. This table is not execution attestation. Future source changes require
review and an appropriate new closure; no active run may change underfoot.

Stop and retain a negative if any budget, missing dependency, native error,
source drift, candidate overflow, unsupported geometry, report/artifact failure or
fixed nominal assertion occurs. Do not raise a budget, shrink a fixture, relax an
error gate, drop a case, cluster near-identical starts or rerun the terminal job
from this protocol alone. Independent source and focused checks precede native
execution; positive constructed evidence precedes any separately authorized
physical-development run. No source-only result is promotion or Gold-F1 progress.

## Current handoff

Only the three new authorized files were authored for this step. No tests,
Python/native import, simulation, real scan/CAD/Gold/held-out/email access or
physical run was performed by the author. Root owns the next independent audit,
focused source controls and any later frozen native execution.

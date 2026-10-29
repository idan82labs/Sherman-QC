# Native diagnostic integrity ledger — source-only design, 2026-09-11

Decision: add **one same-worker supplemental pass after durable CASE_RESULT and
after the unchanged serial _evaluate attempt**, with no new read, native/fixture call, inference,
resampling, registration or refit. Keep the original 39 outcomes and all original
accuracy assertions. Publish a separate <=32 KiB ledger in the existing native
receipt/OUTPUT; preserve four scientific and four administrative files.

This design is not implementation qualification, an exact PRE or execution
authority. Only this document is owned/changed here. Named source and narrative
were read; no target import, test, geometry, physical/control/attempt/archive
read or execution occurred. Connected terminal details and the new allocation
are parent-reported/released narrative; their independent POST is separate.

## Integration interface and immutable meaning

Use the agreed new helper and independent pure test:

- tests/native_depth_diagnostic_ledger_v1.py (not collected as a test module).
- tests/test_native_depth_diagnostic_ledger_v1.py.

Expose:

    evaluate_native_diagnostic_v1(
        fixture_module, record, result_artifact,
        cad_blob, scan_blob, raw, triangles, truth
    ) -> bounded JSON ledger

The last argument is **ignored** solely for compatibility with *fixture.
Integrity admission must never load/read truth, desired radius, endpoints, labels
or oracle poses. Its source test must prove that the truth parameter is unused.
All accuracy testing stays in the unchanged original _evaluate.

Root's _diagnostic_evaluate wrapper may import this helper from the already
source-bound inert tests package and supply sys.modules[__name__]. Declare both
new files in the exact source closure; do not create an unbound scripts package.
Expected policies come from the fixture's fixed _policy_objects constructors,
not returned policies. Existing source-helper modules are already loaded during
fixture construction; do not trigger an old test or workflow through them.

Order in run_single_native_byte_case:

1. Existing source/fixture work, input publication, exactly one V13 invocation
   and post-return source check.
2. Existing complete immutable CASE_RESULT publication.
3. Original _evaluate and its ordinary error handling, unchanged.
4. Supplemental ledger, once, irrespective of ordinary accuracy success/failure.
   Its failures are recorded separately; no integrity success is synthesized.
5. Existing status/positive_geometry_passed/oracle_error semantics, plus separate
   diagnostic_ledger and diagnostic_error fields. Existing native test prints
   the receipt before its unchanged assertions.

The original native qualification can remain FAIL/RED even when the supplemental
integrity ledger is complete. Never set the old positive_geometry_passed from
the new ledger, change a failed oracle to PASS, or catch an accuracy error as
success. Budget interruption/SystemExit/KeyboardInterrupt must not be swallowed
to manufacture a completion. Source/execution/retention failure yields
NOT_EVALUABLE and no scientific predicate call.

Root reports an additive wrapper draft now exists, native test 49,261 B SHA256
fbd2b8e55dedf5d95416ccb0b85f89844166190f3abdd829e37618af23411ee1,
with original _evaluate/native assertions/construction/policies AST-preserved.
That new wrapper is not independently qualified by this design.

### Pre-freeze order correction

Independent integration review identified that running the helper before the
original accuracy oracle could expose that oracle to helper-mutated objects,
even if the helper subsequently detected mutation. Root therefore adopted the
safer order above before any native execution: original accuracy attempt first,
then the always-attempted supplemental ledger after its ordinary caught failure
or success. Both remain after durable original RESULT. BaseException/global
budget interruption remains terminal/unknown, not swallowed. The helper's
initial publication binding detects any mutation by the original oracle;
its final record/raw/triangle hashes detect supplemental mutation. No deep copy,
file reload, normalization, inference or fit is added. The original accuracy
verdict cannot be changed by the later helper.

The prior18576-byte design, SHA256
`ded54547300d4f848e7ecd089859cd6e3666745f622514f5be529ec4d45eea7d`,
is preserved as one exact regular payload in
`cad_engine_v1_synthetic_checks/native_diagnostic_order_pre_amendment.o72duw2e/SOURCE.tar`,
20480 bytes, SHA256
`e34bec20166c31443cd25bc518f3dfd213b0395f61d3ad66ddc10d324fe23250`.
This correction supersedes the preceding parent-reported wrapper checkpoint's
order only; the original accuracy assertions and separate integrity meaning
remain unchanged. The new integration review and exact PRE still must qualify
the final implementation, not this design alone.

## Why an independent predicate pass is necessary

The original native _evaluate first calls a historical binder requiring
PARTIAL/COMPLETE. Actual V13 assigns output.status=measured.status: a completed
cover computation with no measured group is UNAVAILABLE/COMPLETE. It therefore
fails the old positive requirement before later integrity checks.

For a PARTIAL result, line 321 in the pre-wrapper source calls truth-based
measurement assertions in a list comprehension. The first failure skips later
groups, later raw-frame/uncertainty/nominal-deviation checks and endpoint oracles.
No caught AssertionError, oracle=None or receipt FAIL establishes those later
checks as passed.

Do not refactor or weaken that oracle. Isolate common integrity predicates in the
new helper; leave positive/nonempty/accuracy requirements untouched in the old
functions. For copied checks, independent review must map each predicate to its
source and explicitly identify which nonempty/positive requirement was excluded
from *the supplement*, not removed from the original. No status-spoofing shallow
copy from UNAVAILABLE to PARTIAL.

## Eight independent criterion groups

Each group has its own PASS/FAIL/UNREACHED outcome, prerequisites, exact
planned/visited/failed counts and bounded errors. Evaluate every applicable
record; a first bad record cannot leave later records silently marked passed.
Continue independent safe criteria after an ordinary failure. A malformed
required dependency yields UNREACHED with its exact missing boundary, not PASS.

| Group | Exact reuse/isolation and required checks |
| --- | --- |
| 1. Publication/source/input binding | Require complete retained workflow, source_consistency and no execution/publication error; bind record and all supplied bytes/arrays to result_artifact and recorder input_binding/input_artifacts. Confirm the source-bound inference seam accepts only bytes/fixed policies. Actual file identity is completed by parent inventory/POST, not claimed reread here. |
| 2. Raw/topology/native/sampler identity | Split common byte/native predicates from tests/test_cad_guided_connected_v12_native_bytes.py:_assert_byte_and_context_binding. Check scan/CAD and little-endian XYZ/triangle digests, finite 650-row/1,176-triangle shape/index/count identity, raw/measurement/source hashes; native adapter, units/action/provisional calibration, declared feature IDs/counts, 32,768 sampling request and sample/triangle/face/source-triangle report bindings. |
| 3. Context/origin/transform | Isolate the historical binder's complete index/pose/origin loop. Validate each selected context, initialized/exact start-group/hypothesis relations, prefilter origin, pose SHA, ALL_RAW_ROWS transport and no accepted/global-unique pose. Use existing cad_guided_mesh_offline_v2._rigid_transform (1e-8 serialization SE(3) guard), not a truth-fitted transform or new tolerance. |
| 4. Composite and schedule/profile | Reuse _assert_composite_binding/_assert_composite_metadata and _assert_schedule_metadata when their actual report prerequisites exist; reuse V12 _two_plane_schedule_binding for complete equation/match/reservation validation. Check fresh source-requested maps at all publications, bool/int distinctions, all lane/proposal/origin/support digests and every binding row. Add the exact context-V2 profile/joint-slot/input-digest predicates from V12, not only repeated schedule hashes. |
| 5. Raw-support/exact-cover proof | Call unchanged validate_support_cover_publication_v2 once with supplied actual raw XYZ and scan SHA; require exact equality with the retained publication-validation field. It validates nested context/filter/association ownership, raw roles, exact cover/witness/permutations, all group states/policies/counts/visit accounting and exact projection. All-empty/all-refused is valid only if that full proof actually passes. |
| 6. Window/nominal associations | Isolate the original pre-accuracy window and association predicates: exact context×native-feature window set, every original support record and every group association, existing native IDs/source tags, unchanged nominal radius/sweep and canonical signed deviation algebra wherever a measurement exists. Check all groups, not only the prefix before accuracy failure. |
| 7. Quantity/carrier/output semantics | Reuse unchanged _assert_top_level_cover_projection (it accepts a valid empty projection), publication validators and the original post-accuracy semantics predicates. All applicable quantities/axes/endpoints must be finite and self-consistent under existing policy; preserve raw frame, sampled—not physical—trim, unknown count/pose, uncalibrated intervals, indeterminate conformance and accepted_registration=False. Top-level outputs equal exactly supported cover-V2 instances, never nested-member union. Refused groups have no unauthorized output. |
| 8. Branch/denominator/no-prefix closure | Validate the exact V13 phase/status/computation/reason path, complete criterion/group/window/native denominators and no omitted late record. Distinguish complete no-measurement from early refusal, discarded report and exception. Preserve all original scientific outcomes separately; emit no accuracy PASS for unexecuted assertions. |

Group 4 may contain composite and schedule subrows but remains one fixed criterion
group. No new helper calls the original truth-based assert_scan_derived_measurement,
assert_localized_raw_carrier or _evaluate. The original accuracy function remains
responsible for its own assertions and can still stop early; unexecuted accuracy
checks remain unknown, not integrity successes.

### Limits of reusable validators

The report-only _two_plane_schedule_binding rebuilds equation digests and
validates declared schedule/match/reservation structure; it does not rerun the
scheduler or geometry. Composite decoding checks published metadata, not truth.

Do **not** replay full _registration_binding with placeholder CAD arrays or
digest-only stand-ins. V13 removes native triangle, sampler-point and source
arrays from its public CAD sampling result. Runtime checked those arrays before
exposing contexts, but the retained report cannot reconstruct them from hashes.
Label assurance REPORT_BINDING_PLUS_SOURCE_ENFORCED_RUNTIME_CHECK,
not NATIVE_ARRAY_RECOMPUTATION. Exact source/returned bindings are evidence of
that narrow contract, not an independent native tessellation/ownership rebuild.

validate_support_cover_publication_v2 recomputes the raw-set cover proof and
numeric/serialization/coverage-declaration consistency. It explicitly does not
repeat the circle fit or independently prove physical role identity, residual
truth, continuous surface coverage or measurement accuracy. Do not rename it
a new fit or metrology qualification.

## Supplied bytes and actual durable identity

The helper performs **no filesystem read**. It receives the source-bound values
already used by the native case. Recompute CAD/scan and little-endian raw/triangle
hashes and compare recorder input bindings, input_artifact hashes/byte counts
and all applicable workflow fields. Validate shapes/dtypes/ranges before any
array copy; 650×3 and 1,176×3 are fixed native-fixture scope, not a sampling change.

Bind record to the already returned CASE_RESULT publication digest by streaming
the exact existing compact sorted JSON encoding, counting/hash-updating fragments
without retaining an additional 129 MiB byte buffer. Repeat the streaming digest
after predicates to detect mutation. This avoids a second decoded workflow.
Admit only the ordinary types handled by the existing serializer (including
ordinary tuple/list containers where used), preserve scalar types, reject
nonfinite values and bound traversal; do not normalize report contents.

This checks supplied-object/publication-hash consistency. It does not prove the
file was written correctly or reread it. Parent's quiescent four-file inventory
and independent POST must bind actual retained bytes/hash/size to those exact
publication receipts and the one ledger. A short/failed write or changed file
vetoes the diagnostic even if the same-worker ledger looked consistent.

Future retained-only audit can decode the exact PLY/JSON under separate authority
and run the fixed pure predicates without another engine invocation. For sorted
composite metadata, validate the exact codec and restore only the declared outer
format,schemas,root order before source-digest replay; label that serialization
reconstructed, not original raw-array recovery. Missing native arrays stay missing.

## Positive, complete-no-cover and early-refusal admission

Use explicit finite dispositions:

- COMPLETE_CONDITIONAL_INTEGRITY: all applicable common/native/context/support/
  publication criteria completed; all groups/predictions checked. Original
  accuracy can remain failed.
- COMPLETE_NO_MEASUREMENTS_INTEGRITY: measured computation COMPLETE,
  status UNAVAILABLE, zero measured groups/empty top-level predictions and exact
  full refused-group/cover proof verified. Native/context transport must be
  complete. This may describe NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS;
  neither the reason string alone nor a source expectation proves this branch.
- SAFE_EARLY_REFUSAL_LIMITED: exact valid earlier return, no downstream output,
  reached-stage evidence checked; later rows explicitly
  NOT_APPLICABLE_BY_VALID_REFUSAL. It does not qualify the positive/full native
  route. Return the precise missing boundary for the next independent admission.
- FAIL: an applicable predicate/denominator/bound/immutability check fails.
- NOT_EVALUABLE: complete source-bound durable workflow or another prerequisite
  is unavailable.

The ledger's overall complete state requires all applicable criteria to have
actual outcomes; no FAIL/UNREACHED required criterion can yield complete integrity.
REPORT_PUBLICATION discard, WORKFLOW_EXCEPTION or MEASUREMENT_REFUSAL is not
automatically complete-cover evidence. Do not fabricate absent arrays/groups to
satisfy a test or infer physical missing bends from no measurements.

Per-criterion error detail can retain the first eight examples, but failed
counts and visited counts must cover the full admitted record set; mark detail
truncation explicitly. Exceeding any cap yields FAIL/NOT_EVALUABLE, never a
successful prefix. Do not use truth or part identity to filter bad records.

## Bounded work within the proposed native allocation

Proposed supplemental ceilings, to freeze before execution:

| Item | Bound |
| --- | ---: |
| Supplemental CPU admission, within native total | 90 seconds |
| Supplemental wall, clipped to remaining global work | 120 seconds |
| Ledger canonical JSON / fixed criterion groups | 32 KiB /8 |
| Maximum detailed error examples per group |8 |
| New filesystem reads / engine or fit calls |0 /0 |
| Supplied STEP /PLY /record encoding |8 MiB /1 MiB /129 MiB per record pass |
| Record stream binding passes |2, before/after |
| Ordinary record traversal |8,000,000 nodes; depth64 |
| Contexts/native features/windows |48 /64 /4096 |
| Source records/groups/predictions |512 each |
| Existing cover proof |4M graph visits,4M links,32M cover-row checks,262144 comparisons |
| Existing context/cover publication |existing separate4M associated/refit/publication-row guards |
| Overall sampled RSS/whole OUTPUT |unchanged4 GiB /1 MiB |

Node/depth preflight must be iterative and bounded before invoking nested
validators. Validate list lengths and prerequisite types before all-record
loops. Existing decoder/validator bounds remain; do not claim an exact new
aggregate counter for their internal uninstrumented JSON passes/CPU.

90-second CPU admission is checked at phase boundaries; it is not a separate
OS sandbox. A nested-safe 120-second deadline must not weaken a shorter global
timer, and the worker's whole RLIMIT/supervisor remains decisive. Check remaining
budget before each phase. Overrun/interruption cannot qualify the ledger.

Root's prospective [allocation](2026-09-11_two_plane_depth_native_diagnostic_allocation.md)
is 2,520 CPU/2,815 work/2,820 parent:
ceil((1.5×(1462.253261+1.437055)+300)/60)×60.
Charge supplemental 90 CPU/120 wall inside—not in addition to—the 300-second
non-registration allowance. The remaining nominal 210 CPU seconds is not measured
sufficient time for fixture/native work. Neither 50% headroom nor 295-second
work/CPU wall difference guarantees completion; no 115× sample multiplier or
second invocation is introduced. Any final cap failure stays terminal.

## Retention, tests and final review

Keep CASE_START/CAD.step/SCAN.ply/CASE_RESULT and their existing
138 MiB+4096 B aggregate unchanged. Never reopen or append CASE_RESULT.
The existing STARTED/OUTPUT/ARTIFACT_INVENTORY/COMPLETED admin layout is unchanged.

The new <=32 KiB diagnostic_ledger carries schema/version, result/CAD/scan/raw
hashes, fixed-request hash, helper identity, eight criterion outcomes/counts,
exact applicability/failure boundary and supplemental resources. No whole report
copy or self-referential hash. diagnostic_error is bounded and separate.

The original native test prints its receipt before asserting accuracy. Thus
OUTPUT can retain the ledger even when original science stays FAIL. Independent
POST must find exactly one complete matching receipt/ledger, verify actual four
scientific bytes/source/resource/process custody and retain every old outcome.
Missing/duplicate/truncated output or an oversized traceback is a veto; never
suppress an original assertion to fit 1 MiB. Add a pure output-envelope check.

Prefer a separate focused pure test file: fixed tests for complete conditional,
complete all-refused cover, limited early refusal, wrong bytes/late malformed
records, pose/origin inconsistency, stale/bool-aliased schedule/composite fields,
projection/semantic/deviation contradictions, record mutation, bounded work/
output failure, zero additional native/inference/fit calls and untouched truth.
Use spies/invented records, not another engine execution. A fixed 12-scenario
matrix is sufficient as a proposal; its actual parametrized denominator must
be independently counted before freeze.

The original 39 nodes/assertions remain unchanged. Declare the supplemental test
denominator separately; if exactly 12 new outcomes are authored, total would
be 51, not 39. No predicted test count is a pass. Root must add helper/test to the
full exact source closure, read all new sources, mechanically compare unchanged
original oracle AST, update prospective literals/metadata and obtain fresh
independent PRE/archive. Existing mocked recorder tests may stub only the new
ledger for their invented workflow, preserving their original assertions.

Current reference source before root's wrapper: native tested-source draft
ed2acb8dde5881e1f36697bec056b672b2a48e25612559612feb88d38fd12974;
harness f9b145c95f1586f8b72726570813975d0808bb269ee39acd341e305ecb6ea6d9;
protocol 013dfc4c2f80455eac2925e66da859b6b820a57605aed9ac87dcd8b9a3da5e48.
The actual new final closure/hashes and timing guards need revalidation.
No preservation archive was inspected or independently attested in this design.

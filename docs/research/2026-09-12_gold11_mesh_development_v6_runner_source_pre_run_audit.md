# V6 runner: independent exact source PRE

Date: 2026-09-12. Decision: GREEN for the one exactly frozen invented-only
132-outcome gate below, subject to root's exact PRE/archive acceptance and
separate no-overlap check before the sole invocation. No test has run in this
review. GREEN is a prospective engineering gate, not a passing result or
physical execution authority.

This audit is for exactly 132 invented detached-AST boundary outcomes, not V6
physical execution, actual backend/sampler execution, a native process test,
scoring, the separately planned consumed diagnostic, or full11 release. The
reviewer authored neither the candidate, tests nor harness. The prior independent
runner audit was this reviewer's source review; it was not a runtime gate.

## Scope and actual checks

I read canonical AGENTS and current status, the entire 987-line test, entire
new harness, complete protocol, test-author review and accepted sampler POST.
The full V6 design/current affected functions were independently reviewed in
the bound independent audit. This PRE additionally read the inherited three
supervisor definitions and current schema/inventory paths used by the mocks.
No old attempt, archive, control, private cache, raw, Gold, held-out or physical
input was read, searched, statted or hashed. No target import, compile/exec of
selected target ASTs, collection, pytest call, native API, thread, process
observation or signal occurred. Only named-source reads, AST parsing/counting,
hashes and stdlib-only ordinary runtime metadata were performed.

The exact application Python was used with -I -S -B for independent AST/hash
and runtime inspection. It did not import the harness, runner, pytest, backend,
supervisor, sampler, scorer or projection. Pytest distribution metadata and its
ordinary initializer bytes were inspected without executing that initializer.
The root package initializer is absent; tests/__init__.py is the 24-byte inert
comment '# Sherman QC Test Suite'. The harness installs its own inert tests
namespace, not this initializer.

## Resolved narrow protocol discrepancy

The new parent correctly guards its OUTPUT parsing, source/runtime/PRE
postflight and final inventory behind process_group_confirmed_empty=True.
However, the unchanged extracted short-worker supervisor itself always computes
its output_sha256 by reading OUTPUT.txt at return, including when its group
empty check failed. This log descriptor has already been closed by its sole
supervisor writer; the tests do not write the retained log directly. It is not
a private-cache or scientific-result read. The initial protocol's broad sentence
that output/postflight reads require confirmed empty writers was stronger than
the inherited helper. Root narrowed that statement before PRE: the inherited
closed-log hash is explicit, while only the new harness's subsequent parsing
and content postflight are claimed quiescence-gated. I read the entire corrected
protocol and independently rebound all15 sources. No helper, harness, runner or
test changed. This resolves the documentation discrepancy without changing the
frozen helper or implying generation-aware custody for it.

## Exact source closure

At the final complete check: 15 unique files, 559092 bytes. Every pinned
dependency matched. Each Python source parsed as AST. The new harness and
protocol are also bound in the source map even though they cannot self-pin
their own hashes. Canonical JSON means sorted keys, compact separators,
allow_nan=False, UTF-8, no trailing newline.

Canonical path-to-SHA256 map digest:
`68d0e6cb490316fa44add620f6987eae14ad70a385e5d372112001fa58a8706b`.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| scripts/check_gold11_mesh_development_v6_runner.py | 24509 | 4fb3c4b48b3e3af599f59e090110d72ae668751f7275de953ef9fc0273208a76 |
| scripts/check_cad_guided_connected_v12.py | 19437 | 0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694 |
| scripts/run_gold11_mesh_development_v6.py | 166049 | 7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80 |
| scripts/run_gold11_mesh_development_v5.py | 153975 | 71d55915db6e6fcb8ae7f8c658de7d7e0c4b93b553eb78413f3914b037e4e481 |
| backend/cad_guided_two_plane_context_v1.py | 36075 | 9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a |
| backend/cad_guided_two_plane_context_v2.py | 11130 | d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663 |
| backend/cad_guided_two_plane_context_v3.py | 25845 | 9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d |
| scripts/gold11_engine_stage_sampler_v1.py | 8142 | 80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0 |
| tests/test_gold11_mesh_development_v6_runner.py | 52033 | dbe4ea245acbb2f93cd0b8cf8fe4e59620b4f5ce68aad7a85cc20c759a49f3f5 |
| tests/__init__.py | 24 | 5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b |
| docs/research/2026-09-12_gold11_mesh_development_v6_runner_design.md | 17868 | a943ad18577f1a4ee450227c9e879215fccb218dc94a167b0979ab17b96747f2 |
| docs/research/2026-09-12_gold11_mesh_development_v6_runner_source_review.md | 14771 | 39beb1b503bc158548e60e899fa3638c184c41864c05035bead19f18da39f080 |
| docs/research/2026-09-12_gold11_mesh_development_v6_runner_independent_audit.md | 11785 | a344b145282dbe1f8a096fcd7fef56d3eb6609eca8110016140b2165d89eaf50 |
| docs/research/2026-09-12_engine_stage_sampler_v1_source_post_run_audit.md | 9040 | 2fca751d05065eb053a3def327bbdbf83963690a23259d4809416788d7bda17b |
| docs/research/2026-09-12_gold11_mesh_development_v6_runner_source_protocol.md | 8409 | cc7bd96740309e533422ff659b1875f60c218567a9db7f6c79ed1b490c85e17d |

The test-time source-read allowlist is only V6, frozen V5 and context V1/V2/V3,
all individually pinned and presently below 256KiB. The harness separately
binds the 15 files above with bounded no-follow regular final-component reads.
Source path/descriptor identities are rechecked; this is not an atomic hostile
filesystem snapshot or a complete transitive runtime archive.

## Independent denominator and fixture reachability

Direct AST inspection of literal decorators and the three named parameter
tables gives 29 functions / 132 prospective outcomes. I did not collect tests.
The two sampler decorators form a six-state by two-engine-outcome cross product.

| Test function suffix after test_ | Outcomes |
| --- | ---: |
| preserved_science_limits_and_reference_pins | 9 |
| preserved_returned_diagnostics_reference_selector_and_scorer | 3 |
| release_guard_precedes_every_entry_effect | 16 |
| local_guard_copy_has_no_truthiness_or_single_flag_bypass | 6 |
| local_guard_only_copy_rejects_invalid_timing | 4 |
| full_context_request_and_original_bytes_reach_v2_projection | 2 |
| context_maxima_are_source_grounded_without_backend_import | 1 |
| both_successor_constructor_paths_use_explicit_v3 | 2 |
| constructor_effective_declaration_drift_refuses | 3 |
| strict_v3_quiescence_table | 20 |
| fresh_leaf_cache_cleanup_is_quiescence_and_retention_gated | 6 |
| cache_identity_rejects_nonprivate_or_rebound_path | 6 |
| fresh_cache_has_exact_external_creation_scope_before_dispatch | 3 |
| retention_refuses_before_private_enumeration | 3 |
| retention_positive_reads_only_after_receipt_and_fresh_identity | 2 |
| supervisor_wrapper_latches_custody_before_call_and_blocks_next_leaf | 3 |
| coordinator_highwater_is_finite_platform_bound_not_process_table | 8 |
| score_refuses_before_reference_or_bundle_access | 2 |
| scorer_failure_always_attempts_receipt_and_never_cleans_uncertain_bytes | 4 |
| coordinator_abort_never_dispatches_second_leaf_or_scores | 3 |
| outer_postflight_and_all22_fill_follow_whole_group_quiescence | 3 |
| full_schema_preserves_scientific_cap_and_excludes_private_cache | 1 |
| unknown_cache_name_is_metadata_only_veto_never_recursed | 2 |
| no_automatic_cache_finalizer_and_no_diagnostic_release | 1 |
| sampler_preload_closure_and_worker_order_before_backend_hook | 1 |
| unchanged_backend_hook_has_no_broad_scripts_read_exception | 3 |
| sampler_hash_failure_stops_worker_before_hook_input_or_entry | 1 |
| sampler_missing_or_wrong_preload_refuses_before_entry | 2 |
| sampler_diagnostics_cannot_replace_engine_result_or_exception | 12 |
| Total | 132 |

The detached sandbox admits only explicit assignment/function AST allowlists.
Imports and open default to an AssertionError sentinel, so an unintended import
or I/O call cannot satisfy a ValueError expectation. LexicalPath resolution is
inert; fake metadata objects replace all tested stat/identity operations.
Operations use an explicit guard spy while the copied candidate flags remain
False/unreleased. Guard-only copies test local invalid values without reaching
process entrypoints. No real constructor, geometry or policy validation is
performed by these fixtures; the source review of actual call compatibility is
separate from these wiring mocks.

Specific earlier-failure checks:

- All 16 entry definitions first call require_release after any docstring/global.
  Their generated positional/keyword arguments match their AST signatures; they
  stop with V6_DRAFT_BLOCKED before an effect. Flag controls distinguish exact
  True from truthy 1 and then require the timing sentinel. The four timing
  mutations affect Boolean CPU, nonfinite wall, fractional integer CPU and an
  invalid reserve. No backend policy exists on those paths.
- Both ordinary constructor paths use the complete unchanged CONFIG before
  three independent declaration drifts. Context identity and V1-default drift
  hit FROZEN_CONTEXT_V3_OR_SCHEDULE_DEFAULT_BINDING; V2 drift passes the earlier
  context/default check and hits EXPLICIT_COMPOSITE_OR_SCHEDULE_CONSTRUCTOR_BINDING.
  Positive controls cover both arms. The actual source registration policy is
  .25 <= .5 <= 2 mm; no invalid invented Huber/P95 constructor is copied from the
  prior failed numerical integration fixture. The spies intentionally validate
  exact arguments, not real dataclass invariants.
- Quiescence begins with one complete clean V3 receipt. Each negative mutates
  one relevant field. Leaf group None is expressly admitted; outer False/int1
  is refused. False-vs-int0 and discarded-count bool/float are tested distinctly.
- Cache cleanup tests expose body completion, identity and remove ordering.
  Unknown/retention-failed states permit only creation and body events; identity
  rebound forbids remove. Cleanup failure requires a false final cleanup flag.
  Positive complete/no-result retention controls require quiescence, identity,
  enumeration and metadata in that order before any read/publication.
- Supervisor exception leaves the quiescent latch false; a second call must
  refuse before the mock supervisor. Memory tests intentionally distinguish
  finite nonnegative Darwin lifetime HWM, including zero, from current RSS;
  wrong platform prevents getrusage. They do not weaken V3 native RSS validation.
- Scorer fixtures pass matching all22 terminal hashes and request validation
  before their four injected faults. Assertions require an attempted scorer,
  final receipt, false cleanup, no SCORED and no forbidden retention/removal.
  Coordinator fixtures bind token, parent/cwd/environment/cache/start hashes
  before the first case. Abort assertions require exactly one dispatched leaf
  and all22 durable-or-explicitly-nondurable positions, not any caught exception.
- Outer clean and failure fixtures supply matching frozen/start/dispatch hashes,
  a fresh external cache and finite resource data. The unknown-group case must
  reach the quiescence predicate but none of the content operations. Source
  drift must reach the source audit and remain an explicit failure after all22
  terminal fill, with no FINAL_GATE. The clean control reaches final publication.
- Full-schema fixture keeps11 invented cases, two arms, 44 required case terminal
  paths and46 dispatch/claim slots. Unknown names trigger metadata-only status,
  read_bytes=0, one scientific charge and no recursion. All absent declared
  directories fail through inert mocks. This is neither maximum-envelope nor
  physical artifact qualification.
- Hook tests retain the actual backend-only predicate with inert lexical path
  operations. Sampler hash failure occurs in the real extracted loader before
  module construction/compile or hook installation, verified by exact events.
  Missing/wrong preload fails before engine entry. Twelve wrapper controls
  require the same single call, argument/object identities and exact engine
  exception identity, plus finish/write ordering, independent startup-failure
  metadata, <=4096-byte framing and whole oversized-record omission.

These checks found no hidden earlier constructor/keyword/policy failure that
would invalidate this denominator, and no wrong-reason pass blocker. Some tests
use broad exception classes but their path/event/terminal assertions constrain
the intended failure. This is a source conclusion, not a prediction that all132
will pass. Unexpected runtime failures must remain terminal evidence.

## Harness and runtime gate

The harness demands the exact PRE SHA before creating a retained attempt. It
binds all15 sources and ordinary runtime, emits an exclusive STARTED request,
then invokes one serial application-Python worker with -I -S -B -u. Only the
three original supervisor AST bodies are extracted; historical module imports,
old suite selection and main do not execute. Its numerical process-group
supervision is the existing bounded short-worker mechanism, not the qualified
generation-aware V3 physical supervisor.

The worker sets RLIMIT_CPU before source/runtime/PRE checks and pytest loading.
No site/.pth runs; bytecode writes and pytest autoplugins/config/conftest/cache
are disabled. Only one exact absolute test file is passed. It imports stdlib
and pytest; backend, NumPy, SciPy, OCP and Open3D imports and root/backend
initializer execution are vetoed and final loaded-module inventory must be empty.
The audit hook does not claim a hostile-code sandbox; fixed-source review is
part of the gate.

The custom plugin requires132 unique selected items from the named test file.
It tracks ordinary setup/call/teardown reports per node, detects duplicate or
invalid phases and counts failure, errors, skip, xfail, xpass, deselection and
collection errors. All132 ordinary passing phase triplets and zero other
dispositions are required together with pytest exit0, source/runtime/PRE equality
and CPU compliance. A plain '132 passed' summary alone is insufficient.

Parent independently checks unchanged source/runtime/PRE even after ordinary
worker failure when group custody permits its reads. It parses exactly one
bounded worker marker, verifies the supervisor's output hash and requires exact
agreement among parent and worker postflights. Outcome and final publication
gates are distinct. Only exact STARTED.json, OUTPUT.txt and COMPLETED.json may
remain in the fresh retained namespace; changed or unknown names veto release.
The parent does not enumerate/read the retained private supervisor cache.

Fresh runtime object, independently observed without pytest/target import:

```json
{"distribution_versions":{"pytest":"9.1.1"},"executable":"/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11","executable_stat":[16777231,17404606,17257904,1783888533314804818,1783888533315379399],"implementation":"cpython","machine":"arm64","package_initializers":{"pytest":{"bytes":5644,"sha256":"ee37885c9583f843390684dfb496907f87a0905128e7a3cf4ad227f49ff34c2d"}},"platform":"darwin","python_version":"3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]","scope":"ORDINARY_LOADER_ENTRYPOINTS_METADATA_NOT_FULL_RUNTIME_BINARY_CLOSURE"}
```

Its canonical SHA256 is
`2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689`.
Executable stat is device/inode/size/mtime_ns/ctime_ns; pytest is its ordinary
initializer and installed distribution metadata, not full transitive binaries.
Any differing runtime object fails this exact PRE.

## Resources, release sequence and stop rules

Prospective worker caps:20 CPU seconds,25 work seconds,30 supervision seconds,
536870912 sampled group RSS bytes and1048576 output bytes. One worker and one
invocation only. Source reads are <=2097152 bytes each; admin records <=131072
bytes. Ordinary fixtures stay below100KiB. The sampler mock's oversized metadata
is5000 characters; no real512KiB stream, native output or daemon is produced.
Production sampler30-second cadence and V6 physical1860-CPU/3540-wall carryovers
are not exercised or released. Parent pre/postflight and synchronous publication
I/O are outside the30-second clock. Sampled RSS is not a hard allocation bound.

After the final exact PRE is accepted, root must create and verify a NEW external
archive with exactly16 unique regular payloads: these15 sources plus this PRE.
Names, sizes, hashes and byte equality must match without extraction/execution.
No archive was inspected or attested by this PRE. Root must separately verify no
overlapping heavy Sherman writer, then run the sole protocol command using the
accepted PRE digest. This reviewer performs no process check or invocation.

Any failure, skipped/xfail outcome, wrong count, resource stop, missing receipt,
source/runtime/PRE drift, unknown writer, publication failure or interruption
is terminal. Do not repair, recollect, rerun, drop cases, raise caps or reuse the
attempt based on this PRE. Preserve the original evidence and require a newly
versioned/frozen action for any successor. Independent POST must reconcile all132
outcomes, exact source/runtime/PRE/archive bindings, resources, actual parent
exit/final stdout and the exact three-file inventory before qualification.

Passing this gate would qualify only these detached invented orchestration
boundaries. The selected tests do not execute supervisor V3, policy dataclasses,
context numerics, sampler cadence, projection/scorer, native OCP or the physical
runner's publication/fsync and workflow. The harness separately performs real
short-worker supervision and its own administrative publication/fsync. This
gate neither repairs the terminal prior integration94 failure
nor repeats its successful dense case or accepted sampler26. No all-positive
geometry, calibrated uncertainty, physical score, deployment or full11 claim
follows. Only this new PRE is being authored; all frozen sources remain intact.

## Exact machine-readable decision

The marker below binds the final corrected protocol/source map and independently
observed runtime. It is exactly the harness's closed six-field PRE schema. The
exact file SHA supplied at invocation additionally binds every narrative byte.

GOLD11_MESH_DEVELOPMENT_V6_RUNNER_EXACT_PRE_JSON={"decision":"GREEN","expected_outcomes":132,"runtime_binding_sha256":"2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689","schema":"GOLD11_MESH_DEVELOPMENT_V6_RUNNER_EXACT_PRE","scope":"INVENTED_ONLY","source_map_sha256":"68d0e6cb490316fa44add620f6987eae14ad70a385e5d372112001fa58a8706b"}

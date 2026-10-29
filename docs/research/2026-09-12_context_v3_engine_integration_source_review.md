# Context V3 engine integration and dense-support source review

Date: 2026-09-12. Status: SOURCE-ONLY; 94 PROSPECTIVE OUTCOMES; NOT EXECUTED.

## Scope, authorship and disposition

This review covers initializer V13, workflow V14, their design, the new
93-outcome integration test source and one new dense-support context test. The
reviewer authored both new test files, independently of the implementation
author. It is therefore not an independent review of its own tests. Root reports
a separate full read of the 93-outcome source and the dense fixture/counter
derivations without a blocking finding. An independent exact PRE remains
necessary before execution. This document is neither that PRE nor release.

The bounded source review found no blocking production-binding defect in the
new integration delta. That conclusion does not qualify its runtime behavior.
Subsequent independent exact-PRE review found a test-bootstrap defect before
any execution: a private-prefix V5 policy and V7's absolute backend V5 import
created distinct dataclass identities. The strict production type check would
correctly refuse the fixture before its intended context stage. Root corrected
only the unexecuted test to use one coherent inert backend graph, preserving
complete module teardown and suppressing backend initialization. Production
classes/type checks and the94-outcome denominator remain unchanged. The hashes
below reflect this corrected pre-freeze source; renewed independent PRE is
required. This finding was not a failed physical or synthetic invocation.
The candidate/test files were not imported, collected or executed. Checks here
were source reading, standard-library AST parsing and source byte hashes only.
No physical inputs, prior attempt artifacts, private caches, control records,
native geometry or measurement fitting were accessed. Frozen files were not
edited. The only new files owned in this review turn are this narrative and the
one dense-support test.

The current-status top records supervisor V3 host 8/8 with independent POST in
progress; that finite host run did not exercise the actual INEXIT transition.
It also records context V3's accepted 76/76 POST, which must not be rerun.
Neither result executes the new initializer/workflow connection. The latest
approved sequence is a connected capacity qualification, then a separately
frozen small already-consumed diagnostic before a fresh full Gold11 score run.
No new score, physical accuracy or promotion claim follows from this review.

## Exact source identity and denominator

These source bytes were re-read and hashed on this review turn:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_plane_initialization_v13.py` | 38651 | `79ee4de537f1dddcc1dee40d33b25fe3fe8fb667bcd4e30ecc0a811ebe2d989f` |
| `backend/cad_guided_mesh_offline_v14.py` | 33435 | `76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749` |
| `tests/test_cad_guided_context_v3_engine_integration.py` | 42073 | `3b40f2a16a1a6ee48b19716473d57b64880e4f087059f7a7c3e14c370d2db76b` |
| `tests/test_cad_guided_context_v3_dense_support.py` | 11533 | `4ccde0dae1aa6d29c56d0fdfa23291a9b7de1ebdc2ad4093e8e6178b9163acf0` |
| `docs/research/2026-09-12_context_v3_engine_integration_design.md` | 12231 | `ea424a526797080cf717344e74b9d4ce1ca2ff391b566b5206d376a827d14985` |

The denominator is computed from literal test decorators by AST, not from
pytest collection: 20 integration functions expand to 93 outcomes, and the
undecorated dense-support function adds exactly one. Total: 21 functions and
94 prospective outcomes. The integration expansion is:

| Integration test suffix (after `test_`) | Outcomes |
| --- | ---: |
| `independent_fixture_and_frozen_numeric_aliases` | 1 |
| `actual_initializer_scheduler_proposer_finite_prefilter_chain` | 2 |
| `configuration_owns_context_caps_independent_of_schedule_mode` | 4 |
| `context_mode_is_strict_and_precedes_array_preparation` | 7 |
| `context_v3_cannot_silently_disable_pair_lane` | 5 |
| `context_configuration_snapshot_rejects_tamper` | 9 |
| `requested_returned_publication_requires_exact_config_and_hash` | 4 |
| `tampered_actual_context_cannot_enter_finite_prefilter` | 14 |
| `actual_initializer_whole_refusal_keeps_no_usable_start_prefix` | 2 |
| `actual_workflow_binds_all_three_contexts_and_full_raw_transport` | 1 |
| `every_selected_context_is_bound_before_any_measurement` | 12 |
| `empty_pair_schedule_does_not_bypass_mandatory_v3_publication` | 10 |
| `invalid_complete_context_index_table_cannot_publish_prefix` | 4 |
| `late_workflow_refusal_never_exposes_measurement_prefix` | 3 |
| `workflow_top_and_policy_publications_are_independently_bound` | 4 |
| `workflow_invalid_context_mode_never_reaches_native_or_array_decoding` | 4 |
| `context_capacity_is_not_misidentified_as_v2_scheduler_marker` | 1 |
| `actual_workflow_v2_schedule_request_retains_mandatory_context_v3` | 1 |
| `context_budget_selector_rejects_untyped_or_legacy_profiles` | 4 |
| `direct_selected_binding_requires_explicit_expected_context_config` | 1 |

The additional function is
`test_single_dense_support_context_complete_cumulative_capacity`. No old
qualified test module is imported by either new file. The old 76-outcome gate
is retained evidence, not part of the 94 denominator.

## What the integration source actually connects

The real numerical fixture contains 50 original scan rows on two 5-by-5 sheets,
64 scan triangles, independently triangulated CAD sheets with 80 triangles and
two CAD face IDs. Its proper cyclic axis permutation and integral translation
are independently specified fixture premises. The bank cap is two. Both
scheduler modes run the actual initializer, proposal extraction, complete pair
schedule, context V3 and the reused finite-face prefilter. The intended schedule
is two ordered matches, eight signed families and 64 slot records, with each
assigned support retaining all 25 original rows. Registration refinement,
native parsing, local bend fitting and scoring are outside this fixture.

The byte-orchestration fixture instead contains 48 unique raw rows, 36 selected
registration rows, two scan triangles, three CAD triangles, 36 exact dyadic CAD
samples and three selected contexts. Native decode/extraction/sampling,
registration and measurement/validation boundaries are explicit mocks. The
real requested/returned configuration checks and every selected array, pose
and support binding remain live. The empty pair schedule and triplet origins
are deliberate adversarial cases, not evidence that the real pair initializer
returned those contexts. A separate V2-schedule fixture retains the actual
scheduler ledger checks. The mocked byte workflow is not native end-to-end
qualification and does not authenticate physical bytes.

The following production delta is supported by the read source and challenged
by the prospective tests:

- Context mode is an exact closed `v3` selector, independent of scheduler V1/V2.
  Only the exact V3 budget type or `None` is admitted; the pair lane cannot be
  silently disabled. A fresh owned validated budget snapshot is bound before
  array work. Lower budgets retain the dependent extrema-query ceiling.
- Both new comparator arms explicitly use V3. The old V12 registration and V13
  workflow wire schemas remain compatibility transports, with distinct V13
  initializer and V14 workflow implementation identities. They do not imply
  historical replay or legacy BLAS equivalence.
- The exact 12-key context configuration includes the complete 23-key budgets
  and eight arithmetic/scope declarations. Canonical sorted compact JSON is
  independently hashed; the expected configuration is not obtained by trusting
  an unvalidated returned dictionary or an alias of it.
- Workflow top, policies, registration evidence and nested initialization must
  all bind the request. Every selected array binding must bind the initializer
  identity, context mode, context implementation and configuration SHA. The
  checks occur outside the nonempty-pair branch; empty schedules and triplet
  origins cannot skip them.
- Nonempty helper output must bind full input digests, V3 declarations/caps,
  complete schedule, signs, slots and candidate provenance before the finite
  prefilter. It receives one complete schedule, not an accepted prefix.
- The V1 schedule binder does not mistake V3 context capacity metadata for a V2
  scheduler marker. Actual V2 schedule validation remains delegated to the
  frozen equation/configuration binder.
- Context-helper failure clears all usable starts; selected-context failure
  clears the complete binding/index list before measurement. First, middle and
  last selected contexts are separately corrupted. Late workflow failures
  clear usable measurement output rather than retaining a success prefix.
- The unchanged detailed initialization-report refusal envelope may omit new
  metadata because it discards the whole detailed report. It cannot qualify a
  usable selected context. This is not a reason to append out-of-budget fields
  or invent not-reached configuration evidence.

Frozen numerical helpers are reused rather than copied and adjusted by this
integration. Source aliases and the tiny real path are both checked. The tests
do not attempt to requalify every frozen numerical helper or native adapter.
Configuration hashes are consistency bindings, not cryptographic signatures or
proof of execution; a source freeze and exact attempt audit remain separate.

## One representative dense-support capacity case

The accepted context 76/76 narrative reports a 900000-row full buffer but only
four rows in each assigned support: 900008 cumulative indexed memberships.
That evidence is valuable but does not exercise dense assigned support unions
or a near-default triangle workload. The new single case retains the same raw
scale and fills this specific gap without introducing a geometry campaign.

It constructs two disjoint 500-by-900 planar integer grids in memory:

- Sheet 0, original row `900*i+j`: `(0, i+2, j)`.
- Sheet 1, original row `450000+900*i+j`: `(i+3, 0, j)`.
- `0 <= i < 500`, `0 <= j < 900`; all 900000 XYZ rows are distinct.
- Each sheet has 450000 assigned raw rows and 897202 connected grid triangles;
  the whole scan has 1794404 triangles, below the unchanged 2000000 cap.
- Four independently written CAD triangles describe the bounding rectangles.
  Plane declarations are explicit fixture premises; no plane extraction or
  physical fit is performed by this capacity case.

There is exactly one ordered match, four signed families and all 32 slots.
One whole-buffer index plus two disjoint assigned indexes means exactly
1800000 cumulative memberships. No prefix, row reduction, leaf change, raised
cap or retry is permitted. The default 256-row leaves produce 8192, 4096 and
4096 node slots. The fixed geometry yields an analytic staged scan program:
subtract zero, multiply by the positive Z axis, add the two zero products, then
add `j` and zero. All arithmetic is exact here. Thus each scan interval is
`[0,899]`; original minimum/maximum witnesses are `0/899` for whole/sheet 0 and
`450000/450899` for sheet 1. CAD signs produce `[0,899]` or `[-899,0]`, with
original triangle/vertex ordering breaking ties, not the deliberately
nonmonotonic source triangle IDs. No implementation query result is used to
derive these oracles.

The test independently asserts cumulative construction and later query work.
Writing `E` for all query rows actually evaluated, the source-derived totals
are helper row visits `3600000+E`, index visits `9000000+E`, coordinate visits
`34200000+9*E`, sort work 35100000 and hash bytes 14400000. Parent row visits
are `4500000+E`, index visits `23360860+E`, sort work 153530676, hash bytes
100621312 and CAD vertex visits 60. The latter includes subsequent CAD work,
not merely the extrema constructor. Helper-owned payload is 15579648 bytes;
parent-owned payload is 86221376 bytes. The dominant second support-union
payload estimate is about 219 MB, below the unchanged 268435456-byte parent
cap. This is an accounting derivation, not a measured RSS or runtime result.

The test allows the query's bounded full-leaf worst case (`0 < E <= 1800021`),
so it does not demand an unmeasured sublinear speedup. It separately checks
default caps, temporary-byte release, reported model peak and complete schedule.
One compact stdout record, at most 8192 encoded bytes including its fixed
prefix and newline, retains raw SHA, constructor/query/work/memory counters,
support/index descriptions and extrema/witnesses. The record explicitly requires process RSS
from the supervisor, not the model counters. It is evidence of invented dense
support capacity, not realistic full-workflow timing, physical registration
accuracy, all 129 indexes or the 4000000-membership boundary.

This is the smallest requested additional case in outcome count: one case at
the existing 900k scale, with two dense supports sufficient to force cumulative
ownership beyond the whole index. Adding many parts, arbitrary geometry or
mandatory positive physical results is not needed for this narrow decision.

## Concrete static import superset

The integration fixture explicitly maps 11 modules. Standard-library AST
traversal of their project imports, including function-local/lazy branches,
finds the following 45 backend source files. The dense fixture adds no new
backend dependency. This is a conservative static superset, not an observed
loaded-module list or a claim that branch-only native code executes.

```text
backend/cad_guided_cad_sampling_v1.py
backend/cad_guided_cad_surface_v2.py
backend/cad_guided_composite_surface_roles_v1.py
backend/cad_guided_context_measurement_v1.py
backend/cad_guided_context_surface_roles_v1.py
backend/cad_guided_exact_support_cover_v1.py
backend/cad_guided_face_box_screen_v1.py
backend/cad_guided_global_registration_v1.py
backend/cad_guided_global_registration_v2.py
backend/cad_guided_global_registration_v3.py
backend/cad_guided_instances.py
backend/cad_guided_local_measurement_v1.py
backend/cad_guided_measurement.py
backend/cad_guided_mesh_input_v1.py
backend/cad_guided_mesh_offline_v1.py
backend/cad_guided_mesh_offline_v12.py
backend/cad_guided_mesh_offline_v14.py
backend/cad_guided_mesh_offline_v2.py
backend/cad_guided_mesh_offline_v8.py
backend/cad_guided_mesh_plane_proposals_v4.py
backend/cad_guided_mesh_topology_v4.py
backend/cad_guided_native_features.py
backend/cad_guided_native_features_v1.py
backend/cad_guided_noisy_mesh_planes_v5.py
backend/cad_guided_noisy_mesh_planes_v7.py
backend/cad_guided_plane_hypotheses_v4.py
backend/cad_guided_plane_initialization_v12.py
backend/cad_guided_plane_initialization_v13.py
backend/cad_guided_plane_initialization_v4.py
backend/cad_guided_plane_initialization_v8.py
backend/cad_guided_plane_search_v6.py
backend/cad_guided_raw_extrema_index_v1.py
backend/cad_guided_report_envelope_v6.py
backend/cad_guided_scan_plane_preparation_v1.py
backend/cad_guided_seeded_plane_proposals_v1.py
backend/cad_guided_stage_policies_v1.py
backend/cad_guided_support_cover_measurement_v2.py
backend/cad_guided_triangle_distance_bounds_v3.py
backend/cad_guided_triangle_surface_v2.py
backend/cad_guided_triangle_surface_v3.py
backend/cad_guided_two_plane_context_v1.py
backend/cad_guided_two_plane_context_v2.py
backend/cad_guided_two_plane_context_v3.py
backend/cad_guided_two_plane_schedule_v1.py
backend/cad_guided_two_plane_schedule_v2.py
```

The 45 backend files total 1075804 bytes. The canonical map keyed by dotted
`backend.module` names, each value containing integer `bytes` and hexadecimal
`sha256`, encoded with sorted compact JSON and `allow_nan=False`, hashes to
`00f2ea238ee4e31e4f95f0f4b7169a49c03f7e73e4931a5e87fa18c74d640891`.
Changing keys to the exact slash-separated paths above and adding the two test
paths produces 47 project sources, 1129410 bytes and map SHA256
`49737747b23c6d338ba6bd145e071f88676ab13d6392f26b3d30428653668206`.
These counts do not include a future harness, supervisor, protocol, PRE,
runtime libraries, this narrative or the design document.

The fixtures create inert package namespaces and restore/remove them afterward;
they do not execute `backend/__init__.py`. The corrected integration fixture
uses one coherent inert backend namespace for both relative and absolute imports;
the dense helper has only relative imports and keeps its separate private graph. Eager numerical
dependencies include NumPy and SciPy; pytest supplies the prospective test
runner. Static closure also includes lazy OCP import sites in native adapters
whose calls are explicitly mocked. The future harness must freeze runtime
identity, disable uncontrolled pytest plugin autoload/cache side effects, define
the isolated bootstrap and enforce the native/data boundary. Source graph
membership alone is not permission to execute a lazy native branch. Root and
the PRE reviewer should independently reproduce the closure rather than trust
this map as a self-authorizing manifest.

## Exact next qualification and stop boundary

The proposed gate is one serial worker, exactly these 94 outcomes, unchanged
45 CPU seconds, 55 work seconds, 60 supervision seconds, 1 GiB sampled RSS and
1 MiB output. These are prospective maxima, not predictions or execution
authority. A new exact harness/protocol, complete source/runtime closure,
independent PRE, verified frozen archive and owned-process finalization are
root's remaining tasks before any target import. Every outcome must account
for setup/call/teardown and the complete result must be retained. Failure or
timeout stays a terminal qualification result; no same-attempt retry, cap raise,
smaller dense fixture or partial denominator is inferred from this document.

An accepted POST would support connected V3 configuration/binding and this
dense-support engineering capacity case. It would not resolve the known
exploratory registration/coverage negatives by assertion. The shortest further
evidence-producing step is the separately authorized small already-consumed
diagnostic: freeze its exact case/arm/input/reference scope, unchanged physical
criteria and resource caps, then retain capacity/refusal/CPU dispositions even
when inaccurate. Do not require all-positive geometry as an unrelated
prerequisite, do not reuse terminal V5 authority, and do not promote a consumed
diagnostic into unseen-family accuracy. A fresh full 11-case score gate follows
only the resulting decision and its own exact source/custody prerequisites.

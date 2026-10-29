# Context V3 schedule fixture V3 — focused repair design

Date: 2026-09-12. SOURCE ONLY, five prospective outcomes, zero executed.

Author: process_v2_qualification. The author previously reviewed the predecessor and authored its independent failure POST; because this is now authorship of new test code, a different reviewer must perform the independent source review/PRE. Root owns the new harness/protocol and sole eventual invocation. This note is not execution authority.

## Why a focused successor exists

The frozen integration V2 attempt remains terminal FAIL93/94. Its accepted failure POST is
`docs/research/2026-09-12_context_v3_engine_integration_v2_source_post_run_audit.md`,
12368 bytes, SHA256 `c40df2790d336fa8deb614788976e9745c1f04b26e2882c6adad4521cac4f5c0`.
The retained pytest output showed UNAVAILABLE at WHOLE_PART_REGISTRATION in the sole V2-schedule byte-workflow case; its full reason was elided. Source review found the new patch_evidence container supplied only cad, while the real legacy V7 scan binder indexes its scan member whenever the container exists.

This new test repairs only that invented boundary representation. Production sources, original test, V2 test, both terminal attempts, their PREs/archives and successful dense evidence remain unchanged. Do not rerun the 93 successful V2 outcomes or the dense case. This five-outcome gate is not a replacement or favorable recount of either prior denominator.

## Actual codec, explicitly invented metadata

The new selected path is
`tests/test_cad_guided_context_v3_schedule_fixture_v3.py`.
It explicitly imports only the two pytest fixtures `modules` and `plumbing`, and helpers `_canonical`, `_expected_configuration`, `_sha` from the exact frozen V2 test module. No wildcard or old test-function import occurs.

Importing that helper module defines its functions and decorators, but the eventual pytest selection must collect only the new file. The 94 old parameter outcomes are not selected or invoked. The shared module fixture installs one inert backend graph, preserving absolute/relative import identity, and removes/restores it in finally. The harness must separately keep the tests namespace inert and root/backend initializers forbidden.

The repaired fixture uses the exact V7 `encode_metadata_v7` and `decode_metadata_v7` codec through the already-loaded V12 -> V8 -> compact-V7 alias. The payload has only:

- schema_version = noisy_mesh_planes_v7;
- fixture_scope = INVENTED_V7_METADATA_CODEC_ONLY_NOT_EXTRACTION_OR_GEOMETRY;
- status = INVENTED_BOUNDARY_METADATA_NOT_EXTRACTOR_RESULT.

The codec round-trip must reproduce that payload exactly and the encoded object must have the real format/schemas/root envelope and <=4096 encoded bytes. This is an actual codec-generated V7-schema boundary envelope, not an actual V7 extraction report. No COMPLETE extraction, geometric support, count, role preparation, registration quality or native result is invented. The inherited V7 binder checks this root schema only; it does not fully decode/qualify extraction metadata. That limited binder contract is what this fixture is intended to exercise.

The existing byte fixture still supplies its explicitly mocked native read/extract/sample, registration, measurement and publication-validator boundaries: 48 declared raw rows, 36 selected rows, three declared context records. No new geometry/extraction/refinement/fitting calls are added. The real V2 empty scheduler creates its own equation-digest and joint-budget ledgers with retained triplets 3/3 and max joint 8192/1024. Both scan/CAD equation banks remain empty, so it is a request/publication binding case, not a physical plane-count or pose-recovery case.

## Live binders and exact callback evidence

No scan, schedule, context or selected-context binder is mocked to return success. Three delegating spies wrap the actual context publication binder, actual schedule binder and actual scan binder. They call the saved actual function with the unchanged arguments. On an exception they record only its actual class name and args, then use bare raise to rethrow unchanged. On success they record PASS and return the actual result. The frozen plumbing fixture's selected-context wrapper likewise delegates unchanged.

The existing workflow remains responsible for catching binding errors, clearing usable bindings/context indices, and returning UNAVAILABLE. A compact diagnostic includes phase, status, reasons, registration reasons, attempted indices and the small spy ledger, capped at4096 UTF-8 bytes. This avoids a future full-answer repr eliding the decisive reason as in the predecessor. No frame locals, external data or fixture arrays are logged.

Static call sequence for the positive case: four top/policy/evidence/initialization context publication checks, then two context checks plus one schedule and one scan check for each selected index0,1,2. Expected successful counts are context10/schedule3/scan3, selected bindings [0,1,2], then one mocked measurement and one mocked validation. All three mandatory binders remain real.

For every malformed scan envelope, the first four context checks plus the first selected context's two checks succeed, then the real schedule check succeeds, then the real scan binder raises. Required ledger: context6 PASS, schedule1 PASS, scan1 exact FAIL; selected-binding attempts [0]; no measure/validate callback. The outer workflow must expose WHOLE_PART_REGISTRATION and the exact WORKFLOW_EXCEPTION class, retain the exact registration-binding refusal reason, and clear all usable prefix state.

## Fixed five-outcome matrix

Two test functions, five prospective outcomes:

| Case | Required result |
| --- | --- |
| Repaired actual-codec V2-schedule workflow | PARTIAL / COMPLETE / empty reasons; unchanged V14 implementation with V13 wire schema; all three bindings; all four context publications and all binding hashes exact; owned V2 cap and V3 request forwarded; full raw transport; one mocked measurement/validator; top predictions only; production false. |
| missing_scan | Actual scan KeyError with args exactly (scan,); outer WORKFLOW_EXCEPTION:KeyError. Reproduces the malformed-container contract without rerunning the old test. |
| missing_root | Actual scan ValueError with args exactly (INVALID_LEGACY_SCAN_PLANE_ENVELOPE,); outer WORKFLOW_EXCEPTION:ValueError. |
| invalid_root | Actual scan ValueError with args exactly (INVALID_LEGACY_SCAN_PLANE_ROOT,); outer WORKFLOW_EXCEPTION:ValueError. |
| wrong_schema | A codec-valid round-tripped composite_scan_planes_v1 envelope rejected by actual V7 scan binding with exactly UNREQUESTED_OR_INVALID_SCAN_PLANE_SCHEMA; outer WORKFLOW_EXCEPTION:ValueError. |

Negative cases additionally require registration status UNAVAILABLE, transform None, exact reason INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT, empty registration_array_binding, empty usable measurement_context_indices, attempted indices [0,1,2], empty predictions/measurements, observed count None and production false. No generic ValueError assertion can satisfy this matrix at an earlier policy site.

The positive test retains the failed predecessor's intended V2 schedule and mandatory V3 ownership/publication assertions and strengthens reach/phase/measurement ordering. It does not rerun the real small 50-row numerical chain or any dense fixture.

## Premises checked in source

The frozen V2 helper explicitly uses the valid .05 <= .1 <= .5 policy tuple; V14/V2/V1 workflow and native/support constructors remain valid and unchanged. The actual V2 schedule binder reads the supplied empty CAD patch list and both empty equation tables. Its residual capacity is255 ordered/127 both-order bundles/one odd remainder; zero scheduled matches and three triplet reservations are consistent.

Default V7 scan binding admits a valid real-codec envelope whose root has schema_version noisy_mesh_planes_v7. The actual V8 producer's source writes both scan and CAD patch evidence; this repair conforms to that container structure without pretending to execute the producer. The malformed fixtures target, in order, the exact missing-key lookup, envelope shape predicate, root shape predicate and schema predicate in frozen V12 scan binding.

No later call is bypassed in the positive test. The original raw/selection/CAD array binder, selected-context origin/pose/support binding and requested/returned context configuration checks remain live. Actual future execution can still reveal an additional fixture or implementation issue; source review is not a pass claim. Preserve every outcome if it does.

## Source dependency handoff

New test:9545 bytes, SHA256
`06edc31c1953ce5220d292211364a3398452b4ca0d335c8e43981c619b95e7b4`.

Frozen imported V2 test:43998 bytes, SHA256
`67b705809d2b96a557981673d9c571a462c8b9a5ed196bf7593b53a1247e677b`.

All45 backend source files remain unchanged,1075804 bytes. Canonical compact sorted JSON repo-relative path-to-SHA256 map:
`2a86e3807e73f42e4dc11a9ee0a90037c9ef9af6111aba6c80d13ae80c61fb30`.
The two test source files plus those45 modules form47 Python source files; byte-bind the24-byte inert tests initializer as well, making48 source-custody files totaling1129371 bytes, map:
`2e43ead5227c266751826d1e3bd50cf9c8037231312dbe2c6daef8e957a1757f`.

The tests initializer SHA256 is
`5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b`.
It is evidence only when the future harness supplies an inert tests package. The eventual complete harness closure must additionally include its own harness, chosen supervisor, protocol, this design and required accepted historical/source-review evidence. This48 map is not an already frozen complete execution closure or PRE.

Exact unchanged45-backend import superset:

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

The new test adds no project module outside this list and the exact frozen V2 helper module. Ordinary external dependencies remain NumPy/SciPy/pytest plus stdlib via frozen helpers. Native OCP branches remain lazy and mocked/forbidden. Application-server initialization, raw/filesystem decode, physical inputs and unrelated artifacts remain outside scope.

Author checks were full new source reading, inspection of the frozen codec/scan/schedule/context/selected-binding paths, stdlib AST syntax/parameter counts and source hashing only. All48 listed Python sources parse. No source compile/exec, target import, collection, pytest, native call or process action occurred.

## Next gate and stop conditions

A different reviewer must independently verify this new code, exact fixture premises, five-case denominator, source/import closure and protocol. Root must freeze a dedicated fixed selection/harness, runtime and budgets, accept the independent exact PRE, archive the complete closure and ensure no overlapping heavy worker before a sole invocation. No execution is released by this author's note.

The existing short supervisor's unconditional hash read of its closed capture output on unconfirmed quiescence is a known limitation recorded in the V2 POST; a future protocol must not repeat a whole-supervisor no-read guarantee unless its new supervisor actually supplies it. No private cache may be traversed.

Any new failure stays terminal. Do not modify frozen predecessors, weaken binders, drop a case or label old successful outcomes rerun. Full physical diagnostic/scoring authority remains separate. This author must not issue the independent PRE for their own new test.

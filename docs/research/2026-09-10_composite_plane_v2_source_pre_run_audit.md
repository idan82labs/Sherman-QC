# Independent exact PRE: fixture-corrected composite adapter V2

Date: 2026-09-10. Reviewer: `/root/cad_blocker_closure_review`.

## Decision

**GREEN for one new frozen 11-outcome source-only attempt**, after root reads
this complete PRE and archives the exact 66 sources below plus this PRE as
67 approved regular payloads. No import, collection or test execution occurred
during this review. This is prospective approval, not a passing result.

Approved command from the repository root:

```
.venv/bin/python -I -S -B scripts/check_cad_guided_composite_plane_v2.py
```

Exactly 11 passed and zero failures, errors, skips, xfails, xpasses or deselections
are required, with all resource/custody conditions and independent POST.
Exit zero alone is insufficient. No suite, budget or per-test override.

The [V1 POST](2026-09-10_composite_plane_v1_source_post_run_audit.md), SHA256
`5a21c96cafd8c1931d74a9d64573fbf8d87061730e24c0824916fe1dd34c99ae`,
was fully read and is included in this source closure. V1 remains terminal RED:
167 passed, 11 failed before the intended public workflow call. Its intact
167 passes are separate evidence, not a relabeled successful 178-outcome run.

V2 selects only the 11 previously unreached checks under the corrected invocation
fixture. It does not rerun the successful 167 or the terminal V1 suite. A genuinely
GREEN V2 POST plus the explicitly retained 41 producer, 59 role, 24 integration
and 43 older-adapter passes can support the unchanged engine's source-contract
qualification for a separately frozen connected synthetic successor. Neither
attempt becomes “178 passed,” and no connected/native/physical gain is inferred.

## Independent exact-difference review

I read the complete 224-line new test module, entire new harness and protocol,
the inherited harness chain, and relevant constructor, schema and mock paths.
I did not author these new candidate files. The prior review, mathematical
boundaries and full terminal reconciliation are retained in the
[V1 PRE](2026-09-10_composite_plane_v1_source_pre_run_audit.md) and V1 POST;
this audit does not repeat or broaden their evidence.

Pure standard-library AST comparison, without importing project code, confirms:

- The original imports and `ROOT` expression are identical.
- Exactly five original top-level definitions remain: `modules`, `_policy`,
  `adapter_case` and the two final test functions.
- `modules`, `_policy`, both test functions and all their parameterizations,
  expected reasons and assertions are AST-identical to V1.
- Removing exactly the five newly supplied `NativeGuidedPolicy` keywords from
  `adapter_case` makes its entire AST identical to the old fixture.
- No other executable top-level statement is added. The new module docstring
  describes its narrower scope; omitted old helpers/tests are not imported
  through the V1 test module.

The one corrected constructor now supplies:

| Policy field | Frozen test value |
| --- | --- |
| `flange_window_mm` | (2.5, 12.5) mm |
| `search_margin_mm` | 3.0 mm |
| `max_fit_residual_mm` | 0.05 mm |
| `plane_exclusion_mm` | 0.075 mm |
| `max_tangency_error_mm` | 0.1 mm |
| `fit_model` | `free_circle`, unchanged |

The first five fields are exactly the dataclass's required fields. The complete
six-value declaration equals the existing frozen V13 fixture's declaration.
Static inspection of its reused `CADGuidedFeature` admission confirms the
ordered window and `0 < 0.05 < 0.075 < 3` conditions, positive tangency bound
and admitted fit model. The four `SupportPolicy(4., 5., 20., 3)` positional
arguments supply all required fields within their declared bounds. Workflow
V12/V13 inherit the V2 policy fields; the 36-row/36-sample settings and the
registration helper's 36-point/one-iteration settings satisfy their unchanged
constructor bounds. This is source inspection, not a claim those calls have
already executed in V2.

All 62 V1 candidate files remain byte-identical to the exact V1 PRE ledger.
There is no backend default, algorithm, error ceiling, fixture input, fault,
assertion, expected output or mock-scope change. Supplying an existing explicit
test policy repairs an invocation; it does not relax engine admission or infer
a policy from the observed failure.

## Eleven checks and actual execution boundary

The full new module has two test functions. Literal parameter products yield
five faults times two adapter versions, plus one reuse control:

| Prospective check | Outcomes |
| --- | ---: |
| Missing returned configuration, V12/V13 | 2 |
| Top-level budget mismatch, V12/V13 | 2 |
| Initialization budget mismatch, V12/V13 | 2 |
| Unrequested composite configuration, V12/V13 | 2 |
| Hidden composite schema without configuration, V12/V13 | 2 |
| V13 metadata prepared once, same token for both contexts, intentional measurement refusal | 1 |
| Total | 11 |

The ten negative checks must reach the public workflow, observe exactly the
intended binding reason, and publish no measurements/predictions or accepted
transform. They also check requested configuration transport, copied budgets
and omission of composite parameters on requested V7. Merely failing earlier
at construction/input admission does not pass these assertions.

The eleventh check must reach both context bindings, see one full composite
metadata preparation and one shared private token, then reach the intentionally
refusing measurement stub. It is not a successful measurement or native/
registration positive.

The shared fixture still performs actual tiny composite preparation, codec
encoding, role admission and two direct binding controls. Invented bytes/arrays
come from two ordinary constructors in the already frozen V13 test source.
That module's pytest fixtures and tests are not called. The literal backend
namespace is inert and consistent for actual dataclass identity; the server
package initializer is not imported.

The workflow, strict immutable-byte/array custody and configuration binding
are actual source. Scan parsing/preparation, native read/extraction/sampling,
registration returns and measurement refusal use the unchanged explicit
doubles. Function-local OCP imports are not reached. NumPy/SciPy/pytest are
prospective numerical/test dependencies, not native CAD execution. Mock
`allow_ocp=True` is not a native execution permission or native proof.

No original-ID lineage, shared all-lane role conflict rule, all-row witness,
uncertainty declaration, finite-search policy or publication veto changes.
The retained PRE's conditional-role and sampled-budget limitations continue
to apply; no manufacturer/root identity, physical conformance or calibration
is established by this adapter gate.

## Exact closure, custody and required POST

The literal source closure is the unchanged V12 supervisor's 49 names, the
unchanged V1 harness's 13 additions and V2's four additions: **66 unique files**,
60 Python and six Markdown, totaling **1,621,860 bytes**. All Python files parsed
as AST. Regular-file reads were capped at 2 MiB and checked no symlink plus
stable file-descriptor/path identity, size and timestamps around reading.
This is a bounded snapshot check, not arbitrary filesystem-interference security.

V2 loads V1 harness definitions, which load the original supervisor definitions.
Neither older main nor older test suite is called. Only the complete new
`tests/test_cad_guided_composite_plane_integration_v2.py` module is selected.
The other source/test/development runner names remain frozen dependencies or
provenance, not additional execution targets.

One new exclusive `composite_plane_v2_source-` attempt uses one worker,
`-I -S -B -u`, numeric threads one, a 45-second CPU limit, an inert backend,
no repository conftest/plugin autoload/pytest cache and a private retained
HOME/TMP/cache. Limits remain 60 seconds monitored parent wall, 55 seconds
work, 45 seconds CPU, 1,073,741,824 bytes sampled process-group RSS and
1,048,576 bytes captured output. Cleanup shares the absolute parent deadline.
Unexpected descendants, monitoring errors or unconfirmed empty group fail
custody. These are sampled engineering limits, not hard-real-time/allocation
guarantees; startup source hashing and terminal receipt I/O are outside the
enforced work clock.

Before execution, root must completely read this PRE and preserve the exact
66 ledger files plus this PRE as 67 unique regular source payloads. No existing
source archive or terminal attempt is altered. This reviewer created no archive.

Independent POST, after confirmed empty writer group, must verify the exact
11/zero-other denominator, exit zero, full output hash/size, identical frozen
limits, complete CPU/RSS/wall receipt, no budget/primary/cleanup/source/inventory
error, and confirmed empty group. Check exactly four administrative files
(`STARTED.json`, `OUTPUT.txt`, `ARTIFACT_INVENTORY.json`, `COMPLETED.json`)
and an empty scientific `ARTIFACTS` directory. Both before/after source maps
must equal this exact 66-file ledger, and all archived sources plus PRE must
equal their frozen bytes. The harness does not parse the 11 denominator itself.

Any failure, missing/extra evidence, source drift or resource/custody error makes
V2 RED and terminal; no favorable prefix, cap increase, omission, relabeling or
silent retry. This PRE grants no connected packet execution, native bytes,
physical/Gold11 access, score/count/F1 claim, default product routing or floor
promotion. Those remain separate frozen decisions.

No project imports, test collection/execution, native/application services,
physical inputs, pools/manifests, held-outs, email, controls, authorizations or
one-shot records were accessed in this review. Only this new PRE was written.

## Exact 66-file source ledger

Archive this PRE separately as the 67th approved regular payload.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `backend/cad_guided_cad_sampling_v1.py` | 26439 | `f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a` |
| `backend/cad_guided_cad_surface_v2.py` | 7759 | `b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1` |
| `backend/cad_guided_composite_surface_roles_v1.py` | 25114 | `f87f83f9e235059f6d33a92fba4bd329197a61de6a23039212acd8ef90b3b68f` |
| `backend/cad_guided_context_measurement_v1.py` | 39387 | `a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9` |
| `backend/cad_guided_context_surface_roles_v1.py` | 25369 | `c042408109d260888834ebacc9ad3b62256fda589f1ea8cbba31164b2bea9a62` |
| `backend/cad_guided_exact_support_cover_v1.py` | 10167 | `70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47` |
| `backend/cad_guided_face_box_screen_v1.py` | 6121 | `c8f66cc25f5c4cb6a6ba3c1e1beeb205540813796bb2ab30d72eb596cdf3abb7` |
| `backend/cad_guided_global_registration_v1.py` | 21030 | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_global_registration_v2.py` | 20608 | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_global_registration_v3.py` | 35987 | `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0` |
| `backend/cad_guided_instances.py` | 10296 | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
| `backend/cad_guided_local_measurement_v1.py` | 12215 | `5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b` |
| `backend/cad_guided_measurement.py` | 25964 | `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb` |
| `backend/cad_guided_mesh_input_v1.py` | 29182 | `c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917` |
| `backend/cad_guided_mesh_offline_v1.py` | 12249 | `55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba` |
| `backend/cad_guided_mesh_offline_v11.py` | 22290 | `dbd60ebc8c2c159866c212506e46e35d56af17b196827ba5243c637c6dfe6dfa` |
| `backend/cad_guided_mesh_offline_v12.py` | 37319 | `70c94280716cc6b6855c79fd076e5e4100710373b3241773c5fb90bbc119f8f8` |
| `backend/cad_guided_mesh_offline_v13.py` | 16134 | `5cf3589acd22bf428cffaba6be79bd1153dea75c464d366919ca4ce6ed644657` |
| `backend/cad_guided_mesh_offline_v2.py` | 18036 | `fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf` |
| `backend/cad_guided_mesh_offline_v8.py` | 18745 | `afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec` |
| `backend/cad_guided_mesh_plane_proposals_v4.py` | 22680 | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `backend/cad_guided_mesh_topology_v4.py` | 5153 | `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734` |
| `backend/cad_guided_native_features.py` | 17966 | `f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f` |
| `backend/cad_guided_native_features_v1.py` | 41518 | `3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b` |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | 31259 | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | 17767 | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `backend/cad_guided_plane_hypotheses_v4.py` | 15192 | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `backend/cad_guided_plane_initialization_v10.py` | 38737 | `398c40055c00fd3432410242a7a2ee399d9d716ea8cb3c1755916904b6074331` |
| `backend/cad_guided_plane_initialization_v11.py` | 50799 | `31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e` |
| `backend/cad_guided_plane_initialization_v12.py` | 58791 | `164008c6e64e80a718573758b466cca4ba76499c2e84c7acc6975d226b50af51` |
| `backend/cad_guided_plane_initialization_v4.py` | 38276 | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| `backend/cad_guided_plane_initialization_v8.py` | 32544 | `3b4220637fa19aba1e0384071f55e091705476046468327a306486078d7f7438` |
| `backend/cad_guided_plane_initialization_v9.py` | 29482 | `c64ab591ee07b91d7f20cb0d663cc2bb2a98e5baa681e072f97353b01e82401d` |
| `backend/cad_guided_plane_search_v5.py` | 34165 | `13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98` |
| `backend/cad_guided_plane_search_v6.py` | 38804 | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `backend/cad_guided_report_envelope_v6.py` | 12480 | `c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e` |
| `backend/cad_guided_scan_plane_preparation_v1.py` | 11319 | `efea482eb3066a43028f4f3a3b5bfa7ae994fc6f1efc165aace109c05e30e5b5` |
| `backend/cad_guided_seeded_plane_proposals_v1.py` | 13087 | `14754498d5465a97ff0d8ab92fa587a5383df3cb45f9828b62532eee94e74564` |
| `backend/cad_guided_stage_policies_v1.py` | 3186 | `5c3d1252ac5f38f422f8a9a22c98a3d4db1094cd0956c928e9800f75e3ee826c` |
| `backend/cad_guided_support_cover_measurement_v2.py` | 27567 | `8783f95bfce161ae4d9cd0c5a2532114efa510907b50479a42896747e20d56aa` |
| `backend/cad_guided_triangle_distance_bounds_v3.py` | 10776 | `2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7` |
| `backend/cad_guided_triangle_surface_v2.py` | 23511 | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `backend/cad_guided_triangle_surface_v3.py` | 32098 | `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10` |
| `backend/cad_guided_two_plane_context_v1.py` | 36075 | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `backend/cad_guided_two_plane_schedule_v1.py` | 26562 | `1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f` |
| `docs/research/2026-09-10_composite_plane_v1_source_post_run_audit.md` | 9004 | `5a21c96cafd8c1931d74a9d64573fbf8d87061730e24c0824916fe1dd34c99ae` |
| `docs/research/2026-09-10_composite_plane_v1_source_protocol.md` | 7046 | `d1950d07fd415d3f6f1b2674935e97c625ac674902955e1a73fbde095439c559` |
| `docs/research/2026-09-10_composite_plane_v2_source_protocol.md` | 4623 | `60783a9a0aaa0fd9efe156709a8cbd8811da08b79b15a89edf8e56ba5b95c1ce` |
| `docs/research/2026-09-10_connected_v12_fixture_design.md` | 11016 | `5818f76ba084ccf726c9a2fffce15991ee94c8b8a068da95573318f4ae2c048f` |
| `docs/research/2026-09-10_connected_v12_gate_protocol.md` | 7202 | `671aa78a6d68fc2d42f34ed464337fadaa184d68d4ac61c3bfe7e52ac3fa85e4` |
| `docs/research/2026-09-10_seeded_plane_proposal_successor.md` | 7711 | `dcb4a346eb50e80d35f3b0bed35389e5890bce00618c7fd3268e703d5cb64952` |
| `scripts/check_cad_guided_composite_plane_v1.py` | 6145 | `5a28f8ed1c9306ab141bf55dd434cff43fac2f48ac9b63e954415d50ea62b309` |
| `scripts/check_cad_guided_composite_plane_v2.py` | 5767 | `b549dbc2367e4d6d6319a35ea8e122594c1ebae24e159e3fa08579583d83dffc` |
| `scripts/check_cad_guided_connected_v12.py` | 19437 | `0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694` |
| `scripts/run_gold11_mesh_development_v3.py` | 89034 | `77bbcffbf10f05a22d0c72c8a645112e489469b4af92b4dbba3ca6243fcecea7` |
| `tests/test_cad_guided_composite_plane_integration_v1.py` | 31147 | `4fa8ce9735e82878a6eac24d78a9b3476fd986f1f93504ae2509331fbecd6f55` |
| `tests/test_cad_guided_composite_plane_integration_v2.py` | 13116 | `a357a8d1234609ffec0ed1cc1e6baf0abc4bc54a14cf60518f710b9397ebb2af` |
| `tests/test_cad_guided_composite_surface_roles_v1.py` | 20000 | `2bb96edaf8a21508287db54c6ae895206e5ee84fd623566ceb2604fc1849d1a7` |
| `tests/test_cad_guided_connected_v12.py` | 49931 | `fbbbbc148c564c222013c68f81922e2aed19bdfc1033c7677117a1186582bdfd` |
| `tests/test_cad_guided_context_measurement_v1.py` | 46703 | `1567780921771510ac761ed8d41c3b55d82de731b5b1cf409e8c53466a40589b` |
| `tests/test_cad_guided_context_surface_roles_v1.py` | 17645 | `930baccbfe1a878cdce4c1543e79b1a6111f739c37c6bfa47bc7499945d87e4d` |
| `tests/test_cad_guided_mesh_offline_v13.py` | 26867 | `0149d5a9473af1d8e847c94f6c2ea8284e798cad486c425e9b0ab9498706f93e` |
| `tests/test_cad_guided_plane_initialization_v10.py` | 69176 | `4882b701af3806fd03ddde9bb0b1af823bd0124e859e6655207f683db1563eb0` |
| `tests/test_cad_guided_plane_initialization_v11.py` | 25591 | `16a646404b4cf35975956b717a9143b52d8bca1e7ec5467f40096aefbc0edf1e` |
| `tests/test_cad_guided_plane_initialization_v8.py` | 39940 | `e78e5bc17705334b9e7dd1ebe1f70271ae1cdca0a118bbfbea24d95f633a684a` |
| `tests/test_cad_guided_scan_plane_preparation_v1.py` | 24554 | `e9c9e07d5ca5ca08c1c034c555ff73e51d012576cf644798b8aa1b9fa42e25ab` |


# V5 runner source-gate PRE — BLOCKED draft, 2026-09-11

Decision: **BLOCKED_DRAFT_NOT_FROZEN_NOT_EXECUTION_AUTHORITY**.

This document prepares the independent review structure only. It does not
authorize a launch, source archive as a qualified bundle, native work, physical
access or a claim of 85 passing outcomes. All hashes below describe the current
blocked draft, not future final source. No missing evidence, future hash, timing,
attempt identifier or result is guessed.

Owned change: this document only. Source-only reads, hashes and AST/symbol-table
analysis were performed; no target import, compile-exec, pytest/collection,
subprocess experiment, geometry/native call, raw data, manifest, control,
authorization, attempt, archive or scientific artifact was accessed. The exact
repository initializer path was checked only as an import-closure boundary.
No candidate source, test, protocol or active frozen source was changed.

## Explicit blockers

1. Runner RELEASE_AUTHORIZED remains False. All seven physical timing values
   remain None: case wall/CPU, total wall, scorer wall/CPU, parent-read and
   terminalization reserve.
2. Harness FROZEN_FOR_SOURCE_GATE remains False; RUNNER_FINAL_SHA256 and
   TEST_FINAL_SHA256 remain None. The test pins only the current blocked runner
   (7c8a2ca6...), not a finalized timing/release candidate.
3. Completed, independently accepted native timing evidence and the resulting
   uniform all-22 resource allocation have not been supplied for this PRE.
   Active progress is not substituted for terminal/native qualification.
4. There is no final 66-source byte freeze, independent GREEN PRE, exact
   source-plus-PRE archive verification or authorized first source invocation.

These are intentional release barriers, not evidence that the new 85-outcome
suite failed. Its execution and actual outcome denominator are unknown.

## Reused review and current static checks

This draft inherits the complete source/mock assessment in
[runner source review](2026-09-11_gold11_mesh_runner_v5_source_review.md)
(12176 bytes, SHA256
98849b18d6d4cee254e43960f0d0480d6d1611c316f4c771773046cfdb0e7e39)
and [harness review](2026-09-11_gold11_mesh_runner_v5_harness_review.md)
(9975 bytes, SHA256
d37448fcf41e3c476928b81ab56c8fab0f7b91743ef25904d260c43163836f97).
The harness/protocol were reread completely for this draft. Exact unchanged
hashes permit reuse of the prior full runner/test/source review, rather than
claiming an old result qualifies new bytes.

Current static evidence:

- 66 distinct declared source paths, 1,890,563 aggregate bytes. All were read and
  hashed narrowly; each is a nonempty regular file within the 2 MiB source cap.
- The runner's 61-path declaration has 61 distinct entries. Every one matches
  the prior complete review's size/hash. The remaining five paths are exactly
  the two ordinary invented constructor sources, new harness, new source
  protocol and old supervisor preservation oracle listed below.
- 25 top-level test functions expand statically to 85 prospective outcomes.
  In source order, their parametrized counts are
  5,8,1,7,4,2,2,4,1,4,1,2,2,8,5,2,4,1,8,4,2,2,1,4,1.
  This is AST counting, not pytest collection or execution.
- 43 selected runner functions and two named exception classes are admitted by
  the test's AST selector; unsafe entry functions are excluded from its normal
  namespace. Static global-reference review matches the explicitly supplied
  mocks and the pre-reference stop boundary described below.
- CONFIG, CERTIFICATE_CAPS, REVIEW_SHA, CATALOG_SHA and SELECTED_SHA are AST-equal
  to the exact pinned V4 source. This reads source literals only, not the files
  those literals name.
- Static internal backend import dependencies have no missing path from the
  declared runner closure. The complete closure is conserved, including files
  used only as source-preservation oracles rather than imported dependencies.
- tests/__init__.py is 24 bytes, hash-bound and AST-inert.
  FullProject/__init__.py is presently absent, including the dangling-link
  check. Both conditions must be repeated at final PRE and execution.

Current closure equality:

    61 runner-declared paths
    + tests/test_gold11_contact_scoring_v2.py
    + tests/test_gold11_schedule_projection_v1.py
    + scripts/check_gold11_mesh_runner_v5.py
    + docs/research/2026-09-11_gold11_mesh_runner_v5_source_protocol.md
    + scripts/check_gold11_mesh_v4_custody.py
    = 66 distinct paths

The external PRE is additional, not silently one of the 66. A future exact
archive would therefore need 67 regular payloads after the final PRE is accepted;
no archive identity, bytes or existence is asserted here.

## Import, mock and scientific boundary

The harness imports only standard-library modules before its release guard.
Both parent and worker refuse their false/unpinned gate before source preflight,
pytest/project imports, argument dispatch or attempt creation. ROOT path
resolution is not a runtime source/physical-campaign execution. The old V4
supervisor is a hash-bound source-review oracle, never imported or run.

The future worker would use isolated Python -I -S -B, the reviewed local
site-packages path, /dev/null pytest configuration, --noconftest, no cache
provider and disabled third-party plugin auto-load. The one test module is
fixed; no subset, policy, timeout or test override exists. Its test package
initializer is explicitly included. Backend and pure scripts package objects
are inert namespaces; the application package initializer is not imported.
OCP import presence vetoes the worker outcome. The external installed Python,
pytest and numerical dependencies are environmental premises, not part of the
66-project-source hash ledger or an attested immutable OS.

The test AST-loads selected definitions, constants and standard-library imports,
not the physical runner main. Unsafe entry definitions are invoked only in
detached namespaces with an explicitly false first-statement guard; the positive
timing test calls the guard alone, never an enabled entry. Setting the final
runner release literal true therefore does not make these tests run a campaign.

The constructor test loads real policy types under the inert backend namespace,
checks V13/nativeV1/free-circle declarations and distinct V1/V2 arm maps, and
never calls the returned workflow or native adapter. It is declaration
qualification, not OCP/native-byte or measurement geometry qualification.

The two old test sources contribute only named ordinary invented constructors
(and the named schedule literals). Their fixtures/tests are not imported or
invoked. The pure schedule projection and its qualified contact/matcher graph
run on invented original JSON bytes. Real exclusive publication, source/result
binding, tree schema, whole-case reconstruction and inventory functions remain
connected; they are not replaced by tiny-schema success mocks.

Unselected globals have explicit boundaries:

- Timers, monotonic clock, runtime metadata and artifact-module provider are
  replaced for invented fixture work; no real timer/OS resource qualification
  follows from those tests.
- Actual per-case orchestration uses synthetic input tokens, a checked
  invented-directory reader, fake dispatch and fake supervisor. No real process
  or scan/CAD parser is invoked. Result retention and projection are real.
- The all-22-before-reference test supplies a synthetic verify_bundle and a
  reference seam that raises immediately when reached. With 21 terminals the
  reference seam is unreachable; with 22 it stops before reference/scorer access.
  Later score_after_terminal globals are deliberately not exercised by that
  test, not silently claimed as tested end-to-end scorer supervision.

The full invented fixture has 11 cases, 71 references, 22 results, 119 scientific
files, all 61 runner source paths and 46 issued/claimed dispatch records.
Three real inventory phases bracket exclusive COMPLETED and FINAL_GATE
publications. Files, IDs, source contents and runtime metadata in that fixture
are invented: this is structural custody testing, not an actual source-freeze,
native authority, physical-input or process-identity attestation. Ordinary
fixture payload is below 16 MiB; the separate 8 MiB wrapper boundary and its
one-byte-over negative do not establish maximum 128 MiB-result throughput.

Malformed, changed, partial or late-bound results cannot contribute a favorable
prefix. Whole supervisor failures and no-carrier outputs retain the full
11-case/71-reference denominator. No manufacturing count, accepted pose, native
coverage, calibrated error, conformance verdict or new physical F1 is implied.

## Fixed prospective source-gate resources and outcome

These are the reviewed source/mock gate limits, not the missing physical
campaign timings:

| Resource or disposition | Required value |
| --- | ---: |
| Collected/passed outcomes | 85 / 85 |
| Failed, skipped, errors, deselected, xfailed, xpassed | zero each |
| Worker CPU, soft and hard | 120 seconds |
| Parent-monitored work / total wall | 175 / 180 seconds |
| Cleanup and postflight reserve inside total | 5 seconds |
| RSS admission | 2 GiB |
| Complete combined output | 1 MiB |
| Each administrative JSON | 256 KiB |
| Each declared source / source count | 2 MiB / 66 |

Source preflight, attempt creation, work, cleanup, source postflight, publication
and final checks share the same parent clock. Worker CPU includes preflight,
imports and tests. Thread settings remain one. These are monitored deadlines,
bounded waits and sampled/high-water RSS gates, not hard real-time I/O or
hostile-process containment. Parent CPU is not independently capped.

The helper's final file open uses O_NONBLOCK and O_NOFOLLOW before regular-file
admission. Full parent pre/post source maps and worker preflight are mandatory;
errors or unattempted tails veto, never qualify a matching prefix. Root
initializer absence is checked before and after the worker.

The worker's outcome observer distinguishes xfail/XPASS using wasxfail. Parent
admission requires exactly one complete worker receipt, exact 85/85/zero-other
counts, exit0, worker CPU/HWM within limits, no OCP import and initializer
absence. Complete output is retained and inspected only after owned-group
emptiness, not replaced by a parsed pytest success line.

Supervision charges observed log bytes before writes, preserves uncertain
write accounting, attempts owned-child cleanup before closing streams, and
requires confirmed empty process group. Unknown RSS is not zero. Ordinary
cleanup/close errors veto. Combined group sample plus parent high-water and
final reaped-child high-water plus parent high-water must meet 2 GiB; the latter
is conservative, not a measured simultaneous tree peak.

Only STARTED.json, OUTPUT.txt and COMPLETED.json belong to the future source
attempt inventory. Invented pytest files live in a separately retained private
cache, not in the three-file evidence directory and not as retained physical
science. No blanket cleanup is authorized. Final inventory, post-publication
clock, observed parent exit0 and independent exact POST are necessary; an
earlier pending status alone cannot qualify.

## Finalization checklist — still required, no command released

1. Root supplies independently accepted completed connected/native evidence and
   the prospective resource decision. Apply the accepted
   [resource-planning review](2026-09-11_gold11_v5_resource_planning_review.md):
   uniformly fund all 22 opportunities, scorer/coordinator overhead and final
   reconstruction/three-inventory reserve. Missing or killed stages stay unknown.
2. Root finalizes seven runner time literals, release declaration and physical
   protocol; mechanically updates the test RUNNER_SHA; finalizes this harness's
   pins/flag and source protocol. This draft does not authorize those values.
3. Independently read every changed source in full or its complete exact diff
   against this reviewed snapshot. Recheck scientific AST equality, guard-only
   safety, all mocks/free globals and actual 85-outcome AST expansion. A changed
   mechanism or test denominator requires a new review, not a clerical pin edit.
4. Mechanically reread/hash the complete final 66-file declaration, confirm
   exact closure/import boundaries and initializer conditions, and replace this
   blocked draft with a final independently decided PRE. Do not merely append
   future hashes to the current ledger or assert unchanged bytes without reading.
5. Only if GREEN: root reads the complete final PRE, archives exactly the
   66 sources plus that PRE, independently verifies regular members/bytes/hashes,
   then uses the one fixed source-gate command recorded in that final PRE.
   No command is released here.
6. A sole terminal result requires exact independent POST over the authorized
   three artifacts, final 66 sources, archive and observed process/resource
   evidence. Any failure stays terminal. No test rerun, started-cap change,
   post-qualified physical timing mutation or physical execution follows
   automatically from this source gate.

## Current blocked-draft byte ledger

This entire ledger must be revalidated after final edits. Aggregate current
payload: 1,890,563 bytes across 66 regular files. It is not an archive receipt,
qualification or attestation of future final source.

| Current source path | Bytes | SHA256 |
| --- | ---: | --- |
| backend/cad_guided_cad_sampling_v1.py | 26439 | f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a |
| backend/cad_guided_cad_surface_v2.py | 7759 | b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1 |
| backend/cad_guided_composite_surface_roles_v1.py | 25114 | f87f83f9e235059f6d33a92fba4bd329197a61de6a23039212acd8ef90b3b68f |
| backend/cad_guided_context_measurement_v1.py | 39387 | a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9 |
| backend/cad_guided_context_surface_roles_v1.py | 25369 | c042408109d260888834ebacc9ad3b62256fda589f1ea8cbba31164b2bea9a62 |
| backend/cad_guided_exact_support_cover_v1.py | 10167 | 70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47 |
| backend/cad_guided_face_box_screen_v1.py | 6121 | c8f66cc25f5c4cb6a6ba3c1e1beeb205540813796bb2ab30d72eb596cdf3abb7 |
| backend/cad_guided_global_registration_v1.py | 21030 | 7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7 |
| backend/cad_guided_global_registration_v2.py | 20608 | 5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3 |
| backend/cad_guided_global_registration_v3.py | 35987 | 088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0 |
| backend/cad_guided_instances.py | 10296 | d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda |
| backend/cad_guided_local_measurement_v1.py | 12215 | 5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b |
| backend/cad_guided_measurement.py | 25964 | d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb |
| backend/cad_guided_mesh_input_v1.py | 29182 | c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917 |
| backend/cad_guided_mesh_offline_v1.py | 12249 | 55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba |
| backend/cad_guided_mesh_offline_v10.py | 17079 | 81d9411ccfff00c9f9fe5aa9b6fc25d7384acf993d34cfdbb0349dfe109c7fa4 |
| backend/cad_guided_mesh_offline_v11.py | 22290 | dbd60ebc8c2c159866c212506e46e35d56af17b196827ba5243c637c6dfe6dfa |
| backend/cad_guided_mesh_offline_v12.py | 46206 | 7c444bab2c197b0a6d9949ec6a15162286edb78bdfd081137e2907f16152c33e |
| backend/cad_guided_mesh_offline_v13.py | 18129 | 22c5bc280bc1d350f90bf6d93bf441d233fc330c69ca7041d29dd6fd866173bb |
| backend/cad_guided_mesh_offline_v2.py | 18036 | fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf |
| backend/cad_guided_mesh_offline_v8.py | 18745 | afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec |
| backend/cad_guided_mesh_plane_proposals_v4.py | 22680 | 55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48 |
| backend/cad_guided_mesh_topology_v4.py | 5153 | 4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734 |
| backend/cad_guided_native_features.py | 17966 | f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f |
| backend/cad_guided_native_features_v1.py | 41518 | 3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b |
| backend/cad_guided_noisy_mesh_planes_v5.py | 31259 | 9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75 |
| backend/cad_guided_noisy_mesh_planes_v7.py | 17767 | b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f |
| backend/cad_guided_plane_hypotheses_v4.py | 15192 | 7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370 |
| backend/cad_guided_plane_initialization_v10.py | 38737 | 398c40055c00fd3432410242a7a2ee399d9d716ea8cb3c1755916904b6074331 |
| backend/cad_guided_plane_initialization_v11.py | 50799 | 31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e |
| backend/cad_guided_plane_initialization_v12.py | 73393 | 57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87 |
| backend/cad_guided_plane_initialization_v4.py | 38276 | a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724 |
| backend/cad_guided_plane_initialization_v8.py | 32544 | 3b4220637fa19aba1e0384071f55e091705476046468327a306486078d7f7438 |
| backend/cad_guided_plane_search_v6.py | 38804 | aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676 |
| backend/cad_guided_report_envelope_v6.py | 12480 | c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e |
| backend/cad_guided_scan_plane_preparation_v1.py | 11319 | efea482eb3066a43028f4f3a3b5bfa7ae994fc6f1efc165aace109c05e30e5b5 |
| backend/cad_guided_seeded_plane_proposals_v1.py | 13087 | 14754498d5465a97ff0d8ab92fa587a5383df3cb45f9828b62532eee94e74564 |
| backend/cad_guided_stage_policies_v1.py | 3186 | 5c3d1252ac5f38f422f8a9a22c98a3d4db1094cd0956c928e9800f75e3ee826c |
| backend/cad_guided_support_cover_measurement_v2.py | 27567 | 8783f95bfce161ae4d9cd0c5a2532114efa510907b50479a42896747e20d56aa |
| backend/cad_guided_triangle_distance_bounds_v3.py | 10776 | 2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7 |
| backend/cad_guided_triangle_surface_v2.py | 23511 | 7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1 |
| backend/cad_guided_triangle_surface_v3.py | 32098 | 8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10 |
| backend/cad_guided_two_plane_context_v1.py | 36075 | 9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a |
| backend/cad_guided_two_plane_context_v2.py | 11130 | d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663 |
| backend/cad_guided_two_plane_schedule_v1.py | 26562 | 1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f |
| backend/cad_guided_two_plane_schedule_v2.py | 25030 | 07b24b8e53b08a408f56ab94dc442e7db7f4e3ef6a5cd4729629860cace99977 |
| docs/research/2026-09-11_gold11_mesh_development_v5_protocol.md | 11780 | 5aa1b7c0bcc54f9b6eb2a41d000ed9a8c7e7460ea95b74b25b47d37a8027d874 |
| docs/research/2026-09-11_gold11_mesh_runner_v5_source_protocol.md | 12267 | d0c8041f7a2dd51c0865b780fe619d7086610b5e4cdb928eddbb4dc1b49066a6 |
| scripts/check_gold11_mesh_runner_v5.py | 30205 | 36714fda5dab2a716b0bc5f18249566745ea57dfcd73c96cc23147c01c2c1bf7 |
| scripts/check_gold11_mesh_v4_custody.py | 10310 | 579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b |
| scripts/evaluate_f1_against_manual_gold.py | 13134 | 1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62 |
| scripts/f1_physical_evaluation_contract_v2.py | 8329 | 28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7 |
| scripts/gold11_contact_scoring_v2.py | 50413 | 9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff |
| scripts/gold11_mesh_scoring_v1.py | 13492 | aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234 |
| scripts/gold11_owned_process_supervision_v1.py | 17073 | 1b90b86ef77510e32d72d3e08e85417860a8fb06ba0ad3294274a83bfb65ba74 |
| scripts/gold11_schedule_projection_v1.py | 17231 | a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245 |
| scripts/gold11_terminal_artifacts_v4.py | 24341 | 3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838 |
| scripts/review_gold11_inputs_v1.py | 25616 | 41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955 |
| scripts/run_gold11_mesh_development_v2.py | 59477 | 94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422 |
| scripts/run_gold11_mesh_development_v3.py | 89034 | 77bbcffbf10f05a22d0c72c8a645112e489469b4af92b4dbba3ca6243fcecea7 |
| scripts/run_gold11_mesh_development_v4.py | 130538 | 559095a685d6ae5badecdb9704952d5e80304dcd728f8afa0f131a25707ebd85 |
| scripts/run_gold11_mesh_development_v5.py | 153950 | 7c8a2ca60ea7671dd1c2b56339d6c632e70e00df7e6081fd18cb421a6678121f |
| tests/__init__.py | 24 | 5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b |
| tests/test_gold11_contact_scoring_v2.py | 38711 | f9576b9d4b7267565657ee18a4b344e150a48da9c86f9bf5da2259db3d9fe2cb |
| tests/test_gold11_mesh_development_runner_v5.py | 52489 | ee7861235ad345d7d807112cf3471536252e9b55f9ee437d0de0098b91ae2f76 |
| tests/test_gold11_schedule_projection_v1.py | 26868 | 8d3113d8017d3e50e096696563daf38c4f62deacb87ca552263a6adcac8faf72 |


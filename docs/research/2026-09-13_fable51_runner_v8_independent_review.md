# Independent source review: Gold11 V8 runner (2026-09-13)

Reviewer: independent (Claude Fable 5.1, reviewer role; not the author of the
reviewed files). Read-only review of the working tree under
`/Users/idantal/ShermanQC/FullProject`. Nothing was executed except `diff`,
`grep`, `sed`, `shasum`, `wc`, `awk` and `.venv/bin/python -c` snippets that
parse/introspect source with `ast` (no runner import, no freeze/execute, no
geometry, no network). No attempt directory, `data/`, `output*/`, frozen
snapshot, `.env`, credentials, Gold/reference file or the review script's case
list was opened. Scope extension from the coordinator (pose-admission gate) is
addressed in its own section.

## Decision and limits

- **No P1 found.** The V8 runner is the V7 text plus the claimed identity,
  engine-wiring, scoring-transport and three declared CONFIG hunks, with two
  small undeclared deviations (one stale identity, two dead-declaration
  edits) that do not change a scientific outcome, custody or the run.
- Engine binding is statically consistent: every constructor keyword set, every
  identity string, the nine context-V4 declarations, the 23 context budget keys
  and the `REGISTRATION_IMPLEMENTATION` label agree across runner, V15, V14,
  context V4, extrema V3, surface V4, registration V4, stage policies V2 and
  projection V3. The two at-cap CONFIG values (256 ties, 512M coordinate
  visits) are accepted inclusively by the engine validators.
- Import closure of `backend/cad_guided_mesh_offline_v15.py` (48 modules,
  module-level and function-local imports) is fully covered by `SOURCES`.
- Limits: I did not run any test or the runner; correctness of the V15/V14/
  V4/V3 engine mathematics was not re-reviewed (identities, constants and
  validators only); documentation numbers that come from V7 attempt data or
  invented development runs are recorded as unverifiable here (see "What was
  not checked"). The two protocol/design documents are recorded at the hashes
  below; the coordinator has announced a post-review documentation-only wording
  change to both, so their post-review hashes will differ.

## Examined bindings (table)

Hashes/sizes were taken at start and again at the end of the review; all were
identical (no file changed during the review). `V7` hash matches the required
`e910d562...bfc7a52`.

| File | SHA-256 | Bytes | How examined |
| --- | --- | ---: | --- |
| scripts/run_gold11_mesh_development_v7.py | e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52 | 169180 | full diff vs V8; targeted reads/greps; ast |
| scripts/run_gold11_mesh_development_v8.py | 98ae00a0ed63ae1efd15e032a5468c6b9e5fab8d7be4e865df0519c4333cd09d | 171255 | full diff; lines 30-345, 385-430, 455-545, 1015-1110, 1140-1185, 1360-1466, 2470-2510, 2850-2870; greps; ast |
| scripts/run_gold11_mesh_development_v6.py | 7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80 | 166049 | grep lines 93-94 only (sampler path precedent) |
| scripts/gold11_schedule_projection_v2.py | 728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db | 14677 | full diff vs V3 |
| scripts/gold11_schedule_projection_v3.py | 763e0cd032521f237aaa999601f67e75644fe430b9837bedf1dc2c4d9be82d30 | 16058 | read in full |
| scripts/gold11_schedule_projection_v1.py | a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245 | 17231 | hash only |
| scripts/gold11_contact_scoring_v2.py | 9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff | 50413 | grep (schema strings, thresholds) |
| scripts/gold11_mesh_scoring_v1.py | aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234 | 13492 | grep (thresholds) |
| scripts/gold11_engine_stage_sampler_v1.py | 80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0 | 8142 | lines 1-140 (pin verified = STAGE_SAMPLER_SHA) |
| scripts/gold11_owned_process_supervision_v3.py | 9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea | 42751 | hash only (pin verified = SUPERVISION_SOURCE_SHA) |
| scripts/gold11_terminal_artifacts_v4.py | 3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838 | 24341 | grep (entry caps); pin verified = ARTIFACT_SOURCE_SHA |
| scripts/review_gold11_inputs_v1.py | 41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955 | 25616 | hash only (= REVIEW_SHA); contents not read |
| scripts/evaluate_f1_against_manual_gold.py | 1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62 | 13134 | hash only |
| scripts/f1_physical_evaluation_contract_v2.py | 28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7 | 8329 | hash only |
| docs/research/2026-09-13_gold11_mesh_development_v8_delta_design.md | 6847dff1bc8276be3ecd509cd1dfbbd147ee3ed2d1a588ef323a06241341c921 | 5975 | read in full (pre-wording-change version) |
| docs/research/2026-09-13_gold11_mesh_development_v8_protocol.md | 65ed24e86453bcc094d08bc09b7c0d7fb15fd814787446c55a0a1dbd0a25cca5 | 7230 | read in full (pre-wording-change version) |
| tests/test_gold11_mesh_development_v8_source.py | a6d8433436d3b0d7e721bca3ee7e6644c2f65d5d607153b8019b57738df804c4 | 10687 | read in full |
| tests/test_cad_guided_global_registration_v4.py | 1cf31b12f373f1c0b8d72eecf5c6bd5ae86f704feeaaf0d14e5dda8031735204 | 36702 | test names; lines 100-123, 484-620 |
| tests/test_cad_guided_plane_initialization_v14_registration.py | b60896c36b56b825aa600a9f7a6064e9cb69fc93f14ae22238a82156c26aa9e3 | 3916 | grep (test names, gate strings) |
| tests/test_cad_guided_context_v4_engine_integration_v1.py | d82cf5df6bd758a2c2fb6786707e17d3e2bd9346381d44982b41f56087d7180e | 43929 | grep (imports, test names) |
| backend/cad_guided_mesh_offline_v15.py | 5edaae998a56c121a0aabf45fcc5b71d19e4b4aeab10209d42b89897dbd0f4f7 | 33955 | read in full; ast |
| backend/cad_guided_mesh_offline_v14.py | 76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749 | 33435 | ast (class fields/constants) |
| backend/cad_guided_mesh_offline_v12.py | 7c444bab2c197b0a6d9949ec6a15162286edb78bdfd081137e2907f16152c33e | 46206 | grep; `_publish_contexts` (1101-1120) |
| backend/cad_guided_mesh_offline_v2.py | fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf | 18036 | ast (MeshWorkflowPolicyV2 fields) |
| backend/cad_guided_plane_initialization_v14.py | eb1ad06a1847634585aeaa8e79779995d85f6311d17da4aea0dc96b7891b21d7 | 65110 | lines 1-170, 725-760, 790-935; greps |
| backend/cad_guided_plane_initialization_v12.py | 57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87 | 73393 | `_publish_contexts`; grep |
| backend/cad_guided_global_registration_v4.py | 5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273 | 37673 | lines 113-119, 430-560; greps; ast |
| backend/cad_guided_global_registration_v3.py | 088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0 | 35987 | ast; grep line 458/465 |
| backend/cad_guided_global_registration_v2.py | 5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3 | 20608 | grep line 219 |
| backend/cad_guided_global_registration_v1.py | 7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7 | 21030 | lines 25-70, 280-320; ast |
| backend/cad_guided_two_plane_context_v4.py | e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25 | 26196 | lines 55-80; greps; ast |
| backend/cad_guided_two_plane_context_v3.py | 9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d | 25845 | ast (fields) |
| backend/cad_guided_two_plane_context_v2.py | d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663 | 11130 | ast (fields) |
| backend/cad_guided_two_plane_context_v1.py | 9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a | 36075 | ast (fields) |
| backend/cad_guided_raw_extrema_index_v3.py | 83dea6529003f1d5deae592c9d35e566a77e2c9f6f3af8f641928f5465845bc0 | 23978 | lines 44-67; greps |
| backend/cad_guided_triangle_surface_v4.py | f79b73b37be2684f73f7d467d5aa1074f8a27d968cdf52e9fb1d652dae02422c | 37248 | `_budgets` (134-147); greps; ast |
| backend/cad_guided_triangle_surface_v3.py | 8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10 | 32098 | ast (fields) |
| backend/cad_guided_stage_policies_v2.py | 699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d | 3328 | read in full |
| backend/cad_guided_context_measurement_v1.py | a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9 | 39387 | lines 1-60, 160-260; greps |
| backend/cad_guided_support_cover_measurement_v2.py | 8783f95bfce161ae4d9cd0c5a2532114efa510907b50479a42896747e20d56aa | 27567 | lines 1-70, 124-207; greps |
| backend/cad_guided_exact_support_cover_v1.py | 70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47 | 10167 | grep |
| backend/cad_guided_local_measurement_v1.py | 5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b | 12215 | grep |
| backend/cad_guided_measurement.py | d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb | 25964 | grep |
| backend/cad_guided_instances.py | d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda | 10296 | grep |

Closure modules parsed only by `ast` for import edges (no other content
opened), start = end:

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| backend/cad_guided_cad_sampling_v1.py | f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a | 26439 |
| backend/cad_guided_cad_surface_v2.py | b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1 | 7759 |
| backend/cad_guided_composite_surface_roles_v1.py | f87f83f9e235059f6d33a92fba4bd329197a61de6a23039212acd8ef90b3b68f | 25114 |
| backend/cad_guided_context_surface_roles_v1.py | c042408109d260888834ebacc9ad3b62256fda589f1ea8cbba31164b2bea9a62 | 25369 |
| backend/cad_guided_face_box_screen_v1.py | c8f66cc25f5c4cb6a6ba3c1e1beeb205540813796bb2ab30d72eb596cdf3abb7 | 6121 |
| backend/cad_guided_mesh_input_v1.py | c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917 | 29182 |
| backend/cad_guided_mesh_offline_v1.py | 55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba | 12249 |
| backend/cad_guided_mesh_offline_v8.py | afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec | 18745 |
| backend/cad_guided_mesh_plane_proposals_v4.py | 55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48 | 22680 |
| backend/cad_guided_mesh_topology_v4.py | 4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734 | 5153 |
| backend/cad_guided_native_features.py | f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f | 17966 |
| backend/cad_guided_native_features_v1.py | 3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b | 41518 |
| backend/cad_guided_noisy_mesh_planes_v5.py | 9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75 | 31259 |
| backend/cad_guided_noisy_mesh_planes_v7.py | b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f | 17767 |
| backend/cad_guided_plane_hypotheses_v4.py | 7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370 | 15192 |
| backend/cad_guided_plane_initialization_v4.py | a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724 | 38276 |
| backend/cad_guided_plane_initialization_v8.py | 3b4220637fa19aba1e0384071f55e091705476046468327a306486078d7f7438 | 32544 |
| backend/cad_guided_plane_search_v6.py | aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676 | 38804 |
| backend/cad_guided_report_envelope_v6.py | c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e | 12480 |
| backend/cad_guided_scan_plane_preparation_v1.py | efea482eb3066a43028f4f3a3b5bfa7ae994fc6f1efc165aace109c05e30e5b5 | 11319 |
| backend/cad_guided_seeded_plane_proposals_v1.py | 14754498d5465a97ff0d8ab92fa587a5383df3cb45f9828b62532eee94e74564 | 13087 |
| backend/cad_guided_stage_policies_v1.py | 5c3d1252ac5f38f422f8a9a22c98a3d4db1094cd0956c928e9800f75e3ee826c | 3186 |
| backend/cad_guided_triangle_distance_bounds_v3.py | 2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7 | 10776 |
| backend/cad_guided_triangle_surface_v2.py | 7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1 | 23511 |
| backend/cad_guided_two_plane_schedule_v1.py | 1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f | 26562 |
| backend/cad_guided_two_plane_schedule_v2.py | 07b24b8e53b08a408f56ab94dc442e7db7f4e3ef6a5cd4729629860cace99977 | 25030 |

Also touched: one `grep -rn max_inlier_plane_p95_mm` over `backend/ scripts/
tests/` printed single matching lines from 18 further test files (hashes
recorded in the review transcript; none is cited here), and one
`grep -l importlib|__import__ backend/cad_guided_*.py` name-only scan over 81
files (no match). Existence-only checks: `docs/research/
2026-09-12_fable51_review_response_v1.md` (exists, not read); `tests/` listing.

## Hunk classification (V7 -> V8)

Complete `diff` (exit 1, 32 hunks, all listed). Classes: I = identity-only,
E = engine wiring, S = scoring transport, C = declared config delta,
O = OTHER.

| # | V7 lines | V8 lines | Content | Class |
| --- | --- | --- | --- | --- |
| 1 | 1-8 | 1-12 | module docstring | I |
| 2 | 35-36 | 39-40 | `RUN`, `PROTOCOL` | I |
| 3 | 42-44 | 46-48 | `SOURCE_BASIS`=V7 runner, `SOURCE_BASIS_SHA`=e910d562... (verified), `SOURCE_PROFILE.implementation` | I |
| 4 | 47-49 | 51-55 | `source_delta` strings; `scientific_config_or_scorer_change` False->True; new `scorer_change=False` | I (declaration; consumed only by equality at V8:956) |
| 5 | 53 | 59 | own runner path in `SOURCES` | I |
| 6 | 88-90 | 94-99 | `SOURCES`: +v6 runner (was implicit via basis), +extrema V3, context V4, surface V4, registration V4, stage policies V2, init V14, mesh V15, projection V3; -mesh V14, init V13, context V3, extrema V1, projection V2 (70 -> 74 entries, all unique, all exist) | I/E closure |
| 7 | 92 | 101 | delta-design doc path | I |
| 8 | 98-99 | 107-108 | `TIMING_RELEASE_STATUS` (consumed at V8:393), `STAGE_TELEMETRY_STATUS` (unconsumed) | I |
| 9 | 109 | - | `PENDING_THREE_CASE_DIAGNOSTIC` deleted; V7 had no other reference | **O** (dead constant; P3-1) |
| 10 | 113-114, 116 | 121-122 | `DRAFT_RELEASE_BLOCKERS`: two V7 entries replaced, `MAXIMUM_119_FILE_FINALIZATION_AND_256KIB_CONTROL_ENVELOPE_NOT_QUALIFIED` dropped; tuple is referenced nowhere else (V8:120 only) | **O** (unconsumed declaration; P3-1) |
| 11 | 122-125, 129-132 | 129-133, 137-141 | `ARM_DEFINITIONS`: V15/V14 identities, `two_plane_context_mode="v4"`, new `registration_implementation` | I/E |
| 12 | 137 | 146-157 | `COMPARATOR`; new `CONFIG_DELTA_FROM_V7` (3 entries with reasons) | I + C |
| 13 | 171 | 191 | `CONFIG.registration.max_inlier_plane_p95_mm` .5 -> 1. | C |
| 14 | 221 | 241 | `CONFIG.surface_budgets.max_ties_per_query` 32 -> 256 | C |
| 15 | 239-243 | 259-263 | `TWO_PLANE_DECLARATION` context strings; `CONTEXT_V4_REQUEST` mode/implementation/schema | I |
| 16 | 253 | 273 | `max_extrema_coordinate_visits` 256M -> 512M | C |
| 17 | 255a, 258, 260 | 276, 279, 281 | `scan_extrema_partition` added; chunk scope LEAF256 -> LEAF64; guard status ..._V3_INVARIANTS | I (context-V4 declarations; equal to `_base()` in context_v4:66-78) |
| 18 | 263 | 284 | `CONTEXT_REQUESTS_BY_ARM` from V4 request | I |
| 19 | 289, 297 | 310, 318 | `SCORING_SOURCES` projection V3; `wrapper_schema` ..._V3 | S |
| 20 | 371-383 | 392-404 | `require_release` error codes | I |
| 21 | 397-399 | 418-421 | `arm_request` identities + new `registration_implementation` field | I (one new request field) |
| 22 | 440, 462, 486, 517 | 462, 484, 508, 539 | private scoring namespace name; `modules["gold11_schedule_projection_v3"]`; `project_..._v3`; `revalidate_..._v3` | S |
| 23 | 904-906, 932-933, 950 | 926-929, 955-957, 974 | freeze/verify: error codes, bundle schema V8, `config_delta_from_v7` frozen and verified on load | I + C |
| 24 | 1092-1093 | 1116-1117 | `LEAF_SCHEMA`, `RUNNER_SOURCE` | I |
| 25 | 1186 | 1210 | leaf cache prefix | I |
| 26 | 1344-1382 | 1368-1407 | `workflow_entry_and_policies`: `TriangleSurfaceBudgetsV4`, `SurfaceRegistrationBudgetsV4`, `MeshWorkflowPolicyV15`/`inspect_cad_guided_mesh_v15`, V14 `_context_configuration("v4")`, `TwoPlaneContextBudgetsV4`, added `REGISTRATION_IMPLEMENTATION` equality check, error code, mode "v4" | E |
| 27 | 1390 | 1415 | sampler module name | I |
| 28 | 1474 | 1499 | leaf executes `source/scripts/run_gold11_mesh_development_v8.py` | I |
| 29 | 1859-1861 | 1884-1887 | report schema/arm identities; `config_delta_from_v7` in report | I + C |
| 30 | 1972, 2120, 2347 | 1998, 2146, 2373 | custody schema strings; `experiment_schema` check | I |
| 31 | 2642-2660 | 2668-2686 | prestart error codes | I |
| 32 | 2719 | 2745 | coordinator cache prefix | I |

Unchanged lines that should have changed (not a diff hunk, found by reading):
`STAGE_SAMPLE_SOURCE_PATHS` (V8:112-113) still appends
`"scripts/run_gold11_mesh_development_v7.py"`; V6 (v6:93-94) and V7
(v7:103-104) each named their own runner. See P2-1.

Confirmed unchanged: `REVIEW_SHA`/`CATALOG_SHA`/`SELECTED_SHA` (V8:42-45 =
V7:38-41; review script hash re-verified); `LIMITS` (V8:164-175);
`CERTIFICATE_CAPS` (178-179); `STAGE_SAMPLER_SHA` (110), `ARTIFACT_SOURCE_SHA`
(326), `SUPERVISION_SOURCE_SHA` (1113), all three re-verified against the
files; `retained_scorer_request_transport` (57); `POSITION_ERROR_DECLARATION`,
`RETAINED_CONTROL_POLICY`, all other CONFIG maps, `SEEDED_PLANE_BUDGETS`,
`COMPOSITE_PLANE_BUDGETS`, `SCHEDULE_V2_BUDGETS`, `SCORING_TRANSPORT` except
the wrapper string, `ARTIFACT_POLICY`, `STAGE_LOG_DECLARATION`; no hunk touches
custody, supervision, terminal fill, prestart admission, audit hook, scoring
reconstruction or matcher code beyond string renames. Scorer thresholds remain
5 deg / 4 mm / .60 / .60 (`gold11_contact_scoring_v2.py:684-685, 749-750`;
`gold11_mesh_scoring_v1.py:137-138`).

## Projection V2 -> V3

Diff (13 hunks): docstring; `SCHEMA` ..._V3 (V3:20); identities V14/V15/
kd-partition context (25-27); new `REGISTRATION_IMPLEMENTATION` (28); scope
string (29); 512M maximum (39); added `scan_extrema_partition`, LEAF64, ..._V3
guard status (43, 46, 48); `_ARRAY_KEYS` + `registration_implementation`
(53-55); `_context_request` mode/schema v4 (59-60); workflow-level
`registration_implementation` check (78-82); registration-evidence-level check
(104-109); helper schema v4 (137); array-binding key set and equality
(168-175); function renames (189, 230, 241). Everything else is the V2 text.

- `project_gold11_schedule_workflow_v3` (V3:189) and
  `revalidate_gold11_schedule_projection_v3` (V3:230) exist with the V2
  signatures; the runner calls them at V8:508 and V8:539 with the same keyword
  set as V7 (`projection_request`, V8:513-523 unchanged).
- Array bindings: V15 `_registration_binding` returns
  `dict(binding, **context_binding, registration_implementation=..., ...)`
  (V15:230); `context_binding` is V14 `_bind_context_publication` ->
  `initialization_implementation`, `two_plane_context_mode="v4"`,
  `two_plane_context_implementation`, `two_plane_context_configuration_sha256`
  (V14:109-112). Key filter at V3:168-169 therefore yields exactly
  `_ARRAY_KEYS` (V3:53-55). Consistent.
- Registration evidence: V14 sets `evidence["registration_implementation"]`
  unconditionally before the try block (V14:812) and V15 re-checks it
  (V15:72-73). V3:104-109 consistent.
- Workflow top level: V15 base output carries `registration_implementation`
  (V15:280) and `context_request_scope` (281); V3:76-86 consistent.
- Parent key sets: V15 publishes `_context_publication` on the workflow and on
  `policies` (V15:354-355), V14 on evidence (V14:842) and on the initialization
  result (V14:700); each is exactly `_PARENT_KEYS` (V3:51-52). Other
  `two_plane_schedule*` keys do not match the `two_plane_context` prefix.
- Budget maxima/declarations: V3:30-50 equal the runner request (V8:267-283)
  and `_context_v4._base()` (context_v4:66-78) and V4 maxima (context_v4:43-46,
  extrema_v3:62-67).
- Contact scorer: `WORKFLOW_SCHEMA = "CAD_GUIDED_MESH_OFFLINE_V13"`
  (`gold11_contact_scoring_v2.py:30`, required at 519, connected at 537); V15
  publishes exactly that schema (V15:247, 271). The scorer never binds
  `implementation`; that binding lives in projection V3 (76-77). Accepts V15
  output.
- Caveat: the strict branch of `_context_evidence` runs only when
  `payload["phase"] == "COMPLETE"` (V3:217). V7 produced no COMPLETE payload,
  so the V3 additions have not been exercised by any real result and there is
  no `tests/test_gold11_schedule_projection_v3.py` (only V1/V2 tests exist).
  See P2-2.

## Closure check

`ast` walk from `backend/cad_guided_mesh_offline_v15.py` following
`from . import x`, `from .x import y` (module-level and inside functions),
`import backend.x`, `from backend.x import y`: 48 modules. `SOURCES` declares
53 backend modules; **missing: none**. Declared but outside the V15 closure
(preservation oracles, harmless): `cad_guided_mesh_offline_v10/v11/v13`,
`cad_guided_plane_initialization_v10/v11`. No `importlib`/`__import__` in any
`backend/cad_guided_*.py`. `backend/__init__.py` does not exist; the worker
synthesizes the namespace package (V8:1094-1097) and the audit hook admits
`snapshot/backend`, the cache and the runtime roots (V8:1015-1019), so every
module the engine can import is in the frozen snapshot. Function-local imports
covered: V15:305-309, 328, 339, 343, 373, 379, 465-466; V14:129, 243, 873, 875.
Also verified: `leaf_sources("INFERENCE")` (V8:1140-1143) ships all
`backend/` sources plus runner/artifact/sampler.

Constructor keyword sets (ast) vs runner CONFIG: `TriangleSurfaceBudgetsV4`
(12 fields = 11 CONFIG keys + certificate cap), `SurfaceRegistrationBudgetsV4`
(7 = 6 + cap), `TwoPlaneContextBudgetsV4` (18 inherited V1/V2 fields + 5 = 23
request keys), `MeshWorkflowPolicyV15` (V2 fields: 5 CONFIG keys +
`native_adapter_version`). All match. At-cap values accepted inclusively:
`max_ties_per_query=256` (`_budgets`, surface_v4:141-147, `MAX_TIES_PER_QUERY`
= 256 at :56); `max_extrema_coordinate_visits=512M` (context_v4:59-61,
extrema_v3:145-146). Policy order `huber .25 <= p95 1.0 <= corr 2.0` passes
`WholePartRegistrationPolicyV1.__post_init__` (registration_v1:64-67).

## Provisional pose-admission gate (0.5 -> 1.0 mm)

Applies identically to both arms (`CONFIG["registration"]` is shared, V8:189-
197; `arm_request` embeds it as `common_config`, V8:421).

Consumers of `max_inlier_plane_p95_mm` on the V8 engine path:

1. `WholePartRegistrationPolicyV1.__post_init__` (registration_v1:43-68):
   type/finite/range checks, then the order constraint `huber_delta_mm <=
   max_inlier_plane_p95_mm <= max_correspondence_distance_mm` (64-67). 1.0
   passes. Re-run on every `_policy_snapshot` (registration_v4:113-118) and in
   V14 `_frozen_configuration_v4` via `_old._registration_policy` (V14:151).
2. `registration_v4._diagnose` (428-537): line 481 adds candidate reason
   `SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE` when the p95 **upper**
   bound over definite, unambiguous support exceeds the gate; line 488 adds
   possible-eligibility reason `POSSIBLE_RESIDUAL_LOWER_BOUND_OUTSIDE_GATE`.
   In connected mode these are recorded on the candidate row (V14:911) but do
   **not** gate context admission: `_publish_contexts` (plane_init_v12:1101-
   1120) selects by `conditional_context_support.status == "SUPPORTED_CONTEXT"`
   only (1110-1111); V15 requires only that status for a context index
   (V15:139), and `context_measurement_v1:210-222` re-derives the same index
   list. Diagnostic-only in V8.
3. `registration_v4._select` (547-...): consumes `reasons`/possible
   eligibility and the ambiguity margins (`pose_class_distance_mm`,
   `ambiguity_*`). **Confirmed not used in connected mode**: V14:925-928
   `if connected_measurement_contexts: _publish_contexts(result) else:
   _v3._select(...)`; V15 refuses `connected_measurement_contexts is not True`
   (V15:286-287); the runner passes `connected_measurement_contexts=True`
   (V8:1390). Consequently the only cross-candidate veto in the engine
   (`AMBIGUOUS_POSSIBLE_GENERATED_WHOLE_PART_POSES`) is off the V8 path; there is
   no ambiguity check between admitted contexts at all (unchanged from V7, but
   the looser gate widens the admitted set).
4. `plane_initialization_v14._context_support` (730-752) is the deciding gate.
   Rows = `definite_proximity` = `_support_masks(query, 2.0)["definite_support"]`
   (distance upper bound <= `max_correspondence_distance_mm` = 2 mm; no
   ambiguity exclusion, stage_policies_v2:61-67), over the registration row
   selection (<= `max_registration_scan_rows` = 50000). Requires: count >= 36;
   coverage lower bound >= .75 (743); reverse CAD-sample coverage at 2 mm >=
   .25 (743, value from `_diagnose`, V14:914); `p95_upper` of the unsigned
   point-to-triangle-union distance intervals over those rows <= 1.0 (745, was
   .5). No observability/rank term, no `_diagnose` reasons, no pose-class
   comparison. Every SUPPORTED candidate (<= `max_refined_start_groups` = 48)
   becomes a measurement context.

What 1.0 admits that 0.5 did not (other gates unchanged): the 2 mm coverage
gates still require >= 75% of selected rows within 2 mm and >= 25% of CAD
samples within 2 mm of scan rows, so gross alternatives (flipped sheet with
bends of ordinary height, a pose sliding a flange off its neighbour) remain
rejected by coverage as before. The gate change only admits poses whose
proximal-row p95 upper distance lies in (0.5, 1.0]:

- Shifted/tilted near-true poses: translations normal to the dominant surfaces
  up to ~1 mm, or small tilts (about 0.2 deg over a 300 mm sheet). Several such
  contexts of one true pose will typically be admitted together.
- Sheet-thickness offsets: a pose registered to the wrong skin, or to the CAD
  mid-surface, has residual about the thickness t; V7 rejected t > 0.5, V8
  admits t <= 1.0. This is evidently the intended effect (the recorded
  CG4BDD p95 upper 0.666 mm).
- Alternative panel correspondence on nearly symmetric parts: a pose mapping
  the scan onto a similar CAD region whose asymmetric surfaces differ by
  0.5-1.0 mm is now admitted; differences > 2 mm on > 25% of rows are still
  rejected. Reflections are impossible (det = +1 enforced, registration_v4:542;
  V14 `_pose`); a 180 deg flip of a nearly planar part is a rotation and is
  admitted at either gate only if the flipped bends stay within 2 mm on 75% of
  rows, i.e. essentially never for ordinary flange heights.

Downstream protections before a prediction is emitted:

- A context only places windows; predictions are free-circle fits of raw scan
  rows in the scan frame (`context_measurement_v1:62-69`, `_raw_feature` ->
  `_crop_rows` at 285). A wrong pose cannot fabricate geometry; it selects
  rows. Window placement tolerates the admitted offset: `search_margin_mm` = 3,
  flange window 2.5-12.5 mm along the axis, `plane_exclusion_mm` = .075
  (`cad_guided_measurement.py:302-325`), all >> 1 mm.
- Fit gates: free-circle residual <= .05 mm (measurement.py:132, 211, 278),
  tangency <= .1 mm (349; context_measurement_v1:106), sweep >= 20 deg (355;
  context_measurement_v1:116), plus `max_circle_condition`,
  `min_plane_spread_ratio`, `max_normal_deviation_deg` (FIT_FIELDS). A flat or
  non-cylindrical region does not pass these; a real bend of another radius can
  (radius is free; nominal radius only bounds the crop).
- Finite support instance (instances.py:88-166): axial gap <= 4 mm, angular
  gap <= 5 deg, sweep >= 20 deg, `min_local_points` = 3 per slab.
- Cover merging: all support records from all contexts are grouped into
  maximal raw-row ownership components and only the full cover is refit
  (`support_cover_measurement_v2:154-189`; `plan_exact_support_covers_v1`).
  Duplicate contexts of the same physical bend collapse to one prediction;
  member records are never concatenated.
- Publication validation (`validate_support_cover_publication_v2`, V15:465-
  467) and per-context binding (V15:449-451) precede any prediction.
- Scorer: one-to-one minimum-cost matcher with 5 deg / 4 mm / .60 / .60. A
  <= 1 mm pose offset is far inside the 4 mm transverse tolerance, and the
  prediction geometry is raw-row geometry anyway, so the gate cannot convert a
  correctly measured bend into a false positive by displacement.

Residual risks the gate introduces (both arms):

- Budget refusals from more admitted contexts (whole-case zero predictions):
  `CONTEXT_WINDOW_PREFLIGHT_BUDGET` (contexts x features <= 4096) and
  `FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET` (contexts x features x raw rows <= 64M)
  are checked before any window (`context_measurement_v1:240-241`); e.g.
  500k raw rows x 8 features x 16 contexts already reaches 64M. Aggregate row
  budgets (`MAX_TOTAL_LOCAL_ROWS` 4M at 290/341/399, cover refit 4M at
  support_cover_v2:178) scale with context count too. A case can therefore
  flip from predictions to a retained refusal purely because more near-true
  contexts pass at 1.0 mm. Not a bug; must be a declared expected outcome
  class (P2-4).
- Wrong-correspondence contexts that window a real bend not in the 71
  references, or only part of a bend (span overlap < .60): false positives or
  FP+FN pairs. No cross-context ambiguity veto exists in connected mode.
- No new false-negative mechanism: a context admitted at 0.5 is admitted at 1.0
  (monotone), and covers only add records.

Uncertainty labelling the result/report must carry. Present today: V15
top-level `uncertainty_status="UNCALIBRATED"`, `conformance="INDETERMINATE"`,
`registration_mode="FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE"`,
`production_use_allowed=False`, `position_error_model.status` (V15:271-284);
evidence `global_pose_uniqueness="NOT_ESTABLISHED"` (V14:815), policy snapshot
with `max_inlier_plane_p95_mm: 1.0` (V14:895), per-context `scope=
"REPRESENTED_COORDINATE_PROXIMITY_NOT_POSE_OR_CORRESPONDENCE_CERTIFICATE"` and
`accepted_registration=False` (V14:747-752); runner `config_delta_from_v7` with
reason text frozen in the bundle (V8:929), verified on load (957) and copied
into the enriched report (1887). Missing/misleading: (a) the evidence string
`residual_gate="FINITE_DISTANCE_P95_UPPER_USING_UNCHANGED_LEGACY_NUMERIC_
THRESHOLD"` (V14:824) will now be published under a changed numeric value; it
must be read as "the policy field's value", which V8 sets to 1.0, and POST must
not read it as 0.5; (b) no per-arm/per-case label such as
"POSE_ADMISSION_PROVISIONAL_1P0MM_DEVELOPMENT_GATE_NOT_CALIBRATED_CONFORMANCE"
exists in the report arm identities or per-case dispositions; every emitted
support in V8 is conditional on a 1.0 mm development admission gate that is not
a tolerance, calibration or conformance statement, and the protocol's claims
section should say so next to the F1 numbers (P3-4).

Existing adversarial evidence and gaps (no new broad phase requested):

- `tests/test_cad_guided_global_registration_v4.py`:
  `test_possible_only_distinct_competitor_blocks_definite_winner` (512),
  `test_noncompetitive_possible_row_does_not_invent_pose_ambiguity` (529),
  `test_possible_eligibility_alone_never_publishes_a_pose` (558),
  `test_incomplete_or_invalid_late_candidate_clears_prior_selection` (567) all
  drive `_select` with declared scores ("NOT geometric inference", 485) under a
  strict constructed policy (p95 .05, 120). They establish the non-connected
  selection contract only; none of that path runs in V8.
- `tests/test_cad_guided_plane_initialization_v14_registration.py:50-67`: one
  real-chain positive control (two-sheet fixture yields SUPPORTED_CONTEXT). No
  negative control (displaced/alternative pose -> REJECTED_CONTEXT).
- `tests/test_cad_guided_context_v4_engine_integration_v1.py`: workflow
  plumbing with p95 .1 (143); synthetic SUPPORTED_CONTEXT rows (457). No
  geometric admission test.
- Gap: no test in the repository evaluates `_context_support` at 1.0 mm, none
  shows a pose that is REJECTED at .5 and SUPPORTED at 1.0, none exercises
  connected-mode cover merging of two near-duplicate admitted contexts, and
  none exercises the context-count budget refusal. Every existing policy uses
  .05-.5 mm. The V8 outcome for the gate therefore rests on the physical run's
  retained ledgers, exactly as the protocol states; the paired before/after
  against V7 remains the only efficacy evidence.

## Findings

**P1: none found.**

P2-1 (contract; stale identity). `STAGE_SAMPLE_SOURCE_PATHS` (V8:112-113)
appends `scripts/run_gold11_mesh_development_v7.py` instead of the V8 runner
the leaf executes (V8:1499). The sampler maps `co_filename` to names lexically
(`gold11_engine_stage_sampler_v1.py:34-50`, no file read, so no audit-hook or
freeze failure) and names only mapped frames (57-66); V8 runner frames will
never be named and V7 frames can never occur. Telemetry-only (samples cannot
alter results, V8:1435-1466), but it contradicts "identities only" and
`STAGE_TELEMETRY_STATUS="UNCHANGED_SAMPLER_WIRING_V8_RELEASE"`, and the V8
source test does not cover it. Repair: replace the literal with
`"scripts/run_gold11_mesh_development_v8.py"`; add to
`tests/test_gold11_mesh_development_v8_source.py` an assertion that
`runner.RUNNER_SOURCE in runner.STAGE_SAMPLE_SOURCE_PATHS` and no other
`run_gold11_*` path is present. (One-line source change; re-hash and re-freeze.)

P2-2 (test gap). Projection V3's additions (V3:78-82, 104-109, 168-175) have
no test: no `tests/test_gold11_schedule_projection_v3.py`; the V8 source test
only checks the file name (69-70, 107). The strict `completed` branch was
never exercised by V7 output. Repair: copy the V2 test with V3 identities plus
one synthetic COMPLETE payload carrying `registration_implementation` at the
workflow, evidence and array-binding levels, and negative cases removing or
altering it at each level (expect `REGISTRATION_V4_IMPLEMENTATION_REQUIRED:*`,
`MISSING_COMPLETE_REGISTRATION_IMPLEMENTATION:*`,
`CONTEXT_ARRAY_BINDING_MISMATCH`).

P2-3 (test gap, gate). No adversarial evidence for the 1.0 mm gate (see
section above). Minimal repair without a broad phase: one `_context_support`
unit test on the existing two-sheet fixture with a 0.75 mm normal displacement
(REJECTED at .5, SUPPORTED at 1.0) and one connected-mode test that two
near-duplicate SUPPORTED contexts produce a single cover prediction.

P2-4 (protocol contract). The protocol does not declare the new outcome class
"more admitted contexts -> `CONTEXT_WINDOW_PREFLIGHT_BUDGET` /
`FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET` / aggregate-row refusal with zero
predictions" (`context_measurement_v1:240-241, 290, 341, 399`). Add it to the
expected retained negative outcomes so POST does not misread such a refusal as
an engine fault.

P3-1 (documentation/declarations). Hunks 9-10: `PENDING_THREE_CASE_DIAGNOSTIC`
removed (dead) and `DRAFT_RELEASE_BLOCKERS` rewritten; neither is consumed.
The new entries read as satisfied conditions ("...REVIEWED_BEFORE_FREEZE",
"...TESTS_PASSED"), asserted before this review existed, and the dropped
119-file/256 KiB entry is not substantiated in-source (though harmless: caps
are per directory, `scientific_manifest_entries`=128 at V8:331/2491, helper
`_MAX_ENTRIES`=1024; `source/backend` holds 53 files, `source/scripts` 17; the
inert freeze test asserts the bundle < 256 KiB, test:185). Reword as blockers
or drop the tuple; record the release conditions in the protocol instead.

P3-2 (documentation). Claims not verifiable from source and recorded as such:
V7 refusal tallies (10/4/2/2/2/1/1), "CG4BDD 98.8%/99.1% coverage, p95 upper
0.666 mm", all exactness/cost numbers (122.0M, 10.19M, 150.3M, 4.8 s, ...),
"Independent preliminary reviews (Codex, no P1)". Cite retained ledger paths and
hashes for each. Verified against source: V7 basis size/hash, case authority
hashes (review script re-hashed; catalog/selected not checked), the budget
table and 22*3540+180+2400 = 80460, scorer thresholds, "nine" context
declarations (V14:56-60), 64-row leaves (extrema_v3:68), 512M, 256 ties,
V4 registration label, stage-policies-V2 binding.

P3-3 (documentation). Delta design item 4 says "arm request field ...
unchanged"; the arm request gained `registration_implementation` (V8:420).
Reword ("one identity field added; all values unchanged").

P3-4 (labelling). Add an explicit provisional-gate label (see "Uncertainty
labelling" above) to the enriched report's arm identities and to the protocol
claims section, and document that `residual_gate=...UNCHANGED_LEGACY_NUMERIC_
THRESHOLD` (V14:824) denotes the policy field whose V8 value is 1.0.

P3-5 (declaration). `STAGE_TELEMETRY_STATUS` (V8:108) is unconsumed and, until
P2-1 is fixed, inaccurate.

## What was not checked

- No test, freeze, verify, execute or geometry job was run; the assertions in
  `tests/test_gold11_mesh_development_v8_source.py` were read, not executed.
- No attempt directory, frozen snapshot, `data/` (catalog/selected hashes not
  re-verified), Gold/reference file, review-script case list, `.env` or
  credentials were opened.
- Engine mathematics of extrema V3, context V4, surface V4, registration V4,
  initializer V14 and V15 were not re-reviewed; only identities, constants,
  validators, publication keys and control flow relevant to the runner binding
  and the pose gate were read. `docs/research/2026-09-12_fable51_review_
  response_v1.md` was not read.
- `gold11_schedule_projection_v1.py`, `gold11_contact_scoring_v2.py` internals
  beyond schema/threshold strings, the supervision and artifact helpers beyond
  hash pins and entry caps, and the five preservation-only modules were not
  read.
- Numerical behaviour of the 1.0 mm gate on the eleven cases is unknown to
  this review; only the mechanism was traced.
- Post-review wording changes to the two documents (announced by the
  coordinator) are outside this review; the reviewed versions are the hashes
  in the table above.

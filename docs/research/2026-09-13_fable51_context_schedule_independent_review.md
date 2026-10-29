# Independent review: measurement-context schedule delta (2026-09-13)

Reviewer: independent (Claude Fable 5.1, reviewer role; not the author of the
reviewed files). Read-only review of `/Users/idantal/ShermanQC/FullProject`,
2026-09-12 ~23:00-23:30 UTC. Executed only `cat/sed/grep/diff/shasum/wc/date`,
`git status --short` on seven paths, and `.venv/bin/python` heredocs that parse
source with `ast` and print definitions (no `backend`/`scripts` module imported;
no test, freeze, execute or geometry job; no network). No `data/`, `output*/`,
attempt/frozen directory, Gold/reference file, `.env` or credential opened.
Disclosure: one scratch file (`_v7_v8.diff`, the V7->V8 `diff` output) was
written into the reviewer's tmp directory and deleted in the same command; all
later diffs were piped. This report (rewritten once after a corrupted first
save) is the only file that remains.

Scope: only the deltas since the three earlier reviews (wiring V14/V15, runner
V8, review-delta recheck): items A-E of the brief.

## Decision and limits

- **No P1 found.** I consider the schedule a legitimate capacity policy: it
  selects among contexts the unchanged V12 publication has already admitted,
  using only each context's own scan-vs-CAD evidence, the frozen V2 action
  metric with the frozen `pose_class_distance_mm`, and the candidate index; it
  is fixed source plus frozen CONFIG, runs inside the inference leaf with no
  Gold/label/file/random access, is identical for both arms, keeps every
  omitted/duplicate context's evidence intact, and cannot break any consumer
  on the result path (each re-derives or only shape-checks the index list).
  One framing caveat (P3-2): the 0.1 mm pose-class dedup is unconditional (it
  also runs below the cap and with `maximum=None`), so the honest label is
  "dedup + capacity cap", not capacity alone; the reporting label
  `DETERMINISTIC_EVIDENCE_RANKED_CAPACITY_SCHEDULE_NOT_POSE_ACCEPTANCE_OR_UNIQUENESS`
  is otherwise accurate.
- P2: four (two test gaps, one incomplete frozen declaration, one post-review
  hash drift to record before freeze). P3: six.
- Limits: nothing executed, so "tests pass" is not reproduced. The reviewed-V8
  runner bytes (`98ae00a0...`, 171255 B) and the reviewed surface-V4 /
  registration-V4-test bytes are unrecoverable (all untracked), so the runner
  delta was isolated by subtracting the runner review's hunk table from the
  current V7->V8 diff, and the two drifted files were checked only at the lines
  the recheck's P3-2/P3-3 repairs name. Actual registration-report sizes
  against the 96 MiB cap and the number of SUPPORTED contexts the two-sheet
  fixture yields are unknown here.

## Examined bindings

### In-scope file identities (SHA-256, bytes): start == end, no change during review

| SHA-256 | bytes | file |
|---|---:|---|
| 46248b941313f1fafd9eb8cdd1ea0a01453dea4b92cde72d5051da39bc0e99d2 | 70521 | backend/cad_guided_plane_initialization_v14.py (wiring review: eb1ad06a.../65110) |
| 681ad505b39f148a6e1d9b72e5d58e1e3d54a4e3937b2d617fce71ca96454c6f | 35409 | backend/cad_guided_mesh_offline_v15.py (was 5edaae99.../33955) |
| ff052f720575542b3d0841a2b2901fcaafc9e03a4e48b0e9501cb2a1e15ce42b | 172613 | scripts/run_gold11_mesh_development_v8.py (runner review: 98ae00a0.../171255) |
| e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52 | 169180 | scripts/run_gold11_mesh_development_v7.py (= `SOURCE_BASIS_SHA`) |
| a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9 | 39387 | backend/cad_guided_context_measurement_v1.py (unchanged) |
| 9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff | 50413 | scripts/gold11_contact_scoring_v2.py (unchanged) |
| a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245 | 17231 | scripts/gold11_schedule_projection_v1.py (unchanged) |
| 763e0cd032521f237aaa999601f67e75644fe430b9837bedf1dc2c4d9be82d30 | 16058 | scripts/gold11_schedule_projection_v3.py (unchanged) |
| 6a9c32746e519b045a399a393d59776c04fd93909b8e201a405f628a8ab39efb | 13843 | tests/test_cad_guided_plane_initialization_v14_registration.py (was b60896c3.../3916) |
| 760247446c0e2e7db96ff1fb9537f783acffaeb204d965819637f6da600282ef | 46168 | tests/test_cad_guided_context_v4_engine_integration_v1.py (was d82cf5df.../43929) |
| 8a29d3104acd260ad98bf71b143c97d4ccc2f51635071f4a9e30ae3677959971 | 28814 | tests/test_gold11_schedule_projection_v3.py (new) |
| c4c7d389cf408a847d5cc134b86ebfc28871580757dee3b0439a085c8e63b13f | 8795 | docs/research/2026-09-13_gold11_mesh_development_v8_protocol.md |
| 5b8709b8a3b56fa996be52626d567b6b04a5e56016e4bce1194e16342f6a06f8 | 9204 | docs/research/2026-09-13_gold11_mesh_development_v8_delta_design.md |
| 436e21a89b18211fa9c5a5e08d0c83d51d2aea11cb00f0fee6b3c8efb4c9236c | 22035 | docs/research/2026-09-13_fable51_wiring_v14_v15_independent_review.md |
| 576007a86daf10bd51757b58f3c268c40d70fbcb6808bf221a237ec90acbce65 | 37980 | docs/research/2026-09-13_fable51_runner_v8_independent_review.md |
| 40ad51fbdab0a971296e59028e3f0f6ff9599024e7d1d6904d3248defb | 14076 | docs/research/2026-09-13_fable51_review_delta_independent_recheck.md |

### Auxiliary files (partial reads / ast extraction / grep), hashed at end

All equal the hashes the three earlier reviews bound, except the two marked
DRIFT and the expected V8 source-test change:

| SHA-256 | bytes | file |
|---|---:|---|
| 57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87 | 73393 | backend/cad_guided_plane_initialization_v12.py (`_publish_contexts` 1101-1120) |
| a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724 | 38276 | backend/cad_guided_plane_initialization_v4.py (`_Unavailable`, `_require`, `_pose`, `_reason`, `_LIMITS`) |
| 3b4220637fa19aba1e0384071f55e091705476046468327a306486078d7f7438 | 32544 | backend/cad_guided_plane_initialization_v8.py (`_registration_rows`) |
| 5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273 | 37673 | backend/cad_guided_global_registration_v4.py (`_policy_snapshot`, `_candidate_pose`, `_select`, `_refine`, `_metric_bounds`, `_p95_endpoint`, `_fraction_bound`, `_validate_query_result`, `_support_masks`, constants) |
| 7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7 | 21030 | backend/cad_guided_global_registration_v1.py (`WholePartRegistrationPolicyV1`, `_positions`, `_spatial_sample`) |
| 5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3 | 20608 | backend/cad_guided_global_registration_v2.py (`_norm`) |
| f0a01f666f143184960f78f4cec2aab5b7a39510485b136b85b63fdb4225f0d4 | 37320 | backend/cad_guided_triangle_surface_v4.py **DRIFT** (reviews bound f79b73b3.../37248); lines 28-36 and constants 57-61 read |
| 699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d | 3328 | backend/cad_guided_stage_policies_v2.py (`classify_stage_rows_v1`, `stage_row_counts_v1`) |
| 8783f95bfce161ae4d9cd0c5a2532114efa510907b50479a42896747e20d56aa | 27567 | backend/cad_guided_support_cover_measurement_v2.py (63-71, 124-207) |
| c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e | 12480 | backend/cad_guided_report_envelope_v6.py (`bound_initialization_report_v6` 225-253) |
| 55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba | 12249 | backend/cad_guided_mesh_offline_v1.py (`MeshWorkflowPolicyV1`) |
| fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf | 18036 | backend/cad_guided_mesh_offline_v2.py (`MeshWorkflowPolicyV2`) |
| 7c444bab2c197b0a6d9949ec6a15162286edb78bdfd081137e2907f16152c33e | 46206 | backend/cad_guided_mesh_offline_v12.py (grep) |
| 76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749 | 33435 | backend/cad_guided_mesh_offline_v14.py (grep) |
| 3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b | 41518 | backend/cad_guided_native_features_v1.py (grep: `features=()`/`tuple(features)`) |
| f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f | 17966 | backend/cad_guided_native_features.py (grep) |
| c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917 | 29182 | backend/cad_guided_mesh_input_v1.py (grep: `MIN_INSPECTION_ROWS = 12`) |
| a7c20b0c887af1d92726be54a976f63d53b12a6da5a7e0072f2d912796e0bdb4 | 11320 | tests/test_gold11_mesh_development_v8_source.py (grep; was a6d84334.../10687 - expected P2-1 test addition) |
| e108dc3a94651883013d15da655bb25c2a716851277b0b761c64e790f0969752 | 36927 | tests/test_cad_guided_global_registration_v4.py **DRIFT** (recheck bound 1cf31b12.../36702); lines 674-700 read |
| aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234 | 13492 | scripts/gold11_mesh_scoring_v1.py (grep) |

Hash-only drift check, all equal to the earlier reviews' bindings: context V4
e73be23d.../26196, extrema V3 83dea652.../23978, stage V1 5c3d1252.../3186,
registration V3 088c0b5e.../35987, surface V3 8e3222bd.../32098, init V13
79ee4de5.../38651, tests extrema V3 f78c18d1.../33604, context V4
ef57b38f.../37955, surface V4 cc80b5c7.../26622, schedule fixture
e7549772.../9786, projection V2 728920fe.../14677, sampler 80316d00.../8142
(= `STAGE_SAMPLER_SHA`), supervision 9eec779c.../42751, artifacts
3f4c5754.../24341, review script 41a79da1.../25616 (= `REVIEW_SHA`), runner V6
70355413.../166049. `git status --short`: V14, V15, runner V8, both docs and
both new tests are untracked (`??`).

### A. Schedule: consumer consistency (all checked against current text)

`_schedule_measurement_contexts` (V14:771-835) runs only after
`_publish_contexts` (V14:1009) and only when `result["status"] ==
"CONDITIONAL_MEASUREMENT_CONTEXTS"` (V14:1010-1011). It changes exactly:
`conditional_context_support.status` of non-kept SUPPORTED rows to
`SUPPORTED_CONTEXT_DUPLICATE_POSE_CLASS` / `SUPPORTED_CONTEXT_OMITTED_BY_MEASUREMENT_BUDGET`
(V14:816, 820), adds `supported_before_schedule=True` to every pre-schedule
SUPPORTED row and `duplicate_of_context_index` to duplicates, and replaces
`evidence["measurement_context_indices"]` by `sorted(kept)` (V14:826-828) plus
`measurement_context_selection` and `measurement_context_schedule` (V14:829-835).
No other row field (`execution_status`, `reasons`, `numerical_consumer_status`,
`numerical_support_policy`, `distance_model`, `optimizer_objective`,
`proximity_policy`, `accepted_registration`, `pose_sha256`, `rotation`,
`translation`) is touched.

| Consumer | What it requires of the ledger | Effect of the new statuses/keys | Result |
|---|---|---|---|
| context_measurement_v1:210-222 | `indices == [i for i,row in enumerate(candidates) if status == "SUPPORTED_CONTEXT"]`; every row ATTEMPTED, `CONSUMER_COMPLETE`, policy strings, `proximity_policy`, `accepted_registration is False`; `row["index"] == i` | re-labelled rows drop out of both sides; kept rows stay SUPPORTED; `kept` is sorted; `row["index"]` is positional (V14:675-676 `enumerate` -> `index=index`; candidates appended in `start_groups` order V14:971-972) | consistent |
| context_measurement_v1:236 | `pose_sha256` binds every candidate's pose | untouched | ok |
| context_measurement_v1:240-241 | `len(indices)*features <= 4096`, `*points <= 64M` | `len(indices) == len(kept) <= cap` | ok (the purpose) |
| V15:139-148 `_registration_binding(context_index=i)` | `i in measurement_context_indices`; `status == "SUPPORTED_CONTEXT"` | called for `i in kept` only (V15:478-480) | ok |
| V15:448-450 | list, non-empty, <= 48, unique | `kept` non-empty by the guard | ok |
| gold11_contact_scoring_v2:470-473 | list, 1..48, ints in [0,48), unique | shape only | ok |
| gold11_schedule_projection_v1:180-186 | <= 48, ints, unique; `len(bindings) == len(indices)` | V15 emits one binding per kept index | ok |
| gold11_schedule_projection_v3:164-175 | bindings <= 48, five array keys | unchanged binding dict | ok |
| V12 `_publish_contexts` (V14:1009) | status in {SUPPORTED, REJECTED} for every row | runs before the schedule; grep over backend/scripts/tests: nothing on the result path re-runs it or reads `REJECTED_CONTEXT` (only frozen mesh V12/V14 and test-side diagnostic ledgers) | ok |
| V14:1027-1038 `finally` + 1040-1044 gate | counts from `execution_status`/`reasons`/`numerical_consumer_status` | untouched; gate passes exactly as before | ok |
| `_bounded_report` (report_envelope_v6:225-253) | JSON `allow_nan=False`, size <= `max_report_bytes` (runner: 100_663_296, V8:247) | adds int/str/bool/None/list only (<= 48 dispositions, ~<= 6 KB) | ok; size margin not verifiable without data |
| exact key-set checks | contact V2:573/583 apply to the projection payload; projection V3:118-124/168-175 filter by `two_plane_context*`/`initialization_implementation`/`registration_implementation` prefixes | `measurement_context_cap` (workflow), `measurement_context_selection`/`_schedule` (evidence), `supported_before_schedule`/`duplicate_of_context_index` (rows) match none | ok |

Hidden/favourable selection: `rank` (V14:796-803) reads only
`finite_distance_metrics.p95_upper` and `geometric_scan_coverage_lower` of the
same row (both produced by `_context_support` V14:744-768 from scan rows vs CAD
triangles), tie-break by index -> total order, `sorted` deterministic
(V14:804). Dedup (V14:806-818): `refine@R.T+t` over the same `refine` sample
that `_refine` used, `max(_norm(diff)) <= policy.pose_class_distance_mm`, i.e.
the metric of `_select` (registration V4:611-612, 619) with the frozen policy
field (CONFIG `pose_class_distance_mm=.1`, V8:207); comparison against `kept`
only, first match wins -> deterministic. No Gold, label, file, environment or
random access anywhere in the function; V15 passes the cap from frozen CONFIG
(bundle `config` V8:942/971; `arm_requests.common_config` V8:433). Gold is
reachable only by the post-terminal scoring path (protocol). Identical policy
and cap for both arms (same scan/CAD/features).

Edge cases: empty supported list -> `_publish_contexts` sets UNAVAILABLE, the
schedule is not called; `maximum` must be exact `int` in 1..48 or `None`
(V14:789-790; `bool`/float/np.int refused -> `_Unavailable` (a `ValueError`,
init V4:58) -> outer except -> whole UNAVAILABLE, `attempted_measurement_context_indices`
= the pre-schedule list because the ledger replacement is the last statement);
V15 always passes `max(1, min(8, ...))` (int, 1..8). Non-finite/negative p95 or
coverage outside [0,1] refused (V14:800-802); for SUPPORTED rows they are
always finite Python floats (`_metric_bounds` requires finite >= 0 and
`_p95_endpoint` returns `float(...)`, registration V4:309-339;
`_fraction_bound` clamps a Python float, :350-356; no JSON round trip before
the schedule), so the exact-type check cannot misfire. `_candidate_pose`
(registration V4:539-544) repeats `_pose` (init V4:207-214) on the lossless
`tolist()` copy (V14:993, 1000) -> cannot newly fail; an earlier `_pose`
failure leaves the row without `conditional_context_support`, which
`_publish_contexts` refuses before any schedule. `_positions(refine, 20000)`
holds (refine = spatial sample of >= 36 rows, `refine_points <= 20000`). The
guard `bool(kept) == bool(supported)` (V14:827) cannot fire for `maximum >= 1`
(first ranked context is always kept) and is a defensive invariant.

### A. V15 cap derivation (V15:415-427)

`feature_count = max(1, len(features["features"]))`, `raw_rows =
len(scan["raw_points"])`, `context_cap = max(1, min(policy max,
MAX_CONTEXT_WINDOWS//features, MAX_CROP_ROW_VISITS//(features*raw_rows)))`.
Quantities are the consumer's: `scan["raw_points"]` is passed unchanged to
registration (V15:430) and to `inspect_support_cover_context_windows_v2`
(V15:490) -> `_owned_points` keeps the row count -> V1 `len(points)`;
`features["features"]` is passed as-is (V15:490), cover V2 re-instantiates the
same tuple (1..64), the V1 consumer requires all `free_circle` or refuses
whole -> `len(features)` equal. Constants imported from the consumer module
(V15:419; 4096 / 64_000_000, context_measurement_v1:28-30). `features` is a
tuple for every native status (native_v1:237 `features=()`, :737; r0 :96,
:325), so `len` never raises; `raw_rows >= 12` (`MIN_INSPECTION_ROWS`,
mesh_input_v1:283) so no division by zero. Zero features -> cap computed with
1 -> registration proceeds -> `NATIVE_FEATURES_UNAVAILABLE` at V15:485-486 as
before. Floor 1 when `f > 4096` or `f*rows > 64M`: the consumer refuses with
one context anyway (pre-existing refusal, not new). Both arms derive the same
cap. `MeshWorkflowPolicyV15.__post_init__` (V15:49-55) pops the new field and
re-validates the six V2 fields through `MeshWorkflowPolicyV1` (V1 and V2 have
identical fields, mesh V1:19-24, V2:19-24); `frozen=True` on both (V2:18,
V15:44). Runner builds it from `CONFIG["workflow"]` (V8:1400).

### B. Tests

- Synthetic schedule test (registration test :105-141), hand-verified: keys
  (p95, -coverage, index) = 0:(.9,-.9,0) 1:(.3,-.95,1) 3:(.3,-.95,3)
  4:(.6,-.99,4) 5:(.3,-.90,5) -> order [1,3,5,4,0]; actions = refine + t;
  1 kept; 3 vs 1: |(0,5,-.05)| = 5.0002 > .1 -> kept (cap 2 reached); 5 vs 1/3:
  7.07/5 -> OMITTED; 4 vs 1/3: 4.95/7.07 -> OMITTED; 0 vs 1: .05 <= .1 ->
  DUPLICATE of 1. kept = [1,3]; statuses [DUP, S, REJ, S, OMIT, OMIT];
  dispositions [SCHEDULED, SCHEDULED, OMITTED, OMITTED, DUPLICATE]; counts
  5/2; uncapped -> [1,3,4,5]; bad caps 0/49/2.0/True all refused
  (`_Unavailable` is a `ValueError`). All assertions match the source.
- Only-supported guard test (:144-149): one context, cap 1 -> SCHEDULED. ok.
- `_context_support` gate test (:152-199): the mocked query is a complete V4
  query under `_validate_query_result` (registration V4:147-220): 20 array
  keys + 8 metadata + `diagnostics` = 29 keys, 15 stat keys; strings equal
  registration V4:62-70 and surface V4:59-60; `work_units` 44r = 2r + 2r +
  8*2r + 8r + 0 + 16*r (`VERTEX_LOOKUP_WORK_UNITS` = 16); `certificate_pairs`
  8r = 4*2r; passes 2, failures 0, `tie_batches_completed` 1 > 0,
  `triangle_pairs` 2r >= 2r; `lower <= tri_lower <= upper == tri_upper`;
  guard consistency holds within one ulp (`_down/_up` slack); unit normals;
  `near_tie_counts` in [1,256]. 60 rows at upper .75 -> p95_upper >= .75 ->
  REJECTED at .5 (only `CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE`: count 60 >=
  36, coverage 1, reverse .5), SUPPORTED at 1.0; reverse .2 -> INSUFFICIENT.
  Consistent. It is a consumer test, not a registration/pose test (as its
  docstring says).
- Unmocked cap=1 registration test (:202-219): asserts 1 index, counts,
  selection label, statuses in the four-set. See P2-2.
- Plumbing test (integration :776-796): 1 mocked feature, 48 raw rows -> cap
  = min(8, 4096, 1_333_333) = 8; asserts the kwarg and the `measurement_context_cap`
  dict; policy max 2 -> kwarg 2; policy 0/49/True/2. -> `INVALID_MAX_MEASUREMENT_CONTEXTS`
  (V15:55). See P2-1.
- Runner source test (a7c20b0c...): pins `RUNNER_SOURCE in STAGE_SAMPLE_SOURCE_PATHS`
  and no other `run_gold11_*` path (:70-71), the four `CONFIG_DELTA_FROM_V7`
  keys (:66-68), `POSE_ADMISSION_DECLARATION.max_inlier_plane_p95_mm ==
  CONFIG` (:72), V7/V8 CONFIG differing only by `max_measurement_contexts`
  (:95), workflow policy max 8 (:151), bundle fields (:188-189). Closes the
  runner review's P2-1.

### C. Runner delta since the runner review (61 raw hunks in the V7->V8 diff; 55 are the review's table)

| V7 | V8 | Content | Class |
|---|---|---|---|
| 98-99 | 107-108 | `STAGE_TELEMETRY_STATUS` -> `UNCHANGED_SAMPLER_WIRING_V8_RUNNER_PATH_NAMED` | declaration (unconsumed), closes P3-5 |
| 104 | 113 | `STAGE_SAMPLE_SOURCE_PATHS` names the V8 runner | P2-1 repair; consumed at V8:1448 |
| 113-116 | 121-123 | `DRAFT_RELEASE_BLOCKERS` reworded as requirements; new comment 118-119 | unconsumed declaration |
| 118a | 126-131 | `POSE_ADMISSION_DECLARATION` | new; frozen 941, verified 970, report 1901; value 1.0 == CONFIG 202 |
| 137 | 152-167 | `CONFIG_DELTA_FROM_V7` gains `workflow_max_measurement_contexts` (v7 None, v8 8) | declared config delta |
| 167 | 197-198 | `CONFIG["workflow"]["max_measurement_contexts"]=8` | consumed 1400; frozen via `config` 942/971 and `arm_requests.common_config` 433 |
| 906, 933 | 939-941, 969-970 | bundle carries/verifies `pose_admission_declaration` | freeze/verify |
| 1859-1861 | 1897-1903 | enriched report: `pose_admission_declaration`, `result_interpretation`, `arm_comparison_scope` | report-only |

Byte delta since the reviewed V8 (+1358) is consistent with these edits alone.
No hunk touches custody, supervision, scoring, matcher, limits or any other
CONFIG value. Unintended change: none found (but see P2-3 for a declaration
the delta should have updated).

### D. Docs vs source

Protocol: "cap of at most 8 contexts" (:22-23) = CONFIG 8; "further bounded by
the frozen consumer's window/crop preflight budgets" = V15:422-423; expected
retained negative outcomes (:103-110) name `CONTEXT_WINDOW_PREFLIGHT_BUDGET`,
`FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET`, aggregate row budgets = consumer
:240-241, :290/:341/:399 and `POSE_ADMISSION_DECLARATION.expected_new_outcome_class`;
report labels (:108-110) = V8:1901-1903; interpretation boundary and arm
comparison scope match `result_interpretation`/`arm_comparison_scope` strings.
Mismatch: ":70 the three declared deltas" (P3-1). Delta design: item 4 lists
all four deltas and the "exactly one identity field" wording (closes the
runner review's P3-3); schedule section: ranking, `pose_class_distance_mm` =
0.1 (CONFIG :207), cap formula, "never below 1", both statuses, "17 features
on 581K rows -> at most 6" (64M // 9_877_000 = 6; 4096 // 17 = 240) all match
source. Neither document says the dedup is unconditional (P3-2).

### E. Projection V3 test vs script

`test_registration_implementation_is_bound_at_workflow_and_evidence_levels`
(2 locations x missing/stale/non-string -> rejected) matches V3:78-82 and
:104-109; `test_missing_registration_implementation_is_tolerated_only_before_completion`
matches the `_require(not completed, ...)` branches; array level:
`test_first_middle_last_context_binding_every_key` over `ARRAY_KEYS` including
`registration_implementation` x missing/different x 3 positions matches
V3:168-175; positive controls: `test_complete_both_successor_arms_and_schedule_cross_controls`
(8 cases, all three levels published, `verified_array_binding_count == 3`,
contact projection byte-equal to V2's inner projection) and
`test_requested_submaximal_policy_is_bound_not_replaced_by_defaults`;
`test_v3_identities_are_the_published_v15_v14_v4_strings` pins `_ARRAY_KEYS`
and the four identity strings; the literal test pins `CONTEXT_BUDGET_MAXIMA`
/`CONTEXT_DECLARATIONS` against the context V1/V2/V4 AST defaults and the
import closure (six script modules, no backend). Identity strings equal
V14:53/58, V15:33, registration V4:39, context V4. Consistent with the script.

## Findings

**P1: none found.**

- **P2-1 (test gap): V15 cap-derivation budget branches untested.**
  `tests/test_cad_guided_context_v4_engine_integration_v1.py:776-796` only
  reaches `min(8, 4096, 1_333_333)`; neither `MAX_CONTEXT_WINDOWS//features`
  nor `MAX_CROP_ROW_VISITS//(features*raw_rows)` (V15:422-423) ever binds, and
  the floor-1 branch is unexercised. Minimal repair: one parametrized case
  where `plumbing.extract` returns `features=(state.feature,)*600` and the
  test asserts `cap["value"] == 6` (= min(8, 4096//600, 64M//(600*48))) and
  `kwargs["max_measurement_contexts"] == 6`; one case monkeypatching
  `state.m.cover._v1.MAX_CROP_ROW_VISITS` (the consumer module) to 40 and
  asserting `cap["value"] == 1`.

- **P2-2 (test gap): the unmocked cap test cannot fail if the cap never bites.**
  `tests/test_cad_guided_plane_initialization_v14_registration.py:202-219`
  asserts one index and a superset of statuses; if the two-sheet fixture
  yields a single SUPPORTED context the test is the first test plus a no-op
  cap. Minimal repair: assert
  `schedule["supported_context_count"] == sum(row["conditional_context_support"].get("supported_before_schedule") is True for row in candidates) == len(schedule["dispositions"])`
  and state in the docstring how many SUPPORTED contexts the fixture yields;
  if it is one, say that the cap/dedup logic is pinned by the synthetic test
  only.

- **P2-3 (frozen declaration incomplete): `SOURCE_PROFILE.source_delta` omits the schedule/cap.**
  `scripts/run_gold11_mesh_development_v8.py:50-53` lists identity, engine,
  projection and the three numeric CONFIG changes; the module docstring :7-11
  likewise lists three. `SOURCE_PROFILE` is frozen (:939) and verified (:968)
  and will be read by POST as the delta inventory, but `CONFIG_DELTA_FROM_V7`
  (:155-167) declares four deltas including a NEW policy. Minimal repair: add
  one entry, e.g. `"WORKFLOW_MEASUREMENT_CONTEXT_SCHEDULE_POSE_CLASS_DEDUP_AND_CAP_8"`,
  to `source_delta` and one docstring line; re-hash (the source test :66-68
  already pins the four delta keys).

- **P2-4 (post-review hash drift to record before freeze).**
  `backend/cad_guided_triangle_surface_v4.py` is now f0a01f66.../37320 B
  (both earlier reviews bound f79b73b3.../37248) and
  `tests/test_cad_guided_global_registration_v4.py` is e108dc3a.../36927
  (recheck bound 1cf31b12.../36702). Surface V4:33-35 now carries the
  recheck's P3-2 wording ("V4: a wide competitor marks its rows uncertain
  instead of refusing the query") and the test :678-685 carries its P3-3
  mechanism comment plus `assert float(answer["distance_lower"][0]) == 0.`;
  the size deltas (+72 B, +225 B) are consistent with those repairs alone,
  but the old bytes are unrecoverable so "docstring/test-only" is not
  byte-verified here. `docs/research/2026-09-12_fable51_progress.md:98` still
  lists f79b73b3. Minimal repair: record both new hashes and a one-line diff
  summary in the progress ledger / review response before `--freeze`; if any
  non-docstring line of surface V4 changed, re-review that hunk.

- **P3-1 (documentation numbers).** Protocol :70 "the three declared deltas
  recorded in `CONFIG_DELTA_FROM_V7`" -> four; runner docstring :7-11 lists
  three configuration changes. Repair: "four", add the schedule/cap.

- **P3-2 (framing precision).** The pose-class dedup runs before and
  independently of the budget test (V14:813-819) and also with `maximum=None`;
  below the cap the measured set can therefore differ from a
  measure-everything publication by the rows of a <= 0.1 mm-shifted window
  margin (the cover planner would otherwise merge the near-identical windows).
  Label-free and arm-symmetric, hence not P1/P2, but neither the V14
  docstring (:772-785), the delta design (:92-106) nor the scope string says
  so. Repair: one sentence "the pose-class collapse applies regardless of the
  cap"; optionally rename `measurement_context_selection` (V14:829) to
  `measurement_context_schedule_policy` so it cannot be read as pose
  selection.

- **P3-3 (cosmetic).** Row statuses are mutated (V14:816-824) before the guard
  at :827; if the guard ever fired (it cannot for `maximum >= 1`), the
  UNAVAILABLE report would carry re-labelled rows. Repair: collect
  dispositions first, apply `support.update` after the guard.

- **P3-4 (phase attribution).** The cap block V15:415-427 runs while
  `output["phase"] == "CAD_SURFACE_SAMPLING"`; an exception there would be
  reported under the sampling phase. Repair: set an explicit phase (or move
  the block after V15:428).

- **P3-5 (declaration).** `DRAFT_RELEASE_BLOCKERS` (V8:120-125) remains
  unconsumed; the new comment (:118-119) now says so. No repair required.

- **P3-6 (test suggestion).** No projection V3 / integration positive control
  carries the new additive keys (`measurement_context_cap`,
  `measurement_context_selection`, `measurement_context_schedule`,
  `supported_before_schedule`, `duplicate_of_context_index`). I verified no
  exact-key check rejects them today (contact V2:573/583 are projection-level;
  projection V1:180, V3:118-124/168-175 are prefix/shape filters); a one-line
  positive case would pin that tolerance for future projector versions.

Closure of the runner review's items, as far as this delta shows: P2-1
(sampler path) fixed and pinned; P2-2 (projection V3 test) fixed; P2-3 (gate
adversarial evidence) partially - the `_context_support` gate test exists, the
"two near-duplicate SUPPORTED contexts -> one cover prediction" test does not
(the schedule now collapses such duplicates before measurement, which changes
that question); P2-4 (protocol outcome class) fixed; P3-3 (arm-request
wording) fixed; P3-4 (labels) addressed by `POSE_ADMISSION_DECLARATION`, the
report fields and `residual_gate_value_scope` (V14:907).

## What was not checked

- Nothing executed: no test outcome, freeze, verify, execute, geometry job
  or benchmark; the author's test-pass claims are not reproduced.
- The reviewed-V8 runner bytes and the reviewed surface-V4 /
  registration-V4-test bytes (untracked) are unavailable; the runner delta is
  inferred by subtracting the runner review's hunk table from the current
  V7->V8 diff, and the two drifted files were read only at the repair sites.
- Actual registration-report sizes versus the 96 MiB initialization-report
  cap (V8:247) and the 128 MiB workflow/result caps; the number of SUPPORTED
  contexts the two-sheet fixture yields; any physical behaviour of the cap on
  the eleven cases.
- Scientific/numerical correctness of the bound V4/V3 engine modules,
  `_context_support`'s gate semantics beyond the 1.0 mm consumer test, the
  cover planner's merging behaviour, and mesh V8 `_registration_binding`
  internals (relied on the wiring review's "digests/row selection only").
- `data/`, `output*/`, attempt/frozen directories, Gold/reference files,
  `.env`, credentials, `docs/research/2026-09-12_fable51_review_response_v1.md`
  and the runner's scoring/custody code beyond the lines cited.

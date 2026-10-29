# Independent review: V14 initializer / V15 workflow / stage policies V2 wiring (2026-09-13)

Reviewer: independent, read-only. Scope: `backend/cad_guided_plane_initialization_v14.py`,
`backend/cad_guided_mesh_offline_v15.py`, `backend/cad_guided_stage_policies_v2.py` and the three
named tests, checked against their claimed predecessors and bound V4/V2 modules. No test was run,
no geometry job executed, no network. The only Python executed was `ast`-based source extraction
(no `backend` module was imported). Scratch diffs were written only to
`/Users/idantal/.claude/jobs/1519fdc2/tmp/`.

## Decision and limits

**Decision: the wiring claims hold as stated; no P1 found.** Every alias in V14 resolves to the
V4/V2 modules; the four claimed-verbatim V12 definitions are byte-identical except for the disclosed
constructor-name substitution (plus one rename/docstring line in the configuration copy, see P3-2);
stage policies V2 is V1 text with only the import line and docstring changed; V15 is V14 text with
the initializer rebinding and an added `registration_implementation` equality gate. The unchanged
measurement consumer's string contracts are preserved by construction (identical constants in
registration V4 and stage policies V2). Findings are P2 (test gaps) and P3 (stale publication
strings / dead branch / stale comments).

Limits that bound this decision:

1. **Frozen predecessors cannot be attested against a committed baseline.** In this working tree
   `cad_guided_plane_initialization_v12.py`, `_v13.py`, `cad_guided_stage_policies_v1.py`,
   `cad_guided_mesh_offline_v14.py`, and all candidate files are **untracked** (`git status` shows
   `??`). I can only attest that their SHA-256 did not change between the start and end of this
   review (tables below), not that they equal any earlier frozen release.
2. The V4 surface/registration/context modules were checked **only for bindings, key names and
   contract strings**, not for scientific correctness (vertex-bound pruning soundness, 256-tie
   storage, wide-interval-to-uncertainty conversion, kd-partition extrema). See "What was not
   checked".
3. Test counts were verified statically (ast over `parametrize`), not by execution.

### File identities (start == end; no file changed during review)

| SHA-256 | bytes | file |
|---|---|---|
| eb1ad06a1847634585aeaa8e79779995d85f6311d17da4aea0dc96b7891b21d7 | 65110 | backend/cad_guided_plane_initialization_v14.py |
| 79ee4de537f1dddcc1dee40d33b25fe3fe8fb667bcd4e30ecc0a811ebe2d989f | 38651 | backend/cad_guided_plane_initialization_v13.py |
| 57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87 | 73393 | backend/cad_guided_plane_initialization_v12.py |
| e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25 | 26196 | backend/cad_guided_two_plane_context_v4.py |
| 9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d | 25845 | backend/cad_guided_two_plane_context_v3.py |
| f79b73b37be2684f73f7d467d5aa1074f8a27d968cdf52e9fb1d652dae02422c | 37248 | backend/cad_guided_triangle_surface_v4.py |
| 8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10 | 32098 | backend/cad_guided_triangle_surface_v3.py |
| 5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273 | 37673 | backend/cad_guided_global_registration_v4.py |
| 088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0 | 35987 | backend/cad_guided_global_registration_v3.py |
| 699b3f00fc73e311bb5c391e660f9fd04d8690d1d2c0f817c932be9dc66e612d | 3328 | backend/cad_guided_stage_policies_v2.py |
| 5c3d1252ac5f38f422f8a9a22c98a3d4db1094cd0956c928e9800f75e3ee826c | 3186 | backend/cad_guided_stage_policies_v1.py |
| 5edaae998a56c121a0aabf45fcc5b71d19e4b4aeab10209d42b89897dbd0f4f7 | 33955 | backend/cad_guided_mesh_offline_v15.py |
| 76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749 | 33435 | backend/cad_guided_mesh_offline_v14.py |
| a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9 | 39387 | backend/cad_guided_context_measurement_v1.py |
| d82cf5df6bd758a2c2fb6786707e17d3e2bd9346381d44982b41f56087d7180e | 43929 | tests/test_cad_guided_context_v4_engine_integration_v1.py |
| e75497725ed3c7d7b5f208d2d9094d5650bedc894d7c11d4a772508644af45a6 | 9786 | tests/test_cad_guided_context_v4_schedule_fixture_v1.py |
| b60896c36b56b825aa600a9f7a6064e9cb69fc93f14ae22238a82156c26aa9e3 | 3916 | tests/test_cad_guided_plane_initialization_v14_registration.py |
| 3b4220637fa19aba1e0384071f55e091705476046468327a306486078d7f7438 | 32544 | backend/cad_guided_plane_initialization_v8.py (partial read: `_prepare`, `_bound_scan`, `_registration_rows`) |
| a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724 | 38276 | backend/cad_guided_plane_initialization_v4.py (partial read: `_fail_prefilter`, `_reason`, imports, alias attribution) |
| 7c444bab2c197b0a6d9949ec6a15162286edb78bdfd081137e2907f16152c33e | 46206 | backend/cad_guided_mesh_offline_v12.py (partial read: header, function map, binder greps) |
| afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec | 18745 | backend/cad_guided_mesh_offline_v8.py (partial read: `_registration_binding`) |
| 8783f95bfce161ae4d9cd0c5a2532114efa510907b50479a42896747e20d56aa | 27567 | backend/cad_guided_support_cover_measurement_v2.py (grep only) |

## Examined bindings (table)

| Site | Binds to | Verified how | Status |
|---|---|---|---|
| V14:26 `_frozen` | `cad_guided_plane_initialization_v12` | read | ok; only attribute reuse (23 distinct `_frozen.<name>` uses, listed below) |
| V14:27 `_context_v4` | `cad_guided_two_plane_context_v4` | read; no residual `_context_v3`/`propose_two_plane_context_v3` (grep) | ok |
| V14:30 `_v3` | `cad_guided_global_registration_v4` | read; all 20 `_v3.` uses in V14 resolve here | ok |
| V14:31 `_surface` | `cad_guided_triangle_surface_v4` | read; uses at V14:144,151,390-391,462-463,895 | ok |
| V14:32 `_stages` | `cad_guided_stage_policies_v2` | read; uses at V14:737,751,753,818-819,833-834 | ok |
| V14:33 `_box` | `cad_guided_face_box_screen_v1` | same module as V12:18 | ok |
| V14:36-49 reused frozen helpers | `_old` (init V4), `_previous` (init V8), `_noisy`, `_patch`, `_search`, `_v1`, `_context`, `_pair_schedule`, `_require`, `_digest`, `_pose`, `_LIMITS`, `PlaneInitializationBudgetsV12`, `_bounded_report`, `_prepare_triplets`(=V8 `_prepare` 90-189), `_bound_scan`(V8 192-201), `_registration_rows`(V8 369-383), `_publish_contexts`(V12 1101-1120), `_two_plane_schedule_configuration`(V12 62-73), `_schedule_equation_digests`(V12 76-85), `_validate_two_plane_schedule_v2`(V12 106-202), `_bind_initialization_schedule_configuration`(V12 88-103), `_base`(V12 244-283) | grep of every `_v3./_surface./_stages./_box.` use in V12 attributed by ast to functions; none of the reused helpers touches V3 surface/registration computation. V8 helpers use no surface/registration module. V4 helpers reused (`_registration_result`, `_base`, `_fail_prefilter`, `_reason`, `_array`, `_bound_support`, `_registration_policy`) touch only `_v1.WholePartRegistrationPolicyV1` and `_patch._immutable` | ok |
| V12 `_publish_contexts` (V14:926) | compares `numerical_consumer_status == _v3.CONSUMER_COMPLETE` where `_v3` is **registration V3** (V12:16) | reg V3:30 and reg V4:38 both `"COMPLETE_V3_INTERVAL_SUPPORT_POLICY"`; V4 `_diagnose` emits it (reg V4:533) | ok: string-only reach into frozen V3; no V3 computation |
| V12 `_base` (V14:173) | `numerical_support_policy=_stages.PROXIMITY` with **stage V1** | V1/V2 diff: only lines 1-5 (docstring) and the import line differ; `PROXIMITY` identical | ok |
| V14:53 `REGISTRATION_IMPLEMENTATION` | reg V4:39 `"SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4"` | read | ok |
| V14:54-58 `_CONTEXT_DECLARATIONS` (9 keys, adds `scan_extrema_partition`) | context V4 `_base()` 66-78 | all 9 keys present; `scheduling_capacity`/`production_use_allowed` inherited from `_v2._base()` as in V13 | ok |
| V14:83 `_context_v4._MAXIMA`; V14:68-69 `TwoPlaneContextBudgetsV4`/`_configuration` | context V4:35-63 | read | ok |
| V14:285 `propose_two_plane_context_v4` | context V4:362 | signature unchanged vs V3 (diff) | ok |
| V14:144/151 `_surface._budgets` → `TriangleSurfaceBudgetsV4` | surface V4:71,136-147 | type-checked; a V3 `TriangleSurfaceBudgets` instance is refused, no fallback | ok |
| V14:465/471 `_v3._diagnostic_snapshot`, `_v3._validate_query_result` | reg V4:120-144 (STAT_KEYS now include `interval_width_uncertain_pairs`, `vertex_lookup_rows`; work formula adds `16*vertex_lookup_rows`) | surface V4 emits exactly those keys (surface V4:432-433) | ok |
| V14:613/615/639, 740-741 `_v3._unambiguous/_support_masks/_fraction_bound/_metric_bounds` | reg V4:273-277, 262-270, 350-356, 328-339 | unchanged text vs V3 (diff) | ok |
| V14:849-851 `SurfaceRegistrationBudgetsV4`, `_validated_execution_budgets` | reg V4:74, 104-110; `_EXECUTION_MAXIMA` unchanged | read | ok |
| V14:898-928 `_v3._Queries/_norm/_refine/_diagnose/_SurfaceFailure/_select` | reg V4:223-259 and unchanged remainder | diff shows no signature change | ok |
| V14 evidence strings (V14:817,821,822) | `DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION`, `MATERIALIZED_..._SURROGATE...`, `CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL` | literals; equal to reg V4:35-37 and reg V3:27-29 | ok |
| Measurement consumer `cad_guided_context_measurement_v1.py` | imports stage V1 (:21) and reg V3 (:22) for constants; contract at :191-223 (`schema_version`, `status`, `stage_policies == V1.ablation(connected=True)`, `numerical_consumer_status == V3.CONSUMER_COMPLETE`, `SUPPORT_POLICY`, `DISTANCE_MODEL`, `SURROGATE`, `proximity_policy == V1.PROXIMITY`) | every constant identical in V4/V2; no implementation-label check exists there | accepts V14 evidence |
| `cad_guided_support_cover_measurement_v2.py` | delegates to the V1 consumer (:17, :146); no label check | grep | ok |
| V15:18-20 `_v12`, `_initialize_v14`, `_stages` (V2, replacing `_v12._stages` V1) | read | strings identical | ok |
| V15:22-30 reused mesh V12/V8 binders | `_v2`=mesh V2; `_previous`=mesh V8 (`_registration_binding` 64-104: digests/row selection only); `_scan_plane_binding` (mesh V12:71-148), `_scan_plane_configuration_snapshot` (54-68), `_two_plane_schedule_binding` (183-246 → uses plane_initialization_v12 `_validate_two_plane_schedule_v2`/`_search`/`_pair_schedule`, i.e. the same helpers V14 reuses) | ast map + greps; none reaches V3 surface/registration or stage V1 | ok |
| V15:33 `REGISTRATION_IMPLEMENTATION` | `_initialize_v14.REGISTRATION_IMPLEMENTATION` | read | ok |
| V15:68-73 `_registration_binding` gates | two `_bind_context_publication` calls (evidence + initialization) then `registration_implementation == REGISTRATION_IMPLEMENTATION` | read | ok |
| V15:112-126 consumer contract check | `numerical_support_policy == _stages.PROXIMITY` (V2) vs report value set by V12 `_base` (V1) — identical; `stage_policies == V2.ablation(...)` — identical dict | read | ok |
| V15:323-333 context-mode refusal ordering | `initialize._configuration` (→ `_context_configuration` requires `mode == "v4"`, V14:67) runs before `prepare_mesh_scan_v1` (:362) and before `NATIVE_IMPORT` (:381); exception → `INPUT_PREPARATION_EXCEPTION:*` (:377-378) | read | ok |
| V15:343-345 registration budgets | `SurfaceRegistrationBudgetsV4` only | read | ok |
| V15:407 | `register_measurement_contexts_v14` | read | ok |
| V15:452-461 binding failure | whole invalidation (`measurement_context_indices=[]`, `attempted_...` retained, `registration_array_binding=[]`, `owned_surface_roles=None`) then re-raise → `WORKFLOW_EXCEPTION:*` | read; matches V14 text (diff) | ok |
| Stage V2:15 `_v3` | `cad_guided_global_registration_v4` | diff vs V1: only docstring (1-5) and import line | ok; `ablation_stage_policies_v1(connected=False)` reads `_v3.SUPPORT_POLICY`, identical string |

Distinct `_frozen.<name>` uses in V14 (each once): `PlaneInitializationBudgetsV12, _LIMITS, _base, _bind_initialization_schedule_configuration, _bound_scan, _bounded_report, _context, _digest, _noisy, _old, _pair_schedule, _patch, _pose, _prepare_triplets, _previous, _publish_contexts, _registration_rows, _require, _schedule_equation_digests, _search, _two_plane_schedule_configuration, _v1, _validate_two_plane_schedule_v2`. Notably absent: `_frozen._v3`, `_frozen._surface`, `_frozen._stages`, `_frozen._prefilter`, `_frozen._configuration`, `_frozen._context_support`, `_frozen._FaceQueries`.

### Failure boundaries (checked against predecessor text)

- Late fatal finite query in prefilter: `_prefilter` populates `result["start_groups"]` only after the whole scheduled loop (V14:659-662); on any exception `_initialize` (V14:712-725) calls V4 `_fail_prefilter` (clears `start_groups`, `usable_initialization=False` on every hypothesis row, status `UNAVAILABLE`) and then nulls every `all_passing_start_groups` pose with `INVALIDATED_BY_FATAL_INITIALIZATION_FAILURE`. Text identical to V13:345-358. Covered by `test_actual_initializer_whole_refusal_keeps_no_usable_start_prefix[late_finite_query]` (integration :367-391).
- Registration: `_SurfaceFailure` re-raised (V14:918-920) → outer except (V14:929-941) sets `UNAVAILABLE`, `NOT_ATTEMPTED_AFTER_FATAL_FAILURE` / `INVALIDATED_BY_FATAL_REGISTRATION_FAILURE`, clears `measurement_context_indices` and retains `attempted_measurement_context_indices`. Identical to V13 (diff hunks 417a812, 454c849, 456c851 are the only changes inside the function).
- Binding mismatch: `_bind_context_publication` (V14:98-111) raises on label/config/sha mismatch; `_registration_binding` (V15:68-73) raises on missing/wrong registration label; workflow invalidates the whole registration (V15:452-461).
- Whole refusal on wrong budget types: surface V4 `_budgets` and registration V4 `_validated_execution_budgets` are `type(...) is` checks; V3-typed budgets are refused, never coerced.

### Threshold / policy values

No numeric threshold, policy value or default inside V14, V15 or stage V2 differs from V13/V14/V1 respectively (diffs are identity strings, bindings and the one added equality gate). Values that **do** change are inside the bound V4 modules and are declared there, not silent: surface V4 default `max_ties_per_query` 32→256 (surface V4:56,83; becomes the V14 default when `surface_budgets=None`, and is published in the report's `surface_budgets`), context V4 default/maximum `max_extrema_coordinate_visits` 256M→512M (context V4:39,44; published in `two_plane_context_configuration.budgets`), and registration V4 no longer requires `interval_width_rejections == 0` for completeness (reg V4:138-143) because surface V4 marks rows uncertain instead of refusing (surface V4:567-574). These are V4-module design changes outside this review's correctness scope (see "What was not checked").

## Findings

**No P1 found.**

- **P2-1 (test gap): wrong registration label is not exercised.** `tests/test_cad_guided_context_v4_engine_integration_v1.py:750-757` only `pop`s `registration_implementation`. The gate at `backend/cad_guided_mesh_offline_v15.py:72-73` is an equality, so a stale/foreign label (e.g. a hypothetical V3 consumer label) would be refused, but nothing pins that. Minimal repair: parametrize `damage` with `registration["evidence"]["registration_implementation"] = "SURFACE_V3_NOT_V4"` (and with a non-string) and assert the same refusal.

- **P2-2 (test gap): verbatim assertion omits the configuration copy.** `integration :184-188` checks `_prefilter`, `_context_support`, `_FaceQueries` only; `_frozen_configuration_v4` (`V14:114-151`) is the fourth claimed copy and is unchecked. Minimal repair: add a mapped case `("_configuration", "_frozen_configuration_v4")` and compare after stripping the `def` line and the added docstring line (or compare `ast.dump` of bodies minus the docstring).

- **P3-1 (documentation / published strings): V14 initialization reports still declare V3 surface semantics.** Reused `V12 _base()` (via `V14:173`) publishes `work_unit_formula="BOX_SCREEN_ROWS_PLUS_UNCHANGED_V3_QUERY_WORK"` (V12:258) and `counter_semantics.completed_query_rows_and_calls="VALIDATED_COMPLETE_V3_SURFACE_OUTPUTS"` (V12:263); the verbatim `_FaceQueries` publishes `box_binding.scope="OWNED_COMPLETE_V3_FACE_INDEX_ROOT_ALL_TRIANGLE_VERTICES"` (V14:405). Under surface V4 the query work is not "unchanged V3 work" (it adds `16*vertex_lookup_rows`, reg V4:129-131). The prefilter-only API (`prefilter_plane_initializations_v14`) carries no surface/registration implementation label at all; only `initialization_implementation` disambiguates. Retained consumer strings (`SCHEDULED_FOR_V3_REFINEMENT` at V14:661 ↔ V15:188; `COMPLETE_V3_INTERVAL_SUPPORT_POLICY`) must stay. Minimal repair: in V14 `_base()` (V14:172-175) add an additive field, e.g. `result["finite_surface_implementation"] = REGISTRATION_IMPLEMENTATION` (or a dedicated surface V4 label), leaving the retained strings untouched.

- **P3-2 (documentation): verbatim claim is slightly imprecise for the configuration copy.** Besides the constructor name, `_frozen_configuration_v4` differs from V12 `_configuration` by the `def` name and one added docstring line (`V14:118`). Harmless; state it in the V14 docstring (V14:7-9) for exactness.

- **P3-3 (dead branch): `two_plane_enabled=False` path unreachable.** `_two_plane_records` V14:209-216 (`DISABLED_BY_EXPLICIT_POLICY`) cannot run because `_configuration` V14:161 requires `two_plane_enabled is True`. Inherited from V13:149-156. No repair required; optionally replace with `_require(configuration["two_plane_enabled"] is True, ...)` to make the invariant explicit.

- **P3-4 (stale comments/names):** `V15:50-51` comment still says "V3 context capacity ... handles V3" (it is V4 now); test name `test_empty_pair_schedule_does_not_bypass_mandatory_v3_publication` (integration :625) tests V4 publication; `V14:36` binds `_previous` which is unused in V14. Cosmetic.

- **P3-5 (test-report claim check):** static ast count of parametrized cases = 96 (integration) + 5 (schedule fixture) = 101, plus 1 (registration) = 102, consistent with the author's "101 + 1". Not executed here.

## Verbatim-copy verification

Method: `ast`-extracted source of each top-level definition from V12 and V14 written to the scratch dir and compared with `diff`.

| V12 definition | V14 definition | diff result |
|---|---|---|
| `_FaceQueries` (V12:515-615) | `_FaceQueries` (V14:374-474) | 2 lines differ, both `_surface.TriangleSurfaceBudgets(` → `_surface.TriangleSurfaceBudgetsV4(` (V14:391, 463) |
| `_prefilter` (V12:618-819) | `_prefilter` (V14:477-678) | byte-identical (diff exit 0) |
| `_context_support` (V12:1074-1098) | `_context_support` (V14:730-754) | byte-identical (diff exit 0) |
| `_configuration` (V12:205-241) | `_frozen_configuration_v4` (V14:114-151) | 3 differences: `def` name; added docstring line V14:118; `TriangleSurfaceBudgets(` → `TriangleSurfaceBudgetsV4(` at V14:151 |

V13 → V14 whole-file diff (34 hunks) contains only: docstring; alias rebinding (V14:27-33, 36-37, 44-46); `REGISTRATION_IMPLEMENTATION` (V14:53); `_CONTEXT_DECLARATIONS` gains `scan_extrema_partition`; V3→V4 identity/mode/error strings in the context helpers; the four inserted definitions above; `_frozen._configuration` → `_frozen_configuration_v4` (V14:162); `evidence["registration_implementation"]` (V14:812); `SurfaceRegistrationBudgetsV3` → `V4` (V14:849, 851); function renames. No other line changed.

Stage policies V1 → V2 diff: line 1 docstring, lines 3-5 added docstring, line 12/15 import `cad_guided_global_registration_v3` → `_v4`. All policy strings, dataclass, and functions identical.

Mesh offline V14 → V15 diff: docstring; `_initialize_v13` → `_initialize_v14`; `_stages` now imported from stage V2 (was `_v12._stages` = V1); identity strings V14/V3 → V15/V4; `SUCCESSOR_ARM_MECHANISMS` context field `"v3"` → `"v4"`; `MeshWorkflowPolicyV14` → `V15`; added `REGISTRATION_IMPLEMENTATION` gate (V15:72-73) and field (V15:230, 280); `SurfaceRegistrationBudgetsV3` → `V4`; `register_measurement_contexts_v13` → `_v14`; `two_plane_context_mode="v3"` → `"v4"`; refusal-string version numbers. Nothing else.

## What was not checked

- No tests, geometry, or module imports were executed; the "101 + 1 passing" figure is the author's, only its case count was reproduced statically.
- Scientific/numerical correctness of the bound V4 modules (`cad_guided_triangle_surface_v4.py`: nearest-vertex pruning soundness, 256-tie storage, wide-interval-to-uncertainty; `cad_guided_global_registration_v4.py`: relaxed completeness predicate; `cad_guided_two_plane_context_v4.py` / `cad_guided_raw_extrema_index_v3.py`) — only their bindings, key names and contract strings were checked. In particular I did not evaluate whether prefilter coverage (distance-only `definite_support`, V14:639-641) should exclude rows newly marked uncertain by surface V4.
- Whether V12/V13/stage V1/mesh V14 equal any committed frozen baseline (all untracked here; only intra-review hash stability attested).
- Modules only reached transitively and not read: `cad_guided_face_box_screen_v1`, `cad_guided_two_plane_context_v1/v2`, `cad_guided_two_plane_schedule_v1/v2`, `cad_guided_report_envelope_v6`, `cad_guided_scan_plane_preparation_v1`, `cad_guided_context_surface_roles_v1`, `cad_guided_composite_surface_roles_v1`, `cad_guided_global_registration_v1/v2`, `cad_guided_mesh_offline_v2`, `cad_guided_plane_search_v6`, `cad_guided_noisy_mesh_planes_v5/v7`.
- Other callers/runners that may still route to V13/V14-mesh, and any runner-level policy values (policy values are caller-supplied by design).
- The registration test's "2 mm axial ambiguity" tolerance (`test_cad_guided_plane_initialization_v14_registration.py:74-77`) as a physical/registration claim.
- Anything under `data/`, `output/`, `outputs/`, `docs/research/*attempt*`, Gold/reference files, `.env`, credentials, or temp caches (not opened, by rule).

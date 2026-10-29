# V10 candidate — source closure checkpoint for Codex's bounded closure review — 2026-09-13 15:57 UTC

Author: Claude Code (`claude-fable-5-1`, effort max), sole engine/test/runner editor under the
standing consumed-Gold11 development authority. This is a source-closure checkpoint with actual
executed test receipts and stable hashes. It is NOT a freeze, PRE, launch authority or physical
result; no physical data, attempt artifact, control, opaque cache or V9 frozen source was read or
touched. V9 and every earlier attempt remain terminal.

## Final executed receipt (the one to review against)

Command (inert namespace, no conftest, no plugins, single numerical thread):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
.venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider \
  tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py \
  tests/test_cad_guided_global_registration_v5.py \
  tests/test_cad_guided_plane_initialization_v16_registration.py \
  tests/test_cad_guided_plane_initialization_v16_completion.py \
  tests/test_cad_guided_plane_initialization_v16_allowance.py \
  tests/test_cad_guided_context_measurement_v3.py \
  tests/test_cad_guided_support_cover_measurement_v4.py \
  tests/test_gold11_contact_scoring_v4.py \
  tests/test_gold11_schedule_projection_v5.py \
  tests/test_cad_guided_v10_connected_chain_v1.py \
  tests/test_cad_guided_completion_v1_engine_integration_v17.py \
  tests/test_gold11_mesh_development_v10_source.py \
  tests/test_gold11_v10_execute_waiter.py -q
```

Result: **885 passed, 1 skipped, 0 failed; pytest exit 0**; started 2026-09-13T15:52:07Z, finished
2026-09-13T15:55:11Z (183.62 s). The one skip is the pre-existing V17 integration skip inherited from
the V16 port (unchanged). Output retained in the session job directory; the counts above are the
actual pytest summary line, and the exit code is the process exit, not a pipeline head.

## Stable hashes at that receipt (SHA256)

| File | SHA256 |
| --- | --- |
| backend/cad_guided_native_feature_bridge_v1.py | fef3980a5d5d4594b48dacaee763451c798d081044b4cf47630f5d7cd6126fea |
| backend/cad_guided_mesh_offline_v17.py | 5c4fa6b8f3fb97a23398b9fb00cb418f2f47f0477b93693c6a3c0c4cfd8a5a63 |
| backend/cad_guided_global_registration_v5.py | ee05a03eb564c9fcc8c3cee631e4f429d0c66d9ec5468bc932377368105750e7 |
| backend/cad_guided_plane_initialization_v16.py | 7dad12ae569c2044681864bb136105e0fea8750a4c1e454797a9f9fd874cb63b |
| backend/cad_guided_context_measurement_v3.py | e67d1c35cc66a48ddd0ee521ca4c4f23c08f78606751554c17e8b4cdfd042891 |
| backend/cad_guided_support_cover_measurement_v4.py | f6972fd538d27b24d2142a453b56b3846be34e2dc5d79ab30f54d198ce960230 |
| scripts/gold11_contact_scoring_v4.py | c0bdfadadec64447acb3db6e6509264c53980e8f9b02a130eb22311b8aceba10 |
| scripts/gold11_schedule_projection_v5.py | 996fc87cb85f659d9d604bddb359497abd7bade5306f178703dbac0052499179 |
| scripts/run_gold11_mesh_development_v10.py | f7fd083f8b74a22e7fd1afe872ae8b9b2bfbc94f4b7024fc8c2b46494208da92 |
| scripts/gold11_v10_execute_waiter.py | e7ade783ea5adae1d2a97f9d1696913dd614120209a8b875d97147ed6b1282bc |
| docs/research/2026-09-13_gold11_mesh_development_v10_delta_design.md | 0a68d42ac50264cff41189b67960a6f42d4d282c761c227bee11989b2e23d23c |
| docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md | e59aff47a8312bfce44139a0a1977a33dff4a0a86039502b8c34fe5f1f27889b |
| tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement.py | e251ed01325b028d348610e3220e2b66fae8300721283f2f9b9c3a45fa94ffa1 |
| tests/test_cad_guided_global_registration_v5.py | 14dcb8bc29765bc9687e12930dd120a288d475fc1b88d0c0b1daeeb876095bb0 |
| tests/test_cad_guided_plane_initialization_v16_registration.py | a3c950a6188cf799fb1945525590b1358cdf148cf27edf895cd2a18eb3a00126 |
| tests/test_cad_guided_plane_initialization_v16_completion.py | 699194ebf6b20d6433f6eb4bd1096f7044a72db8d3f60b76bced774dc9b3f3a8 |
| tests/test_cad_guided_plane_initialization_v16_allowance.py | cc34fc7f9f63b18ba54b596edba7147985327efc199520fa6cd9562471b96e65 |
| tests/test_cad_guided_context_measurement_v3.py | d5a81f27cea85b8a720d6b8c8a38a5c51f1417fa3c8390e9395bc803dfa6e306 |
| tests/test_cad_guided_support_cover_measurement_v4.py | 062ce6e276ee87b6c2ec0e69282461892643eceb5dce04af30e5031f5b2ced6e |
| tests/test_gold11_contact_scoring_v4.py | a40fbbfc377d9627ecd8a397a52476496f1f6e3b38ffa9c18b74ab3784cbee68 |
| tests/test_gold11_schedule_projection_v5.py | d4985a5b1b161aec4e661e459a00d9e2a5833dd6e4629ccf5a35dd0615cd7416 |
| tests/test_cad_guided_v10_connected_chain_v1.py | 99a56ab47e82f6b863a73b517f104173546c62c74fec2c79d422f8a6c9b9a62e |
| tests/test_cad_guided_completion_v1_engine_integration_v17.py | 5f2862f31f2a8d66547ecb4a8ca6e1a279fc0cb266da86468cf34b390310633c |
| tests/test_gold11_mesh_development_v10_source.py | 7116db3500a9541b133c04dd9e3f57ae7e188060bd466df2e3b3160d87e6e37e |
| tests/test_gold11_v10_execute_waiter.py | 0df6e722723441c0b4b13e795a51479cc1cae50bfd31f8b3a988856e60649cf9 |

Files whose hashes changed after the 15:47 UTC (883-pass) receipt and why: registration V5
(row-quota scope sentence: enumeration/binding/index builds charge no rows; docstring only),
runner V10 (whole-cover-before-windows bridge provenance wording; `allowance_interaction`
declaration string), waiter V10 (overlap regex `_gold11_v\d+_exact_scoring`), waiter test (+1
regex regression), allowance test (+2 guard-expiry-between-chunks tests), design/protocol docs
(D1 allowance-check timing inside the authorized worker, D2 unchanged metric-aggregation contract
wording, whole-cover refusal wording, row-quota scope). No engine logic changed after 15:43 UTC.

## Receipt history (all actual runs; failures included)

| UTC | Scope | Outcome |
| --- | --- | --- |
| 14:35 | bridge suite (B1 delta) | 13 passed |
| 14:50 | 11 suites (bridge, V5, V16 ×3, context V3, cover V4, contact V4, projection V5, chain, V17) | 837 passed, 1 skipped |
| 15:18–15:20 | 11 suites | 862 passed, 1 skipped, **1 FAILED** (`test_independent_fixture_and_frozen_numeric_aliases`: a between-build hook line had been inserted into the V15-verbatim base `_FaceQueries` text) |
| 15:24–15:26 | alias + phase-stop/chunk/accounting/rollback tests after moving the between-build stop into `_FaceQueriesV16` | 6 passed |
| 15:33–15:36 | 13 suites (+ runner source, waiter) | 879 passed, 1 skipped |
| 15:43–15:46 | 13 suites (+ B3 ordering fix, seam test, chunk tests, forged-whole-flag fix) | 882 passed, 1 skipped, exit 0 |
| 15:47–15:50 | 13 suites (+ wording corrections, waiter regex) | 883 passed, 1 skipped, exit 0 |
| 15:52–15:55 | 13 suites (+ guard-expiry-between-chunks tests) | **885 passed, 1 skipped, exit 0** |

## Closure of every Codex finding, with the executed test that proves it

| Finding (reviewer) | Source closure | Executed test(s) in the final receipt |
| --- | --- | --- |
| B1 P1 cap 16 vs the 64-feature cover boundary | bridge cap 64 (cover/context grammar) | `test_bridge_limit_is_the_cover_and_context_boundary_of_64_not_the_kernel_entrypoint_cap`, refusal matrix, both real factory branches through real cover V4 / context V3 / contact V4, real V17 dispatch (13 tests) |
| B1 fixture truth limits | rows labelled factory-conditioned transport; independent constructor-frame check; 1 mm-skin kernel refusal retained | `test_independently_constructed_geometry_confirms_the_extractor_frame_through_the_real_chain[v1,r0]`, `test_one_mm_skin_remains_a_named_kernel_refusal` |
| B4 design (strict flag, deterministic quotas, typed expiry, ledger schema) | initializer V16 + registration V5 | `test_active_allowance_reproduces_the_v15_supported_context_with_full_ledger_receipts_reuse_and_checkpoints`, `test_tiny_main_row_quota_expires_the_first_candidate_deterministically_and_omits_the_rest`, `test_main_group_quota_bounds_the_refinement_schedule_deterministically`, `test_allowance_validation_and_deterministic_quota_semantics`, `test_cpu_guard_is_absolute_process_cpu_and_declared_timing_dependent` |
| B4 P1 rollback left duplicate-main provenance | pre-transaction main snapshot restored verbatim | `test_completion_rollback_restores_duplicate_main_provenance_and_the_real_v17_binder_accepts_the_retained_main` (real `register_measurement_contexts_v16`, byte-equivalent groups vs a lane-disabled run, real V17 binder accepts; dangling pre-fix provenance refused) |
| B4 P2 attempted/completed accounting in completion queries | charge before the primitive, complete after COMPLETE | `test_completion_face_queries_charge_attempted_rows_before_work_and_complete_only_after_success`, `test_a_completion_support_longer_than_the_cooperative_chunk_is_screened_and_queried_in_capped_chunks` (1500 rows → [1024, 476]) |
| B4 P2 guard coverage / cooperative chunk | phase-boundary stops, pre-build first-query check, between-build stop, chunk cap | `test_completion_lane_typed_stops_before_preparation_at_zero_row_allocation_and_between_face_builds`, `test_completion_lane_plans_and_issues_every_query_within_the_cooperative_chunk`, `test_guard_expiry_between_chunks_of_a_long_support_stops_before_the_next_primitive_with_exact_counts[first_screen_chunk,first_query_chunk]` (no next primitive, exact counts, never PASSED, rollback marks it EXPIRED_BY_ALLOWANCE_UNKNOWN), `test_completion_lane_skips_without_allocation_and_rolls_back_only_on_typed_expiry` |
| B4/B2 P2 shallow checkpoint snapshot | deep-owned reconciled snapshots | `test_checkpoint_snapshots_are_owned_and_reconciled_at_each_publication_depth`, `test_checkpoint_ledger_is_bounded_and_declares_truncation` |
| B4 P2 fatal attempted candidate reported completed | FAILED disposition (surface and unclassified) | `test_fatal_failure_after_a_completed_candidate_is_failed_not_completed_and_later_candidates_stay_not_attempted[surface,unclassified]` |
| B4 acceptance: conditional/skip-based core tests | explicit fixtures, checkpoint-controlled clock | `test_first_candidate_supported_then_cpu_guard_omits_the_second_with_exact_ledger_and_real_v17_binding` (+ the fatal test above) |
| Root: subset route only with handcrafted evidence | actual initializer subset through the real V17 binder | same test: real V16 output, real `_registration_binding`, omitted candidate and bare claim refused |
| Root: forged whole flag labelled STRICT_WHOLE_SET before rejection | helper refuses a strict claim contradicted by new disposition evidence; legacy no-ledger meaning kept | same test (helper False, legacy True, real binder `INCOMPLETE_INITIALIZATION_CANNOT_REGISTER`); context V3 `strict_claim_contradicted_by_ledger` (helper) and `strict_claim_with_omitted_row` (full consumer denominator) |
| B3 P1 corrupted retained query recomputed | intrinsic integrity first, fatal `_SurfaceFailure`; index-bound validation after a full key match | `test_diagnosis_query_is_reused_exactly_for_context_support_and_any_changed_binding_requeries` (stale status, changed array, changed shape, changed diagnostics, changed rows, foreign record → fatal with no fresh query/reuse increment; genuinely different index and identical rebuild → requery; write re-enable raises) |
| B3 P2 immutability / shape / byte identity | bytes-backed read-only views, JSON-text metadata, shape in digest | same test; `test_chunked_queries_are_byte_identical_to_whole_queries_and_charge_the_same_rows` (literal `tobytes`) |
| B3 attempted vs completed in V5 | charge before, complete after | `test_attempted_rows_stay_charged_when_the_primitive_fails_and_completed_rows_do_not` |
| Cost postmortem: false completion budget prose; duplicate query | corrected declaration (values unchanged); exact reuse | runner source test `test_basis_identities_and_declared_config_deltas`, `test_unchanged_v9_values_outside_the_declared_deltas` |
| Runner declarations D1–D3 | design/protocol wording; waiter regex `\d+` | `test_overlap_pattern_matches_the_two_digit_v10_private_scoring_namespace`; docs above |
| Runner → V16 seam | constructors from CONFIG reach the registration entry through the real V17 workflow | `test_runner_v10_constructor_policies_reach_the_v16_registration_seam_with_the_allowance_and_checkpoint_printer`, `test_real_constructor_bindings_for_both_arms`, `test_inert_freeze_and_verify_round_trip_bind_the_v9_basis`, `test_stage_checkpoint_printer_is_bounded_and_never_raises` |
| Consumer cascade (context V3 → cover V4 → contact V4 → projection V5) | subset contract validated by every consumer; int-count diagnostic | context V3 (subset acceptance/fault matrix), cover V4 suite, contact V4 suite, projection V5 suite, chain `test_explicit_subset_registration_contract_is_measured_by_cover_v4_and_scored_while_context_v2_and_a_bare_claim_refuse` |

## What this checkpoint does not claim

No physical execution, no physical F1, no speedup measurement, no calibrated accuracy, no comparison
with the historical 0.4580152672. The synthetic scenes (context-V4 engine scene, completion scene,
plumbing doubles) qualify contracts and stop plumbing, not physical coverage. CPU-guard stops are
timing dependent by declaration; the realized trace is published by the engine, not predicted here.

## Requested next step

One bounded closure review of exactly these hashes. On acceptance: exclusive `--freeze` of the V10
runner (basis V9 4567a0f5…), independent non-author exact physical PRE plus archive, fresh no-overlap
check, ONE waiter-launched execute (`.venv/bin/python scripts/gold11_v10_execute_waiter.py`),
ps-only monitoring, observed outer exit, independent POST, honest result narrative (null stays null;
a low numeric score is a baseline with its dominant failures reported).

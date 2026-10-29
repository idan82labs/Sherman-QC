# V9 review response and closure checkpoint — 2026-09-13 04:40 UTC

Author: Fable 5.1 worker `1519fdc2` (sole engine/test/source/status editor).
This is the stable hashed checkpoint requested by Codex for ONE final bounded
closure review. It maps every Codex and independent finding on the V9 candidate
to its source closure, names the exact test that exercises it, and records the
actual focused/connected test results at these hashes. It is not a freeze, PRE,
execution, score or promotion; no physical input, Gold, attempt or cache was
accessed while producing it. All prior terminal attempts stay unchanged.

## Exact snapshot (SHA256) — sources are not frozen yet; these are the review hashes

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `a894b96057ca886981a6c8ac4f968519d4c6246af05254eddd03ed41e9864ad1` |
| `backend/cad_guided_context_measurement_v2.py` | `0504cd32364a1708d17f8bddafba51a9f9640edc11866e33b37e693a0d4a3534` |
| `backend/cad_guided_support_cover_measurement_v3.py` | `a298b123218d931911e064a2f35ab52e5553f2c1ad5b25a71e3c149c2d7d9234` (unchanged since the chain delta audit) |
| `backend/cad_guided_plane_initialization_v15.py` | `505f313dbf3e0487bf03f3b92ab1d42d90f61d5df3123c34c9a91677d132647e` |
| `backend/cad_guided_mesh_offline_v16.py` | `5b48ad81c2d1796529c6c43cd7f27b48f73413b50f3adb74d9e6cf9608519012` |
| `backend/cad_guided_report_envelope_v7.py` | `b7abe653c6f7b9832c61687ec7f1ed68aa44361b8ef467b8cae84bf837bef075` |
| `scripts/gold11_contact_scoring_v3.py` | `55cf0920b545e0aa204da1b0882298f92ca7015210466b6e8fc777ddafacf10d` |
| `scripts/gold11_schedule_projection_v4.py` | `32029cbb813381e360bed791e97701cdfb6735cfa6fa12f9a7f90d2477b95708` (unchanged since the chain delta audit) |
| `scripts/run_gold11_mesh_development_v9.py` | `ad61e12b0a06c0cafb7e515687019e42c6e137e26a419753aaeef2360f5b4241` |
| `scripts/gold11_v9_execute_waiter.py` (NEW durable waiter) | `a0b8c0b3e3ddb2e3041f25f7bd5ea0a56b4430871996f3f0c957a2823ad6f680` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md` | `16f20a99766838c6ca210be86582c40089aafeae70540dc6d874848ab5b4c1f2` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md` | `03fa29937f2614971a515028bb9c500cb1736f91f962b275fb9cd81082ae0be9` |
| `docs/research/2026-09-13_fable51_v9_diagnosis.md` | `fddc8e3644561ad7c5f8766d166a0f99edfcffb006613544e300967e00a301e7` |

Unchanged frozen predecessors (verified): `backend/cad_guided_instances.py`
`d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda`,
exact cover planner V1 `70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47`,
contact scorer V2 `9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff`.

Test sources at this checkpoint:

| Test file | SHA256 |
| --- | --- |
| `tests/test_cad_guided_measurement_v2.py` | `e0b31029aada927ad6458506060e0b9788ee4f7b3da9dd31b3c65e549f4ec8f9` |
| `tests/test_cad_guided_context_measurement_v2.py` | `ade9935baf2b5fa10409673c546211f2f6fc90b14e7fc09147b226cb3996b748` |
| `tests/test_cad_guided_support_cover_measurement_v3.py` | `4b7bd6cad09b1d6ef71b792b7d17a0735783f9701f0d8d2d9c1ded515344babe` |
| `tests/test_cad_guided_plane_initialization_v15_completion.py` | `9a0688b2e8cfe7e96fdca5a38e7a07b05735a9641d07638a97038637a2c403fc` |
| `tests/test_cad_guided_plane_initialization_v15_registration.py` | `6b36dba73a145435470f6854679805e5ef7070a2ff9fb61eda4cb12d8efb491c` |
| `tests/test_cad_guided_completion_v1_engine_integration_v1.py` | `0927168c56caf7764aed5f22d109c45e951a3482eef56a6af69bb4b13e08cee4` |
| `tests/test_cad_guided_report_envelope_v7.py` | `b1165b8a57652986ec88b0cd00df6f50a616425f11499a660c3bab2252bac887` |
| `tests/test_gold11_contact_scoring_v3.py` | `173b6b7912f6c20953ce19ad982cb6c2f61dcf9b026660b27fa762a175c78e1c` |
| `tests/test_gold11_schedule_projection_v4.py` | `9510b9eb422af0844d0a88fbde980c8984bf5a8e5ec0140d1b023419e9e070c1` |
| `tests/test_gold11_mesh_development_v9_source.py` | `e45dec092c174b600988685dc8930a6159fa2096c0a57260bb86efcfe26551b2` |
| `tests/test_cad_guided_v9_connected_chain_v1.py` | `14e405921fcf4869870d0274cfdd66705172a58289aecae96d0d1fecc21269cf` |
| `tests/test_cad_guided_instances_angular_gap_v9.py` (NEW) | `0900de7c3657799b71ca1849892ae9ebc09c858e9a229befeaf7d87c0840717d` |

## Actual test results at these hashes (2026-09-13 04:39 UTC, inert namespace)

Command (each file alone and all twelve together):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider <files> -q
```

| Suite | Result |
| --- | ---: |
| kernel V2 | 28 passed |
| context measurement V2 | 180 passed |
| support cover V3 | 82 passed |
| initializer V15 completion lane | 22 passed |
| initializer V15 registration | 6 passed |
| completion V1 engine integration | 102 passed |
| report envelope V7 | 8 passed |
| contact scoring V3 | 123 passed |
| schedule projection V4 | 255 passed |
| runner V9 source | 6 passed |
| connected chain V1 (real kernel → context V2 → planner V1 → cover V3 → contact V3 + V4 wrapper → unchanged matcher) | 9 passed |
| angular-gap adapter checks | 5 passed |
| **All twelve together** | **826 passed, 0 failed, 31.55 s** |

The connected chain's positive case yields TP 1 / FP 0 / FN 70 on eleven
invented cases whose first reference is COPIED from the emitted carrier: this
is a transport and matcher-arithmetic check on real emitted geometry, not
independent localization accuracy. No physical F1 is claimed anywhere here.

## Finding → closure map

Reports: LD = layer design audit, CD = completion design audit, CS = completion
source audit, MC = material-identity chain audit, AP = arc-policy audit,
MCΔ = chain delta audit (`3b4d51c4…`), LSΔ = layer source delta audit
(`96c11062…`), CΔ = completion delta audit (`9c58a6d1…`), AG = angular-support
policy audit (`b0a43d8a…`), COORD = coordination 04:00/04:27 sections,
IR-K = independent kernel/chain review (`7d0afddd…`), IR-C = independent
completion/wiring/runner review (`4ee023d0…`).

| # | Finding (report) | Closure at source | Test |
| --- | --- | --- | --- |
| 1 | Unresolved identity still emits inner geometry; nominal-R exclusion is not a certificate (LD P1, MC) | Kernel V2: `INNER_FACE_ELIGIBLE_STATUSES = (TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED,)`; ineligible measurements carry `radius_deviation_mm`/`canonical_radius_deviation_mm = None` and the generic carrier role; `outer_arc_nominal_heuristic` published with `certificate=False`, explicit assumptions, domain refusal | kernel V2 outer-only/single-skin tests; chain `test_single_skin…`, `test_outer_only…`, `test_disabled_layer_policy…` |
| 2 | Disabled selection granted inner eligibility from `CONTEXT_ASSUMED_NOT_VERIFIED` (MCΔ P1) | removed from the eligible tuple in kernel V2 and contact V3 | kernel V2 disabled-mode tests; chain disabled-control test (generic carrier, no deviation) |
| 3 | Association eligibility only after a successful canonical fit; ordinary `_Unavailable` became a fatal publication mismatch (MCΔ P1) | context V2 binds identity and initializes both canonical deviations to `None` at association creation; overlap invalidation clears deviations (`record.get("associations", [])` tolerant) | context V2 suite; chain `test_ordinary_canonical_refusal_coexists_with_a_positive…` (forced canonical refusal + retained positive, whole publication and transport validate) |
| 4 | Per-context measurement identity never revalidated; `_bind_identity` accepted 0/1 (MCΔ P2) | validator binds `association["per_context_measurements"]` to the record identity; `_bind_identity` is type-strict | chain tamper test: `inner_face_claim_eligible = 1` → MALFORMED_RESULT; per-context status tamper → publication ValueError |
| 5 | Scalar fit policy vs layer-selection grammar: planner V1 `INVALID_COVER_FIT_POLICY_FIELDS`, cover V3 equality bug (MC, COORD) | context V2 `_fit_policy` scalar seven fields; separate `layer_policy` record (eight fields) bound on records/groups/witnesses | cover V3 suite; chain positive (`set(group["fit_policy"]) == FIT_FIELDS`, layer policy equality) |
| 6 | Projection V4 delegated to frozen contact V2 which rejects every V9 success (MC P1) | contact scoring V3 (schemas V3/V2, identity fields, matcher untouched); V4 calls V3 directly | projection V4 suite (V2 projector must not be called; V2 returns MALFORMED_RESULT on V9 payloads); chain `test_emitted_measurement_bytes_transport_through_the_v4_wrapper…` |
| 7 | Bounded closure needs actual emitted bytes through V4, not only direct V3 (COORD 04:27) | — | chain V4 transport test: wrapper ADMITTED, `contact_projection` byte-equal to the direct V3 projection, `revalidate…v4` scorer input byte-equal, frozen V2 refuses |
| 8 | Layer telemetry is a bounded summary, not lossless provenance (LSΔ P2) | `_select_flange_layer` publishes `summary_scope`, `sort_normal`, `sort_center_mm`, `split_threshold_mm`, `kept_rule`, bounded `qualifying_splits`, reported/omitted split counts, `unqualified_large_gaps` and counts | kernel `test_selected_partition_is_reconstructible…` (kept mask reconstructed exactly), `test_sparse_extra_population…` |
| 9 | Numerical exits lose diagnostics; shared-support invalidation drops them (LSΔ P2) | `_attach` on every refusal/numerical failure with `phases_completed` / `failure_phase`; public rows keep `attempted_diagnostics` on shared-support invalidation | kernel `test_numerical_failure_inside_layer_selection…`, `test_successful_measurement_lists_every_phase…`, `test_shared_support_invalidation_preserves_attempted_diagnostics…` |
| 10 | "Footprint overlap" is a bounding-interval proxy; count name overstates skins; curved-contaminant docstring (LSΔ P2) | renamed `projected_bounding_interval_overlap`, policy field `min_projected_interval_overlap`, reason `GROUP_PROJECTED_BOUNDING_INTERVALS_DO_NOT_OVERLAP`, published basis/intervals/scope; `qualifying_groups_detected` + scope; docstring corrected | kernel `test_island_inside_a_hole_scores_full_interval_overlap_and_is_labelled_a_proxy` |
| 11 | Completion children appended only after evaluation (CS P1) | every selected child published before support binding/preflight | completion `test_selected_children_are_published_before_fallible_work…` |
| 12 | Envelope V6 ignores completion children (CS P1) | envelope V7 counts `completion_lane.hypotheses` at both depths; malformed `evidence` propagates unknown completion counts; docstring V7 | envelope V7 suite |
| 13 | Candidate sign copied from the old face (CS P2) | `_candidate_sign` per candidate via atan2 | completion recovery tests (both shift signs) |
| 14 | Completion-only groups refused by the V16 origin binder (COORD P1) | V16 accepts completion-only groups with validated completion origins, never fabricated main origins | integration suite; completion `test_completion_only_origins_bind_through_the_live_v16_seam…` |
| 15 | Lane byte cap checked before the last mutation and skipped on empty/disabled lanes (CΔ P2-D1) | `_cap_completion_lane` re-applied on every terminal path and after `refinement_schedule`; `ledger_bytes_scope` declared | completion `test_completion_ledger_cap_is_enforced_after_the_last_mutation…` (fixed point, −1 byte refuses, zero-child and disabled paths) |
| 16 | Origin binder did not bind the replaced slot / completed face / signs / parent record (CΔ P2-D2) | unique rejected parent slot derived; replaced face, kept scan plane, completed face/sign at that slot, unchanged passing correspondences and signs, lane parent record (position, ids, axis, child index) all required | completion live-seam tamper matrix (nine stale-field refusals, honest mapping binds) |
| 17 | Vacuous `or True` shift assertion (CΔ P2-D3) | replaced by signed expectations derived from the fixture and the lane's declared axis (`−shift·(axis·u)` and `(z_face − z_origin)·u_z`) | completion recovery tests |
| 18 | No connected completion-only or late-fatal test (CΔ P2-D4) | — | completion live-seam test; `test_late_failure_after_a_passing_completion_retains_children_but_no_surviving_start` |
| 19 | Inactive inherited budget fields and uncharged counters undeclared (CΔ) | runner `completion_budget_declaration` (active/inactive fields partition the budget keys, work-unit formula, combined allowance) | runner source tests |
| 20 | Angular-gap 5 → 30 missing from protocol/design; runner claim overstated (AG P2 ×2) | protocol and design list the delta with fixture-only scope; runner reason rewritten to the uniform-column fixture statement with `spacing_assumption`, `evidence_scope`, `monotonicity`, `value_status` | runner source tests; `tests/test_cad_guided_instances_angular_gap_v9.py` (threshold/monotonicity, phase and numeric uniqueness, missing interior samples, local-vs-global with axial-only neighbours, chord arithmetic); chain `test_frozen_5_deg_angular_gap_refuses_the_fixture_support_and_30_deg_admits_it` |
| 21 | Arc policy: restricted theorem, H1 fixture-only, pair-count caveat, proxy relabel, coupled deltas, uniqueness (AP, COORD) | diagnosis Parts B/E/F corrected; runner `MEASUREMENT_POLICY_DECLARATION` carries `finite_sampling_dependence`, `coupled_effects`, `uniqueness`, `physical_claim` | runner source tests |
| 22 | Connected fixture scope (COORD 04:27): registration/roles are substitutes; copied-carrier reference tests transport | stated in the chain module docstring and the design | — |
| 23 | Outer exit status unobserved in V8 (COORD) | durable waiter `scripts/gold11_v9_execute_waiter.py` records the actual return code beside the attempt directory; protocol binds it | parses; exercised only at execution |

Independent reviews IR-K and IR-C (copied verbatim into `docs/research/`,
hashes above) raised the same connected blockers (completion-only origins,
publish-before-evaluate, envelope counts, scalar-vs-layer policy, contact V2
rejection, eligibility) and are closed by rows 1–16.

## What is not claimed

No physical score, no calibrated accuracy, no comparability with historical
F1 .4580152672, no proof that any V9 mechanism yields a measurement on the
cohort, no phase-robust support guarantee, no material-side verification. The
derived δ/e and 30° policies are pre-declared prospective development values,
not unique or calibrated. Registration evidence and surface-role selection in
the connected fixture are declared substitutes. The 826 passing tests are
consumed-development evidence at these exact hashes, not gates.

## Next

1. Codex's ONE final bounded closure review on these hashes.
2. Exclusive `--freeze` of V9 (new attempt directory), independent PRE.
3. Fresh no-overlap check, then the single `--execute` through the waiter
   (actual outer return code recorded), ps-only monitoring, POST, result
   narrative with honest numbers (null stays null).


## Append-only delta checkpoint — 2026-09-13 05:06 UTC (the 04:40 UTC checkpoint above is preserved unchanged)

Bounded edits after the 04:50–04:58 UTC final closure reviews (coordination
document) and the runner/waiter review. Every prior section, hash and the
826-test result above stay as history; only this delta is new.

### Delta 1 — no present material-side status is inner-face eligible (prior P1)

`INNER_FACE_ELIGIBLE_STATUSES = ()` in kernel V2 and contact V3 (comment states
why: the two-layer selection is a scan-derived side HYPOTHESIS, not isolation
evidence; disabling supplies none). Consequences at source: every measured
window carries `inner_face_claim_eligible=False`, the generic carrier role,
`radius_identity = OBSERVED_SURFACE_RADIUS_NOT_INNER_RADIUS`, no
`radius_deviation_mm`/`canonical_radius_deviation_mm`; the side-hypothesis
metadata (`material_side_status`, layer-selection provenance, arc-filter
diagnostics) and the generic measured contact geometry are retained; the
scorer, endpoints and matcher are untouched; `inner_face_eligible_carrier_count`
stays in the grammar and is expected to be zero. Tests updated: kernel two-skin
case, context V2 (`test_two_layer_selected_status_is_a_side_hypothesis_not_an_inner_face_claim`,
alias test deviations `[None]*4`), cover V3 fixture and deviation assertions,
contact V3 `IDENTITY_ELIGIBLE` (eligibility False, generic role), the
connected chain (two-skin case now asserts the generic carrier with the
two-layer hypothesis, counts 0 eligible / 1 unresolved, no nominal deviation,
also through the V4 wrapper). Protocol, design and diagnosis eligibility
sentences corrected.

### Delta 2 — strict boolean at the two remaining publication sites

Context V2 association eligibility (`is` plus `type(...) is bool`) and cover V3
instance eligibility (`is`). Isolated mutation checks added: context tamper
matrix `association_eligibility_int`, `record_eligibility_int`,
`window_eligibility_int`, `per_context_eligibility_int` (0 in place of False
refuses); cover fault matrix `instance_eligibility_int`, `group_eligibility_int`;
the existing `eligibility_only`/`role_only` tampers now assert True / inner
role are refused.

### Delta 3 — completion-only and mixed origins through the FULL V16 binding and workflow dispatch

`tests/test_cad_guided_completion_v1_engine_integration_v1.py` gains
`_completion_registration(..., mixed)`: group 0 re-originated from the
completion lane (completion-only: its former main hypothesis becomes the
REJECTED two-of-three parent; mixed: a fourth REJECTED parent and a child whose
exact pose merges into the main group, as V15 does on an equal pose key).
`test_completion_origins_bind_through_the_full_v16_binding_and_dispatch_to_measurement[False|True]`
calls the actual `_registration_binding` (lanes `["TRIPLET_COMPLETION"]` /
`["MAIN", "TRIPLET_COMPLETION"]`) and then runs `inspect_cad_guided_mesh_v16`
end to end (all three contexts bound, measurement dispatched, the lane label
published in `registration_array_binding`). Fault matrix (13 cases, one
legitimately skipped): forged main origin on a completion-only group, missing
completion index (mixed and completion-only), child not passed, parent slot
unmarked, lane unusable, parent is a passing hypothesis, replacement names a
passing face — each refuses the direct binding with the named reason and the
workflow refuses without measurement. Source delta in V16: whenever a
completion lane is published, the selected group's completion origins must
equal the PASSED children that claim it, even when that list is empty (a mixed
group can no longer silently drop a passed child; origins without a lane
refuse).

### Delta 4 — waiting launcher bound and exclusive

`scripts/gold11_v9_execute_waiter.py` is now in the runner's frozen `SOURCES`
(and the source-closure assertion). The launcher reserves the three outer
records exclusively (`O_CREAT|O_EXCL`) before starting anything; any existing
record refuses and is never appended to or truncated. The waiting parent is
admitted only with the launcher's one-time token, an exclusive advisory lock on
the reservation, a still-pending exit record and an empty stdout log; a direct
`--waiter` invocation refuses without touching records; the exit record is
replaced atomically once; a waiter launch failure or a waiting-parent failure
records the return code as UNOBSERVED. Scope comment corrected (the launcher
lists attempt-directory entry names only). Fake-child tests
(`tests/test_gold11_v9_execute_waiter.py`, 9): observed return codes 0 and 3,
second launch / duplicate / direct / wrong-token waiter refusals with
byte-identical records, no-reservation refusal, lock serialization, overlap /
non-pristine / existing-record admission refusals, launch failure, waiting-parent
failure, and the real `--waiter` process entry with a fake child. The runner's
own one-shot gate is unchanged.

### Nonblocking precision notes (carried as requested)

- Lane byte cap: `ledger_bytes` is normalized with the field null on every
  non-fatal terminal path; fatal-path mutations are covered by the outer
  serialized report cap only.
- The zero-child cap case now really schedules zero children (all candidates
  axis-unresolved, parent ledger populated) and asserts
  `scheduled_completion_count == 0` before the cap refusal.
- Malformed top-level `evidence` now has a dedicated envelope test: completion
  and candidate counts unknown, direct lists a known lower bound.
- Kernel diagnostics retain the completed prefix on wrapped failures; a
  pre-diagnostic refusal (for example insufficient flange rows before the
  diagnostic block) carries no spectra, by construction.
- The jittered two-column angular test now asserts the exact outcomes
  (admitted at 30°, refused at 5°) and states the exact-uniqueness limitation.

### Delta hashes (SHA256)

| File | SHA256 |
| --- | --- |
| `backend/cad_guided_measurement_v2.py` | `12bb9ce9e595ada6b4cf86947089ef96f2158ddf36009584442e0a414e6ae008` |
| `backend/cad_guided_context_measurement_v2.py` | `9b48c72b4e3eb87899f8e6911929a869c63d0182dd95e254e5deb5671e623f7c` |
| `backend/cad_guided_support_cover_measurement_v3.py` | `6c6bd75813f6e985fd4e841eb401ae330a5be614a6e874b2e5704256e00ca901` |
| `backend/cad_guided_plane_initialization_v15.py` | `505f313dbf3e0487bf03f3b92ab1d42d90f61d5df3123c34c9a91677d132647e` (unchanged) |
| `backend/cad_guided_mesh_offline_v16.py` | `469c444f49cdcd631f04635516304b6af530fe3c813dc1f07cadb51389ed16a7` |
| `backend/cad_guided_report_envelope_v7.py` | `b7abe653c6f7b9832c61687ec7f1ed68aa44361b8ef467b8cae84bf837bef075` (unchanged) |
| `scripts/gold11_contact_scoring_v3.py` | `4f0cbea8cc97056295ab49092465f27230d0707ff53eaa8915f78feea031927b` |
| `scripts/gold11_schedule_projection_v4.py` | `32029cbb813381e360bed791e97701cdfb6735cfa6fa12f9a7f90d2477b95708` (unchanged) |
| `scripts/run_gold11_mesh_development_v9.py` | `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1` |
| `scripts/gold11_v9_execute_waiter.py` | `85913855908ac3a965cb37d7fdd36a8c3464aabd05621184ecd9b4b6e8107688` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md` | `e6fcd96aeb85bf3d91543f63f2001511ca22ff13a46c2bd1f1a51ca810fef54f` |
| `docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md` | `d00991ce3ec38d89c668830eeb28cb505086cb64ab0c62c11c0addfb8c48efce` |
| `docs/research/2026-09-13_fable51_v9_diagnosis.md` | `da7d2c5c4d253e0877e5049de408fcc6c8bb02f2456da7480424e5151599ecba` |
| `tests/test_cad_guided_measurement_v2.py` | `6573fd2643af819ffc97f403027476a296a3e7b953e6d76db8d2f7afa85db95f` |
| `tests/test_cad_guided_context_measurement_v2.py` | `f001906219048910d0742ff96d327b38efb53537e1b90a26c79ada60f6f11b47` |
| `tests/test_cad_guided_support_cover_measurement_v3.py` | `3716947e2caf8c49e931863a128f810d30f1868ffdec490375cfb6408daf1013` |
| `tests/test_cad_guided_plane_initialization_v15_completion.py` | `367a059196a9d3c0df894a5c808a6b4f73c9b552774db82ba67ec790cbf381a7` |
| `tests/test_cad_guided_plane_initialization_v15_registration.py` | `6b36dba73a145435470f6854679805e5ef7070a2ff9fb61eda4cb12d8efb491c` (unchanged) |
| `tests/test_cad_guided_completion_v1_engine_integration_v1.py` | `20695f7faf2e70a6b2d0c33ffaea1a05ca3f8170303ea0cfd33254e690536875` |
| `tests/test_cad_guided_report_envelope_v7.py` | `6c643d4d55d8ae5d7508f4a76ccf06175e951e02eca35f37ac75aa44f46b3eea` |
| `tests/test_gold11_contact_scoring_v3.py` | `5e84791e2a2547b872a16a2f3b39079e1de95f594b411258829197f5188c18dd` |
| `tests/test_gold11_schedule_projection_v4.py` | `9510b9eb422af0844d0a88fbde980c8984bf5a8e5ec0140d1b023419e9e070c1` (unchanged) |
| `tests/test_gold11_mesh_development_v9_source.py` | `88845b14b18c9714d9c585063a2640bcf916c33277b7521d9df3b90a5c2d028c` |
| `tests/test_cad_guided_v9_connected_chain_v1.py` | `8b0399202af65ef5e63d92998cee97e999cc437832e2770c796ff6c14962ed46` |
| `tests/test_cad_guided_instances_angular_gap_v9.py` | `d666eab9a5cbff72c774cdf354266ccfe449569f0d832f190a4b31090980c566` |
| `tests/test_gold11_v9_execute_waiter.py` (NEW) | `20feb44003cb82e5e3583a6b0a4360afd23491e521ae99af300ea8e6b625c04c` |

### Actual test results at the delta hashes (2026-09-13 05:05 UTC, inert namespace, same command)

| Suite | Result |
| --- | ---: |
| kernel V2 | 28 passed |
| context measurement V2 | 184 passed |
| support cover V3 | 82 passed |
| initializer V15 completion lane | 22 passed |
| initializer V15 registration | 6 passed |
| completion V1 engine integration (now with completion-only and mixed full binding/dispatch) | 117 passed, 1 skipped (mixed forged-main-origin: a mixed group legitimately carries a main origin) |
| report envelope V7 | 9 passed |
| contact scoring V3 | 123 passed |
| schedule projection V4 | 255 passed |
| runner V9 source | 6 passed |
| connected chain V1 | 9 passed |
| angular-gap adapter checks | 5 passed |
| execute waiter (fake child) | 9 passed |
| **All thirteen together** | **855 passed, 1 skipped, 0 failed, 31.32 s** |

Nothing else changed. No freeze, PRE, physical run, raw/Gold/attempt/cache
access or score; V8 and all earlier attempts remain terminal. Next, on Codex's
delta acceptance: exclusive `--freeze`, independent PRE and external archive,
one waiter-launched execution with the observed outer exit, independent POST.


## Append-only test-wiring correction — 2026-09-13 05:12 UTC

Codex's delta review found that the cover V3 fault matrix had the
`instance_eligibility_int` and `group_eligibility_int` branches but neither
name in its parametrize list, so those two cases were never executed. Both
names are now in the list. Only those two cases were run (no broad suite rerun,
as instructed):

```text
PASSED test_publication_rejects_tampered_proof_sources_quantities_and_projection[instance_eligibility_int]
PASSED test_publication_rejects_tampered_proof_sources_quantities_and_projection[group_eligibility_int]
2 passed, 82 deselected in 0.25s   (2026-09-13 05:12 UTC, inert namespace)
```

`tests/test_cad_guided_support_cover_measurement_v3.py` is now
`f53fe50dcbe70e8a6119c583b3d574c43caae76a57e5f7812f1a1163e2c41732`
(the parametrize list is the only change; the suite now holds 84 cases, a
count not re-claimed here because the broad suite was not rerun). No source
file changed.

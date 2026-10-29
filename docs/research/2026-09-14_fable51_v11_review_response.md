# V11 review response and source-closure checkpoint — 2026-09-14

Author: Claude Code (`claude-fable-5-1`, effort max), sole engine/test/runner implementation owner, working under
the standing consumed-Gold11 development authority. Codex coordinates the independent audits. V10 and every
earlier attempt remain terminal and unchanged; V11 is NOT frozen, PRE-audited or launched by this document.

Reconciled inputs (hash-verified before reading):
- Kernel audit `docs/research/2026-09-13_gold11_v11_layer_kernel_codex_audit.md`
  (8815d7cd26032a3cedab35da333a1e4aa8feac8ecccc0815df18f176892e3a8d; pinned kernel aa37dd4f…, test 835a0db3…).
- Consumer-chain audit `docs/research/2026-09-13_gold11_v11_consumer_chain_codex_audit.md`
  (521db7207fa1844701016b4dc15a8a1fa87834bd8fba0230bee587ec3fc4e83a; pinned V18 b26d32c2…, bridge a2b0256c…,
  kernel 117cd188…, context 0084c921…, cover 7a0b40cf…, contact cb3072b4…, projection adae1deb…, chain test
  72790f54…).
- Codex coordination messages of 2026-09-14 (POST reconciliation; protocol/design wording; read-only preliminary
  review of the full-bytes fixture).
Both audits recorded later unreviewed source drift; every closure below is against the CURRENT bytes listed in the
closure receipt at the end of this document, not the pinned drafts.

## Kernel audit — closures

| Finding | Closure (source) | Evidence (test) |
| --- | --- | --- |
| P1 band premise recorded, not enforced | `_measure` raises `LAYER_MEMBERSHIP_BAND_MUST_EXCEED_DECLARED_RESIDUAL_POLICY` for an enabled policy whose band is ≤ the feature's `max_fit_residual_mm` (joint contract; disabled policies keep V1 crops; the feature's residual gate is unchanged) | `test_membership_band_must_exceed_the_declared_residual_policy_when_selection_is_enabled`: band = δ and band < δ refused, band = δ + 0.001 accepted, disabled policy not subject to the contract |
| P1 final premise check on previous-membership planes; unbudgeted refit | Membership must reach a FIXED POINT within `layer_refit_iterations` completed refits (published planes fitted from exactly the final labels, final labels exactly their membership); an exhausted loop abstains `TWO_MODES_MEMBERSHIP_UNRESOLVED` / `MEMBERSHIP_NOT_A_FIXED_POINT_WITHIN_DECLARED_REFITS`; no extra iteration | `test_membership_must_reach_a_fixed_point_within_the_declared_refits_or_abstain`: a 2.5°-tilted outer skin (departure < 3°) forces a genuine membership change; with ONE declared refit the exact abstention (all rows kept, 1 completed refit, 1 loop entry, phase PLANE_FIT_ROLE_0), with six the measurement using exactly two completed refits |
| refit accounting counts a starved entry | `refit_iterations_used` = completed two-plane TLS refits with membership recomputed; `refit_loop_entries` = attempts (a starved entry is counted only there); `refit_accounting` string published; `GROUP_BELOW_THREE_ROWS_DURING_REFIT` bound to the starved flag | same test (entries == used on clean fixtures) |
| P2 third-population admission uses single-bin support | Admission is band-population support: band mass ≥ `min_layer_rows` (and ≥ the ambiguity fraction of the smaller skin's mass, separated from both modes) — `third_population_support` published | `test_third_population_admission_is_band_population_support_and_band_mass_is_exact`: the Codex probe (40 at 0, 40 at 1.5, 6 at 2.98, 6 at 3.02) is ambiguous with band mass 12 from two 6-row bins; 11 rows is not |
| P2 quantized “band mass” | Band mass = EXACT inclusive row count within [centre − band, centre + band] on the sorted coordinate (two binary searches per local maximum; `mass_semantics`, `mass_band_mm` published; `mass_half_bins` removed); the interval bounds are evaluated in binary64 and the test reproduces them in the same form | same test: rows 0.0999 mm from the centre count, 0.1001 mm do not (a five-bin window would count 18 instead of 12); widened bins keep exact masses around the published centres |
| kept_rule omits nearest-plane / tie clause; reconstruction only subset-checked | `MEMBERSHIP_RULE` published per selection: argmin of absolute plane distances, ties → plane index 0 (the primary-mode plane), member when within the inclusive band, UNASSIGNED only when outside both bands AND strictly between the planes, outside-sandwich rows assigned to the nearest plane and subject to the unchanged gates; `kept_rule` names the complete rule | `test_membership_rule_is_complete_nearest_plane_ties_sandwich_and_exactly_reconstructible`: tie → 0; overlapping non-parallel bands → nearest; between/outside cases; EXACT equality of group rows, unassigned rows, outside-sandwich members and the kept row SET on the rebuilt flange crops of two public-entry fixtures |
| warped-surface gap: displaced original rows absorbed as “bridging” | Rows outside both bands are unassigned ONLY between the two planes (`bridging_rows` semantics: `UNASSIGNED_BETWEEN_SKIN_ROWS_NOT_VERIFIED_CUT_EDGE_OR_BRIDGE_IDENTITY`); a displaced population outside the sandwich stays in its nearest group — in the SELECTED population it meets the unchanged gate (refused when displaced), in the OPPOSITE population only the pair planarity checks (that skin is never measured; conformance unknown; the `far_rows` reconstruction fixture measures with 1 % rows 0.7 mm beyond the upper skin by design); `diagnostics["flange_support"]` carries per-role kept / excluded-opposite / excluded-unassigned / outside-sandwich counts with COVERAGE_NOT_ESTABLISHED semantics | `test_a_dented_flange_is_refused_not_trimmed_and_a_warped_span_is_not_certified`: single-skin dent refused; two-skin dent BETWEEN the skins = explicit counted exclusion (kept + excluded = crop, no displaced row measured, coverage NOT_ESTABLISHED), above the bound refused with the exact reason; the same dent toward the AIR side kept in the group (outside-sandwich count == displaced rows in the rebuilt crop) and refused NOT_PLANAR; contiguous warp refused |
| projection test vacuous (OR-list) | — | `test_projection_premise_is_validated_after_refits_and_published`: a narrow 0.3 mm strip 1.5 mm above a flat skin, its own plane tilted 12°, projects as one clean mode, reaches a fixed point, and its refitted normal departs 11–13° from the crop normal: `PROJECTION_PREMISE_UNRESOLVED` / `REFITTED_SKIN_NORMAL_DEPARTS_FROM_CROP_TLS_NORMAL_BEYOND_DECLARED_ANGLE` is the only possible disposition; pair compatibility never reached |
| dense-edge “third population” | Under exact band masses a uniformly dense edge is not a population (≈ 330 rows against 0.25 × 1456); it is refused by the unassigned-fraction bound (0.41 > 0.35) with the exact reason | `test_bridging_fraction_above_the_declared_bound_refuses_the_separation_with_all_rows_kept` |
| header/preamble narrative | Kernel header and test preamble state the V10 evidence as an inferred mixture hypothesis with the exact counts (34 single-unresolved role-0 fits: 32 no qualifying gap, 2 singleton-tail; 7 selected; 2 incompatible; 24 selected fits beyond δ: 7 role 0, 17 role 1), 86 diagnostic pairs / 43 attempts, between-skin rows as unassigned rows | — |

Honest scope of the changed mechanism (not a universal guarantee): a between-skin displacement of original rows is
an explicit, exactly counted, unverified exclusion bounded by `max_bridging_row_fraction` (0.35) — the measured
flange is then a skin-membership subset with coverage NOT_ESTABLISHED and no conformance claim — while a displaced
or foreign population outside the sandwich stays in its group: on the SELECTED (measured) skin it is refused by the
unchanged residual gate; on the OPPOSITE skin it faces only the pair planarity checks and that skin's conformance
stays unknown (no upper-surface gate is added). Population fitting is residual-based row selection (nearest
fitted plane within a band); what never selects is any downstream measurement, nominal or Gold quantity.

## Consumer-chain audit — closures

| Finding | Closure | Evidence |
| --- | --- | --- |
| P2 serialized evaluator accepts a broader layer-policy domain | `scripts/gold11_contact_scoring_v5.py` `_layer_policy` mirrors the kernel constructor: `LAYER_POLICY_SCALAR_DOMAIN` ((low, high] ranges), `LAYER_POLICY_INTEGER_DOMAIN`, `LAYER_POLICY_CROSS_FIELD_RULES` (max gap > split gap, bin ≤ split/3, band ≤ split/2), strict JSON types, no engine import | `tests/test_gold11_contact_scoring_v5.py`: 26 out-of-domain cases with EVERY serialized copy mutated together (source policy, every record, every cover group) → `MALFORMED_RESULT` naming the layer policy; 13 boundary cases accepted; `test_serialized_layer_policy_domain_is_bound_to_the_kernel_constructor` proves kernel/mirror agreement on every scalar/integer bound and the cross-field boundaries (binary64 arithmetic identical: 0.1 > 0.3/3 in both). Equivalence is claimed for the strict serialized JSON NUMERIC domain: the constructor float-coerces numeric strings, the scorer rejects strings by design |
| P2 machine labels still claim bridge V1 | `CAD_GUIDED_MESH_OFFLINE_V18_CONTEXT_V4_COMPLETION_V1_KERNEL_V3_NATIVE_BRIDGE_V2_ALLOWANCE_V1` and `FRESH_V18_COMPLETION_V1_KERNEL_V3_NATIVE_BRIDGE_V2_…` in mesh offline V18, projection V6, runner V11 (arm definitions and scoring request) and their tests; V18 header names bridge V2; bridge V2 alias/docstring name kernel V3 (historical quoted headers retained) | runner source / projection V6 / V18 integration tests assert the V2 identifiers |
| full connected fixture WIP | `tests/test_cad_guided_v11_full_bytes_connected_v1.py` (see below) | receipts below |
| acceptance family expectations | `tests/test_cad_guided_v11_connected_chain_v1.py` acceptance family: nominal 2 mm / scan 2.06 mm, skins 2.5 mm apart, cut-edge rows in every inter-mode gap, ≤ 0.005 mm noise, two oblique poses + row-order variant; measured radius 2.06 (never 2.0), constructor-transported carrier within 0.10 mm / 1°, TP 1 against the CONSTRUCTED reference through contact V5; controls: between-skin dent (exact exclusion, coverage NOT_ESTABLISHED, no displaced row measured), air-side dent (NOT_PLANAR, outside-sandwich count == displaced rows in the crop), excess dent fraction, in-crop third population, excess unassigned fraction, no arc support, thin-skin arc mixture | 20 passed |

## Full-bytes real-connected fixture (`tests/test_cad_guided_v11_full_bytes_connected_v1.py`)

Chain (no mocked seam): STEP bytes of an independently constructed strip (CAD nominal radius 5.0 mm, 2.5 mm
skin, 15 mm flanges, 30 mm prism) → real owned STEP reader → real native factory (v1) → real bridge V2 → real
mesh-PLY preparation / topology / CAD sampling → real initializer V16 + registration V5 under the frozen V11
CONFIG registration, noisy-plane, CAD-patch, plane-search, completion, surface and context policies/budgets
(the CPU allowance is the fixture's own: 900 s budget / 720 s guard) → real context surface roles → real context
V4 / kernel V3 / cover V5 / publication validator through `inspect_cad_guided_mesh_v18`.

Scan: a two-skin mesh from declared constants with the NON-NOMINAL inner radius 5.06 mm, skins 2.5 mm apart,
sheared (35°) cut surfaces at both flange ends inside the feature span (between-skin rows whose incident triangles
are not cap-like, so the real surface-role filter retains them: the first fixture revision with square
axis-normal cut patches lost all 558–567 of them to that filter before the kernel, which is exactly what V10's
physical ROIs did NOT do — zero cap-role exclusions — so that revision did not exercise the changed mechanism),
per-coordinate noise |e_i| ≤ 0.005 mm (row norm ≤ √3·0.005) from a seed independent of every deformation, oblique
rigid pose. The strip is symmetric under S: (x, y, z) → (y, x, 30 − z), a proper rotation, so the declared pose
truth is the family {truth, S∘truth}; the real registration finds both members.

Expectations from the constants only: for EVERY context feeding an emitted group, pose action ≤ 0.5 mm on all raw
rows against the pose family (declared after observing 0.085 mm without and 0.32 mm with the between-skin rows
inside the frozen 2 mm correspondence distance; the frozen admission gate is p95 ≤ 1.0 mm; measurement runs on raw
rows and is pose-independent within the crop margins); measured radius within 0.02 mm of 5.06 and more than 0.04 mm
from the nominal 5.0; carrier line within 0.10 mm / 1° of the transported mid-angle generator (invariant under S)
and ≥ 20 mm long — a line-distance/direction/length check, not exact finite endpoint truth; exact per-role
accounting on every measured window (excluded unassigned rows == the 136 sheared rows per flange crop, outside
sandwich 0, kept + excluded == crop, `excluded_cap_like_roi_rows` 0); nonempty retained raw support (arc and both
flange roles, unique in-range original row ids) for every emitted group.

Controls (same chain): a DECLARED SPATIALLY CONNECTED patch of 160 original +x inner-flange rows (flange distance
4.5–9.0 mm × height 8–18 mm; background noise unchanged) displaced 0.2 mm toward the air side → an
ATTRIBUTABLE-CONTEXT LOCAL DENT REFUSAL: every context within the pose-family bound refuses
`FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL` at the role the pose maps the patch to, with `outside_sandwich_member_rows`
== 160 in that role and 0 in the other and the sheared interval intact; out-of-bound contexts admitted by the frozen
p95 gate are reported, never attributed, and any emission must come from them only — this is NOT a whole-workflow
guaranteed dent rejection. The same patch toward the outer skin → on every attributable window the patch is an EXACT
counted exclusion on top of the sheared interval (per-role excluded unassigned rows in {[90, 152], [90, 152] + 160}),
coverage NOT_ESTABLISHED, no patch row and no sheared row in any record's retained raw support (arc and both flange
roles, original ids); the cover level then either emits (retained support disjoint) or abstains by the existing
whole-component cover rule (campaign 1 observed the abstention).

Execution contract (`FRESH_PROCESS_ENVELOPE`): the V5 CPU guard is ABSOLUTE process CPU, so each case runs in its
own fresh process (in-test guard: < 60 s process CPU at start) under wall 1200 s (alarm), CPU rlimit 900 s,
OMP_NUM_THREADS 1, max RSS sampled by `/usr/bin/time -l`; a case whose candidates were expired/omitted by the guard
fails with `EXECUTION_ENVELOPE_CPU_GUARD_OR_QUOTA_TRIPPED_NOT_A_GEOMETRY_RESULT`, never as a geometry result.

Attribution and the between-skin outcome (campaign 1 diagnosis): the frozen p95 ≤ 1.0 mm gate also admits far
members of the S family (observed 4.0–4.5 mm max row action) whose crop boundaries the construction cannot predict,
so exact/interval claims are asserted only on windows whose OWN context pose is within the bound (≥ 1 required);
out-of-bound windows carry invariant accounting and are listed as unattributable; an emitted group is attributed
through every member context. The sheared rows are claimed as an interval per flange crop — 90 rows certainly inside
after eroding every crop bound by the pose bound (15 columns d 3.3–11.7 × 3 shear levels s ≥ 0.9 × 2 ends), 152
possibly inside, 136 nominal — and never as retained support. The between-skin case asserts the mechanism
unconditionally on attributable windows (patch = exact counted exclusion on top of the sheared interval) and then
accounts for BOTH declared cover-level outcomes: predictions with retained support disjoint from the patch and the
sheared rows, or the abstention `NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER` verified from the published records
(campaign 1 observed the abstention: the cross-context arc-ownership rule invalidated all six records and no member's
support covered the union of the eight pose-family contexts — the existing cover rule, no gate weakened, no
prediction forced).

### Codex read-only fixture audit (consumer_chain audit 540aa7bf… / 2ac8b116…) — closures

| Finding | Closure |
| --- | --- |
| optional `finite_support.instance` flange keys made the displaced-row check vacuous | `_assert_group_pose_and_support`: nonempty groups required; `group.raw_support.flange_rows` (both roles) and `arc_rows` must be nonempty, unique, in range and DISJOINT from the patch's original ids |
| seeded random 12 % rows are not a connected warped patch; noise re-drawn | declared connected rectangle `DENT_PATCH` of the +x inner flange grid (160 rows, 16 × 10); background noise from `NOISE_SEED` drawn before any displacement (verified identical between variants) |
| count actual context/role-filtered ids, account cap exclusions separately | the patch and the sheared rows lie ≥ 0.2 mm inside every crop boundary (margins ≫ the pose bound); each window asserts `excluded_cap_like_roi_rows == 0`, kept + excluded == crop, the exact per-role unassigned / outside-sandwich counts, and the receipts publish `pre_role_filter_roi_rows` / `local_input_rows` per window |
| CPU guard is absolute process CPU; one pytest process shares 720 s | `FRESH_PROCESS_ENVELOPE` + in-test fresh-process guard + `_envelope_guard`; the earlier shared-process run (fixture bytes 61f680e9…) is retained below as an execution-envelope outcome: case 1 registered/measured but failed the then-declared 0.2 mm pose bound at 0.32 mm; cases 2–3 ended `PROVISIONAL_REGISTRATION_UNAVAILABLE` after the guard, not a geometry result |
| exact sheared-row identity too strong (rows 0.2 mm from the crop bounds; interior precondition 90 not > 100) | interval claim [90, 152] declared from the construction; tagged sheared rows never retained support; `SHEARED_NOMINAL/INTERIOR/POSSIBLE` declared constants |
| refused contexts need their own pose evidence; windows used for interval claims need their own pose check | `_window_pose_error` per window; attributable windows only; unattributable windows reported |
| RSS: `/usr/bin/time -l` is a final HWM, not a supervisor | `rss_budget_bytes` + `rss_check` declare the HWM/ps-sample check honestly |
| claim limits | noise per coordinate; line checks are not endpoint truth; pose truth bound to the emitted groups' contexts and to the declared symmetric pose family; pose bound 0.5 mm declared with its reason |


## Closure receipt — 2026-09-14 (stable hashes; all executed receipts)

Light focused suites (10 files, inert invocation, actual pytest exit recorded): `912 passed, 1 skipped in 13.67s`, PYTEST_EXIT=0.

Full-bytes real-connected fixture: final bytes `8b0556ccfa2f152887a88505e0760f307c99f773c43fe02abcd811803204335a` differ from the campaign-2 bytes `6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110` ONLY by the between-skin abstention oracle block (21 diff lines: one component holding every record, every member's ARC support missing rows of the component's arc union); that oracle was checked offline against the retained campaign-1 diagnostic dump `7e463b2ae6282881cb22697753dc7f4f49d02fd13161accb4c2a497b2fb3186f` (35,702,457 B: 6 records, component of 6, arc union 1169, arc missing per record [59, 208, 89, 78, 52, 217]) and the module collects (3 tests); no geometry rerun for the oracle change. Launcher `208279c865d72c9a7e5997322980d649fb7057cafcbe898981f1bf425d203b6c` (per case: fresh process, ulimit -t 900, perl alarm 1200, OMP_NUM_THREADS=1, allowance 900/720 s, /usr/bin/time -l final HWM; 4 GiB RSS budget checked against the HWM and ps samples, not continuous enforcement).

| Campaign / case | fixture bytes | START | END | real / user s | HWM B | pytest | outcome |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2 / non_nominal | 6ff4d154… | 2026-09-14T01:34:35Z | 2026-09-14T01:45:12Z | 637.53 / 634.01 | 516112384 | exit 0 (1 passed, 2 deselected in 637.38s (0:10:37)) | {"attributable_windows": 5, "cad_sha256": "9bdfe871348bbba10dd97776844e9abea0d9502c0a1a99b926f6ae1dc19af74b", "cpu_seconds": 636.12, "faces": 17048, "fixture_file_sha256": "6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110", "line_distance_mm": [0.0002604365474593418], "measured_radius_mm": [5.061236680431572], "measured_windows": 5, "predictions": 1, "process_cpu_seconds_total": 636.88, "raw_rows": 9100, "receipt": "V11_FULL_BYTES_REAL_CONNECTED_POSITIVE", "scan_sha256": "fa00260f23bf14dfaa9272331cbbf44277444157bb8a0731258385873cb2afa8", "supported_candidates": 5, "wall_seconds": 636.67, "windows": 5} |
| 2 / air_side | 6ff4d154… | 2026-09-14T01:45:12Z | 2026-09-14T01:55:27Z | 615.03 / 611.74 | 500678656 | exit 0 (1 passed, 2 deselected in 614.86s (0:10:14)) | {"attributable_windows": 3, "cad_sha256": "d83a7980fbc180841c7b0e71782cb1478cc6cd62cbef44514b4cd5fe22529874", "cpu_seconds": 613.64, "faces": 17048, "fixture_file_sha256": "6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110", "patch_rows": 160, "predictions_from_unattributable_contexts": 0, "process_cpu_seconds_total": 614.41, "raw_rows": 9100, "receipt": "V11_FULL_BYTES_REAL_CONNECTED_AIR_SIDE_DENT", "scan_sha256": "e5221894e62012fb6677d54d21d2d1e48de281a575d6f741760d26c1606818dc", "wall_seconds": 614.14, "windows": 8} |
| 2 / between_skin | 6ff4d154… | 2026-09-14T01:55:27Z | 2026-09-14T02:05:46Z | 618.43 / 616.19 | 499056640 | exit 0 (1 passed, 2 deselected in 618.30s (0:10:18)) | {"attributable_windows": 3, "branch": "ABSTAINED_NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER", "cad_sha256": "f62d19b5cf6ec1e7e29032735ed6b50d591816480d9e55bd969f947e9fcf8094", "cpu_seconds": 617.04, "faces": 17048, "fixture_file_sha256": "6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110", "measured_windows": 8, "patch_rows": 160, "process_cpu_seconds_total": 617.81, "raw_rows": 9100, "receipt": "V11_FULL_BYTES_REAL_CONNECTED_BETWEEN_SKIN_DENT", "scan_sha256": "ee1b0c38bc84379909d2356681c9f31f2e8a47e0ea892382eb2077b434d83f15", "wall_seconds": 617.54, "windows": 8} |
| 1 / non_nominal | 5e1eb4c0… | 2026-09-14T00:46:18Z | 2026-09-14T00:56:57Z | 639.15 / 635.67 | 498253824 | exit 1 (1 failed, 2 deselected in 638.90s (0:10:38)) | retained: every geometry/pose/radius/carrier assertion passed; failed only the over-strong exact sheared count 131 vs 136 |
| 1 / air_side | 5e1eb4c0… | 2026-09-14T00:56:57Z | 2026-09-14T01:07:14Z | 617.60 / 614.51 | 511279104 | exit 1 (1 failed, 2 deselected in 617.36s (0:10:17)) | retained: empty predictions, INDETERMINATE, supported registration and first NOT_PLANAR window reached; failed the same exact sheared count 131 vs 136 (nothing after that line claimed) |
| 1 / between_skin | 5e1eb4c0… | 2026-09-14T01:07:14Z | 2026-09-14T01:17:36Z | 621.72 / 618.34 | 501383168 | exit 1 (1 failed, 2 deselected in 621.48s (0:10:21)) | retained: UNAVAILABLE / COMPLETE NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS: the cover-level abstention diagnosed above (all 8 windows measured with the exact exclusion) |

Campaign 2 window-level receipts: positive — 5 supported contexts, 5 measured windows all attributable (pose-family action 0.15–0.36 mm, both family members), 1 prediction, radius 5.0612 mm, carrier line distance 0.00026 mm, retained support arc 1140 / flanges 990 + 1020, unassigned rows per role 119–136 (interval [90, 152]), outside-sandwich 0, cap exclusions 0; air side — 8 windows all refused FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL, 3 attributable (0.12 / 0.32 / 0.36 mm) with outside-sandwich 160 in the dented role and 0 in the other and unassigned within the interval, 5 unattributable (0.52–4.50 mm) reported, 0 predictions; between skin — 8 windows measured, 3 attributable with the patch as an exact exclusion on top of the sheared interval (291 = 131 + 160, 296 = 136 + 160, 279 = 119 + 160), outside-sandwich 0, no patch/sheared row in any record's retained support, cover branch ABSTAINED_NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER (6 records [830, 990, 1110] … [689, 848, 952]). Earlier shared-process run (fixture 61f680e9…, all three cases in ONE pytest process, 818 s, exit 1): case 1 registered/measured, best truth-family pose action 0.32 mm against the then-declared 0.2 mm bound; cases 2–3 PROVISIONAL_REGISTRATION_UNAVAILABLE after absolute CPU-guard exhaustion — execution-envelope outcomes, not geometry results.

The full-bytes evidence is integration evidence of the changed mechanism through the real V18 chain on synthetic bytes; it is not a physical result, a Gold score or launch authority.

| File | SHA256 | Bytes |
| --- | --- | ---: |
| `backend/cad_guided_measurement_v3.py` | `8fd5c3e6a7f6437638f068052d8c71e085980b25fa7d14cca955cf5f66c5d6e6` | 71573 |
| `backend/cad_guided_native_feature_bridge_v2.py` | `ecac003e07b5cb7e91432352c6d88fe6b4bfd46f7fcfbeab19addaba671380a4` | 5947 |
| `backend/cad_guided_context_measurement_v4.py` | `7d12b92204334bce27e0012886fa41b75bc8b256f3c5c31de83cebdd243666b2` | 51403 |
| `backend/cad_guided_support_cover_measurement_v5.py` | `7a0b40cfb888811a039bc677e9c186c415195823db709525d702d76b349d95a6` | 29737 |
| `backend/cad_guided_mesh_offline_v18.py` | `57ae0310135c915bab4820ed55bd0d7595ca069ac47e10e74978933163374fe6` | 50892 |
| `scripts/gold11_contact_scoring_v5.py` | `dd0963c2cbc8173fe94ce686b9fbe04f9bac9998ce1881afd5d4ec7cc1f000c7` | 60121 |
| `scripts/gold11_schedule_projection_v6.py` | `b419903ca8d7e1aa8d934d3e2e4088568db8c39391b8d7118b7f45e4f549bf9d` | 19899 |
| `scripts/run_gold11_mesh_development_v11.py` | `8432203ec6c475644b2f4bf25aaf12f613a6451bdfed48355997a8b279f2d978` | 193724 |
| `scripts/gold11_v11_execute_waiter.py` | `1a7be9fda3d9551e0606764f178f69967c68dfce9d5be576fcee8c6a883e84b9` | 9213 |
| `tests/test_cad_guided_measurement_v3.py` | `9bf3cb07d517951e64ba237642aababf45e7b800194d6ffbb179a8514a3085c5` | 39083 |
| `tests/test_cad_guided_context_measurement_v4.py` | `7105136f89a58d360bf8370e8db0b19e97149c296cdfa18c0eab870e46efcc57` | 59290 |
| `tests/test_cad_guided_support_cover_measurement_v5.py` | `0385a2dac5d236ae186b2945fabd03c8f22746309d1e1ddc5e74ad98231280a8` | 31317 |
| `tests/test_gold11_contact_scoring_v5.py` | `12e9944a0c77a4ec34e9b7df7abd22f3c2e6013d01426152086e6dd08222abdb` | 46313 |
| `tests/test_gold11_schedule_projection_v6.py` | `7d56061c5ad70981265353e52189958030cbe1ebdd0d897b79de32375bf976c8` | 31977 |
| `tests/test_cad_guided_v11_connected_chain_v1.py` | `aa5901f2151e2241cebe5d4d85129b6695d64bf865d4b27700239e823be41891` | 47556 |
| `tests/test_cad_guided_completion_v1_engine_integration_v18.py` | `623692465ce6fa81dc36cc44b079b796e5f03a951eb7ed00e3ed855353948623` | 61437 |
| `tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement_v18.py` | `a87d06d249e475817e80cf1b3632e0faeb16c5d139d4895c92e535edbcb5d38a` | 30510 |
| `tests/test_cad_guided_v11_full_bytes_connected_v1.py` | `8b0556ccfa2f152887a88505e0760f307c99f773c43fe02abcd811803204335a` | 38716 |
| `tests/test_gold11_mesh_development_v11_source.py` | `7c14ecf53df0adc124c57a1357b175bb028712bb8c4c85b58a77015cf868456d` | 21062 |
| `tests/test_gold11_v11_execute_waiter.py` | `4ee1448153f39d5ca5aaf7c833e944ebeb19f4f62d0c90ad25705335922e9e2f` | 8751 |
| `docs/research/2026-09-14_fable51_v11_diagnosis.md` | `664a7a37ca39576513dee6c83563b69807fd79725506b0ea3a292abf8652334b` | 17770 |
| `docs/research/2026-09-14_gold11_mesh_development_v11_delta_design.md` | `468fbfc509893e5ac060ed326d731e6993819c46e4706ddfbac35b9169139f35` | 11583 |
| `docs/research/2026-09-14_gold11_mesh_development_v11_protocol.md` | `987e6a165c820ff5a77918dcf64ee2d04b86167556ec2ae8a050e238324ef1e9` | 13162 |
| `docs/research/2026-09-13_gold11_mesh_development_v10_result.md` | `ca11f41abb8b8f5c76df9e2e66ba21ab9dd3e4afd66482b8f0c544a1d2f7d1ae` | 15707 |


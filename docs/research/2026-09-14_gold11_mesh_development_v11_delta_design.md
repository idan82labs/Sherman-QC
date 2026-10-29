# Gold11 V11 delta design — 2026-09-14

Source-only design record for the V11 successor authored by Claude Code (`claude-fable-5-1`, effort max)
under the standing consumed-Gold11 development authority. Not a qualification, freeze, PRE, execution
request or physical result. V10 and every earlier attempt remain terminal and unchanged.

## Frozen basis and exact delta inventory

Basis: `scripts/run_gold11_mesh_development_v10.py`, SHA256
`f7fd083f8b74a22e7fd1afe872ae8b9b2bfbc94f4b7024fc8c2b46494208da92`, declared as `SOURCE_BASIS` and frozen for
preservation only (the V9, V8 and V7 runners are retained as preservation oracles). The V11 runner is the V10
text with these classes of change:

1. Identities: attempt/protocol/design paths, `GOLD11_MESH_DEVELOPMENT_V11`, `ONE_USE_PARENT_BOUND_LEAF_V11`,
   custody/inventory schemas, private cache prefixes (`gold11-v11-…`), private scoring/sampler namespaces, V11
   error codes, comparator and arm identities. Runner custody, supervision, terminal fill, limits, certificate
   caps, sampler, the allowance and failure handling are the V10 text.
2. Engine wiring in `workflow_entry_and_policies`: mesh offline V18 entry and policy; `LayerSelectionPolicyV2`
   (measurement kernel V3) constructed from the thirteen-field `CONFIG["layer_policy"]`. Everything else the
   worker constructs (registration V5 allowance, initializer V16, context V4 budgets, surface V4, native V1) is
   the V10 constructor set with the V10 values.
3. Scorer projection: `scripts/gold11_schedule_projection_v6.py` (V5 text with the V18 identity, wrapper schema
   `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V6`) calls the NEW `scripts/gold11_contact_scoring_v5.py`
   (V4 text recognizing support cover V5 / context V4, the thirteen-field layer-policy record and the two new
   unresolved material-side statuses, neither inner-face eligible). Endpoint copy, matcher, thresholds (5°,
   4 mm, .60/.60), one-to-one matching, denominator and failure arithmetic are unchanged.
4. Declared configuration delta (`CONFIG_DELTA_FROM_V10`): `layer_policy` gains the five kernel-V3 fields
   (`layer_mode_bin_mm` 0.05, `layer_membership_band_mm` 0.10, `max_bridging_row_fraction` 0.35,
   `layer_refit_iterations` 6, `layer_mode_ambiguity_fraction` 0.25); the eight V1 fields keep their V10 values;
   every other CONFIG value, limit, cap, gate and policy is byte-identical to V10 (the V9→V10, V8→V9 and V7→V8
   deltas are restated as inherited).

## Engine modules (NEW files; frozen predecessors untouched)

| Module | Relation to predecessor | Reason (V10 evidence) |
| --- | --- | --- |
| `backend/cad_guided_measurement_v3.py` | V2 text with ONE changed mechanism: the flange layer separation. Two-mode detection on the material-side coordinate (fixed 0.05 mm bins; the tallest local maximum + the tallest other local maximum with separation in [0.3, 6] mm whose band mass is ≥ 25 % of the primary's; band masses are EXACT inclusive coordinate-band row counts around the bin centre, never a quantized bin window; a further local maximum separated from both with band mass ≥ 12 rows (population support, not a single-bin peak) and ≥ 25 % of the smaller skin's band mass is ambiguous), band membership around each refitted skin plane under the published MEMBERSHIP_RULE (nearest plane, equal distances → plane index 0 = the primary-mode plane, inclusive 0.10 mm band; a row outside both bands that lies STRICTLY BETWEEN the planes is an UNASSIGNED between-skin row — published as `bridging_rows`, never measured, not a verified cut-edge identity; a row outside both bands that is NOT between the planes stays in its nearest group — in the SELECTED population it meets the unchanged gates, in the OPPOSITE (never measured) population only the pair planarity checks, its conformance unknown — so a displaced or foreign population outside the sandwich is never discarded; a membership fixed point within at most 6 completed refits, else abstain; `refit_iterations_used` counts completed two-plane refits, `refit_loop_entries` counts attempts), a declared maximum unassigned fraction (0.35), a projection-premise check (each refitted skin normal within 3° of the crop TLS normal, else abstain), the UNCHANGED pair compatibility and the UNCHANGED smaller-coordinate choice; the unchanged flange gate forms run on the retained skin rows; the band must exceed δ = 0.03 mm (joint feature/policy contract enforced at measurement, `LAYER_MEMBERSHIP_BAND_MUST_EXCEED_DECLARED_RESIDUAL_POLICY`) so the gate is never satisfied by construction; `diagnostics["flange_support"]` publishes per-role kept / excluded-opposite-skin / excluded-unassigned / outside-sandwich counts with coverage NOT_ESTABLISHED semantics. Two new unresolved statuses (`TWO_MODES_MEMBERSHIP_UNRESOLVED`, `PROJECTION_PREMISE_UNRESOLVED`); `INNER_FACE_ELIGIBLE_STATUSES` stays EMPTY | 43/43 physical windows refused at the first flange fit; crops consistent with two populations plus between-skin rows defeat the V1 empty-gap partition — role 0: 34 fits unresolved as a single population (32 with no qualifying gap, largest gaps 0.028–0.230 mm; 2 the duplicated CG1E0 singleton-tail gap 0.433 mm failing the 12-row minimum), 7 selected pairs, 2 incompatible pairs — and leave contaminants in selected groups (24 selected fits failing the 0.03 mm maximum residual: 7 role 0, 17 role 1); 13 selected role-1 populations meet δ (rms 0.005–0.011 mm), identity unverified |
| `backend/cad_guided_native_feature_bridge_v2.py` | V1 text targeting kernel V3's `CADGuidedFeature` (the exact-type consumer guard); implementation `NATIVE_FEATURE_BRIDGE_V2_EXACT_KERNEL_V1_TO_KERNEL_V3_LOSSLESS_FIELD_RECONSTRUCTION`; every current machine identifier of the chain (`…_KERNEL_V3_NATIVE_BRIDGE_V2_ALLOWANCE_V1` in mesh offline V18, projection V6, runner V11 and their tests) names bridge V2 | the V9 class-contract lesson: the bridge must target the consumer's exact class |
| `backend/cad_guided_context_measurement_v4.py` | V3 text bound to kernel V3 (thirteen-field policy record, new statuses); schema V4 | consumer binding |
| `backend/cad_guided_support_cover_measurement_v5.py` | V4 text bound to context V4; schema V5 | consumer binding |
| `backend/cad_guided_mesh_offline_v18.py` | V17 text bound to bridge V2, context V4, cover V5 and `LayerSelectionPolicyV2` | consumer binding |
| `scripts/gold11_contact_scoring_v5.py`, `scripts/gold11_schedule_projection_v6.py` | see item 3; the inert serialized layer-policy validator mirrors the kernel constructor's exact numeric domain and cross-field rules (`LAYER_POLICY_SCALAR_DOMAIN`, `LAYER_POLICY_INTEGER_DOMAIN`, `LAYER_POLICY_CROSS_FIELD_RULES`; a policy the kernel would reject is malformed evidence even when every serialized copy agrees; the scorer still imports no engine module and the mirror is bound to the kernel by test) | consumer binding |
| `scripts/run_gold11_mesh_development_v11.py`, `scripts/gold11_v11_execute_waiter.py` | see items 1, 2, 4; V10 waiter text with V11 paths/env names/schema | runner, outer-exit custody |

## Observation model, evidence and claims

The mechanism is judged against the declared V10 observation model (two parallel planar populations of a
sheet ≈ 1.3–1.8 mm apart in this cohort, per-skin noise rms ≈ 0.005–0.011 mm, a minority of between-skin rows
continuous in the material-side coordinate; their cut-edge identity is a hypothesis, so the kernel publishes them
as UNASSIGNED rows). Evidence (all synthetic algorithm checks, never physical truth or calibration):
1. Kernel V3 modeled fixtures, `tests/test_cad_guided_measurement_v3.py`: two skins with cut-edge rows sampled at
   the scan spacing measured by V3 and refused by V2; the unassigned-fraction bound (including uniformly dense
   edges refused by the bound, not absorbed as a third population); exact third-population admission on the Codex
   bin-boundary probe (40 / 40 / 6 + 6 rows) and exact inclusive band mass (0.0999 in, 0.1001 out; widened bins);
   the complete membership rule (ties → plane 0, nearest plane on overlapping non-parallel bands, unassigned only
   between the planes, outside-sandwich rows kept) with EXACT reconstruction of the published partition and kept
   row set on the rebuilt flange crops; an exact projection-premise rejection (a narrow 12°-tilted second
   population); a deterministic fixed-point abstention with one declared refit and exactly two completed refits
   with six; the joint band > δ contract at, below and just above equality; dented ORIGINAL rows toward the air
   side refused by the unchanged gate with the displaced rows kept in the group, the same dent between the skins
   published as an exact counted exclusion, above the bound refused; a warped span never certified.
2. The Codex acceptance family through the REAL kernel → context V4 → cover V5 → contact V5 path,
   `tests/test_cad_guided_v11_connected_chain_v1.py` (nominal radius 2 mm, scan radius 2.06 mm, skins 2.5 mm
   apart, cut-edge rows filling every inter-mode gap, ≤ 0.005 mm noise, two oblique poses and a row-order
   variant; expectations from the constructor only; TP 1 against the CONSTRUCTED reference line; controls:
   between-skin dent = explicit counted exclusion with coverage NOT_ESTABLISHED, air-side dent refused
   NOT_PLANAR, excess dent fraction refused, in-crop third population ambiguous, excess unassigned fraction
   refused, missing arc support refused, the documented thin-skin arc-mixture refusal).
3. The full-bytes real-connected fixture, `tests/test_cad_guided_v11_full_bytes_connected_v1.py`: STEP bytes →
   real native factory (v1) → real bridge V2 → real mesh-PLY preparation, topology and CAD sampling → real
   initializer V16 + registration V5 under the frozen V11 CONFIG policies (fixture-declared CPU allowance) →
   real surface roles → real context V4 / kernel V3 / cover V5 / validator through `inspect_cad_guided_mesh_v18`,
   on an independently constructed NON-NOMINAL two-skin mesh scan (inner radius 5.06 mm against the CAD's 5.0
   mm, skins 2.5 mm apart, cut-edge rows, oblique pose): pose action, measured radius (never the nominal),
   carrier line, exact support/exclusion accounting; controls: air-side dent refused, between-skin dent
   published as an exact exclusion.
V11 targets a first non-empty output on the slots V10 already admitted; the arc, axis/sweep, canonical-refit
and finite-support gates were never exercised on physical data and may refuse.

## Interpretation boundary

Combined consumed-development changes on top of V10; arm comparison scan-plane lane/schedule only on the same
V18 engine; no speedup, calibration or production claim; historical 0.4580152672 non-comparable and never claimed
beaten; a numeric F1 exists only if the unchanged matcher and aggregation contract return one; a low numeric
score is a baseline with its dominant failures reported. No score arithmetic, matcher, threshold, δ/e, admission
gate or physics bound is changed; no CAD nominal value replaces a measured quantity; no Gold in inference.

## Deferred (recorded, not changed in V11)

Registration admission (18 near poses at p95 1.20–1.62 mm), per-candidate refinement cost (all 31 completed
candidates hit the 30 + 30 iteration limits; median 416 s CPU), early stop of registration when native features
are unavailable (six INVALID_NATIVE_SOLID slots consumed 8914.7 s = 39.4 % of metered lane CPU), CG65/CG84 no
finite start, CGE4B noisy-plane unavailability, INVALID_NATIVE_SOLID solids.

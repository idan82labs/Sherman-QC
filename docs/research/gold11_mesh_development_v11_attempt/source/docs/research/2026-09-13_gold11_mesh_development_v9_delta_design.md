# Gold11 V9 delta design — 2026-09-13

Source-only design record for the V9 successor authored by Claude Code
(`claude-fable-5-1`, effort max) under the standing consumed-Gold11 development
authority. Not a qualification, freeze, PRE, execution request or physical
result. V8 and every earlier attempt remain terminal and unchanged.

## Frozen basis and exact delta inventory

Basis: `scripts/run_gold11_mesh_development_v8.py`, SHA256
`0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5`, declared
as `SOURCE_BASIS` and frozen for preservation only (never imported at runtime;
the V7 runner is likewise retained as a preservation oracle). The V9 runner is
the V8 text with these classes of change:

1. Identities: attempt/protocol/design paths, `GOLD11_MESH_DEVELOPMENT_V9`,
   `ONE_USE_PARENT_BOUND_LEAF_V9`, custody/inventory schemas, private cache
   prefixes, private scoring/sampler namespaces, V9 error codes, comparator and
   arm identities. Runner custody, supervision, terminal fill, scoring
   transport, limits and the contact scorer are the V8 text.
2. Engine wiring in `workflow_entry_and_policies`: mesh offline V16 entry and
   policy, initializer V15 (completion lane V1, `CompletionLanePolicyV1` and a
   separately declared completion budget envelope of the V12 budget type),
   measurement kernel V2 through `LayerSelectionPolicyV1`. Context V4, surface
   V4, registration V4, stage policies V2 and native V1 are unchanged.
3. Scorer projection: `scripts/gold11_schedule_projection_v4.py` (V3 text with
   the V16/V15 identities) replaces V3 in `SCORING_SOURCES` and calls the NEW
   `scripts/gold11_contact_scoring_v3.py` directly (the V2 text recognizing
   the support-cover V3 / context V2 grammar and the material-side identity
   fields); wrapper schema `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V4`,
   projection schema `GOLD11_CONTACT_WORKFLOW_PROJECTION_V3`. The frozen V2
   projector is retained for preservation only and refuses V9 payloads. The
   endpoint copy, matcher, thresholds (5°, 4 mm, .60/.60), one-to-one
   matching, denominator and failure arithmetic are unchanged.
4. Declared configuration deltas (`CONFIG_DELTA_FROM_V8`, frozen in the
   bundle together with `MEASUREMENT_POLICY_DECLARATION` and
   `completion_budget_declaration`):
   `measurement.max_fit_residual_mm` 0.05 -> 0.03 and
   `measurement.plane_exclusion_mm` 0.075 -> 0.04 (derived policy, Part E of
   the V9 diagnosis; one defensible prospective candidate, not unique or
   calibrated; sampling-phase sensitive; coupled effects declared);
   `support.max_angular_gap_deg` 5 -> 30 (prospective conditional
   sample-support policy; fixture and nominal-arithmetic evidence only, as
   declared in the runner's `spacing_assumption`, `evidence_scope`,
   `monotonicity` and `value_status` fields); NEW `completion_policy`,
   `completion_budgets` (equal to the initialization budgets, with
   `completion_budget_declaration` naming the five inherited fields the lane
   does not enforce and the work-unit formula) and `layer_policy` (eight
   fields; `min_projected_interval_overlap` names the bounding-interval proxy
   honestly). Every other CONFIG value, certificate cap, sampler policy, limit
   and the 1.0 mm pose-admission gate with its declaration are unchanged from
   V8. The combined allowance is main initialization budgets PLUS the separate
   completion envelope under the unchanged outer limits.

## Engine modules (all NEW files; frozen predecessors untouched)

| Module | Relation to predecessor | Reason (V8 evidence) |
| --- | --- | --- |
| `backend/cad_guided_measurement_v2.py` | V1 text; both-flange/both-branch diagnostics computed without raising and retained on every refusal and numerical failure with explicit `phases_completed` / `failure_phase`; qualifying-gap partition on the material-side-positive coordinate (every gap above the floor with ≥ 12 rows on both sides; unqualified large gaps published too), scan-derived pair compatibility (per-group RMS/rank, parallel normals ≤ 3°, mean-normal centroid offset in [0.3, 6] mm, projected bounding-interval overlap ≥ 0.5 — a proxy, published with its basis and scope), exactly one compatible split keeps the SMALLER-coordinate group (`TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED`), everything else keeps all rows with an unresolved identity; bounded diagnostic summary with lossless partition provenance (sort normal, centre, split threshold, kept rule); enabled arc filter keeps rows on the axis side of both fitted planes, disabled = V1 absolute band; nominal outer-arc heuristic published with `certificate=False`; `INNER_FACE_ELIGIBLE_STATUSES` is EMPTY (the two-layer selection is a retained side hypothesis, not isolation evidence), so every measurement carries no nominal inner-radius deviation and the generic carrier role; attempted diagnostics preserved when shared support invalidates a row | 43/43 windows refused without a published spectrum; whole-part evidence of two skins; sign, eligibility and provenance audits |
| `backend/cad_guided_context_measurement_v2.py` | V1 text bound to kernel V2; scalar seven-field `fit_policy` unchanged and a SEPARATE eight-field `layer_policy` record in the identity and on every record; identity fields (status, eligibility, carrier role) on windows, records, associations at creation, canonical measurements, instances and the retained per-context measurement, validated type-strictly; association deviations None unless eligible and measured, cleared on overlap invalidation; schema V2 | consumer binding; chain audit P1/P2 |
| `backend/cad_guided_support_cover_measurement_v3.py` | V2 text bound to context V2; binds scalar fit policy and layer policy separately (planner V1 grammar intact); carries the record's identity into the canonical fit, groups, witnesses and validator; schema V3 | consumer binding; chain audit P1 |
| `scripts/gold11_contact_scoring_v3.py` | V2 text: recognizes cover V3 / context V2 / kernel V2 identities, binds layer policy and identity equality across witness/measured/instance, counts eligible and unresolved carriers; matcher and thresholds untouched | chain audit P1 (V2 rejects every V9 success) |
| `backend/cad_guided_plane_initialization_v15.py` | V14 text + additive completion lane after the frozen prefilter loop: for TRIPLET hypotheses with two PASSED and one completed REJECTED support, the third correspondence is re-assigned to every normal-compatible CAD face along the passing pair's intersection direction (λ = m·(a − y)/(m·u); sign cancels; candidate-specific normal sign); every selected child is published before any fallible work; all three supports are rescreened/requeried with the unchanged rules; separate ledger and budget envelope; omission and unresolved counters; lane byte cap re-applied after the last mutation on every terminal path; main-lane strata first; NON-exhaustive |λ|-then-face-id ranking | 12 no-passing-start slots; two-of-three parents in 5/6 scans with feasible completions in 4/6 (derived-record test); completion audits |
| `backend/cad_guided_report_envelope_v7.py` | V6 text + retained completion-child counts at both depths; malformed evidence containers propagate unknown completion counts | completion audit |
| `backend/cad_guided_mesh_offline_v16.py` | V15 text on initializer V15 / support cover V3 / context V2; publishes the completion and layer policies and both implementation identities (the budget declaration is a runner-level frozen declaration); binds completion-only start groups to their PASSED children, the parent's unique rejected slot (scan plane kept, CAD face/sign replaced, passing correspondences and signs unchanged), the lane parent record and parent + shift·axis pose; never fabricates main origins | completion delta audit |
| `scripts/gold11_schedule_projection_v4.py` | V3 text with V16/V15 identities; projects through contact V3 directly | scoring binding |

Consumed-development evidence (invented fixtures, not gates; exact counts and
hashes in the closure checkpoint of
`docs/research/2026-09-13_fable51_v9_review_response.md`): kernel V2 tests
(V1-identical single-skin outputs, inner-layer recovery for two skins at both
sweep signs, unresolved identity for outer-only scans, partition provenance,
sparse-population and holed-footprint proxy labelling, numerical-failure
diagnostic retention, frozen-policy infeasibility at r 0.8 / 90° under the
clean model); context V2, support cover V3, contact scoring V3, projection V4,
envelope V7, initializer V15 registration and completion lane (signed
recovery of the correct third face for both shift signs, ambiguity,
slide-off, publish-before-evaluate, cap fixed point, live V16 origin binding
with a stale-mapping tamper matrix, late fatal after a passing completion),
engine integration, angular-gap adapter checks, the V9 runner source tests and
the real connected chain (kernel → context V2 → planner V1 → cover V3 →
contact V3 and the V4 wrapper on the same emitted bytes → unchanged matcher:
positive, generic-unresolved, outer-only, frozen-policy refusal, 5°/30°
support refusal/admission, coexisting canonical refusal, disabled control,
tamper matrix) pass in the inert namespace. The connected chain's scoring
reference is copied from the emitted carrier: it checks transport and matcher
arithmetic on real emitted geometry, not independent localization accuracy.

## Interpretation boundary

V9 outcomes are combined consumed-development changes relative to V8: the
completion lane, the kernel-V2 layer selection and diagnostics, and the derived
measurement policy act together with the unchanged V8 mechanisms and the
unchanged provisional 1.0 mm admission gate, together with the derived
support-gap policy. No per-change attribution and no calibrated accuracy claim
follow; the in-run arm comparison remains scan-plane lane/schedule only on the
same V16 engine. A completion that passes the prefilter is a proposal that
still has to survive refinement, admission and measurement; its ranking is
non-exhaustive and its omissions are published. Eligibility contract: NO
present material-side status is inner-face-claim eligible
(`INNER_FACE_ELIGIBLE_STATUSES = ()`); a window whose both flanges select a
compatible two-layer pair keeps that selection as a side HYPOTHESIS
(`TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED`, with the layer
diagnostics as metadata), and it, like the single-layer, ambiguous,
incompatible, mixed and disabled-selection outcomes, is a generic conditional
contact line with `inner_face_claim_eligible=false`, no nominal inner-radius
deviation, and is scored exactly like any other contact line. Disabling the
selection is the V1 control and supplies no material-side evidence. A future
eligibility path needs separately bound side evidence and its own review. The
derived measurement policy is a pre-declared engineering choice from the
nominal geometry and the flat-witness inequality under the exact-nominal clean
model, not a value chosen on Gold or per case, not unique and not calibrated;
windows whose flange residual exceeds 0.03 mm are expected to refuse as
NOT_PLANAR, and finite sampling phase can refuse arcs, all retained as
negatives. The support-gap change admits sparser angular stations under the
same abstention rules; it is a chosen development margin, not a calibrated
threshold, and promises no prediction.

## Deferred blockers (unchanged, expected to persist)

CGE4B membership cap (frozen V5 hard maximum), CG65 (no two-of-three parent),
CG64 INVALID_NATIVE_SOLID, CG191/CG4FAF surface-query CPU, the cone-feasible
normal-witness fallback and the finite-support axial proposal search
(finite-start audit items 1–2), and the reusable flange-plane context for
short native windows (flange audit). Removing the finite-start and window
refusals does not promise a measurement or a numeric F1.

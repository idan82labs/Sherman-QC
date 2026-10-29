# Gold11 V10 delta design — 2026-09-13

Source-only design record for the V10 successor authored by Claude Code
(`claude-fable-5-1`, effort max) under the standing consumed-Gold11 development
authority. Not a qualification, freeze, PRE, execution request or physical
result. V9 and every earlier attempt remain terminal and unchanged; no V9
frozen source, attempt artifact, control or cache is edited or reopened.

## Frozen basis and exact delta inventory

Basis: `scripts/run_gold11_mesh_development_v9.py`, SHA256
`4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1`, declared
as `SOURCE_BASIS` and frozen for preservation only (never imported at runtime;
the V8 and V7 runners are likewise retained as preservation oracles). The V10
runner is the V9 text with these classes of change:

1. Identities: attempt/protocol/design paths, `GOLD11_MESH_DEVELOPMENT_V10`,
   `ONE_USE_PARENT_BOUND_LEAF_V10`, custody/inventory schemas, private cache
   prefixes (`gold11-v10-…`), private scoring/sampler namespaces, V10 error
   codes, comparator and arm identities. Runner custody, supervision, terminal
   fill, limits, certificate caps, sampler and failure handling are the V9
   text.
2. Engine wiring in `workflow_entry_and_policies`: mesh offline V17 entry and
   policy, initializer V16 (`_context_configuration`, `CompletionLanePolicyV1`),
   registration V5 identity and `RefinementAllowanceV1` constructed from
   `CONFIG["refinement_allowance"]` and bound to `LIMITS["case_cpu_seconds"]`
   (a mismatch refuses INSIDE the authorized worker, after leaf authorization
   and worker initialization and before the scan/CAD reads and the engine
   call; the same seam is exercised by the runner source test without any
   leaf). The worker passes a bounded
   `stage_checkpoint` printer (65536 bytes of `STAGE_CHECKPOINT` stdout lines
   inside the unchanged 1 MiB whole-leaf log cap; a cap notice line, then
   silence; emitted evidence of what a killed leaf reached, never a result).
3. Scorer projection: `scripts/gold11_schedule_projection_v5.py` (V4 text with
   the V17/V16/V5 identities, wrapper schema
   `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V5`) calls the NEW
   `scripts/gold11_contact_scoring_v4.py` directly. Contact V4 is the V3 text
   plus (a) recognition of the support-cover V4 / context V3 grammar, (b) the
   registration-evaluation contract check (strict whole-set completeness OR the
   explicit evaluated-subset contract with its disjoint disposition ledger;
   anything else is `MALFORMED_RESULT`), and (c) one integer diagnostic,
   `subset_registration_count` (1 when the registration published an
   evaluated allowance subset, 0 otherwise). The endpoint copy, matcher,
   thresholds (5°, 4 mm, .60/.60), one-to-one matching, denominator and
   failure arithmetic are unchanged; `scorer_change=True` is declared with
   that exact scope. The frozen V3/V2 projectors are retained for preservation
   only and refuse V10 payloads.
4. Declared configuration deltas (`CONFIG_DELTA_FROM_V9`, frozen in the bundle
   together with the inherited `MEASUREMENT_POLICY_DECLARATION` and the
   corrected `completion_budget_declaration`):
   - NEW `refinement_allowance` (enabled; `main_refined_groups` 12,
     `completion_refined_groups` 4, `main_query_rows` 6 000 000,
     `completion_query_rows` 3 000 000 as ONE allocation for the whole
     completion lane, `cpu_budget_seconds` 1860 = the unchanged leaf CPU cap,
     `cpu_guard_seconds` 1500, `cooperative_chunk_rows` 32 768). This is a
     declared work allocation INSIDE the unchanged outer caps, not a raised cap.
   - `completion_budget_declaration`: `max_node_visits`, `max_triangle_pairs`
     and `max_normal_pairs` move from the "inactive" to the "active" list
     because the completion lane's finite face queries allocate and charge
     them (and certificate pairs against the certificate caps); the V9 prose
     declaring them inactive was false (Codex cost postmortem). Every budget
     VALUE is byte-identical to V9.
   - Every other CONFIG value, certificate cap, sampler policy, limit, the
     derived δ/e measurement policy, the 30° support gap, the completion policy,
     the layer policy and the 1.0 mm pose-admission gate with its declaration
     are unchanged from V9 (the V8→V9 and V7→V8 deltas are restated as
     inherited).

## Engine modules (all NEW files; frozen predecessors untouched)

| Module | Relation to predecessor | Reason (V9 evidence) |
| --- | --- | --- |
| `backend/cad_guided_native_feature_bridge_v1.py` | NEW adapter: explicit lossless reconstruction of kernel-V2 `CADGuidedFeature` objects from the kernel-V1 objects both frozen native factories (`cad_guided_native_features` r0 and `_v1`) return, every dataclass field in declared order, provenance and unknown semantics preserved; refuses a wrong class, duplicate ids, a malformed factory result and any set above 64 features WITHOUT truncation (64 is the cover V3/V4 and context grammar boundary; the standalone kernel entry cap 16 is not the consumer boundary); no authority, no coordinate conversion | CG4BDD control: `INVALID_SUPPORT_COVER_FEATURES` — the cover's exact-type guard refused the frozen factories' kernel-V1 objects, so no V9 measurement ran on physical data |
| `backend/cad_guided_mesh_offline_v17.py` | V16 text; bridge dispatched immediately after the unchanged native extraction and BEFORE CAD sampling/registration (refusal phase `NATIVE_FEATURE_BRIDGE`, native diagnostics retained); binds initializer V16, registration V5, context V3 and cover V4; the array-binding seam accepts strict whole-set completeness OR the explicit subset contract (context V3 rule) and labels `registration_evaluation`; completion-origin validation unchanged (completion-only, mixed and main origins) | bridge closure; subset contract consumer |
| `backend/cad_guided_global_registration_v5.py` | Re-exports the V4 helpers; `RefinementAllowanceV1` (validated; deterministic group/row quotas, declared timing-dependent CPU guard on ABSOLUTE process CPU including imports/preparation, headroom is not a guarantee); `_CpuMeter` (lane/candidate CPU, wall, attempted rows charged BEFORE the primitive, completed rows only after a validated COMPLETE answer, typed exhaustion reasons, stopped lanes); `_QueriesV5` chunked cooperative full-scan queries with exact merge and typed stops between chunks (an incomplete merged query is discarded whole); exact bound reuse of the diagnosis full-scan query: an owned IMMUTABLE payload (read-only views over private bytes that cannot be made writable again, face ids as a tuple, metadata/diagnostics as canonical JSON text) with a digest binding key, dtype, SHAPE and bytes of every array plus both texts, bound to rows, point bytes, pose bytes, query/index identity, gate and an exact policy/implementation snapshot; integrity of the retained record is checked FIRST and any failure is the fatal `_SurfaceFailure` (never a fresh query, never a typed expiry), a legitimate binding mismatch requeries; `_AllowanceExhausted` is deliberately not a `_SurfaceFailure` | 14/22 leaves killed by the 1860 s CPU cap inside `_paired_candidates` refinement with nothing published; duplicate full-scan query per candidate (cost postmortem) |
| `backend/cad_guided_plane_initialization_v16.py` | V15 text plus: bounded `_checkpoint` ledger (512 records / 65536 bytes, emitted and retained, not a durability attestation; deep-owned snapshots reconciled count/bytes at every publication depth); `_FaceQueriesV16` (V15 face queries with attempted-before-work / completed-after accounting, typed stops before each screen/query and between the eager per-face index builds, the V15 builder text itself unchanged); completion lane: cooperative chunk cap, typed phase-boundary checks before/during enumeration, before binding, before the face-index build (refusing a lane whose first admissible query already exceeds the remaining row allocation so a zero allocation never pays for index construction) and before support evaluation; `_rollback_completion_lane` discards completion groups only and RESTORES every main group to its exact pre-transaction record (a passing duplicate-pose child's completion index appended to a main group is undone; attempted child evidence retained; children unbound and marked unknown); deterministic lane quotas inside the frozen 48-group schedule (main strata first); refinement loop with OMITTED (cannot start) / EXPIRED (typed expiry mid-candidate, nothing partial published, later same-lane candidates omitted) / FAILED (fatal surface or unclassified failure in flight, never reported completed; earlier completed candidates keep their disposition, later ones stay NOT_ATTEMPTED) dispositions; strict `candidate_evaluation_complete` FALSE with any omission, NEW `evaluated_subset_complete`, disjoint `candidate_disposition_ledger` (schema `candidate_disposition_ledger_v1`, partition of the selected set), `publication_contract` `EVALUATED_ALLOWANCE_SUBSET_EXPLICIT_DISJOINT_DISPOSITION_LEDGER_V1`; contexts only from completed candidates; cost receipts and stage checkpoints published on the initialization and the registration evidence | CPU kills without publication; V15 orders all refinement before any publication; lane attribution of kills unproven |
| `backend/cad_guided_context_measurement_v3.py` | V2 text bound to the V16 contract: `_registration_evaluation_acceptable` (strict True OR the explicit subset contract validated type-strictly: contract string, `evaluated_subset_complete`, omission status, ledger schema/partition/selected count, no failed or not-attempted member, at least one expired/omitted member, contexts ⊆ completed, execution statuses agree); the denominator, pose snapshot and window loops run over COMPLETED candidates only; an expired/omitted row carrying a pose or context-support claim is refused (`UNKNOWN_CANDIDATE_CARRIES_CONTEXT_CLAIM`); a strict claim over a non-attempted row is refused; schema V3 | consumer of the subset contract; V2 grammar (strict only) refuses the altered meaning by design |
| `backend/cad_guided_support_cover_measurement_v4.py` | V3 text bound to context V3; schema V4 | consumer binding |
| `scripts/gold11_contact_scoring_v4.py` | see item 3 | consumer binding; every consumer validates the contract |
| `scripts/gold11_schedule_projection_v5.py` | see item 3 | consumer binding |
| `scripts/run_gold11_mesh_development_v10.py` | see items 1, 2, 4 | runner |
| `scripts/gold11_v10_execute_waiter.py` | V9 waiter text with V10 paths, token/config environment names and schema (exclusive outer records, one-time token, advisory lock, atomic exit record, unobserved on failure) | outer-exit custody |

## Allowance semantics (what a run can and cannot claim)

- Deterministic parts: which start groups each lane refines (strata within the
  declared lane quotas, main first) and the row quotas are functions of the
  inputs and the fixed candidate order only; the same inputs give the same
  OMITTED/EXPIRED dispositions from row quotas. Row quotas count QUERY rows
  only (box screens, finite face queries, refinement queries; attempted before
  the primitive, completed after a COMPLETE answer); enumeration, support
  binding and face-index builds charge no rows and are covered only by the
  CPU-guard phase checks.
- Timing-dependent part: CPU-guard stops depend on the machine and the moment
  (absolute process CPU including imports and preparation). They are declared
  as such; the realized trace (disposition ledger, per-candidate cost receipts
  with attempted/completed rows and calls, stage checkpoints with process CPU)
  is published so a POST can bind what actually happened. Two runs may differ
  in which candidates the guard omits; neither is wrong, both are explicit.
- What an omitted or expired candidate means: UNKNOWN. It is never a rejected
  pose, never evidence of absence, never counted as evaluated. Strict whole-set
  completeness is FALSE with any omission; `evaluated_subset_complete` is TRUE
  only when every candidate has a typed disposition, none failed and none is
  left not attempted. Global pose uniqueness and continuous family coverage
  remain NOT_ESTABLISHED in every case.
- Fatal is fatal: surface failures, corrupted retained query evidence,
  checkpoint ledger reconciliation failures, budget refusals of the declared
  envelopes and unclassified errors clear every usable context and return no
  owned measurement support; only the typed `_AllowanceExhausted` produces the
  subset publication.
- Headroom: `cpu_budget_seconds − cpu_guard_seconds` = 360 s is reserved for
  context publication, measurement, publication validation and serialization.
  It is not a guarantee. The largest uninterruptible operations are one
  cooperative chunk query of at most 32 768 rows under the surface budgets'
  per-call abstract work caps and one face-index build under the face build
  caps; both are bounded in work, not in seconds. A kill inside them is a
  retained negative outcome, never a cap extension.

## Interpretation boundary

V10 results are COMBINED consumed-development changes (bridge, allowance,
reuse, subset contract, corrected declarations) on top of the unchanged V9
mechanisms, policies and the provisional 1.0 mm admission gate. The arm
comparison inside V10 isolates only the scan-plane lane and scheduler on the
same V17 engine under the same allowance; timing-dependent guard stops can add
arm differences that are not mechanism differences and are reported as such.
No result of this design is a speedup measurement, a calibrated accuracy
improvement, a production result or a comparison with the historical
0.4580152672 (non-comparable without evidence; never claimed beaten). A numeric
F1 exists only if the unchanged matcher and metric-aggregation contract
(`evaluate_f1_against_manual_gold` / `f1_physical_evaluation_contract_v2`,
imported unchanged by contact V4, whose own schema and diagnostics changed)
return one; null stays null; a low numeric score is a baseline with its
dominant failures reported, not roadmap completion.

## Deferred blockers (unchanged, expected to persist)

CG65 has no two-of-three parent; the CGE4B, CG64, CG191 and CG4FAF blockers of
the V8/V9 diagnoses persist; windows whose flange residual exceeds 0.03 mm
refuse NOT_PLANAR; sampling-phase-sparse arcs refuse; single-layer or
ambiguous-partition windows keep an unresolved identity. Candidates the
allowance omits or expires stay unknown; a lane that expires at a phase
boundary publishes no completion group. All are retained zero-prediction
outcomes of this configuration, never grounds for a change after launch.

## Codex findings closed at source (bounded reviews of 2026-09-13)

| Finding | Closure |
| --- | --- |
| B1 P1: bridge cap 16 would newly refuse the 17-feature CG4BDD case | cap 64 = real cover grammar boundary; 17/64 accepted, 65 whole-refused by bridge and cover; fixture rows labelled factory-conditioned transport; independent constructor-frame check; 1 mm-skin kernel refusal retained as a named regression |
| B4 design: subset must not redefine strict completeness; CPU selection timing dependent | strict flag preserved; `evaluated_subset_complete` + disjoint ledger in a NEW schema validated by every consumer; deterministic group/row quotas with CPU as a declared guard and the realized trace published |
| B4 P1: rollback left completion indices on duplicate-pose main groups | pre-transaction snapshot restored verbatim; real V16 registration through the real V17 binder accepts the retained main; dangling provenance refused |
| B4 P2: completion queries charged after work, never completed | attempted before the primitive, completed after a COMPLETE answer; screen/query/pre-admission tests |
| B4 P2: guard narrower than declared; chunk ignored cooperative size | phase-boundary checks (enumeration, per parent, binding, pre-build with first-query rows, between builds, evaluation); cooperative chunk cap; zero-allocation and between-build tests |
| B4/B2 P2: shallow checkpoint snapshot | deep-owned records, count/bytes reconciled at both depths, later mutation isolated |
| B4 P2: surface-failed attempted candidate reported completed | FAILED disposition for surface and unclassified fatal failures; completed/failed/not_attempted exact sets tested |
| B4 acceptance: conditional/skip-based core tests | explicit two/three-candidate fixtures with a checkpoint-controlled clock; positive and fatal outcomes forced |
| B3 P1: corrupted retained query silently recomputed | integrity first, fatal `_SurfaceFailure`; only legitimate binding mismatches requery; same-query index replacement tested |
| B3 P2: owned copy not immutable; shape unbound | bytes-backed read-only views (write re-enable raises), JSON-text metadata, digest binds shape; tamper matrix; literal byte identity in the chunk-equivalence test |
| Root: subset route tested only with handcrafted evidence | an ACTUAL initializer-V16 subset (first candidate supported, second omitted by the guard) binds through the real V17 binder; a bare claim is refused there |
| Cost postmortem: false completion budget prose; duplicate full-scan query | corrected declaration (values unchanged); exact reuse |

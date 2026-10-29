# V10 diagnosis: what V9's terminal records and the two Codex post-run reports establish, and the successor mechanisms

Date: 2026-09-13. Author: Fable 5.1 worker `1519fdc2` (sole implementation
owner). Inputs: the terminal V9 attempt's DERIVED records only, the frozen V9
source, Codex's scoring-ledger POST seam
(`2026-09-13_gold11_v9_scoring_ledger_codex_post_audit.md`,
`87e7f905e3a33c18b13e6b2261f7c545b9ff4e18102e2cf39a21c5a707a5b9ac`, accepted in
its limited scope) and Codex's completion-cost postmortem
(`2026-09-13_gold11_v9_completion_cost_codex_postmortem.md`,
`5edc6ee890310eb4b22b2d367077affe423a608cae96266fe675adb07fb5618f`, accepted in
its limited scope). The full physical POST is still outstanding; nothing here
edits, re-scores or reinterprets V9, which stays terminal with F1 null. No raw
scan/CAD, Gold, cache or credential was opened. Every mechanism below is a
separately versioned successor candidate that needs its own review, freeze and
independent PRE before any physical run.

## A. Established facts (from the records and the reports)

1. Both arms: TP 0 / FP 0 / FN 71, precision and F1 null, 22 slots in the
   denominator, no arithmetic or omission defect (ledger seam).
2. Fourteen slots (seven cases × two arms) ended with exit −24 (SIGXCPU at
   the 1860 s per-case CPU cap), `budget_stop: null`, supervisor reason
   `INFERENCE_WORKER_FAILED`, NO returned workflow stages. Their internal
   stage is unknown in the receipts. The sampled-stack lines in `OUTPUT.txt`
   (surface V4 `_paired_candidates` / `leaf` / `walk`, distance bounds V3) show
   where the interpreter was at 1770–1861 s, not which lane or candidate was
   being refined; lane attribution of the kills is UNPROVEN.
3. The only slot that reached MEASUREMENT (CG4BDD control, main group
   admitted at inlier-plane p95 0.666 mm) was refused by a contract failure:
   `INVALID_SUPPORT_COVER_FEATURES`. Source: `cad_guided_native_features_v1.py`
   imports `CADGuidedFeature` from the V1 kernel (line 33) and constructs the
   windows with that class (line 692); mesh offline V16 hands those objects to
   support cover V3 (V16 line 621), whose grammar requires the kernel-V2 class
   (cover V3 line 144). V8's chain was consistent (V15 → cover V2 → V1 class).
   Every real V9 measurement dispatch would refuse this way. This is the
   author's defect: the connected-chain test constructed kernel-V2 features
   directly and the engine integration test mocked native features and the
   cover call, so V16's real feature construction never reached the real
   consumer in any test.
4. V15 order of work (cost postmortem, source lines): the completion lane's
   prefilter (candidate enumeration, all three-support rescreens/requeries,
   face-index builds, ledger encoding) runs BEFORE any registration refinement;
   ALL refinements (main and completion groups) then run before ANY context is
   published (V12 publisher requires every candidate attempted and fully
   diagnosed). "Main strata first" orders the refinement list only; it protects
   neither the CPU budget nor an already-passing context. A completion budget
   exception also uses the whole-initialization failure path, invalidating
   already-found main groups (second coupling, deliberately conservative in V9).
5. Per-group source bound in registration: up to 415392 surface-query rows in
   63 calls (coarse/fine iterations, `_diagnose`, `_context_support`); 48
   groups fit the abstract row/call quotas while the leaf has at most 38.75 CPU
   seconds per group before any other work. Abstract quotas do not imply CPU
   feasibility.
6. An exact duplicate full-scan surface query exists per completed candidate:
   `_diagnose` (registration V4 line 440) and `_context_support` (V15 line
   1208) query the same rows at the same pose with no intervening update.
7. The frozen V9 runner's `completion_budget_declaration` is false at source:
   `_FaceQueries` charges node visits, triangle pairs, certificate pairs, normal
   pairs and surface work units (V15 lines 535–548), so `max_node_visits`,
   `max_triangle_pairs` and `max_normal_pairs` are NOT inactive and the
   work-unit formula is not row-only. Enforcement is more extensive than the
   prose says; the frozen record is not repaired, the successor corrects it.
8. The completion lane produced new passing finite starts only on CG4BDD
   (3 control, 2 challenger; all rejected at the unchanged 1.0 mm context
   gate: 0.720–1.338 mm); CG84 enumerated 1 parent / 44 children with ZERO
   passing; CG65 had no parent; CGE4B never reached the lane.
9. Nonblocking: the inherited scorer's `comparison_scope` string carries a
   legacy `NEW_V13…` label; the enriched report's experiment/arm identities are
   correct. The successor reports the actual bindings beside it.

## B. Mechanisms for V10 (each versioned separately; predecessors untouched)

### B1. Feature-class contract (closes fact 3) — per Codex's native feature-contract follow-up

Both accepted native adapter branches (`r0` `cad_guided_native_features` and
`v1` `cad_guided_native_features_v1`) return kernel-V1 `CADGuidedFeature`
objects; the two classes have identical fields but different Python
identities. The frozen factories stay untouched and cover V3's exact-type
check stays. Repair: a NEW successor adapter (`backend/
cad_guided_native_feature_bridge_v1.py`) that, immediately after the unchanged
factory returns, requires EXACT kernel-V1 objects and performs an explicit
lossless kernel-V2 reconstruction from every dataclass field (no class
aliasing, no geometry rederivation, no omitted field): tuple order and count,
feature IDs, nominal source, signed sweep / orientation / span and the native
candidate/window/owner/face/edge/trim-exclusion metadata (including
PARTIAL/unavailable reasons) are preserved byte-for-byte; the bridge publishes
a provenance record (source class, target class, per-feature field digests,
count) and establishes no material-side, unit/frame or measurement authority;
the existing context transform remains the sole feature-to-scan coordinate
conversion. An invalid schema/type, duplicate IDs or an over-limit set fails
before any costly registration with the native diagnostics retained and no
partial measurement prefix; the set is never truncated. Mesh offline V17
dispatches through the bridge for both branches. Regression (no mocked
factory/bridge/crop/fit/cover/validator seam): a tiny independently
constructed OCP bend through BOTH real factory branches and the actual V17
dispatch/bridge, into the real context V2 / kernel V2 / planner V1 / cover V3
/ publication validator with independently constructed analytic scan rows and
a declared identity registration fixture (labelled factory-to-measurement
integration, not full byte-workflow or registration qualification): at least
one supported generic carrier, exact field/provenance preservation, unchanged
input objects, generic-contact semantics (inner eligibility false, inner
deviation unavailable, observed count unknown, conformance INDETERMINATE),
endpoints transported by the existing projection; wrong classes/schema,
duplicate IDs and over-limit inputs yield no partial predictions; and a
proof that the actual successor dispatch calls the bridge (a detached helper
test is insufficient). Negative control: the frozen V1 objects handed
directly to cover V3 still refuse with exactly the V9 reason.

### B2. Observable cost funnel (addresses fact 2 without asserting attribution)

Per-stage, per-lane, per-group cost receipts inside the engine: process CPU
and wall deltas (`resource.getrusage(RUSAGE_SELF)` / monotonic clock behind an
injectable clock seam for tests), surface-query counter deltas, dispositions
(attempted / completed / omitted / expired) and the last durable checkpoint
before each costly phase (input preparation, scan-plane proposals, context
V4, main prefilter, completion prefilter, each refinement candidate, context
publication, measurement). Published in the result and, so that a killed leaf
still leaves evidence, emitted as bounded `STAGE_CHECKPOINT` JSON lines on the
worker's stdout (retained in `OUTPUT.txt` under the unchanged 1 MiB leaf log
cap; the supervisor's retention list is not changed). No raw rows or
coordinates are logged. This lets the POST attribute a kill to a stage and a
lane instead of inferring.

### B3. Reuse of the identical final surface query (fact 6)

Registration V5 (V4 text): `_diagnose` returns its owned, immutable query
result bound to (pose bytes, row count and digest of the rows, index identity,
query policy identity, status COMPLETE); `_context_support` accepts that bound
result and reuses it only when every binding matches exactly, otherwise it
queries. Accounting distinguishes performed from reused calls
(`reused_query_calls`, `reused_query_rows`). Never reused after a partial or
unavailable query. Acceptance: a counted deterministic query double proves one
full-scan query serves both consumers; independently generated finite meshes
(holes, edges, multiple face identities, ties, awkward scales,
threshold-near rows) show byte-identical geometry/gate/uncertainty outputs
between the original and reused paths; a changed pose, row permutation, index,
policy or incomplete result invalidates reuse. This removes one of the
63-call per-group bound; it is a candidate optimization, not a measured speedup.

### B4. Protected verification and measurement time (facts 4 and 5)

Initializer V16 (V15 text) and registration V5 with a declared
`RefinementAllowanceV1`: inside the unchanged 1860 s leaf cap, a main-lane CPU
allowance, a completion-lane CPU allowance and a reserved
final-verification/measurement allowance (values frozen in the runner CONFIG
and published). Rules: (a) the completion prefilter runs only when a
completion refinement allocation exists (remaining slots > 0) and only within
its allowance; (b) a completion budget or allowance exhaustion marks the lane
unusable and its children OMITTED, and does NOT invalidate main groups
(lane-level transaction, explicit change from V9's conservative coupling);
(c) refinement candidates are evaluated in the deterministic main-first order,
each lane checked against its allowance before every candidate; candidates not
started are `OMITTED_BY_CPU_ALLOWANCE` (unknown, never rejected), a candidate
whose surface work exhausts the allowance is `EXPIRED_BY_CPU_ALLOWANCE`
(unknown) and the remaining candidates are omitted; (d) context publication
then proceeds under an EXPLICIT finite-subset contract: the new evidence
fields `candidate_omission_status`, `omitted_candidate_indices`,
`candidate_set_scope = EVALUATED_ALLOWANCE_SUBSET_NOT_WHOLE_FINITE_SET`,
`global_pose_uniqueness` not claimed; `candidate_evaluation_complete` remains
True only for the evaluated subset and is read together with the omission
status by the successor consumers (context V3 grammar, V17 binder). V9's
completeness gate is not removed silently: the successor's consumers require
the omission status to be present and declared, and every omitted candidate is
retained as unknown in the ledger. Acceptance (fake deterministic cost counters
for exhaustive scheduler tests; one tiny process-level CPU-stop test, never a
1860 s burn): main-valid fixture plus an expensive completion tail reaches the
unchanged main evidence within the declared budget with the tail omissions
retained; no-main completion success; completion-all-rejected; completion
burst; cap edge; terminal failure; ambiguous alternative poses; no selected
whole-workflow result disappears because optional completion work consumed
its allocation.

### B5. Corrected declarations (facts 7 and 9)

Runner V10: `completion_budget_declaration` lists node visits, triangle pairs,
certificate pairs, normal pairs and surface work units as CHARGED by
`_FaceQueries` and states the actual work formula; the enriched report carries
an explicit `comparison_scope_note` naming the frozen runner/arm bindings
beside the inherited scorer label (the scorer text is not edited).

### B6. Deferred, not taken now

The inexpensive proposal optimizer plus unchanged conservative verifier (cost
postmortem recommendation 4) is deferred until B2's receipts show whether
refinement or prefilter CPU dominates; a new proposal mechanism without that
evidence would be another untested guess. Raising any cap is excluded.

## C. Order and evidence discipline

B1 and B5 first (small, closes the reachable-measurement defect and the
false prose); then B2 (receipts) so that the successor's own records can
attribute cost; then B3 and B4 with their synthetic acceptance tests; then
protocol/design, independent source reviews, freeze, independent PRE, one
waiter-launched execute, POST. Expected honest outcome classes: the kernel-V2
selection and the derived δ/e and 30° policies get their first exercise on
cohort windows only if a context is admitted within the allowances; slots may
still expire (retained unknown), refuse at the gates, or measure nothing; no
numeric F1 is promised and none may be fabricated where the scorer says null.

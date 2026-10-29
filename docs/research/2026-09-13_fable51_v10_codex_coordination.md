# V10 repair coordination — Codex independent review

Date: 2026-09-13. Fable5.1 session1519fdc2 remains the sole implementation
owner under the existing clean-F1 handoff and standing consumed-Gold11 scope.
Root and Codex reviewers perform bounded independent review, not parallel
engine edits. V9 and all earlier attempts and frozen source remain terminal
and immutable. V10 is unfrozen and not released for physical execution.

## V9 POST correction closure — 14:16 UTC

Root fully read and hash-verified the original physical POST (unchanged,
`bcdd7e4b414d308ca347f40ddda5c78fb53d24f623ae6b6083bb25c1f0b33dca`) and the
reviewer's append-only addendum
`2026-09-13_gold11_mesh_development_v9_post_run_audit_addendum.md`, SHA256
`67237db70c3d2232c62c1ce0ca72e88c21011ce424a5d18ef25a48cf27362a2e`.
All three requested corrections are addressed: archive member-byte identity
is not claimed from a container hash/listing;10/11 case pairs share the same
cause; returned-worker CPU totals with different denominators are not called
paired. The addendum additionally reports the matched eight-slot CPU sum
763.757s ->2535.780s, distinguishing missing final CPU on killed workers.
Acceptance stands within the original custody/observation limits, not as
engine performance or historical comparability. No numeric F1 exists.

## B1 review at the author's eight-test checkpoint — 14:18 UTC

Read/hash anchors before independent source review:

- Native bridge V1:
  `72cb142f28f5af17b0812e31da7437cda99752dae5a256a6aa6dff523998c1cf`.
- Mesh offline V17:
  `3cb73f0860e00e1778c6883bff873d37e48e7f886acf3ce94c55c3cca32a1f35`.
- Factory-to-measurement test:
  `d2a0f9efe87d5e91e9465441648ae0faf6a450916ad100980aac259f24472246`.

The actual V17 diff inserts a lossless bridge after either unchanged native
factory and before CAD sampling/registration; measurement consumes the bridged
tuple. The author's eight passing tests are reported, not rerun by root.
Root read the full bridge and test plus the complete V16-to-V17 diff.

**Not green at this snapshot.** Independent reviewer found, and root verified
in source, that the bridge derives its capacity from the two standalone
kernel `MAX_FEATURES=16` constants. The actual connected cover V3 admits64
features. Thus a purported class-only bridge rejects valid17–64-feature cover
inputs, including the retained17-feature CG4BDD case that reached measurement
in V9. Correct to the actual connected boundary, with explicit17/64 acceptance
and65 whole-set refusal tests; never truncate or alter grouping. This is not
a relaxation of a started experiment's cap: V10 is unqualified, and V9 stays
untouched. Finding sent to Fable (`8edcb4`).

Test limitation: `_analytic_rows(feature)` builds rows from the extracted
feature's origin/axis/radial/span/radius/sweep. That can test factory-conditioned
transport, but is not independent geometric truth. The claimed supported
carrier is additionally under identity-registration and keep-all-role
substitutes. The separately tested public dispatch still uses measurement/
registration doubles. Keep these seam-specific claims explicit. The copied
carrier reference proves scoring transport only. The failed1mm-skin fixture
must remain a named negative regression, not only a comment explaining its
replacement by a2.5mm-skin positive. This bridge repair does not establish
improved kernel handling of realistic thin skins or real measurement accuracy.
Root's scalar check of the comment's formula at R5/e.04 gives68.48434deg for
skin1 and97.19672deg for skin2.5; this is a nominal heuristic only, not a
noise/coverage certificate. A fuller bounded B1 audit is being written by
`v9_material_identity_chain` in its own V10 native-bridge report.

## B2/B3/B4 design review: minimal required safeguards

The non-author cost reviewer read the current V10 diagnosis and relevant
frozen consumer source without any execution or physical access. B2 telemetry
and B3 exact-query reuse are sound directions, not yet qualified implementations.
B4 needs the following narrow corrections before source acceptance:

1. Preserve the original whole-selected-set completeness field. With any
   expired/omitted candidate, `candidate_evaluation_complete` stays false.
   Add `evaluated_subset_complete` under a new schema and reconcile disjoint
   completed/expired/omitted ID sets whose union equals the selected set.
   Every new consumer must validate this explicitly; old context V2 requires
   every candidate complete and cannot silently accept the altered meaning.
2. Fixed ordering does not make CPU-clock admission deterministic. Prefer
   prospectively frozen work/group allocations, with CPU as a safety guard.
   If timing controls the subset, explicitly declare timing dependence and
   bind its realized selection trace; do not promise bit-identical reruns.
3. Account absolute process CPU, including imports and preparation. Charge
   completion enumeration, prefilter and refinement to one allocation.
   Candidate-boundary checks alone cannot stop a long query crossing a reserve;
   use bounded cooperative query/iteration stop points. Headroom must cover
   final verification, measurement, publication validation and serialization.
   Without a bound on the largest uninterruptible operation, do not describe
   this as guaranteed protected completion time.
4. Roll back an optional lane only for a typed, declared allowance/resource
   omission condition. Unclassified errors, malformed provenance, corrupted
   query results, invariant/numerical and publication failures remain fatal
   and clear usable output. Never treat arbitrary `_SurfaceFailure` or
   `ValueError` as graceful omission or publish a successful prefix afterwards.
5. Omitted alternatives remain unknown, including alternatives to a completed
   supported pose. Measurements remain conditional on evaluated contexts;
   no global uniqueness, observed count, missing bend or conformance follows.
6. Bound checkpoint count/bytes before execution so telemetry cannot consume
   the retained-log cap unexpectedly. Distinguish emitted/flushed from durable
   retention, and avoid double-counting nested CPU. Query reuse requires a
   complete immutable result with exact index/rows/pose/policy bindings.

Small acceptance matrix: successful main plus expired completion; no-main
completion success; mid-query expiry; corruption after earlier success;
omitted symmetric alternative; exact-budget edge with preparation charged;
old/malformed subset grammar refusal; reuse hit versus changed pose/rows/
index/policy. Counted synthetic costs plus one tiny process-stop test suffice
for these scheduler contracts; no1860s burn or broad new readiness programme.
Completeness/reproducibility findings already queued to Fable (`cd0918`);
the remaining reserve/fatal/checkpoint safeguards are to be read from this
checkpoint. These are source-review inputs, not new physical-run authority.

No root/agent tests, geometry, inference or rescore ran in this review turn;
no raw/Gold/reference/cache/held-out access. The14:15 process-identity check
showed no Python benchmark/test/geometry job. Fable was busy on V10 source.
Next: complete the bounded B1 review, check concrete fixes and synthetic
receipts, then independently audit the actual successor closure before its
new frozen protocol/PRE/one-shot run. No extra user proceed is needed.

## Review delivery and in-progress correction

The complete B1 audit is now published, fully read and hash-verified by root:
`2026-09-13_gold11_v10_native_bridge_codex_audit.md`, SHA256
`c7fe924eda4e0ea18d6dbcb58fe79c39680bb829a44e5afc9a6f6283138a78f3`.
Its pinned snapshot and findings agree with the root source reads above.
This document and the audit were queued to Fable (`8bb58e`), then the attach
session was safely detached (`6897b5`). The cost-design reviewer is complete;
its entire actionable finding set is recorded above. Both reviewers are idle.

An ensuing narrow source read shows Fable has changed the bridge capacity to64
and added explicit boundary-test code. That is an in-progress correction,
not yet a stable-hash or executed-test acceptance. The original author's
eight-pass receipt applies to the earlier rejected snapshot only. Check the
new bounded test results and final bridge/test hashes at the next checkpoint;
do not claim the revised suite passed merely because it exists.

## B1 delta accepted; fresh author execution receipt — 14:49 UTC

Root fully read the B1 reviewer's append-only delta and verified report SHA256
`36b00f4b1346aec23c0543eb538e29251adadcb841c330fa6b04fcf01101684a`.
The requested source closures are accepted at bridge
`fef3980a5d5d4594b48dacaee763451c798d081044b4cf47630f5d7cd6126fea`,
test `6b3442507936f45a361f711b949801540a002513bd815587d524b171022a7ec9`,
and V17 `3cb73f0860e00e1778c6883bff873d37e48e7f886acf3ce94c55c3cca32a1f35`:
cover-bound cap64, explicit17/64/65 tests, independently constructed scan
geometry, named1mm-skin negative and actual malformed-factory bridge refusal.
The independent review does not turn the known thin-skin refusal into a success.

The author's14:36 progress receipt reports a NEW focused execution at14:35:44
UTC at those exact hashes:13 passed in2.60s, using the inert-namespace command
recorded in `2026-09-12_fable51_progress.md`. Root read that receipt; root and
the source reviewer did not rerun the suite. The earlier8-pass receipt remains
evidence only for its rejected predecessor snapshot. Subsequent integration
edits have already changed the test to
`3e5e94145b71ecaf78c1ee62427c13a063c0d843fc4d9892baab11ccaca51f82`
and V17 to `5c4fa6b8f3fb97a23398b9fb00cb418f2f47f0477b93693c6a3c0c4cfd8a5a63`;
the13-pass result is not claimed for those later drafts. Bridge unchanged.

## B3 implementation findings delivered — 14:34 UTC; recorded14:49 UTC

Root's complete read of registration V5 draft
`5e0ed7120feef16260e9eba12027172a21318c5238531b4a3f2f287b9e26535a`
and narrow V4/surface dependencies identified three concrete issues:

- `_ReusableQuery` is a frozen dataclass but retains the original mutable
  query dictionary and writable result arrays. Surface V4's result arrays
  are not its immutable index arrays. The cached evidence needs owned,
  immutable bytes/metadata or a bound digest verified before reuse, plus
  payload-mutation regressions.
- Query/index object identities and the distance gate do not bind the full
  effective query policy or implementation. Same-object policy/budget mutation
  must invalidate reuse; tests should cover rows, pose, index and full policy.
- `_QueriesV5._one` charges lane rows only after the underlying query returns.
  A failed attempt is counted by V4 but disappears from lane cost. Preserve
  attempted versus completed query work, including failure receipts.

These findings were queued to Fable (tool receipts `4383b3`, `1d7018`), with
safe terminal detachment (`34c2a9`). They are source findings, not executable
counterexample results. The14:49 source hash is now
`4bba69e6ace23fed07c815a6c6291d0e93822f7eae154f8b1e90224bb54f9134`;
an independent bounded review will check the actual delta and its tests.

At14:49 Fable session1519fdc2 was busy/working. A command-identity process
check showed no Python benchmark/test/geometry process. V10 remains an active
unfrozen implementation, not a physical run. B3 reuse and B4 allowance reviews
are delegated on source-only snapshots; root checks connected subset consumers
and dispatch. Fable remains sole engine/test/status editor. No raw inputs,
Gold/reference records, attempt artifacts, opaque caches or held-outs accessed.

## Independent B2/B3/B4 draft audits delivered — 14:55 UTC

Both bounded reports are complete, fully read and hash-verified by root:

- `2026-09-13_gold11_v10_allowance_codex_audit.md`, SHA256
  `d06f1ffc9206e28213fc71b08fad2ff0d5e90cf811b6fa9cb73cb3e3aa943df5`.
  Pinned initializer `6ad06db0419da4109aee3105d164653100cfb011f79bb6819c9d098ddab54209`,
  allowance test `c7855cd7b3d18a6c9b75508de6dcbc7cc2dacbcc1296219bb23d2fbe0d1221c1`.
- `2026-09-13_gold11_v10_query_reuse_codex_audit.md`, SHA256
  `26530f3c58ad01669b3f04c7629274e897d1014aba7784e58660b0b4d181ff43`.
  Pinned registration V5 `4bba69e6ace23fed07c815a6c6291d0e93822f7eae154f8b1e90224bb54f9134`,
  query test `8d61304067583190d854539279a8bfdfb40378a8ef306d85173856fd567368ea`.

**These draft snapshots are not green.** Accepted changes include full effective
query-policy binding, V5 attempted/completed work separation, strict whole-set
versus explicit subset completeness, fatal-output clearing, absolute process
CPU and honest timing-dependent guard/headroom wording. Required narrow fixes:

1. Completion rollback must restore additions to preexisting main-group origin
   metadata, not just remove appended completion groups. Root independently
   confirmed that stale `completion_indices` reach V17's real
   `_validate_completion_origins`, which requires a COMPLETE/usable lane and
   correctly refuses the retained main context. Exercise a passing duplicate
   main-pose child followed by a later expiry, through the real binder.
2. Completion face queries must match V5's attempted-before-work/completed-after-
   success accounting. Include completion preparation guard checks and bind
   completion chunk length to the declared cooperative limit; retain honest
   limits for still-uninterruptible index construction.
3. Published checkpoint snapshots must own their records. The shallow
   `dict(ledger)` lets later refinement append to initialization's record list
   while its count/byte fields remain old. Check both published depths.
4. A surface-failed attempted candidate must be in the failed set, not completed.
   The draft only recognizes `CANDIDATE_NUMERICAL_FAILURE` as failure. Earlier
   completed and later unattempted candidates must remain distinguishable.
5. The core preservation/corruption tests must force the claimed two-candidate
   behavior; conditional assertions and optional skips currently allow vacuous
   passes. Require preserved supported output for typed expiry and no usable
   output plus exact dispositions for fatal corruption.
6. Retained-query integrity failure is not a legitimate lookup-key miss. A changed
   pose/rows/index/policy may query again; corrupted status, payload or digest
   must fail fatally before a replacement query. Use immutable-byte-backed
   arrays and complete tamper-evident metadata/shape bindings, with mutation
   tests. The current test explicitly expecting recovery from tampering must
   change. Distinguish value equality from literal byte-identity claims.

Root confirmed the concrete rollback, shallow snapshot, fatal disposition and
query-tamper paths in source. Findings were queued to Fable: rollback/accounting
(`57bfc6`, `263e28`), query integrity/checkpoint snapshot (`714b0a`, `0fbbf0`),
and the full allowance report with disposition/vacuous-test findings (`5ab67b`).
All attach sessions safely detached. Subsequent narrow reads already show
in-progress author corrections; this is not yet delta acceptance or test proof.

## Root connected-consumer check — 14:55 UTC

Root read complete predecessor-to-successor diffs for context V3, cover V4,
workflow V17, contact scoring V4, schedule projection V5 and the connected-chain
test, plus relevant full validation/call-path bodies. The new subset route is
connected at source: initializer V16 -> V17 array/origin binding -> context V3
-> cover V4 -> contact V4 -> schedule V5. Context V3 retains omitted/expired
candidate rows as unknown, measures only completed supported indices and
refuses unknown rows carrying pose/context claims. Contact V4 recognizes the
new measurement schemas and explicitly counts subset registrations. Matcher,
thresholds and denominator arithmetic are unchanged in the inspected diff.

Read-completion source anchors (not a full pre/post stability qualification):

```text
2f1439fb01e0959e58f3275975b0150afc3d59ed72b3ea22ba0f510fb2a13c56 backend/cad_guided_context_measurement_v3.py
f6972fd538d27b24d2142a453b56b3846be34e2dc5d79ab30f54d198ce960230 backend/cad_guided_support_cover_measurement_v4.py
5c4fa6b8f3fb97a23398b9fb00cb418f2f47f0477b93693c6a3c0c4cfd8a5a63 backend/cad_guided_mesh_offline_v17.py
c0bdfadadec64447acb3db6e6509264c53980e8f9b02a130eb22311b8aceba10 scripts/gold11_contact_scoring_v4.py
996fc87cb85f659d9d604bddb359497abd7bade5306f178703dbac0052499179 scripts/gold11_schedule_projection_v5.py
99a56ab47e82f6b863a73b517f104173546c62c74fec2c79d422f8a6c9b9a62e tests/test_cad_guided_v10_connected_chain_v1.py
```

The new positive connected-chain subset fixture supplies handcrafted
registration evidence, identity pose and keep-all surface roles. It tests real
measurement/projection transport, not actual initializer-to-byte-workflow
integration or physical F1. Requested final bounded regression must send a real
initializer subset through V17 binding and connected consumers; do not repeat
V9's untested factory/interface seam. Fable is actively extending these tests.

No root/reviewer tests, geometry, inference, physical access or rescoring ran.
Next action is one concrete delta review of the author's completed fixes and
focused execution receipts, then the separately frozen successor and independent
PRE. Do not launch from these draft audits, reopen V9 or broaden the dataset.
No new user decision is required. Both bounded reviewers are now complete.

## User reaffirmed Fable ownership and outcome goal — 2026-09-13

The user explicitly asked to push further using Fable and tell it to work until
reaching a goal. Root queued the following concrete outcome to existing Fable5.1
MAX session1519fdc2 (receipts `b81fcc`, `4e0534`; safely detached `e0dd30`):
continue implementation, boundary tests, independent delta review, new frozen
protocol/PRE, single execution and independent POST until a clean reproducible
numeric Gold11 localized F1 is delivered for all11 cases/22 arm outcomes and the
unchanged71 references. Do not stop at passing tests or rerun-ready and do not
request another proceed. If a successor fails, retain it as terminal, diagnose
the mechanism and pursue a separately versioned/frozen/audited successor under
the existing standing authority. A low numeric baseline is not roadmap completion.

All scope, no-Gold-in-inference, historical-comparability, failure-retention,
single-heavy-job, fixed-resource and no-new-paid-cost restrictions were repeated.
Fable remains the implementation owner; Codex coordinates independent reviews.
At the15:18 process check, Fable's11-suite synthetic/integration test command was
actively running as PID64198, not a physical benchmark. Its final result was
not yet observed; no new pass count or physical score is asserted here.

## B3 final source acceptance and interim integration evidence — 15:42 UTC

Fable's 15:38 UTC ledger now supplies the missing outcomes. The 15:18–15:20
eleven-suite run returned 862 passed, 1 skipped, 1 failed: the numeric-alias
test detected an inserted hook in the verbatim predecessor face-query builder.
The author restored the base text and placed the stop in the V16 subclass;
six focused checks passed. The subsequent 15:33:15–15:36:07 thirteen-suite run
reports **879 passed, 1 skipped, 0 failed** at the ledger's listed hashes.
Root observed that process running, then read the author's dated receipt;
root did not independently execute these suites. Later edits require their
own receipts, so this is not blanket acceptance of the moving candidate.

The final B3 source delta is independently accepted in the append-only query
audit, SHA256 `17920e5406695937ce6a985d550faf86f1b42ac32e244bc3edd9f87b3ea96a23`.
Root read and verified both appended deltas, preserving the rejected drafts.
Accepted registration V5 is
`00c1eab81fe016faa44f0a569e40e06b6b37f4275badadb2dc16656e4bcc6f19`;
test is `14dcb8bc29765bc9687e12930dd120a288d475fc1b88d0c0b1daeeb876095bb0`.
Intrinsic retained integrity is checked before lookup keys; only a full key
hit validates against the current index. A genuinely different valid stepped
plate index now triggers a fresh query, while corruption remains fatal. The
author reports 14 focused passes for this post-integration ordering repair.
No B3 blocker remains in the bounded source scope; no speedup is established.

The B2/B4 delta audit, SHA256
`080afcfce573c206c1a00f6410f71f008f1f57e843eeda0c709ca134f3a7d346`, closes
the five original implementation findings at initializer
`7dad12ae569c2044681864bb136105e0fea8750a4c1e454797a9f9fd874cb63b` and
allowance test `b50e781a422a4a5a334151ee5a9eecc609db22df5b6cfcea2247e6aebc1c1168`.
The real initializer now exercises typed expiry after a supported candidate,
fatal failure after success, and duplicate-main completion rollback through
the real V17 binder. Root inspected those test bodies. This closes those
source seams, not a complete physical native-byte workflow or calibration.
Two small corrections remain: a genuinely longer-than-1024-row chunk/stop
test (the original fixture had only 25 rows), and truthful documentation of
query-row charging versus atomic enumeration/binding/eager face-index work.
Fable is implementing these without a new broad experiment or changed cap.

Root reconciled the reviewer's whole-completeness concern: context V3's full
consumer rejects omitted rows under a forged whole=True claim, so the observed
helper issue does not currently bypass measurement validation. But the helper
returns True before inspecting a contradictory new disposition ledger, and
V17's array binder can consequently label that evidence STRICT_WHOLE_SET.
Root requested a narrow helper/binder refusal for contradictory new evidence,
preserving legacy no-ledger whole-set support, with regression coverage. The
request, accurate test-exit accounting and apply_patch-only reminder were queued
to Fable (`bacef2`), then safely detached (`b0cf8d`). No prediction bypass,
physical failure or new metric is claimed from this source finding.

## Runner/waiter delta reviewed; final closure remains prospective — 15:42 UTC

Root completed the V9-to-V10 runner diff (including a separately reread middle
section truncated in the first output), the entire waiter diff, source-test
delta and relevant constructor/checkpoint seam. Source-only review anchors:

```text
ef73dfacf475052ad9b59727a49f80c7a0d6b2210c2651a63c9c7182ddc80217 scripts/run_gold11_mesh_development_v10.py
c2648778195b34ee39b45e217c8b566d0998e4d721634866501305ad6b4952fc scripts/gold11_v10_execute_waiter.py
7116db3500a9541b133c04dd9e3f57ae7e188060bd466df2e3b3160d87e6e37e tests/test_gold11_mesh_development_v10_source.py
9a1542b378118dfec12b245fb5b87ea9b4c758be942e87ddb170e9206e855d37 tests/test_gold11_v10_execute_waiter.py
```

The port keeps the Sep9 input/reference anchors, serial 22-slot denominator,
outer budgets and exclusive waiting-parent exit design. It binds the new V17/
V16/V5 and measurement/scorer schemas, fixed allowance values for both arms,
and bounded checkpoint stdout inside the unchanged 1 MiB leaf log. The fresh
arm comparison is still scan-plane/scheduler on the same engine; it is not an
old-engine/new-engine ablation. Matcher and numerical match gates are unchanged.
Root requested factual provenance wording: only CG4BDD control reached V9
measurement; the other 21 slots failed earlier. The author reports that wording
fixed at runner `163c46658bf2ebe3aa258866d3e7ef431bc9a22974d499b79de10cbe7753db71`.
The V10 prospective protocol was read fully; a separate bounded declaration
review is checking consistency while the final tests are prepared.

No V10 physical freeze/PRE/execute or new numeric F1 has been observed. No root
tests or physical input/reference/cache access ran in this checkpoint. Fable
continues as sole implementation owner toward final receipt, accepted source
closure, exclusive freeze, independent PRE, one waiter execution and POST.

## Continuing Fable assignment and final bounded-review handoff — 15:49 UTC

The existing ten-minute heartbeat `finish-and-audit-sherman-gold11-benchmark`
was updated through the app tool, retaining its name, cadence, ACTIVE status
and this task target. Its stale V7 launch description is replaced by the current
Fable5.1 MAX ownership, V9 terminal/V10 prospective state, exact score goal,
scope/budget/terminality safeguards and quiet-on-unchanged policy. The app
confirmed the update; the saved status/cadence/target were read back. This is
ongoing coordination, not a new experiment or a claim of unattended completion.

The declaration reviewer completed
`2026-09-13_gold11_v10_runner_declarations_codex_audit.md`, SHA256
`cbe7ba9dc2a9d16119fc3027d3d9a852ca5c5d553ddd23f47ea1fd1df8b48af1`:
GREEN in its bounded scope with nonblocking timing/metric wording and a
multi-digit namespace-regex correction. Root fully read/hash-verified it.
Those corrections and the whole-cover-before-windows wording were queued to
Fable (`54e02a`, safe detach `6e7487`); none creates another physical gate.

The final B2/B4 source-only appendix is SHA256
`3a27a38511e465831a825753e60da860e04f777a565ffff3a347cb42fe3e3eba`.
At allowance-test `2afa4f69aee4addcd993137aeb97a8309fca2b5007041cbf754290109c80e3e7`
and V5 `30e66d9c5cdc05d830a635557b24c710d2d16072758ab6c95e4aa954d27ddb2c`,
the real 1500-row split/accounting test and atomic-work disclosure are accepted.
The repeated 25-row fixture tests arithmetic/stop plumbing, not new physical
coverage. Remaining narrow items: inject expiry between chunks of that support
and require no next primitive/exact retained counts/no passed partial child;
remove enumeration from the query-row-charge sentence (CPU still counts it).
Root read/hash-verified the appendix and queued those exact residuals plus the
final report locations (`94ffab`, safe detach `69fdfb`).

Root also read the author's new contradictory-whole-flag helper branch and
the real V17 binder refusal regression at context-V3
`e67d1c35cc66a48ddd0ee521ca4c4f23c08f78606751554c17e8b4cdfd042891`,
context test `d5a81f27cea85b8a720d6b8c8a38a5c51f1417fa3c8390e9395bc803dfa6e306`
and the allowance-test hash above. The specific reported claim is now rejected
in source, preserving legacy no-ledger behavior; matching final execution
receipts remain pending. No blanket new malformed-input qualification is
inferred from this narrow fix.

Fable's later final-suite process was observed running through the attached
session, then was absent from a bounded process check; its final receipt has
not yet been published in the ledger at this handoff. The earlier 879-pass
receipt remains the latest published integration outcome, not proof for later
edits. Fable is busy and continues the agreed goal without another proceed.
Next check: author's closure/actual test exits at final hashes, narrow residual
acceptance, exclusive freeze and independent PRE, then exactly one run and
POST. No V10 physical execution or numeric F1 is claimed by this checkpoint.

## V10 bounded source closure ACCEPTED — 2026-09-13 16:01 UTC

Decision: **ACCEPT the final V10 candidate for exclusive freeze and independent
physical PRE.** No remaining blocking finding in the reviewed V10 delta.
This is not PRE, permission to skip PRE, an execution receipt, a new physical
score, calibrated metrology or roadmap completion.

Root read the complete author closure
`2026-09-13_fable51_v10_review_response.md`, SHA256
`ff6c94daa686b72335562888f0594a9218d3d369c103e94b41e4cd6c71b49e5b`.
All 25 source/test/protocol/design hashes in its table independently match
before and after the final review (`e7e3f9`, `415654`). That table is the exact
accepted source checkpoint; it is not a substitute for the forthcoming frozen
runtime/source/input bundle. The author reports 885 passed, one inherited
mixed-origin skip, zero failed, actual pytest exit 0, 15:52:07–15:55:11 UTC.
Root read the receipt and independently verified its source anchors, but did
not rerun or independently observe the final test process's exit. Earlier
negative test outcomes remain in the author's receipt history.

Final independent deltas, read and hash-verified by root:

- B2/B4 allowance audit:
  `7315fc11851cdfbb1a47967533cccde528cd44f72770a0b85edbaaefde0091a9`.
  Both remaining items are closed. The screen/query variants stop after one
  successful chunk, retain exact counts and incomplete support, and use real
  rollback to unknown. Query-row disclosure excludes enumeration/binding/build.
- Runner/declaration audit:
  `c3e5fdcd6a0684d09b09a496fdd08eacb1757f34918dcf092f6a15e1295d4684`.
  Final D1/D3, whole-cover wording and row-quota declarations accepted. The
  residual phrase "unchanged scorer" in protocol line 141 is nonblocking
  shorthand because adjacent text explicitly states that only metric arithmetic
  and matcher are unchanged. Do not claim that phrase was literally removed;
  no further author/test cycle is requested for it.
- B3 remains accepted by audit
  `17920e5406695937ce6a985d550faf86f1b42ac32e244bc3edd9f87b3ea96a23`;
  root reread final intrinsic-integrity/key-match/current-index validation
  paths and the final allowance disclosure. The final V5 pin is
  `ee05a03eb564c9fcc8c3cee631e4f429d0c66d9ec5468bc932377368105750e7`.
  The earlier B1 bridge acceptance remains applicable to unchanged bridge
  `fef3980a5d5d4594b48dacaee763451c798d081044b4cf47630f5d7cd6126fea`.

Root's connected seams are closed in the bounded source scope. The exact
contradictory-whole-flag case is refused by context V3's helper and real V17
binder; the full consumer still separately refuses legacy-shaped omitted
rows. The real-initializer subset and rollback tests bind actual registration
output through V17. The new runner-constructor test exercises the real V17
workflow and verifies identical allowance/printer objects reach the V16 entry;
its registration, raw/CAD parsing/sampling and measurement remain declared
plumbing doubles, with the .025 mm/36-CAD-row fixture overrides explicitly
shown. This tests dispatch, not physical end-to-end geometry. Root also checked
the B1 real-factory/measurement-chain bindings now target V3/V4 consumers.

Authorized next action under standing consumed-development scope: Fable
performs the exclusive V10 freeze, commissions independent non-author exact
physical PRE and verified archive, then—only after GREEN and a fresh no-overlap
check—launches ONE execute through the frozen durable waiter. Verify archive
member payload hashes, not just archive container hash/listing; state any
remaining evidence limitation. All 22 outcomes/71 references and fixed limits
remain binding. Do not reopen V9 or earlier attempts. Monitor only processes
while live, observe the actual outer exit and writer quiescence, then independent
POST and the honest physical score ledger. No additional user proceed is needed.

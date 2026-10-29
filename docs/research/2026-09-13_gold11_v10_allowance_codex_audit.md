# V10 allowance implementation — independent draft-snapshot audit

Date: 2026-09-13, snapshot checked through 14:52 UTC.
Reviewer: Codex `v9_completion_cost_postmortem`.
Decision: **NOT GREEN at this draft snapshot.** The principal completion/fatal
semantics are improved, but rollback, accounting and acceptance-test gaps below
must be closed. This is source review, not an executed-test receipt, frozen-source
PRE, physical POST or launch authority.

## Scope and pins

Read canonical AGENTS/current status, the complete V10 Codex coordination
checkpoint, the complete initializer V15-to-V16 diff plus relevant V16 bodies,
the complete 246-line allowance test, relevant registration V5 interfaces and
the report-envelope return path. No tests, imports, geometry, physical inputs,
attempt artifacts, raw/Gold/reference content or caches were accessed. Fable is
the sole engine/test editor; this audit is the only write.

Before and after source reads, hashes were identical:

```text
6ad06db0419da4109aee3105d164653100cfb011f79bb6819c9d098ddab54209 backend/cad_guided_plane_initialization_v16.py
c7855cd7b3d18a6c9b75508de6dcbc7cc2dacbcc1296219bb23d2fbe0d1221c1 tests/test_cad_guided_plane_initialization_v16_allowance.py
4bba69e6ace23fed07c815a6c6291d0e93822f7eae154f8b1e90224bb54f9134 backend/cad_guided_global_registration_v5.py
b7abe653c6f7b9832c61687ec7f1ed68aa44361b8ef467b8cae84bf837bef075 backend/cad_guided_report_envelope_v7.py
```

Root independently reviews connected context/cover/publication binding. This
report does not substitute for that consumer review or the independent B3
numerical/chunk-equivalence review.

## Positive source closure

- V16:1767–1775 preserves strict whole-selected-set
  `candidate_evaluation_complete` and adds `evaluated_subset_complete`; expired
  or omitted rows do not make the old completion flag true.
- V16:832–860 publishes contexts only from fully diagnosed attempted candidates,
  with the explicit subset contract and unknown alternative-pose scope.
  The disposition constructor (812–829) checks disjoint membership and union
  against the internally generated selected IDs. Downstream validation of
  arbitrary malformed IDs/ledgers remains the root's consumer-review seam.
- V16:1715–1726 treats `_AllowanceExhausted` separately from `_SurfaceFailure`
  and candidate numerical errors. Genuine fatal paths clear context indices
  (1745–1757), make evaluation incomplete and return no owned measurement
  support (1785–1794). No accepted-prefix leak was found in those inspected paths.
- V5:75–77,201–216 checks absolute process CPU rather than time since meter
  construction, so earlier imports/preparation count toward the guard. The
  deterministic row/group quotas and timing-dependent CPU guard are explicitly
  distinguished. V5:101–107 honestly calls the remaining seconds headroom, not
  a guaranteed completion interval.
- V5:294–309 checks between global-query row chunks and discards incomplete
  merged queries on typed expiry. V16 checkpoint count/byte caps are explicit,
  and emitted checkpoints are not misrepresented as durable by themselves.

## Findings

### P1 — Completion rollback does not restore main-group provenance

V16:1074–1076 appends each passing completion index to whichever exact-pose
group is found. That includes **preexisting main groups** when a completion
pose duplicates one. A later typed expiry calls `_rollback_completion_lane`
(786–809), which removes only groups and lookup entries created after
`main_group_count`, and clears the children's `exact_start_group_index`.
It does not undo `completion_indices` added to retained main groups.

Consequently a retained main group can reference a completion child whose group
binding has been cleared and whose lane is unusable. This violates the advertised
transactional preservation even though the main pose/group index survives. A
strict connected binder can correctly reject those dangling origins, losing the
earlier main context. Root independently confirmed the connected consequence:
V17 `_validate_completion_origins` requires the lane COMPLETE and usable and is
invoked for nonempty origins. That connected-source confirmation is root's
review evidence, not an additional source inspection by this reviewer.

Small correction: stage completion-origin mutations until the completion lane
commits, or snapshot/restore all touched main-group metadata on rollback. Do not
remove preexisting main provenance or hide the attempted completion evidence.

Acceptance: first completion child duplicates an existing main pose and passes;
a later child expires. Assert byte-equivalent main-group provenance before/after
rollback, no dangling child indices, retained attempted child evidence and a main
context accepted by the real successor binder. The present tiny-row test expires
early and does not force this successful-duplicate-then-expiry sequence.

### P2 — Completion query accounting violates the attempted/completed contract

V5 `_CpuMeter` defines separate attempted rows/calls (154–157,185–194) and its
global-query wrapper charges before the primitive and completes after success
(279–292). V16 `_FaceQueriesV16.screen` and `__call__` (771–782) instead invoke
`charge_rows` **after** successful work and never invoke `complete_rows`.

Thus a failed completion query consumes work without the attempted lane charge;
a successful completion query increments attempted work but leaves completed
work at zero. In a subsequent completion refinement, those zero completed
prefilter counts remain mixed into the shared lane summary. This directly
undermines the cost attribution introduced to diagnose V9.

Small correction: charge attempted rows/calls after successful quota admission
but before execution, and completed rows/calls only after the primitive's
validated COMPLETE result. Apply equally to box screens and finite queries.
Test a successful screen/query, a primitive failure after admission, and a
pre-admission typed stop; reconcile lane and stage deltas without double-counting.

### P2 — Completion guard coverage is narrower than its declared envelope

The completion lane still enumerates parents/faces, binds supports and eagerly
builds all needed face indices before its first `_FaceQueriesV16._stop` call.
The subclass constructor performs `super().__init__` at 761 before storing its
meter. V16:1029 also chooses `min(lane_budget.chunk_rows,
surface.max_query_rows)` without the allowance's `cooperative_chunk_rows`.
The default completion chunk may therefore be 65536 rows while the declared
cooperative chunk is 32768.

With an already-exhausted absolute guard, optional completion preparation still
runs before the check. Enumeration/build CPU is included in the eventual meter
receipt but is not stopped by the claimed completion guard; enumeration does
not spend a deterministic row token either. The V5 allowance prose at 96–107
overstates the whole-completion scope and the maximum uninterruptible operation.
This is not a demonstrated physical timeout or a reason to enlarge any cap.

Small correction: check the typed guard before completion preparation and at
bounded parent/face/build boundaries, and enforce the declared cooperative row
limit on completion support chunks. Explicitly retain any still-uninterruptible
index-build/primitive work limit and distinguish row-query quotas from
enumeration counters. A zero row allocation should not need expensive index
construction merely to discover the first row cannot be admitted.

Acceptance: guard already exceeded before completion; zero completion-row
allocation; expiry between two face builds; a support longer than the
cooperative chunk. Instrumented doubles should prove no forbidden subsequent
operation starts and no completion query exceeds the declared row chunk.

### P2 — Checkpoint snapshots share a mutable records list

`_publish_cost` at V16:1558–1559 uses `dict(ledger)`, a shallow copy.
`_initialize` publishes that snapshot at 1393 before later refinement checkpoints
append to `ledger['records']`. The nested initialization report therefore keeps
its old `count` and `bytes` integers while its shared records list grows.
`bound_initialization_report_v7` returns the same detailed result at line 273;
it does not detach the snapshot.

Small correction: publish an owned snapshot of checkpoint records and their
contents. At each published depth, assert count equals list length and bytes
equals the declared encoded-record sum. A later checkpoint must not mutate an
earlier snapshot. The existing test checks top-level count and only a lower
bound on initialization count, so it misses this ordinary successful-run path.

### P2 — A fatal attempted candidate can be reported as completed

`_reconcile_dispositions` at V16:817–818 treats every `ATTEMPTED` row as completed
unless its reasons contain exactly `CANDIDATE_NUMERICAL_FAILURE`. A
`_SurfaceFailure` at 1722–1724 leaves that execution status unchanged and stores a
different reason. The overall result correctly becomes unavailable, but its
disjoint ledger labels the unfinished fatal candidate completed.

Small correction: assign an explicit failed/fatal disposition for an interrupted
candidate, or require a completed diagnostic/context record before classifying
it as completed. Keep earlier genuinely completed evidence distinguished from
the failed current candidate and from later not-attempted candidates.
Test a surface failure mid-candidate after an earlier success and require the
exact completed/failed/not-attempted sets, as well as zero usable output.

## Essential acceptance-test correction

The main-preservation test named
`test_cpu_guard_omits_later_candidates_and_publishes_the_completed_context_under_the_subset_contract`
does **not** require a completed supported context: lines 139–147 make those
assertions conditional on a nonempty completed set, and allow either
CONDITIONAL_MEASUREMENT_CONTEXTS or UNAVAILABLE. It can pass when the guard
omits everything. Its fake clock advances per clock read, so incidental receipt
or checkpoint reads move the stopping point. The corruption-after-success test
also makes its principal assertions conditional on having at least two candidates.

Use an explicit two-candidate fixture and a controlled clock advanced at the
first candidate's END checkpoint. Require: first candidate fully supported;
second expired/omitted; exact ledger IDs; whole-set completion false; subset
completion true; nonempty preserved measurement context and owned support.
Separately trigger the fatal variant and require no usable output. Do not
condition away either behavior or rely on a skip to establish this core claim.

This is a small necessary strengthening of the existing matrix, not a request
for another broad test programme. Other existing tests already cover basic
group-quota scheduling, all-expired outcome, no-allocation skip, early completion
rollback and checkpoint truncation at source; none was executed in this review.

## Next action

Fable should close the named seams in new draft source, record focused executed
test receipts and provide new hashes for a delta review. No source is frozen or
released by this audit. No conclusion about physical F1, metrology accuracy or
performance improvement follows from the source closures alone.

## B2/B4 delta review — 2026-09-13 15:37 UTC

This append-only delta preserves the initial snapshot and its NOT GREEN decision
above. Scope: the five named allowance defects and their strengthened synthetic
test source. Canonical instructions/current status and the existing audit were
reread. No tests, imports, physical inputs, attempt artifacts, caches or engine
edits were performed. The author remains the sole engine/test editor; root owns
the connected consumer and runner review. Test assertions below are inspected
source, **not independently observed passing execution**. The author's expanded
suite requires its own final exit/count receipt at matching source hashes.

The following before/after pins were identical during this review:

```text
7dad12ae569c2044681864bb136105e0fea8750a4c1e454797a9f9fd874cb63b backend/cad_guided_plane_initialization_v16.py
b50e781a422a4a5a334151ee5a9eecc609db22df5b6cfcea2247e6aebc1c1168 tests/test_cad_guided_plane_initialization_v16_allowance.py
19137bd8008c43de03fef1f6501c9c4e58eab8dd17868e103d9a67a937370406 backend/cad_guided_global_registration_v5.py
```

These are active unfrozen draft snapshots. Later edits are not covered by this
decision even if their filenames are unchanged.

Final-write pin check: initializer and allowance-test hashes remained unchanged,
but V5 moved while this report was being appended to
`00c1eab81fe016faa44f0a569e40e06b6b37f4275badadb2dc16656e4bcc6f19`.
The V5 findings below apply to the inspected `19137bd8...` snapshot only;
the moving revision has not been reviewed here. This final pin does not erase
or reopen the completed snapshot review.

### Original findings: source closure

1. **Duplicate-main rollback is repaired.** V16:1362 takes a deep snapshot of
   all main-group records plus the pose lookup before completion. The typed
   expiry path passes that snapshot at 1369; rollback at 833–841 restores the
   main records and lookup, including removal of appended `completion_indices`.
   Attempted child evidence is retained but its published group binding is
   cleared (842–850). The new test at allowance.py:340–393 requires an actual
   passing duplicate-main child followed by a later expiry; compares retained
   main/start-group JSON exactly with completion disabled; requires a supported
   retained main context through the real V17 binder; and requires that binder
   to refuse a deliberately restored dangling completion origin. This is
   nonvacuous source coverage of the reported transaction defect, under its
   explicitly modified small-scene coverage policy (331–332).

2. **Attempted/completed completion work is repaired.** V16:784–802 checks
   admission first, charges attempted screen/query rows before the primitive,
   and charges completed rows only after a successful return. V5:185–194 keeps
   the counters distinct. The test at 396–431 requires successful screen/query
   accounting, a post-admission primitive failure with attempted but not
   completed work, and a pre-admission typed stop with no charge. The completion
   receipt now also includes completed rows at V16:1372–1374.

3. **The missing early stops and inconsistent chunk selection are repaired in
   source.** V16:945/966 checks the guard before enumeration and between parents;
   1091/1103/1117/1122 checks preflight, support binding, face-index preparation
   and support evaluation. A positive first-query row admission is checked
   before face indexes are built. `_FaceQueriesV16` preserves the whole-face
   count/triangle preflight and checks between individual eager builds
   (764–776). Actual query chunks and planned-call denominators use the capped
   chunk (1083–1098), including `cooperative_chunk_rows` when enabled. The tests
   at 434–476 require no enumeration/build on an already-expired CPU guard, no
   build with a zero row allocation, and exactly one built face before a guard
   stop at the next build. The remaining chunk-test and scope-disclosure gaps
   are listed below; this is not a proof of bounded elapsed time.

4. **Checkpoint ownership is repaired.** V16:1634–1638 deep-copies the records
   and verifies both count and encoded-byte reconciliation before publishing
   each snapshot. The test at 506–518 requires a strictly smaller initialization
   prefix than the final ledger, reconciles both, verifies distinct record/list
   identity and mutates the outer list without changing the inner snapshot.
   The bounded emitted-not-durable checkpoint contract remains explicit at
   V16:731–751.

5. **Fatal disposition accounting is repaired.** V16:1801–1813 marks the
   in-flight surface/unclassified failure `FAILED` before rethrowing;
   862–879 assigns that ID to the failed set and checks an exact disjoint
   partition. Outer fatal handling still clears usable publication and leaves
   later candidates not attempted (1832–1868). The strengthened test at
   allowance.py:281–313 forces three candidates, requires first-completed,
   second-failed, third-not-attempted exact IDs, and requires no transform,
   owned roles or published measurement contexts for both fatal variants.

The previous vacuous preservation test is replaced by a two-candidate fixture
at 239–278. Its clock advances only at the first candidate END checkpoint.
It now unconditionally requires a supported first context, omitted second
candidate, exact ledger IDs, false whole-set completion, true subset completion,
preserved owned support, and successful/refused real V17 binding for the first/
second candidates respectively. This establishes a meaningful proposed test,
not evidence of default-schedule speed or physical registration accuracy.

### Two small remaining corrections

- **Make the chunk-boundary test exercise a boundary.** The test named
  `test_completion_supports_longer_than_the_cooperative_chunk_are_queried_in_capped_chunks`
  (479–503) explicitly uses only 25-row supports with a 1024-row chunk
  (494–495). It therefore exercises neither splitting nor an intra-support stop;
  `max(calls + screens) <= 1024` would pass the old 65536-row chunk behavior for
  this fixture. Small acceptance: use a support longer than 1024 rows under the
  same declared budget, assert exact chunk lengths/planned counts, and advance
  the injected guard after the first successful chunk. Require no next primitive
  call, exact attempted/completed counts and an expired/unknown child with no
  published partial support. This closes the named regression without a large
  simulation or another readiness programme.

- **Narrow the allowance documentation to what is charged/stoppable.** V5:96–99
  still calls the query-row allocation an allocation for enumeration as well
  as prefilter/refinement. Enumeration, binding and face-index construction have
  no query-row charge; their process CPU is covered by the absolute guard and
  completion section receipt, with phase-boundary checks. V5:105–107 also says
  the largest uninterruptible operation is one chunk query, although an eager
  face-index build and bounded binding/enumeration sections remain atomic
  between checks. State these limitations explicitly. Retain the existing
  distinction between deterministic row/group quotas and timing-dependent CPU
  stops; retained headroom is not guaranteed measurement/serialization time.

One connected consumer issue was sent to root rather than independently
expanded here: allowance.py:267 expects a subset evidence object with omitted
ID 1 to remain acceptable after forcing `candidate_evaluation_complete=True`.
This expectation appears inconsistent with preserving an honest whole-set
claim. Root must reconcile the actual consumer/binder grammar; accepting the
five initializer fixes does not accept that forged whole-set claim.

### Narrow delta decision

The five original initializer/accounting defects are addressed in inspected
source, and the main-retention, rollback, ownership and fatal tests are now
nonvacuous. **B2/B4 acceptance is conditional on the two small corrections above,
matching executed-test receipts, and root's connected consumer resolution.**
This is not full V10 acceptance, frozen PRE, run authority, performance evidence,
or a physical F1 improvement. No new benchmark, changed cap or broad redesign
is requested by this delta.

## Final chunk/disclosure snapshot check — 2026-09-13 15:46 UTC

At root's request, read only the revised chunk-test source and V5 allowance
documentation. No tests/imports or physical/control/cache artifacts were
accessed. Before/after pins were identical:

```text
2afa4f69aee4addcd993137aeb97a8309fca2b5007041cbf754290109c80e3e7 tests/test_cad_guided_plane_initialization_v16_allowance.py
30e66d9c5cdc05d830a635557b24c710d2d16072758ab6c95e4aa954d27ddb2c backend/cad_guided_global_registration_v5.py
```

- **Multi-chunk coverage is now nonvacuous.** Allowance test 508–543 tiles an
  existing 25-row support to 1500 rows and invokes the real completion-support
  evaluator and metered finite-face query. It requires screen lengths
  `[1024, 476]`, finite-query lengths `[1024, 476]`, all 1500 rows completed by
  each operation, and exactly 3000 attempted/completed charged rows. The
  unchanged small full-lane test is now named/scoped accurately (489–505).
  Repeated rows are adequate for this numeric chunk/accounting contract, not
  independent scan geometry, additional physical evidence or coverage truth.
- **Between-chunk expiry remains untested in this fixture.** Its injected clock
  remains fixed at zero and every chunk succeeds. The separate query admission,
  candidate-end and between-face-build stop tests do not invoke an expiry
  between two chunks of one completion support. The already-requested smallest
  remaining acceptance is to reuse this fixture, advance the guard after the
  first successful screen or finite-query chunk, and require that the next
  primitive does not execute, exact attempted/completed counts are retained,
  and no partial support/child is published as passed. No larger scene is needed.
- **Atomic-work disclosure is corrected, row-allocation wording is not yet.**
  V5:101–113 now enumerates stop points, explicitly lists uninterruptible
  preparation/index builds, and states that headroom is not guaranteed in
  seconds. V5:96–97 still includes enumeration in `completion_query_rows`.
  Enumeration/binding/build consume metered process CPU but no query rows;
  exclude them from that row-charge sentence. Root has already requested this
  author edit; this snapshot does not chase or qualify a later revision.

This closes the missing long-support split coverage and the largest-atomic-work
wording issue at source. The two exact residuals above remain; executed author
receipts must match the final sources. Root owns the contradictory-whole-set
consumer/binder regression and reports that the old full consumer refused it,
so this review does not claim an accepted-prediction bypass. No PRE or broader
engine/physical-score acceptance follows from this narrow check.

## Final B2/B4 residual closure — 2026-09-13 15:59 UTC

**ACCEPT for this bounded B2/B4 source-review seam. No remaining finding from
this review.** This supersedes only the residual conditions in the preceding
deltas; all earlier findings/snapshots remain preserved. It does not constitute
full V10 acceptance, a frozen PRE, physical access or launch authority.

Read the complete author's final closure checkpoint and exactly the two
requested residual source seams. Before/after hashes were identical and match
the final closure table:

```text
ff6c94daa686b72335562888f0594a9218d3d369c103e94b41e4cd6c71b49e5b docs/research/2026-09-13_fable51_v10_review_response.md
ee05a03eb564c9fcc8c3cee631e4f429d0c66d9ec5468bc932377368105750e7 backend/cad_guided_global_registration_v5.py
cc34fc7f9f63b18ba54b596edba7147985327efc199520fa6cd9562471b96e65 tests/test_cad_guided_plane_initialization_v16_allowance.py
7dad12ae569c2044681864bb136105e0fea8750a4c1e454797a9f9fd874cb63b backend/cad_guided_plane_initialization_v16.py
```

The initializer is unchanged from the accepted original-five-defect delta.
The author has paused edits for closure review. This is a stable review
snapshot, not an experiment freeze; a later changed source is outside the pin.

1. **Between-chunk typed expiry: closed.** The new parameterized test at
   allowance.py:574–617 exercises both `first_screen_chunk` and
   `first_query_chunk` through the real evaluator and metered face queries on
   the 1500-row numeric fixture (546–571). The clock changes only after the
   selected first primitive has successfully returned. The next operation
   raises the typed CPU-guard exception. The screen variant requires only
   `[1024]` screen rows, no finite query, 1024 attempted/completed rows and one
   attempted/completed call. The finite-query variant requires full
   `[1024, 476]` screening, only `[1024]` queried, 2524 attempted/completed rows
   and three attempted/completed calls. Both require incomplete support/child,
   zero evaluated/passing hypotheses, a tripped guard and conversion by the
   actual rollback helper to `EXPIRED_BY_ALLOWANCE_UNKNOWN`. No partial child
   becomes PASSED. The prior full-lane typed rollback and real-binder regression
   supply the connected transaction evidence; this added fixture deliberately
   isolates the intra-support boundary, not geometry generalization.
2. **Query-row scope wording: closed.** V5:96–101 now explicitly limits the
   completion row allocation to box screens, finite-face queries and refinement
   queries, excluding enumeration, support binding and face-index builds from
   row charges and assigning their stop coverage to CPU phase checks. The
   explicit timing-dependent guard, uninterruptible preparation/build caveats
   and nonguaranteed headroom remain at 103–115. No increased caps or promised
   elapsed-time bound are inferred.

Executed evidence is **author-reported**, not rerun or independently observed
by this source reviewer: the hash-bound closure records the final 13-suite
command, 885 passed / 1 skipped / 0 failed, pytest process exit 0, from
15:52:07 to 15:55:11 UTC. Both new parameter values are named in its final
closure matrix. The single skip is identified there as pre-existing; it does
not condition away these new assertions. Root owns reconciliation of that
receipt and the remaining consumer/runner seams.

No test/import, raw/Gold/physical/attempt/control/cache access, engine change,
budget change or experiment execution occurred in this review. Only this
append-only audit was edited. No physical score, speedup or measurement-accuracy
claim is made.

# Gold11 V6 runner: unreleased source draft

Date: 2026-09-12. This is a design/draft handoff, not a freeze, protocol release,
PRE, test result, physical-input permission, or launch decision. Only the new
runner and this document were authored for this subtask. Frozen V5 remains
terminal, unchanged and unscored. No physical/private/attempt-control artifact
was read. No target import, compilation, collection, test, process observation,
native operation, or runner invocation occurred.

## Release and preserved scientific boundary

`scripts/run_gold11_mesh_development_v6.py` is separately versioned. Its literal
RELEASE_AUTHORIZED is False, and TIMING_RELEASE_STATUS is
UNRELEASED_V5_CARRYOVER_VALUES_NOT_V6_APPROVAL. require_release rejects before
process entrypoint source/control/input access and additionally rejects that
unreleased timing declaration even if the boolean alone changes. There is no
environment override. Freeze, execute, inference worker, scorer worker, internal
coordinator and main are blocked; new direct supervision, per-case, policy
constructor, reference-selection and post-terminal scoring seams also require
release before their scoped work.

Inherited numerical LIMITS are retained solely as draft values. They are not
newly qualified caps or authority for another 11-case run. The separate future
physical protocol path remains declared but unimplemented; this design is not
its replacement. No source freeze or attempt directory was created.

Textual source comparisons, without executing Python, found these preserved:

- The complete CONFIG expression and complete LIMITS expression are identical
  to frozen V5. Numeric measurement/registration/search budgets, provisional
  position-error declaration and existing certificate caps remain unchanged.
- Returned workflow-stage diagnostics are source-identical. Missing/killed
  internal stages remain UNKNOWN; this draft adds no inferred stage attribution.
- Reference selection is source-identical except for an added release guard.
  The same review source/hash, catalog hash, selected71 hash, eleven scans,
  six CAD models (17 raw inputs), eleven cases and both arms remain declared.
- The contact scorer, matcher and physical evaluation contract sources are
  unchanged. No fitted values, endpoints, carriers, unit interpretation, scoring
  threshold or scientific denominator is repaired or relaxed by the runner.

Fresh completed/refused case arms still have all 22 fixed terminal positions.
Failed arms contribute zero usable predictions rather than disappearing.
Only all22 durable reconstructed terminals can precede selected71 reference
access. No score follows a coordinator abort or unconfirmed leaf custody.

## Explicit successor integration

Both arms now request initializer V13/context V3 through workflow V14:

| Arm | Scan-plane mechanism | Schedule | Context | Native adapter |
| --- | --- | --- | --- | --- |
| control | v7 | v1, original effective defaults | explicit v3 | v1 |
| challenger | composite_v1 | v2, unchanged residual budgets | explicit v3 | v1 |

The comparator is
FRESH_V14_CONTEXT_V3_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1.
This is not a historical paired-control replay. V14 retains V13 wire transport
for the unchanged measurement/scorer consumers, while mandatory distinct
workflow/initializer implementation identities and complete context policy
declarations expose the changed implementation honestly.

CONTEXT_V3_REQUEST independently declares all twelve configuration keys,
23 exact integer budgets and eight arithmetic/scope declarations. Both arm
requests contain separate JSON-stable copies. The contextual maxima include
129 indexes, four million total indexed memberships, sixteen million node
visits, 256 million coordinate units, and 3072 extrema queries; the existing
64M scan/index visit, 1024-family, 8192-slot, 256MiB accounted working and 32MiB
report limits remain in the explicit context policy. These new context guards
are prospective engineering limits, not old V1/V2 invariants or RSS limits.

workflow_entry_and_policies constructs the real TwoPlaneContextBudgetsV3 from
the requested map and verifies the initializer V13 configuration snapshot
against the complete independent declaration before scientific input reads.
It passes real two_plane_context_mode and two_plane_context_budgets keywords to
inspect_cad_guided_mesh_v14. The numerical PlaneInitializationBudgetsV12 type
is reused explicitly because V13 reuses that frozen type; no invented V13 type
or alternate schema is fabricated. Remaining arm policies/schedules retain
their original constructors and effective values.

The frozen-buffer projection module graph now adds schedule projection V2 after
its V1 dependency. projection_request passes the full expected_context_request
alongside the existing complete scan/schedule request. Projection and retained
reconstruction use the V2 APIs on the original RESULT bytes. The outer wrapper
schema is GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V2; the untouched
inner contact projection remains the input to the unchanged scorer. New runner
request/control schemas are V6; no arbitrary wrapper-unwrapping path was added.

## Supervisor V3 and conservative coordinator memory

The runner compiles only the verified frozen supervisor V3 buffer and requires
the independently reported qualified source hash
9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea.
The signature is unchanged, so orchestration does not wrap numeric process
signalling or reinterpret V3 accounting. Frozen supervisor V3 remains untouched.
Root reports its accepted eight-case host POST separately from the 166-case
mock gate. Host success did not observe actual exit-transition events and does
not prove the old race's causal fix or qualify this new runner integration.

V1-only process_table/descendants functions are removed. Between leaves,
coordinator_memory_check uses Darwin resource.ru_maxrss as lifetime high-water
bytes, with platform/finite/nonnegative checks and the unchanged coordinator
limit. The returned scope explicitly says lifetime high water, not current RSS.
This is stricter/conservative after an earlier transient peak. Supervisor V3
still owns its separate sampled current tree/parent memory accounting; no
global process-table child is created by this coordinator check.

writers_quiescent is a strict V3 receipt predicate: exact supervisor schema,
direct child reaped, no attention, no observed group escape/unconfirmed escape,
no still-present unknown identity, empty cleanup/control-close errors, exact
integer zero discarded cleanup errors, and additionally owned group empty for
the outer new-session coordinator. It grants content-retention/cleanup order
only, never new process observation or signal authority. For a leaf, V3 reaps
only after its observed owned descendants are gone; this remains subject to
V3's documented inability to observe a descendant born/escaped/reparented
entirely between censuses. The runner does not convert sampling into OS
containment or an atomic hostile-filesystem guarantee.

## Fresh private cache custody and cleanup

The coordinator cache is now freshly created under explicit Darwin /private/tmp,
outside RUN. Its canonical path and device/inode/uid metadata are bound into
COORDINATOR_DISPATCH before launch. The coordinator checks the same private
directory identity, outside-attempt location, cwd, TMPDIR and HOME before
activation/leaf dispatch. The outer receipt retains only that already known
opaque metadata, never cache contents. The coordinator cache is retained on
success or failure and is not recursively inventoried or automatically removed.

The old declared coordinator_work inventory branch is removed. If such a name
or any other undeclared cache appears under the new attempt, the complete-tree
inventory still vetoes it as unknown metadata; nothing is ignored or promoted
into an allowed namespace. No old cache path is followed. Predispatch checks
continue to reject spent/terminal names and no retry is exposed.

Each inference/scorer cache uses explicit mkdtemp custody rather than a
TemporaryDirectory finalizer. The initial canonical private directory identity
is retained before dispatch. The supervisor wrapper marks possible outstanding
leaf writers before calling V3, and clears that state only from a strict
quiescent V3 receipt. An exception or absent receipt leaves the state unknown.

retain_worker_result checks strict quiescence before any cache path enumeration,
stat or result content read. It then requires the original private cache path
and identity before looking for its one complete/partial derived result. No
partial is parsed as a successful result, and no raw input is copied into the
retained attempt. A failed retention leaves the private cache opaque and intact
apart from any already completed operation; no cleanup follows it.

The cache context's final block may call shutil.rmtree only for the exact fresh
owned directory after identity revalidation and either proven absence of any
dispatch or strict writer quiescence plus successful derived-result retention.
It uses no cleanup glob and no automatic garbage-collection finalizer. On
uncertain writers or failed retention, it records an explicit opaque-retained
status without any further enumeration/content reads. A cleanup exception
records possible partial cleanup honestly, keeps remaining private contents,
and vetoes subsequent inference/scoring. There is no claim that an interrupted
recursive deletion is atomic or fully reversible.

Inference receipts retain private-cache status and cleanup completion. Scorer
receipts are published from an outer finally after cache handling, including
failure metadata even when retention/supervision fails. Metadata publication
does not authorize a private content read. Publication failure still propagates
to terminal failure handling; it is not repaired by deleting the cache.

When the coordinator is killed before receiving an inner receipt, those leaf
caches have no destructor and remain outside RUN. Their issued neutral dispatch
records retain original path/identity for custody; the outer runner does not
reopen them after the abort. A later separately authorized cleanup decision is
required. Successful outer quiescence does not retroactively authorize this
draft to discover or reclaim those private caches.

## Failure ordering and whole denominator

After every attempted case, uncertain leaf custody, failed retention or failed
cache cleanup aborts further dispatch and scoring. The coordinator sets an
explicit abort state. If leaf writers remain unconfirmed, its error branch
does not terminal-fill case files; it reports the complete 22-position list as
not yet durable and defers filling to the outer coordinator supervisor.

The outer execute path now computes strict whole-group quiescence first. Only
then may it perform source/runtime/control content postflight, terminal filling,
artifact verification or complete-tree inventory. Source postflight failures
remain failures while terminal materialization can still preserve all22 slots
under its remaining deadline. Without quiescence, no such content reads or
terminal fill occur; the outer administrative terminal receipt explicitly
reports nondurable positions and unknown writer custody. No later scoring call
exists in the outer fallback.

Existing files are never overwritten during terminal fill. Unknown names still
charge the original scientific128 limit and veto the full inventory. The
expanded source closure is derived into the same separate schema metadata
limits; it does not raise scientific128, source file caps, wrapper caps or the
256KiB control envelope. Maximum finalization/envelope capacity is still an
unqualified draft blocker, not assumed safe merely because source was copied.

## Source-only stage telemetry connection; physical work still unreleased

Root added a source-only `engine_call_with_stage_sampler` seam. The inference
leaf source closure includes the exact sampler; after authorize_leaf and before
installing the backend-only filesystem hook, the worker compiles its pinned
source buffer and records the verified module identity. This pre-hook ordering
repairs an independently found initial wiring error: reopening a scripts path
after the backend-only hook would be denied. The hook is not broadened. The
call wrapper requires that verified preload before any sampler or engine call,
then
starts the bounded observational sampler immediately around the one unchanged
engine call and finishes it in finally. It passes only the explicit frozen
backend source paths plus this runner to the sampler, never scan/CAD objects or
frame locals. Thread construction/start/finish errors remain separate diagnostic
metadata; they neither replace the engine's return object nor mask its exception.
The exact source/hash check itself is mandatory custody and fails before calling
the engine. No callbacks were added to frozen mathematics or native consumers.

At most524288 sample bytes plus4096 terminal-receipt bytes (including framing
and both newlines) leave520192 bytes for ordinary output under the unchanged
1048576-byte complete leaf log. This arithmetic is a declaration, not a separate
native-output quota: the existing whole-log supervisor cap remains authoritative
and an overrun is a terminal failure. A too-large terminal diagnostic is omitted
whole into a short refusal; a failed diagnostic write is not promoted into a
complete receipt. Any still-live sampler is reported unknown/incomplete until
the external supervisor confirms worker quiescence. RESULT bytes and projection
are unchanged. The source seam and standalone sampler are pending exact runtime
qualification; no physical diagnostic is released. Coordinator output has its
own separate cap and is not a leaf-log allowance.

PENDING_THREE_CASE_DIAGNOSTIC is likewise unimplemented. A small consumed
three-case diagnostic must be a separate runner/protocol after V6 core source
checks, with independently frozen selection, purpose, limits, stdout/custody,
PRE and terminal rule. There is no new subset argument, three-case list,
physical path lookup or implicit reuse of V5 authority in this file. Passing
that future diagnostic would not automatically release an all11/both-arm run.

This design leaves source qualification, runtime/transport review, concrete
timing release, telemetry, diagnostic protocol, exact physical PRE/archive and
run authority outstanding. It does not propose another broad test suite here.

## Minimal bounded source-test plan (not authored or run)

Use one new isolated invented-only gate after independent review, not blanket
pytest or an old qualified suite rerun. Keep actual process/native/physical
operations mocked and fixture bytes small. The minimal assertions are:

1. Static preservation of CONFIG, inherited scientific caps, complete reference
   selector aside from release guard, returned-stage diagnostics and fixed
   all11/both-arm loops. Both unreleased guards reject every process entrypoint
   before source/control/input/import hooks; no old attempt or cache is touched.
2. Two actual policy-construction request controls using inert mocked backend
   constructors: explicit V3 mode/budgets in both arms, correct V14 callable,
   unchanged v1/v2 schedule mechanisms, exact independent context declaration,
   ordered frozen projection graph and full V2 reconstruction request.
3. Strict quiescence mutation table: missing/wrong/bool fields, unreaped child,
   unknown identity, escape, cleanup/control errors and outer nonempty group
   refuse before private enumeration/read/cleanup. Clean leaf and clean outer
   receipts admit their different group predicates.
4. Fresh cache identity/cwd/TMPDIR/HOME binding, no attempt-contained cache,
   changed inode/private mode/symlink refusal. Cleanup spies verify no call on
   uncertain writer or failed retention, exact owned-path removal only after
   quiescence and retained result, and explicit cleanup-failure veto/status.
5. Inference/scorer supervisor exceptions and missing receipts retain opaque
   caches. The scorer failure receipt remains attempted independently of its
   derived-result failure. No later inference, reference read or scoring occurs.
6. Order spies show outer source/control postflight, terminal fill and inventory
   occur only after whole-group quiescence. Unknown writer paths yield all22
   nondurable records, then stop; a confirmed killed coordinator may fill all22
   without scoring, and existing terminals are never overwritten.
7. New external coordinator cache is not a schema branch. Any unexpected name
   under RUN still causes a metadata-only veto; source closure expansion cannot
   silently enlarge the scientific128 or 256KiB envelope limits.
8. Darwin high-water checks use ru_maxrss only, enforce unchanged limits, reject
   invalid/platform cases, and never call process_table/descendants or label
   high water as current RSS. The new telemetry wrapper preserves exactly one
   engine call/result/exception under startup/finish/write failures and frames
   only bounded separate log metadata. Physical diagnostic selection stays absent.

Freeze a small exact denominator and resource envelope only after writing and
reviewing those focused tests. No denominator or pass result is claimed here.

## Source-only handoff

Agent's initial draft runner (superseded by root's unexecuted telemetry wiring):
162,286 bytes, SHA256
c292ca6a776169fa6a1cda879ec0d1fb7cb8ad7aab3294175387b3eb4a8e2133.
Frozen V5 source remains 153,975 bytes, SHA256
71d55915db6e6fcb8ae7f8c658de7d7e0c4b93b553eb78413f3914b037e4e481.
These are source identities, not attempted-run evidence. The final handoff must
report updated hashes if any review-driven edit follows this checkpoint.

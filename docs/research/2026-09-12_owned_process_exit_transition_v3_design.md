# Owned-process exit transition V3 — source-only design

Date: 2026-09-12. Status: NEW / UNQUALIFIED. This document accompanies
`scripts/gold11_owned_process_supervision_v3.py`. It does not authorize an import,
test, native process query, signal, host qualification, or physical input run.
V2 and every terminal attempt remain unchanged and terminal. Independent source
review, separately authored adversarial tests, and any exact protocol/PRE/archive
required for execution remain future work. No new result or scientific score is
claimed.

## Retained evidence and bounded inference

Only the explicitly released terminal `CASE_03_RESULT.json` and
`CASE_04_RESULT.json` under
`docs/research/cad_engine_v1_synthetic_checks/owned_process_supervision_v2_host-cpab0qfk/`
were read for this diagnosis. Root reports the actual eight-case host disposition
as six PASS and two FAIL; this source change cannot revise that result.

CASE_03, `outer_exec`, retained an exit-zero child, one exec-version update,
four coherence retries, and a transient unknown identity for PID17028. Cleanup
recorded a token-stale-or-exit observation, then reaped the child and confirmed
the group empty. CASE_04, `outer_waited_descendant`, retained the inspector failure
on descendant PID17030. Cleanup accepted SIGKILL against direct child PID17029,
whose exit code was -9; the descendant was later verified absent and the group
confirmed empty. Both retained `PROCESS_IDENTITY_OR_RSS_DID_NOT_STABILIZE`.

Those receipts do not preserve the failed TASKINFO size/errno, the two BSD rows,
or INEXIT. They therefore do not prove the precise native sequence. The proposed
ordinary-exit mechanism is a source-supported inference, not retrospective
instrumentation or a claim that a specific race was directly measured.

Apple's current source permits the inferred sequence. `proc_prepareexit` moves
the process to `zombproc` and sets `P_REF_DEAD` before `proc_exit` later writes
`p_stat=SZOMB`; exit work between these points can block.
[Apple kern_exit.c](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_exit.c)

`proc_find` rejects dead references; `proc_find_zombref` instead checks the
retained exited process. This distinction is about lookup lifecycle, not proof
that every task resource has already been destroyed.
[Apple kern_proc.c](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_proc.c)

Flavor18 with arg1 enables zombie-inclusive lookup, whereas TASKINFO does not.
BSD information copies `p_stat` directly and separately reports INEXIT from
`P_LEXIT`. Coherent non-SZOMB identity can consequently coexist with TASKINFO
ESRCH during exit.
[Apple proc_info.c](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/proc_info.c)

The public `PROC_FLAG_INEXIT` value is4. It is evidence of exit underway, never
evidence of completed exit, zero resident memory, or permission to reap.
[Apple proc_info.h](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info.h)
The existing unique-id/version structure remains a separately host-qualified
private interface.
[Apple proc_info_private.h](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info_private.h)
These moving-source references are not a match attestation for the exact host
kernel build, nor a guarantee that exit resolves within one second.

## Exact V3 change boundary

V3 is a separate copy of V2. The native ABI structures, `_coherent_key`, lifetime
comparison, PID enumeration, native-call accounting, sample deadline calculation,
custody admission/retirement, ownership and ancestry checks, memory arithmetic,
subprocess launch, output handling, cleanup, direct-child anchor, and token-based
signal math are unchanged. V3 adds flags to the decoded BSD row, an inspector
transition branch, bounded diagnostic bookkeeping, a typed transition deadline
error, and the corresponding V3 receipt metadata. No runner imports V3 through
this change. No existing qualified source is edited.

Flags are decoded from an existing unsigned32 field in the exact-size BSD record.
No flags are added to `_coherent_key`. Arbitrary mutable flag changes therefore
do not become identity changes. No validation, coherence field, permission check,
or process-row trust condition is removed.

## Observation states and branch predicates

Every iteration performs the existing first BSD read, optional TASKINFO read,
and second BSD read. Acceptance remains limited to the existing categories:

1. Explicit ESRCH from zombie-inclusive flavor18 returns absence. The caller
   still refuses a missing unreaped direct-child anchor; descendant retirement
   remains subject to the existing custody code.
2. An equal full `_coherent_key` pair, non-SZOMB, and exact-size TASKINFO yields
   the current bracketed resident size when RSS was requested.
3. An equal full `_coherent_key` zombie pair yields RSS `None`, not a fabricated
   measurement. Existing zombie/group/anchor rules apply.
4. With `rss=False`, the original coherent identity-only result remains available
   to callers such as signal authorization. The new transition branch is not
   reachable from that path because no task-missing observation is made.

The new wait branch requires all of the following in the same iteration:

- TASKINFO returned exactly size0 and errnoESRCH, with RSS requested.
- Both BSD reads are valid and non-SZOMB.
- Their existing full coherent keys compare equal: PID, uniqueid, idversion,
  PPID, parent_uniqueid, PGID, credentials, start time and zombie discriminator.
- The second BSD record carries `PROC_FLAG_INEXIT`.

This branch returns no row and makes no custody mutation. It records entry once
per `inspect` invocation, yields for at most1ms and the remaining sample time,
checks the clock after the yield, then performs the complete read sequence again.
It is a bounded opportunity to obtain acceptable evidence, not a fourth accepted
RSS/identity result.

## Unchanged limits and persistent incoherence allowance

`COHERENCE_ATTEMPTS` stays2. V3 counts iterations that fail ordinary acceptance
and do not meet the narrow wait predicate. The count is initialized once at
`inspect` entry and never reset by a coherent exit-transition observation. For
example, mismatch, eligible wait, mismatch still fails after two mismatches.
The wait branch does not consume that mismatch count because the identity pair
was coherent, but it consumes real sample time and all actual native calls.

V3 never calls `begin_sample` from `inspect`, creates a per-PID deadline, adds a
grace deadline, or replenishes work/cleanup call budgets. The existing sample
deadline is `min(caller_end, begin_sample_time+1 second)`. All repeated reads
continue through `_call`, including its precharged per-sample and cumulative
native-call limits and before/after clock checks. Multiple inspected processes
share the same sample deadline; one slow transition cannot grant another more
time. The fixed1ms value is pacing, not an empirical success threshold.

Native calls and OS scheduling are not cancellable hard deadlines. A late
return or oversleep fails the existing clock check; V3 cannot promise exact
wall-clock cancellation. A transition may legitimately remain unresolved when
the unchanged budget expires, including during blocking kernel exit work.

The mismatch allowance is not lifetime authorization. Every accepted replacement
row still goes through the existing caller generation/ownership checks. No
previous RSS value is carried into another iteration or generation. A changed
coherent replacement may be returned only through the ordinary acceptance path;
there is no new signal, retirement or admission shortcut.

## Failure semantics and retained diagnostics

Short/malformed/oversized records, permission errors, unsupported symbols and
other non-ESRCH native failures retain V2's fail-closed behavior. A size0 result
with errno0 is not absence. Coherent TASKINFO ESRCH without INEXIT does not gain
the wait exemption. Live RSS is never substituted with zero, a previous sample,
or an omitted row. No partial snapshot is accepted while a transition is pending.

After transition entry, any normal Exception records an unresolved outcome.
If that exception is a TimeoutError, `_ExitTransitionDeadline`, a TimeoutError
subclass, is raised with the original exception chained and its bounded textual
cause retained. Its message begins `EXIT_TRANSITION_UNRESOLVED_DEADLINE:`.
Non-timeout failures keep their original exception type. Existing outer
failure/cleanup orchestration handles these errors; nothing suppresses a prior
budget stop, true ownership failure, escape, or unresolved cleanup state.

`native_process_work` remains a numeric dictionary. Added counters are:

- `task_rss_esrch_observations`;
- `exit_transition_entries`, `exit_transition_waits`;
- `exit_transition_wait_seconds`, the cumulative measured duration of the sleep
  calls, including oversleep, not a replacement sample deadline;
- `exit_transition_resolved_rss`, `exit_transition_resolved_zombie`,
  `exit_transition_resolved_absence`;
- `exit_transition_unresolved` and
  `exit_transition_diagnostics_discarded`.

The receipt separately publishes `exit_transition_diagnostics`, retaining the
first16 ENTERED/resolved/unresolved events, and `exit_transition_first_failure`,
retaining the first unresolved event independently of routine-history saturation.
Each event has fixed-field before/after identity views, task size/errno, phase,
sample native-call count, elapsed transition time and a cause of at most160
characters. A resolved or unresolved event also retains the initiating ENTERED
event in its `entry` field; ENTERED itself has `entry=None`. Thus nesting depth
is fixed, and the initiating size0/ESRCH/INEXIT evidence survives a later failure
before a fresh BSD row can be read. No executable names, arguments, environment,
raw science data, or unbounded row histories are added.

Initialization for independently authored mocks must mirror the new numeric
work fields, diagnostic list, first-failure field, and flags-bearing BSD rows.
Tests constructing `_MacOSProcessAPI` via `__new__` must initialize these fields
explicitly; actual code is not weakened to accommodate incomplete old fakes.

## Required prospective verification

Independent source review and additive invented tests must cover at least:

- More than two eligible observations followed by a coherent zombie, verified
  absence, or fresh exact RSS; no signal or fake zero from waiting.
- Persistent mismatch counts across waits, changed generations/versions,
  ownership/ancestry/credentials/start-time changes, and unchanged group escape
  and replacement retirement behavior.
- ESRCH without INEXIT, all malformed/short/error cases, a real measured zero,
  oversized resident memory, and no reuse of a previous RSS observation.
- Deadline expiration before/after a native call and during sleep; unresolved
  cause and initiating evidence retained; call caps precharged without a real
  excess call; no sample renewal between PIDs or between wait observations.
- History saturation plus independent first-failure retention, fixed nesting,
  discard accounting, and numeric-only `native_process_work`.
- Unchanged `rss=False`, token-stale/exit semantics, sole-waiter anchor,
  cleanup, no signal after reap, log accounting and parent-inclusive RSS scope.

Existing regression coverage must be separately adapted only for explicit V3
row/receipt contracts, then run solely under new exact authorization. No test
count or success is inferred from this document or the existence of source.

A future independently frozen native gate should include ordinary exit,
exec-then-exit and a parent that waits for its descendant, while retaining the
other custody/cleanup controls. Prospective expectation: a transition resolving
within existing budgets should not trigger supervisor-induced SIGKILL. A native
run might observe zero eligible transitions; that is not deterministic branch
coverage, so mock evidence remains necessary. Never repeat the terminal V2
attempt or increase its limits to obtain a passing result.

## Handoff

This is source-only engineering. No target import, syntax execution, test
collection, native call, process observation or signal was performed by this
author. Only the new V3 source and this new design document are authorized edits.
Independent tests/review and the required exact gates remain outstanding before
any qualification or integration claim.

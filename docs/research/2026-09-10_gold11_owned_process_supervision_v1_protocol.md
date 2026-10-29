# Owned-process supervision V1: independent invented qualification

Date:2026-09-10. Stdlib/source/table/mock tests and one separately opted-in
same-process-group child. No real runner, input/reference/cache/Gold access,
inference, native geometry, new process-group stress or old-attempt reuse.

## Mechanism and limits of the claim

New helper `scripts/gold11_owned_process_supervision_v1.py` replaces the old
descendants-only supervisor for newly versioned runs. It accepts explicitly
validated finite numeric limits; has no repository/data constants or import-time
file access; and is loaded by callers from exact verified frozen source bytes.

For an outer new-session child, sampled custody is the union of its owned PGID,
rooted descendants and retained observed PID/PGID identities. Reparenting after
leader exit must not turn an owned survivor into zero-RSS success. Normal return
requires the direct child reaped and observed owned group empty. Foreign-group
escape causes a permanent refusal, is counted/flagged separately and is never
group-signaled; own-group emptiness does not prove escaped descendants removed.

An inner child shares its coordinator group. Never kill the whole coordinator
group or claim it should be empty. Its direct child/observed descendants are the
scope; create/reparent-before-observation, hostile native group escape and PID
reuse remain limitations. No unrelated process inventory is retained.

Missing/malformed monitoring is failure, not a fabricated zero. A genuinely
short-lived child may finish before a sample; report the unobserved child peak
as unknown with explicit sample counts. Caller worker/reaped high-water evidence
is separate. Sampling is not a hard OS allocation cap. Control-close failure must
not bypass cleanup, and an ordinary cleanup/poll/selector/stdout error vetoes
success with retained evidence. Observed and stored log bytes remain distinct.

The helper relies on its caller's absolute deadline for synchronous Popen/stdin/
log operations. Failed-only kill/reap observation is bounded to five additional
seconds, included in its returned wall time and within the caller's reserved
deadline. No self-contained hard-wall/native-containment/persistence guarantee.

## Independent source/mock matrix

Independent test file: `tests/test_gold11_owned_process_supervision_v1.py`.
The frozen matrix has104prospective outcomes:103source checks plus one declared
smoke skip, then104checks with the single actual child enabled. The exact source
closure is bound in the independent pre-run audit before first execution.
Tests must exercise real helper code with
invented process-table/clock/pipe/process doubles, not reproduce its algorithm
as the test result. Required controls include normal and fast completion, owned
PGID survivor after leader+EOF, unrelated groups, observed foreign escape,
monitor/cleanup/control-close faults, exact budgets and log retention. The matrix
also covers both previously owned and previously foreign observed PID/PGID
identities changing after reparenting, plus write/partial-write/flush uncertainty.
First observed identities stay fixed; changed identities permanently veto a clean
result, remain conservatively accounted while present, and grant no new signal
authority. Failed log storage must not erase bytes observed from the child or
assert an invented exact stored length. Every
fault has a retained refusal/unknown outcome, not an omitted test.

## One actual same-group invented child

The optional child uses only stdlib, sets its own2sCPU and3s alarm before reading
a tiny invented request, emits a tiny result and exits. Invoke the actual helper
with `new_session=False`, wall5s, log64KiB and1GiB sampled tree. It remains inside
the outer test worker's process group; no separate group/orphan stress is run.
The helper's failed cleanup is additionally bounded5s. The complete bounded
child log and compact receipt are forwarded before assertions so failures remain
visible even if the temporary invented directory is removed.

This checks actual pipes/monitor/reap/recording for the inner mode. Outer group
and reparenting branches are independently mocked, not claimed actual orphan
qualification. No caller authorization, cached scoring or actual runner CLI is
exercised. The source route never executes a subprocess.

## Freeze, serial execution and advancement

Freeze helper, independent test, this protocol and the shared harness together
(plus any explicitly read source oracle added by the independent tests). Exact
pre-audit and GREEN required; no source edit between source and smoke or parallel
shared-closure gates. Preserve earlier harness snapshots and all terminal tests.

Source gate once:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_owned_process_supervision_v1_source`

Only after independent retained source post-GREEN, smoke once:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_owned_process_supervision_v1_smoke`

Each outer gate60s monitored wall/45s workerCPU/1GiB sampled groupRSS/1MiB output,
isolated HOME/TMPDIR and one-thread environment. ChildCPU is separately bounded,
not added to the worker's own45s limit; group sampling includes both. Retain
STARTED/OUTPUT/COMPLETED, all outcomes and exact before/after source hashes.
Any failure, unexpected skip, missing outcome, drift or stop remains terminal;
no changed cap, oracle, fixture or retry. Independent post-audit precedes release.

Passing qualifies this process-supervision scope only. Cached runtime V2 still
needs its own source/guard tests and exact actual-run bundle audit. Physical V3
needs its own workflow/runner qualification and source/config/reference freeze.
No F1, geometry accuracy, historical comparison or production/floor release claim.

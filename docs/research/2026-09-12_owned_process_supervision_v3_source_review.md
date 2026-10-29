# Owned-process supervision V3 independent source review

Date: 2026-09-12. Reviewer: independent AI technical reviewer, not a human
approver. Decision: SOURCE-REVIEW GREEN for the amended mock qualification
candidate below. This document is not exact PRE, execution authority, a test
result, host qualification, integration promotion or physical accuracy evidence.
There are 166 prospective outcomes and zero V3 outcomes executed at this review.

## Scope and exact candidate

The reviewer read the complete V3 supervisor, complete 1,231-line test file,
complete mock harness and protocol, and complete exit-transition design. The
previously fully reviewed unchanged V12 short supervisor, tests initializer and
released V2 host failure POST were byte-verified again. Full V2-to-V3 source and
test deltas were reviewed against the previously fully read V2 baseline. No
target import, test collection, test execution, native process API, signal,
host PID observation, old attempt reopening or private-cache read occurred.
Only isolated standard-library source parsing and hashing were performed.

The following eight files will join this new review in the nine-source closure:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| scripts/check_owned_process_supervision_v3.py | 22940 | 8f666fd22f98819412a8672a898f97101a57bb970f52bed7aff64ace3b079e73 |
| scripts/check_cad_guided_connected_v12.py | 19437 | 0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694 |
| scripts/gold11_owned_process_supervision_v3.py | 42751 | 9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea |
| tests/test_gold11_owned_process_supervision_v3.py | 57565 | d1b98c2d3cc8dbc7240103b667e4e5cc586347cd7755fa687853b411360c8ca8 |
| tests/__init__.py | 24 | 5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b |
| docs/research/2026-09-12_owned_process_exit_transition_v3_design.md | 12535 | 71452356e24219069c3ac6f67b06c7d519987d78e35204c2aee47cf8b8a9992d |
| docs/research/2026-09-12_owned_process_supervision_v2_host_post_run_audit.md | 13451 | ce4dcbc0f627b7b6b0f6c1288c411c6621fb77b5d9388b78ce052c1e764f24c7 |
| docs/research/2026-09-12_owned_process_supervision_v3_source_protocol.md | 7721 | 6b9929ca5bb40a7dda04607d1e847678c69220c597ec9ac463bfbc7505ae679e |

One concrete preflight blocker was identified and repaired by the source owner
before any execution: the new harness enumerated nine sources but its inherited
`_sources` assertion still required seven. The final 22940-byte harness requires
exactly nine and names `SOURCE_CLOSURE_NOT_NINE`. The full final delta was read;
there is no remaining blocking source finding for this narrowly scoped mock
gate. Exact PRE must independently bind this review and the final runtime.

## Mechanism and preserved failure evidence

V2's sole installed-host experiment remains terminal FAIL, six of eight cases
passing. Ordinary exec and a parent waiting for its descendant failed with
PROCESS_IDENTITY_OR_RSS_DID_NOT_STABILIZE and historical unknown-identity peaks.
Both eventually became quiescent, but cleanup also retained failures, and the
waiting parent was killed. The earlier mock PASS did not predict this host
behavior. V3 cannot rewrite these receipts or promote V2 retrospectively.

The proposed cause is plausible rather than observed in those failed native
brackets. The retained V2 evidence lacks the TASKINFO size/errno and both BSD
rows including INEXIT. Apple source distinguishes ordinary process lookup from
zombie-inclusive lookup and separates exit-in-progress from SZOMB publication.
This supports an intermediate coherent non-zombie identity with task lookup
ESRCH, but does not prove its occurrence on these exact cases or this kernel
build. The primary-source review used the linked Apple sources; moving main is
not a host-build attestation. See [Apple proc_info.c](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/proc_info.c),
[kern_exit.c](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_exit.c),
[kern_proc.c](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_proc.c)
and [proc_info.h](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info.h).

V3 decodes the existing unsigned flags field and uses INEXIT bit4 only as
permission for another bounded observation, not as exit, RSS, ownership or
signal evidence. The ABI structures and sizes remain unchanged. Flags are not
added to the coherent identity key, so unrelated mutable flag transitions do
not become generation failures. All original identity-key fields and the
zombie discriminator remain mandatory.

The added branch requires RSS requested, exact TASKINFO size0/ESRCH, two valid
non-zombie BSD records with equal complete keys, and INEXIT in the second row.
It returns nothing and changes no custody. Every retry reconstructs the full
bracket; resident memory is initialized afresh. Only ordinary exact measured
RSS, coherent zombie evidence or explicit zombie-inclusive absence resolves
the observation. A real measured zero remains valid; missing RSS is not zero.
Changed generations may return only through the old ordinary acceptance path
and still face the unchanged caller's lifetime/admission checks.

The wait preserves the current sample deadline and all cumulative native caps.
It does not call begin_sample, reset mismatches or grant a per-PID extension.
The sleep is min(1ms, remaining sample time), with pre/post time checks and
actual sleep duration accounting. Native calls remain precharged and checked
both before and after invocation. Two genuine unresolved/mismatched iterations
still fail even when eligible coherent waits occur between them. Multiple PIDs
share one sample end; the existing at-most-one-second sample horizon remains.
Late OS returns and oversleep cause failure, not accepted late measurements.
No hard cancellation of synchronous kernel or filesystem I/O is claimed.

## Generation, signal and lifecycle invariants

The complete source delta leaves `_Custody`, `_coherent_key`, `_lifetime`,
native enumeration, begin_sample/check_time/_call, memory arithmetic and token
signaling unchanged. Same-lifetime exec refreshes pidversion only after coherent
observation. Known reparented descendants remain known; fresh ancestry requires
matching parent unique identity. Verified absence/replacement may retire a
descendant, while unavailable identity remains unknown. A stale signal-token
ESRCH is ambiguous between exit and exec and still grants no retirement proof.

Signal authorization remains fresh, generation-aware and per-process; there
is no numeric kill, killpg, poll or fallback in the candidate. The root must be
authenticated and unreaped. Same-generation group escape remains failure, and
no foreign group target is signaled. Keeping the direct child unreaped anchors
its PID until observed descendants/group are empty. Reap releases that anchor,
after which only a read-only final group census is allowed. This is not atomic
group-membership authorization and cannot observe a child born, escaped and
reparented wholly between censuses.

Public cleanup still retains the first stop, historical unknown/error evidence,
and explicit attention when owned writers are not confirmed gone. A transition
timeout is not suppressed: after entry it becomes a typed TimeoutError subtype
with the original cause chained; non-timeout exceptions retain their type.
The normal public failure/cleanup path handles both. No new success exception,
silent omitted row, stale RSS acceptance or cleanup authority is introduced.

## Diagnostic boundedness and uncertainty

The numeric work dictionary gains task ESRCH, entry/wait/resolution/failure and
discard counters. Diagnostic records are separate from those numeric fields.
At most sixteen routine events are retained, plus one independently preserved
first unresolved event. Every event has fixed identity fields, task size/errno,
phase, elapsed time, sample calls and a cause truncated to160 characters. A
resolution/failure references one initiating ENTERED event whose entry is None;
there is no recursively growing event chain. Discard accounting remains visible.

The entry record preserves the initiating full before/after fields and exact
task evidence even if the next native call fails before returning a row. A
later unresolved diagnostic need not itself contain a valid current bracket;
its nested entry provides the earlier causal observation, not proof that the
same lifecycle state persisted continuously. Diagnostics neither expand signal
authority nor constitute a complete process history. Historical error evidence
is not cleared just because eventual cleanup succeeds.

## Independent test-source reconciliation

AST comparison confirmed all 51 original test function ASTs, including decorators
and assertions, are identical to V2. The supporting fake changes are restricted
to V3 source/module identity, flags-bearing rows, numeric counters, diagnostic
list and first-failure initialization. The prior127 outcomes are preserved.
Counting literal list/tuple lengths in parametrization ASTs, without evaluating
parameter expressions or collecting tests, independently yields64 functions,
166 total prospective outcomes and13 added functions/39 outcomes.

| Added control family | Outcomes |
| --- | ---: |
| Three eligible waits then fresh RSS zero/positive/uint64 maximum, zombie, first/second BSD absence | 6 |
| Eleven coherent-key mismatches retain two-mismatch allowance across waits | 11 |
| Second-row INEXIT and unrelated flags/runnable status | 4 |
| Denied, zero-without-errno, short, oversized and EIO task records | 5 |
| Original short sample deadline in work/cleanup | 2 |
| Oversleep reports actual duration and fails | 1 |
| Deadline before/after next native read retains initiating evidence | 2 |
| Precharged work/cleanup/per-sample call caps | 3 |
| Two PIDs share original sample deadline | 1 |
| Fresh replacement cannot inherit previous RSS | 1 |
| RSS-false identity path has no task read/wait | 1 |
| Sixteen-event saturation, discarded count and independent first failure | 1 |
| Actual public receipt publishes fake transition diagnostics | 1 |

The new native-boundary controls instantiate the real adapter via __new__,
then supply complete fake numeric/diagnostic state and a fake libproc method.
They do not replace the methods under test: real `_bsd`, `_call` and `inspect`
execute against scripted ctypes structures, fake errno accessors and virtual
time. Script exhaustion and exact flavors/arguments/record sizes are asserted.
Over-budget calls are counted as attempted but must not reach the fake native
function. Unconsumed scripted evidence checks distinguish refusal before a
call from failure after a late call. No actual library, process, signal, errno
native accessor or wall-clock sleep is used by this boundary fixture.

The preserved public-loop fixtures invoke real supervise_process and _Custody,
replacing only OS/native/time/stream boundaries with invented objects. They
cover root-anchor retention, owned/unrelated descendants, failed authentication,
unknown identity, post-reap census, cleanup failures, log uncertainty, control
close faults, prelaunch refusals, memory scopes and work expiration. The new
public diagnostic test intentionally injects a diagnostic into the fake API;
it verifies publication, not a connected real-native transition through the
public loop. Native-boundary logic and public orchestration are covered in
separate mock layers. Their composition on the installed host remains unqualified.

## Mock harness and execution boundary

The amended harness retains the prior mock harness algorithm, with only V3
identifiers/pins, the nine-file closure and166 denominator changed. The exact
three V12 functions extracted by AST are _remaining, _group_sample and
supervise_native_context_test_v3. No historical module imports, main, source
selection or historical constants are evaluated by that extraction. This short
ordinary-worker wrapper is not the V3 candidate under test.

The fixed future gate permits one worker:20 CPU seconds,25 work seconds,30 total
supervision seconds including cleanup,512MiB sampled group RSS and1MiB output.
Parent metadata/read/archive/publication I/O lies outside that clock. Python
runs-I-S-B with no site/.pth initialization, pytest plugin autoload, conftest,
project config or cache. A synthetic tests namespace suppresses its initializer;
the root initializer must remain absent. The worker tripwire rejects process
creation/signaling, named ctypes loading, native geometry imports and project
server initialization. It is a reviewed-source guard, not a hostile-code sandbox.

Exact PRE binds nine file hashes and the whole runtime metadata object before
attempt creation. Exclusive receipts, bounded no-follow regular-file reads,
source/runtime/PRE postflight and exact three-file final inventory support
custody, not an atomic hostile-filesystem snapshot. Output is credited only
after confirmed writer quiescence. All166 unique nodes need passing setup,
call and teardown, with zero other dispositions/errors and unchanged evidence.
Any missing receipt, ambiguous cleanup, unexpected file or publication mismatch
vetoes qualification. One-shot discipline is external; no automated retry is
authorized by a successful PRE or a failed run.

## Handoff

No remaining source blocker was found for this exact mock candidate after the
nine-source correction. Next is a separate exact PRE that binds this document,
all eight companions and fresh standard-library runtime metadata, followed by
root's full acceptance, external ten-payload archive verification and sole
invocation. A separate exact POST must reconcile any terminal result. Even a
future166/166 mock PASS would authorize only a separately frozen host gate,
not integration, real-input work or a physical score. V2 host FAIL6/8 and Gold11
V5 terminal unscored failure remain unchanged; no retry is permitted here.

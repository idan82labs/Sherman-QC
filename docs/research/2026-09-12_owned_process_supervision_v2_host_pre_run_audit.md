# Owned-process supervision V2 — independent exact host PRE

Date: 2026-09-12. Reviewer: Codex AI technical reviewer
`/root/process_v2_qualification`, independent of the host harness/protocol author.
This is AI technical review, not human approval or a result of host execution.

## Decision and authorized boundary

GREEN for one prospective eight-case owned-process host qualification under the
exact frozen protocol, conditional on root accepting this PRE, verifying the
five-payload external archive and confirming no overlapping heavy writer.
No host case has executed as part of this review. No target module import,
native process API call, child fixture, process observation or signal was used
to reach this decision. This PRE does not independently launch anything.

The scope is the eight new owned-child controls, not physical inputs, Gold11,
held-outs, email, prior attempts, private caches or arbitrary PID targets. A
successful run would still require independent POST before runner integration
review. Terminal V5 remains failed and unscored; no restart, rescore or new F1
claim follows. The existing 127/127 mock POST is prerequisite evidence, not
proof of installed private-API semantics.

## Exact source and runtime binding

The final amended host harness and protocol were read completely. The unchanged
supervisor and mock POST were previously fully reviewed in this same independent
qualification chain; their final complete raw byte hashes were checked again.
Both Python files were parsed as source AST without execution. The four-file
closure is 73524 bytes and all independently observed hashes match final release.

| Repository-relative path | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/check_owned_process_supervision_v2_host.py` | 17395 | `3241e4d110cc9355ae7a2b59785ea8ee9e6d32311a7bd2fec096c343ead09174` |
| `scripts/gold11_owned_process_supervision_v2.py` | 37329 | `649815cad6dc10d70b34cf9ae079fe77146e371ee12d1bcc4b7ea64e340492e5` |
| `docs/research/2026-09-12_owned_process_supervision_v2_source_post_run_audit.md` | 9263 | `46d677a89826a25d282575a6903083fc516ae94c25605e68e499ab630a329d9b` |
| `docs/research/2026-09-12_owned_process_supervision_v2_host_protocol.md` | 9537 | `94eb5c63993ae9ef867f049168bcc1d70e90e5381bae4a3593f90b7033cb37db` |

Canonical path-to-SHA map digest, sorted-key compact JSON without newline:
`5abb395b3e54146801515d6f01d6f48a7131abaa53b72d5a0420ae70998926c9`.

The following runtime metadata was independently collected using isolated
stdlib-only Python; no candidate was imported. Its canonical sorted-key compact
JSON digest without newline is
`50bb15d426d20537f5aa33dace33610ff07eee9bf45008a1556c44a61bf9a587`.

```json
{"executable":"/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11","executable_stat":[16777231,17404606,17257904,1783888533314804818,1783888533315379399],"implementation":"cpython","kernel_release":"25.5.0","kernel_version":"Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050","machine":"arm64","platform":"darwin","python_version":"3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]","scope":"EXECUTABLE_METADATA_AND_KERNEL_IDENTITY_NOT_TRANSITIVE_BINARY_CLOSURE"}
```

Executable stat fields are device, inode, size, mtime_ns and ctime_ns. The root
package initializer was independently confirmed absent. This is metadata and
kernel identity, not the full interpreter/libproc/shared-cache binary closure.
The prospective harness binds the same object before each case and at postflight.

## Closed source-review findings

1. **Scheduler-dependent deadline classification.** The two deadline controls
   now admit only WALL_LIMIT with no supervisor_failure_reason, or
   SUPERVISOR_FAILURE:TimeoutError with exact reason
   NATIVE_PROCESS_SAMPLE_DEADLINE. The latter is the unchanged candidate's
   work-sample boundary. Both require exit -9, a native signal-call witness,
   clean reaping and no identity/custody/cleanup uncertainty. Other generic
   supervisor failures remain failures. This finite rule was frozen before any
   host execution; it is not a post-result relaxation.
2. **Unknown RSS was arithmetically treated as zero.** Every case now requires
   positive integer tree and parent peaks and a positive child-tree sample
   count. Their conservative sum must not exceed 536870912 bytes. None cannot
   pass. EOF/nonzero payloads delay 0.15s to permit observation, but absence of
   evidence still fails; the delay does not guarantee sampling or reduce the
   denominator.
3. **Missing publication versus not-invoked state.** Candidate-load failure is
   caught before case invocation and attempts eight truthful not-invoked
   records. A result-publication failure halts new invocations, preserves the
   in-memory result for COMPLETED when possible, and vetoes final success.
   Missing records from disk/process failure remain NOT_OBSERVED /
   INVOCATION_STAGE_UNKNOWN, not inferred no-start. A returned receipt is not
   used as the definition of invocation_started.

No additional blocking source defect was identified within this finite owned-
process qualification scope. These closures were reviewed as source, not tested
by executing the new harness.

## Frozen eight-case matrix and witnesses

| Case | Expected direct exit | Accepted stop and specific witness |
| --- | ---: | --- |
| outer_eof | 0 | none; reaped and owned group empty |
| outer_nonzero | 7 | none; actual nonzero exit retained |
| outer_deadline | -9 | one of the exact two deadline dispositions; signal call |
| outer_exec | 0 | none; at least one same-lifetime exec-version update |
| outer_waited_descendant | 0 | none; at least two tracked lifetimes |
| outer_surviving_descendant | 0 | OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT; two tracked lifetimes and signal call |
| inner_eof | 0 | none; direct-child/observed-descendant custody |
| inner_deadline | -9 | one of the exact two deadline dispositions; signal call |

All eight must independently satisfy the common gate: matching exit/stop,
direct child reaped, no process attention, no cleanup/control-close errors,
no observed foreign escape, zero historical unknown-identity peak, certain log
storage, required positive observed RSS and correct outer-group confirmation.
Expected nonzero/killed payload exits are explicit negative test controls,
not suppressed failures or successful benchmark inference arms.

Each command is the fixed application interpreter and this exact script with a
fixed internal payload role. The role logs a small PID/PPID witness, but no
receipt PID is later promoted to cleanup authority. The exec control reexecs
the same interpreter/script; no arbitrary command, foreign escape or unrelated
PID target is exposed by the approved invocation. The waited and surviving
descendant controls create one additional same-group child each. Descendants
inherit the restricted environment and run the same fixed self-limited payload.

The unchanged candidate alone performs process signaling after fresh coherent
generation/group authentication. It uses audit-token SIGKILL, never numeric
kill/killpg, and refuses absent/replaced/foreign/zombie targets. An unreaped
direct child anchors outer group ownership until cleanup; no post-reap signal
authority is restored. The host harness adds no stale-PID fallback. Inner
controls use only direct-child/observed-descendant scope in the inherited group.
The caller must remain sole waiter with SIGCHLD default, as checked by the SUT.

## Work limits and remaining uncertainty

The frozen campaign has eight serial direct payloads and two descendant
lifetimes, with at most three live Python processes including the qualification
parent. New cases stop after cleanup uncertainty; therefore a potentially live
unconfirmed case is not followed by another fixture. No concurrent heavy
Sherman experiment is allowed, and the repository 16 GiB ceiling remains binding.

Parent CPU is capped at 10s; each payload caps itself at 2 CPU seconds and arms
a four-second SIGALRM handler that exits 98. Deadline payloads sleep 3s;
descendants sleep 1.5s; EOF/nonzero delay 0.15s; exec waits 0.2s on each side;
the exiting parent waits 0.2s before leaving its same-group descendant. The
waited parent uses a two-second wait timeout. These timeouts are failure bounds
and observation opportunities, not a promise that all controls pass under load.

Self-limits are installed on entry to payload(), after interpreter startup,
module imports and argument parsing. They do not hard-bound startup or a
blocked native/I/O operation. The exec phase installs its limits again. The
protocol's explicit non-hard synchronous-I/O limitation remains essential; no
self-alarm is treated as proof that a process has exited or that output may be
read. An OS/process/disk failure can prevent final records despite the bounded
ordinary control flow.

The between-case guard is 55 wall seconds, and the outcome gate also requires
55s or less before publication. Work caps are 2s except the two 0.3s deadline
controls; failed-case cleanup adds at most its declared five-second observed
reserve. These are Python checks, not hard native-call cancellation. Parent
and payload CPU limits are per-process, not an aggregate token bucket. Case
RSS is sampled at 512 MiB with a conservative sum of parent/tree observed peaks;
it is not continuous memory containment. Logs cap at 16384 bytes per case,
administrative files at 262144 bytes each, and a successful final namespace has
18 files. The thread environment is fixed to one through the SUT environment.

## Admission, custody, retention and publication

Exact source/runtime/PRE admission occurs before attempt creation. The source
closure pins the unchanged SUT, and the PRE hash binds its own exact bytes plus
the four-source map and full runtime object. Each case rechecks these bindings
before invoking the candidate. Candidate source is read, hash checked and loaded
once as definitions; no application package, pytest or numerical module is
needed. The approved command uses -I -S -B and therefore no site/.pth startup.

STARTED records the full ordered eight-case matrix and fixed limits. Each case
records invocation_started separately from receipt presence. A failure before
invocation may truthfully yield not-invoked; a missing record cannot. No case
log is read until its receipt confirms quiescence, and no new case is invoked
after uncertain custody. Postflight content reads are withheld when owned
writers remain unconfirmed. The private cache is retained as metadata only and
must not be traversed or reused.

Source/runtime/PRE equality, all eight passing cases, clear postflight and
publication errors, CPU/wall limits and quiescence enter the outcome gate.
Exclusive publication prevents overwrite. Failed CASE publication stops further
invocations; failed COMPLETED publication vetoes final success. A favorable
COMPLETED status remains pending final publication and independent POST. Final
stdout must also report a passing exact-namespace/content-hash gate. If storage
or the parent fails, STARTED remains the denominator authority; missing records
cannot shrink eight cases to a favorable subset.

## Required release sequence and stop rules

Root must fully read/accept this exact PRE, create an external archive of the
four frozen sources plus this PRE as five unique regular payloads, verify exact
archive/live bytes without extraction, and confirm no overlapping heavy writer.
Only then issue the single approved invocation:

```
.venv/bin/python -I -S -B -u scripts/check_owned_process_supervision_v2_host.py --execute --pre-sha256 <accepted-exact-PRE-sha256>
```

No retry, extra case, limit increase, native fallback, retrospective witness
relaxation, source mutation, installation or unrelated cleanup is released.
Changes to source/runtime/PRE invalidate this exact release. Keep every failure,
unknown and unavailable disposition. Parent termination must not be interpreted
as permission to signal a historical numeric PID or read a private cache.

Mandatory independent POST must reconcile all eight case records/logs and actual
native-call/identity witnesses, exact source/runtime/PRE before and after,
resources, invocation terminal status, final inventory/stdout and external
quiescence evidence. Until then, no host success or integration readiness may
be claimed. Even a successful finite host POST would not prove every PID/exec
race, continuous descendant containment, atomic group membership at signal time,
hard I/O deadlines or cross-release private-API compatibility. Physical engine
qualification and F1 remain separate and unperformed here.

OWNED_PROCESS_SUPERVISION_V2_HOST_PRE_JSON={"decision":"GREEN","expected_cases":8,"runtime_binding_sha256":"50bb15d426d20537f5aa33dace33610ff07eee9bf45008a1556c44a61bf9a587","schema":"OWNED_PROCESS_SUPERVISION_V2_HOST_PRE","scope":"OWNED_PROCESS_HOST_ONLY","source_map_sha256":"5abb395b3e54146801515d6f01d6f48a7131abaa53b72d5a0420ae70998926c9"}

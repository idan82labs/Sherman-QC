# Owned-process supervision V2 — independent exact mock PRE

Date: 2026-09-12. Reviewer: Codex AI technical reviewer
`/root/process_v2_qualification`, independent of the final harness/protocol
author. This is AI technical review, not human approval, a passed test result,
installed-host qualification, integration approval or physical-run authority.

## Decision and scope

GREEN for exactly one prospective invented-only qualification invocation under
the frozen protocol below, after root accepts this PRE, verifies the required
external archive and confirms no overlapping heavy writer. No target module
import, pytest collection, test execution, native process API operation, process
observation or signal was performed in this review. There are **51 source-counted
test functions / 127 prospective outcomes / zero executed outcomes**.

This review read the complete seven-file closure, including all 706 lines of the
candidate supervisor, all 953 lines of its expanded tests, the full new harness
and protocol, the full inherited V12 file, the prior review and the 24-byte test
initializer. It separately recomputed the source map and ordinary runtime
metadata using an isolated stdlib-only Python invocation; no candidate or
pytest module was imported. No raw/Gold/held-out/email/private-cache/attempt
contents were accessed. No existing source, test, harness, protocol or receipt
was edited. The terminal V5 benchmark remains failed and unscored; no restart,
rescore, inferred F1 or amendment of its evidence is released.

## Exact seven-source closure

Each SHA256 is over the complete raw file bytes. The canonical source-map hash
is over the path-to-SHA256 object, sorted-key compact JSON with no newline.

| Repository-relative path | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/check_owned_process_supervision_v2.py` | 22782 | `97370d8a30a30b75ff761bf94e77346e41f4b19133ad58b082ee3d6bb7a46594` |
| `scripts/check_cad_guided_connected_v12.py` | 19437 | `0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694` |
| `scripts/gold11_owned_process_supervision_v2.py` | 37329 | `649815cad6dc10d70b34cf9ae079fe77146e371ee12d1bcc4b7ea64e340492e5` |
| `tests/test_gold11_owned_process_supervision_v2.py` | 43465 | `dc6587fbee11815a3c7036fb78742da2560fa229aeda9f8017a7cc8479b4125f` |
| `tests/__init__.py` | 24 | `5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b` |
| `docs/research/2026-09-12_owned_process_supervision_v2_source_review.md` | 8631 | `5851495c944b220595cd380f10609d1ed34dafbe3014d39175c3e77d7adf8876` |
| `docs/research/2026-09-12_owned_process_supervision_v2_source_protocol.md` | 7950 | `16e45389b12a3388d0b7012b5b09a07ffbfc33381704d7fba87d9a524a2d532e` |

Total: 139618 bytes. Source-map SHA256:
`cbd13011199d0b3469665d9af810e24811156dd6da75c2b9c7bcc7bcf1fc9950`.
All independently observed hashes agree with the author's final release.
The earlier review covers this unchanged supervisor but only the original
21-function/40-outcome test prefix. This PRE additionally reviews the appended
30 functions / 87 prospective outcomes; the older review is not relabeled as
having covered them or as execution evidence.

## Runtime binding independently observed

The complete runtime object is reproduced here; its canonical sorted-key compact
JSON hash (without newline) is
`2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689`.

```json
{"distribution_versions":{"pytest":"9.1.1"},"executable":"/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11","executable_stat":[16777231,17404606,17257904,1783888533314804818,1783888533315379399],"implementation":"cpython","machine":"arm64","package_initializers":{"pytest":{"bytes":5644,"sha256":"ee37885c9583f843390684dfb496907f87a0905128e7a3cf4ad227f49ff34c2d"}},"platform":"darwin","python_version":"3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]","scope":"ORDINARY_LOADER_ENTRYPOINTS_METADATA_NOT_FULL_RUNTIME_BINARY_CLOSURE"}
```

Executable stat fields are device, inode, size, mtime_ns and ctime_ns. The root
package initializer was independently confirmed absent. Metadata discovery read
only the pytest distribution metadata and named initializer; it did not import
pytest or load a candidate library. This binds ordinary loader entry points,
not all transitive packages, bytecode, shared libraries or Python executable
content bytes. The harness additionally checks the actual loaded pytest path,
version and initializer before and after the prospective test invocation.

## Candidate and test assessment

No new blocking source defect was found within the explicitly sampled scope.
Lifetime tracking keys PID and uniqueid; verified absence or replacement retires
an old generation. Same-generation exec refreshes idversion, and known surviving
descendants remain tracked after ordinary reparenting. New parent-based admission
requires PPID and parent-uniqueid agreement; inner scope does not admit unrelated
shared-group members. Outer group scope depends on a directly authenticated,
unreaped child. The root remains tracked as a zombie until cleanup can reap it.

RSS is bracketed by coherent identity reads. Permission, short-record and
unstable-identity cases fail rather than fabricate absence or zero memory.
Fresh signal authorization checks generation and group before constructing the
audit token. The code neither polls nor numerically signals its children; token
ESRCH is not retirement proof, and accepted signaling is not confirmed exit.
After reap, the final group census is read-only and cannot restore authority.
Historical unknown peaks and separate bounded first-error records retain causal
evidence despite routine churn. Cleanup failures preserve first failure and
explicit process-attention state.

The public controls call the real `supervise_process` and `_Custody`, replacing
only the native API factory, child/selector/log/pipe objects, clock, signal and
OS boundaries. They cover EOF and fast zombie/no-RSS cases, surviving children,
initial/fresh authentication failures, post-reap nonempty census, dispatch and
I/O faults, uncertain storage counts, cleanup failure and resource limits.
The fake child asserts zombie-before-wait and absence of modeled same-group
descendants; this is a model assertion, not observed OS lifecycle behavior.

The adapter controls instantiate through `__new__`, never its native library
constructor. Fake call implementations cover exact and overflowing PID lists,
duplicates/malformed values, sample/call limits, separate cleanup reserve,
pre-call expiry versus late effects, RSS failure/zombie transitions, and fresh
exec tokens. The stale-token control exercises two `signal_owned` calls with
fresh mocked inspections, not a full public cleanup-loop exec retry. The
4096-generation churn and 4096-integer enumeration are invented and bounded.
No test invokes real subprocess, signal, libproc, geometry or application code.

## Harness admission and prospective result gate

The harness admits the exact PRE before creating an attempt. It binds all seven
sources, enforces the candidate/test pins, and requires one unique bounded PRE
declaration with exactly the frozen scope, count and map/runtime hashes.
The enclosing worker supervisor comes from exactly three top-level V12 AST
definitions: `_remaining`, `_group_sample` and
`supervise_native_context_test_v3`. No V12 imports, constants, main, historical
suite selection or other source functions execute. The entire V12 file is
nevertheless in the source closure.

Frozen limits are one serial worker, 20 CPU seconds, 25 work seconds, 30 total
supervision seconds including five seconds cleanup reserve, 536870912 bytes
sampled group RSS and 1048576 output bytes. Parent pre/postflight, archiving and
publication I/O are outside that clock. The repository 16 GiB ceiling and no
concurrent-heavy-work rule remain binding. Limits and denominator cannot change
after invocation begins. The inherited short supervisor rejects observed worker
children and uses process-group cleanup. It is not V2 and does not independently
solve hostile PID/PGID reuse; its documented bounded lifecycle assumption and
sampled/non-hard-I/O limits remain relevant to this short reviewed mock worker.

The worker runs application Python with `-I -S -B`, imports only the declared test
path through pytest with configuration, conftest, plugin autoload and cache
disabled, and supplies a synthetic `tests` package to suppress its bound
initializer. The audit hook rejects reviewed subprocess/fork/spawn/signal events,
named ctypes library loading, specified numerical/native imports and named
backend/root initializer execution. It allows ctypes' current-process loader.
These are defense-in-depth tripwires, not a security sandbox or exhaustive
syscall confinement. No claim relies on the tripwire making arbitrary code safe.

Outcome accounting requires 127 unique collected node IDs, all 127 setup/call/
teardown triplets passing, and zero failure, error, skip, xfail, xpass,
deselection, collection error or invalid report. Missing phases cannot pass.
Source/runtime/PRE equality, expected loaded environment, worker exit zero and
CPU usage within the fixed cap also enter the worker gate.

The parent does not parse or credit the worker ledger before the enclosing
supervisor confirms empty writers. It independently attempts source, runtime
and PRE postflight even if tests fail or the worker dies. A passing decision
additionally requires no supervision/cleanup/postflight error and an exact final
three-file retained namespace with expected hashes. The inherited supervisor's
output hash calculation is not itself independent final-publication custody.
`COMPLETED.json` marks any favorable result as pending final publication and
independent POST; the final stdout publication gate must also be reconciled.

## Conditions, stop rules and remaining boundaries

Root must read and accept this exact PRE, archive these seven files plus this
PRE as eight unique regular payloads outside the attempt, verify archive/live
bytes without extraction, and confirm no overlapping heavy worker. Only then:

```
.venv/bin/python -I -S -B -u scripts/check_owned_process_supervision_v2.py --execute --pre-sha256 <accepted-exact-PRE-sha256>
```

One invocation only. No retry, suite/limit override, installation, private-cache
inspection, previous-attempt reuse or physical input access is released. Any
changed source/runtime/PRE or failed admission invalidates this exact release.
Failure, partial output, missing phases, cleanup uncertainty or publication
failure stays failure; do not reduce the denominator or infer a successful run.
An independent POST must reconcile final retained output, all 127 dispositions,
source/runtime/PRE custody, quiescent writers and the final namespace/stdout.

Mock success would qualify only these invented source contracts. Installed-host
private API/ABI behavior and actual owned-child signaling/lifecycle remain
separately gated. No atomic group-membership guarantee, continuous process-tree
containment, hard I/O cancellation, continuous RSS limit, physical engine
accuracy, integration readiness or F1 follows from this PRE or these mocks.

OWNED_PROCESS_SUPERVISION_V2_EXACT_PRE_JSON={"decision":"GREEN","expected_outcomes":127,"runtime_binding_sha256":"2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689","schema":"OWNED_PROCESS_SUPERVISION_V2_EXACT_PRE","scope":"INVENTED_ONLY","source_map_sha256":"cbd13011199d0b3469665d9af810e24811156dd6da75c2b9c7bcc7bcf1fc9950"}

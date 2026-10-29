# Owned-process supervision V3: prospective installed-host qualification

Date: 2026-09-12. SOURCE CANDIDATE FOR EXACT PRE, not execution release. Eight serial
owned-child cases; none executed at authoring. Independent exact PRE and archive
are required after final source freeze. No Gold11 run or physical input access.

## Reason and scope

V2's host gate remains terminal FAIL6/8, including ordinary exec and a
waited descendant. Failed native brackets/errno were not retained; Apple's exit
lifecycle makes a coherent non-zombie identity/TASKINFO ESRCH transition plausible,
not retrospectively proven. V3's new INEXIT-gated bounded wait and causal evidence
are the documented engineering reason for this successor, not an unchanged retry.
It never accepts missing RSS, resets a sample deadline or relaxes identity checks.

The166-outcome V3 mock gate is prospective at this protocol's authoring. Its
independent accepted PASS POST at
2026-09-12_owned_process_supervision_v3_source_post_run_audit.md is a mandatory
prerequisite to the exact host PRE and invocation. No pass is implied here.
Mocks do not prove the installed macOS private libproc interfaces, actual zombie retention,
exec version updates or generation-coupled signal permissions. This gate tests
those mechanisms on newly created self-limited processes only. It neither
reopens terminal V5 nor changes the benchmark's science, inputs or score rules.

The exact four-source closure is:

1. scripts/check_owned_process_supervision_v3_host.py
2. scripts/gold11_owned_process_supervision_v3.py
3. docs/research/2026-09-12_owned_process_supervision_v3_source_post_run_audit.md
4. This protocol.

Candidate supervisor is42751 B SHA256
9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea.
Host harness is17637 B SHA256
349669f88b378211a1585d1f172a8dd20431180933c87e981e34f8e229c61cde.
No target imports or execution were used for authoring. Source-only AST/hash
checks are permitted. Existing frozen sources and mock artifacts remain intact.

Read only these sources, PRE, interpreter/kernel metadata and this gate's own
bounded terminal files. No raw scans, CADs, Gold, datasets, predictions, email,
old attempts, private caches or unrelated PID content. No packages/services/
credentials/hardware/external changes. Native observation is restricted to this
supervisor and its scoped owned group/parent descendants. No process is selected
from an external process listing for signaling. No foreign-escape test is included.

## Fixed eight-case matrix

| Case | Mode | Expected direct exit / stop | Additional witness |
| --- | --- | --- | --- |
| outer_eof | new owned session | 0 / none | reaped, group empty |
| outer_nonzero | new owned session | 7 / none | failure exit retained as7 |
| outer_deadline | new owned session | -9 / WALL_LIMIT | native signal call |
| outer_exec | new owned session | 0 / none | same lifetime's exec version update |
| outer_waited_descendant | new owned session | 0 / none | at least two tracked lifetimes |
| outer_surviving_descendant | new owned session | 0 / OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT | at least two lifetimes, native cleanup signal |
| inner_eof | inherited group | 0 / none | direct child/observed descendant custody only |
| inner_deadline | inherited group | -9 / WALL_LIMIT | native signal call, no group signal |

For only the two deadline controls, the finite accepted stop alternatives are
WALL_LIMIT with no supervisor_failure_reason, or SUPERVISOR_FAILURE:TimeoutError
with the exact reason NATIVE_PROCESS_SAMPLE_DEADLINE. The latter is the unchanged
candidate's boundary when the scheduled sample reaches its work deadline.
Both still require native signaling, exit-9 and complete clean custody; no other
generic supervisor failure is accepted. This rule is frozen before host execution.

Every payload first sets CPU2 seconds and a four-second self-alarm whose handler
exits98, then consumes only its <=16384-byte dispatch and emits a small fixed
JSON PID/role witness. Deadline payloads sleep3s; descendant payloads sleep1.5s.
EOF/nonzero payloads wait0.15s to permit actual RSS observation; if no RSS sample
is obtained the test refuses, rather than substituting zero for unknown memory.
Exec payloads wait0.2s on each side of an exec of the same fixed interpreter and
script, allowing actual observations. Waited-parent owns/waits its descendant;
exiting-parent deliberately exits after0.2s, leaving the same-group self-expiring
descendant. No payload creates new sessions except the supervisor's explicit
outer spawn; no escape, arbitrary command or unbounded descendant tree.

All signals are performed by the unchanged candidate using fresh generation/
group authentication and the audit-token API. No numeric kill/killpg fallback,
retained-receipt PID signaling, post-reap signaling or external cleanup shortcut
is authorized. The parent remains sole waiter for its direct children, with
SIGCHLD default. A cleanup uncertainty stops new cases; remaining cases receive
explicit not-invoked dispositions. Do not retry or credit uncertainty as success.
All payloads self-expire even if candidate cleanup fails, but that alone does
not qualify custody or authorize content reading/reaping via stale receipt IDs.

## Fixed budgets and limits of enforcement

One serial qualification parent, maximum three live Python processes including
it; eight direct payloads and at most two descendants over the whole matrix.
ParentCPU10s; each payloadCPU2s. These are per-process limits, not one shared
aggregate CPU token bucket. Parent's between-case wall guard is55s and the final
outcome gate also requires <=55s before publication. Individual work caps are
2s except the two deadline controls at0.3s. Candidate failure cleanup reserves5s.
Synchronous process/native/log I/O and final publication are not forcibly
cancelled by a hard wall deadline; this limitation remains explicit.

512MiB sampled case RSS; all existing candidate counters/caps remain unchanged.
Inner mode separately bounds parent RSS; every case requires a positive actual
child-tree sample and positive integer parent/tree peaks. Final admission
requires their conservative peak sum <=512MiB. An unobserved child peak fails
this memory-evidence gate; it is never substituted by zero. The numeric sum
does not convert sampling into an OS allocation limit. Repository16GiB ceiling
and no other heavy Sherman experiment remain binding. Thread environment is1.
Per-case output16KiB; each administrative file256KiB; at most18 final owned files.

Use application Python3.11 with-I-S-B, no site/.pth/package initialization. No
NumPy/pytest/application import. Bind exact executable identity/version/platform,
kernel release/version before and after. This metadata does not attest the full
interpreter or shared-cache/libproc binary closure. Successful host controls
qualify observed behavior on this declared host, not all macOS releases.

## Exact freeze and invocation

The independent PRE is a NEW file:
docs/research/2026-09-12_owned_process_supervision_v3_host_pre_run_audit.md.
After reviewing final source, mock POST and this protocol, its single marker
OWNED_PROCESS_SUPERVISION_V3_HOST_PRE_JSON= must contain exactly schema
OWNED_PROCESS_SUPERVISION_V3_HOST_PRE, decision GREEN, scope OWNED_PROCESS_HOST_ONLY,
expected_cases8, source_map_sha256 and runtime_binding_sha256. Hash sorted-key
compact JSON without newline: path-to-SHA map for sources, complete runtime
object for environment. It must describe AI technical review, not human approval.

Root must accept exact PRE, archive four sources plus PRE as five unique regular
payloads outside any attempt, verify archive/live bytes and confirm no other
heavy worker. Then issue exactly one command:

```
.venv/bin/python -I -S -B -u scripts/check_owned_process_supervision_v3_host.py --execute --pre-sha256 <accepted-exact-PRE-sha256>
```

No retry/case/limit override. One unique owned_process_supervision_v3_host-*
attempt receives exclusive STARTED.json, all eight CASE_nn_RESULT.json files,
up to eight CASE_nn_OUTPUT.txt logs and COMPLETED.json. STARTED predeclares all
cases/limits/bindings. Invocation-start state is distinct from presence of a
returned receipt; missing receipts do not imply no process ever started.
Private cache path is retained as metadata only, never traversed.
Candidate-load failure is caught before any case invocation and attempts eight
truthful not-invoked records. Case-publication failure stops new invocations,
retains the in-memory disposition for COMPLETED when storage permits, and vetoes
the final gate. A disk/process failure may prevent any further record; missing
records remain NOT_OBSERVED / INVOCATION_STAGE_UNKNOWN, never inferred not-invoked
from absence. STARTED's full eight-case denominator remains authoritative.

## Pass, negative retention and POST

Each case must match its exact exit/stop expectation, have confirmed reaping and
no process-attention, cleanup/control-close errors, unknown identity peak or
foreign escape. The only allowed supervisor_failure_reason is the exact frozen
deadline alternative above. Outer groups must be confirmed empty. Expected failing-child
outcomes are test controls, not suppressed failures or successful benchmark arms.
Logs must be present, bounded and size-reconciled after writer quiescence.
Source/runtime/PRE remain unchanged before each case and after the whole gate.
Every receipt must additionally declare GOLD11_OWNED_PROCESS_SUPERVISION_V3,
exit_transition_unresolved0 and no exit_transition_first_failure. The original
eight-case matrix, timing and all prior acceptance predicates remain unchanged.
A typed unresolved exit-transition timeout is not an accepted deadline control.
All eight cases must pass; no removal, skip or favorable-subset acceptance.

Final publication requires the exact declared namespace and every content hash.
Mandatory independent POST reconciles invocation, all eight dispositions,
actual native-call/identity/cleanup evidence, transition entry/resolution counts
and bounded initiating/terminal diagnostics, resources, source/runtime/PRE,
final inventory and external terminal/quiescence observations. Any failure
remains terminal and retained; a successor needs documented engineering reason,
new frozen protocol and independent PRE.

The gate may observe zero eligible exit transitions; deterministic mock coverage
is therefore separate and mandatory. No fixture sleep, timing cap or success
condition may be adjusted after invocation to force a favorable observation.

This finite host test is not a proof of every exec race, continuous containment,
atomic group membership at signal time, hard I/O cancellation or Gold11 accuracy.
A successful POST permits separately audited runner integration only. No physical
benchmark executes here; the engine/F1 goal remains open.


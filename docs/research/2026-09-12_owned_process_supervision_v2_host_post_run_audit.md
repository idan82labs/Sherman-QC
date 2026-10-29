# Owned-process supervision V2 — independent exact host failure POST

Date: 2026-09-12. Reviewer: Codex AI technical reviewer
`/root/process_v2_qualification`, independent of the host harness author/operator.
This is AI technical reconciliation, not human approval or promotion evidence.

## Disposition

**FAIL / RED: 6 of 8 host controls passed.** The single attempt is terminal.
`outer_exec` and `outer_waited_descendant` failed the frozen identity/RSS
stabilization and clean-custody criteria. All eight were actually invoked and
have retained result and output records; none is missing or reclassified as
not-invoked. Final owned writers were confirmed quiescent, but eventual cleanup
does not erase the two failures or qualify V2 for runner integration.

The source/runtime/PRE/archive and final-publication custody checks reconcile.
This is a correctly retained failing qualification, not a corrupted or
unpublished passing result. No V2 retry, source amendment, denominator reduction,
failure-rule relaxation, native fallback or integration is released by this POST.
The earlier 127/127 invented mock result remains valid in its separate ledger.
Terminal V5 remains failed and unscored; no physical input or F1 is involved.

## Evidence scope and invocation provenance

The sole released attempt is
`docs/research/cad_engine_v1_synthetic_checks/owned_process_supervision_v2_host-cpab0qfk`.
I read the complete STARTED/COMPLETED records, all eight case records and all
eight logs, then independently reconciled every case object against COMPLETED,
the exact 18-file namespace, log byte accounting, frozen case predicates,
accepted source archive, four live source bindings, exact PRE and runtime
metadata. Verification used only isolated stdlib parsing/hashing/metadata, not
imports or execution of the target or host harness.

Root supplies the execution provenance: tool `7d7903` invoked the sole command
and returned session 16998; tool `4df246` observed terminal exit 1. Final stdout
reported outcome false, publication true, no publication error and an 18-file
inventory. Root's external process-only check `193f42` found no Sherman Python
writer. These are operator-supplied observations, not fresh process observations
made by this reviewer. This audit independently verifies the final namespace and
retained custody after that release.

No native process API, subprocess fixture, signal, rerun, raw/Gold/held-out/email,
old attempt or private-cache content was accessed. The retained cache path was
treated only as receipt metadata and not followed. This NEW POST is the only
file written; no frozen source or retained result was edited.

## Complete eight-case disposition

All invocation_started fields are true. Each case's outer error and
unavailable_reason fields are null; the two substantive failures are in their
returned supervisor receipts and frozen pass predicates.

| Index / case | Actual exit | Actual stop | Result | Wall seconds |
| --- | ---: | --- | --- | ---: |
| 00 outer_eof | 0 | none | PASS | 0.178282125 |
| 01 outer_nonzero | 7 | none | PASS | 0.185837459 |
| 02 outer_deadline | -9 | SUPERVISOR_FAILURE:TimeoutError | PASS | 0.330164750 |
| 03 outer_exec | 0 | SUPERVISOR_FAILURE:RuntimeError | FAIL | 0.509997000 |
| 04 outer_waited_descendant | -9 | SUPERVISOR_FAILURE:RuntimeError | FAIL | 1.582367041 |
| 05 outer_surviving_descendant | 0 | OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT | PASS | 0.285788750 |
| 06 inner_eof | 0 | none | PASS | 0.199395958 |
| 07 inner_deadline | -9 | SUPERVISOR_FAILURE:TimeoutError | PASS | 0.332930917 |

Both deadline cases have exact reason NATIVE_PROCESS_SAMPLE_DEADLINE and satisfy
the finite alternative frozen before execution; neither is a post-result
exception. They each record one accepted native audit-token signal, exit -9,
zero unknown peak, no cleanup/control-close error and final clean custody.
The nonzero case correctly retains exit 7 as its expected negative control.

Every case records direct_child_reaped true and requires_process_attention false.
All six outer groups are confirmed empty; the two inner group-confirmation
fields are null, as required for inner scope rather than an unrelated shared-
group claim. No foreign-group escape or final unresolved identity is recorded.
No case reports uncertain log storage, and each log's actual bytes equal all
four forwarded/stored/write-attempted/write-accepted counters and its binding.

## Failure analysis and limits of diagnosis

### 03: outer_exec

The log contains both exec_before and exec_after witnesses for the same direct
PID 17028. The receipt records one exec_version_update for lifetime uniqueid
2789626. Thus the transition was observed at least once; this was not a failure
to collect any exec witness. Nevertheless the frozen gate fails because:

- supervisor_failure_reason is PROCESS_IDENTITY_OR_RSS_DID_NOT_STABILIZE;
- historical unknown-identity peak is 1;
- one cleanup RuntimeError retains the same stabilization reason;
- four coherence retries and two OWNED_IDENTITY_QUERY_UNAVAILABLE events were
  retained for PID 17028 / idversion 5698522 / uniqueid 2789626.

The one signal call returned TOKEN_STALE_OR_EXIT_NOT_RETIREMENT_PROOF, not an
accepted signal. The direct child eventually exited 0 and was reaped, with
outer group empty and final unknown state cleared. Neither ordinary exit nor
later cleanup cancels the historical uncertainty or cleanup error.

### 04: outer_waited_descendant

The log witnesses waited_parent PID 17029 and descendant PID 17030. Two
lifetimes were tracked. The causal unavailable-query events identify descendant
17030, uniqueid 2789628, idversion 5698526, parent/group 17029. This case also
retains four coherence retries, historical unknown peak 1, the same supervisor
stabilization failure and one cleanup RuntimeError with the same reason.

Cleanup recorded ATOMIC_SIGNAL_ACCEPTED for the direct parent 17029, then an
ambiguous TOKEN_STALE_OR_EXIT_NOT_RETIREMENT_PROOF for descendant 17030. A later
VERIFIED_KERNEL_ABSENCE retired the descendant, and final group custody was
confirmed. The direct parent's actual -9 differs from the frozen expected 0,
so the case fails on exit as well as stabilization/unknown/cleanup criteria.

### What the receipts do and do not establish

The exact source raises this stabilization error after its two coherence
attempts when identity keys do not match or an RSS query remains missing despite
identity reads. The retained receipt does not include both raw native identity
rows, the differing field, or each task-RSS return/errno pair. Therefore it
cannot distinguish an identity-key transition from a temporary task-RSS
unavailability, or conclusively place the exec-case failure at exec rather than
a subsequent exit transition. The source's immediate bounded retry mechanism
and these naturally ending fixtures motivate separately versioned transition
handling work, but the exact mechanism remains an inference, not a proven errno
or kernel-state diagnosis.

No generation replacement or foreign escape is recorded. That absence does not
prove every possible race is harmless. Token ESRCH remains explicitly ambiguous
between stale token and exit and was not treated as retirement evidence. The
later verified absence/reap observations supply terminal custody, not retroactive
qualification of the earlier failed samples.

The passing surviving-descendant control provides useful narrower evidence:
it tracked two lifetimes, observed one reparent update, recorded an accepted
audit-token signal for descendant 17033 after PPID became 1, then verified its
absence before final clean custody. This supports that tested path on this host,
not full V2 promotion despite the two failed controls.

## Resources and native activity

Parent CPU was 0.096387s; parent high-water RSS 27099136 bytes. Campaign wall in
COMPLETED is 3.622842166107148s. Root's final stdout reports through-inventory
wall 3.624909375s; that latter value is operator-supplied final-output evidence.
Maximum sampled case budget RSS is 73170944 bytes, in case 04, below the fixed
536870912-byte cap. All cases have positive tree/parent samples and conservative
sum within the cap; no unknown child peak was substituted by zero.

Across all eight receipts there are 865 attempted native calls and six signal
calls: four ATOMIC_SIGNAL_ACCEPTED events and two stale-or-exit outcomes. The
two failing controls each retain four coherence retries; the other six retain
zero. All frozen parent/payload CPU, case work, cleanup, log and administrative
limit fields match STARTED and the protocol. No limit was raised during this
attempt. Ordinary host timings are below the campaign limits; this does not
convert sampled RSS or Python deadlines into continuous memory/IO containment.

COMPLETED has outcome_gate false, status FAIL, expected_cases 8 and passed 6.
Candidate-load error and halt reason are null. Postflight_errors and
publication_errors are empty, and all_owned_writers_confirmed_quiescent is true.
Failed but fully cleaned cases were followed by later cases as the frozen
protocol allowed; no uncertain writer was used as authority to continue.

## Exact final inventory

Exactly 18 expected regular files are present; all were checked independently.
Each CASE result equals its corresponding COMPLETED result object. Each log
matches its per-case output binding. No extra member or missing case exists.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| STARTED.json | 2793 | `589892ee81bb52b76826e13dcd7c8ecfb8d8746498b2f9fb15091e9e04110020` |
| CASE_00_RESULT.json | 2675 | `17b20c44b510b60909c794a58d28ccd28f0e8cda92cdadadb9a017b779fff2b2` |
| CASE_00_OUTPUT.txt | 61 | `7dd320ae1fe561f574a0ec8ffa39a01e3895021237553dca87a4df4c61747813` |
| CASE_01_RESULT.json | 2680 | `7760ff68076e8848ab527965266a12a1c554dc485b1be318f891b8848b76a844` |
| CASE_01_OUTPUT.txt | 65 | `867871ffb70a8b6cfa7b2d1d8e72513b3d6df5c7f56028d8492edf3e313a9068` |
| CASE_02_RESULT.json | 2947 | `ef22d25517cb18c7670c9617b6d247847c90a0e3452682fb56b8a5427dc6ab40` |
| CASE_02_OUTPUT.txt | 66 | `6d0f51a2579a634c658e168c2b17357c654a3d674d63fded08e3a7f79cc7d6b0` |
| CASE_03_RESULT.json | 3615 | `d4effd66e12f81d5b67d08853d04c759c3adf88b37722c8a0103102b2d3763bc` |
| CASE_03_OUTPUT.txt | 137 | `59acdfa2bb55d372ae248870e947d29070d977627b320653389930ddc6355d37` |
| CASE_04_RESULT.json | 3977 | `e4d918fb9c8353eb0a770ad615b9e5589e3f6770d5f013c56c2f0ae15e9ab1ea` |
| CASE_04_OUTPUT.txt | 139 | `3de180ed84241f973b344bfbf870164f19993744e34f987c05bd58ceca5e59d2` |
| CASE_05_RESULT.json | 3074 | `89f770b4a005ff2cf7f4747e661f3a821cc0684d36bf953dcfc04fc03a6070a7` |
| CASE_05_OUTPUT.txt | 140 | `70ec72b3dc9ec1956f983a21723da2054935d5f26c69c82e3a5e2475411b0892` |
| CASE_06_RESULT.json | 2643 | `3e8965862356ae34a50af8692f8feb57b2afc838fab5b1158d412ec6d3bf2150` |
| CASE_06_OUTPUT.txt | 61 | `ebce085d76bf2908fbff91c40efeb31806c1ef9ce14c57f66f1f4028f82af50f` |
| CASE_07_RESULT.json | 2914 | `a6449f524226a8d76a3c873f5e95802903997e2d9713df48e98435edff227785` |
| CASE_07_OUTPUT.txt | 66 | `38e9094ab230fb5459fef809dbce08235c327357e25c83bdba213e61a595ccba` |
| COMPLETED.json | 27087 | `77c7b206da3f341d95bf99b9e9deb19cb4110cd3901be61890f6bb3dad106f41` |

## Frozen custody reconciliation

Accepted archive:
`docs/research/cad_engine_v1_synthetic_checks/owned_process_supervision_v2_host_frozen.3tKI6J/SOURCE.tar`,
91136 bytes, SHA256
`26ed1ad646cba5f990fda61d083f87ab52f01492144807499eef8a30485bc27c`.
Independent inspection found exactly five unique expected regular payloads,
86588 bytes total: four sources and exact PRE. Every payload matches its live
counterpart and declared binding. No filesystem extraction occurred.

All four source bindings in STARTED, COMPLETED postflight and live files agree.
They total 73524 bytes and retain canonical source-map digest
`5abb395b3e54146801515d6f01d6f48a7131abaa53b72d5a0420ae70998926c9`.
Exact individual paths/bytes/hashes remain those in the accepted host PRE.

The host PRE is 13064 bytes, SHA256
`7d73c41f85d6f30f12b213da327b3e1b1bc4e1e17f4641f8d3ad1a5c613b630f`.
Its live bytes and STARTED/COMPLETED declarations match. The independently
recollected runtime matches both retained objects with digest
`50bb15d426d20537f5aa33dace33610ff07eee9bf45008a1556c44a61bf9a587`:
same CPython 3.11.15 executable/stat identity, darwin/arm64, kernel 25.5.0 and
exact kernel-version string. The root initializer remains absent. This metadata
is not a complete interpreter/libproc/shared-cache binary attestation.

## Next decision and prohibitions

Accept this audit as an exact retained **failure**, preserving 6/8 and all
eight records. V2 host qualification did not pass, so it must not be promoted
to the benchmark runner on the strength of the six favorable cases or mock
success. Do not rerun this terminal gate or silently weaken its frozen failure
rule. Any successor should be separately versioned and justified from the
observed transition-uncertainty failure, with source review, discriminating
mock controls, bounded receipt evidence and new exact protocol/PRE before any
authorized execution.

The required engineering work remains distinct from OS containment, arbitrary
race proof and physical accuracy. No stale PID cleanup, old-attempt mutation,
private-cache access or physical benchmark is authorized by this POST. The
broader engine/F1 goal remains open; this host failure creates no new score.

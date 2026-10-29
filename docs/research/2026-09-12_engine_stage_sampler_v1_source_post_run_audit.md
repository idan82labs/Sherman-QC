# Engine stage sampler V1: independent exact POST

Date: 2026-09-12. Decision: PASS for the declared invented-only sampler
qualification: 26/26 ordinary passing phase triplets, exact publication and
unchanged source/runtime/PRE/archive custody. This does not qualify production
30-second cadence, arbitrary blocked writes, runner integration, physical stage
diagnosis or engine accuracy. No rerun occurred in this audit.

The reviewer authored the independent PRE but neither candidate nor tests.
This is AI technical audit, not human approval. I read the full three released
terminal files, verified the accepted archive and all bound live sources/PRE,
and independently collected stdlib-only runtime metadata. No target import,
collection, thread, pipe, native operation or process signal/query was executed.

## Terminal and exact retained evidence

Root reports full read/acceptance of the exact PRE, archive verification in
5b592d, one sole invocation41bba1 returning exit0, outcome/publication gates
true and through-inventory wall approximately0.3078475 seconds. Root external
process assessment638842 found no Sherman worker. These are root-supplied
actual terminal/process observations, not a new process assessment by this
reviewer. Their released final-stdout file bindings match the independent
measurements below.

Attempt:
`docs/research/cad_engine_v1_synthetic_checks/gold11_engine_stage_sampler_v1-2o06u6p8`.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| STARTED.json | 2746 | b912b6146f1d38a08c64f81893130f7082360d942523cf2052289e407d123174 |
| OUTPUT.txt | 3154 | 2f15cae97c0d060c014376cf86ebc21044f91359cfc44bce1287fdda6daa073e |
| COMPLETED.json | 6311 | 1d1123a26e50eec355475ed6b48aa86a134a81f67a0700c966239350e4548c3d |

The exact names-only inventory contains these three names, totaling12211
bytes. Stable descriptor/path identities, regular no-follow reads and byte
counts matched. Canonical sorted compact JSON name-to-{bytes,sha256} map digest:

`3f6df64cf74c2e6ab4e8b79c8d238627979d3123e5d93d7824b5744c8e707630`.

OUTPUT's sole bounded worker marker parses exactly to COMPLETED.worker.
The supervisor output hash and observed byte count match OUTPUT. Parent final
publication is a separate root-stdout/independent-inventory check; COMPLETED
does not self-certify its own final hash.

Accepted archive:
`docs/research/cad_engine_v1_synthetic_checks/engine_stage_sampler_v1_frozen.882syg2s/SOURCE.tar`,
118272 bytes, SHA256
`5f6703f2571549ea61f68d8020457aeb4bd30368efee7d8ae82eb321740cfa11`.
Independent inspection found exactly nine unique regular payloads,110787
payload bytes, exact expected names and byte-for-byte equality to all eight
live sources plus PRE. Nothing was extracted to disk or executed.

The PRE remains15637 bytes, SHA256
`048473c9140d58efb0be0b3670f978225f9401d8b0c583c0631873521da41810`.
All eight source identities remain unchanged,95150 total bytes, canonical
path-to-SHA map
`4f60feb080c8ac084b6290a2293365f94a48f4a6f039caa83217f12723379b31`.
STARTED, worker postflight, parent postflight, accepted archive and live source
bindings all agree. The PRE declaration/bytes agree across the same chain.

Fresh ordinary runtime metadata matches all three retained runtime objects:
Python3.11.15 on darwin arm64, exact executable/stat identity and installed
pytest9.1.1 initializer5644 bytes/SHA256
`ee37885c9583f843390684dfb496907f87a0905128e7a3cf4ad227f49ff34c2d`.
The complete runtime object is in the bound PRE and receipts; its canonical
digest remains
`2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689`.
This is ordinary-loader metadata, not complete transitive binary attestation.

## All26 outcomes reconciled

The receipt records collected26, passed26 and phase_triplets_passed26.
Failures, setup/teardown errors, skips, xfailed, xpassed, deselections and
collection errors are all0; invalid_report=false. Pytest and supervisor exit
codes are0. Forbidden loaded-module lists and parent postflight errors are
empty. Pytest's text summary is26 passed in0.07s.

I independently reconstructed the ordered26 node IDs from the17 frozen test
functions and literal decorators, without collection or target import.
The canonical node-list digest matches the actual worker:

`5860837909db2471b6b0692cfe321f01411a2456281ef0158f5dd5750feccea1`.

This covers the frozen lexical source/no-I/O controls, invalid scopes, bounded
location-only stack traversal,4096/4097-byte serialization boundary, whole-stack
overflow refusal, partial/invalid/eight-call writes, precharged attempts and
incomplete-byte accounting,128-attempt/524288-byte stream control, zero-sample
and terminal lifecycle semantics, main-thread ownership, startup failure and
live-thread uncertainty, plus the one real-thread control. The fake stream
case is not a real512KiB filesystem write or a64-minute timing run. Source
review of premises and intended error sites remains in the exact PRE; the
new outcomes establish execution of this fixed suite, not universal coverage.

## Real-thread record and limits

The unique retained short-thread summary is358 bytes including its marker and
newline. It records exactly:

- one sample,526 complete sample bytes, confirmed stopped thread;
- artificial test interval0.01 seconds and nominal event wait1.0 second;
- unchanged nominal finish join0.25 seconds;
- production interval30.0 seconds;
- scope INVENTED_SHORT_THREAD_CONTROL_NOT_PRODUCTION_CADENCE_OR_ENGINE_STAGE_EVIDENCE.

Pytest progress characters precede the marker on the same line; I parsed the
unique fixed marker and complete following JSON. The summary, not the raw526-
byte pipe sample, is retained in OUTPUT. The passing frozen test asserts that
its original pipe sample is one complete <=4096-byte JSON line, observed main-
thread stack with at least one named test-source frame, sample_index0, matching
attempted/written byte accounting, no locals/geometry claims, no partial-byte
uncertainty and unchanged production policy. This audit did not reconstruct
or regenerate that original sample.

The control used its own fresh nonblocking pipe and one daemon. Its wrapper
stopped sampling after the first complete write; reads and descriptor closure
followed a stopped-thread receipt. This demonstrates ordinary short-interval
thread scheduling, one write and stopping/accounting on the declared host.
It is not long-running30-second scheduling or arbitrary blocked-descriptor
recovery. Fake live-thread controls preserve uncertainty; the actual control
did not need such a fallback.

## Whole-worker resources and quiescence

| Metric | Actual | Frozen limit |
| --- | ---: | ---: |
| Worker CPU seconds | 0.226153 | 20 |
| Worker HWM bytes | 51642368 | diagnostic |
| Sampled group peak RSS bytes | 51642368 | 536870912 |
| Supervisor wall seconds | 0.302515666000545 | 30 |
| Work stopped seconds | 0.28775874990969896 | 25 |
| Complete output bytes | 3154 | 1048576 |

These cover the whole suite, not an isolated sampler-performance benchmark.
Parent prepublication time is0.30752787506207824 seconds; root reports
through-inventory time approximately0.3078475 seconds. Parent pre/postflight
and synchronous publication I/O remain outside the30-second supervisor clock.
Sampled RSS is not an OS allocation cap or hard real-time guarantee.

The only observed group member was20183. Retained phases show work stopped
at0.28775874990969896 seconds, reap at0.28776158299297094, empty-group check
at0.30241791578009725 and terminal decision at0.302515666000545.
process_group_confirmed_empty=true and requires_group_attention=false.
budget_stop, primary_error and cleanup_error are null. Both worker and parent
outcome gates are true; actual final stdout reports final_publication_gate=true.
No source/runtime/PRE drift or publication discrepancy was found.

The private cache path appears only as retained receipt metadata; its contents
were not traversed, statted or read. No old attempt, scientific input,
raw/Gold/heldout/email content or unrelated process was accessed. Only this NEW
POST was written; frozen candidate/tests/PRE/archive/attempt remain unchanged.

## Decision and next boundary

Accept the exact26-outcome invented sampler qualification and its publication
custody. A separate source-reviewed runner connection may consume this evidence;
this POST does not itself release that integration or any physical diagnostic.

The sampler is observational: no frame locals/array values or inference result
are inputs, no sample certifies stage completion, and zero/last samples cannot
prove absence of later work. Logging overhead remains charged to any future
experiment. Generation-aware production process custody, physical input/
reference scope and any consumed diagnostic/full score require their own
frozen gates.

The separate94-outcome integration failure remains terminal21/73, including
its four wrong-reason passes and narrow dense-support pass. No frozen gate is
rerun or promoted here. No new F1, physical accuracy or engine-completion claim
follows from the sampler PASS.


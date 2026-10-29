# Engine stage sampler V1: independent exact source PRE

Date: 2026-09-12. Decision: GREEN for this exact invented-only 26-outcome gate,
conditional on root's full read/acceptance, verified archive and separate
no-overlapping-heavy-writer check before the sole invocation. No target import,
collection, test, thread, pipe or native operation was executed in this review.
All 26 outcomes remain prospective. This is AI technical review, not human
approval, runner-integration qualification or physical-input authorization.

I am independent of the sampler and test authors. I read the complete sampler,
test file, design, implementation review, harness and protocol. The complete
unchanged V12 source had been read in the earlier exact short-worker reviews;
its exact hash is rechecked here and only its three named AST functions may
be reused. Standard-library source reads, AST parsing, metadata and arithmetic
checks found no remaining blocking defect in the declared seam. No source or
test was edited; only this NEW PRE was written.

## Exact eight-source identity

Paths are relative to /Users/idantal/ShermanQC/FullProject. All eight are unique
regular files, totaling 95150 bytes. All Python files parse by AST;
no target AST was compiled or executed. All literal harness pins match.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `docs/research/2026-09-12_engine_stage_sampler_v1_design.md` | 3991 | `4cd71e3e8f1d40086c54b5207d6a2af18ed6ed51415a1d92c8df83fd9a819cd1` |
| `docs/research/2026-09-12_engine_stage_sampler_v1_source_protocol.md` | 6754 | `801e61520547db8d3848e14206b593821fe5c8177ee29c5c227193c984a3ff20` |
| `docs/research/2026-09-12_engine_stage_sampler_v1_source_review.md` | 12652 | `c0514f957b88e429184cfb7b0c64b1ede7ccfb7759a74edc974847b51467b633` |
| `scripts/check_cad_guided_connected_v12.py` | 19437 | `0003b32fb4a605cfb82a6edd167aa8d2cc024303e50921928efd21666d4df694` |
| `scripts/check_gold11_engine_stage_sampler_v1.py` | 23179 | `814511ce369893f3a126ccbea197321dd08333b6a1bcf777ee38ab9c5cee7c8b` |
| `scripts/gold11_engine_stage_sampler_v1.py` | 8142 | `80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0` |
| `tests/__init__.py` | 24 | `5332e0ec7edf6ab084943384a46bb543762f0401b94ea5c4276a27fc63ef211b` |
| `tests/test_gold11_engine_stage_sampler_v1.py` | 20971 | `455b75f37ac238512d40d78b28da3661d3d7d149d5ac0c450eb74903a77033c2` |

SHA256 of the sorted-key compact JSON exact path-to-SHA string map, without
newline, with allow_nan=False and default ensure_ascii:

`4f60feb080c8ac084b6290a2293365f94a48f4a6f039caa83217f12723379b31`

Do not substitute the table's {bytes,sha256} objects for that marker map.
The only imported project implementation is the sampler. Its imports are
json, os, pathlib, resource, sys, threading and time. The test adds stdlib and
pytest, plus a named read of this same sampler source for static assertions.
No backend, geometry, NumPy/SciPy, OCP/Open3D or old test module is required.

The private inert scripts namespace bypasses scripts/__init__.py; an inert
tests package bypasses test/root initialization. The sampler has no project
dependencies, avoiding the duplicate class-identity problem found in the
separate integration gate. Private fixture modules are removed in finally.
Root __init__.py is absent. No files or scientific inputs are discovered by
the sampler's lexical source-map validation.

## Denominator, premise checks and deterministic coverage

Exactly one test file is selected. Literal decorator expansion gives
17 functions and 26 outcomes, not a runtime collection result:

| Test suffix after test_ | Outcomes |
| --- | ---: |
| `static_source_and_fixed_observational_policy` | 1 |
| `source_scope_is_exact_lexical_allowlist_without_io` | 1 |
| `invalid_frozen_source_scope_is_rejected` | 4 |
| `stack_names_only_exact_sources_and_never_locals` | 1 |
| `stack_has_exact_twelve_frame_and_sixtyfour_walk_caps` | 1 |
| `full_line_byte_boundary_and_whole_overflow_refusal` | 1 |
| `partial_writes_complete_exactly_with_suffixes` | 1 |
| `zero_negative_bool_or_excess_write_fails` | 4 |
| `eight_write_call_cap_and_os_error_never_retry` | 1 |
| `attempt_precharge_and_incomplete_byte_accounting` | 3 |
| `exact_128_attempt_and_524288_full_line_stream_caps` | 1 |
| `zero_samples_and_finish_before_start_are_honest_terminal_state` | 1 |
| `start_finish_is_single_owned_daemon_and_policy_copy` | 1 |
| `thread_start_failure_is_separate_precharged_and_not_retried` | 2 |
| `main_thread_ownership_and_constructor_validation` | 1 |
| `still_live_thread_is_explicitly_incomplete_without_signals` | 1 |
| `one_short_self_owned_real_thread_control` | 1 |

The harness requires 26 unique collected nodes from only that path, 26
ordinary passing calls and complete passing setup/call/teardown triplets.
Zero failures, errors, skips, xfail/xpass, deselections, collection errors or
invalid/duplicate phase reports are permitted. An empty xfail reason remains
an xfail marker. No subset qualifies.

I explicitly checked constructor premises and the intended failure sites,
rather than counting a broad exception assertion as evidence by itself.
The normal invented sampler uses absolute ASCII root /invented, one relative
engine.py path, exact integer fd91 and the actual main thread on the frozen
Darwin runtime. That descriptor is never used for a real write: relevant tests
replace os.write or _write, stop before output, or fail at the deliberately
selected frame/serialization stage. The sole actual write uses the real
control's fresh pipe descriptor instead.

Invalid source-scope cases call _sources directly with one invalid dimension;
there is no preceding policy constructor that can mask the intended rejection.
Write rejection tests check exact error categories and actual write-call counts.
Injected frame/wire/write failures assert samples_attempted before raising and
then verify distinct completed-byte/failure receipts. Constructor and startup
failure controls count the attempted factory call and verify no retry.
Ownership controls mutate only the identity query temporarily and check that
start/finish state was not advanced. Other negative controls use explicit
state or reached-call assertions. This does not promise every possible failure
path is covered; it avoids the specific wrong-reason exception pattern exposed
by the separate failed integration gate.

The fake 70-frame chain requires exactly 64 code/link inspections, 12 copied
frames and 52 omitted named frames. Only allowlisted exact code filenames,
bounded function names and line numbers are copied. A raising f_locals property
and source AST checks guard against locals/globals inspection. Returned records
contain no frame objects or array values, and explicitly deny a complete-stack
or stage-completion claim. Equal basenames elsewhere and suffixed filenames
are not admitted.

The independent JSON boundary test constructs exactly4096 bytes including
newline, then4097, and checks whole-stack replacement by a small refusal
record. Unicode JSON escaping counts toward that cap. The 128-attempt/524288-
byte test uses a fake serializer/write sink and real sampler accounting; it
does not wait64 minutes or write512KiB to disk. Real serializer boundaries
are separately tested. Eight write calls are permitted, including completion
on the eighth; invalid progress, an incomplete eighth call or OS error stops
without retry. Partial failed-line bytes are deliberately not included in
completely_written_sample_bytes and remain explicitly unknown.

Start construction and thread.start are inside the sampler's guarded diagnostic
boundary; start_attempted is set before either. Main-thread ownership is
required at construction/start/finish. Finish-before-start prevents later
startup, repeated startup is rejected, and a still-live thread after the join
is not called stopped. finish returns a copy of the production policy.
A failure receipt is separate from inference and cannot be read as proof that
engine work did not happen. No inference result is supplied to the sampler.

## The sole actual-thread control

Exactly one prospective case creates one daemon through the actual public
sampler and one fresh empty nonblocking pipe. It creates no process and opens
no filesystem path. The real _write is forwarded by a wrapper that sets the
sampler stop event and test event immediately after its first complete line.
The next loop checks that stop before precharging another sample, so the
successful path cannot fill the pipe with repeated samples.

Frozen test-only timings are interval0.01 seconds, one event wait1.0 second,
and one normal finish join0.25 seconds. Production POLICY and original
INTERVAL_SECONDS are asserted30.0 before the test; only the test's module
interval is temporarily changed. The receipt must still declare production
30.0, and the <=4096-byte stdout summary explicitly names both timings.
This is ordinary scheduling evidence at an artificial short interval, not a
test of production30-second cadence.

The main thread waits within the named test frame. The actual sampler looks
up its owner in sys._current_frames and traverses only bounded code locations.
The emitted sample must have at least one named test-source frame, observed
status and index0; exactly one complete line, one attempted/written sample,
matching byte count, no diagnostic failure, no partial-byte uncertainty and
confirmed stopped thread are required.

After finish confirms stop, the test reads at most4097 bytes from its own
nonblocking pipe, then closes both descriptors. If stop is unknown, it does
not close/reuse the live writer's descriptors; the outcome fails and outer
worker custody must end it. No forced cancellation, process signal, polling
retry or additional successful-path join is hidden in the control. Startup
failure is also terminal for that control. OS scheduling and synchronous
operations remain outside any hard real-time guarantee.

## Fixed bounds and scope

One serial worker:20 CPU seconds,25 work seconds,30 supervision seconds with
nominal cleanup reserve,536870912 sampled process-group RSS bytes and1048576
complete output bytes. No resource/selection override or retry is exposed.
No other heavy Sherman writer may overlap. The sampler's128*4096 =524288-byte
maximum complete sample stream fits within, but does not replace, the overall
worker log cap. At most128 source names,12 retained frames,64 walked links and
eight write calls per attempted sample remain unchanged.

The sampler's process CPU and Darwin RSS high-water values are whole-process
diagnostics, not instantaneous or whole-tree memory. Sampling overhead still
consumes any later experiment budget. A missing/last sample does not establish
absence/completion of a stage; opaque native work may only expose its Python
calling location. A bounded join does not forcibly cancel synchronous I/O, and
a live-thread receipt is not an atomic final counter/output certificate.
The real control's nonblocking owned pipe does not qualify arbitrary blocked
descriptors or long-running sampling.

## Exact ordinary runtime

Fresh stdlib-only metadata, without importing pytest or the target, matches:

```json
{"distribution_versions":{"pytest":"9.1.1"},"executable":"/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11","executable_stat":[16777231,17404606,17257904,1783888533314804818,1783888533315379399],"implementation":"cpython","machine":"arm64","package_initializers":{"pytest":{"bytes":5644,"sha256":"ee37885c9583f843390684dfb496907f87a0905128e7a3cf4ad227f49ff34c2d"}},"platform":"darwin","python_version":"3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]","scope":"ORDINARY_LOADER_ENTRYPOINTS_METADATA_NOT_FULL_RUNTIME_BINARY_CLOSURE"}
```

The pytest initializer is
/Users/idantal/ShermanQC/FullProject/.venv/lib/python3.11/site-packages/pytest/__init__.py.
The large nanosecond stat fields above are exact decimal integers. Canonical
runtime-object SHA256:

`2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689`

This binds ordinary executable metadata and the installed pytest entrypoint/
version, not the complete Python runtime or all transitive binaries. No
dependency installation/change is part of the gate.

## Harness, archive and terminal custody

The parent requires -I-S-B application Python and admits the exact eight
sources/runtime/PRE before creating an attempt. The worker independently binds
its exclusive STARTED request bytes and the same sources/runtime/PRE/limits
before loading pytest/tests. Site/.pth, plugin autoload, conftest, project
configuration, bytecode and cacheprovider are disabled. The import hook blocks
backend, NumPy/SciPy, OCP/Open3D and root/backend initializer execution; final
forbidden-module inventory must also be empty. Source review and private
package construction remain required; the hook is not a hostile-code sandbox.

Only _remaining, _group_sample and supervise_native_context_test_v3 AST
definitions are taken unchanged from the pinned V12 source. Historical imports,
main and suite selection are not executed. This is an ordinary short-worker
numeric-group supervisor, not generation-aware production supervision,
confinement or atomic PID/group custody.

Worker outcome and parent publication gates remain separate. Parent postflight
and output reads require confirmed empty owned writers. Even an ordinary
failed test gets independent source/runtime/PRE postflight when quiescence is
known. Unknown custody retains attention and prevents a pass. Output must bind
the supervisor hash and exactly one bounded worker marker. Final exclusive
publication requires exactly STARTED.json, OUTPUT.txt and COMPLETED.json with
matching identities. Unknown names veto without their content being read.
Private supervisor cache is retained outside that namespace and stays opaque.
Parent pre/postflight and synchronous publication I/O are outside the30-second
clock; no hard OS allocation/I/O bound is claimed.

Before the sole invocation, root must fully read/accept these exact PRE bytes,
create a NEW external archive with exactly nine unique regular payloads (the
eight bound sources plus this PRE), verify every archived/live byte without
extraction/execution, and independently confirm no overlapping heavy worker.
Only then may root issue the exact protocol command with the accepted PRE SHA.

Every invocation is terminal. Missing receipts, failed phases, wrong reasons,
overrun, source/runtime/PRE drift, unknown writer or publication failure remain
failures. No cap raise, same-attempt retry, smaller denominator, frozen edit or
artifact overwrite follows from GREEN. A later independent exact POST must
reconcile all26 outcomes, the real-thread record, actual terminal stdout,
archive/live/runtime/PRE, resources and quiescence before qualification.

The separate94-outcome integration gate remains terminal FAIL21/73, including
its four wrong-reason passes and narrow dense-capacity pass. It is not being
rerun or promoted here. This sampler gate creates no engine accuracy evidence
or physical score, and cannot qualify its future runner connection by itself.
A consumed diagnostic or full Gold11 successor retains its own exact source/
input/reference custody and release prerequisites.

GOLD11_ENGINE_STAGE_SAMPLER_V1_EXACT_PRE_JSON={"decision":"GREEN","expected_outcomes":26,"runtime_binding_sha256":"2d3308cd2d3e9a109fed3b9fb283dcdb23eee99c343dfea6a12d9a679d6fd689","schema":"GOLD11_ENGINE_STAGE_SAMPLER_V1_EXACT_PRE","scope":"INVENTED_ONLY","source_map_sha256":"4f60feb080c8ac084b6290a2293365f94a48f4a6f039caa83217f12723379b31"}


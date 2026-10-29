# Engine stage sampler V1: independent bounded source review

Date: 2026-09-12. SOURCE-ONLY / 26 PROSPECTIVE OUTCOMES / NOT EXECUTED.

## Disposition and ownership

The sampler is a small observational diagnostic for a new runner and is not
currently wired into inference. This reviewer read the complete candidate and
design, independently identified four source defects, and authored the new
focused test source. Root owns the candidate and made its repairs. No candidate
or design edit was made by this reviewer. The reviewer is independent of the
implementation author, but not an independent reviewer of its own tests; root
and the exact PRE reviewer must independently inspect those tests before a run.

No remaining blocking defect was found in the repaired, bounded source seam.
This is not runtime qualification. No target import, pytest collection, test,
thread, pipe, geometry or physical diagnostic was executed by this reviewer.
Only named source reads, standard-library AST parsing, byte hashes and textual
checks were performed. No arbitrary input files, raw/Gold data, older attempt,
private cache or control artifact was accessed. Earlier frozen context and
projection files were not edited or rerun.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/gold11_engine_stage_sampler_v1.py` | 8142 | `80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0` |
| `tests/test_gold11_engine_stage_sampler_v1.py` | 20971 | `455b75f37ac238512d40d78b28da3661d3d7d149d5ac0c450eb74903a77033c2` |
| `docs/research/2026-09-12_engine_stage_sampler_v1_design.md` | 3991 | `4cd71e3e8f1d40086c54b5207d6a2af18ed6ed51415a1d92c8df83fd9a819cd1` |

## Concrete review findings and repairs

The initial candidate was not yet frozen. Four findings were reported directly
to root and resolved in the above final source:

1. `start()` constructed `threading.Thread` outside its exception guard. A thread
   constructor failure would therefore escape the diagnostic startup boundary.
   Construction and `.start()` are now both guarded, and `start_attempted` is
   charged before either operation. Failed construction cannot authorize a retry.
2. Constructor ownership originally used the caller's identifier without checking
   that it was the actual main thread. A helper-thread constructor could silently
   sample the wrong thread. Construction now requires actual main-thread identity,
   and `start()`/`finish()` require that same owner.
3. `finish()` before `start()` originally left startup permitted, producing a
   new daemon with an already-set stop event. A terminal `finished` flag now
   rejects startup after finish, while repeated finish remains a bounded
   observation and duplicate startup is rejected.
4. The valid root string `/` originally produced `//relative.py` by string
   concatenation and missed the corresponding exact frame filename. PurePath
   lexical joining now yields `/relative.py`; no file discovery or resolution
   was introduced.

The new tests cover both thread-construction and thread-start failure, all
ownership stages, the finish-before-start state and root `/`. The initial
defects were source findings, not observed failing test outcomes. No speculative
fix was applied to inference or frozen numerical code.

## Bounded semantics checked in source

`_sources` accepts only an exact tuple of 1 through 128 unique relative ASCII
Python filenames, each at most 192 characters, under an absolute canonical
ASCII root of at most 2048 characters. NULs, parent traversal, noncanonical
separators, non-Python suffixes and duplicates are rejected. The mapping is
lexical. There is no `open`, stat, directory discovery, symlink resolution or
source-content read in that function. An exact filename match is required;
the same basename elsewhere or a filename with a suffix is not admitted.

`_stack` inspects only code filename/name, line number and predecessor link.
It does not inspect `f_locals` or `f_globals`, hold a returned frame in the
record, or expose an unlisted code filename. A function name is capped at 64
characters. At most 64 frame locations/links are inspected and at most 12
named locations are copied. Additional named locations within the walked
prefix are counted as omitted. A remaining frame after the walk is explicitly
reported as truncated. Even an untruncated sample declares that it is not a
complete Python-stack claim. The sample is last-observed work, not evidence of
stage completion or proof that later work did not occur.

`_wire` encodes compact sorted ASCII JSON with its newline inside the 4096-byte
cap. JSON escaping, including Unicode expansion from code names, counts toward
the cap. An oversized connected sample is replaced by one small record with
the complete stack omitted, rather than keeping a misleading selected prefix.
The connected sampler's sample index is bounded to 0 through 127. Private
helper calls with arbitrarily invented unbounded indexes or blobs are not a new
public transport and are not claimed universally bounded by this review.

The event wait is 30 seconds in production. An expired wait precharges one of
128 attempts before frame collection, resource sampling, serialization or output.
A failed sample is not refunded or retried. Exactly eight OS write calls at
most may complete a line, with strict integer positive/in-range progress. A
zero, negative, boolean, excessive, short-after-eight or exceptional write ends
diagnostic sampling. At the connected line and attempt limits, the maximum
sample stream is 524288 bytes, inside but not replacing the worker's complete
log budget. Fully written overflow-refusal records count as written samples;
that counter is not the number of valid observed stacks.

`completely_written_sample_bytes` deliberately excludes any bytes from the
failing partial sample. For example, one 8-byte complete line followed by two
bytes of a failed line leaves the counter at 8, not 10. The failure receipt
declares partial bytes unknown. This is a lower-bound completed-line count,
not a claim that only those bytes reached the shared log. Startup failures also
conservatively mark diagnostic byte completeness unknown. A live thread after
the nominal 0.25-second join remains explicitly incomplete; its counters are
not an atomic final output certificate. Only external writer quiescence permits
later log inspection.

The sampler does not receive or mutate an inference result. Its receipt always
states `inference_result_modified=False`, and zero samples explicitly mean no
timed sample, not no engine work. It neither signals processes nor closes the
caller-owned log descriptor. Its resource values are whole-process CPU and
Darwin process high-water RSS, not instantaneous or process-tree memory. CPU,
RSS and output overhead still consume the future experiment's ordinary budgets.

## Exact prospective denominator

Literal AST expansion gives 17 new functions and exactly 26 outcomes. No test
was collected to obtain this count.

| Test suffix after `test_` | Outcomes |
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

All but the last control use deterministic invented frames, event/thread fakes
and captured writes. A frame's locals property raises if touched. The 70-frame
chain checks exactly 64 location/link accesses, 12 copied frames and 52 omitted
named frames. The 4096/4097 boundary is independently derived from JSON bytes
including newline. The 128-sample case uses a fake 4096-byte serializer/write
path and checks exact attempt/byte accounting; it does not wait 64 minutes or
write 512 KiB to a real file. Real serialization boundaries are checked separately.
Failure controls precharge at frame access, serialization and a partial OS write.

## One short real-thread control, explicitly not production timing

The final test is the only prospective real-thread control. It creates one
self-owned daemon through the actual sampler and one fresh empty nonblocking
pipe. No filesystem path is opened. The real `_write` writes one bounded line
to the pipe, then the wrapper sets the sampler stop event and signals a test
event. Stopping after the first completed write prevents pipe filling.

Its explicit proposed timing is:

- Test-only sampling interval: 0.01 seconds.
- Main-thread event wait: at most 1.0 second nominally.
- One unchanged sampler finish join: at most 0.25 seconds nominally.
- Exactly one complete sample; no retry, polling loop or child process.

Only the module's interval used by the test is temporarily monkeypatched; the
frozen production POLICY continues to declare 30 seconds. A bounded stdout
record explicitly labels both the artificial test interval and production
interval so this receipt cannot be mistaken for 30-second cadence evidence.
The sample must contain the named current test source in the real main-thread
stack. After confirmed stop, the test reads at most 4097 bytes from its own
nonblocking pipe, checks exactly one bounded line and closes both descriptors.

If finish cannot certify stop, the test fails and does not close/reuse a live
writer's descriptors. External worker custody must terminalize that failure;
there is no forced thread cancellation or hidden extra join/retry. OS scheduling
and synchronous Python operations are not hard real-time guarantees. The finite
control demonstrates ordinary thread scheduling and stop/write accounting,
not long-run production sampling or arbitrary blocked-descriptor recovery.

## Import closure, prospective gate and integration boundary

The only project module imported by the new test is the sampler, through a fresh
private inert scripts package. `scripts/__init__.py`, backend and legacy tests
are not imported. Candidate imports are only `json`, `os`, `pathlib`, `resource`,
`sys`, `threading` and `time`. The test adds ordinary standard-library modules
and pytest, plus a named AST read of the candidate source. It does not need
NumPy, SciPy, OCP, Open3D, an engine initializer or any physical input. The Darwin
constructor contract is retained; this is not a skipped cross-platform suite.

Root owns a future exact harness/protocol and independent PRE/archive. A small
single-worker gate consistent with existing source-check wrappers is proposed:
20 CPU seconds, 25 work seconds, 30 supervision seconds, 512 MiB sampled RSS
and 1 MiB output, with exactly 26 setup/call/teardown outcome triplets. These
are prospective caps, not a prediction or execution release. The exact test
timing above, source bytes, ordinary runtime identity, import boundary and all
writer/descriptor stop conditions must be independently frozen before import.
The source closure is small but must also include the future harness/supervisor,
protocol and PRE; this review does not preauthorize whichever wrapper is chosen.

Qualification would permit only the separately reviewed new runner connection:
start after source/runtime/input binding, immediately around the engine call,
finish in its finally path, and retain the separate diagnostic receipt in the
already bounded worker log. Inference bytes, scoring projection and acceptance
criteria must remain unchanged. No arbitrary log path or broader source map,
30-second policy change, CPU cap raise or physical diagnostic is released here.
Samples can miss short stages and opaque native work; a last observed Python
location is not a completed stage or proof of a historical cause.

This is a focused diagnostic addition for the approved failure analysis, not a
new all-positive geometry prerequisite or an F1 improvement. Root's independently
frozen small consumed diagnostic remains the next physical evidence boundary;
all old terminal attempts remain terminal.

# Bounded engine-location sampling for the successor runner

Date: 2026-09-12. NEW source-only design; not tested or integrated.

The terminal V5 CPU-limited arm returned no workflow report, so its deepest
internal stage cannot be reconstructed. The successor needs observations during
execution without changing frozen geometry functions, injecting a measured
answer, or requiring a result to return successfully before diagnostics exist.

`scripts/gold11_engine_stage_sampler_v1.py` provides one optional runner-owned
daemon thread. Every30 seconds it snapshots the worker's main Python thread,
copying only code locations whose absolute filename belongs to the runner's
explicit frozen source map. Each location contains relative source filename,
function and line number. It never reads frame locals, arrays, scan/CAD filenames,
Gold, input bytes or files. It names no unlisted runtime frames. It is a diagnostic
consumer, never an inference or scoring input.

The fixed prospective limits are128 samples,12 named frames per sample,64 frame
links inspected per sample and4096 encoded bytes including newline. Total sample
stream is at most524288 bytes, leaving part of the existing1MiB per-leaf log cap
for ordinary output. A too-large sample discards its entire stack and writes an
explicit small refusal record; it does not mislabel a selected frame prefix as
complete. Sampling is precharged, with no retry/refund. At most eight write calls
complete one sample; a partial write/error makes diagnostic completeness unknown.

Each sample records elapsed wall time, whole-process CPU and Darwin process RSS
high-water bytes. These are not instantaneous memory, whole-tree resources,
calibrated uncertainty or proof of stage completion. Zero timed samples means
none were observed, not that the engine never worked. A last frame before a CPU
kill identifies last observed work only; later stages remain unknown. Samples
may miss short stages and cannot resolve work performed wholly inside an opaque
native call beyond its calling Python location.

Start immediately before the actual engine call, after source/runtime/input
binding. Stop in its `finally` block and retain the sampler's separate receipt in
the already bounded worker log. Inference RESULT and scorer projection remain
unchanged. Thread startup or sampling failure is reported separately and does
not turn an otherwise valid measurement into a fabricated unavailable geometry
answer. CPU/RSS/log overhead still belongs to the frozen experiment budget.

The thread uses the existing worker log descriptor, not a new filesystem path.
Its final join is at most0.25 seconds nominally; if it is still alive, the receipt
says so and does not certify complete output. Synchronous OS writes are not
forcibly cancelled by this timeout. The external owned-process supervisor and
log cap remain responsible for the complete worker lifetime. Any later POST
reads logs only after that supervisor confirms writer quiescence. No process
signal, stale PID operation or external cleanup is part of this sampler.

Before integration, independent source review and focused invented tests must
cover exact source-path allowlisting, no locals access, frame/walk/byte limits,
whole-stack overflow refusal, partial/zero/error writes, precharged sample count,
stop/start failures, zero samples, and accurate incomplete receipts. Include one
short self-owned real-thread control if independently frozen; fake-thread tests
alone do not demonstrate scheduling. The fixed production30-second policy must
not be silently replaced by the test timing. No sampling run, target test or
physical diagnostic is authorized by this design.

This is a small diagnostic addition for the already approved failure analysis,
not a new research phase. It does not fix registration, increase F1, prove that
the historical race is gone, or replace a future all-case benchmark. The three
previous failures stay terminal; only a separately frozen successor may use it.

# V2 guarded runtime import smoke — 2026-09-09

## Prospective scope and prerequisite

This separately versioned, small smoke checks actual guarded imports and tiny
owned temporary-file cleanup. It is not scan/CAD parsing, inspection, geometry
evaluation, scoring, physical input review, source freeze, production promotion
or an all-case benchmark. No physical authority/control artifact is opened.
The existing V2 pure test/protocol remains byte-unchanged, including its
unconditionally skipped old smoke placeholder.

Root reports the prerequisite V2 pure suite has passed 89 controls with one
declared skip under its independently reviewed gate. That does not by itself
release this new native smoke. Require a new full source/closure and supervisor
review, then this file's pure gate, before the single actual smoke invocation.
The author has not imported, collected or executed these new tests or any native
module. A native success is not presumed here.

## Source and runtime binding

Test source: `tests/test_gold11_mesh_guard_smoke_v2.py`, SHA-256
`c56623967119a201b2c4db296f3e950d728642e48f0b28ce8c0a575b438d8e3d`.
Runner source: `scripts/run_gold11_mesh_development_v2.py`, SHA-256
`94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422`.
The test's literal `SOURCE_BINDINGS` contains the exact 24-file source closure:
this runner and its 23 named `backend/` inference files. It must match the
runner's ordered inference-leaf source denominator. No review, protocol, scorer,
catalog, app/package initializer, manifest or control file is copied.

Root's two additive harness routes bind those 24 files, the new test, this
protocol and the harness itself: 27 source files. Before execution and after
terminal completion require identical declared live-source digests. This smoke
does not emit runtime version metadata and makes no runtime-version or binary
attestation claim. The separate physical freeze captures its own runtime
metadata; that responsibility is not satisfied by this smoke. The proposed
harness revision is SHA-256
`4f2a778150fa106d7e9c8155333a12d010eace0020b2e9c39f9836b7be8cce19`.
Any later harness/source change requires a new exact review and binding.

The parent smoke fixture reads only the named runner source, checks its pinned
digest, and imports its inert stdlib declarations without calling its CLI. The
actual smoke copies exactly the 24 named files into a new pytest-owned snapshot.
Each source read is bounded at 2 MiB, the total at 48 MiB, with regular-file,
no-symlink, stable-read and pinned-hash checks through the existing reader.
An independent exact-byte/digest check precedes publication. All 24 admitted
blobs are staged in bounded memory before any source file is written, so a late
source/hash admission failure publishes no successful prefix. Existing snapshot
directories are refused. A storage exception prevents child launch; partial
owned writes after a storage failure are not a usable admitted snapshot.

The copied runner is rechecked by the child before loading. Backend imports
resolve through a fresh inert `backend` namespace pointing only at the copied
closure. The application package initializer is not imported. Source copying is
not protected against a hostile local process or native filesystem tampering;
the existing Python audit hook is not an OS sandbox.

## Fixed case denominator

Prospective source count: **32 expanded tests**. No collection was executed by
the author; parent collection must agree with this declared denominator.

- 31 pure/source/mock cases: exact closure (one), full invented copy (one),
  missing/extra/nonhex binding refusal (three), early/late digest or byte-type
  refusal (three), fresh-snapshot requirement (one), exact/one-byte-over total
  source bound (two), strict new opt-in (five), child guard/order/no-inference
  source inspection (one), exact constructor set (one), inert initialization
  source check (one), mocked same-group launch/budgets (one), failed-child
  receipt/log metadata before refusal (two), malformed artifact-directory
  paths (five), exclusive fresh directory and symlink refusal (two), and
  unavailable-log metadata preserving the original child receipt (two).
- One actual guarded runtime import/cleanup test, skipped unless the exact new
  `SHERMAN_RUN_GUARDED_RUNTIME_SMOKE_V2` value is `1`. The old opt-in cannot enable
  it. The first source route must report 31 passed and one declared skip. The
  later actual route must report all 32 passed, after independent source GREEN.

The pure copy tests use only explicitly invented byte strings and synthetic
digests, all under the supervisor-owned pytest temporary root. They never copy
or import backend files. Mock launch tests never create a subprocess. No source
test invokes `freeze`, `execute`, physical authority, reference selection, an
inspection entrypoint or an actual native import.

## Single actual child

The child starts under isolated Python `-I -S -B -u`. It first imposes a
10-second hard/soft CPU limit, stdlib-loads the exact copied runner, and calls
`initialize_worker(snapshot, cache, "backend", 10)` before any backend/native
import. That routine installs the same runtime path, inert namespace and Python
I/O audit hook used by the candidate worker.

After the guard, it imports actual NumPy, SciPy, OCP `gp_Pnt`, the V8 workflow
API and all nine constructor types: V5 noisy policy/budgets, V4 CAD-patch
policy/budgets, V6 search policy/budgets, V8 initialization budgets, finite-
triangle budgets and V2 surface-registration budgets. It instantiates those
nine plus the V8 workflow policy using the unchanged runner configuration,
including native V1, tuple restoration for `[128, 1024]`, .05 declaration and
the 1,024 unsigned admission ceiling. It checks configured field values.
Neither the imported inspection function nor any geometry/scoring entrypoint
is called. Constructors accepting these values do not qualify the physical
noise assumption, correspondence schedule, registration or measurement.

The only native constructed object is `gp_Pnt(1, 2, 3)`. The only data payload is
the literal small string `invented guarded V2 runtime smoke`, written, bounded-
read and removed inside one owned `TemporaryDirectory` under the cache. The
test checks cleanup completion. No scan/CAD bytes, PLY/STEP parsing, source
datum, pose, case identity, original data, Gold or prediction enters the child.
It emits a small fixed JSON success record only after all these checks pass.

## Supervision and terminal rule

Both named parent routes are bounded at 60 seconds outer wall, 45 seconds worker
CPU, 1 GiB complete process-tree RSS and 1 MiB retained output. Run serially under
the reviewed harness with project conftest and plugin autoload disabled; no
unrestricted pytest or concurrent heavy/native task.

The child uses the unchanged runner supervisor with 20 seconds wall, 10 seconds
CPU, 512 MiB sampled child-tree RSS and 1 MiB output. It explicitly uses
`new_session=False`: it remains in the pytest worker's existing process group.
It creates no new session, detached group or group registry. Parent and child
remain covered by the outer process-tree/group limit. The existing inner
supervisor's kill/wait reserve is subordinate to the independent 60-second
outer limit. Native extensions are not allowed to create tasks or escape the
guard; the hook forbids Python-level external/process/group actions.

The actual route alone supplies a fresh absolute retained `ARTIFACTS` path in
`SHERMAN_GOLD11_GUARD_SMOKE_V2_ARTIFACT_DIRECTORY`. The test requires a normalized
absolute path and creates that final directory exclusively with mode 0700,
without creating parents, following an existing final symlink, overwriting an
existing directory or falling back to pytest temporary storage. Root owns and
retains its parent. This environment value is not forwarded to the child.
There is exactly one allowed artifact, `SMOKE_OUTPUT.txt`, capped at 1 MiB.
The parent supervisor directs combined child stdout/stderr there before
interpreting the child outcome; the child itself writes only inside its cache.
The file therefore survives pytest temporary cleanup even after a child failure.
If output exceeds its cap, the bounded retained prefix and output-limit receipt
remain a failure, not a claim to retain bytes beyond the declared cap.

Before any success assertion, the parent emits at most 4 KiB of metadata
containing the original supervision receipt and log status, byte count and SHA.
A missing/unreadable log is explicit `UNAVAILABLE` with null count/digest and
exception type only; the original receipt remains intact and the test fails.
No child output text or base64 is forwarded. No manifest file is needed or
created. Success requires zero child exit, no budget stop, the available log's
exact final JSON status, ten constructed policy objects, no inference executed,
and owned-temp removal.
Require the outer expected case count and no resource stop or 27-file drift.
Preserve the first terminal attempt, including any failure or skipped actual
test. Do not retry, raise caps, change source/fixture to rescue an executed
failure or call success merely because the pure route passed. A later actual
pass establishes guarded runtime-import compatibility only; physical bundle
audit, physical-development accuracy and floor/production release remain separate.

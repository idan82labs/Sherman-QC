# V5 runner source harness: independent source-only review

Date: 2026-09-11. Decision: **no remaining source blocker found after two
pre-freeze corrections.** Continue the declared finalization/PRE sequence;
do not execute this draft. This is not an exact PRE, test result, native
qualification or physical release.

I read the full new harness and protocol, compared the lifecycle mechanism with
the complete hash-pinned old V4 supervisor source, and reviewed the final
localized corrections. Static AST/source checks only were performed. No project
import, test collection/execution, FIFO probe, native operation, attempt/control/
scientific-artifact access or protected input access occurred. Existing active
connected-source files were not modified.

## Exact reviewed draft

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/check_gold11_mesh_runner_v5.py` | 30205 | `36714fda5dab2a716b0bc5f18249566745ea57dfcd73c96cc23147c01c2c1bf7` |
| `docs/research/2026-09-11_gold11_mesh_runner_v5_source_protocol.md` | 12267 | `d0c8041f7a2dd51c0865b780fe619d7086610b5e4cdb928eddbb4dc1b49066a6` |
| `tests/test_gold11_mesh_development_runner_v5.py` | 52489 | `ee7861235ad345d7d807112cf3471536252e9b55f9ee437d0de0098b91ae2f76` |
| Old supervisor, source oracle only | 10310 | `579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b` |

The harness still has `FROZEN_FOR_SOURCE_GATE=False` and both final runner/test
hash pins unset. Parent and worker require that source-frozen release before
target-source reads, dispatch or pytest import. There is no environment/test-
subset/budget override. Final source qualification must use the concrete
native-derived physical-runner timings and its final reviewed release bit,
not temporarily qualify false/unset declarations and change them afterward.

## Findings closed before freeze

1. The initial draft opened an expected file before checking regular type
   without O_NONBLOCK. A replaced FIFO could wait inside open before a deadline
   or type check. The final open now includes O_NONBLOCK with O_NOFOLLOW,
   followed immediately by the regular-file and byte-cap checks. This closes
   that avoidable malformed-file wait; it does not turn synchronous OS I/O
   into a hard real-time operation.

2. The initial observer could count a non-strict XPASS as an ordinary passed
   call. The final observer records `wasxfail` outcomes explicitly, and both
   worker and parent require zero `xfailed` and `xpassed`. A normal xfail
   also increases skipped; strict XPASS fails through the normal failure path.
   The reviewed 85-test source contains no xfail mark/call, so this was a
   robustness gap, not evidence that a prior/current result falsely passed.

Both changes were reviewed in source. No validation run or started attempt
existed or was retried. A separate observer/FIFO experiment is not being
claimed. If the observer is later generalized, a small invented-report matrix
covering pass, call/setup/teardown/collection failure, skip, xfail, non-strict
XPASS and deselection would be useful; it is not authority to execute a new
unfrozen test now.

## Source closure and early-execution boundary

Static resolution confirms **66 distinct files / 1,890,563 current payload
bytes**: the runner's 61 declarations plus two ordinary-constructor test
sources, this harness, this source protocol and the old supervisor oracle.
Every Python member parsed, and harness globals resolve without a missing
source-level dependency. The inherited 61-file backend/script closure and
25-function/85-outcome test review are documented in the separate
[V5 source review](2026-09-11_gold11_mesh_runner_v5_source_review.md),
SHA256 `98849b18d6d4cee254e43960f0d0480d6d1611c316f4c771773046cfdb0e7e39`.

The old supervisor is authenticated as a source-preservation oracle only:
its definitions, main and old tests are never imported or executed. The new
self-contained lifecycle code declares its own 120/175/180 limits; it does not
patch a 45-second receipt on an old object.

The parent checks exact final runner/test hashes and parses only their declared
AST constants/source list. It neither imports nor compile-executes the runner.
Its preflight requires the final runner release literal, seven positive finite
timings, exact integral CPU caps/reserve relationship, the test's final runner
SHA and exact 61-plus-five source equality. It does not itself approve physical
timing adequacy or the external independent PRE.

Sources are read through no-follow parent components and nonblocking final
opens, bounded to regular 1..2 MiB files, with descriptor/parent-entry identity,
size, mtime and ctime checks. Each full pass is bounded by 132 MiB returned
source bytes. Late errors/unattempted paths are retained and prevent positive
map equality. This is quiescent source custody, not an adversarial atomic
namespace snapshot. Tests' initializer is hash-bound and AST-inert; any
enclosing repository initializer object refuses before the worker and is
checked again afterward.

The isolated worker repeats preflight before importing pytest. Only the reviewed
site-packages path is added; pytest uses disabled external plugin autoload,
`-I -S -B`, `/dev/null` configuration, no conftest/cache and the one fixed test
module. That test creates inert backend/pure-script namespaces and extracts
only the reviewed ordinary constructors. Positive runner process entries are
not executed; OCP import detection additionally vetoes the worker result.
Interpreter/NumPy/SciPy/pytest/OS binaries remain outside repository hashes.

## Denominator, output and CPU/RSS evidence

The observer records collection, calls, setup/teardown and collection failures,
skips, deselection and expected-failure outcomes. Parent admission requires one
complete final worker receipt, 85 collected, 85 passed and zero of every other
disposition, pytest exit 0, no OCP import and the initializer guard. Worker CPU
must be finite/nonnegative and at most 120 seconds; Darwin high-water byte
semantics are explicitly required. Missing/duplicate receipts or wrong field
types/values veto rather than falling back to a success text line.

Observed output bytes are charged before writes. Short/uncertain writes and
close errors are retained; a positive requires exact retained length against
write-return accounting. More than 1 MiB observed output stops the worker even
though only the bounded prefix is retained. Complete OUTPUT is not read by
the parent until group emptiness is confirmed. Independent POST still must
read the full output and reconcile the observer's disposition with it.

Fixed prospective limits are **120 worker CPU seconds, 175 work seconds,
180 parent-monitored seconds, 2 GiB RSS admission and 1 MiB output**.
The parent clock begins before source preflight/attempt creation and is passed
into supervision, so preflight consumes—not extends—the 175-second worker
opportunity. Worker source preflight/import/tests are inside its CPU limit.
The five-second work-to-parent remainder must also cover cleanup/postflight;
it is a planning reserve, not a completion guarantee.

During work, group samples plus parent high-water are checked against 2 GiB.
Unsampled group RSS is null/unknown rather than zero. Final admission uses
reaped-child high-water plus parent high-water as a conservative extra check,
not a measured simultaneous tree peak; worker self-high-water is checked too.
No hard parent CPU ceiling is claimed.

## Failure cleanup and final artifacts

The copied polling/reaping structure creates one owned process group, uses
bounded selector/ps waits, refuses unexpected same-group children and attempts
owned-child reaping before selector/stream closes. Ordinary cleanup/close
failures veto. Known residual group members can be killed and resampled within
the remaining deadline; unfamiliar survivors remain explicitly unconfirmed.
The mechanism does not discover escaped groups or provide hostile-process/
native-extension isolation. The reviewed tests spawn no real subprocesses.

Popen, regular-file/pipe I/O, fsync and some OS calls remain synchronous.
The clock bounds monitored waits and postchecks, not uninterruptible OS calls.
A status written as PASS_PENDING_INDEPENDENT_POST cannot release evidence if
the later final inventory, clock/stdout check or observed parent exit fails.

A fresh exclusive attempt retains exactly STARTED, OUTPUT and COMPLETED.
Each administrative JSON is bounded to 256 KiB. Final enumeration checks all
names, including extras/dotfiles, requires exactly those three regular files
and hashes the exact published STARTED/COMPLETED and retained OUTPUT.
Unknown content is not read. No fourth artifact or self-referential COMPLETED
hash is created; the final inventory is reported on stdout for independent
POST binding. The separately retained private cache contains invented pytest
fixtures only and is not inspected or blanket-cleaned by the parent.

The full-namespace test checks 119 scientific paths, 46 issued/claimed records
and three finalization phases with a 256 KiB completed envelope. Its ordinary
invented footprint and explicit fake timing/process seams do not establish
maximum 128 MiB RESULT throughput, native physical runtime or OS-level cleanup
guarantees. Oversized/incomplete evidence remains a terminal negative.

## Next boundary

The final concrete runner/test pins, harness flag and protocol must be frozen
and independently checked before exact PRE and the **67-payload source-plus-PRE
archive**. No current draft invocation is allowed. Any final literal/diff change
requires corresponding source review; no begun limit may increase and no
failed gate may be retried or narrowed. This review neither changes the active
connected packet nor authorizes physical input/reference/control access.

No tests or scientific experiments were run by this review, and no new F1,
native qualification, accepted pose, manufacturing count or production claim
was obtained.


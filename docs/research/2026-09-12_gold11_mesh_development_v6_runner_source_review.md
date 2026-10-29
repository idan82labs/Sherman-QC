# V6 runner focused source review and prospective mock gate

Date: 2026-09-12. Status: source-only review and test authoring, not execution,
collection, qualification, PRE, physical authority, or a new score. The reviewer
owns only this NEW document and NEW
`tests/test_gold11_mesh_development_v6_runner.py`. Root owns the candidate runner
and design; no frozen source, attempt, or private cache was changed or inspected.

## Candidate and bounded review

Reviewed candidate `scripts/run_gold11_mesh_development_v6.py`: 166,049 bytes,
SHA256 7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80.
The candidate's RELEASE_AUTHORIZED remains False and its timing declaration is
UNRELEASED_V5_CARRYOVER_VALUES_NOT_V6_APPROVAL. No invocation is implied by this
review. Root's sampler integration changes supersede the original uninstrumented
V6 draft hash; tests pin the new bytes, not the old draft.

The predecessor V5 source was read fully during draft work. This review examined
the V6 changed custody, supervisor, context/projection, scorer-finalization and
outer postflight seams, and read both new sampler integration functions and
their worker/source/audit-hook dependencies. The sampler implementation itself
is under separate independent review/qualification; it is not imported, executed
or independently qualified by this test module. No physical input, reference
catalog, attempt control, runtime native library or private artifact was read.

No target import, Python compilation, AST execution, test collection, process
operation, signal or timer was performed while authoring this review. Shell
source reads, bounded named source hashes, textual comparisons and whitespace
diff checks are the actual checks performed. Test source below describes future
isolated execution, not an observed passing result.

## Found blocking issue and root-owned correction

The first sampler-wired V6 candidate called `stage_sampler_module(snapshot)`
from `engine_call_with_stage_sampler` after `initialize_worker` had installed
the inference worker's backend-only file-access hook. Loading the mandatory
sampler from `snapshot/scripts/gold11_engine_stage_sampler_v1.py` would therefore
fail WORKER_PATH_OUTSIDE_FROZEN_SCOPE before the engine call. The sampler was
also absent from the inference `leaf_sources` verification tuple. This was a
connected source defect, not a physical-run result or a test observation.

The reviewer reported it before freeze; root corrected the candidate:

- The inference leaf source tuple explicitly includes STAGE_SAMPLER_SOURCE.
- Worker order is authorize_leaf, exact pinned sampler load, initialize_worker
  (install backend-only hook), then the unchanged engine policy/input/call path.
- The verified module is retained in `_STAGE_SAMPLER_MODULE` with its exact
  source SHA. The engine wrapper requires that preload and performs no source
  read or loader call under the installed hook.
- `worker_audit_hook` remains source-equivalent to V5. There is no broad new
  scripts-directory read allowance, metadata discovery, or cache exception.

The current source closes this reported issue. Three mocked hook controls,
worker call-order/source-closure assertions, a hash-refusal worker control, and
missing/wrong-preload controls are included prospectively. None has run.

The reviewer also noted that a start() failure could be replaced by a later
successful finish() receipt. Root added a separate startup_failure_type field
to preserve that attribution even after finish; the constructor/start failure
controls now require it, without changing the engine result.

No additional blocking runner defect was found in this bounded source review.
This is not an assertion of exhaustive correctness or qualification.

## Test isolation and custody

The new test module has 29 test functions and exactly 132 prospective outcomes,
counted directly from its literal parameter tables, including the sampler
cross-product. It imports only standard-library modules and pytest. It does not
import the runner, any backend package, a scorer, the projection module, the
supervisor, artifact helper, sampler, or an old test module. There is no inherited
85-case denominator and no old suite rerun hidden inside this gate.

Five named source files are the entire test-time source-read allowlist: candidate
V6, frozen V5, and context V1/V2/V3. All are SHA-pinned. Source reads are bounded
at 256 KiB each and parsed as AST. An explicit assignment/function allowlist
extracts the candidate seams; no top-level import, CLI, main invocation, runtime
loader or unrestricted initialization is admitted. Context maxima use literal
AST data and an independently spelled exact V2 override expression; no context
function or numerical implementation is executed.

Operational tests retain literal RELEASE_AUTHORIZED=False and unreleased timing
in each detached namespace. A named `admit_mock` helper checks those values and
substitutes only that namespace's release guard with an event spy. Process,
authority, runtime, file, publication, native and clock seams are inert mocks or
fail-on-call sentinels. Separate guard-only copies change local flags/timings to
test rejection; they invoke only require_release, never a released entrypoint.
The 16 entry controls call real extracted definitions solely behind their actual
false guard, after asserting that it precedes every effectful statement.

LexicalPath supports path arithmetic but does not resolve, stat or create an
invented host path. Dedicated cache identity doubles supply invented metadata.
No temporary directory, filesystem write, signal, sleep, subprocess, native API,
real daemon or stack observation is needed by these tests. Mocked backend
constructors use a closed local import interceptor; they are never installed
into the real backend namespace. Context/projection wiring spies explicitly do
not claim actual constructor validation, numerical execution or real projection
reconstruction. Those belong to their separate exact qualifications.

Ordinary invented JSON payloads are bounded below 100 KiB. The largest admitted
namespace fixture declares all 11 invented cases, two arms, six invented CAD
identities, all candidate source slots, 44 required case terminal paths and 46
issued/claimed dispatch slots. Its content is invented metadata, not a source
freeze or physical-input attestation. The sampler oversized-receipt fixture is
only a 5000-character invented diagnostic. All times, process IDs, resource
numbers and file identities are explicit fake values.

## Exact prospective denominator

| Focus | Outcomes |
| --- | ---: |
| Nine preserved constants and three preserved functions | 12 |
| Sixteen blocked entries, six flag controls, four timing controls | 26 |
| Both-arm projection wiring, source maxima, constructors and drift | 8 |
| Strict leaf/outer V3 quiescence table | 20 |
| Cleanup state, identity, fresh creation and retention controls | 20 |
| Supervisor custody latch and later-dispatch veto | 3 |
| Darwin finite lifetime-high-water memory controls | 8 |
| Abort-before-reference, scorer-finally, coordinator and outer order | 12 |
| Full schema arithmetic, unknown cache veto and no automatic finalizer | 4 |
| Sampler preload/hook/hash and observational exception controls | 19 |
| Total | 132 |

This table is a planned collection denominator, not PASS counts. The eventual
harness must verify exactly these outcomes once, with no xfail, skipped cases,
unexpected collection, conftest initialization, autoplugins, cache provider or
bytecode output. A resource envelope still needs independent PRE. A bounded
starting proposal is 20 CPU / 25 worker-work / 30 supervision seconds, 512 MiB,
1 MiB log; those are not timing release for V6 inference or the physical run.
No harness or protocol was authored by this reviewer.

## What the assertions establish if they pass

Preservation controls compare complete CONFIG, LIMITS, certificate/position
declarations, seeded/composite maps and review/catalog/selected-reference hashes
to the pinned V5 AST. Returned-stage diagnostics and scorer_worker are unchanged;
selected_references differs only by its leading release guard. The fixed nested
case/arm loop iterators and both arm names are preserved. This does not execute
the reference selector or open any of its paths.

Both successor arms retain distinct v7/v1 and composite/v2 scheduling mechanisms
while explicitly carrying the same full V3 context request and V14 implementation
identity. Fake constructors verify exact supplied kwargs, effective declaration
round trips and independent drift refusal. Projection/reconstruction spies prove
that original result bytes and the full context/scan/schedule request reach the
V2 APIs without reencoding the result; they do not duplicate projection237.

Receipt controls reject wrong schema, unreaped/attention/escape/unknown identity,
cleanup/control failures and invalid counter types. Only the outer new-session
predicate requires group-empty True; leaf group state is not fabricated. Missing
RSS is never interpreted by these runner tests and no signal permission follows
from a quiescence result.

Cache controls exercise real extracted owned_leaf_cache and retention functions
using only invented paths/metadata. Unknown writers or failed derived retention
cause no metadata refresh, enumeration, result read or removal. Clean/no-dispatch
cleanup uses exactly the newly returned directory after identity validation;
identity or cleanup failure produces an explicit false cleanup result. Actual
retention positive controls require receipt and identity before enumeration or
read, and distinguish a complete retained result from no returned result.

Scorer controls preserve the final administrative receipt on supervisor,
retention and cleanup failure. Unknown writers are never passed to result
retention; failure does not produce SCORED. Real extracted coordinator logic
stops after the first faulty leaf and cannot reach a second dispatch, coordinator
memory check, reference access or scoring. Unknown writer custody leaves all 22
terminal slots explicitly nondurable; quiescent retention/cleanup failures may
materialize the fixed denominator but cannot become a score.

The mocked outer execute path checks whole-group quiescence before content
source postflight, terminal fill, artifact verification or inventory. It binds
the external coordinator cache path/identity before dispatch. Unknown custody
skips those content operations; source-audit failure survives subsequent all22
terminal fill and cannot produce FINAL_GATE. These are call-order tests with no
live coordinator, external cleanup, or source inventory.

The real extracted campaign schema has no coordinator_work/private-cache branch.
Unknown names still consume the unchanged scientific128 accounting and produce
metadata-only vetoes. Invented unknown directories are not recursed or read.
The complete schema's derived read ceiling remains below the unchanged 8 GiB
helper bound. The test asserts the original 256 KiB custody envelope cap, not
that every possible maximum receipt fits it.

Sampler controls require the verified preload before engine entry and preserve
the exact single entry call, arguments, returned object, and engine exception
identity. Constructor/start/finish/write failures and oversized diagnostics are
tested both with an engine result and with an engine exception. The separate
terminal log line is at most 4096 bytes including its prefix and suffix newline;
oversize is omitted whole rather than truncating an arbitrary JSON prefix.
No sampler result is merged into the measured engine result.

## Remaining limits and release blockers

The tests do not run supervisor V3, Darwin proc APIs, timers, signals, native
readers, backend geometry, real policy validation, the numerical context chain,
actual V2 projection, scoring math, real publication/fsync or process cleanup.
They do not repeat supervisor mock166/host8, context76, projection237, sampler
qualification or the separately terminal integration94 gate. Their finite
evidence must remain separate from physical accuracy and the terminal history.

The schema fixture proves small declared-path arithmetic only. It is not the
maximum 119-file retained campaign with all long process/partial/deadline/error
ledgers, nor a worst-case 256 KiB combined COMPLETED envelope. The candidate's
maximum-finalization/envelope blocker remains outstanding. No file count, cap,
denominator or log limit has been raised to make the small mock fixture fit.

Sampler stream 524,288 plus terminal receipt 4096 leaves 520,192 bytes inside
the unchanged 1,048,576-byte whole leaf log cap. This is arithmetic, not a promise
that ordinary/native output fits: there is no independent native-output quota.
Whole-log overflow stays terminal. The sampler only catches ordinary Exception
failures at this wrapper boundary; process termination and BaseException control
flow are not converted into successful telemetry. The separate
startup_failure_type field now preserves a startup exception independently of
finish. Whole omission on oversize or failed stderr publication can still lose
diagnostic evidence; POST must not infer complete telemetry from a missing or
incomplete diagnostic. None of this rewrites an engine result.

Independent full source review of the new test/harness closure, exact PRE and one
bounded mock gate remain necessary. The sampler and connected engine capacity
have separate qualification obligations. Concrete V6 inference timings, the
separately frozen small consumed diagnostic, physical input/reference authority,
archive, PRE and any all11/both-arm launch remain unreleased. No old attempt or
old cache is made reusable by this review.

## Source identities for handoff

The test source at this checkpoint is 52,033 bytes, SHA256
dbe4ea245acbb2f93cd0b8cf8fe4e59620b4f5ce68aad7a85cc20c759a49f3f5.
Its five source-read pins are:

- V6 runner: 7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80.
- Frozen V5: 71d55915db6e6fcb8ae7f8c658de7d7e0c4b93b553eb78413f3914b037e4e481.
- Context V1: 9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a.
- Context V2: d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663.
- Context V3: 9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d.

The reviewed root-owned V6 design is 17,868 bytes, SHA256
a943ad18577f1a4ee450227c9e879215fccb218dc94a167b0979ab17b96747f2.
Future qualification closure should freeze these six core source/test files,
this review and the root-owned design, plus the new harness/protocol and their
explicit supervising dependencies. No physical protocol or runner-source
authorization can be inferred from that mock-only closure.

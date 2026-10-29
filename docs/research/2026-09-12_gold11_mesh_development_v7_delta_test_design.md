# V7 delta: focused prospective test design — 2026-09-12

Status: source-only authored, not imported, collected, compiled or executed during
authoring. New test `tests/test_gold11_mesh_development_v7_delta.py` declares
21 prospective outcomes across four parametrized functions. This is not a pass
claim, physical release, freeze, or permission to overlap the running diagnostic.
The unchanged V6 132-outcome suite is neither imported nor recollected.

## Exact source-read and execution boundary

Future test execution reads only these two named target source files:

| Target | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/run_gold11_mesh_development_v7.py` | 169028 | `27737eff034891e8f7a6d09ba63f5bd17a3947c67ffd553ddfb73a931fef5cb3` |
| `scripts/run_gold11_mesh_development_v6.py` | 166049 | `7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80` |

Complete byte lengths and SHA256 values must match before AST selection. The V7
buffer supplies an explicit allowlist of 39 assignments and 23 functions. Only
those definitions/constants are compiled in a detached namespace at the future
gate. No V7 or V6 module import or top-level main execution occurs. The V6 buffer
is only the exact source-basis payload in the invented freeze store; no V6 code
is compiled or invoked by this test. No backend/helper/scorer module, native
library or external service is loaded.

Standard-library modules plus pytest suffice. Candidate namespace imports and
open operations are denied. Lexical paths never resolve/stat a host path;
`mkdir` is inert. Clocks, PID/group/parent values, environment, metadata,
descriptor closure, preflight controls, publication, supervision, cache creation,
reference selection and scorer entry are explicit in-memory doubles. Unprovided
I/O/process/native/reference operations raise a test failure. Real target paths
are used only for the two source reads above; all runtime paths are invented.

Candidate release stays false in every behavioral test. A detached function
double admits selected pure boundaries. The release control separately flips a
local Boolean solely to verify the timing refusal, then invokes no source or
profile operation. No source/environment bypass or execution authority is added.

## Fixed 21-outcome ledger

| Group | Outcomes | Actual seam and required observation |
| --- | ---: | --- |
| Release/basis/profile | 3 | Seven guarded entry points refuse unreleased source; local Boolean cannot release timings. Actual freeze/verify round trip precedes changed basis and profile controls, which require their exact errors. |
| Pre-start tree | 9 | Valid; root extra; spent root; missing source; file symlink; directory symlink; changed second enumeration; changed directory fingerprint; declaration conflict. Actual helper is reached from actual execute before STARTED. |
| Input binding | 6 | Actual Boolean True; False; absent field; None; integer 1; string `true`. Actual authenticated coordinator branch, original negative receipt retained, no subsequent dispatch or scoring on non-True. |
| Request/cap | 3 | Actual small all22 producer/consumer; exactly-cap admission; cap+1 refusal before scorer-cache entry. |
| **Total** | **21** | **Four parametrized functions; no inherited132 denominator.** |

All counts are prospective ordinary pytest outcomes, without skips or xfails.

Every tree variant first admits the same valid actual 70-source declaration.
The metadata fixture has seven known directories. Negative variants change only
their named predicate. The helper runs inside actual `execute()` with earlier
bundle/PRE/control I/O doubled; refusal must escape before any STARTED or other
publication. The valid variant deliberately stops its inert STARTED publisher
after recording that boundary. Actual execute returns its expected no-dispatch
publication failure rather than entering real coordinator work. Descriptor
open/close event lists must match for every branch, including refusals. Unknown
entry contents are never offered to a read seam.

Each binding outcome first exercises an authenticated positive coordinator with
22 invented negative receipts having Boolean-True bindings. The coordinator
records all22 and reaches only the scorer spy, establishing that scientific
negativity does not itself trigger the new administrative abort. Non-True cases
then use the same valid dispatch/cache/start premise, retaining the first original
receipt bytes and reason. They must attempt exactly one leaf, preserve 22 terminal
positions, omit INFERENCE_TERMINAL, skip the memory/scorer spies, set aborted, and
publish the exact binding-not-established reason. The actual extracted scorer
entry is then called and must reject the aborted state before any reference seam.
This is not a claim that an input hash actually mismatched or that a live writer
was quiescent.

## Fixture honesty and limits

The freeze/verify control uses actual candidate orchestration but an invented
70-slot source store. All slot bytes are inert named literals except the exact
V6 source-basis buffer. The local artifact/supervision pins are explicitly changed
to the digests of their invented store bytes so the actual hash checks remain
internally coherent. This does not qualify those helper bytes or change their
production source pins. Review-source parsing is an explicit stub yielding 11
invented cases; the unchanged authority parser is not requalified. No original
review script, source backend, physical input or reference is opened. The new
V7 source and V6 basis complete hashes remain independently verified by the test.

The request fixture uses actual frozen arm requests and actual
`make_scoring_request()`/`scorer_arguments()` with 22 exact supervisor-refusal
objects and 71 tiny invented reference-shaped rows across 11 invented cases.
The 71 count preserves denominator shape only: those rows are not physical Gold,
and no geometric matcher is invoked. The actual complete small request must
serialize below 128 KiB and carry all22 projection bindings. Both producer and
consumer retain the explicit V6 request transport schema, as the V7 design states.

For the two large-cap comparison controls, the real encoder is retained for all
ordinary objects. Only the whole, already consumer-validated scoring request is
replaced by `LengthOnlyEncodedRequest`, whose `__len__` returns exactly 192 MiB
or 192 MiB plus one. It is not bytes, cannot be passed off as serialized data,
and allocates no large buffer. The actual `score_after_terminal()` ordering and
comparison run: exactly-cap reaches an inert scorer-cache boundary which raises
an explicit stop before creation/dispatch; cap+1 raises
`SCORER_REQUEST_BYTE_BUDGET` before that boundary. The exact-cap receipt-finally
path is observed without a worker. The cap+1 case first reaches that same
exact-cap boundary, clears only its event list, then changes the length by one.
The reconstruction/reference operations in
this ordering test are small fixture callbacks, not real owned-data reads.

The arithmetic 22*8 MiB = 176 MiB and request cap192 MiB is asserted against the
actual candidate constants. The 16 MiB difference does not establish maximum
serialization fit, reference headroom, memory usage, scorer throughput, enriched
32 MiB report fit, or final control/inventory envelope capacity. No real192-MiB
allocation or maximum-shape qualification is claimed. Small frozen bundle/request
serialization is bounded below128 KiB; source buffers are separately disclosed
above and must not be misreported as sub128-KiB fixtures.

The test does not rerun actual projection V2, context V3, native V14, the scorer,
supervisor, sampler, or prior V6 numerical/custody suites. It does not observe or
operate on a live process, private cache, filesystem race or physical attempt.
Actual metadata and process safety remain separate source/host evidence. The
tree controls qualify the declared comparisons and pre-dispatch ordering, not
an atomic filesystem snapshot or immunity to arbitrary future mutation.

## Gate and handoff

Independent review of exact test/harness/protocol/PRE must precede any collection
or execution. Only this new test may be collected, with no conftest, automatic
plugins, cache, bytecode or root/backend initialization. Root owns the later
finite resource caps, harness, source archive, PRE and sole gate. Execution waits
for accepted diagnostic POST and no heavy-job overlap. A test failure or unexpected
count is terminal evidence to retain/review, not permission to retry, loosen
assertions, change the frozen source or enlarge a physical cap.

Authoring touched only this new test and this document. Named source/review reads,
manual fixture inspection, hashing and whitespace checks were performed; no
candidate import, compilation, collection, test, native call, raw/Gold/private/
attempt/control read, diagnostic-source change or physical process operation
occurred. The next action is independent source review, not execution.

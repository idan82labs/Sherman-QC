# Diagnostic V1 focused source-gate test design — 2026-09-12

Status: source-only authored, not collected or executed. This is a prospective
46-outcome, 15-function new-orchestration gate. It does not rerun the frozen V6
132-outcome suite, qualify physical execution, authorize an attempt, or establish
an engine result. The candidate remains release-false with unreleased timings.

## Exact target and read boundary

The new test is `tests/test_gold11_context_v3_diagnostic_v1.py`. Its only two real
target-source reads are:

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/run_gold11_context_v3_diagnostic_v1.py` | 57923 | `08f9e50b52f0225ca54e1920ef21fed2d0840a19941fb20b338b43e6b141a44f` |
| `scripts/run_gold11_mesh_development_v6.py` | 166049 | `7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80` |

Each read is restricted to one of these named paths, capped below 256 KiB, and
checked against its complete SHA256. No physical review script, original input,
catalog, reference, prior attempt, authorization/control artifact, private cache,
backend source, projection source, artifact-helper source, supervisor source, or
sampler source is read by the test. The test runner may separately freeze an
explicit source closure; that is not permission for this test to open that closure.

At a later separately authorized gate, the test parses the diagnostic source and
compiles only its explicit 24 assignments and 28 named function definitions in a
detached namespace. It does not import the candidate or execute its module body.
The real `profile()` then parses the pinned V6 buffer and compiles its actual
closed 21-constant, 55-function, 2-class allowlist. The real seven diagnostic
callbacks are installed by the actual candidate profile implementation. No full
V6 module import, V6 main/freeze/execute/scorer extraction, backend initialization,
native operation, subprocess, thread, or signal operation is permitted.

The test uses only Python standard-library modules and pytest. No numpy/scipy or
native geometry dependency is needed. Candidate namespace imports and `open` are
deny functions; process/native/module-loader surfaces are explicit deny doubles.
`LexicalPath.resolve()` returns its lexical value, `mkdir()` records an in-memory
event, and `lstat()` denies access. Thus the profile's pre-bootstrap path
resolution does not touch a host path. Process identity, clocks, alarm remaining
time, file descriptors, source reads, metadata enumeration, publication, and
supervision are independently supplied inert doubles for the functions under
test. Any still-unmocked operation must fail the test.

## Fixed outcome denominator

Each row below describes one parametrized test function. Counts are prospective
ordinary outcomes, not observed pass counts. There are no xfail/skip controls.

| Test seam | Outcomes | Required observation |
| --- | ---: | --- |
| False guard at six entry points | 6 | Exact draft refusal before effects for profile, freeze, execute, coordinator, main, bootstrap read |
| Boolean-only timing guard | 1 | Local Boolean change alone still refuses unreleased timings |
| Actual closed profile | 1 | 55 definitions, 2 classes, 21 constants, 7 callback identities, 64 unique declared source slots including 50 backends, no scorer/reference globals |
| Closed extraction deltas | 2 | After real positive construction, absent function or forbidden scorer extraction receives exact denominator refusal |
| Source CASES authority parser | 7 | Invented valid 11-row/17-path authority; exact hash, row count, duplicate, source form, shared-hash conflict, missing selected case refusals |
| Actual freeze/verify round trip | 3 | Invented 64-slot publication and equivalent verified bundle; changed six-slot profile or changed source bytes refuses |
| Exact six-slot PRE | 4 | Positive profile/limits/no-score binding; changed profile, 22-slot denominator, or scorer permission refuses |
| Closed scorer routes | 3 | Inference positive premise; role, command, and publication scorer paths each refuse with their own exact reason |
| Actual source-tree shape | 3 | Exact metadata-only tree passes; unknown name and symlink refuse after an exact-tree positive premise; zero content reads |
| Six-slot schema into actual V6 inventory | 2 | Exact invented tree passes; unknown SCORED name is metadata-only veto, never read; 12 scientific terminal files retained |
| Actual six-slot reconstruction | 4 | Six negative slots retained; one exact original-result/transport path; altered transport bytes and altered closed-log count refuse after positive reconstruction |
| Closed log evidence | 3 | Explicit post-quiescence hash scope; claimed missing log refuses; unknown failed-attempt count is not treated as known zero |
| Coordinator source-profile authentication | 2 | Actual six serial dispatches and new context schema; wrong source-profile digest refuses before another dispatch after a valid premise |
| Unestablished input binding | 1 | Pre-read reserve failure retains its original receipt/reason, aborts after first control slot, and publishes generic binding-not-established reason with six terminal positions |
| Outer ordering and terminal evidence | 4 | Clean, unknown writers, coordinator abort, and evidence digest mismatch; strict quiescence precedes postflight/fill/reconstruction/inventory; unknown writers prohibit these operations |
| **Total** | **46** | **15 test functions; no previous 132 outcomes included** |

The selected order remains capacity `CG08506AE97D`, registration/near-tie
`CG4BDD1371B6`, and CPU `CG4FAF273093`, control then challenger for each. These
strings identify the frozen scheduling choice only. Every fixture authority path,
hash input, result, projection, terminal, and metadata row is invented; the actual
three physical cases are not read or evaluated.

## Positive premises and exact branch reach

The profile extraction controls first build the real unchanged profile. Parser
controls first establish the same valid invented 11-row/17-path authority. PRE
controls first validate their exact baseline. Freeze controls first complete the
actual freeze/verify round trip in memory. Scorer-route controls first check
permitted inference routes and use separate exact error expectations. Tree-shape
adversaries first admit the same exact tree. Reconstruction controls first
reconstruct the same retained byte buffers. Wrong coordinator profile first runs
the same valid synthetic coordinator path, discards only that fixture's transient
state, then asserts no second dispatch or context publication. The outer unknown
case first checks its clean receipt using the real V6 strict quiescence predicate
before changing the identity-unknown flag. Four outer fixtures also construct
their six-row evidence through real reconstruction before replacing operation
seams with ordering spies.

The unestablished-binding control intentionally uses the original pre-read
`TOTAL_WALL_RESERVE_UNAVAILABLE` receipt. `source_binding_valid=False` is not
claimed to prove a hash mismatch. The expected outer reason is exactly
`DIAGNOSTIC_INPUT_BINDING_NOT_ESTABLISHED_ABORT`; the original per-case bytes and
reason remain unchanged, only the first control leaf is attempted, and neither
the memory check nor successful reconstruction is reached. No synthetic retry of
a physical one-shot is involved.

## Explicit fixture substitutions and limits of the claim

The candidate release flag stays false during actual profile/control tests. An
explicit detached `require_release` event double admits selected pure functions;
it does not modify either source file or provide an environment-based release
path. One separate guard-only test changes a local Boolean solely to establish
that the timing guard remains closed, then performs no profile operation.

The parser receives an invented `CASES = json.loads(...)` source buffer. Its
test-local `REVIEW_SHA` is the digest of that invented buffer. The production
Sep9 review-source literal is retained in the pinned candidate and is recognized
before this substitution; no original review source or input is opened. Parser
adversaries update only the invented source hash when necessary to reach the
intended later parser branch.

The freeze/verify test is a source-store protocol control, not a qualification of
64 executable payloads. It fills those exact 64 named slots with small inert
buffers, except for the already-read actual V6 provider and invented CASES source.
Only in this detached fixture, the artifact, supervision, and sampler pin values
are changed to the digests of their inert slot bytes. This makes the actual
freeze/verify closure logic testable without reading, importing, or executing any
helper. It must not be reported as evidence that those invented payloads are
usable, or that real helper pins were verified by this gate. The candidate and
provider complete-source pins remain independently checked.

The actual V6 inventory body is exercised with the actual new six-slot schema,
in-memory two-pass directory metadata, and a minimal artifact helper double that
checks admitted bytes/hashes and exposes explicit read counters. No directory is
walked on the host. The exact-tree fixture includes the six pairs of required
terminal/projection files and no optional issued/claimed files; authentication of
all possible issued/claimed adversaries is not requalified here. The separate
actual coordinator test covers the new profile digest and context publication
boundary. The unchanged V6 leaf machinery and prior 132 source gate remain
separate evidence, not outcomes added to this denominator.

The reconstruction test runs actual diagnostic `reconstruct_six()` and inherited
V6 `reconstruct_scoring_input()` on invented original and retained projection
bytes. The projection V2 revalidator is an explicit contract double that checks
those exact original bytes, projection digest, and mandatory context request;
the actual projection algorithm is not imported or requalified. Negative slots
use the actual inherited unavailable/reconstruction branch. No contact math,
measurement, registration, certificate, catalog, scoring, or F1 is executed.

The outer supervisor is an inert receipt double. The strict quiescence predicate
is real, but no operating-system process is observed, signaled, waited on, or
claimed quiescent. Source postflight, six-slot fill, and inventory are ordering
spies in the outer test; their pure new-delta branches are exercised separately.
Opaque cache custody is asserted without reading or removing any cache. These
tests cannot establish physical scheduling/resource behavior, stage telemetry,
race freedom, native input binding, cache cleanup on a live host, or exact
supervisor capture-time log hashes. Closed-log hashes retain their narrower
post-quiescence scope.

## Bounds and next gate

Invented CASES payloads are below 16 KiB, bundle serialization below 64 KiB,
inventory receipts below 100 KiB, and outer publications below 256 KiB. Original
result/projection/log fixtures are tiny literals. The actual provider source is
166049 bytes and is deliberately not misreported as a sub-100-KiB fixture.
There are no expanded arrays, geometry buffers, background writers, or repeated
host polling. The whole future worker, including repeated small AST construction,
must fit the independently frozen source protocol rather than assuming a pass
from these source-level bounds.

Root owns the single new source harness/protocol/PRE/archive. Intended gate:
only this new test collected, exactly 46 ordinary outcomes, pytest/stdlib only,
20 CPU seconds, 25 work seconds, 30 outer wall seconds, 512 MiB worker limit,
1 MiB total log limit, no conftest/autoplugins/cache/bytecode/root initialization.
Those declarations require independent exact-source PRE and a sole bounded
invocation before any result exists. A failure or unexpected collection count
must be retained and reviewed; this document does not authorize a retry, widening
the limit, suppressing an assertion, or launching a physical diagnostic.

Authoring checks here are named-source rereads, textual inspection and whitespace
diff checking only. No Python parsing, compilation, import, collection, test,
native operation, or physical/control/attempt access was performed during
authoring. Independent test-source review and the bounded qualification remain
the next authorized steps; the candidate and frozen V6 source were not edited.

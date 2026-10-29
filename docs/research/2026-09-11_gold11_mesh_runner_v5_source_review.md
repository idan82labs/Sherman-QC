# Gold11 V5 runner: independent source-only review

Date: 2026-09-11. Decision: **no additional source blocker found in the reviewed
candidate; proceed to the declared qualification sequence, not physical
execution.** This is neither an exact source-gate PRE nor a physical release.
The runner still has `RELEASE_AUTHORIZED=False` and seven unset time limits.
No test, import/collection, native operation, physical input, control record,
retained scientific artifact or authorization marker was accessed or executed.

## Reviewed revision and method

I read the complete 2,620-line runner, all 877 lines of the new test module,
the full draft physical protocol, the selected ordinary fixture constructors,
and the complete qualified terminal-artifact helper. Static AST/symbol-table
checks inspected the declared source/import graph without loading project
modules. The unchanged predecessor was read only as source for its pinned AST
preservation checks; no old runner or old suite was invoked.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/run_gold11_mesh_development_v5.py` | 153950 | `7c8a2ca60ea7671dd1c2b56339d6c632e70e00df7e6081fd18cb421a6678121f` |
| `tests/test_gold11_mesh_development_runner_v5.py` | 52489 | `ee7861235ad345d7d807112cf3471536252e9b55f9ee437d0de0098b91ae2f76` |
| `docs/research/2026-09-11_gold11_mesh_development_v5_protocol.md` | 11780 | `5aa1b7c0bcc54f9b6eb2a41d000ed9a8c7e7460ea95b74b25b47d37a8027d874` |
| `scripts/run_gold11_mesh_development_v4.py` | 130538 | `559095a685d6ae5badecdb9704952d5e80304dcd728f8afa0f131a25707ebd85` |

Full-runner global references resolve to its declared imports/constants/
definitions or builtins. The new tests select 43 runner function definitions,
two exception classes and top-level imports/assignments into a detached
namespace; none of the eight access-bearing process entries is installed by
that fixture. Separate entry-guard tests install an explicitly false first
statement guard before invoking those definitions, never a released entry.

The selected namespace's omitted globals are accounted for: deadlines, time,
signal, artifact loader and runtime metadata are explicit fixture substitutes;
projection uses the real separately loaded pure helper; the case orchestration
test supplies fake memory/process/dispatch functions. The reference-barrier
test supplies `verify_bundle` and a reference seam that raises before any
reference/scorer access. Its later process/private-request globals are not
reachable in that test. No unexplained callable-path global was found.
The selected contact/schedule dictionary constructors' globals also resolve
within their declared ordinary-function/constant namespace.

Independent static decorator counting yields **25 test functions / 85 outcomes**.
No outcome is claimed passed by this review. Test definitions permit the future
concrete timing/release constants to be frozen: bad-time/false-release cases
set those conditions explicitly rather than requiring permanent draft values.
The runner SHA pin in the test must be updated mechanically if the final
reviewed timing/release literals change, before exact source qualification.

## Scientific comparison and constructor binding

The complete `CONFIG`, `CERTIFICATE_CAPS`, `REVIEW_SHA`, `CATALOG_SHA`
and `SELECTED_SHA` AST expressions match the hash-pinned V4 source exactly.
The eleven-case selector is preserved. This verifies source preservation, not
the physical contents or historical comparability of those hashes.

Both arms call V13 with native adapter V1, connected measurement contexts,
V3 finite-triangle consumers and free-circle measurements from original raw
scan rows. Control requests V7 / scheduler V1 / context V1; challenger requests
composite V1 / scheduler V2 / context V2. The new source explicitly labels this
a combined proposal/scheduling/admission change, not a one-factor causal result.
Neither fresh arm is a reconstructed historical control.

`arm_request` returns a detached canonical declaration containing common
scientific settings, certificate caps, scan configuration, full scheduler/
initializer budgets and the expected context schema/default budget map.
Its hash is included in leaf dispatch identity, case start/terminal receipts,
campaign terminal evidence and scoring bindings. The constructor seam checks
the actual native/context/scheduler classes against those maps; it does not
invent an unsupported context-budget keyword. The tested constructor call
does not invoke the workflow or native code.

Scoring remains the qualified contact-sector generator projection plus the
unchanged geometric matcher. The original finite contact carrier is not
renamed physical-trim truth. Unknown count, absent measurement, uncalibrated
0.05 mm bound, unverified common frame/datum and indeterminate conformance
remain explicit. A future positive delta means same-input consumed-development
improvement over this fresh control; historical superiority and unseen-family
generalization remain unestablished.

## Original-byte and all-22-before-Gold barrier

The runner feeds the exact returned RESULT byte object and expected SHA into
the qualified schedule projection. SCORING_INPUT retains that wrapper in its
existing slot; neither strip/re-encoding nor a sixth per-case artifact is added.
Parent reconstruction receives the same original bytes, retained wrapper hash
and independently frozen arm request. A clean case must have the exact wrapper
kind and original result; a failed case must have the exact six-key, empty-
prediction supervisor refusal recorded in its terminal receipt.

Before the only scoring coordinator path opens Gold, it reconstructs all
eleven cases in both arms from their durable terminals/results, checks canonical
agreement with coordinator outcomes/receipts and validates the 22-terminal
marker/request hash. The 21-terminal test stops before the reference seam;
the 22-terminal test reaches a deliberate exception at that seam, not Gold.
Malformed late records do not get replaced with convenient fresh outcomes.

After scoring, the parent repeats reconstruction from retained original bytes,
rebuilds the expected scoring request, checks its hash and requires every
original scorer field to be unchanged in the enriched report. This verifies
declared metadata/transport consistency; it does not independently recompute
native geometry, schedules or raw-support membership. Those remain obligations
of the separately qualified engine and source-bound worker.

A supervisor/projection failure remains an all-arm empty-prediction refusal,
with available result/partial evidence retained. The all-failure scorer test
retains all eleven cases and 71 references per arm. A terminal-fill mismatch
or later artifact inconsistency vetoes campaign custody rather than allowing
a useful prefix or a rewritten terminal.

## Source/import and execution boundaries

The current runner declaration contains **61 distinct source/test/protocol
paths**. Static traversal found no omitted backend import from any declared
backend module. The private scoring graph is exactly the five declared pure
script modules: physical-evaluation contract, geometric matcher, legacy scorer,
contact scorer and schedule projection. Their relative dependencies remain
within that graph; NumPy/SciPy are installed-runtime dependencies, not repository
payloads. The native inference leaf uses the declared backend subset; its
private namespace prevents application initialization.

The source/mock gate must additionally bind the two ordinary-constructor test
sources, its own harness/protocol and its supervisor, as well as the declared
test initializer and absent enclosing initializer boundary. The physical
runner's 61 count alone is not that test-gate closure. This review does not
certify the separately authored harness or replace its exact PRE.

The preserved artifact helper is
`3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838`.
The selected contact fixture is
`f9576b9d4b7267565657ee18a4b344e150a48da9c86f9bf5da2259db3d9fe2cb`;
the selected schedule fixture is
`8d3113d8017d3e50e096696563daf38c4f62deacb87ca552263a6adcac8faf72`.
Neither whole old test module nor any decorated fixture is called as an
ordinary fixture constructor.

Physical entries refuse first while draft or timing-invalid. Future execution
still requires exact frozen source/bundle/runtime metadata, an independent
GREEN bundle audit, exclusive unused state and one-use parent-bound leaf
claims before backend/input access. Source/mock tests do not execute positive
authorization/process paths; process audit hooks remain engineering guards,
not an OS sandbox or hostile-extension guarantee.

## Full inventory, transport and resource limits

The campaign fixture uses eleven invented case tokens, six invented families,
71 invented reference lines and 22 actual pure-scored projections. It contains
all **119 scientific paths** (22 times five per-case files plus nine campaign
files), the full 61 source-path namespace, ordinary control ledgers and all
**46 issued/claimed records** for 23 leaves. Source contents, PIDs, cache paths,
version metadata and source/control authority are explicitly invented inputs.
No fake PID/cache path is queried or followed.

The test keeps the real mixed-tree schema, dispatch verifier, terminal artifact
helper, parent reconstruction and inventory functions. It exercises inventory
before COMPLETED, after exact COMPLETED publication and after FINAL_GATE
publication, with source/control bindings, no-write existing terminal validation
and 256 KiB completed-envelope admission. Unknown names, aliased or separate
partials, wrong issuance/claim, missing result, source/link changes and late
equal-byte inode replacement are vetoes. Every unknown name is charged to the
scientific 128-entry limit; unknown directories are not recursively read.
The full schema permits explicit complete/partial alternatives but their
simultaneous inconsistent presence does not become a positive.

The test's fake signal/clock/deadline and fake subprocess boundaries mean it
does **not** qualify real alarm delivery, process cleanup latency or maximum
128 MiB RESULT throughput. The ordinary constructed footprint is limited to
16 MiB, and one separate case checks the actual retained wrapper at exactly
8 MiB including whitespace/newline versus one byte over. The prepublication
body-plus-newline check uses an explicitly labeled transport mock.

The current transport caps are coherent: 128 MiB returned result, 8 MiB retained
wrapper/inner projection, 192 MiB private scorer request, 32 MiB original/enriched
score, 256 KiB control envelopes and source-derived aggregate read ceiling below
8 GiB. That is bounded fail-closed admission, not assurance every maximal
combination can succeed. The finalizer may legitimately veto an oversized
report. Passing the planned full-schema test would establish representative
structural capacity, not worst-case metadata/path/payload throughput.

No physical timing adequacy can yet be concluded. Final concrete limits must be
derived/reviewed from the corrected connected/native work, not the rapid prior
quota refusal. Keep the seven timing literals and final release state fixed
before the final source/mock gate and physical PRE; a begun cap never increases.
The stated memory limits remain 8 GiB leaf, 4 GiB coordinator, 12 GiB whole tree
and 2 GiB scorer, with the disclosed sampled-RSS—not hard allocation—semantics.

## Remaining gates

There is no new code-change request from this source review. Required work is
the existing sequence: corrected all-four connected qualification, constructed
native-byte qualification, independently reviewed concrete physical timings,
final exact-source 85-outcome runner qualification, then exact physical
PRE/freeze. Preserve all terminal predecessors and the original case/reference
denominators. Recheck the final candidate diff/hashes after timing literal
selection; this draft review cannot attest later bytes.

No F1, new physical predictions, accepted registration, manufacturing event
count, calibrated uncertainty or floor readiness was obtained here.


# Cached-control V1: separately versioned qualification V2

Date:2026-09-10. Invented source/mock/stdlib inputs only. No runner CLI, actual
source-freeze command, cached prediction geometry, Gold, raw scan or CAD access.

## Engineering reason and preserved failure

The predecessor qualification is terminal132PASS/4FAIL/one declared skip, exit1,
no budget stop or six-file drift. Independent retained audit establishes a
test-double interface defect: `_Entries` was a context manager but not iterable;
all four partial-usage checks failed before their intended assertions. Preserve
[that failure](2026-09-10_gold11_cached_control_v1_qualification_result.md), OUTPUT
`0824416c12d25e0aa291910f642cb1f8a1e52f1ea7ffd0b4b78d6fa51c385ee2` and its frozen
source. The conditional V1 smoke is NOT released or executed.

Runtime `scripts/run_gold11_cached_control_v1.py` remains byte-identical SHA256
`75af29ad6314c63a98ae5176d348a82e55fd408e975f92766879cfb09939721f`.
The new test is `tests/test_gold11_cached_control_v1_qualification_v2.py`.
Its only predecessor-body changes are a faithful single-pass iterator/context
manager `_Entries` and the separately named V2 smoke opt-in. Every original test
assertion and parameter is retained, enforced by an exact predecessor SHA and
AST-preservation check. New controls cover iterator identity, consumption and
context closure, plus actual `os.scandir` accounting of a private invented tree.
The latter uses newly created regular files and no project data.

Prospective outcomes:141pure/source/mock/stdlib checks and one default-skipped
invented child (142when opted in). These count targets must be independently
reconciled against exact source before release; they are not reported passes.

All custody, descriptor, grant, source verification, byte/preflight/report
binding, retained-partial accounting, resource and terminal/publication oracles
from the [original qualification protocol](2026-09-10_gold11_cached_control_v1_qualification_protocol.md)
remain unchanged. The actual child uses the same extracted source/owned invented
paths and assertions; only its opt-in is separately named
`SHERMAN_RUN_CACHED_CONTROL_GUARD_SMOKE_V2`. No runner top-level or real BINDINGS,
scorer, actual Gold selection or native geometry executes.

## Freeze and one-shot gates

Freeze eight files: unchanged runner, frozen V2 support, old failed test, new
test, actual cached-run protocol, old qualification protocol, this successor
protocol and the new narrowly additive harness. Preserve exact predecessor
harness `source_snapshots/cached_runner_e11d1343/check_cad_guided_measurement.py`.
Independent exact pre-GREEN is required. No common-file edits between gates.

First execute once, expected141PASS/one declared smoke skip:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_cached_control_v1_qualification_v2_source`

Only after independent retained source post-GREEN, execute once, expected142PASS:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_cached_control_guard_smoke_v2`

Each gate retains unchanged60s monitored wall,45s workerCPU,1GiB sampled groupRSS,
1MiB output, isolated HOME/TMPDIR and one-thread numeric environment. The smoke
child retains8s wait/5sCPU/64KiB log file-size limit and at most1s timeout reap;
it stays in the outer process group. ChildCPU is separately bounded, not included
in the worker's own45s. Sampled groupRSS includes both; no hard childRSS claim.
The complete bounded child log is forwarded before assertions. Python audit
hooks protect against accidental Python I/O, not hostile native syscalls.

Retain STARTED/OUTPUT/COMPLETED, all outcomes and exact before/after hashes.
Any unexpected skip, failed assertion, missing outcome, source drift or resource
stop is terminal. No changed cap/oracle, successful-prefix result or retry.
Existing sampled-RSS and whole-parent receipt/cleanup timing limits remain.

## Relationship to the actual cached-control bundle

This is qualification V2 of unchanged runtime V1, not a reopened V1 qualification
or a new cached-input execution. The actual runner's16-file SOURCES retains the
old test as historical frozen evidence, never as a passed qualification. After
both new gates independently pass, the actual PRE_RUN_AUDIT must explicitly bind
this successor's eight-file closure and both terminal receipts/output hashes.
That audit's full SHA enters STARTED/worker authentication. Do not silently
substitute the old test entry or modify the actual runtime/protocol.

Successful qualification permits source-only creation of the first actual
cached-control bundle under its [existing protocol](2026-09-10_gold11_cached_control_v1_protocol.md).
That exact bundle still requires independent pre-audit before any cached/Gold
execution. None of these checks establishes physical F1, historical
comparability, observed count, calibration, engine promotion or floor release.

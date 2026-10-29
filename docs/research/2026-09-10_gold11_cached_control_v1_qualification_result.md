# Cached-control V1 qualification: retained fixture failure

Date:2026-09-10. This is a terminal invented-input qualification failure, not a
cached prediction/Gold scoring result. Preserve the original runner, test,
protocol, harness and attempt; no source-gate retry or conditional smoke run.

The exact independently pre-released six-file source gate ran once:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_cached_control_v1_source`

Attempt: `cad_engine_v1_synthetic_checks/gold11_cached_control_v1_source-eks363q7`.
Result:132PASS/4FAIL/one declared smoke skip, exit1; no outer budget stop or
six-file source drift. Parent1.646140499971807s, workerCPU1.6143109999999998s,
worker/sampled peak56344576B. OUTPUT SHA256:
`0824416c12d25e0aa291910f642cb1f8a1e52f1ea7ffd0b4b78d6fa51c385ee2`.

All four failures are parameterizations of the retained partial-usage test
(`None`, unknown directory, symlink, entry-count overflow). Its `_Entries` mock
implements a context manager but is not itself iterable. The runtime correctly
uses `with entries:` followed by iteration over `entries`; the test raises
TypeError before reaching its intended success/ValueError oracles. This does not
demonstrate a runtime accounting defect, nor establish those four checks passed.
Independent `/root/cad_measurement_integration_review` verified all six live and
before/after hashes, OUTPUT, counts and runtime/mock interface: valid terminal
qualification FAILURE, no runtime defect established by these four checks. The
reviewer explicitly acknowledged missing this interface mismatch in pre-review.
No smoke release.

The justified successor is a separately named qualification test/protocol with
a faithful single-pass iterator/context-manager fixture, all predecessor
assertions preserved, and additional protocol/AST-preservation controls. The
runtime stays byte-identical unless independent audit identifies a runtime
finding. The predecessor test remains frozen and included as a source oracle.
The new qualification needs independent exact pre-audit before execution. Keep
the old test in the actual runner's16-file source bundle as historical evidence;
bind the successor qualification closure and receipts explicitly in the actual
run's independent PRE_RUN_AUDIT, whose full digest enters worker authentication.
Do not silently replace its source entry or claim the old test passed.
No cached-run bundle freeze or actual data execution follows from this failure.

Actual runtime SHA256:
`75af29ad6314c63a98ae5176d348a82e55fd408e975f92766879cfb09939721f`.
Failed test SHA256:
`67ec0c5dccebfd58298818e6c43df2651eeee2492c636427f15d4776241dd5e6`.
Harness SHA256:
`e11d1343acc1d7b1b25d3112fbf7fc6aba1a97cf6f70e2efac0702e1163e1ef0`.
The exact harness is retained at
`source_snapshots/cached_runner_e11d1343/check_cad_guided_measurement.py` before
adding a separately named successor route.
See the [frozen pre-audit](2026-09-10_gold11_cached_control_v1_qualification_pre_run_audit.md)
for all six bindings and unchanged60wall/45CPU/1GiB/1MiB limits.

No real prediction geometry, Gold coordinates, raw scan or CAD was accessed by
this gate. No new physical score, historical comparison or promotion is claimed.

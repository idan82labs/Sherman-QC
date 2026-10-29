# Cached-control V1 qualification: exact prospective freeze

Date:2026-09-10. Independent final pre-run release PENDING. No gate execution
until exact GREEN on these six files. Source136PASS/one declared smoke skip,
then smoke137PASS on unchanged common files after source post-GREEN.

Root read the full911-line test and all runner source plus reviewed terminal
repairs. Runtime author `/root/production_architecture_review`; root owns the
small terminal-marker/publication/deadline repairs, protocols and harness.
Independent `/root/cad_support_lineage` authored tests. Independent
`/root/cad_measurement_integration_review` reviewed runtime/protocol and is
reviewing the full new tests/harness. No author's self-review substitutes for
independent exact pre-run release.

[Qualification protocol](2026-09-10_gold11_cached_control_v1_qualification_protocol.md)
binds136pure/mock controls plus one opt-in invented audit-hook child. No real
runner CLI/source-freeze, BINDINGS evaluation, cached/Gold/raw access, scorer or
native execution. Outer60wall/45workerCPU/1GiB sampled groupRSS/1MiB output;
child8s timeout/5CPU/64KiB log with1s post-timeout reap, no independent childRSS cap.
All artifacts/failures stay retained; no changed caps/oracles or retries.

| File | SHA256 |
| --- | --- |
| `scripts/run_gold11_cached_control_v1.py` | `75af29ad6314c63a98ae5176d348a82e55fd408e975f92766879cfb09939721f` |
| `scripts/run_gold11_mesh_development_v2.py` | `94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422` |
| `tests/test_gold11_cached_control_v1.py` | `67ec0c5dccebfd58298818e6c43df2651eeee2492c636427f15d4776241dd5e6` |
| `docs/research/2026-09-10_gold11_cached_control_v1_protocol.md` | `7f6dc2894544bbe2adf4ce45ca73a521bed7f7407a431f8d097c223448d54d37` |
| `docs/research/2026-09-10_gold11_cached_control_v1_qualification_protocol.md` | `60a4edc4f2ab914e4dc59c3d39c9dafaba1aa675944beb99446b09e8fc49e362` |
| `scripts/check_cad_guided_measurement.py` | `e11d1343acc1d7b1b25d3112fbf7fc6aba1a97cf6f70e2efac0702e1163e1ef0` |

Harness difference against archived `cc1f1b6a` adds only two suite mappings,
the four dependency entries,60/45/1GiB routing, retained-attempt routing and the
single invented-smoke opt-in. No backend namespace or physical/native opt-in
is added. Archive:
`source_snapshots/mesh_connected_v10_cc1f1b6a/check_cad_guided_measurement.py`.

After exact independent GREEN, run each once serially:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_cached_control_v1_source`

After independent retained source post-GREEN:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_cached_control_guard_smoke_v1`

A valid post-audited pair permits source-only creation of the separate exact
cached-run bundle. It does NOT release cached/Gold reads: actual execution still
needs that frozen bundle's independent PRE_RUN_AUDIT. No physical accuracy,
historical-score-beaten, engine integration or factory-floor claim follows.

# V3 finite-triangle search — independent pre-run audit

Date2026-09-09. Decision **GREEN_FOR_NAMED_PURE_SURFACE_GATE**; blocking findings0.
Reviewer: Astra agent `/root/cad_measurement_integration_review`. Surface author:
`/root/cad_support_lineage`; root integrates the lower-envelope repair and owns
the public-search tests/protocol/harness. This is AI technical review, not a human
manufacturing signature, physical qualification or production approval.

| Frozen file | SHA-256 |
| --- | --- |
| backend/cad_guided_triangle_surface_v3.py | 8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10 |
| backend/cad_guided_triangle_distance_bounds_v3.py | 2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7 |
| tests/test_cad_guided_triangle_surface_v3.py | e0d0be929915829c95c43e794127818d8a9c1b7fede18981926844c71c09e1bc |
| tests/test_cad_guided_triangle_distance_bounds_v3.py | 09deb22ee510fe62e1c69dd1469005ffb949fbea25179755bf713998d2ab254a |
| docs/research/2026-09-09_triangle_surface_v3_protocol.md | a47c9b3f591e91e52f3e6e801209a7df895792ac0e68c3098db2c369319269fd |
| scripts/check_cad_guided_measurement.py | 025ae1d4632dab1430a3f42cc6c80c7d55e54da539c9e69b3dca7020f5e2a959 |

The reviewer read the complete surface implementation, exact helper/oracle
closure, public tests, protocol and named harness routing. Conditional feasible
primal witnesses, max-lower/min-upper combination, fixed-upper conservative
AABB/tie search and selected-triangle optimizer plus world-restoration error
are coherent. Normals, regions and point-comparison flags remain engineering
diagnostics; their explicit uncertainty is not a certificate of CAD or metrology.

Before any execution, independent source review closed negative-index aliasing,
missing region assertions and certificate-channel budget coverage in the tests.
It also confirmed root's lower-envelope finding: a sound but weak first-pass
certificate from a later AABB-excluded triangle could unnecessarily widen the
final interval. Rebuilding that envelope in the complete fixed-upper pass is
sound because a truly minimizing triangle cannot be pruned by its certified AABB
lower bound. The added scheduling control deliberately weakens only a valid lower
bound to zero; upper witnesses stay unchanged and exact KKT truth stays separate.

All52 prospective tests/543 exact KKT calls reviewed. Includes local scales/world
origins, mixed soup scales, finite holes/edges/vertices, ties/order permutations,
long-centroid sentinel, strict source identity and whole-result clearing. The
long-triangle local width criterion was fixed prospectively before execution as
explained in the protocol; no failed result prompted a changed threshold.

Six-file before/after freeze; one pure worker,60seconds parent wall,45seconds
workerCPU,1GiB sampled groupRSS and1MiB output. Independent certificate-pair cap
and abstract work charging are explicit; exact7-versus8certificate and28work
boundary controls match the two-pass source. No native switch, backend startup,
physical inputs, Gold or other suite enters this command.

Authorized first execution under these exact bytes, once:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite triangle_surface_v3
```

Retain every outcome in a fresh exclusive attempt. Do not edit started source,
tests, protocol or limits. Independent post-run review is required. Passing
permits separately versioned caller/registration integration work only; frozen
callers do not consume the new uncertainty and cannot inherit its bounds. No
physical rerun or goal completion follows from this pure gate.

## Terminal result and independent post-run audit

Executed once at
`cad_engine_v1_synthetic_checks/triangle_surface_v3-cdxuhpo6`.
All52tests passed,0skips; the fixed matrix includes543 exact KKT invocations.
Parent0.7578618340194225s,workerCPU0.727166s,worker/samplepeak62,881,792B.
Exit0,no60wall/45CPU/1GiB/1MiB stop and no six-file source drift.
OUTPUT SHA-256:
`b552c8f5a3f79f7324e67c509dc7864d020dcfce717052fbf5b5489090c6e201`.

The independent reviewer read retained STARTED/OUTPUT/COMPLETED, recomputed the
output digest and verified every before/after/live binding against the exact
pre-run closure: **POST_RUN_GREEN_FOR_TESTED_PRIMITIVE_CONTRACT**. No replay,
native import or physical input access occurred during that audit.

This candidate/test/protocol is frozen and this attempt terminal. The exact used
harness is archived at
`source_snapshots/triangle_surface_v3_025ae1d4/check_cad_guided_measurement.py`
with the same025ae1d4 full hash above, allowing future named harness additions
without losing executed bytes. No callers or default product routes changed.

Next: additive bound-aware registration/finite-face consumers, complete connected
workflow checks, then a separately frozen and independently audited Gold11
comparison. The existing physical V2 attempt remains terminal, both arms0lines/
71FN and undefined F1; historical comparability is still unestablished. Passing
this primitive does not imply recovery of the nine numerical-admission cases.

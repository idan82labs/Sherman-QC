# V3 triangle-distance helper — independent pre-run audit

Date2026-09-09. Decision **GREEN_FOR_NAMED_PURE_HELPER_GATE**. Blocking findings:0.
Independent reviewer: Astra agent `/root/cad_measurement_integration_review`;
root authors the helper/harness/protocol, `/root/cad_support_lineage` authors the
exact-rational tests. This is an AI technical review, not manufacturing approval.

| Frozen file | SHA-256 |
| --- | --- |
| backend/cad_guided_triangle_distance_bounds_v3.py | 2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7 |
| tests/test_cad_guided_triangle_distance_bounds_v3.py | 09deb22ee510fe62e1c69dd1469005ffb949fbea25179755bf713998d2ab254a |
| docs/research/2026-09-09_triangle_distance_bounds_v3_protocol.md | 5341d7cb03ae6adb6a43d5319fb7560bf3c09e17fa96bf531e06e2ff59bc85e8 |
| scripts/check_cad_guided_measurement.py | 66eb2f5877a4bc9c49d1bbaf07e09d8d3af4b8db24d2ddcdbb9f8c59c2a68644 |

Independent full-source review checked exact TwoSum feasibility, outward local
interval arithmetic, primal/dual lower and feasible-point upper bounds, zero
residual and restoration, exceptional inputs and whole-result clearing under
the declared conditional IEEE model. Descriptive width telemetry is not a
certified width-admission bound. Model sentinels are not binary attestation.

The complete398-line test source and protocol were read. Original351 lines hash
`f2068d4697f469c45b62211b4cb43d485f57ec50328f2516156cff4e2db20b3a`,
preserving all80 first-draft tests; seven additional branches were added before
any execution. Count independently verified:87cases,122exact KKT oracle calls.
Squared rational containment, candidate-vs-optimizer distinction, global KKT
inequalities, restoration and fixed local accuracy criteria are coherent.

Harness diff against exact used4f2a7781 archive contains only this named route,
two source/protocol bindings and membership in existing60wall/45CPU/1GiB retained
execution handling; output1MiB. Direct NumPy-only source import, no backend
package/native switches, raw/Gold or other suite. Four-file before/after binding.

Authorized first command, once under this exact freeze:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite triangle_distance_bounds_v3
```

Retain failure/abstention/skip output and do not mutate this started source/test/
protocol or budgets. This release is not BVH/workflow integration or physical-run
qualification. Independent post-run review is still required after execution.

## Terminal result and independent post-run audit

Executed once: `cad_engine_v1_synthetic_checks/triangle_distance_bounds_v3-c9yh4uq6`.
All87tests pass,0skips;122exact-oracle invocations under the fixed source/test.
Parent0.3152538752183318s,workerCPU0.275642s,worker/samplepeak54,591,488B.
Exit0,no60wall/45CPU/1GiB/1MiB stop. All four before/after/live hashes above match.
OUTPUT SHA-256 `d364bc00f83073a6a725b947d4f3a48d7e1e22a8885e4688c29fa76b91314a69`.
The same independent reviewer read retained STARTED/OUTPUT/COMPLETED and hashed
all four live sources, without replay/import or physical access: POST_RUN_GREEN
for the conditional represented-hull helper only. No tightness claim is extended
beyond the specific24optimizer scenarios and separate world-rounding criterion.

This revision is now frozen/terminal. Next implement and independently test the
whole finite-triangle search, conservative BVH/tie passes and numerical-bound
consumption in registration. Passing this helper is not physical F1 progress or
permission to rerun the terminal Gold11 V2 protocol. The roadmap remains active.

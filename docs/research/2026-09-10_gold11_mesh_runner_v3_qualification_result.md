# Physical runner V3: invented qualification passed

Date:2026-09-10. One execution,167PASS/one unconditional native/runtime skip,
exit0/no stop/seven-file source drift. This qualifies source/mock orchestration,
not physical geometry. All worker/coordinator-looking log messages are deliberately
invented doubles; no native process, input scan/CAD or Gold was used by the suite.

The [exact seven-file freeze](2026-09-10_gold11_mesh_runner_v3_pre_run_audit.md)
SHA256 `2485487c379168c9c67ba5da4c7bfce9084b7e813ad59d144ec4b85bf294bcdd`
received independent pre-GREEN. Test author cad_measurement_integration_review
reviewed the source/protocol; root separately read the full1210-line test and
435-line physical protocol. The author reviewed root's narrow harness routing
and exact bindings, not self-certifying test expectations. Different reviewer
production_architecture_review inspected retained evidence and issued post-GREEN.

Executed once:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_mesh_runner_v3_source`

Attempt `cad_engine_v1_synthetic_checks/gold11_mesh_runner_v3_source-v9fb0xsh`:
parent1.9675192502327263s, workerCPU1.903993s, worker/sampled peak63602688B.
Unchanged outer60wall/45CPU/1GiB/1MiB; OUTPUT2512bytes.

| Record | SHA256 |
| --- | --- |
| STARTED | `56f56d02a6e5d312efbc09ce26dedac21954992b814da491721b3ce899ad4f31` |
| OUTPUT | `8a6f2509a62d381e473fa906050e026599cbfd4ca69fe52a5818f7545795803c` |
| COMPLETED | `47f954bb589bdeda6962129afc6e2047e0c3990dd02b0c8b012bdc0889b05979` |

Independent post-review checked every outcome, limits and all seven live/before/
after source hashes. Exact test/runtime source remained unchanged. Actual native
guard execution is unconditionally absent from this suite, not secretly enabled
by an environment flag. The shared helper's separate same-group qualification
and existing V10 native4/4result retain their own narrower meanings.

After post-GREEN released the first source-only freeze, root invoked:

`.venv/bin/python -I -S -B scripts/run_gold11_mesh_development_v3.py --freeze`

Observed exit0/FROZEN_NOT_EXECUTED. Exact38-source bundle SHA256:
`ffee4c1012393ff67607160c888bd5d00ea0ac5b81784b6f4ab32d40d3fa9a27`.
Fresh destination `gold11_mesh_development_v3_attempt`; no original geometry or
reference input was opened during freeze. Its independent exact PRE_RUN_AUDIT is
still required before first physical execution. The preserved prospective physical
protocol describes the candidate/limits; the actual bundle and audit establish
the later release without rewriting that qualified document's original status.

The [new cached UNION control](2026-09-10_gold11_cached_control_v2_result.md)
now has independently audited F1.4580152671755725. The future V3 freshV8/V10 run
does not load it or claim historical replay. No fresh-engine F1 improvement,
observed-count truth, calibrated measurement, production integration or floor
release is established by this source gate or freeze.

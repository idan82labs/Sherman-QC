# Shared supervision V1: qualified invented scope

Date:2026-09-10. Both gates ran once on the exact joint
[thirteen-file freeze](2026-09-10_gold11_cached_v2_supervision_pre_run_audit.md).
Independent cad_measurement_integration_review pre/post-GREEN; test author
production_architecture_review did not self-certify its tests.

| Gate | Result | Parent seconds | Worker CPU seconds | Worker / sampled outer RSS bytes |
| --- | --- | --- | --- | --- |
| Source |103PASS/one declared smoke skip|0.25358041701838374|0.223373|50855936 /50855936|
| Same-group child |104PASS/no skip|0.29870862513780594|0.23634199999999997|51216384 /53952512|

Both exit0, no budget stop or four-file gate/thirteen-file union drift.
Source attempt:
`cad_engine_v1_synthetic_checks/gold11_owned_process_supervision_v1_source-sf2ym6b7`;
OUTPUT SHA256 `23379e914b2e90347214b56ee43091b50c7f01ca273b9eba46991809fbcd2c7c`.
Same-group attempt:
`cad_engine_v1_synthetic_checks/gold11_owned_process_supervision_v1_smoke-z6b5z13j`;
OUTPUT SHA256 `083177e8555021b7ae80f131ae6a866f8a362ec13ff57e28c44a4fe5654305a5`.
STARTED/COMPLETED/output hashes, full outcomes and live/before/after source maps
were independently inspected. No old attempt was reopened.

The actual tiny child46241 shared parent/workerPGID46230, was reaped, retained
all105logbytes and returned no cleanup/control errors. One process-table sample
reported childRSS0; this is not a zero peak, continuous bound or allocation cap.
Outer retained sampling and worker high-water values above remain separate.
Inner-mode owned-group fields correctly remain null: the coordinator group is
not killed or declared empty. Outer/reparenting/foreign-identity cases are mocked,
not an actual orphan/escape stress qualification. Stored-log uncertainty and
changed first-observed identities are explicit refusals in the qualified controls.

This qualifies source/table/mock logic and ordinary same-group pipes/monitor/reap
only. No cached prediction, reference, raw scan/CAD, actual runner or native
geometry was accessed by these gates. Cached runtimeV2 still needs its own two
gates and exact actual-bundle pre-audit; physicalV3 has separate prerequisites.
No F1, engine-output gain, historical replay or floor release follows here.

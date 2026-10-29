# Cached runtime V2: source and invented guard qualification passed

Date:2026-09-10. Both gates ran once on the exact joint
[thirteen-file freeze](2026-09-10_gold11_cached_v2_supervision_pre_run_audit.md).
Independent production_architecture_review pre/post-GREEN. It reviewed the
cached test authored by cad_measurement_integration_review; shared-helper tests
were independently reviewed in the opposite direction, not self-certified.

| Gate | Result | Parent seconds | Worker CPU seconds | Worker / sampled RSS bytes |
| --- | --- | --- | --- | --- |
| Source |160PASS/one declared smoke skip|1.842500708065927|1.801864|60784640 /60784640|
| Invented guard |161PASS/no skip|1.8823177078738809|1.802422|60850176 /60850176|

Both exit0, no budget stop or eleven-file gate/thirteen-file union drift.
Source attempt:
`cad_engine_v1_synthetic_checks/gold11_cached_control_v2_source-_zz8091a`;
OUTPUT SHA256 `e93f43390d5c96758187cae1ddec7a0e613b16276b8e24a96a02463e68abbc9a`.
Guard attempt:
`cad_engine_v1_synthetic_checks/gold11_cached_control_v2_guard_smoke_v1-1xj97ouk`;
OUTPUT SHA256 `49bc83e3b149c1c4bdcf4c9c4cb5a9d3dc435597c098cb50eedcec2b3e1c9794`.
Independent review checked full STARTED/OUTPUT/COMPLETED, output digests, every
outcome and exact live/before/after source maps. The real invented child reports
component grants observed and original paths locked after preflight, no physical
input/inference. This is an audit-hook guard check, not actual scorer execution.

The [shared helper gate](2026-09-10_gold11_owned_process_supervision_v1_result.md)
also independently passes103source checks/declared skip and104same-group checks.
Its outer/reparenting/escape cases remain mocked; observed memory is not a hard
allocation cap. Timed setup and explicit clean-group admission are qualified in
the new runner's independent control tests, not inferred from predecessor passes.

The reviewer released the first source-only actual bundle freeze. Root invoked
`scripts/run_gold11_cached_control_v2.py --freeze` once; observed exit0 and
`FROZEN_NOT_EXECUTED`. Bundle SHA256:
`f6129f40e48f6c1c95712bfdaac4433903b2fa7ef3d1f529ede37b88ed2b7213`.
Destination `gold11_cached_control_v2_attempt` contains21byte-exact named source
copies plus fixed source-derived case/reference metadata and unchanged budgets.
No cached prediction, catalog, raw scan or CAD was read by freeze/qualification.
Actual execution still requires this exact bundle's independent PRE_RUN_AUDIT.

Preserve old V1 runtime, failed fixture qualification and its passing successor;
none is reopened or retroactively admitted. This result is not a physical score,
fresh engine-output gain, historical replay, calibration or factory-floor release.

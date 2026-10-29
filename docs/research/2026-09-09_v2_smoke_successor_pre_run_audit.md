# Invented smoke successor: independent pre-run gate

September 9, 2026. This records the independent Astra technical review by
`/root/cad_support_lineage`, followed by root's complete changed-source/test/
protocol review. It is an AI engineering audit, not a human approval or physical
run authorization. The previous negative remains terminal and unchanged.

Exact reviewed source hashes:

- Harness `scripts/check_cad_guided_measurement.py`:
  `cf31c93eee0c947ecbbe6a9c7b3d82d1b045544b2bca92f1f5276f3ef3082692`.
- Supervisor tests `tests/test_v2_native_validity_supervisor.py`:
  `4d013bf3631f4c23d22458e283dee7238187108b08e8ce274d3f04b057254692`.
- Custody tests `tests/test_cad_smoke_group_custody.py`:
  `59d43fd58b70b4bfdeac0828bcd298062a6c48b4747ea92126abe8cec36d25b0`.
- [Prospective successor protocol](2026-09-09_v2_smoke_custody_successor_protocol.md):
  `5152f3651523b2e24921e10afbde441a63fcdc1dfa51697540055f336e0cea35`.

Independent decision: source GREEN for the54 pure controls; conditional GREEN
for one new attempt of the same two invented stdlib smokes after those controls
pass. No new P1 found. Capture-before-process-sample, retained staging links,
cleanup without ACK, original-error retention and independent direct-child/group
cleanup address the identified source defects. The old generic failure cannot
be attributed retroactively to either race. Sampled containment and pre-request/
PID-identity limitations remain explicit; this is not OS-enforced containment.

The prerequisite ran once as `cad_smoke_group_custody-mokgunxf`:54 pass,
parent .3481326248s, CPU .294029s, peak48611328B, no stop or source drift under
60s wall/45s worker CPU/1GiB sampled RSS/1MiB output. Output SHA
`459ccbe7ab94587984dc78da2a9a83f261fc7442a4cb08f2497b9060a9e44374`.
Printed registration-failure records are deliberately injected pure controls,
not actual subprocess execution or ignored test failures.

Root may now run exactly one new named `native_validity_v2_guard_smoke` attempt.
Its STARTED receipt must freeze the above unchanged source closure before work.
Both original smoke cases, budgets and failure rules remain selected. Success
requires all tests, two main-admission ACKs, no original/cleanup error or budget
stop, unchanged sources and all owned groups confirmed empty. Failure is retained
and terminal, not an automatic retry. No physical freeze or CAD read is approved
by this document; the distinct physical exact-bundle gate remains mandatory.

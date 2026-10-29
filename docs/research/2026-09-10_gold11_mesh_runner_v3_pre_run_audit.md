# Physical V3 runner: exact invented qualification freeze

Date:2026-09-10. Prospective source/mock gate, awaiting final independent GREEN.
This is not an actual physical bundle or a physical execution release. The
[qualification protocol](2026-09-10_gold11_mesh_runner_v3_qualification_protocol.md)
binds167pure/mock outcomes plus one unconditional native skip. All budgets remain
60s monitored wall/45s workerCPU/1GiB sampled groupRSS/1MiB output.

The new runner and scorer-local deadline received independent complete source
review by cad_measurement_integration_review. That agent authored the1210-line
test; root independently read the entire test and435-line physical protocol,
finding no blocking oracle/custody issue. Root's harness change is only one route,
five explicit dependencies, existing limits and retained-output routing; its exact
diff against archived05df3772 was read before this freeze. A different reviewer
must verify this complete seven-file binding/protocol and harness delta before
root runs once. Post-audit is required before any actual V3 source freeze.

| Source | SHA256 |
| --- | --- |
| `scripts/run_gold11_mesh_development_v3.py` | `77bbcffbf10f05a22d0c72c8a645112e489469b4af92b4dbba3ca6243fcecea7` |
| `scripts/run_gold11_mesh_development_v2.py` | `94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422` |
| `scripts/gold11_owned_process_supervision_v1.py` | `1b90b86ef77510e32d72d3e08e85417860a8fb06ba0ad3294274a83bfb65ba74` |
| `tests/test_gold11_mesh_development_runner_v3.py` | `9b6c1059ac53c8b788dfbe7b8f4d86f61ed87bef41648230526e69c0d29852f2` |
| `docs/research/2026-09-10_gold11_mesh_development_v3_protocol.md` | `12d5ed1e5b11af055f6a7702f66cfe75dde4686a0d0adf934698a0d417becd85` |
| `docs/research/2026-09-10_gold11_mesh_runner_v3_qualification_protocol.md` | `aac14e06c24fdbf9b77add28f2889b4dc17842e889d13a44fbcf61d7d5c68f56` |
| `scripts/check_cad_guided_measurement.py` | `eae849f8da3930913290c02d0aa41ef49c0f50ffd1cc9a93f51b847ac66f0c83` |

Once after independent pre-GREEN:

`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite gold11_mesh_runner_v3_source`

No source edit, actual process/native child in the test, old-attempt reopening,
changed cap, repeated passing suite or favorable-prefix qualification. Preserve
any terminal failure. A standalone pytest command or private runner CLI is not
an alternative route. The fixed cached-control actual bundle is separate and
its21source bytes remain unchanged by this new harness-only routing extension.

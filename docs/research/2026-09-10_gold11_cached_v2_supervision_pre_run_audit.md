# Cached runtime V2 and shared supervision: joint qualification freeze

Date:2026-09-10. Source/invented-only qualification, not actual cached execution.
This thirteen-file union freezes the shared helper/harness across four serial
gates. No input/reference/Gold access, runner CLI or old attempt is released.
Root read the complete runtime delta, test adapter delta, helper/test and narrow
harness routes. Independent reviews: production_architecture_review read the
cached runtime/test/protocols; cad_measurement_integration_review read the shared
helper/test/protocol. Neither author substitutes for the independent reviewer.
Exact final reviewer GREEN is required before the first gate.

## Serial gates and unchanged budgets

1. `gold11_owned_process_supervision_v1_source`:103PASS/one declared smoke skip.
2. After independent source post-GREEN,
   `gold11_owned_process_supervision_v1_smoke`:104PASS/no skip.
3. `gold11_cached_control_v2_source`:160PASS/one declared smoke skip.
4. After independent source post-GREEN,
   `gold11_cached_control_v2_guard_smoke_v1`:161PASS/no skip.

Invoke each once through `.venv/bin/python -I -S -B
scripts/check_cad_guided_measurement.py --suite <name>`. Each outer gate has
60s monitored wall/45s workerCPU/1GiB sampled process-group RSS/1MiB output.
The respective incorporated protocols bind child limits, same-group scope,
source-only routes, retained evidence and failure handling. No source edits
between these four gates. Every failure/stop/skip/drift remains terminal; no
cap/oracle/fixture change or retry. Review each complete retained attempt before
advancing. Passing permits only subsequent exact actual-bundle source freeze
and independent pre-run audit, not unreviewed cached data access.

## Exact source union

| Source | SHA256 |
| --- | --- |
| `scripts/run_gold11_cached_control_v2.py` | `1e5e1d4ad086182b453388846308ade85dbcf3682d6663d384b638aad4c91abd` |
| `scripts/run_gold11_cached_control_v1.py` | `75af29ad6314c63a98ae5176d348a82e55fd408e975f92766879cfb09939721f` |
| `scripts/run_gold11_mesh_development_v2.py` | `94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422` |
| `scripts/gold11_owned_process_supervision_v1.py` | `1b90b86ef77510e32d72d3e08e85417860a8fb06ba0ad3294274a83bfb65ba74` |
| `tests/test_gold11_cached_control_v1.py` | `67ec0c5dccebfd58298818e6c43df2651eeee2492c636427f15d4776241dd5e6` |
| `tests/test_gold11_cached_control_v1_qualification_v2.py` | `2b9f1339b76ecbb5813f575ec54849477532bd0c1b89bb2ca7fc4d722b47934f` |
| `tests/test_gold11_cached_control_v2.py` | `332bba3ed8e84764939736115d2fe5ff72c5d06d77a77ca2e083988ee68ddd1d` |
| `tests/test_gold11_owned_process_supervision_v1.py` | `51cce05857c899dcd364643458e4821787d9e8499c1977c1818dd66ec6ff43c3` |
| `docs/research/2026-09-10_gold11_cached_control_v1_protocol.md` | `7f6dc2894544bbe2adf4ce45ca73a521bed7f7407a431f8d097c223448d54d37` |
| `docs/research/2026-09-10_gold11_cached_control_v2_protocol.md` | `7f5ca736a9e9789c52d5245d3cfb0dac7c19d267b3c463b35220463d990cf2cf` |
| `docs/research/2026-09-10_gold11_cached_control_v2_qualification_protocol.md` | `0ef097abe39787af6facbc27239143f9e3900d14e95cf881dc1968c22467aec2` |
| `docs/research/2026-09-10_gold11_owned_process_supervision_v1_protocol.md` | `90eff520f6ed9b344e7eb5e71907efbbce8ee0f7560752ca453e25aef275f313` |
| `scripts/check_cad_guided_measurement.py` | `05df3772343f3b10642553f0cce308df21a033e6bb4eec7ea4d4f68ddfa81e49` |

The helper gate binds its four-file subset; the cached gate binds its eleven-file
subset. Their union is thirteen, not fifteen: helper and harness are shared.
The preceding qualified harness remains archived byte-exact at
`source_snapshots/cached_qual_v2_0fb68e41/check_cad_guided_measurement.py`.
No source state, count or result from the terminal old qualification is relabeled.

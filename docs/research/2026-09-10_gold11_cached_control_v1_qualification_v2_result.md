# Cached-control qualification V2: passed scope, later runtime blocker

Date:2026-09-10. Both invented-input gates ran once on the same eight-file
[freeze](2026-09-10_gold11_cached_control_v1_qualification_v2_pre_run_audit.md),
independently pre/post-GREEN. Failed V1 qualification remains terminal.

| Gate | Result | Parent seconds | Worker CPU seconds | Worker/sampled RSS bytes |
| --- | --- | --- | --- | --- |
| Source |141PASS/one declared smoke skip|1.5875967498868704|1.554616|57671680|
| Invented guard smoke |142PASS/no skips|1.6396600417792797|1.5791279999999999|57442304|

Both exit0, no budget stops or eight-file drift. Source attempt:
`cad_engine_v1_synthetic_checks/gold11_cached_control_v1_qualification_v2_source-14jdkrlp`;
OUTPUT SHA256 `c74e178fc5e25b6f6c38126c9c1dbf7e033fd87393da96921cfd5516708107fd`.
Smoke attempt:
`cad_engine_v1_synthetic_checks/gold11_cached_control_guard_smoke_v2-mn0uyw59`;
OUTPUT SHA256 `f3cd873f77d0f6ecb3d35e00025e8d9172f6ab3580ab062cecaa0891f6f65d92`.
The full expected actual audit-hook completion record is retained in OUTPUT.
Independent `/root/cad_measurement_integration_review` checked all receipts,
OUTPUT hashes, counts and live/before/after source maps for each gate.

## New source finding blocks actual cached execution

Root's connected V3-supervisor audit found the analogous gap in unchanged cached
runtime V1: post-STARTED cache creation/token/cache-identity/DISPATCH publication
(including fsync) occurs outside an active deadline. Later wall checks reject
late success but cannot interrupt a hung setup or guarantee reaching terminal
evidence. The independent reviewer confirms this is blocking for actual release.
The successful tests did not qualify this boundary; their passing scope remains
valid. In particular, the existing mock explicitly exempts DISPATCH publication
from its active-deadline assertion, explaining why the gate did not catch it.

No actual cached-run freeze, input read or scoring has executed. Preserve runtime
V1 and all qualification bytes/receipts. A narrow separately versioned runtime
successor must enclose setup in the existing absolute remaining-work deadline,
retain all original oracles and add setup-timeout/publication controls. Its new
protocol/source freeze requires independent audit; unchanged successful gates
must not be rerun as though the failure did not exist.

Python audit-hook checks constrain accidental Python I/O, not hostile native
syscalls. No physical F1, historical comparability or engine promotion follows.

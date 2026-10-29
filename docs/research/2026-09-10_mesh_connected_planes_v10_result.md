# V10 unchanged native matrix: retained execution

Date:2026-09-10. Governed by the unchanged
[protocol](2026-09-10_mesh_connected_planes_v10_protocol.md),
[79-file freeze](2026-09-10_mesh_connected_planes_v10_pre_run_audit.md) and
[independent release](2026-09-10_mesh_connected_planes_v10_pre_run_release.md).

## Source gate —56PASS/one declared native skip, post-GREEN

Executed once:
`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_connected_planes_v10_source`

Attempt `cad_engine_v1_synthetic_checks/mesh_connected_planes_v10_source-mnmidi59`:
56passed, one declared native skip, exit0/no budget stop,79stable source hashes.
Parent.47394350031390786s, workerCPU.436396s, worker peak104595456B,
sampled group peak103546880B; those peaks are distinct measurements.
OUTPUT SHA256 `b55e8c5717d4152a95f3605de57f9e1181092f5dcd7cb6cea7c0dc104ad10094`.

Independent `/root/cad_measurement_integration_review` checked all three receipts,
rehashed OUTPUT, matched all79 before/after/live hashes to the exact freeze and
issued source post-GREEN. This releases the already-frozen native gate once,
serially, under unchanged300wall/240CPU/4GiB/1MiB and16MiB/case64MiBtotal.

## Native gate —57PASS,4/4; independent post-audit GREEN

Executed once after source post-GREEN:
`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_connected_planes_v10`

Attempt `cad_engine_v1_synthetic_checks/mesh_connected_planes_v10-p17n6k73`:
57passed, no skips, exit0/no stop,79before/after hashes unchanged.
Parent152.8944267919287s, workerCPU152.52156300000001s,
worker/sampled peak466337792B. Same300wall/240CPU/4GiB/1MiB limits as V9.
OUTPUT SHA256 `b6e64239f447ee1ccd96640d31d933155d5af2a438ac3e298c67c92def6a2ed7`.
Exact used harness SHA256
`cc1f1b6a5a16b02fc16b2c82ebddd7ba0682153749f6626f2b63dc864d18908f`
is retained at `source_snapshots/mesh_connected_v10_cc1f1b6a/check_cad_guided_measurement.py`.

All four exclusive artifacts are complete, total38315673B (each<16MiB,total<64MiB).
Root independently rehashed OUTPUT and all four artifacts and inspected their
geometry, finite-query and box ledgers. Independent retained review is complete.

| Case | Bytes | SHA256 | Outcome |
| --- | ---: | --- | --- |
|00V1 nominal|9806266|`bd58e69de9928593da999f80adc85a3357933b563a7f3dd44f407d5d205c534b`|One required measured bend; original clean oracles pass.|
|01missing arc|9376190|`53021834983d28c20a9bbecabd0a342abf9051b0f10f0572a5f8b0584e2c30d9`|Completed geometric registration abstention, not observed absence.|
|02R0 common-wrapper nominal|9804162|`0167b7cb1b76e04de683d16d972a44ee5e3e5f9c3a90046124f16348042a57a7`|One required measured bend; original clean oracles pass.|
|03bounded3D noise|9329055|`bb97bcf5b29694a8c27e58bcf023c46d7c4a4412278919fb2a47f7de43839e68`|One required measured bend; original noisy oracles pass.|

Both clean nominal cases: radius error1.3322676295501878e-14mm, absolute sweep0,
generator endpoint1.2814417116311057e-13mm, axis endpoint1.2515439318724856e-13mm,
whole-raw-row pose action9.751113574048942e-6mm. These near-arithmetic errors
belong to a constructed exact model, not scanner calibration or physical precision.

Noisy case: radius error7.097692127189248e-5mm, absolute sweep.0002996216672244145
degrees, generator endpoint.0007612244484623532mm, axis endpoint.0007557403158549605mm,
pose action.0018736098271073978mm. Original error gates remain unchanged.

All cases have zero candidate numerical failures. Every one of2366960 scheduled
support-row occurrences is box-screened;214136 receive full finite queries and
2152824 are explicitly excluded. No excluded support is reported as a completed
query or actual inlier. Nominal workflow statuses remain PARTIAL and the missing
arc UNAVAILABLE; observed_bend_count remains null in all four.

For the two cases with retained complete V9 comparisons, initialization certificate
pairs decrease from60812796 to5729896 (90.5778% fewer). This is a mechanism workload
count, not a CPU profile or a full-run speedup ratio. V9 ended at240sCPU with only
two complete artifacts; V10 completes all four in152.522sCPU under the same cap.
No missing V9 case runtime or geometry is inferred.

Independent `/root/cad_measurement_integration_review` verified all three receipts,
all79freeze/before/after/live hashes, all four complete artifact bytes and original
geometry oracles, every scheduled/support/group partition and all4/10/4/8 completed
refinements. In the two V9-comparable records, scan/topology/CAD-array/sample
bindings, patch/search evidence, hypothesis/support identities and dispositions,
starts, candidates/refinement resources, predictions and measurements match exactly.
Fresh STEP file hashes differ between native runs; represented CAD-array/sample
digests match. Do not claim identical STEP bytes. The reviewer issued post-GREEN
for constructed completion and recorded prefilter-work reduction.

This is positive connected constructed-engine completion evidence. Physical
.05mm/free-circle qualification remains distinct
from this .005mm/tangent gate. No Gold11 F1, observed-count accuracy, historical
comparability, product promotion or factory-floor release is established here.
The next output deliverables are the recovered cached control and a separately
frozen all11same-input development comparison, with unsupported cases retained.

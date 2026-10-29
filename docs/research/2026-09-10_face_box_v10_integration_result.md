# V10 face-box integration: retained results

Date:2026-09-10. Governed by the unchanged
[protocol](2026-09-10_face_box_v10_integration_protocol.md),
[39-file freeze](2026-09-10_face_box_v10_pre_run_audit.md) and
[independent release](2026-09-10_face_box_v10_pre_run_release.md).

## Initializer —86PASS, independent post-audit GREEN

Executed once:
`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite plane_initialization_v10`

Attempt `cad_engine_v1_synthetic_checks/plane_initialization_v10-tal0ib5e`:
86passed, no skips, exit0, no budget stop,21before/after hashes match the freeze.
Parent3.5933329579420388s; workerCPU3.556964s; worker peak100630528B;
sampled process-group peak98533376B. Worker high-water and sampled peaks are
different measurements, not interchangeable. Limits60wall/45CPU/1GiB/1MiB.
OUTPUT SHA256 `b160c26663cdd75c4bf0cb47ea308781f735b1d0d5987bdd2f8a7a8cb08aaa5c`.

Independent `/root/production_architecture_review` read all three retained
receipts, rehashed OUTPUT, matched the21source maps to the39-file release and
issued post-GREEN. This permits the already-frozen54-outcome workflow gate once.
Actual array positives and box/counter controls pass; this is not native runtime
qualification or physical output/F1 evidence.

## Workflow —54PASS, independent post-audit GREEN

Executed once after initializer post-GREEN:
`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_offline_v10`

Attempt `cad_engine_v1_synthetic_checks/mesh_offline_v10-iwaet3m8`:
54passed, no skips, exit0, no stop,37before/after source hashes match the freeze.
Parent0.5396656249649823s; workerCPU0.500421s; worker/sampled peak96616448B.
OUTPUT SHA256 `c355d96f6e44e606df554913ffac2593728f46d169b162b1f6bc29c120879c6e`.
The same independent reviewer checked all three receipts, rehashed OUTPUT and
matched the37source maps, then issued post-GREEN. This is mocked workflow
qualification, not native geometry or performance.

The shared harness SHA256
`af7fbdc42caa059378f912d705a6f4ef640ec117eab17d7ba6f5152ba9b9467c`
is retained byte-for-byte at
`source_snapshots/face_box_v10_af7fbdc4/check_cad_guided_measurement.py`.
No common source changed between the two gates. A separately frozen/audited
native successor may now proceed; no native or physical result follows from
these component passes. Frozen V9 CPU-terminal and physical-config0/2 negatives
remain unchanged.

# Pure retained fusion V1:114PASS

Date:2026-09-10. Independent exact pre/post-audit GREEN, pure contract only.
No actual cached payload, scan, CAD or reference coordinate was read by this gate.

One execution under [protocol](2026-09-10_retained_fusion_v1_protocol.md) and
[six-file freeze](2026-09-10_retained_fusion_v1_pre_run_audit.md):

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite retained_fusion_v1
```

Attempt `cad_engine_v1_synthetic_checks/retained_fusion_v1-gqxt1dn3` retains
STARTED/OUTPUT/COMPLETED.114PASS/no skips,exit0,no stop under60monitored-wall/
45workerCPU/1GiB sampled groupRSS/1MiB stdout. Parent.5105301253497601s,
workerCPU.481534s,worker peak62,439,424B,sampled group peak59,408,384B; these peaks
are distinct observations. OUTPUT SHA256
`61f3f4fa1bd4d9430b52a44929dffe88ed6d8d8057597a477319af8fcc2e81a0`.
Independent `/root/cad_support_lineage` verified receipts and all six unchanged
before/after/live hashes at post-audit. Exact supervisorb40bc9bc is archived at
`source_snapshots/retained_fusion_v1_b40bc9bc/check_cad_guided_measurement.py`.

Original3function/6statement ASTs, four variants, order-sensitive greedy union,
shorter-span duplicate rule, endpoint aliases and duplicate display IDs are
preserved. Independent invented tests cover full-case clearing,11-case retention,
owned per-case snapshots and bounded pair/JSON work. The2M limit controls use
explicit mocked boundary seams, not2M measured geometric comparisons.

This qualifies pure reconstruction semantics, not current payload authenticity,
historical execution, F1, measured geometry or observed count. The101case scoring
adapter gate and separately frozen actual cached-control run remain outstanding.
No fresh inference, physical improvement or release is claimed.

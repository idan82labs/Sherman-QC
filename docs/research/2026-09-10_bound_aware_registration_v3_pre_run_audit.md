# Bound-aware registration V3: exact-source pre-run audit

Date: 2026-09-10. Decision: GREEN FOR THE NAMED PURE 68-CASE CONSUMER GATE ONLY.
No native geometry, physical data, Gold, pipeline publication or physical F1
qualification follows. The [prospective protocol](2026-09-09_bound_aware_registration_v3_protocol.md)
remains unchanged; this receipt records final pre-execution evidence.

Independent runtime/protocol/harness reviewer: `/root/cad_measurement_integration_review`.
That reviewer read the complete runtime and pure dependencies and compared the
named harness route against the preserved 025ae1d4 harness. Root independently
read all 68 test cases and their fixtures, the complete consumer, protocol and
relevant harness dispatch/limits/retention. The test author did not approve their
own tests. No imports, collection or test execution preceded this receipt.

The test matrix retains all 47 earlier draft controls and adds five complete
telemetry, two pre-dispatch resource and fourteen candidate-selection cases.
The unexecuted float iteration typo was corrected before freeze. Actual finite
queries use one represented triangle with analytic distances; other correspondences
and candidate scores are labeled interface mocks. Fraction-based quantile/RMS/
coverage oracles are independent of runtime outward-rounding code. There is no
public pose-recovery or CAD numerical-range claim from these controls.

Reviewed support includes inclusive upper/strict lower thresholds, unresolved
and ambiguity exclusions, unchanged trim/no-refill behavior, region/rank roles,
fixed versus possible support denominators, materialized optimizer objective,
retained original resource-refusal reasons, possible-only competitors and stale
success invalidation. Reverse sample coverage, rank, pose-class distance and
transformation rounding remain engineering calculations, not interval certificates.

## Frozen nine-file closure

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_global_registration_v3.py` | `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0` |
| `backend/cad_guided_global_registration_v1.py` | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `backend/cad_guided_triangle_surface_v3.py` | `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10` |
| `backend/cad_guided_triangle_distance_bounds_v3.py` | `2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7` |
| `tests/test_cad_guided_global_registration_v3.py` | `0e774293e1314d81db6567abf782601304508b5cc71b8f9d9d16e4337a441ace` |
| `docs/research/2026-09-09_bound_aware_registration_v3_protocol.md` | `afb95f6b37a64eb7991ea01519b3aef917537de0c52d5564c4f3682664d50108` |
| `scripts/check_cad_guided_measurement.py` | `b67b71eed25afb3e93063b099d3579bbe0accdfbc461261cf643cd503fdb9810` |

Execution budget: 60 s parent wall /45 s worker CPU /1 GiB sampled group RSS
/1 MiB stdout. Numeric libraries single-threaded; private HOME/TMPDIR, isolated
Python and pytest, no backend application/native import or physical input.
One root-owned serial execution, with exclusive STARTED/OUTPUT/COMPLETED and all
source digests before/after. Every selected test must pass without unexplained
skip, budget stop or source drift. Any negative remains terminal and unchanged.

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite bound_aware_registration_v3
```

Execution and independent retained-only post-run audit are still pending at the
time of this pre-run receipt. V9 initialization/workflow are separately reviewed
and require their own gates; no physical comparison is authorized by this command.

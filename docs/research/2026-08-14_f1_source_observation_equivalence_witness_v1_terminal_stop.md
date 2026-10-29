# F1 source-observation equivalence witness — terminal source-only STOP

Date: 2026-08-14  
Status: **SOURCE-ONLY LANE CLOSED AT V20**  
Baseline: V16 retained and frozen  
Runtime/main/production: unchanged and closed

## Authority

GPT-5.6 Sol Pro issued `SCAN_SURFACE_CAUSAL_NONIDENTIFIABILITY_STOP_V21` and authorized exactly one source-only proof artifact, `SOURCE_OBSERVATION_EQUIVALENCE_WITNESS_V1`. The complete ruling is preserved verbatim in:

- `docs/research/2026-08-14_gpt56_sol_pro_post_pnct_ruling_verbatim.md`

No V21 surface-only detector, U02 execution, C017 execution, locked-35 execution, or held-out execution was authorized.

## One-shot execution

The witness was executed exactly once:

- output: `output/research/f1-new-parts-20260813/v21_source_observation_equivalence_witness`
- report: `output/research/f1-new-parts-20260813/v21_source_observation_equivalence_witness/report.json`
- execution lock: `output/research/f1-new-parts-20260813/v21_source_observation_equivalence_witness/execution.lock`
- immutable source observation SHA-256: `39763b2f4406307f80d006c11f302738ec2ea2a5d5cfeeccb08e59ca7f0b25c9`
- complete worker output SHA-256: `287ea048b85bf7adbd01821fa5545fd6f356365abd21e9d39df639a457a63ba5`

The two harness records referenced the same 17,584-byte source-observation slice. Fixture role, physical construction, and expected count remained external to the worker. The worker emitted the same 96-byte output for both roles:

| Harness role | Physical interpretation | Required count | Actual count | Semantic gate |
|---|---|---:|---:|---|
| NI01 | monolithic bend | 1 | 1 | pass |
| NI02 | ground full-penetration joint | 0 | 1 | fail |

The worker cannot satisfy both semantics because its permitted observation is identical. A different output would require nondeterminism or prohibited role/identity information.

## Gates

- proof valid: **true**
- identical input object: **true**
- input byte identity: **true**
- complete output byte identity: **true**
- contradiction established: **true**
- frozen PNCT metamorphic pair corpus: **pass**
- canonical replay: **pass**
- 4096px surface-panel byte identity: **true**
- 4096px surface-panel differing pixels: **0**
- witness time beyond preprocessing: **0.000751 s**
- added peak RSS before visualization: **1,474,560 bytes**
- nonvisual artifact size: **19,603 bytes**
- source mesh copies stored by witness: **0**
- real scans opened: **0**
- gold, labels, CAD, identity, or expected counts passed to worker: **none**

All frozen bounds passed. The witness is terminal and will not be rerun.

## Mandatory native-resolution visual ruling

The four required 4096×4096 artifacts were inspected manually at native resolution:

- `audits/NI01_surface_4096.png`
- `audits/NI02_surface_4096.png`
- `audits/NI01_MONOLITHIC_BEND_audit_4096.png`
- `audits/NI02_GROUND_FULL_PENETRATION_JOINT_audit_4096.png`

The two role-blind surface panels are byte-identical. Visually, each shows the same double-skin constant-offset fold, the same source candidate axis, the same mid-station chord field, the same amber midpoint path, the same axial projection, and the same all-owned PNCT station/bin support. The two captioned audits differ only in the harness caption added after rendering.

**Manual ruling: no visible surface evidence distinguishes the required positive and negative.**

This is not a visual-quality defect or a threshold problem. It is an observation problem: the optical boundary cannot reveal whether the identical exterior was produced by one continuous bent sheet or by a ground joint that must be excluded by the frozen target.

## Consequence

The source-only research direction is formally stopped at V20. V16 remains the research baseline:

- 35 scans / 25 families
- TP 132 / FP 60 / FN 44
- precision 0.6875
- recall 0.7500
- F1 0.717391
- exact count 4/35
- held-out U02–U05: 15 TP / 0 FP / 0 FN, unchanged

No claim is made that the overall Sherman QC F1 objective is complete. The present evidence says only that another surface-only architecture cannot be causally certified under the required bend-versus-ground-joint semantics.

## Smallest scientifically valid opening

The next authorized research capability requires explicit user approval and physical sensing data:

`CO_REGISTERED_THROUGH_THICKNESS_ULTRASONIC_BSCAN`

- candidate generation and ranking remain source-only;
- at most 16 candidate strips per scan;
- B-scans at axial support quantiles 0.25, 0.50, and 0.75;
- candidate-local tiles only; no full volumetric reconstruction;
- first physical gate must pair a monolithic bend and a ground full-penetration weld/formed joint with matched exterior surfaces;
- one preregistered separation rule; no surface-only fallback if the matched pair is not separable.

Until that sensing channel, matched coupons, and explicit authorization exist, U02, C017, the locked 35, the held-out scans, runtime/main, and production remain closed.

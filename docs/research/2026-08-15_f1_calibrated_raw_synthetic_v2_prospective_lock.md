# Calibrated raw-range synthetic V2 prospective replication lock

Date: 2026-08-15

## Purpose

Falsify or replicate the development-only V61 gain on fresh simulated captures before any V2 evaluator truth is opened. This lane remains synthetic and non-confirmatory; it does not substitute for authentic scanner raw-range exports.

## Frozen corpus construction

- 30 opaque cases, O001-O030
- 15 near-neighbour sheet-metal profile families
- capture variants 3 and 4 (`capture_index` 2 and 3), never used by calibrated raw-range V1
- eight 384x288 calibrated organized depth frames per case
- independent geometry noise, holes, occlusion, range dropout, and rigid pose for each new capture
- source cloud fused only from the simulated organized depth frames
- evaluator truth and hidden meshes remain under `evaluator_only` until the execution and truth-free visual audits freeze
- token: `SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY`
- confirmatory eligibility: false

## Frozen code hashes

| Component | SHA-256 |
|---|---|
| V2 corpus generator | `126e065d2d93c2141605b3c3f9df667bad7ecee355d90350706d9185c784db00` |
| V1 calibrated acquisition generator | `def0b7f66e3207e6111ab483dd77eae780a8893cee416e1a10eba66bbd0926e4` |
| base scene generator | `55e7ea0ff266bf5cb488b89ca15b6446408eba5f2bafa6b78fbdef61b7a9c328` |
| V60 per-view segment detector | `9fa6a314d698f6d840880464cdeb75e616749fca7c7499550175827860f7128e` |
| V61 ownership guard | `fb6329ce66cecefdf2fd60528bb74ad730b2e6e6d4ca2bf07a3397df8b8cff80` |

## Precommitted execution

1. Generate the complete V2 corpus once.
2. Validate all 30 public bundles using the frozen calibrated raw-range validator.
3. Run frozen V51/V52 primary and replay from source PLY only.
4. Run frozen V60 primary and replay from public calibrated frames.
5. Run frozen V61 primary and replay with the unchanged 0.010-B ownership guard.
6. Render every V61-added or V61-suppressed case in three 3D views and all eight native depth frames.
7. Record a truth-free visual ruling for every changed case.
8. Only then open V2 truth, score V52/V60/V61, and retain V61 only if it produces a clean gain without a family or visual regression.

No detector threshold, capture setting, case membership, or geometry may change after any V2 prediction or visual artifact is observed. No rerun rescue is allowed.

The first launcher invocation failed before creating the output root because the wrapper omitted the repository import bootstrap. The only correction added `ROOT` to `sys.path`; the corrected generator hash above was frozen before any corpus bytes were created.

## Physical boundary

The newly received U01-U04 scans are PLY-only and therefore cannot test this capability. No organized depth views, validity masks, intrinsics, or camera poses exist locally for those parts. They remain useful mesh-only evidence but cannot be converted into qualifying raw-range bundles.

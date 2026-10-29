# Native STEP topology targeted-verification E2E — 2026-08-22

## Decision

**Do not promote this architecture to the production floor.**

The production-shaped implementation is deterministic and operationally healthy, but its bend-line accuracy is inadequate. Native STEP topology is a useful infrastructure improvement; the current topology-to-inspection-line mapping is not yet a viable replacement for the retained scan-supported CAD fusion.

## What changed

The test reuses the existing automatic CAD/scan overlay. It does not implement a second registration system.

The following production-oriented changes were made:

1. Export the exact raw-scan-to-CAD transform already used by the automatic overlay.
2. Preserve that transform regardless of whether the global, local, or merged bend report wins internal report ranking.
3. Force the overlay for every CAD-targeted inspection rather than skipping smaller scans.
4. Read STEP B-Rep topology directly through OCP instead of tessellating STEP and treating mesh creases as bends.
5. Define nominal bend targets as paired, coaxial inner/outer partial-cylinder surfaces with useful axial span.
6. Project those axes through the existing overlay, require measured scan support, crop to the visible span, and suppress geometric duplicates.
7. Abstain when the overlay is unavailable or below its trust threshold.

No gold lines, expected counts, filenames, or family identity were exposed to inference. Geometry constants were frozen before scoring.

## E2E operation

- 13/13 inference jobs completed.
- Blind inference contract rate: 100%.
- One case produced a controlled overlay abstention.
- 107 native STEP targets were enumerated across the 12 registered cases.
- 46 targets had sufficient scan support; 44 remained after duplicate suppression.
- p95 pipeline latency: 79.709 seconds per scan.
- Full replay produced byte-identical standalone and fusion metric tables.
- Relevant regression suite: 113 passed, 1 skipped.

## Accuracy

### Exact matched 11-scan comparison

| Mode | Precision | Recall | F1 | TP | FP | FN | Exact count | Minimum family recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scan-only A0 | 0.5758 | 0.2676 | 0.3654 | 19 | 14 | 52 | 0/11 | 0.1111 |
| Native STEP targeted only | 0.3171 | 0.1831 | 0.2321 | 13 | 28 | 58 | 2/11 | 0.0000 |
| **Scan + native STEP targeted union** | **0.4333** | **0.3662** | **0.3969** | **26** | **34** | **45** | **1/11** | **0.2000** |
| Retained scan + scan-supported CAD union | 0.5000 | 0.4225 | 0.4580 | 30 | 30 | 41 | 1/11 | 0.2000 |

The new fused lane improves F1 by **+0.0316** over scan-only, but loses **0.0611 F1** to the already retained scan-supported CAD fusion. The targeted-only accuracy is **0.2321 F1**, so CAD targets cannot currently serve as the sole inspection answer.

On the full 13-scan lane, native STEP targeted-only performance is precision **0.2955**, recall **0.1688**, and F1 **0.2149**.

## Visual ruling

All 13 native-resolution audit artifacts were reviewed in six views.

The STEP axes are visibly more meaningful than mesh silhouettes. Simple tray bends sometimes land cleanly. The remaining failure is semantic and local, not merely global registration:

- Compound and rolled regions contain several valid cylindrical surface pairs for what an inspector may consider one bend.
- Some CAD neutral/curvature axes do not coincide with the manually selected inspection line under the current 4 mm matcher.
- Partial scans suppress too many expected targets.
- Short local topology features still generate false positives.
- One scan correctly abstains at registration.

The strongest conflict is `40500000`: the target lane returns seven visually plausible formed transitions—the same as the trusted count—but scores zero localized matches. Before production, the system needs one explicit bend ontology shared by CAD extraction, manual annotation, and the inspection UI.

Visual gate: **FAIL for production**.

## Production-floor requirements exposed by this test

1. **Canonical bend identity:** establish whether the inspection target is the neutral axis, inner tangent, outer tangent, or a manufacturing bend feature ID.
2. **STEP feature consolidation:** merge inner/outer cylinders and compound rounded facets into one manufacturing operation, not merely one coaxial surface pair.
3. **Per-target scan verification:** use two-sided normal/curvature evidence around the registered STEP feature, rather than point proximity alone.
4. **Registration SLA:** expose abstention, residuals, and operator rescan guidance. Never silently return zero bends.
5. **Promotion corpus:** run the frozen corrected ontology on untouched new scans with independently annotated bend identities.
6. **Operator behavior:** until precision and recall are both substantially higher, outputs must be review suggestions rather than automatic pass/fail dispositions.

## Generalization Progress Indicator

**GPI-C — small positive development indicator, no generalization proof.**

The slice is development-consumed and not an untouched family holdout. The architecture is deterministic, but it underperforms the retained CAD fusion and fails the production visual gate.

## Evidence

- Primary raw E2E: `output/research/f1-step-topology-targeted-v6-20260822/raw-e2e/summary.json`
- Targeted result: `output/research/f1-step-topology-targeted-v7-20260822/LINE_COMPARISON.json`
- Fusion result: `output/research/f1-step-topology-targeted-v7-20260822/FUSION_GOLD11_DIAGNOSTIC.json`
- Verifier audit: `output/research/f1-step-topology-targeted-v7-20260822/VERIFIER_AUDIT.json`
- Native visual audit: `output/research/f1-step-topology-targeted-v7-20260822/native-visual-audits/`
- Replay: `output/research/f1-step-topology-targeted-v7-replay-20260822/`
- GPI: `docs/research/artifacts/2026-08-22_step_topology_targeted_generalization_progress_indicator.json`

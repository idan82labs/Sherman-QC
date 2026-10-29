# Sherman QC F1 — post-V52 single-intervention architecture ruling

## Role and required output

Act as the independent research professor for a zero-shot, scan-only sheet-metal bend-line detector. Review the complete evidence below and select exactly one smallest bounded source-only intervention for the next falsification run, or issue a formal STOP for this architecture family.

Ground the reasoning in relevant research where it materially constrains the choice (robust multi-model fitting, geometric consensus, graph ownership, scan reconstruction artifacts, or related areas), but keep the final prescription implementable and falsifiable.

Return:

1. `RULING`: `EXECUTE_ONE` or `STOP_FAMILY`.
2. If `EXECUTE_ONE`, one unambiguous algorithm specification with fixed constants, deterministic ordering, insertion point, and exact abstention conditions.
3. A short causal argument for why it targets the observed remaining defect rather than merely increasing prediction count.
4. Frozen success/failure gates for the one-shot synthetic engineering run.
5. The exact visual signatures that must be manually inspected.
6. A leakage audit explaining why the proposed feature is source-only and part-identity-free.
7. A short research grounding with primary-source citations where applicable.

Do not offer multiple alternatives, parameter sweeps, rescue tuning, per-family rules, or post-hoc choices.

## Fixed objective and constraints

- Objective: trustworthy zero-shot bend inspection on known and near-neighbor sheet-metal families.
- Strict blind scan-only inference. No labels, CAD, drawings, expected counts, filenames, family identity, or benchmark feedback may enter inference.
- Deterministic CPU path.
- Approximately 200,000-point bounded detector input.
- Peak RSS below 16 GiB and P95 latency below 300 seconds.
- Compact cache schemas; no source-point serialization.
- No part-specific thresholds or post-hoc rescue.
- Current production/runtime/main paths remain closed. This is research-only.
- The present corpus is synthetic engineering evidence only and cannot establish generalization.

## Synthetic LiDAR stress V2 corpus

- 30 scans, 15 topology families, two captures per family.
- Explicit non-confirmatory token: `SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY`.
- Real LiDAR reference median-nearest-neighbor range: 0.153753–1.039131 mm.
- Real reference median-of-medians: 0.463105 mm.
- Synthetic range: 0.363922–0.674757 mm.
- Synthetic median-of-medians: 0.476373 mm.
- All meshes below 200,000 vertices.
- Simulated noise, scan pose, ragged holes, occlusions, and capture-B perturbation.
- Important limitation: triangulation remains more uniform than the most pitted/ragged physical LiDAR reconstructions.

All 30 truth renders and all scored overlays were manually reviewed. Native-resolution reviews were performed on representative simple, multi-fold, degraded, and ownership-error cases.

## Frozen V51 result

- Exact count: 3/30 = 0.10.
- TP / FP / FN: 30 / 27 / 70.
- Precision / recall / F1: 0.526316 / 0.300000 / 0.382166.
- Minimum family recall: 0.
- Thirty primary/replay pairs were semantically byte-identical.
- Truth access was denied by the detector sandbox.
- Peak RSS: 771,833,856 bytes.
- P95 latency: approximately 51.32 seconds.

Visual defects included complete misses, short central fragments, capture-pair instability, and diagonal cross-panel false axes.

## Full source-only candidate ceiling

- 100 truth bends total.
- Full candidate-pool ceiling: 91 TP = 0.91 recall.
- Candidate generation was complete in 22/30 cases.
- V51 selected only 30/100.
- Selection recall loss: 61 bends.
- Weak candidate-generation families: SF02 = 0.50, SF10 = 0.75, SF14 = 0.50.
- Largest selection losses: SF12 = 9, SF10 = 8, SF09 = 7, SF15 = 7, SF08 = 6.
- True-candidate representative sources: cylindrical-standard 46, mesh adjacency 22, plane lattice 15, cylindrical-wide 7, bend hypothesis 1.

This isolates selection/ownership as the dominant global bottleneck, while three families still also have candidate-birth gaps.

## V52 bounded cylindrical-consensus recovery

V52 retains V51 and adds only an unrepresented candidate satisfying:

- both `cylindrical_standard` and `cylindrical_wide` source agreement;
- at least three source routes in the cluster;
- candidate span at least 0.50 of mesh bounding-box scale;
- deterministic generic ordering;
- representation suppression at 7 degrees, 6 mm, and 0.5 overlap.

No label, expected count, family identity, filename, CAD, or truth-derived scalar is used.

### V52 result

- Exact count: 6/30 = 0.20.
- TP / FP / FN: 52 / 27 / 48.
- Precision / recall / F1: 0.658228 / 0.520000 / 0.581006.
- Minimum family recall: 0.
- Delta from V51: +22 TP, 0 FP, -22 FN, +3 exact cases, +0.198840 F1.
- V52 added 22 predictions; every added prediction matched a truth bend under the locked scorer.
- All 30 primary/replay pairs were semantically byte-identical.

### V52 visual audit

All five scored contact sheets (S001–S030) were reviewed. Native-resolution overlays were reviewed for S005, S007, S011, S016, S024, and S029.

- S005: both previously missed bends recovered as long fold-aligned axes; exact, no FP.
- S007: fourth bend recovered; exact, no FP.
- S011: both missing folds recovered; one pre-existing short false fragment remains.
- S016: both missing folds recovered; three pre-existing diagonal cross-panel false axes remain.
- S024: four of six folds recovered; two misses and one short false fragment remain.
- S029: one of five folds recovered; four misses and one old short diagonal false fragment remain.

The recovered lines are full-length or near-full-length, collinear with visible physical folds, and consistent across orthogonal and oblique views. V52 is retained and frozen as a positive engineering prototype.

## Remaining failure structure

1. Complete misses remain on simple and multi-bend families where no V52-eligible consensus exists.
2. Short central fragments remain even when their orientation is approximately correct.
3. Diagonal cross-panel axes remain from ownership/adjacency errors; V52 did not create these.
4. Higher-count parallel-fold families remain incomplete.
5. Capture-pair swings show sensitivity to holes and reconstruction perturbation.
6. Candidate birth remains incomplete in SF02, SF10, and SF14, but selection loss is larger globally.

## Decision boundary

Select one intervention only. Prefer a source-only mechanism that addresses one causally isolated residual class while retaining V51+V52 unchanged. The implementation must be bounded and one-shot testable across all 30 synthetic scans without reading truth during inference.

The one-shot gate should reject the intervention if it increases false positives, creates new diagonal/cross-panel axes, harms any previously exact V52 case, breaks deterministic replay, exceeds 16 GiB RSS, or exceeds 300 seconds P95. Specify any additional minimum positive gate needed to justify retaining it.

If no such intervention is scientifically defensible from this evidence, issue `STOP_FAMILY` and identify the smallest new source-only representation capability required before another execution.

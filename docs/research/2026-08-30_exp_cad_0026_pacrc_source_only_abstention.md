# EXP_CAD_0026 — PACRC-V1 source-only pairwise conformance

## Ruling

PACRC-V1 is retired without threshold rescue. The final frozen implementation produced no authorized anomaly rank on any of the three source-actionable scan pairs. Native-resolution review confirmed the abstentions. Evaluator roles remain sealed.

Integrity note: the invalidated V1 predecessor performed a hash-only byte read of the evaluator manifest before the audit prohibited even that access. It never parsed the manifest or exposed a role. The final V4 runner accepts no evaluator path and made no evaluator-manifest access. This distinction is recorded explicitly rather than describing the entire superseded sequence as zero-byte-access.

**Generalization Progress Indicator:** `GPI-C_NEGATIVE_SOURCE_ONLY_PAIRWISE_CONFORMANCE_ALL_ABSTAIN_NOT_F1_PROGRESS`.

This is a negative generalization result, not an accuracy or bend-F1 result. It establishes that unsigned whole-surface CAD-to-scan distance contrast is too weak and too registration-sensitive to distinguish these physical good/bad pairs under the frozen conservative gates.

## Frozen denominator

The authenticated intake contains seven opaque two-scan CAD pairs. Before any role was opened, only three pairs had two source-only actionable registrations:

- N005/N006 — frozen EXP15 proper transforms;
- N009/N010 — frozen EXP15 proper transforms;
- N011/N012 — independently corroborated proper orientation from EXP24 and EXP25.

The other four pairs remained pretruth abstentions because of wrong CAD binding, unsafe registration, unavailable decoded CAD, or unilateral registration bias. The maximum source-actionable coverage was therefore 3/7 (42.9%). Final rank coverage was 0/7.

## What was measured

PACRC-V1 built a deterministic equal-area atlas over accepted face-owned CAD geometry and compared both scans symmetrically. It removed a weighted pair bias, required common observable support, retained only topology- and spatially-coherent excess components, and rejected any rank that failed magnitude, support, even/odd, pose-uncertainty, replay, or visual gates.

The final patched implementation also:

- forbids evaluator-path access in the source runner;
- exact-binds all six transforms, three meshes, three mesh reports, source thicknesses, and helper implementations;
- welds duplicate display-list seam vertices at 1e-8 mm before face adjacency;
- prevents one-sided/missing support from creating anomaly mass;
- requires every robustness variant to pass the full central rank gate;
- is byte-exact across the complete primary and replay output trees.

## Results

| Pair | Anchor area | One-sided area | Score | Accepted components | Final |
|---|---:|---:|---:|---:|---|
| N005/N006 | 99.65% | 0.000% | 0.000 | 0 | Abstain |
| N009/N010 | 81.18% | 0.114% | 0.000 | 0 | Abstain |
| N011/N012 | 100.00% | 0.000% | 0.000 | 0 | Abstain |

For N005/N006, the largest rejected areas had sufficient extent but sub-threshold median excess. N009/N010 showed diffuse triangulation, perimeter, and coverage-boundary texture rather than a compact physical morphology. N011/N012 was near-neutral; an even-point-only sign disappeared on the odd partition and was therefore rejected.

## Native visual ruling

All three 3000×2400 images were inspected at original resolution. The scan/CAD overlays were mechanically plausible, and no residual component survived both visual coherence and the preregistered numerical gates. Thresholds were not changed after seeing the results.

## Reproducibility bindings

- final result SHA-256: `bca619879e6c7d88a2f3a0e2949958b4920257dc5d8d4d99941b67bcad47716b`
- source-only summary SHA-256: `abd0cf0676a93200733cd29ef05b3916f619cd1dbef668406e23f11ba9df4e5d`
- manual visual review SHA-256: `1789663ce3ef7cb40b7893dc0c0ccc6e86f01479e42e9d50aec2fd0c4cc85ef8`
- implementation SHA-256: `550660c91aedd5a67f57c5d8e0c86cc9c423b7429533bedec5e404efcc939203`
- tests SHA-256: `77a1530343fc001bbf013915a8afe32ff66b35ee067b47a4e630ec677e261289` (9 passed)

The earlier V1–V3 output roots are invalidated/superseded and cannot support a scientific claim.

## Next action

The next bounded architecture is Fixed-Frame Differential Bend Conformance V1. It will leave the pose frozen and operate only in native CAD bend neighborhoods, using analytic cylinder/plane topology, scan normals, rotating-normal evidence, fixed angular/axial bins, and explicit occlusion abstention. This attacks the failure PACRC exposed: a whole-surface unsigned distance can look almost identical even when manufacturing conformance differs locally.

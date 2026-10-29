# EXP_CAD_0011 - fixed CAD-event registration execution audit

## Decision

**Mechanism positive; protocol failed; no guard authorized.**

Generalization Progress Indicator: **GPI-C_MECHANISM_POSITIVE_NONDETERMINISTIC**.

The audit found the exact failure mode suggested by the direction-execution paper: the plane-datum refinement can look favorable on its selected primary-plane objective while making the complete physical alignment worse at fixed bend neighborhoods. However, the current registration pipeline did not replay deterministically, so the result cannot authorize a production guard or support a generalization claim.

## Frozen experiment

- Track: CAD-assisted.
- Evidence: consumed physical development only.
- Cases: 13 total; 11 had selected `PLANE_DATUM_REFINEMENT` in the frozen parent.
- Inputs: scan geometry, CAD geometry, V2 CAD event templates, deterministic seed 17.
- Prohibited during source execution: gold lines, labels, expected counts, evaluator manifests, identity-conditioned behavior, and unavailable SharePoint bytes.
- Fixed local audit: 201 samples per CAD event axis measured against identical processed scan points under the global-ICP and datum-refinement transforms.
- Predictions were not changed and no F1 score was generated.

## Result

| Measure | Primary | Replay |
|---|---:|---:|
| Frozen parent datum cases with at least one material local reversal | 8 / 11 | 9 / 11 |
| Material reversal events | 22 | 24 |
| Frozen parent datum cases whose whole-scan median worsened | 10 / 11 | 10 / 11 |
| Rerun datum selections | 12 / 13 | 12 / 13 |

Eight reversal cases and their event sets were stable across both runs. One additional case crossed the reversal threshold in replay. More importantly, **0/13 complete semantic case outputs were byte-identical**. The frozen parent and rerun also differed on one alignment selection.

The central geometric issue is real: current selection compares primary-datum residuals, not preservation of every bend neighborhood or the whole-scan residual. A refinement may therefore improve the datum while translating or rotating other inspection-critical regions away from the scan.

## Visual review

The eight stable reversal cases were manually reviewed using their existing native six-view parent overlays: C001, C003, C007, C008, C009, C010, C012, and C013.

- C001 and C009 show coherent long frame-root bends where even a small local registration shift is consequential.
- C003, C007, C008, and C012 contain short, partial, asymmetric, or weakly observed features; these are exactly the geometries a primary-plane-only refinement can sacrifice.
- C010 has weak global coherence and several short or ambiguous axes; abstention is safer than confident refinement.
- C013 has crowded perpendicular and parallel event neighborhoods, so a single global datum objective is insufficient.

This review supports the physical plausibility of a bend-local guard. It is not candidate-bound evidence: the exact primary and replay transforms differ, and the frozen parent images show only the selected datum output rather than an exact global-versus-datum pair.

## Why replay failed

The automatic alignment prepares its reference cloud with stochastic mesh sampling and then uses stochastic RANSAC plane extraction. Open3D reseeding was insufficient to reproduce identical reference samples, plane memberships, transforms, and floating-point audit outputs. The visible symptom was small plane-count and metric variation, plus one threshold-crossing case.

This is a production blocker independent of F1. A QC system cannot reliably approve or abstain when identical CAD and scan bytes can produce different transforms or guard decisions.

## Next action

Do not tune EXP_CAD_0011 and do not rerun it as a rescue. The next bounded experiment must qualify a deterministic reference-surface sampling and datum-extraction substrate. Only after semantic replay is exact should a bend-local step guard be frozen and evaluated on genuinely new authenticated CAD/scan pairs.


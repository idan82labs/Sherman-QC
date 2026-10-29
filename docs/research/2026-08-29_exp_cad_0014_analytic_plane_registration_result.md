# EXP_CAD_0014 — Deterministic analytic CAD-plane registration result

## Ruling

**RETAIN FOR RESEARCH INFRASTRUCTURE. GPI-C_INFRASTRUCTURE_POSITIVE_DETERMINISTIC.** The deterministic analytic CAD-plane extractor passed every preregistered replay gate.

## Evidence

- 13/13 canonical semantic payloads were byte-identical across independent processes.
- 13/13 deterministic reference point/normal hashes matched.
- 13/13 datum-selection decisions matched.
- 13/13 bend-local reversal decision sets matched.
- Maximum global-transform delta: exactly 0.
- Maximum selected datum-transform delta: exactly 0.
- Both executions selected datum refinement on 5/13 cases and failed closed to global registration on 8/13.
- Peak RSS: 1.153 GiB, below the 16 GiB ceiling.
- No gold, labels, expected counts, filename identity, or score feedback entered source-side execution.

## What changed

Stochastic sequential Open3D plane RANSAC was replaced by deterministic grouping of the exact triangle normals and plane offsets already present in the CAD reference surface. Plane groups are ranked by projected area with a canonical key tie-break. Global alignment, datum ICP, datum-selection thresholds, and bend-local reversal thresholds were unchanged.

## Scope and visual status

This establishes reproducibility only. It does not establish accuracy or physical generalization, and the intervention selects fewer datum refinements than the old RANSAC path. No new candidate-bound overlay was rendered in this infrastructure-only replay, so there is deliberately no visual-quality claim.

The next experiment must be prospective: run the fixed bend-local direction-execution guard on authenticated new scan/CAD pairs, render native-resolution before/after overlays for every changed case, and visually grade each result before computing any truth-bound score.

That run is not yet executable because the local new-part intake contains correspondence audits and evaluator-only manifests but not the authenticated scan/CAD bytes.

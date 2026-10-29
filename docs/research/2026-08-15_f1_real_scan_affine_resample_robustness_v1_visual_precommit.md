# Real-scan affine/resample robustness V1 — visual precommit

Date: 2026-08-15  
Status: visually accepted for one development-only V51/V52 primary/replay execution  
Benchmark status: ineligible for locked, prospective, blind, or independent F1 claims

## Why this lane exists

The first “corpus-matched” synthetic PSROC lane was rejected before execution. Its mesh spacing was plausible, but its manufactured geometry was not: repeated rectangular-window brackets, idealized trays, broad clean polygons, and symmetric rib shoulders created artificial line evidence and obvious false-positive mechanisms. Adding scan-like surface noise did not repair that semantic domain gap.

No genuinely untouched real scan is currently present in the local intake. The newest U01–U04 scans and the 47251002/38965002 good/bad pairs have already been inspected or executed. The newly received physical part cannot become evaluation evidence until its scan is exported.

This replacement is therefore a bounded development robustness lane. It starts from the already-exposed U01–U04 real LiDAR-derived meshes and applies only deterministic global affine changes, bounded quadric resampling, and small normal-direction measurement perturbation. It tests whether retained source-only inference is geometrically stable under plausible acquisition and meshing changes. It does not test new-part generalization.

## Native-resolution visual decision

All eight public source-only six-view audits and all eight evaluator-only parent comparisons were inspected before detector execution.

| Cases | Visual decision | Native visual rationale |
|---|---|---|
| R001–R002 | Accept as development controls | The real bracket morphology is preserved: asymmetric ears, openings, rolled/formed boundaries, local surface irregularity, and scanner texture remain visible. R002 is coarser but does not invent a new crease or erase the dominant formed structure. |
| R003–R004 | Accept as development controls | The real tray has rounded transitions, non-ideal corners, holes/ears, and surface wear absent from the rejected ideal CAD tray. Both variants retain the same manufactured topology. |
| R005–R006 | Accept; R006 explicitly a density-stress case | The large formed panel retains real edge damage, openings, surface blemishes, and nonuniform scan support. R006 is visibly coarser; that coarsening is the intervention being tested and must not be presented as a new physical part. |
| R007–R008 | Accept; R008 explicitly a density-stress case | The large panel morphology, apertures, boundary defects, and acquisition artifacts remain real. R008 is visibly coarser and is accepted only as a resampling-stability stress case. |

The lane is visually much closer to the corpus because its semantic geometry is the corpus geometry. Its limitation is equally important: each A/B pair shares a real parent, so the eight cases represent four exposed morphologies, not eight independent parts.

## Truth and evidence contract

- Detector inputs contain only opaque case IDs and public `source.ply` files.
- The generator did not use development pseudo-truth to alter the source geometry.
- Evaluator-only reference lines are affine transforms of previously visually audited V51 outputs on U01–U04.
- Those reference lines are `development_pseudo_truth_from_visually_audited_v51_outputs`, not human gold.
- They may measure self-consistency, missing transformed incumbents, and newly introduced predictions.
- They may not establish absolute precision, recall, F1, promotion readiness, or new-family generalization.
- The detector must finish byte-identical primary/replay prediction artifacts and prediction-only native-resolution review before evaluator-only pseudo-truth is opened for scoring.
- Every case must receive a post-execution visual audit. Numbers cannot retain or retire the lane without visual agreement.

## Authorized execution

One V51 primary/replay execution and one deterministic V52 composition are authorized for R001–R008. The authorization is development-only and non-confirmatory. No post-hoc tuning, case-specific rescue, identity routing, expected-count use, CAD use, or benchmark feedback is authorized.

## Frozen bindings

- Generator: `scripts/generate_f1_real_scan_affine_resample_robustness_v1.py`
  - SHA-256: `d14ada2bfa256284c651876ff1f4742f35ef618a4d7ca31fbc4ec5786ef76ecb`
- Public manifest:
  - SHA-256: `d168b46852d9ee19e72b48eed9f0dc3069bab8385b5c0b52229591925d881b31`
- Evaluator-only manifest:
  - SHA-256: `f6ff50099f5da870fb3a58a4cbe5557ce088f0a89a330f23a2b53267e331ab38`
- V9 router model:
  - SHA-256: `ac48c8b9ed48d352ca5a125137a6128a73f793cef012c967eb3a998e460a39ae`
- V4 selector model:
  - SHA-256: `412946c78a63d8550f9abf1d476c94dcf01b7ed1215cfef28053a6dfd7c966f1`

Per-case public source and visual-audit hashes remain bound by `public_manifest.json`.

## Pre-execution conclusion

Accept the lane for a single robustness execution. Reject any interpretation of it as a replacement for the missing unseen, human-annotated part. The next actual generalization measurement still requires the newly received part to be scanned and independently annotated before detector inference.

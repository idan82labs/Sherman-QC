# Sherman QC F1 — post real-scan resampling ownership packet

Date: 2026-08-15  
Requested reviewer: GPT-5.6 Pro, extended thinking  
Decision requested: exactly one bounded source-only architecture, or a formal capability-opening STOP

## Research objective

Develop a trustworthy zero-shot bend-line inspector for daily use on known and near-neighbor sheet-metal families. The eventual locked benchmark must contain at least 30 scans spanning at least 15 families and achieve at least 90% exact bend count, 90% line precision, 90% line recall, 99% render-admissible marker accuracy, 100% deterministic replay, less than 16 GiB peak RSS, at most 300 seconds P95 latency, at least 99.5% job success, and no major family below 80% recall.

Inference must remain blind and source-only: no CAD, drawings, labels, expected counts, filename/identity, family identity, good/bad status, benchmark feedback, or post-hoc part-specific tuning. CPU execution is deterministic, detector input is bounded to approximately 200,000 points/vertices, and compact caches may not serialize source points, vertices, triangles, section paths, or normal arrays. F1 remains research-only.

## Operational target

A positive is a scan-observable, spatially supported fold-shaped transition with two material surface tails and a localizable finite axis. The detector is not asked to infer hidden manufacturing operations.

## Established evidence before this cycle

### Strict 35-case family-held-out development evidence

The strongest estimator-only development point (V26) has:

- TP 79, FP 55, FN 97
- precision 0.589552
- recall 0.448864
- F1 0.509677
- exact count 3/35
- minimum-family F1 0.0

It is not promotion-safe; one family case collapsed from 4 TP / 4 FP / 1 FN to 1 TP / 2 FP / 4 FN.

V27 collinear fragment stitching later removed three historical false positives without changing TP/FN: 89 TP, 97 FP, 87 FN, precision 0.478495, recall 0.505682, F1 0.491713, exact 3/35. V27 is retained as a narrow precision primitive. V28 paired-return completion has encouraging post-hoc mechanism evidence on one newly received part but no independent positive validation.

### Recent V52-lineage precision work

V63 localized-turn ownership was terminally retired on a sealed source-only lane. It changed 9 TP / 9 FP / 25 FN into 7 TP / 8 FP / 27 FN, deleting two true structural bends for one false positive. Native visual audit proved that missing profile construction had become deletion evidence. No future architecture may treat missing/ambiguous support as negative evidence.

V64 PSROC was frozen as a deletion-only certificate for two well-separated responses on one diffuse smooth ribbon. Its eight source-free scientific fixtures pass, but its prospective synthetic lane was rejected visually before execution because the created CAD-like part grammar was unlike the real corpus. V64 was not forced onto the scored lane below: most observed duplicates are closer than its frozen 10h separation floor, flat-face crossings are not ribbon pairs, and its prospective truth-isolation contract is already unavailable on the consumed lane.

## New real-scan-derived robustness experiment

The user correctly rejected idealized synthetic parts as visually out-of-domain and false-positive-prone. We therefore created a development-only lane from four already exposed real LiDAR-derived scans (U01–U04), two deterministic variants per scan (R001–R008). Each variant applies only:

- mild global affine scale/shear/rotation/translation;
- bounded quadric resampling (approximately 136k–181k vertices, always below 195k);
- zero-mean normal-direction noise of 0.035 mm or 0.070 mm.

The source morphology remains real: holes, tabs, rolled edges, surface wear, damage, boundary irregularity, and scanner artifacts. All eight source-only audits and parent comparisons were visually accepted before detector execution. The lane is not new-part or benchmark evidence because it contains four exposed parent morphologies.

The detector was isolated from evaluator data, historical labels/evaluation, CAD, and drawings. Frozen V51 and V52 ran primary/replay on all eight cases. V51 and V52 semantic bytes matched on 8/8 cases. All prediction-only native-resolution overlays were manually inspected before evaluator pseudo-truth was opened. No tuning or case replacement occurred.

## Pre-truth native visual findings

| Case | V52 count | Visual ruling |
|---|---:|---|
| R001 | 10 | Fail: multiple parallel axes on one long fold plus central-junction fragment clutter. |
| R002 | 4 | Conditional: same parent is much cleaner, but the 10→4 change is severe instability and likely drops shorter junction bends. |
| R003 | 8 | Fail: competing parallel perimeter axes and one visible corner overextension. |
| R004 | 5 | Fail: two unequivocal axes cross the broad flat tray face (one diagonal, one horizontal). |
| R005 | 3 | Fail: all three long axes are nearly coincident responses on one physical flange transition. |
| R006 | 5 | Fail: all five axes cluster on the same physical flange transition. |
| R007 | 3 | Fail/conditional: two axes duplicate one opposing fold; one axis follows the other fold. |
| R008 | 2 | Visual pass: one axis on each dominant opposing fold, no obvious duplicate or flat-face crossing. |

Pairwise count instability for the same real parent morphology is 10→4, 8→5, 3→5, and 3→2.

The full visual contact sheet is attached. Its SHA-256 is `5c40a66ba41bd55448da708c36cb0382c72cea4ac965c20900da380dab57bd90`.

## One-time development pseudo-truth accounting

Only after the visual audit was frozen, transformed previously audited V51 reference lines were opened. These are not human gold and cannot establish F1/generalization.

- 18 matched references
- 22 extra predictions
- 14 missed references
- 0/8 exact cases
- pseudo-truth consistency precision 0.4500
- pseudo-truth consistency recall 0.5625
- pseudo-truth consistency F1 0.5000

Key agreement with the visual audit:

- R001: 3 TP, 7 FP, 2 FN.
- R003: 4 TP, 4 FP, 0 FN.
- R004: 0 TP, 5 FP, 4 FN.
- R005: 2 TP, 1 FP, 1 FN.
- R006: 2 TP, 3 FP, 1 FN.
- R008: 2 TP, 0 FP, 2 FN.

Unmatched prediction attribution shows both full-resolution V11 lineage errors and V52 cylindrical recoveries. R004's three V52 additions are all unmatched; R003 has two unmatched V52 additions. R001 has six unmatched full-resolution-mesh predictions plus one unmatched V52 recovery. Therefore this is not solely a V52 append problem.

## Geometric diagnosis available without labels

The dominant failure is final-line ownership/consolidation under routine mesh resampling:

- R005's three visually duplicate axes have pairwise transverse offsets around 8.1–13.0 mm, direction differences below 0.1 degrees, and nearly full overlap.
- R006 contains close parallel responses around 7.3–14.5 mm apart, with mixed span ratios and sources.
- R007's duplicate pair is about 15.0 mm apart with 0.28-degree direction difference and 0.99 shorter-span overlap.
- The existing final-line near-coincident NMS is too narrow (about 4 mm) and does not activate on any R case.
- A broad geometry-only NMS would be dangerous because true return/channel bends can also be long, parallel, and separated by similar distances.
- R004 demonstrates a distinct unsupported-flat-face problem that pair consolidation alone cannot solve.

## Negative-result constraints

Do not propose:

1. a threshold sweep or constants fitted to R001–R008;
2. a generic broad parallel-line NMS without positive source-geometry evidence;
3. deletion triggered by missing support, sparse sampling, failed section construction, output caps, or ambiguity;
4. filename/family/part routing, expected-count logic, gold use, or CAD;
5. rerunning/rescuing V63 or altering frozen V64 under the same identity;
6. another ideal CAD synthetic prospective lane presented as real generalization evidence.

## Exact question

Select exactly one smallest falsifiable next architecture—or issue a formal STOP and name the smallest scientifically necessary capability opening.

The preferred target is a source-only final-line ownership/consolidation primitive that can recognize multiple route responses to one physical fold while protecting two nearby real structural folds and rejecting flat-face crossings. If one primitive cannot safely address both duplicate ownership and flat-face support, select only the higher-value problem and explicitly leave the other unchanged.

Ground the recommendation in relevant research where possible (for example robust multi-model fitting, crease/ridge extraction, mesh differential geometry, topology-aware non-maximum suppression, uncertainty under point-cloud resampling, or related work), while making clear which parts are novel engineering inference.

## Required response format

Return exactly one of `EXECUTE_ONE` or `FORMAL_STOP`, then provide:

1. architecture identity and new lock token;
2. exact eligible input population;
3. allowed and forbidden fields;
4. deterministic source-geometry construction;
5. exact positive certificate and fail-closed behavior;
6. exact mutation rule and representative ranking;
7. compact audit schema with no source-geometry serialization;
8. CPU/resource caps: ≤200k input, <16 GiB process peak RSS, ≤300 s P95;
9. at least eight preregistered source-free scientific fixtures, including:
   - same-fold multi-route duplicates;
   - two nearby genuine structural folds;
   - U-channel/return/hem protection;
   - rounded continuous ribbon;
   - flat plate/perimeter distractor;
   - flat-face crossing candidate;
   - sparse/missing support that must not delete;
   - replay and reversed-winding determinism;
10. a fresh truth-unopened prospective protocol with native-resolution visual precommit;
11. hard retain/retire gates that cannot be rescued post hoc;
12. whether the newly received physical part must be scanned/independently annotated before execution, or whether a fresh transformation-only robustness lane is sufficient for mechanism falsification but explicitly not promotion evidence;
13. citations/research grounding and a separate statement of the novel, unvalidated assumptions.

The response must not use or request private gold answers from the consumed lane to tune the architecture.

## Evidence bindings

- Outcome report SHA-256: `15443c599f2593ff6caa0847596713b0bad097a24420dbc74000bd34dc225cc7`
- Execution manifest SHA-256: `b7045655bc58c5e7315828e739320239771e32d34710a0d9958efd15c791cde6`
- Pre-truth visual-audit marker SHA-256: `8229e2641cc9acb1bcb38e303e4cac7ccc2f963a5b7d1fdb77286065c105d991`
- Self-consistency score SHA-256: `53e92cf89fecc5b0c88d738fa967ba744a410d1b23980abdb5b4f4635ba83c37`
- V64 implementation SHA-256: `f5cfc0393ce45721840f2149ea002737b469754d58ae499abcd78319d30a7528`
- V64 source-free fixture SHA-256: `be9bb2c476d74ccc7817fb88fd1d49b6a8e6b97833e02ba353bf053ba53f4791`

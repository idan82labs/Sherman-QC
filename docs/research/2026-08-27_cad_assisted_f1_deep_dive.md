# CAD-assisted F1 deep dive: why the headline looked worse

Date: 2026-08-27  
Scope: frozen, consumed physical evidence only; no new model execution or truth opening  
Decision class: diagnostic correction and next-experiment design

## Executive ruling

The statement “scan-only F1 0.518 versus CAD-assisted F1 0.456” is not a valid modality comparison.

- The scan-only number was scored on 12 physical scans from 12 families. Fifteen scans were executed, but three had no line truth and were visual-only.
- The CAD-assisted number was scored on a different 11-scan, 6-family corpus.
- The scan and CAD dataset hashes, scorers, versions, and cases differ; their case and family overlap is zero.
- Both numbers are failed experimental challengers, not the retained incumbents.
- The CAD visual record bound to the unified gate covers the topology-only V9 candidate, not the aggressive three-source union whose F1 is 0.456.

The valid same-case comparison on the frozen Gold11 slice is:

| Same 11 scans | Precision | Recall | F1 | TP | FP | FN | Exact count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scan-only control | 0.576 | 0.268 | 0.365 | 19 | 14 | 52 | 0/11 |
| Pure native STEP guidance | 0.390 | 0.324 | 0.354 | 23 | 36 | 48 | 0/11 |
| **Retained scan + scan-supported CAD** | **0.500** | **0.423** | **0.458** | **30** | **30** | **41** | **1/11** |
| Aggressive three-source union | 0.414 | 0.507 | 0.456 | 36 | 51 | 35 | 2/11 |

Therefore:

1. Conservative CAD assistance improved matched F1 by **+0.0926**, adding 11 true positives at the cost of 16 false positives.
2. Pure CAD guidance was slightly worse than scan-only: **0.354 versus 0.365**.
3. The aggressive union recovered six more true bends than the retained CAD lane but added 21 false positives, reducing F1 by 0.0023.
4. CAD is useful as a constrained prior. It is not yet a reliable autonomous bend-event detector.
5. This is GPI-C positive matched development evidence, not physical unseen-family generalization.

Machine-readable evidence: `docs/research/artifacts/2026-08-27_f1_cad_assisted_comparability_audit_v1.json`.

## Why raw CAD is not automatically better

### 1. CAD geometry and QC bend-event truth are different ontologies

A STEP B-rep describes surfaces, edges, radii, topology, and sometimes Product and Manufacturing Information. It does not imply that every cylindrical axis, tangent transition, or topological boundary is one operator-inspection bend event.

The current extractor sees geometrically valid entities that include:

- the inner and outer radius skins of one physical bend;
- cutout, lip, hem, bead, stiffener, and rolled-feature boundaries;
- tangent transitions that are real in the design but outside the current QC bend ontology;
- multiple STEP primitives that should contract into one neutral manufacturing event.

NIST distinguishes geometry from computer-interpretable semantic PMI and notes that semantic PMI is what enables reliable downstream manufacturing and inspection integration. Geometry-only STEP is therefore an incomplete manufacturing authority, not a bend label oracle. See the [NIST STEP File Analyzer](https://www.nist.gov/services-resources/software/step-file-analyzer-and-viewer), [PMI validation project](https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/enabling-digital-3), and [PMI conformance paper](https://www.nist.gov/publications/conformance-checking-pmi-representation-cad-model-step-data-exchange-files).

This is the dominant false-positive mechanism in the reviewed sheets.

### 2. The scan is partial, while CAD is complete

The scanner observes only visible surfaces. The CAD model contains hidden, occluded, opposite-side, and unscanned geometry. Registration or verification against the full CAD surface lets non-overlapping surfaces influence the fit and lets full-length CAD axes extend through unobserved material or voids.

NIST experimentally showed that self-occlusion can systematically misalign ICP even from a good initial pose, and that fitting only the appropriate visible CAD surface reduced residual error by three to four times. This is directly analogous to the Sherman partial-scan/full-CAD condition: [Franaszek, Rachakonda, and Saidi, 2023](https://www.nist.gov/publications/improving-fitting-cad-3d-point-cloud-acquired-line-sight-sensor).

The low-overlap registration literature reaches the same architectural conclusion:

- [PREDATOR](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Predator_Registration_of_3D_Point_Clouds_With_Low_Overlap_CVPR_2021_paper.html) focuses sampling on the predicted overlap region because registration degrades sharply at low overlap.
- [GeoTransformer](https://openaccess.thecvf.com/content/CVPR2022/html/Qin_Geometric_Transformer_for_Fast_and_Robust_Point_Cloud_Registration_CVPR_2022_paper.html) uses geometric structure and overlap-aware correspondences for low-overlap matching.
- [DeepPRO](https://openaccess.thecvf.com/content/ICCV2021/html/Lee_DeepPRO_Deep_Partial_Point_Cloud_Registration_of_Objects_ICCV_2021_paper.html) documents the domain gap between complete synthetic CAD and real partial scans, including self-occlusion, material effects, edge noise, and reduced shared content.
- [OMNet](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.html) explicitly predicts overlap masks for partial registration.

Registration is not the only problem—the reviewed overlays are often broadly aligned—but it is a necessary upstream uncertainty source and explains controlled abstentions and some line drift.

Classical registration results reinforce the need to treat pose as uncertain rather than final. Original ICP is a local optimizer, [Trimmed ICP](https://doi.org/10.1109/ICPR.2002.1047997) was designed for partial overlap, and [Welsch-loss ICP](https://doi.org/10.1109/TPAMI.2021.3054619) explicitly targets missing data and outliers. For repetitive sheet-metal geometry, the production implication is to retain multiple credible pose hypotheses until cutouts, endpoints, flange adjacency, or other identity anchors disambiguate them.

### 3. The present fusion operator rewards recall faster than precision

Unioning scan and CAD candidates is structurally asymmetric:

- a CAD candidate can recover a missed scan bend;
- every CAD primitive that is not contracted, cropped, or semantically rejected becomes another false-positive opportunity.

That is exactly what the matched counts show. Moving from retained conservative CAD to the aggressive union yields +6 TP, +21 FP, and -6 FN. The additional recall is real, but it is too expensive in precision.

### 4. CAD lines are insufficiently localized to observed support

Native-resolution visual review of all five V9 contact sheets found recurring full or overextended CAD axes crossing voids, continuing beyond observed material, or representing an entire design axis when only a shorter visible bend segment is supported. Even a correct infinite axis can fail a finite-line scorer or be unsafe as a production-floor overlay.

Relevant visual artifacts:

- `output/research/f1-step-topology-normal-verified-v9-20260822/native-visual-audits/contact_sheet_01.png`
- `.../contact_sheet_02.png`
- `.../contact_sheet_03.png`
- `.../contact_sheet_04.png`
- `.../contact_sheet_05.png`

### 5. The target-line convention is not fully signed

The project already has a 63-event CAD ontology review pack. The AI manufacturing advisory classified 60 events as likely QC bends and left three compound rounded-junction events unresolved, but it is explicitly non-authoritative. EXP_CAD_0007 showed why geometry alone cannot settle the target: 50 relocations passed geometric visual review while matched F1 fell from 0.3721 to 0.2791.

Two independent real manufacturing authorities must sign the primitive-to-event and line-placement convention before the model is allowed to learn or hard-code that mapping. This is not bureaucratic delay: without it, scoring may reward one arbitrary skin/axis convention and punish an equally plausible manufacturing convention.

## Per-case matched behavior

Retained CAD assistance improved seven of eleven cases, was F1-neutral on two, and slightly hurt two:

| Case | Family | F1 delta |
|---|---|---:|
| CGE4B0549E41 | 10839000 | +0.267 |
| CG4BDD1371B6 | 37361005 | +0.238 |
| CG65CD575B1C | 38172000 | +0.214 |
| CG4FAF273093 | 49024000 | +0.167 |
| CGF41F49066A | 38172000 | +0.165 |
| CG1E0BBA2933 | 38172000 | +0.095 |
| CG84C8FA12D2 | 10839000 | +0.057 |
| CG08506AE97D | 10839000 | 0.000 |
| CG191DEC7F81 | 49001000 | 0.000 |
| CGEBA9A869AC | 38172000 | -0.011 |
| CG64C412CC73 | 40500000 | -0.038 |

This distribution argues against “CAD is useless.” It argues for a better acceptance and contraction layer.

## Reporting defects corrected in this audit

1. `f1_program_status_v2` now exposes `baseline_metrics`, `candidate_metrics`, and `retained_metrics` separately.
2. A negative-development challenger no longer masquerades as the current retained lane.
3. The scan accuracy denominator is now 12 scored scans / 12 scored families, while the 15 executed scans remain separately recorded as execution coverage.
4. The CAD visual binding is explicitly marked as topology-only V9, not the aggressive three-source union.
5. The cross-lane headline is machine-rejected when dataset hashes, scorers, scan counts, or family counts differ.

The retained headline should now be stated as:

- retained scan-only development incumbent: F1 0.593 on its own consumed 12-case corpus;
- retained CAD-assisted development incumbent: F1 0.458 on its own consumed 11-case corpus;
- no direct cross-lane winner may be declared from those values;
- on matched Gold11, retained CAD assistance beats the scan-only control 0.458 to 0.365.

## Recommended successor architecture

Do not add more CAD sources or another permissive union. The evidence supports a four-stage, fail-closed architecture:

### Stage A — visibility-aware registration

1. Start from the existing automatic CAD/scan transform.
2. Retain multiple plausible global hypotheses on repetitive geometry instead of collapsing immediately to one transform.
3. Estimate the scan-observable CAD surface using sensor viewpoint when available; otherwise use bidirectional nearest-neighbor support and normal compatibility.
4. Refine each pose using trimmed or Welsch point-to-plane residuals on stable predicted-overlap anchors, not the full CAD and not the bend currently being inspected.
5. Publish pose uncertainty, overlap fraction, and perturbation stability. Abstain when identity remains ambiguous.

### Stage B — signed manufacturing-event contraction

1. Prefer AP242 semantic PMI or an explicit bend table when present.
2. Otherwise group inner/outer cylinders and adjacent tangent transitions into one candidate manufacturing event.
3. Resolve compound junctions, hems, beads, rolled features, and cutout-adjacent transitions under the signed CIBEC mapping.
4. Emit one neutral manufacturing axis per event, with provenance back to all source primitives.

### Stage C — scan-supported finite localization

1. Use CAD to define a narrow search corridor, not a final line.
2. Require continuous observed scan support, a two-sided normal turn, and compatible local surface ownership.
3. Crop endpoints to the visible supported interval.
4. Split or abstain across occlusion gaps instead of drawing through voids.

### Stage D — conservative set fusion

1. Preserve strong scan-only candidates.
2. Admit a CAD-only recovery only when Stage C verifies it.
3. Reconcile duplicates jointly by manufacturing-event ID and supported finite span.
4. Keep uncertain CAD events as operator-review suggestions, not autonomous QC failures.

The fusion should be late and structured: the scan branch first creates multi-scale geometric candidates without access to nominal answers; the CAD branch creates a signed event/topology graph; assignment then allows each CAD event to claim one scan candidate or an explicit `null`, with hard exclusivity so several repeated bends cannot claim the same ridge. This follows the strongest generalization lesson from partial-registration work: estimate overlap, visibility, and matchability explicitly rather than allowing full-model features to dominate weak scan evidence.

For production tolerance decisions, global best-fit alignment must be separated from datum/characteristic alignment. Pose uncertainty and local fit uncertainty should propagate to a three-way decision—conforming, nonconforming, or indeterminate—rather than turning weak coverage into a confident pass or fail.

## Fastest falsifiable experiment

Experiment name: `EXP_CAD_0008_VISIBILITY_EVENT_CONTRACTION`.

Prerequisites:

- two independent human 63-event reviews complete;
- disagreement adjudication complete;
- mapping and line convention frozen and hashed;
- no threshold selection from Gold11 outcomes.

Design:

1. Use exactly the same consumed Gold11 cases and scorer for all arms.
2. Freeze the retained scan-only control and retained CAD incumbent.
3. Run a 2×2 ablation:
   - current registration versus visibility-aware registration;
   - current primitives versus signed event contraction plus scan-supported cropping.
4. Bind native-resolution overlays to the exact candidate payload for every arm and every case.
5. Attribute every changed TP/FP/FN to registration, ontology contraction, endpoint cropping, duplicate ownership, or abstention.

Retain only if all of the following hold:

- F1 at least 0.508 and at least +0.05 over retained CAD 0.458;
- precision and recall both at least 0.50;
- exact-count cases at least 2/11;
- minimum-family recall at least 0.20 with no family regression;
- no visually unsupported line, void crossing, duplicate event, or unsafe overlay;
- byte-identical replay and unchanged preregistration.

Falsification:

- If evaluator-checked registration materially improves localization but automatic overlap-aware registration does not, the registration capability is the blocker.
- If signed contraction removes FP without losing TP, ontology was the dominant blocker.
- If both pass but F1 remains below 0.508, CAD geometry plus fused-scan evidence is insufficient under the current target, and the next capability opening should be PMI/bend-table ingestion or organized raw range views—not another union threshold.

## Final conclusion

Something was off, but it was primarily the reporting comparison—not evidence that CAD makes the system worse. CAD provides useful structural prior information on matched scans. The present failure is that complete design geometry is being asked to behave like a manufacturing-event annotation despite partial visibility and an unsigned event/line convention. The next fundable step is visibility-aware registration plus signed primitive-to-event contraction and scan-supported finite localization, evaluated on one matched frozen corpus.

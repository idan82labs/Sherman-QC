# F1 Synthetic Parts — Corpus Similarity Critique and Quarantine

Date: 2026-08-15

## Decision

The user diagnosis is confirmed. The generated parts do not sufficiently resemble the physical scan corpus, and their strongest artificial structures attract false bend predictions. Both affected synthetic lanes are therefore **quarantined from model optimization and generalization claims**.

They may remain available only as morphology-development sketches or explicitly named adversarial fixtures. They must not be used for threshold selection, detector ranking, route promotion, architecture retirement, benchmark-family counts, or evidence that physical-scan F1 improved.

Quarantined lanes:

- `f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815`
- `f1-psroc-v64-corpus-matched-prospective-v1-20260815`

Quarantined cases: `O001`, `O003`, `O005`, `O007`, `O009`, and `O011`.

## Native visual comparison

The synthetic review sheets and scored overlays were inspected at native resolution against physical scan references 48811000_A, C006, and C019.

The generated parts repeatedly use:

- perfectly straight, full-length, axis-aligned seams;
- large rectangular windows with clean perimeter edges;
- abrupt flange endpoints and disconnected relief corners;
- repeated rectangular carrier-plate and paired-lip grammar;
- symmetric shoulders and long parallel returns;
- razor-sharp zero-radius junctions and negligible material thickness;
- clean idealized observation with synthetic fuzz added after geometry creation.

The physical corpus instead contains:

- finite bend radii and visible sheet thickness/returns;
- irregular, chamfered, rounded, or stamped exterior contours;
- interrupted and local bend spans;
- asymmetric ears, attachments, punched features, and formed transitions;
- compound corners and pressed curvature;
- partial observation, dropout, edge erosion, and nonuniform scan texture.

Matching point spacing or TSDF density did not close this gap. The acquisition wrapper became more scan-like, but the underlying shape information remained clean ideal CAD.

### Direct visual verdict after side-by-side reinspection

The six generated cases are not six independent manufacturing families. Visually they are one repeated grammar: a broad plate, one or two long raised lips/flanges, a clean aperture, and a small number of abrupt tabs or shoulders. Rotation, outline changes, or different holes do not create the corpus diversity needed for a generalization test.

By contrast, the inspected physical references show continuous finite-radius material flow, visible thickness, local stamped curvature, unequal end treatments, rounded transitions, compound junctions, and features whose support varies along their length. The physical scans also retain structured acquisition texture and occlusion that follows viewpoint and geometry; the synthetic cases mostly show clean complete surfaces with noise superimposed afterward.

The most damaging visual difference is the synthetic line hierarchy. In O001–O011, the longest and highest-contrast structures are often the artificial flange shoulder, window perimeter, or carrier-plate edge. Those structures are straighter, longer, and more persistent than many designated synthetic bends, so false positives are a predictable response to the images rather than random detector mistakes.

### Per-case visual critique

- **O001:** the large rectangular window and full-span upper/lower returns dominate the part. The abrupt terminal tabs resemble cut-and-join artifacts, not formed sheet transitions. The false response at the flange endpoint is visually unsurprising.
- **O003:** the triangular tray concept is closer to a manufactured blank, but the walls meet as clean sharp planes and the side relief is disconnected/abrupt. The long perimeter seam and wall edge are stronger than the local bend features.
- **O005:** this is essentially a shallow wedge with two near-parallel long structures. It lacks the compound curvature, end relief, thickness transitions, and localized features seen in the corpus; it is a direct long-line false-positive attractor.
- **O007:** the upper plate and lower carrier overlap with paired shoulders that read as two persistent bends from several views. The intended local feature is visually subordinate. All three predecessor predictions being false matches the geometry.
- **O009:** the central window and parallel rails again dominate. Although it produced no false positives in the predecessor score, its dense parallel truth layout is artificial and cannot demonstrate transfer to physical parts.
- **O011:** the shape is mostly a clean plate plus a single long rib/shoulder. It is too simple, too complete, and too sharp; the lone detected line follows the synthetic construction cue rather than corpus-like formed geometry.

### Generator failure, not just detector failure

The false positives show that the generator optimized visible complexity by adding edges and panels, rather than manufacturing plausibility. A valid replacement must build the sheet-metal forming process first—thickness, neutral-axis bend radii, reliefs, local stamping, interrupted support, and coupled contour deformation—and only then simulate scanner sampling, dropout, and noise. Adding more noise to the current meshes would preserve the wrong geometric cues.

## False-positive mechanism

The scored predecessor overlays show that the false positives are structured, not random:

| Case | Baseline TP / FP / FN | Native visual finding |
|---|---:|---|
| O001 | 2 / 1 / 4 | The false line follows a short abrupt flange/end construction cue. |
| O003 | 2 / 2 / 2 | False lines follow a long outer/perimeter seam and an abrupt side edge. |
| O005 | 2 / 2 / 4 | False lines align with long parallel face/lip structures. |
| O007 | 0 / 3 / 4 | All predictions are false: two parallel lip shoulders and one perimeter/edge structure. |
| O009 | 3 / 0 / 7 | No false positives, but the synthetic truth is over-dense and dominated by parallel features; recall evidence is not transferable. |
| O011 | 0 / 1 / 4 | The only prediction is false and follows a long clean rib/shoulder. |

Aggregate baseline: 9 TP, 9 FP, 25 FN, precision 0.5000, recall 0.2647, F1 0.3462, exact cases 0/6.

The V63 suppression attempt reduced false positives from 9 to 8 but also deleted two true positives. Its result was 7 TP, 8 FP, 27 FN, precision 0.4667, recall 0.2059, F1 0.2857, exact cases 0/6. It is worse than baseline and, independently, the lane is not valid physical-generalization evidence.

## Case disposition

- **O001:** retire the geometry; a finite unequal-tab concept may be reused only after scan-derived remeshing and visual corpus matching.
- **O003:** retain only the compound-tray idea; the clean walls, disconnected reliefs, and coarse apertures must not be reused.
- **O005:** retire entirely; it recreates the long parallel extrusion bias.
- **O007:** retain only the finite local-feature idea; the current paired shoulders are an explicit false-positive attractor.
- **O009:** retain only the rounded-form negative-control idea; discard its carrier plate and dense parallel truth layout.
- **O011:** retire the geometry; retain only the small/local-crease testing objective.

## Binding quarantine rules

1. No quarantined prediction, score, or truth item may influence a threshold, rule, route, architecture, or retention decision.
2. No quarantined case may count toward the required physical benchmark size, family diversity, accuracy, exactness, or generalization claim.
3. Results on these cases must be labeled `synthetic_morphology_adversarial_only`.
4. Future synthetic data must pass a native-resolution visual similarity gate **before** engine inference.
5. That gate must compare each proposed case with the nearest physical morphology and reject clean rectangular coupons, repeated full-span parallel structures, zero-radius seams, and unrealistic observation completeness.
6. Source-free mathematical fixtures may still test invariants, determinism, and accounting, but they are not morphology evidence and cannot justify physical F1 claims.

## Replacement standard

A replacement lane should be scan-derived wherever possible. If procedural generation is unavoidable, it must include finite sheet thickness, manufacturing bend radii, non-rectangular contours, compound joined corners, smooth apertures, local/interrupted bend spans, asymmetric attachments, partial observation, and empirically matched noise/dropout. Visual acceptance must precede inference, and one accepted synthetic family may not substitute for an unseen physical family.

## Audited artifacts

Synthetic review sheets:

- `output/research/f1-psroc-v64-corpus-matched-prospective-v1-20260815/review-sheets/review_sheet_01.png`
- `output/research/f1-psroc-v64-corpus-matched-prospective-v1-20260815/review-sheets/review_sheet_02.png`
- `output/research/f1-psroc-v64-corpus-matched-prospective-v1-20260815/review-sheets/review_sheet_03.png`

Scored overlays:

- `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/O001.png`
- `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/O003.png`
- `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/O005.png`
- `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/O007.png`
- `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/O009.png`
- `output/research/f1-calibrated-raw-range-synthetic-v5r3b-ltco-pilot-20260815/v63-score-and-visual/visual-audit/O011.png`

Physical references:

- `output/research/f1-new-parts-20260812/v9/holdout_488/raw_visual/48811000_A_raw_six_view.png`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/visual_review/C006.png`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/visual_review/C019.png`

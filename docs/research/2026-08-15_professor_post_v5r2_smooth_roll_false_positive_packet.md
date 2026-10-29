# Sherman QC F1 — post-V5R2 smooth-roll false-positive architecture packet

Date: 2026-08-15  
Decision authority requested: GPT-5.6 Pro extended thinking  
Requested output: one bounded source-only intervention specification or a formal STOP

## Objective and locked invariants

Develop a trustworthy zero-shot bend-line inspection system for known and near-neighbor sheet-metal families. The eventual locked benchmark must contain at least 30 scans from at least 15 families and achieve at least 90% exact bend count, 90% precision, 90% recall, 99% render-admissible marker accuracy, 100% deterministic replay, <16 GiB peak RSS, <=300 s P95 latency, >=99.5% job success, and no major family below 80% recall.

For this ruling, inference must remain:

- scan-only and blind;
- CPU deterministic;
- approximately 200,000 input points maximum;
- compact-cache only, with no serialized source-point arrays;
- no CAD, labels, expected counts, filenames, identity, benchmark feedback, or post-hoc tuning;
- research-only; runtime/main and production remain closed.

## Why this packet exists

The user visually rejected the earlier synthetic parts because they did not resemble the real corpus and produced misleading false positives. That criticism was correct.

- V4: 30 long cross-section extrusion cases. Rejected as real-generalization evidence because of full-span parallel transitions and excessive smooth tangencies. Apparent V62 improvement is invalid for promotion.
- V5: first finite-plate pilot. Rejected before inference because of zero-radius seams, no thickness, stair-stepped holes, rectangular silhouettes, and an extrusion-like case.
- V5R1: added thickness and finite radii, but native views exposed inward-folded flange direction errors and a floating/inconsistent local raised feature. Rejected before inference.
- V5R2: corrected 1.45 mm thickness, 2–2.5 mm finite bend radii, rounded/chamfered finite outlines, smooth circular/slot apertures, a perforated cover, local vent-wall folds, and an 18 mm broad rolled wall negative. It passed only a limited morphology gate. It is closer to the real scans but remains simpler than compound real cases C019/C030.

## Frozen V5R2 protocol

Four opaque cases were generated, visually accepted before inference, then consumed exactly once with deterministic replay. V51 -> V52 -> V60 -> V61 -> V62 ran blind. V60/V61/V62 admitted no new births, so V52 and V62 were prediction-identical.

Truth-free visual judgments were frozen and hashed before evaluator truth was opened:

- O001: two plausible supported long flange lines; an apparent short fold uncovered.
- O003: one plausible long-wall line plus one nearby/parallel unsupported response; short peripheral folds uncovered.
- O005: one line spanning the parent plate through a local vent instead of terminating on a local fold; local folds uncovered.
- O007: one plausible short mounting-tab line plus three long parallel lines across a broad smooth rolled wall.

After the precommit, evaluator truth was opened once.

## V5R2 score and visual audit

V52 = V62 aggregate:

| TP | FP | FN | Precision | Recall | F1 | Exact cases |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 7 | 0.4444 | 0.3636 | 0.4000 | 0/4 |

Per case:

| Case | TP | FP | FN | Visual ruling |
|---|---:|---:|---:|---|
| O001 | 2 | 0 | 1 | Correct long flange localization; missed short opposite tab. |
| O003 | 1 | 1 | 2 | V16 long wall correct; V52 added a long false edge and missed two short ears. |
| O005 | 0 | 1 | 3 | V16 line diagonally overextended through parent plate/local vent; all local folds missed. |
| O007 | 1 | 3 | 1 | Short V16 tab correct; all three V52 recoveries lie at different parallel positions on the same continuous 18 mm smooth roll; opposite tab missed. |

All four scored six-view native-resolution overlays were manually inspected. The numeric score agrees with the visual interpretation.

## Mechanistic evidence

V52 admits any unrepresented line object when:

- both `cylindrical_standard` and `cylindrical_wide` sources occur;
- cluster source count >=3;
- line span >=0.50 of maximum scan bounding-box span;
- it is not near an existing prediction under 7 deg / 6 mm / 0.50 overlap NMS.

It does not currently require curvature localization, finite feature ownership, or rejection of multiple parallel axes from one continuous smooth form.

O007's three visually false V52 recoveries have:

| Candidate | cluster sources | span/bbox | two-sided angle | tight/wide normal ratio | support balance |
|---|---:|---:|---:|---:|---:|
| `CB3_61` | 3 | 0.6763 | 25.50 deg | 0.9443 | 0.9750 |
| `CB3_96` | 4 | 0.8425 | 34.66 deg | 0.4589 | 0.9862 |
| `CB1_50` | 3 | 0.8403 | 12.21 deg | 0.0000 | 0.6992 |

They are mutually near-parallel and occupy three positions across one broad smooth roll. This is visually not three bend lines.

O003's single false V52 recovery `CB2_251` has source count 4, span/bbox 0.5913, two-sided angle 43.20 deg, tight/wide ratio 0.5208, and support balance 0.7247. A simplistic low-angle cutoff would therefore remove some O007 errors but not O003 and may suppress valid rounded folds.

O005's false positive is inherited from V16, not V52, and represents a separate endpoint/ownership problem. Do not solve both problems by an opaque scalar score.

## Existing relevant mechanisms

The repository already contains older routed-profile utilities for:

- pruning long discrete fold strips on a `rolled_or_mixed_transition` route when span/support-width >10;
- pruning a unique one-support-width parallel doublet on that route.

Those utilities do not govern V52's line-object recovery and do not inspect a continuous source-observed curvature profile. Reusing them by name alone would not answer the current failure.

## Proposed prospective falsification design

The consumed O001/O003/O005/O007 cases may be used for diagnosis only. No threshold tuning, rerun, or rescue is allowed.

After your ruling, we will freeze one intervention before generating a fresh V5R3 batch. V5R3 should include at minimum:

1. multiple broad smooth rolls at different radii and partial occlusions, labeled as formed-surface negatives rather than bend lines;
2. genuine localized finite-radius folds with similar total normal turn;
3. a real hat/channel-like part with three or more close parallel legitimate bends, to falsify naive parallel-bundle suppression;
4. short return flanges and mounting tabs;
5. local vent/bead/emboss-like features with finite endpoints;
6. compound finite silhouettes, not extrusion templates;
7. scan density, dropout, thickness, and view fusion matched to the real LiDAR corpus.

Inference will run blind primary+replay, truth will remain closed through a hashed visual precommit, then scoring and manual inspection of every case will occur once. Changed cases are terminal.

## Decision requested

Select exactly one smallest scientifically defensible, source-only intervention for the next prospective cycle, or issue a formal architecture STOP.

If authorizing an intervention, provide:

1. the physical invariant distinguishing a localized bend corridor from a broad smooth roll;
2. exact input fields or a bounded source-cloud computation (no raw points serialized to cache);
3. deterministic equations and thresholds frozen before V5R3 generation;
4. whether it filters only V52 births or also audits incumbent V16 lines;
5. a fail-closed policy for missing/degenerate evidence;
6. counterexamples, especially legitimate close parallel folds;
7. exact prospective retain/retire criteria based on precision, recall, visual alignment, replay identity, memory, and latency;
8. whether the separate O005 endpoint/ownership failure should remain out of scope for this tranche.

Please ground the recommendation in established geometry/point-cloud literature where applicable (curvature scale space, normal variation, cylindrical fitting, change-point localization, or equivalent), while making clear which elements are research-derived and which are novel engineering hypotheses.

Do not recommend tuning against O001/O003/O005/O007 or opening real held-out gold. The goal is a falsifiable prospective test, not a retrospective score repair.


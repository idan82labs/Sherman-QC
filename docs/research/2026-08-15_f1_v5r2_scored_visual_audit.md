# F1 V5R2 scored visual audit

Date: 2026-08-15  
Evidence class: consumed, non-confirmatory morphology pilot  
Models: frozen V52 and V62 (prediction-identical on all four cases)

## Aggregate result

| Model | TP | FP | FN | Precision | Recall | F1 | Exact cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| V52 | 4 | 5 | 7 | 0.4444 | 0.3636 | 0.4000 | 0/4 |
| V62 | 4 | 5 | 7 | 0.4444 | 0.3636 | 0.4000 | 0/4 |

V60, V61, and V62 admitted no new predictions on this finite-radius geometry. Therefore this pilot evaluates the incumbent V52/V16 composition, not a successful V62 intervention.

## Manual visual audit of every scored case

### O001 — 2 TP, 0 FP, 1 FN

Both predictions are physically well localized to the two long finite-radius flange transitions. The short opposite tab fold is visible and missed. This is the cleanest case: acceptable precision, incomplete short-bend recall.

### O003 — 1 TP, 1 FP, 2 FN

The V16 line correctly covers the dominant long wall. The V52 cylindrical recovery adds a long parallel response on the opposite parent edge rather than either small mounting-ear fold. Two short folds are missed. This is a poor exchange: one new false positive and no recall recovery.

### O005 — 0 TP, 1 FP, 3 FN

The only prediction is a V16/non-mesh line that diagonally spans the parent plate through the local rectangular vent. It is not owned by either local vent-wall fold and does not terminate at the feature. Both vent-wall folds and the outer tab fold are missed. This is a clear finite-support/ownership failure.

### O007 — 1 TP, 3 FP, 1 FN

The short V16 prediction correctly marks one mounting-tab fold. All three V52 cylindrical recovery lines lie on the broad smooth rolled wall, at different parallel locations across the same continuous curvature. None is a localized bend line. The opposite mounting-tab fold is missed. This case isolates the highest-value next intervention: broad-roll rejection without suppressing a localized rounded fold.

## Decision

- The user's visual criticism is confirmed quantitatively and geometrically.
- V5R2 itself is only a partial correction. Its point density, thickness, rounded edges, apertures, and finite folds are credible, but its silhouettes remain too clean and its topology too simple. It lacks the corpus's compound return flanges, clustered small tabs/notches, stamped beads/embossments, locally varying bend radii, self-occlusion, and irregular scan dropout. It must not be expanded by merely parameter-jittering these four templates.
- V62 is not promoted; it has no effect here.
- V52's cylindrical recovery must not be funded unchanged on rounded forms.
- The next bounded source-only intervention should require localized curvature concentration and finite endpoint ownership before accepting a cylindrical recovery line. A broad, approximately uniform normal turn across a wide band should be treated as formed curvature rather than a bend line.
- The intervention must be tested prospectively on fresh generated cases and then on a genuinely new physical scan. O001/O003/O005/O007 are terminal diagnosis cases and cannot be used for threshold tuning or rerun rescue.

## Artifacts

- Score: `output/research/f1-calibrated-raw-range-synthetic-v5r2-pilot-20260815/score_v52_v62.json`
- Score SHA-256: `5d1a01d03a3bd439a0a1a3f691d1c73bbe0dee81c2be709dca94f6af05590b54`
- Scored visual manifest: `output/research/f1-calibrated-raw-range-synthetic-v5r2-pilot-20260815/scored-visual-audit/manifest.json`
- O001 image SHA-256: `fa361ea07a5529652fe065c81ff6ed8e5521069b7d37f3c2b475d3871a5d252a`
- O003 image SHA-256: `db130d343c6fe0f3a5f2e5ec9815b5b1a6f1ccc06ca2dd025ace7f6eca7bf115`
- O005 image SHA-256: `2091a568f7d476026ab575bef14aa99c4698caa804693c214fb3a797964f489f`
- O007 image SHA-256: `04a050095e62beab5d5e9db5ffd86dcf9f379ed7dcb3142e6f716a96572d55e1`

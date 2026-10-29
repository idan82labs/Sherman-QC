# Real-scan affine/resample V51/V52 — pre-truth visual audit

Date: 2026-08-15  
Execution: `f1-real-scan-affine-resample-v51-v52-v1-20260815`  
Status: native-resolution prediction-only audit complete; evaluator pseudo-truth remained closed

## Execution integrity

- Eight opaque development cases were executed once in a primary pass and once in an unchanged replay pass.
- V51 semantic result bytes matched primary/replay for R001–R008.
- V52 semantic result bytes matched primary/replay for R001–R008.
- The detector reported no gold, label, CAD, expected-count, part-identity, or evaluator-pseudo-truth use.
- No post-hoc tuning or rescue was performed.
- Execution manifest SHA-256: `b7045655bc58c5e7315828e739320239771e32d34710a0d9958efd15c791cde6`.

## Visual findings by case

| Case | Predicted lines | Visual critique before pseudo-truth |
|---|---:|---|
| R001 | 10 | Fail. Several near-parallel predictions describe the same long fold, and multiple short fragments accumulate at the central formed junction. The geometry contains plausible bends, but candidate multiplicity is visibly excessive. |
| R002 | 4 | Conditional visual pass. The same parent morphology is much cleaner after coarser resampling; the four axes mostly follow major formed boundaries. The 10-to-4 change versus R001 is itself a severe robustness failure and likely sacrifices shorter junction bends. |
| R003 | 8 | Fail. Most candidates track the tray perimeter, but multiple long axes compete for the same boundaries and one corner candidate visibly overextends beyond supported geometry. |
| R004 | 5 | Fail. Two axes cross the broad flat tray face (one diagonal, one nearly horizontal) instead of following a physical fold. These are unequivocal visual false positives. |
| R005 | 3 | Fail. All three long axes are nearly coincident on one physical flange transition. At least two are duplicate representations of the same visible fold. |
| R006 | 5 | Fail. All five axes cluster on the same long flange transition. Resampling increased duplicate multiplicity from three to five rather than preserving a single physical bend. |
| R007 | 3 | Fail/conditional. One axis follows the opposing flange while two largely duplicate the other visible long fold. The part appears to contain two dominant opposing folds, not three independent ones. |
| R008 | 2 | Visual pass. One line follows each of the two dominant opposing folds, with no obvious flat-face crossing or duplicate. This is the cleanest result in the lane. |

## Cross-pair conclusion

The visual instability is not a synthetic-shape artifact: all four A/B pairs derive from the same real manufactured scan morphology. Mild affine change, resampling, and normal noise alter the V52 prediction counts as follows:

- parent U01: 10 → 4
- parent U02: 8 → 5
- parent U03: 3 → 5
- parent U04: 3 → 2

The dominant failure mode is candidate ownership and consolidation. Multiple routes emit parallel or overlapping lines for one visible physical fold, while some resampled variants also admit unsupported axes across flat faces. V52 added twelve candidates across the eight primary cases, and several visible failures are among the added or competing route families. The correct next optimization target is therefore a source-only, geometry-supported cross-route ownership/NMS gate—not more synthetic part generation and not case-specific threshold tuning.

## Evidence limitation

This audit establishes a visually real robustness failure, not absolute F1. R001–R008 represent four already-exposed parent parts, and their evaluator references are transformed V51 development pseudo-truth rather than human gold. The pseudo-truth may now be opened only to quantify self-consistency and attribute which reference lines were preserved or lost. It cannot promote the model or substitute for scanning and independently annotating the newly received physical part.

## Bound audit artifacts

- R001: `a8bf15663b5db3b930b11c123d00ac98a8eb35f0ea66a2107b9d0cad30c2fc28`
- R002: `3fd89fce3553aac7c24ea2db64695451952c74fca7cfc511012787d5cbc5cbb1`
- R003: `91ee94fe47fe6ac2f6504f70788fe83539c04557c3b56622c03e5fe044825937`
- R004: `de5c9e255e699579c0be565e02de5c67bae5602a2b5bdf3c797e28bebd34b5d6`
- R005: `53d9a0c518f1137aabd68b9458944b0c7a787ae4af6c32372ac6c7d77a889b1f`
- R006: `df66e2265dba5d1acba86c923887f92c3eb6b571efa8b9b6b650c5c29ed1f47f`
- R007: `b82aabc68d48913c31a10553453aaae98c3008d143079d600a0f44e3bd2fd861`
- R008: `bd768685069a466f4dec1f635d93806ed35cfc77ee7c9133bb9270ca5ebd15b2`

## Decision

Retire this lane as an input-generation experiment after one run; retain it as a deterministic development regression. Do not tune on individual R-cases. Open pseudo-truth once for bounded self-consistency accounting, then fund a single general cross-route consolidation intervention only if the accounting agrees with the visual diagnosis.

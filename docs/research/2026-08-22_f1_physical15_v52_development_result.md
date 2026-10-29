# V52 physical-15 development result

Date: 2026-08-22

## Result

The unchanged retained V51→V52 baseline completed 15 primary and 15 replay executions across 15 development-consumed physical families. Every semantic result was byte-identical between primary and replay. Peak stage RSS was approximately 1.53 GB. Primary P95 end-to-end case latency was approximately 390 seconds, exceeding the 300-second target.

V52 recovered 22 additional candidates over V51. Native-resolution review shows that most additions increased duplication or cross-face clutter rather than clearly restoring unique missed folds.

Visual adjudication:

- 1 pass: D015;
- 4 conditional: D003, D007, D010, D012;
- 10 failures: D001, D002, D004-D006, D008-D009, D011, D013-D014.

The dominant error is cross-route candidate ownership and consolidation, compounded by failure to reject lines crossing visibly flat faces. The physical lane confirms that further candidate birth is not the priority.

After both V51 and V52 executions were frozen, a scan-hash join recovered the pre-existing July multiview line annotations for D001-D012. These are development-consumed single-reviewer annotations, not untouched confirmatory truth, but they provide valid development scoring without using sender counts in the model path.

| Development metric (12 physical families) | V51 | V52 | Delta |
|---|---:|---:|---:|
| Precision | 0.5932 | 0.4500 | -0.1432 |
| Recall | 0.5932 | 0.6102 | +0.0169 |
| F1 | 0.5932 | 0.5180 | -0.0752 |
| Exact count accuracy | 0.2500 | 0.2500 | 0.0000 |
| Exact localized target-set accuracy | 0.0833 | 0.0000 | -0.0833 |
| Minimum-family recall | 0.0000 | 0.0000 | 0.0000 |

V52 is therefore a **physical-development regression**. Its 22 recovered candidates yielded only one additional matched line, added 20 false positives, and removed the only exact localized case. V51 remains the retained physical-development reference.

Manual native-resolution comparison of D010 against its frozen gold overlay showed that the official 5° matcher rejects visually close axes at approximately 7°. The official result remains unchanged. A non-authoritative 2×2 sensitivity grid at 5°/7.5° and 4/6 mm confirmed that V52 still regresses at every setting; its F1 delta ranges from -0.0634 to -0.0804. The negative direction is therefore not an artifact of the single 5° boundary.

The machine-readable comparison is `output/research/f1-physical15-development-v52-20260822/GOLD12_DEVELOPMENT_COMPARISON.json`.

## Generalization Progress Indicator

**GPI-C — no clean generalization evidence; development direction negative.**

The execution is valid, deterministic, physical, and memory-safe, and development deltas are now measurable. The grade remains C because the families are development-consumed and the annotations are not a two-annotator untouched-family holdout. Within that limitation, both the F1 delta and visual ruling are negative.

## Ownership funding diagnostic

A deterministic nested leave-one-family-out diagnostic tested source-feature ownership filtering on the retained V51 predictions. The held-out family was excluded from both model fitting and threshold selection.

- Extra Trees: F1 0.5812.
- Random Forest: F1 0.5913, the best result, still 0.0019 below V51.
- The Random Forest improved exact-count accuracy from 0.2500 to 0.3333 but reduced recall from 0.5932 to 0.5763.
- Native-resolution D002 review confirmed the tradeoff: three visible false-positive crossings disappeared, but one real bend disappeared as well.

The diagnostic is recorded at `output/research/f1-physical15-development-v51-20260822/OWNERSHIP_NESTED_LOFO_DIAGNOSTIC.json`. `fund_successor` is false. No V74 was created.

## Next funded work

Do not fund a new candidate generator or the tested source-feature ownership selector. Preserve V51. The next useful input is additional independently annotated physical morphology—preferably the pending seven SharePoint families—or a second independent annotation pass on a fixed subset. New model work should target the zero-recall families only after that evidence is frozen.

No runtime/main or production promotion is authorized.

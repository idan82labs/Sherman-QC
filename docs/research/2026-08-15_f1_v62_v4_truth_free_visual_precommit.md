# F1 V62 on prospective complex V4: truth-free visual precommit

Date: 2026-08-15

## Scope and lock

- Corpus: `sherman_qc_f1_calibrated_raw_range_prospective_complex_v4`
- Token: `SYNTHETIC_CALIBRATED_RAW_RANGE_V4_PROSPECTIVE_COMPLEX_NON_CONFIRMATORY`
- Comparison: retained V61 versus frozen V62 distributed-turn consensus.
- Evaluator-only truth, family identities, intended counts, and scored overlays remained closed throughout this review.
- V61 and V62 primary/replay executions were already complete and byte-identical before review.

V62 produced seven births across six changed cases: O013, O014, O020 (two births), O023, O024, and O029. The review included all six source-only six-view fused overlays and all six eight-view native calibrated depth sheets.

## Visual precommit

| Case | Births | Truth-free visual ruling |
|---|---:|---|
| O013 | 1 | Plausible long transition on a rounded channel wall. It follows coherent geometry through all fused views and is not explained by a hole. Adjacent-parallel duplication risk is visible, so the score must decide whether this is a distinct small facet or an over-segmentation. |
| O014 | 1 | Plausible lower web/return transition, consistently aligned with the channel geometry. No scan-hole or isolated-noise explanation is apparent. |
| O020 | 2 | Both births land on separate long side transitions of the compound rolled section. They are mutually separated, remain coherent in end and oblique views, and are visually the strongest V62 additions. |
| O023 | 1 | Plausible tangent/crease transition on the upper rounded wall. The line is spatially coherent, but its close parallel relationship to an incumbent creates moderate duplicate risk. |
| O024 | 1 | Plausible narrow intermediate facet transition in a deep multifacet trough. It is supported across the raw views; close packing with two incumbents creates moderate over-segmentation risk. |
| O029 | 1 | Plausible missing ridge within the mixed-radius labyrinth. The birth tracks a persistent structural transition rather than a validity boundary. |

Pre-score decision: **no visual veto**. All seven births are spatially credible enough to score. O013, O023, and O024 are explicitly precommitted as duplicate-risk cases; a favorable aggregate will not override poor matching or visibly redundant scored placement on those cases.

## Frozen evidence

- Truth-free comparison root: `output/research/f1-calibrated-raw-range-synthetic-v4-20260815/v62-truth-free-visual-comparison`
- Comparison tree SHA-256: `377a43d477fc166fdcd559b46cc2c2b1ed8dcf056d907cfc1fddab01d84b0dfa`
- Native-depth audit root: `output/research/f1-calibrated-raw-range-synthetic-v4-20260815/v62-truth-free-native-audit`
- Native-depth tree SHA-256: `92f2f9e4d09830a081a9ba8f654e981ab2f43a78aad0c48afc9725d6297e97b6`
- Renderer SHA-256: `a55d78539466c45429ab319ceb6f5838108f342702e3a9ccf19de1292fa4593a`

This precommit closes the truth-free review. Evaluator truth may now be opened once for scoring and scored visual verification.

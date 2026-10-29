# F1 calibrated raw-range synthetic V3 — complex geometry result

Date: 2026-08-15

## Outcome

The retained V61 source-only route materially improved F1 on a new 30-capture, 15-family complex synthetic corpus while also improving precision. The corpus deliberately stresses smaller bends and creases, close parallel features, nested and compound profiles, tight fillets, and broad rounded facets at scanner-like density and spacing.

| Route | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| V52 baseline | 48 | 20 | 200 | 0.7059 | 0.1935 | 0.3038 |
| V60/V61 | 64 | 23 | 184 | 0.7356 | 0.2581 | 0.3821 |

Absolute F1 gain: **+0.0783**. Relative F1 gain: **+25.8%**.

V60 produced 19 new source-only lines across 14 cases. The frozen V61 ownership guard admitted all 19. All 19 additions matched a private synthetic bend after the truth-free audit was frozen. The aggregate net change is +16 TP, +3 FP, and -16 FN because three new matches displace older baseline assignments under the one-to-one evaluator.

## Visual audit

Before scoring, every changed case was inspected in a three-view fused overlay and every proposed line was projected into all eight native 384×288 calibrated depth frames. After scoring, all 14 changed cases were inspected again with matched truth, missed truth, TP, and FP overlays. The cyan V61 additions visually coincide with black matched truth lines in every changed case. The metric gain is therefore consistent with the geometry rather than an obvious matching artifact.

The audit also makes the remaining limitation plain: recall is still low. Dense close-parallel profiles and distributed small-angle rounded transitions retain many magenta missed truth lines. V61 is a meaningful recall tranche, not a complete bend parser.

## Family findings

V61 improved 9 of 15 hard families. Notable gains include:

- HF01 micro-ribbed hat: F1 0.000 → 0.211.
- HF06 compound S-joggle: F1 0.609 → 0.815.
- HF11 close parallel crease shelf: F1 0.000 → 0.222.
- HF12 rounded wide pan: F1 0.400 → 0.500.
- HF15 asymmetric multi-angle labyrinth: F1 0.143 → 0.207.

No gain was obtained on HF08, the 2.2–4 mm close-spaced double-step family, or HF13, the dense miniature joggle train. HF07 fell slightly from 0.421 to 0.400 because recall was unchanged and one additional assignment remained false positive. These families define the next bounded recall problem.

## Decision

Retain V61 as the strongest synthetic research candidate. It remains research-only and is not promoted to physical or production evidence. The next development tranche should target distributed rounded-turn evidence and near-sampling-limit micro-steps with an additional source-only route; it must not loosen V61 globally. Any such tranche developed after opening V3 truth must be validated prospectively on a fresh corpus before retention.

## Artifact hashes

- V52/V60 score: `a503bc6ad7dab7bd8d6500144e3742b222763e65cbf705b8c562034aaf189630`
- V52/V61 score: `ce48e3d1d883c6c15fbc0e62b4cbfd66fe42d0d224e0ed47908148651b855673`
- Aggregate post-score visual hash: `23ba776a3a51d2a8d8674a9eeb62f85e91d56c170bd4d5aac4d7a4aaf15fa33e`
- V3 execution adapter: `99ab9b0d6bd4f17a824e0b04767d668e8c1d66a2002306337d203a35bde61630`
- V3 scoring adapter: `bf99215b3234f9b06c0d607d52fecb1112df7ef0c5b9231457e4874bd9e7253a`

## Evidence locations

- Corpus and execution outputs: `output/research/f1-calibrated-raw-range-synthetic-v3-20260815`
- Frozen truth-free ruling: `docs/research/2026-08-15_f1_calibrated_raw_synthetic_v3_truth_free_ruling.md`
- Post-score overlays: `output/research/f1-calibrated-raw-range-synthetic-v3-20260815/v61-scored-overlays`

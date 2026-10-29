# Calibrated raw-range synthetic V3 construction audit

Date: 2026-08-15

## Frozen ruling

`PASS_CONSTRUCTION_OBSERVABILITY`

The full V3 corpus is accepted for one-shot V52/V61 execution. No detector output existed during this audit, and no detector setting was changed.

## Corpus checks

- 30 cases across 15 new complex-geometry families
- 248 intended bend centers, 6–11 per case
- 240 native 384x288 depth frames
- 30/30 unchanged raw-range preflights passed
- fused source size: 28,328–99,043 points
- sampled backprojection agreement within 0.02B: 1.0 for every case
- native valid fractions: 0.021575–0.261927 across all views

## Manual visual audit

Every case was inspected in:

1. three fused-cloud projections with the evaluator bend centers overlaid; and
2. all eight native depth frames with the same bend centers projected through the frozen calibration.

Findings:

- all intended bends lie on coherent rendered sheet surfaces rather than empty space or unrelated geometry;
- the 2.2–4 mm micro-steps and 2.5–7 mm shallow creases remain visually separable in complementary views;
- close parallel crease pairs remain distinct at the chosen LiDAR-like fusion density;
- large-radius families show distributed rounded transitions rather than collapsing into single sharp hinges;
- compound returns, channels, hats, and labyrinth profiles are geometrically intact after holes, dropout, and fusion;
- O016 is the hardest grazing capture, with several sparse views, but its shallow rounded structure remains observable across the complementary view set;
- line endpoints sometimes cross deliberate holes or capture truncation, as expected, while the surviving portions remain aligned to the intended bend locus.

No case was rejected, regenerated, or substituted after this full-corpus audit.

## Frozen hashes

- generator: `42cfbe2f9cf588d99abc12978cc6241fa63df0c89dbf0041bddbe795c791e0cb`
- public manifest: `e5c38f108b47eb1e70463d6ab5d0eefac897fea884b3685dcd065586e99b43f0`
- private manifest: `147f4e8cfefe4b7ff8f36b261ab1156346663d4e3a1a1e9b39ebef24bf7e3018`
- renderer manifest: `be932a92036036a0d7a79f51fadd1a37533567108892f2b09b1e23890e7114fa`
- ordered preflight-file hash ledger: `7357175b7cc8b710ffe5969e1b8951d3e8718a19ba573120ae88d1dec0ecc532`
- ordered observability-manifest hash ledger: `2799aeae6756dc2ce8a7f1562e27be06cde61898ec77cb3c3b75d79a81bdddeb`

## Artifacts

- corpus: `output/research/f1-calibrated-raw-range-synthetic-v3-20260815`
- preflights: `output/research/f1-calibrated-raw-range-synthetic-v3-20260815/preflight`
- fused/native construction audit: `output/research/f1-calibrated-raw-range-synthetic-v3-20260815/geometry-observability`

## Next action

Execute the frozen V52 baseline and V61 candidate through primary/replay routes without tuning, then perform a truth-free visual audit of every V61 changed case before scoring.

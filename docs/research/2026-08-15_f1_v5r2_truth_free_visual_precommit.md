# F1 V5R2 truth-free visual precommit

Date: 2026-08-15  
Corpus: `f1-calibrated-raw-range-synthetic-v5r2-pilot-20260815`  
Status: frozen visual judgment before evaluator-truth scoring; four cases are consumed and may not be tuned or rerun.

## Scope and evidence limits

V5R2 is a four-case, non-confirmatory morphology pilot. It was built only after V4 and V5 were rejected visually for corpus mismatch. V5R2 includes finite sheet boundaries, 1.45 mm thickness, finite-radius folds, smooth apertures, local formed features, and a broad-radius negative control. It passed only a limited morphology gate: it is materially closer to the real scan corpus than the extrusion-like V4 cases, but it remains simpler than the compound real parts C019 and C030.

The judgments below were made from the fused source cloud plus frozen V62 prediction overlays. Evaluator-only truth files were not consulted while making these prediction-quality judgments. The generator's general case intent is necessarily known, but line-level evaluator matching and metrics remained unopened. V52 and V62 are prediction-identical on this pilot because V60/V61/V62 admitted no new births.

## Visual findings

| Case | Predictions | Truth-free visual judgment |
|---|---:|---|
| O001 | 2 | Both lines lie on visually supported finite-radius flange transitions. The visible part also appears to contain an additional short folded tab with no prediction, so recall is suspect even though precision looks plausible. |
| O003 | 2 | One line follows the dominant long wall transition. The second is a nearby parallel/offset response on the same region rather than a clearly separate physical fold. Short peripheral folds appear uncovered. Likely duplicate/false-positive plus false negatives. |
| O005 | 1 | The single line spans a large portion of the parent plate through the local vent structure instead of terminating on one supported local fold. This is an ownership/span failure and is visually suspicious as a false positive. Local vent-wall folds appear uncovered. |
| O007 | 4 | One short line is supported by a small mounting-tab fold. Three long parallel lines run along the broad smooth rolled wall. Those three are not localized sharp bends; they are visually clear smooth-form false positives. This is the strongest failure and directly validates the user's complaint. |

## Pre-score ruling

1. The user's visual criticism is correct: the earlier synthetic design was too unlike the real corpus and obscured false-positive behavior.
2. V5R2 is useful precisely because its more realistic finite geometry reproduces two concrete incumbent failure modes:
   - smooth-roll/cylindrical over-detection;
   - overextended line ownership through a local formed feature.
3. V62 has no measurable mechanism contribution here; it collapses to V52. It cannot be promoted from this evidence.
4. After scoring, the next architecture may target only source-observable discrimination between a localized bend corridor and a broad smooth roll, together with finite support/endpoint ownership. The four consumed cases may be used for diagnosis but not for tuning or rerun rescue.

## Frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| O001 visual audit | `fc0b2dafa37684ef3717defd39d0604fbf5b309a0debc2abd9723b01f20c092c` |
| O003 visual audit | `c9337123c44d94397d78ca83553b6329e5b6dc2f89bc778894bb92309833cafe` |
| O005 visual audit | `38539c0194914f90ffe16aea877a72f2f22947e3bafb4d76eb461d7b28c123e0` |
| O007 visual audit | `1d85110a2f8e8309f525f1879ca9e1cb33f17c37758fc7e123b43d0b2d3ebbde` |
| O001 semantic result | `17609bf03da55affe4373724e1a7b6e3045c2948167dd738cbdbe07688e16367` |
| O003 semantic result | `6eb54e4a77a1653faa4303f3f484eefe5806065c3f421a8b42965bf8d498fc0b` |
| O005 semantic result | `dc695fa8f1958245fa984ea97f1a5c7cc6acf2fe8e0b4fd3f0e4f8773daa8977` |
| O007 semantic result | `323c049c83ef480c9053b2d2e2843eee82d83f3b8b5c43681aa84efaf1e439fa` |
| V62 execution manifest | `8d4834bc46f0733eb52066342be5be9ead718010bdac7d3100ab410fd5d0a174` |
| Private manifest | `d52f13c6a65753b49aa5c8303dca12595755e3175ec01478b10624feb3b01a36` |


# F1 V43 multi-addition explosion guard — development result

## Ruling

The source-only V43 explosion guard is the strongest **non-regressive tactical
change** found in the 2026-08-31 optimization pass.  It removes 18 false
positives across the consumed 65-case catalog without losing a true positive.
It is replay-deterministic, but it remains post-hoc development evidence and is
not authorized for promotion.

The guard is deliberately small: when frozen V43 reports more than one
post-merge addition, it falls back to the already-produced V16 predictions;
otherwise it preserves frozen V51.  No label, expected count, part identity,
filename, CAD, drawing, or sender metadata enters inference.

## Exact score

| Slice | Version | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Today 28 | frozen V51 | 14 | 63 | 51 | 0.181818 | 0.215385 | 0.197183 |
| Today 28 | guarded | 14 | 52 | 51 | 0.212121 | 0.215385 | **0.213740** |
| Legacy 37 | frozen V51 | 124 | 85 | 58 | 0.593301 | 0.681319 | 0.634271 |
| Legacy 37 | guarded | 124 | 78 | 58 | 0.613861 | 0.681319 | **0.645833** |
| Pooled 65 | frozen V51 | 138 | 148 | 109 | 0.482517 | 0.558704 | 0.517824 |
| Pooled 65 | guarded | 138 | 130 | 109 | 0.514925 | 0.558704 | **0.535922** |

Exact-case counts are unchanged: 2/28, 3/37, and 5/65.  The guard triggers on
today case T020 and legacy runtime cases T016, T033, and T034.  It removes 11,
2, 3, and 2 false positives respectively.

All 65 primary/replay successor payloads are byte-identical.  An independent
run through the canonical gold-catalog scorer reproduced the pooled 138 TP / 130
FP / 109 FN result exactly.

## Why this is retained over the larger fitted gains

Two more aggressive source-only layers were also implemented:

- Precision ownership fits the same 28 cases at 14 TP / 0 FP / 51 FN, F1
  0.354430.  However, applying those 23 fitted cutoffs to legacy-37 collapses to
  7 TP / 3 FP / 175 FN, F1 0.072917.  It is a cohort-specific resubstitution
  ceiling, not a transferable successor.
- Source recovery fits the same 28 cases at 29 TP / 76 FP / 36 FN, F1 0.341176.
  Grouped leave-one-case-out falls to 16 TP / 86 FP / 49 FN, F1 0.191617,
  slightly below frozen V51.

Those experiments are useful diagnostics: existing candidate geometry contains
29–30 of the 65 bends, but the learned selection boundary does not transfer.
They are not retained as a new baseline.

## Bindings

- Guard module SHA-256:
  `d74c7abcaa9f7f1094b631215db252f27e7758ffbcffcaa11ed35979941a39f5`
- Cached evaluator SHA-256:
  `3b5a228d621e25a0e0b79ee56194d074c4853e82843ed74af004942321f5534a`
- Focused test SHA-256:
  `ed231c8ce1edf49d57ecc8cb299a8ae7071afcd462004f0c680cede9e767d7c9`
- Development result SHA-256:
  `d8221b942a5a581b123aceb559f67b5a7e574b7f3910fea20d7fa6e4599b64a6`
- Prediction-map SHA-256:
  `136bb717131e6de10dd327a402b19a36608faa6f476bf7c78042e9c7d635ee3f`

Focused unit tests: 6/6 pass.  Canonical-scoring recomputation also passes.

## Scientific limitation and next gate

The `>1` cutoff was chosen after inspecting consumed development failures.
Therefore the observed no-harm result is not generalization evidence.  Freeze
these bytes now and test once on the next independently labeled, prediction-
hidden daily delivery.  Retain only if that prospective cohort loses zero true
positives and reduces false positives; otherwise retire the guard without
retuning.

The dominant unsolved problem is candidate birth and geometry: 35/51 today
false negatives have no scorer-valid raw-pool candidate.  Further threshold
tuning cannot solve that deficit.


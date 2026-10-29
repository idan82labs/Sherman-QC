# F1 V51 daily-28 axis/span recovery — development result

Status: **development-only / consumed-label optimization / not promotion evidence**.

The cached successor composes the precision-retained V51 predictions with the
final R3 source-only recall additions, then tests a separate source-only
candidate-geometry layer.  That layer normalizes each cached line object's
source axis and emits deterministic center-, low-endpoint-, and high-endpoint-
anchored variants at fixed and scaled spans.  Runtime application cannot open
the catalog, labels, expected counts, CAD, drawings, filenames, or part IDs.

## Result

| Lane | TP | FP | FN | Precision | Recall | F1 | Exact cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| Precision + final R3 recall union | 29 | 14 | 36 | 0.674419 | 0.446154 | 0.537037 | 3/28 |
| Axis/span successor, resubstitution | 37 | 15 | 28 | 0.711538 | 0.569231 | 0.632479 | 5/28 |
| Seven-fold case-group-held-out diagnostic | 29 | 26 | 36 | 0.527273 | 0.446154 | 0.483333 | 3/28 |

The resubstitution selector adds eight true positives and one false positive.
It recovers T003-B001, T005-B003/B004, T007-B001, T020-B001, T022-B001, and
T028-B001/B002, with no regressed matched gold line.  T028 also receives the
single added false positive.

The held-out result is the governing scientific caution: held-out trees add no
true positive and twelve false positives relative to the 29/14 base union.  The
operating point was selected on all 28 cases (it is not nested), so even that
held-out diagnostic is not untouched evidence.  The 0.632479 value is a useful
upper-bound development target, not a credible estimate of production
accuracy.

## Geometric ceiling

At the exact scorer thresholds (5 degrees axis, 4 mm transverse offset, and
0.60 overlap in both directions):

- the original pool can match 30/65 gold bends;
- endpoint-anchored axis/span variants can match 36/65;
- the base union plus all variants can match 37/65.

The selected resubstitution model reaches that 37-bend geometric ceiling.  The
remaining 28 false negatives are not recoverable by this axis/span transform
family and require new candidate axes/offsets or a different representation.

## Reproducibility and custody

- Final model: `data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json`
- Final model SHA-256: `8689db67f8a7f1df04c4cc43b104ef116e0cecb9004d5aca580764091318ec7b`
- Fresh bound output root: `output/research/f1-v51-today-axis-span-recovery-v1-20260831-r3`
- Cached report SHA-256: `c3fa73aca8aa31791e702222591b92491b9e132b7b1ed3fc06a3c7db6c480ed3`
- Final R3 recall model binding: `5f8b5e43cefb9114dafccaf65159270bec7875657a9d3df1acc699b0156a1897`
- All 28 primary/replay successor artifacts are byte-identical.
- Eight focused tests pass; 32 tests pass across axis/span, R3 recall, and precision ownership.

The frozen V51 modules, original daily baseline, and final R3 recovery artifacts
were read only and were not rewritten.

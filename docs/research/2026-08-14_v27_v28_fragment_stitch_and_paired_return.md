# F1 V27–V28 fragment-stitch and paired-return cycle

Date: 2026-08-14

## Outcome

The cycle produced one historically validated precision improvement and one
promising new-case recall mechanism.

- **V27 collinear fragment stitching is retained.** Across strict 35-case
  leave-one-family-out evaluation it removes three false positives without
  changing true positives or false negatives. F1 rises from 0.487671 to
  0.491713. It also converts four selected fragments on U04 into one physical
  right-base bend axis.
- **V28 paired-return completion remains research-only.** It adds no candidates
  in the 35 historical cases, so it has zero measured historical regression but
  no historical positive validation. On the new U04 mechanism case it recovers
  the missing right wall-to-return bend and produces the four visually and
  drawing-confirmed physical axes.
- **The V26 model combined with frozen V16 routes is retired.** V26's score
  semantics are not compatible with V16's guarded route logic; that combination
  changed the batch from 5/4/3/6 predictions to 4/4/4/7 and introduced visible
  duplication on U03 and U04.

No production, runtime/main, or benchmark-feedback path was changed.

## Scientific contract

Inference used source geometry and source-derived candidate evidence only. It
did not use CAD, drawing bend counts, labels, expected counts, filenames, part
identity, good/bad status, or benchmark feedback. U04's drawing and sender count
were opened only after the source-only outputs had been frozen. They are
evaluator-only evidence, not inference inputs.

U01–U03 are useful regression replays but are not new independent evidence in
this batch: earlier assistant visual precommits already existed for them. U04 is
the new mechanism case. Its post-inference drawing adjudication is strong
mechanism confirmation, but it must not be reported as an unbiased blind F1
measurement.

## Quantitative results

| Tranche | TP | FP | FN | Precision | Recall | F1 | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| V22 retained baseline | 89 | 100 | 87 | 0.470899 | 0.505682 | 0.487671 | 3/35 |
| V27 + collinear fragment stitch | 89 | 97 | 87 | 0.478495 | 0.505682 | **0.491713** | 3/35 |
| V28 + paired-return completion | 89 | 97 | 87 | 0.478495 | 0.505682 | **0.491713** | 3/35 |

V27 triggered on only C021 and C024. It removed one and two false positives,
respectively, with no true-positive loss. V28's rare three-lane paired-return
precondition triggered zero times in the historical set.

| Batch output | U01 | U02 | U03 | U04 |
|---|---:|---:|---:|---:|
| Frozen V16 | 5 | 4 | 3 | 6 |
| V27 | 5 | 4 | 3 | 3 |
| V28 | 5 | 4 | 3 | **4** |

## Native-resolution visual adjudication

Every V28 batch artifact was inspected at native resolution.

- **U01:** five supported folds—four long web/fork folds plus a short transverse
  closure—with no obvious broad-face or perimeter alias.
- **U02:** four clean tray base-to-wall folds.
- **U03:** three close return-fold axes. They overlap in plan view but remain
  distinct in the previously adjudicated cross-section curvature zones.
- **U04:** the frozen V16 output contains four collinear fragments for the same
  right-base fold and misses the adjacent right return. V27 fits the four
  fragments into one full-span physical axis. V28 then stitches three
  high-support, high-normal-turn source fragments into the missing counterpart.
  The final overlay contains two bends on each long edge—base-to-wall and
  wall-to-return—and no visible hole, perimeter, or broad-face false line.

The evaluator-only drawing's Detail G independently shows the same two-bend
return profile on each side, for four physical bends total. Because it was
opened after the output freeze, it validates the visual interpretation without
having contaminated inference. Because V28 was designed after inspecting U04's
failure mode, this remains post-hoc mechanism evidence rather than a promotion
score.

## Retain, restrict, and next test

V27 is the funded tranche: it improves strict LOFO precision, preserves recall,
and resolves the new scan's fragmentation failure. V28 should remain behind a
research flag. Its implementation is deliberately narrow: exactly three
selected physical lanes, one close parallel pair, one far parallel base,
high-support/high-normal-turn counterpart fragments, material span gain, and at
most one recovery.

The next decisive test is not another threshold sweep. Run the frozen V28 rule
on a genuinely unseen paired-return scan, freeze the output, then obtain an
independent human bend selection or drawing adjudication. If it recovers the
counterpart without adding aliases, fund it; if it does not, retire it without
rescue tuning.

## Immediate recent-pool trigger sweep

The unchanged V28 selector was subsequently executed across 26 valid recent
source-only candidate pools using 12 parallel workers. The sweep included 21
full-resolution routes, two bounded small-pool routes, and three bounded complex
routes. Two complex-route cases entered V28 with exactly three predictions.

V27 triggered zero times and V28 triggered zero times. This is a favorable
specificity result: the new rule did not opportunistically complete unrelated
three-line profiles. It does not supply the missing positive validation, so the
research-only restriction remains unchanged.

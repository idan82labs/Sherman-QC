# V51 source-only recovery V1 — consumed-development result

## Ruling

The balanced source-only recovery profile materially improves the **same 28-case
consumed development cohort**, but grouped leave-one-case-out does not validate
the gain. Retain this result as candidate-lineage and calibration evidence only;
do not promote it or represent the resubstitution score as generalization.

The frozen V51 cache and baseline outputs were not rerun or modified. Recovery
used only the cached V51 semantic result, cached scan-only candidate pool, and a
frozen feature tree/config. Primary and replay successor bytes matched for all
28 cases. Runtime application opened no catalog, labels, expected counts, CAD,
drawings, or part identity.

## False-negative lineage

V51 has 51 false negatives on the daily cohort:

- 16/51 have at least one scorer-admissible line in the raw candidate pool but
  are lost by selection or ownership.
- 0/51 exist only in a later/earlier retained stage while absent from the pool.
- 35/51 never appear as scorer-admissible geometry anywhere in the cached V51
  line stream.

The raw-pool one-to-one oracle ceiling is 30/65 TP (recall 0.461538), but emitting
all 15,627 candidates has precision 0.001920 and F1 0.003824. Candidate birth,
not only selection, remains the dominant recall ceiling.

## Same-cohort resubstitution score

| Result | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen V51 | 14 | 63 | 51 | 0.181818 | 0.215385 | 0.197183 |
| Recovery V1 balanced | 29 | 76 | 36 | 0.276190 | 0.446154 | 0.341176 |
| Delta | +15 | +13 | -15 | +0.094372 | +0.230769 | +0.143993 |

The successor recovers 15 of the 16 pool-born false negatives with zero
regressions among the 14 V51-matched gold lines. Exact-case count remains 2/28.

## Grouped leave-one-case-out diagnostic

For each fold, the decision tree was refit on the other 27 cases and applied to
the held-out case with the already selected operating constants. Aggregate:

| Diagnostic | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 28-fold grouped LOCO | 16 | 86 | 49 | 0.156863 | 0.246154 | 0.191617 |

This is below frozen V51 F1 0.197183. It is not a fully nested estimate because
the feature set, tree hyperparameters, supplement thresholds, replacement rule,
and operating threshold were selected with awareness of the full 28-case
development cohort. It nevertheless shows that the large 0.341176
resubstitution score is calibration/selection overfit and does not transfer
reliably across cases.

## Implementation contract

- The runtime tree artifact contains source-feature thresholds and leaf
  probabilities, not case IDs, bend IDs, counts, truth endpoints, or per-case
  decisions.
- Candidate ownership uses 5 degrees, 4 mm, and bilateral 0.60 span overlap.
- Incumbents own duplicates by default. A narrowly bounded rerank may replace a
  low-score adjacency owner with a probability-1 cylindrical consensus owner.
- A high-curvature adjacency supplement uses source-only confidence, transition,
  normal-eigenvalue, and corridor-density fields.
- This model was fitted on AI-adjudicated, non-authoritative development labels.
  `promotion_eligible=false` and `generalization_evidence=false` are mandatory.

The next recall attack should operate below this ceiling by creating new line
geometry (for example the separately developed axis-mode/span family), then use
fresh case/family-held-out calibration rather than tuning another selector on
these same 28 cases.

# F1 V23–V26 strict-LOFO optimization cycle

Date: 2026-08-14

## Scope and scientific contract

This cycle evaluates the scan-observable bend target defined in
`data/bend_corpus/annotations/scan_observable_bend_target_v3.json`. Inference is
source-only: no CAD, gold bend lines, expected bend counts, filenames, part
identity, good/bad labels, or benchmark feedback are available to the selector.
The learned bounded-complex route is retrained with the held-out part family
removed. Deterministic V16 geometry routes remain frozen.

The threshold and estimator variants below were explored against the same
aggregate leave-one-family-out corpus. They are therefore development evidence,
not an unbiased final validation result. A genuinely prospective scan or a new
locked family holdout is still required before promotion.

## Results

| Tranche | Change | TP | FP | FN | Precision | Recall | F1 | Exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V22 | ExtraTrees, threshold 0.45 | 89 | 100 | 87 | 0.470899 | 0.505682 | 0.487671 | 3/35 |
| V23 | ExtraTrees, threshold 0.80 | 79 | 64 | 97 | 0.552448 | 0.448864 | 0.495298 | 3/35 |
| V24 | V23 + frozen short-mesh recovery | 79 | 69 | 97 | 0.533784 | 0.448864 | 0.487654 | 3/35 |
| V25 | V23 + source-balanced representative snap | 79 | 64 | 97 | 0.552448 | 0.448864 | 0.495298 | 3/35 |
| V26 | family/class-balanced ExtraTrees, threshold 0.80 | 79 | 55 | 97 | 0.589552 | 0.448864 | **0.509677** | 3/35 |

Candidate coverage is 151/176 gold axes (85.8%). Of 87 V22 misses, 62
already have a matching candidate and 25 do not. The dominant limitation is
therefore selection/representative ownership, although candidate birth remains
a material secondary ceiling.

## Retain and retire decisions

### V23: retain as a precision operating point, not a final model

Raising the threshold from 0.45 to 0.80 removed many obvious diagonal and
parallel hallucinations, but it also removed legitimate tray and edge bends.
It improved aggregate F1 by 0.007627 while losing ten true detections. This is a
useful precision setting, not a complete generalization solution.

### V24: terminally retire

The exact frozen short-mesh primitive added five predictions and every addition
was false positive. It recovered zero true bends. No threshold rescue or rerun
is justified.

### V25: retire as net-neutral

The source-balanced geometric-median snap moved 20 predictions. It helped C003,
C010, and C037 but harmed C012, C014, and C021, producing exactly the same
aggregate confusion counts as V23. There is no evidence to fund this primitive.

### V26: retain as the leading research estimator

V26 gives every held-in part family equal total training weight and divides each
family's mass equally between positive and negative candidate rows. This aligns
training more closely with unseen-family evaluation and prevents large candidate
pools from dominating the learner. It retained all 79 V23 true positives while
removing nine false positives, improving F1 by 0.014379 over V23 and 0.022006
over V22.

This gain is not globally safe. Minimum family F1 remains 0.0, exact-count
accuracy remains 3/35, and recall remains 0.448864. V26 is retained only for
prospective research execution; it is not authorized for runtime/main or
production.

## Native-resolution visual audit of V26 changes

Every changed case was inspected from the native audit renders:
`output/research/f1-new-parts-20260814/v26_t080_family_class_balanced_lofo/native_audits/`.

- Clear improvements: C008 removed a false line; C010 recovered a real long
  fold while removing a false one; C011 and C012 recovered real long edges;
  C024 removed two redundant parallel lines; C025 recovered a short cross bend;
  C028 recovered a true axis; C032 removed five false positives at the cost of
  one true bend.
- Mixed or immaterial: C001 improved precision but still missed many real bends;
  C014 changed geometry without a material visual improvement.
- Regressions: C002 gained a phantom interior plane; C003 gained an offset
  parallel false line; C021 gained a false short tab line; C037 became more
  under-detected and gained a false positive.
- Critical regression: C031 fell from 4 TP / 4 FP / 1 FN to 1 TP / 2 FP / 4 FN.
  The render shows severe under-detection and loss of three legitimate bends.

The visual verdict matches the quantitative one: family balancing is a useful
precision correction, but it cannot be promoted without recovering recall and
eliminating family-level collapse.

## Next evidence

1. Fit a full V9-compatible V26 research bundle using equal-family/equal-class
   sample weights and the exploratory 0.80 operating threshold.
2. Run it prospectively on genuinely new source scans without labels or identity
   features.
3. Render and manually inspect every native-resolution overlay before deciding
   whether the behavior is worth funding.
4. Keep the candidate-local ultrasonic B-scan lane separate. Its acquisition
   contract is ready, but no ultrasonic observations exist yet.


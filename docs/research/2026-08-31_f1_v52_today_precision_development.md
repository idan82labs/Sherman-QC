# F1 V52 development precision ownership — 2026-08-30 scans

## Ruling

`F1_V52_DEVELOPMENT_PRECISION_OWNERSHIP_V1` is a deterministic, source-only
post-filter over the frozen V51 result and cached source-derived candidate pool.
It is a useful in-sample precision experiment, but it is **not** promotion or
generalization evidence.  The same 28 AI-adjudicated cases were used to choose
the gates and measure the score.

V51 and its baseline artifacts were not modified.

## Development result

| Metric | Frozen V51 | V52 precision layer | Delta |
|---|---:|---:|---:|
| Predictions | 77 | 14 | -63 |
| True positives | 14 | 14 | 0 |
| False positives | 63 | 0 | -63 |
| False negatives | 51 | 51 | 0 |
| Precision | 0.181818 | 1.000000 | +0.818182 |
| Recall | 0.215385 | 0.215385 | 0.000000 |
| F1 | 0.197183 | 0.354430 | +0.157247 |
| Exact cases | 2/28 | 4/28 | +2 |

The observed F1 lift is 79.7% relative.  This is entirely a precision gain:
all 14 existing V51 true positives were retained and all 63 existing false
positives were vetoed.  The layer does not recover any of the 51 false
negatives.  T020 falls from 13 predictions to zero; all 13 were false positives
under the development labels.

Retained predictions by evidence family:

- plane lattice: 6
- cylindrical standard: 3
- scan-mesh adjacency: 2
- cross-source consensus midpoint: 2
- fragment-tolerant plane sidecar: 1

## Method

The runtime reads no labels, expected bend counts, CAD, drawing, filename-derived
part identity, or sender-reported counts.  It uses explicit family-specific
quality gates over line length, local normal structure, two-sided transition
support, line residual, and (for scan adjacency) surface gap.  Unsupported or
missing evidence fails closed.  A deterministic high-score owner also suppresses
near-exact cross-family aliases after gating.

The output declares both facts that matter:

- `labels_used: false` — labels are not runtime inputs.
- `development_tuning_labels_used: true` — the thresholds were selected using
  this cohort's AI-adjudicated labels.

## Determinism and verification

- 18 synthetic/source-free unit tests pass.
- The cached V51 primary run and cached V51 replay each produced identical V52
  case payloads for all 28 cases and identical summary bytes.
- A second execution over the primary cache also produced identical case
  payloads and summary bytes.
- The evaluator completes from cached artifacts in about two seconds; it does
  not rerun mesh processing.

## Quantified resubstitution risk

| Quantity | Value |
|---|---:|
| Calibration cases | 28 |
| Evaluation cases | 28 |
| Calibration/evaluation overlap | 28/28 (100%) |
| Independent holdout cases | 0 |
| Prediction outcomes inspected during tuning | 77 |
| Numeric gate cutoffs | 23 |
| Outcomes per numeric cutoff | 3.35 |
| Observed F1 lift attributable to resubstitution | 100% |
| Held-out verified F1 lift | unknown |

The observed retained precision is 14/14.  If this gate had been preregistered,
the two-sided 95% exact binomial interval would still be only approximately
0.768–1.000.  That interval is **not valid here** because those same 14 successes
and 63 failures informed threshold selection.  The effective uncertainty is
therefore larger and cannot be estimated honestly without a new sealed cohort.

The correct next test is to freeze this file and its hashes, then run it once on
new scans whose labels are unavailable until after inference.  Until that test,
the 1.000 precision and 0.354430 F1 are an in-sample development upper bound.

## Artifacts

- implementation: `backend/f1_post_v51_precision_ownership_v52_development.py`
- cached evaluator: `scripts/evaluate_f1_v52_today_precision_cached_development.py`
- tests: `tests/test_f1_post_v51_precision_ownership_v52_development.py`
- primary summary: `output/research/f1-v52-today-precision-development-20260831-r2/summary.json`
- primary score: `output/research/f1-v52-today-precision-development-20260831-r2/today_accuracy_v52.json`
- V51-replay summary: `output/research/f1-v52-today-precision-development-20260831-r2-v51-replay/summary.json`


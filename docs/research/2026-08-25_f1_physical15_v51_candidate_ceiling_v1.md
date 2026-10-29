# Sherman QC physical15 V51 candidate ceiling V1

Date: 2026-08-25

## Result

The frozen V51 candidate pools were scored after execution against the 12
development-consumed, count-backed physical annotations. This diagnostic does
not change the model and is not promotion evidence.

| Quantity | Result |
|---|---:|
| Physical cases | 12 |
| Gold bends | 59 |
| Pool line objects | 5,787 |
| Candidate-ceiling true positives | 50 |
| Candidate-ceiling recall | 0.8475 |
| V51 selected true positives | 35 |
| V51 selected recall | 0.5932 |
| Recall lost in selection | 15 bends |
| Recall lost in candidate birth | 9 bends |
| Cases with complete candidate birth | 8/12 |
| Cases with zero pool recall | 0/12 |

The dominant aggregate bottleneck is selection/ownership, but the remaining
candidate-birth gap is real and concentrated: D012 lacks five of seven gold
bends; D007 lacks two; D001 and D009 each lack one. The selection deficit is
concentrated in D003, D006, D010 and D011.

Among the 50 ceiling matches, the representative source counts are:

- scan-mesh adjacency: 25;
- cylindrical standard: 14;
- plane lattice: 7;
- cylindrical wide: 2;
- bend hypothesis: 1;
- strict blind owned: 1.

## Interpretation

Another unrestricted candidate generator is not the highest-value aggregate
attack. The pool already contains 84.7% of physical truth, but only 0.86% of
its 5,787 lines can be assigned one-to-one to a gold bend. A successful
selector must therefore use relational ownership and aggressive deterministic
duplicate/alias control; a scalar threshold over the existing selected lines
has already failed.

The next funded diagnostic is a nested leave-one-family-out selector over the
complete pool with relational features to the frozen V51 incumbent set. It may
use consumed truth only inside training folds; every outer physical family is
excluded from model fitting and threshold selection.

## Generalization Progress Indicator

**GPI-C — positive diagnostic clarity, no accuracy or generalization claim.**

Evidence:
`output/research/autoresearch-loop/PHYSICAL15_V51_CANDIDATE_CEILING_V1.json`

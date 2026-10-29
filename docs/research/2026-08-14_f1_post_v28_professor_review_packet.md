# GPT-5.6 Pro packet — post-V27/V28 bounded action review

Date: 2026-08-14

## Decision requested

Act as a skeptical professor reviewing an industrial 3D scan bend-inspection
research program. Ground the ruling in relevant published research where it
exists, clearly separate established methods from Sherman-specific hypotheses,
and cite primary sources or authoritative technical literature.

Select **exactly one** next action:

1. retain V27 only and prescribe its exact locked promotion/falsification gate;
2. authorize exactly one frozen prospective V28 paired-return validation and
   prescribe its exact input, precommit, visual, and stop gates;
3. reject both V27 and V28 as incompatible with the prior causal STOP and name
   the smallest scientifically necessary capability opening;
4. choose a different single bounded action only if the evidence below makes
   all three choices scientifically invalid.

Do not propose a threshold sweep, post-hoc rescue, part-specific rule, identity
feature, expected-count feature, gold-assisted inference, or broad architecture
search. If no bounded action is defensible, issue a formal STOP.

## Program target and immutable constraints

The long-term target is trustworthy zero-shot bend inspection for daily known
and near-neighbor sheet-metal workflows. Promotion ultimately requires at
least 30 scans / 15 families, at least 90% exact-count accuracy, precision and
recall at least 90%, marker admissibility at least 99%, deterministic replay,
peak RSS below 16 GiB, P95 latency at most 300 s, job success at least 99.5%,
and no major family below 80% recall.

All current work remains research-only. Inference must remain scan-only and
deterministic on CPU with approximately 200,000 detector points, compact caches
without source-point serialization, no CAD, labels, expected count, filename,
part identity, good/bad status, benchmark feedback, or part-specific tuning.
Runtime/main and production remain closed.

## Prior authoritative STOP

The prior GPT-5.6 Sol Pro ruling established
`SCAN_SURFACE_CAUSAL_NONIDENTIFIABILITY_STOP_V21`: a monolithic bend and a
ground full-penetration joint can have byte-identical permitted surface
observations while the frozen semantic target assigns different labels. The
executed `SOURCE_OBSERVATION_EQUIVALENCE_WITNESS_V1` confirmed the contradiction.

That ruling correctly prevents a universal claim that another surface-only
architecture can identify material provenance. The new user authorization was
explicitly narrower: run source-free tests quickly and learn through action.
V27 and V28 were therefore executed as research postprocessors, not promoted
semantic classifiers. Your review must decide whether that narrower scope is
scientifically legitimate or still violates the STOP.

## Two evaluation views that must not be conflated

The retained V16 full-training research baseline previously scored 132 TP / 60
FP / 44 FN, precision 0.6875, recall 0.7500, F1 0.717391, exact count 4/35.
Those numbers are not family-held-out generalization evidence.

The exploratory strict leave-one-family-out learned-router protocol retrains
with the held-out part family removed. Its V22 baseline is:

| Tranche | TP | FP | FN | Precision | Recall | F1 | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| V22 strict LOFO | 89 | 100 | 87 | 0.470899 | 0.505682 | 0.487671 | 3/35 |
| V27 strict LOFO | 89 | 97 | 87 | 0.478495 | 0.505682 | 0.491713 | 3/35 |
| V28 strict LOFO | 89 | 97 | 87 | 0.478495 | 0.505682 | 0.491713 | 3/35 |

Candidate coverage is 151/176 gold axes. The learned-router metrics remain far
below promotion thresholds.

## V27 — collinear selected-fragment stitching

V27 operates only after selection. It cannot add an unselected candidate. It
builds connected components among selected lines when:

- axis angle is at most 2 degrees;
- transverse offset is at most 6 mm;
- axial gap is at most 20 mm;
- the PCA-fitted union span is at least 1.25 times the longest member.

A qualifying component is replaced with one fitted physical axis. Across the
35 strict-LOFO cases it triggered only on C021 and C024, removed respectively
one and two false positives, and lost no true positives. This is a physical-line
instance reconciliation claim, not a bend-versus-joint semantic claim.

Implementation SHA-256:

- `backend/f1_collinear_fragment_stitch.py`:
  `d67db9e952d8e127cebe4c8117fe3d604830a83a084d73c6aeeb8ec812015527`
- V27 report:
  `d910716ddf325ae5641d3c490dd7cb850a72a044bb0a5cbc7ac0840fecd8b19e`

## V28 — paired-return counterpart completion

V28 can add at most one source candidate. It applies only to the bounded complex
route after V27 and requires exactly three selected physical lanes:

- exactly one close parallel pair, separated by 1.5–8% of bbox span with at
  least 0.75 shorter-span overlap;
- the third lane is parallel and at least 30% of bbox span away;
- candidate counterpart fragments have support at least 0.80, normal turn at
  least 70 degrees, transition score at least 0.25, span at least 4% bbox;
- their separation from the far reference is 0.70–1.50 times the close-pair
  separation;
- stitched union span must materially exceed each member and reach at least 60%
  of the far reference span;
- add at most one deterministic best candidate.

V28 triggered zero times in the 35 strict-LOFO cases. It therefore introduced
no measured historical regression but has no historical positive validation.

Implementation SHA-256:

- `backend/f1_paired_return_completion.py`:
  `0f0408bb5040d11b3cd6180a4ab46224d720f2a82ff431b6f6b6eff1b06cc9af`
- V28 report:
  `3efc7326378ce9b01bef70ab46e33cd3aa97759cb5aed158c8449f8af70dbb1b`

## New Aug13 batch and mandatory visual audit

Four opaque source scans were processed without identity/count/CAD/gold:

| Output | U01 | U02 | U03 | U04 |
|---|---:|---:|---:|---:|
| frozen retained V16 | 5 | 4 | 3 | 6 |
| V27 | 5 | 4 | 3 | 3 |
| V28 | 5 | 4 | 3 | 4 |

U01–U03 were prior regression replays and not independent new evidence. Their
native-resolution artifacts were visually acceptable: five supported
web/fork/closure folds, four tray folds, and three distinct return curvature
zones, respectively.

U04 was the new mechanism case. V16 selected four collinear fragments of one
right base-to-wall fold, creating three instance-level false positives, and
missed the adjacent right wall-to-return fold. V27 replaced the four fragments
with one 499.414 mm physical axis. V28 stitched three unused high-support source
fragments into a 399.821 mm counterpart axis.

The final native overlay and cross-section were manually inspected at original
resolution. They show two folds on each long edge—base-to-wall and
wall-to-return—with no visible hole, perimeter, or broad-face alias. Only after
the output was frozen, an evaluator-only fabrication drawing was opened. Its
Detail G independently shows the same two-fold return profile on each side,
four bends total.

This is drawing-confirmed post-hoc mechanism evidence, **not** an unbiased blind
F1 result, because V28 was designed after U04's failure mode was inspected.

Visual adjudication SHA-256:
`09fd9896f3fe1a60ece9f67b9a68cd9018a92856b620e8435a5cb845039c6a15`

## Immediate specificity sweep

The unchanged V28 selector was then run with 12 workers across 26 other recent
valid source candidate pools: 21 full-resolution routes, two bounded small-pool
routes, and three bounded-complex routes. Two complex cases entered with
exactly three predictions. V27 triggered zero times and V28 triggered zero
times.

This is a zero-opportunistic-trigger audit, not positive validation. Summary
SHA-256:
`56e28f75e5325f14eab80ba697afaebcc6b2da20622c07f482e9f6d2bf7c777d`

## Reliability and implementation evidence

- focused tests: 15 passed;
- all changed modules compile;
- every Aug13 output explicitly records `source_only=true`,
  `expected_count_used=false`, and `part_identity_used=false`;
- no runtime/main or production code path changed;
- no V28 threshold or rerun rescue was performed;
- native-resolution visual artifacts were manually reviewed for all four scans.

## Questions the ruling must answer

1. Does V27 legitimately improve physical line-instance accounting without
   contradicting the causal nonidentifiability STOP? If yes, state the exact
   promotion boundary and tests that would make the claim defensible.
2. Is one frozen prospective V28 run scientifically warranted, given one
   post-hoc positive, zero historical triggers, and zero triggers in 26 recent
   pools? If yes, specify the exact positive/negative fixtures, precommit,
   visual adjudication, and terminal failure conditions.
3. What published work supports or contradicts collinear fragment merging and
   paired structural completion in noisy point-cloud/mesh line extraction?
4. What claim language avoids implying that either primitive solves the
   bend-versus-ground-joint ambiguity?
5. Select exactly one next action. No menu may remain at the end.

## Required response format

1. **RULING** — one exact action or formal STOP.
2. **RESEARCH BASIS** — primary/authoritative citations and what is established
   versus Sherman-specific.
3. **CAUSAL SCOPE** — precisely what the action may and may not claim.
4. **EXACT SPECIFICATION** — frozen predicates, deterministic ties, NMS,
   memory/latency/cache constraints, prohibited information.
5. **FAST FALSIFICATION GATES** — positive, negative, metamorphic, replay, and
   native-resolution visual gates.
6. **REAL-SCAN GATE** — one bounded next execution or an explicit prohibition.
7. **RETAIN/RETIRE RULE** — terminal decision with no rescue tuning.


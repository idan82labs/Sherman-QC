# Sherman QC F1 typed-union end-to-end checkpoint — 2026-08-05

## Disposition

The professor-guided typed-union architecture is implemented and reproducible
as an offline research lane. It is **not** accurate enough for runtime
promotion.

The current authoritative 13-scan result is **1/13 exact**, not the requested
10/13. The full Gold-27/37-scan result is **1/37 exact**.

Runtime/main integration remains prohibited.

Against the active daily-workflow generalization target, 9 of 14 research
gates pass. Corpus size, family count, determinism, memory, latency, job
success, grouped-split integrity, the 200,000-point bound, and the compact-cache
contract pass. Five semantic gates fail: exact-count accuracy (0.027027 versus
0.90), line precision (0.56 versus 0.90), line recall (0.230769 versus 0.90),
reviewed marker accuracy (0.56 versus 0.99), and minimum family recall (0.0
versus 0.80). The machine-readable checkpoint is
`output/research/f1-gold27-20260803/structured_union_exact_v1/generalization_goal_readiness.json`.
All eleven A0 evidence artifacts are frozen by SHA-256 in
`output/research/f1-gold27-20260803/structured_union_exact_v1/a0_evidence_manifest.json`
under disposition `FALSIFIED_REPRESENTATION_NOT_PROMOTABLE`; repeated manifest
generation is byte-identical.

## Implemented contracts

- one global deterministic working cloud capped at 200,000 points;
- no full source-point serialization in candidate pools or research caches;
- compact atom, edge, and plane tables only;
- source-independent geometric path identity;
- explicit `PLANE:`, `FLANGE:`, and `MESH_SURFACE:` namespaces;
- verified `INCIDENCE_V1` completion and unsupported-path prohibition;
- all seven semantic labels active through eligibility, not factor membership;
- normalized atom-mass ownership and two-sided bend minima;
- hard duplicate and panel-boundary ownership constraints;
- certified, endpoint-aware, acyclic continuation proposals;
- disabled generic topology hypotheses retained as audit-only evidence;
- exact deterministic CP-SAT stages with proved lexicographic ties;
- proved second-best semantic assignment;
- per-atom owner output and bounded `BIRTH_V1` integration;
- research-only end-to-end and measured-parallel batch runners.

The exact professor response used as the implementation source is
`docs/research/PROFESSOR_F1_STRUCTURED_FOLLOWUP_RESPONSE_20260804.md`.

## Operational evidence

| Check | Result |
| --- | ---: |
| Bounded candidate-pool cases | 37 / 37 |
| Maximum working-cloud points | 200,000 |
| Source-point serialization | 0 cases |
| Mesh lanes disabled before read when declared vertices exceed cap | 22 cases |
| Exact solver success | 37 / 37 |
| Unsupported non-residual selections | 0 |
| Exact deterministic replay | 10 / 10 byte-identical |
| Exact replay P95 latency on C004 | 14.779444 s |
| Batch workers after measured probe | 12 |
| Batch wall time | 1,151.03 s |
| Maximum reported child solver RSS | 1,662,664,704 bytes |
| Grouped split integrity | PASS: 37 scans / 27 families / 5 folds |
| Focused Python tests | 64 passed |
| Isolated CP-SAT tests | 14 passed |

The candidate-pool probe on C001 used 2,486,484,992 bytes peak RSS. The exact
solver probe used 1,359,134,720 bytes. Both are below the 16 GiB per-job target.

The refreshed enterprise gate remains blocked: 5 of 16 gates pass and 11 are
blocked. Operational determinism, latency, and job-success gates pass; truth
scale and semantic accuracy gates do not.

The evaluation-only split audit confirms that all three frozen manifests have
identical case sets and corpus IDs; every part family, byte-identical scan, and
byte-identical label artifact is confined to one outer fold; and the declared
locked/development memberships are exact complements. No split metadata is
available to inference. The audit does not yet prove that differently numbered
near-neighbor geometries are independent; that requires the preregistered
geometry-family grouping requested from the professor.

Future A1/A2 scores are compared to A0 with deterministic paired resampling of
the 27 part families (`PAIRED_FAMILY_BOOTSTRAP_V1`, 10,000 resamples). The
comparator fails closed if corpus IDs, family sets, case-token sets, or scan
counts differ and reports per-family improvement/tie/regression counts plus 95%
intervals for precision, recall, line F1, and exact-count accuracy. A0 versus
itself returns byte-stable zero deltas for all 27 families.

## Frozen line-level result

| Metric | Gold-37 | Authoritative 13 |
| --- | ---: | ---: |
| Gold bends | 182 | 71 |
| Predictions | 75 | 32 |
| True positives | 42 | 15 |
| False positives | 33 | 17 |
| False negatives | 140 | 56 |
| Precision | 0.560 | 0.469 |
| Recall | 0.231 | 0.211 |
| Line F1 | 0.327 | 0.291 |
| Exact scans | 1 / 37 | 1 / 13 |

Equal-weight part-family metrics are stricter than the global totals:

```text
family-macro precision:       0.482804
family-macro recall:          0.241230
family-macro line F1:         0.307605
family-macro exact accuracy:  0.037037
minimum family recall:        0.000000
families below 0.80 recall:   26 / 27
families with zero recall:    8 / 27
```

The zero-recall families are `48333006`, `48340007`, `49125000`,
`49191023`, `49191030`, `49191037`, `49191054`, and `50556000`. These values
are now emitted directly by the version-2 Gold37 scorer so future aggregate
gains cannot hide a collapsed family.

Every zero-recall family also has zero BIRTH_V1 components reaching the
six-atom minimum. This is not unique to those families—26 of 27 families have
zero size-qualified birth components—so the birth failure is a corpus-wide
representation collapse rather than a threshold that only misses one geometry
family.

On the authoritative user-verified tier alone (13 scans / 10 part families),
family-macro precision is 0.515238, recall is 0.291587, line F1 is 0.358199,
and exact-count accuracy is 0.10. Nine of ten authoritative families remain
below 0.80 recall and one has zero recall. The scorer reports the same family
breakdown independently for each lower-confidence truth tier so development
labels cannot mask the authoritative result.

The single exact case is C004 / part family 15651004.

## Representation versus selection

All incidence-complete, bend-eligible paths cover 112 of 182 gold bends:

```text
supported candidate recall: 61.5%
supported candidate precision: 5.1%
candidate-representation false negatives: 70
represented gold bends selected by MAP: 42 / 112
```

Therefore the 140 final false negatives split into two material problems:

1. 70 gold objects have no admissible compact candidate at all;
2. 70 gold objects are represented but lose ownership/MDL selection.

## Professor ablation results

### BIRTH_V1 precheck

```text
cases with negative-lower-bound birth columns: 0 / 37
eligible birth columns: 0
```

This meets the professor's falsification condition: the current residual atom
and plane representation does not contain enough object-forming information
for the first straight two-sided birth rule. No second exact batch was run,
because it would be identical.

The subsequent truth-free `BIRTH_V1_STAGE_CENSUS_V1` diagnostic localizes the
collapse before duplicate suppression or the MDL lower bound:

```text
plane pairs:                 938 total / 649 angularly eligible
atom/plane-pair trials:      861,688
after transverse gating:     8,571
after normal gating:         2,960
after atom-axis gating:      390
connected components:        312
components with >= 6 atoms:  1
components with bin coverage: 0
```

Of the 312 components, 311 fail the six-atom minimum. The sole surviving
component reaches the axial-span stage and then fails occupied-bin coverage.
No component reaches two-side overlap, TLS fitting, duplicate suppression, or
MDL evaluation. This rules out object-cost tuning as a justified repair and
points to sparse/disconnected transition support plus the atom-axis gate as
the immediate representation blockers. A full 37-case replay is byte-identical
(`8766dc22d065b1853ee08380fd467805bc1074c54af00859c7113766d22684c6`),
takes 9.37 seconds, and peaks at 188,973,056 bytes RSS.

### Two-sided ownership minimum screen

This is a conservative post-solve screen; exact re-optimization may choose
alternatives, but a dominated screen is sufficient to avoid an unnecessary
19-minute exact batch.

| Minimum bins per side | TP | FP | FN | Precision | Recall | Line F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42 | 33 | 140 | 0.560 | 0.231 | 0.327 |
| 2 | 38 | 29 | 144 | 0.567 | 0.209 | 0.305 |
| 3 | 29 | 18 | 153 | 0.617 | 0.159 | 0.253 |
| 4 | 21 | 12 | 161 | 0.636 | 0.115 | 0.195 |
| 6 | 14 | 6 | 168 | 0.700 | 0.077 | 0.139 |

No setting improves line F1. At three bins, false positives fall by 45.5%,
short of the professor's 50% GO threshold, while recall loses more than the
allowed five percentage points. The side representation therefore reaches the
professor's STOP condition.

## Manual visual audit

All 37 six-view sheets were inspected. The scan/gold/F1 coordinate alignment
is correct. No gross floating marker or hole-centered hallucination was found.
False markers remain scan-supported but semantically wrong.

Legend: matched gold green; missed gold orange; matched F1 cyan dashed; false
F1 red dashed.

| Case | TP/FP/FN | Hard visual critique |
| --- | ---: | --- |
| C001 | 1/4/7 | Red dashed lines follow tray/frame outlines; most parallel fold lanes are absent. |
| C002 | 1/0/3 | One long boundary-aligned fold is correct; three enclosing fold edges are suppressed. |
| C003 | 0/1/6 | The only output is an exterior boundary alias; every physical frame bend is missed. |
| C004 | 1/0/0 | Exact finite line, correct surface support and extent. |
| C005 | 4/2/2 | Strong triangular folds are recovered; two long diagonal/perimeter extensions are aliases and two short bends are missed. |
| C006 | 0/0/3 | Complete suppression despite three clearly visible fold transitions. |
| C007 | 1/0/2 | Main long fold recovered; both short return bends are missed. |
| C008 | 1/1/3 | One long fold is correct; a short edge alias remains and three sides are missing. |
| C009 | 1/0/5 | Only one of several long parallel bends survives; repeated-lane representation is inadequate. |
| C010 | 1/4/7 | Multiple red exterior/frame aliases survive while nearly all internal/parallel bends are absent. |
| C011 | 5/0/5 | Precision is good; five internal/short folds are suppressed. |
| C012 | 4/0/6 | Four long structural folds align; six interior/end folds are missing. |
| C013 | 0/1/4 | Bottom/perimeter alias selected; all four intended folds missed. |
| C014 | 2/2/2 | Two correct long folds plus two exterior side aliases; two returns missed. |
| C015 | 3/2/4 | Exterior frame dominates: three correct sides, two opposite-edge aliases, four missed folds. |
| C016 | 1/1/4 | Main long fold aligns; one short outline alias and four missed folds remain. |
| C017 | 0/0/2 | Both long bends absent on a rounded/occluded geometry family. |
| C018 | 1/0/1 | One parallel fold is exact; its sibling is missing. |
| C019 | 0/0/4 | No countable object survives despite four visible annotated folds. |
| C020 | 1/0/2 | One long transition aligns; two end/parallel folds are missing. |
| C021 | 1/2/4 | Main base fold aligns; short end edges are falsely counted and four folds are missed. |
| C022 | 1/0/2 | One short bracket transition aligns; two nearby parallel bends are missing. |
| C023 | 0/0/2 | Straight-line ontology produces no object on a strongly curved/rolled part. |
| C024 | 0/0/8 | Dense long-arm part has zero selected bends; representation/ownership failure, not rendering. |
| C025 | 0/2/7 | Two short end/edge aliases selected; all seven intended bends missed. |
| C026 | 1/0/2 | One long bend recovered; two shorter returns missed. |
| C027 | 0/1/5 | One diagonal/perimeter alias selected; all intended bends missed. |
| C028 | 4/0/4 | Four supported bends align cleanly; four additional folds have no selected object. |
| C029 | 0/0/6 | Entire six-bend arrangement suppressed. |
| C030 | 2/0/3 | Two adjacent long folds align; three small/return bends are missing. |
| C031 | 0/0/5 | Entire five-bend box arrangement suppressed. |
| C032 | 1/3/4 | One fold aligns; three red interior/perimeter aliases and four misses remain. |
| C033 | 0/3/3 | Every prediction traces the outer rectangular frame; all three gold folds are missed. |
| C034 | 1/2/2 | One fold aligns; two exterior outline aliases and two misses remain. |
| C035 | 0/0/1 | Single localized bracket bend missed completely. |
| C036 | 0/0/1 | Single localized bracket bend missed completely. |
| C037 | 3/2/9 | Three major folds align, two exterior/base aliases survive, and nine complex internal/return bends are absent. |

## Next justified architecture step

Do not tune a global span threshold or part-specific count. The evidence says:

1. atomization must retain denser transition-side strips and panel-side modes,
   because BIRTH_V1 has no eligible residual columns and 38.5% of gold objects
   lack an admissible path;
2. non-counted perimeter/corner objects need human-labeled ownership examples,
   because the current one-/two-sided representation cannot separate them
   without unacceptable recall loss;
3. after those labels exist, fit only the professor-approved low-capacity,
   sign-constrained potentials under nested leave-one-family-out validation;
4. preserve this 37-case result as the frozen A0 baseline and never count the
   designed 457 regression as generalization evidence.

The existing compact caps leave substantial representation headroom. Across
Gold37 the largest V1 table contains 1,918 atoms and 14,184 edges. Even a
conservative two-child expansion of every atom would produce at most 3,836
atoms, and a four-way expansion of every existing edge would produce at most
56,736 edges—well below the existing 20,000-atom and 200,000-edge caps. This
does not justify unconditional splitting, but it means `TRANSITION_ATOM_V2`
can preserve separate local normal modes without increasing the serialized
schema limits.

## Evidence artifacts

- `output/research/f1-gold27-20260803/candidate_pool_bounded_v2/manifest.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/batch_summary.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/gold27_score.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/candidate_recall_score.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/birth_v1_precheck.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/side_minimum_posthoc_screen.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/visual_review/review.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/determinism_C004/determinism_summary.json`
- `output/research/f1-gold27-20260803/structured_union_exact_v1/enterprise_promotion_gates_20260805.json`

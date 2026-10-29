# Patch-Graph F1 Next Blocker Handoff

Date: 2026-04-24

## Context

We now have an offline F1 patch-graph lane with:

- scan-family routing,
- fixed atom graph / fixed proposal pools,
- typed flange/bend/residual/outlier ownership,
- latent candidate flange-pair attachment,
- interface-birth hypotheses,
- control-constrained acceptance rules,
- corrected exact/MAP reporting semantics.

Runtime zero-shot is still the production/control path. F1 is not promoted to runtime.

The current F1 lane is useful as a routed research candidate layer, but raw F1 is not yet reliable enough to replace control. The remaining work is no longer "find bends in point clouds" in general. It is model selection between competing typed decompositions, especially on repeated near-90 full scans.

## Corrected Reporting Semantics

We corrected a reporting issue:

- `map_bend_count` is the count of the MAP decomposition.
- `exact_bend_count` is only populated when the F1 decomposition range is singleton.
- A non-singleton range like `[11,12]` with MAP 11 is no longer reported as exact 11.

This matters because the earlier dashboard overstated raw F1 determinism.

## Current Validation Summary

Main validation summary:

`data/bend_corpus/evaluation/patch_graph_f1_mixed_intersection_validation_20260424/VALIDATION_SUMMARY.md`

iCloud copy:

`/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Patch Graph Renders 2026-04-24/MIXED_INTERSECTION_VALIDATION_SUMMARY.md`

Corrected focus run:

`data/bend_corpus/evaluation/patch_graph_f1_focus_exact_semantics_apply_20260424`

Focus aggregate:

| Metric | Value |
| --- | ---: |
| Scans processed | 6 |
| Candidate exact accuracy | 1.000 on 3 exact-evaluated cases |
| Candidate range contains truth | 1.000 |
| Tightened cases | 3 |
| Broadened cases | 0 |
| Raw F1 exact accuracy | 0.250 on 4 singleton raw claims |
| Raw F1 range contains truth | 0.500 |

Key focus cases:

| Scan | Expected | Control | Accepted | Raw F1 |
| --- | ---: | --- | --- | --- |
| `10839000` | 10 | `[9,10]` | exact 10 / `[9,10]` | exact 9 / `[9,9]` |
| `47959001` | 6 | `[6,6]` | exact 6 / `[6,6]` | exact 1 / `[1,1]` |
| `ysherman` | 12 | `[10,12]` | `[11,12]` | exact 11 / `[11,11]` |
| `49204020_04` | 12 | `[11,12]` | `[11,12]` | MAP 11 / `[11,12]` |
| `48963008` | 3 | `[3,4]` | exact 3 / `[3,3]` | MAP 3 / `[2,3]` |
| `48991006` | 9 | `[8,12]` | `[9,12]` | exact 9 / `[9,9]` |

Protected slice status:

- Development: truth containment 1.000, tightened 2, broadened 0.
- Holdout: truth containment 1.000, tightened 1, broadened 0.
- Known-count PLY audit: exact accuracy 1.000 on 5 exact-evaluated cases, tightened 4, broadened 0.
- Repeated-near-90 supplemental: truth containment 1.000, tightened 1, broadened 0.
- Small-tight supplemental: truth containment 1.000, tightened 2, broadened 0.
- Partial/total-bends negative control: 10/10 routed `partial_scan_limited`, 10/10 `control_guard`, 0 exact-count evaluated.

## Remaining Technical Blocker

The main blocker is not safety of the accepted routed layer. The accepted layer is safe because it is still control-constrained.

The blocker is raw F1 model selection:

- `49204020_04`: raw F1 has MAP 11 but range `[11,12]`; expected observed count is 12.
- `ysherman`: raw F1 gives exact 11; expected observed count is 12.
- Stable control anchors still fail raw F1 badly:
  - `10839000`: raw exact 9 while control exact/range is 10 / `[9,10]`.
  - `47959001`: raw exact 1 while control exact/range is 6 / `[6,6]`.

This means F1 is useful as a routed research lane, but the latent decomposition objective is not yet strong enough to stand alone.

## 49204020_04 Diagnostic

Current result:

`data/bend_corpus/evaluation/patch_graph_f1_focus_exact_semantics_apply_20260424/49204020_04/decomposition_result.json`

Replacement-diagnostic rerun:

`data/bend_corpus/evaluation/patch_graph_f1_49204020_04_replacement_diag_20260424`

Pairwise neighborhood diagnostic rerun:

`data/bend_corpus/evaluation/patch_graph_f1_49204020_04_pairwise_diag_20260424`

Frozen local re-label diagnostic rerun:

`data/bend_corpus/evaluation/patch_graph_f1_49204020_04_local_relabel_diag_20260424`

Candidate solution counts:

```text
[11, 12, 12]
```

Candidate solution energies:

```text
[1728.87913, 1740.586452, 1740.707183]
```

Interpretation:

- The correct-count 12 decompositions exist in the current search family.
- They are about 12 energy units worse than the 11-count MAP.
- Therefore, the missing ingredient is not merely proposal generation. It is decomposition scoring / repair move model selection.

The 11-count MAP active labels:

```text
B1
B12
B13
B14
B15
B7
B8
B9
BIRTH_F1_F3_7
BIRTH_F3_F58_1
BIRTH_F44_F6_2
```

The 12-count alternatives add or retain labels such as:

```text
B4
B11
```

But those labels have low decomposition confidence around `0.249`, while the MAP substitutes/keeps the `F1-F3` interface birth with much higher confidence.

So the question is not "can we force 12?" The question is:

> What object-level term would make the physically correct 12-object decomposition score below the 11-object MAP without allowing the failed over-birth behavior?

### Local Replacement Diagnostic

We added a diagnostics-only local move evaluator. It does not change labels or counts. It scores:

- replacing one active bend component with one inactive bend candidate,
- adding one inactive candidate support on currently non-bend atoms.

For `49204020_04`:

```text
current_energy: 1728.87913
active_bend_component_count: 11
inactive_bend_label_count: 11
replacement_candidates: 10
add_candidates: 15
improving_replacements: 0
improving_adds: 0
pairwise_candidates: 16
improving_pairwise: 0
local_relabel_candidates: 0
```

Best replacement:

```text
active B12 -> inactive BIRTH_F1_F6_6
candidate admissible: true
energy_delta: +0.865747
```

Relevant missing-label checks:

```text
B4 as replacement/add: not admissible in local single-object move
B11 as replacement: not admissible, +52.193764 energy_delta in the inspected overlap
```

Interpretation:

- Single-object local add/replacement does not explain the missing 12th.
- Pairwise local neighborhoods also do not explain it.
- Frozen local re-labeling over bounded neighborhoods made no changes.
- Best pairwise move is still `+3.060135` energy:

  ```text
  B12 -> BIRTH_F1_F6_6
  B7 -> B10
  ```

- The correct 12-count alternatives exist globally, but they require a coordinated multi-object neighborhood change.
- Loosening one birth/replacement rule is likely to broaden the range rather than select exact 12.

## ysherman Diagnostic

Replacement-diagnostic rerun:

`data/bend_corpus/evaluation/patch_graph_f1_ysherman_replacement_diag_20260424`

Pairwise neighborhood diagnostic rerun:

`data/bend_corpus/evaluation/patch_graph_f1_ysherman_pairwise_diag_20260424`

Frozen local re-label diagnostic reruns:

`data/bend_corpus/evaluation/patch_graph_f1_ysherman_local_relabel_diag_20260424`

`data/bend_corpus/evaluation/patch_graph_f1_ysherman_local_relabel_count_diag2_20260424`

Current result:

```text
accepted: [11,12]
raw F1: exact 11 / [11,11]
expected: 12
```

Local replacement diagnostic:

```text
current_energy: 1895.75865
active_bend_component_count: 11
inactive_bend_label_count: 12
replacement_candidates: 14
add_candidates: 14
improving_replacements: 0
improving_adds: 0
pairwise_candidates: 16
improving_pairwise: 0
local_relabel_candidates: 4
energy_improving_local_relabels: 4
energy_and_count_improving_local_relabels: 0
```

Best admissible add:

```text
candidate B13
candidate source: detector_bend_seed
component_atom_count: 10
candidate_admissible: true
energy_delta: +50.830474
selected pair: [F13,F6]
```

Interpretation:

- `ysherman` is not missing a trivially addable bend under the current objective.
- It is also not fixed by pairwise local swaps.
- Frozen local re-labeling finds lower-energy cleanup moves, but none recover the missing count:

  ```text
  best local relabel: energy_delta -35.320683, count 11 -> 10
  second local relabel: energy_delta -10.399713, count 11 -> 11
  no local relabel has count_delta > 0
  ```

- Best pairwise move is still `+8.589187` energy and uses inadmissible replacements:

  ```text
  BIRTH_F1_F6_10 -> B9
  BIRTH_F12_F7_7 -> B2
  ```

- The missing 12th also appears to require a coordinated local decomposition change, not a one-object birth.
- The accepted `[11,12]` range remains justified; raw exact 11 should not be promoted.

## Rejected Experiment: Active-Overlap Trimming

Experiment folder:

`data/bend_corpus/evaluation/patch_graph_f1_focus_trim_overlap_apply_20260424`

Idea:

- Instead of skipping interface-birth candidates that overlap active bend support, trim active-overlap atoms and try to claim the remaining connected residual support.

Result:

- Accepted control-constrained layer stayed safe.
- Raw F1 on `49204020_04` broadened from `[11,12]` to `[11,14]`.
- Extra rescued birth labels increased ambiguity instead of selecting exact 12.

Decision:

- Rejected and reverted.
- This suggests simple birth support expansion is too permissive.
- The next successful move needs stronger overlap-aware model selection, not looser birth activation.

## Current Interpretation

The current scorer still lacks a principled "one object vs two objects" comparison for overlapping or adjacent bend supports.

The likely missing terms are:

1. A local BIC/MDL split/merge score for neighboring bend supports.
2. Explicit replacement moves:
   - replace interface birth with one or more seed bends,
   - replace a seed bend with an interface birth,
   - compare the two alternatives under the same atom support.
3. A stronger support topology term:
   - whether a candidate owns a coherent support corridor,
   - whether its support is too small, too isolated, or too duplicate-like,
   - whether two nearby supports explain distinct flange interfaces or the same physical bend.
4. A competition term between interface-birth labels and seed labels when they explain overlapping supports.

## Concrete Questions

1. For `49204020_04`, should the 11-vs-12 decision be formulated as a local replacement move over a shared support region, rather than as global label-cost tuning?

2. How should we score:

   ```text
   one interface-birth bend + adjacent seed bends
   ```

   versus

   ```text
   two/three original seed bends with separate supports
   ```

   when their supports overlap or touch the same flange pair family?

3. What is the right local BIC/MDL penalty for adding a bend object when the support is small but geometrically plausible?

4. Should interface-birth labels be lower-confidence temporary proposals that must beat a seed-label replacement score before becoming countable?

5. Should selected-pair duplicate pruning operate before or after split/merge replacement moves?

6. Should the solver keep a "decomposition family range" when two decompositions differ by about 12 energy units, or is that already enough evidence to suppress the higher-energy count?

7. For `ysherman`, raw F1 is exact 11 while accepted output is `[11,12]`. Is this the same missing-local-birth problem as `49204020_04`, or a separate issue in support decomposition?

## Recommended Next Tranche

Do not promote F1 to runtime.

Do not loosen interface births further.

Next implementation should add a targeted local replacement evaluator:

```text
For each overlapping cluster of active/proposed bend labels:
  collect atoms in the union support neighborhood
  build 2-4 competing multi-object local decompositions
  score each with data + incidence + fragmentation + duplicate + MDL
  accept replacement only if it lowers total objective and does not broaden protected anchors
```

The one-object and pairwise diagnostics already exist and found no improving count-recovery moves on `49204020_04` or `ysherman`.

Bounded frozen local re-labeling also does not recover count:

- `49204020_04`: no local re-label changes.
- `ysherman`: energy-improving local re-labels exist, but they reduce or preserve count; none increase count.

Therefore the next tranche should be larger multi-object neighborhood search with explicit count/evidence constraints, not automatic acceptance of lower-energy local cleanups:

```text
active cluster C = active bends + inactive proposals that overlap/touch C

candidate decompositions:
  keep current active labels
  add one inactive support
  replace one active with one inactive
  swap interface birth plus neighboring seed labels
  split/merge two adjacent active supports
  re-solve atom ownership inside C over 3-6 labels while freezing the outside

only accept if:
  energy improves or stays within calibrated tolerance
  target hard-case count range tightens or moves toward truth
  protected anchors do not broaden
  selected-pair duplicate pressure does not increase
  local re-label does not simply delete weak bend support
```

First test target:

- `49204020_04`: require no widening beyond `[11,12]`; success is exact 12 or a narrower evidence-backed tie.

Secondary target:

- `ysherman`: require no broadening beyond `[11,12]`; success is exact 12 or stronger support for 12 in the range.

Regression anchors:

- `10839000`: must remain accepted exact 10 / `[9,10]`.
- `47959001`: must remain accepted exact 6 / `[6,6]`.
- `48963008`: must remain accepted exact 3 / `[3,3]`.
- `48991006`: must remain accepted `[9,12]` or improve.

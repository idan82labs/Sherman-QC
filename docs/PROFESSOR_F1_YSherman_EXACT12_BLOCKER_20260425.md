# Professor Handoff: F1 Exact-12 Blocker on `ysherman`

## Context

We have continued the offline-only F1 patch-graph latent decomposition lane. Runtime zero-shot remains unchanged.

The current F1 lane now works well as a conservative observed-bend range engine:

- clean known full-scan range containment: `2/2`
- manual partial observed-bend range containment: `10/10`
- partial exact count promotion: intentionally blocked

The next blocker is no longer broad partial containment. It is exact full-scan promotion on a hard repeated-near-90 case:

```text
scan: ysherman.ply
manual observed/full count: 12
current accepted output: [11, 12]
raw F1 MAP: [11, 11]
zero-shot control: [10, 12]
```

So the system is safe, but not exact.

## Current Formulation

The F1 lane uses:

- surface atoms as ownership units
- flange hypotheses
- bend hypotheses from zero-shot strips plus interface births
- candidate-based MAP labeling
- object-level admissibility and duplicate suppression
- scan-family router
- conservative guard layer against exact promotion

For `ysherman`, the router chooses:

```json
{
  "route_id": "fragmented_repeat90_interface_birth",
  "solver_profile": "interface_birth_enabled",
  "enabled_moves": [
    "full_geometric_attachment",
    "interface_activation",
    "duplicate_collapse",
    "repair_before_prune"
  ]
}
```

The feature snapshot is:

```json
{
  "point_count": 897663,
  "local_spacing_mm": 2.7059,
  "dominant_angle_family_deg": 90.702,
  "dominant_angle_family_mass": 0.82,
  "duplicate_line_family_rate": 0.35,
  "single_scale_count": 7,
  "multiscale_count": 13,
  "union_candidate_count": 20,
  "zero_shot_count_range": [10, 12],
  "atom_count": 1538,
  "flange_hypothesis_count": 11,
  "flange_orientation_family_count": 3,
  "bend_hypothesis_count": 13,
  "avg_candidate_attachment_pairs": 3.6154,
  "weak_incident_bend_fraction": 0.0769,
  "multi_attachment_bend_fraction": 1.0,
  "seed_incidence_risk": 0.5,
  "small_tight_bend_signal": 0.3
}
```

## Baseline Result

Baseline F1 is stable at 11:

```json
{
  "raw_bend_count_range": [11, 11],
  "raw_solution_counts": [11, 11, 11],
  "best_energy": 1895.761378,
  "accepted_candidate_range": [11, 12],
  "accepted_candidate_source": "control_constrained_f1",
  "guard_reason": "singleton_below_control_upper"
}
```

The 11 owned regions are:

```text
OB1   atoms=42  angle=89.87  flanges=[F13,F7]
OB2   atoms=17  angle=90.02  flanges=[F13,F3]
OB3   atoms=14  angle=81.80  flanges=[F12,F20]
OB4   atoms=13  angle=98.20  flanges=[F11,F3]
OB5   atoms=8   angle=81.69  flanges=[F13,F20]
OB6   atoms=5   angle=81.80  flanges=[F11,F19]
OB7   atoms=4   angle=89.56  flanges=[F1,F3]
OB8   atoms=4   angle=89.24  flanges=[F12,F7]
OB9   atoms=3   angle=89.56  flanges=[F12,F3]
OB10  atoms=3   angle=89.56  flanges=[F2,F3]
OB11  atoms=3   angle=89.73  flanges=[F1,F6]
```

## Count Sweep

We tested whether the missing 12th bend appears when the active-birth cap is relaxed:

```text
baseline: raw [11,11], energy 1895.761378
cap_11:   raw [11,11], energy 1895.758650
cap_12:   raw [11,11], energy 1895.758650
cap_13:   raw [11,11], energy 1895.758650
cap_16:   raw [11,11], energy 1895.758650
uncapped: raw [11,11], energy 1895.758650
```

Conclusion: the 12th bend is not recovered by simply allowing more active births.

## Proposal Relaxation Probe

We then enabled subspan interface births:

```text
baseline:            raw [11,11], energy 1895.761378, births=10
subspan_births:      raw [17,17], energy 2057.849808, births=24
subspan_no_contact:  raw [17,17], energy 2057.849808, births=24
no_dup_prune:        raw [17,17], energy 2057.849808, births=24
more_iters_no_dup:   raw [17,17], energy 2057.849808, births=24
```

Conclusion: global subspan birth is too permissive. It opens the floodgate to 17 fragments and has much worse energy. However, the 17-solution may contain the missing physical bend among spurious fragments.

## Visual Evidence

We generated a dedicated render pack:

```text
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker
```

Important files:

```text
baseline_11/render/patch_graph_owned_region_overview.png
baseline_11/decomposition_result.json
subspan_17/render/patch_graph_owned_region_overview.png
subspan_17/decomposition_result.json
YSherman_exact_blocker_summary.json
```

Visual read:

- baseline 11 is clean-ish but missing one physical bend according to manual truth.
- subspan 17 adds many fragments/clusters, especially around repeated near-90 local features.
- subspan 17 is not directly promotable.

## Current Interpretation

This no longer looks like a simple routing or count-guard problem.

It looks like a local model-selection problem:

```text
We need to activate exactly one additional physically valid bend support,
but the available broad subspan birth mechanism creates about six extra fragments.
```

The current energy strongly prefers the 11-bend solution over the 17-bend subspan solution. But it has no clean intermediate 12-bend local move.

## Candidate-Ranking Addendum

We added a diagnostic-only ranking pass over the overcomplete subspan solution. It compares each subspan owned region against the stable 11-bend baseline and scores whether it looks like a plausible extra non-duplicate bend support.

Artifacts:

```text
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_candidate_ranking.json
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/EXACT_BLOCKER_CANDIDATE_RANKING.md
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_candidate_contact_sheet_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_local_selection.json
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/EXACT_BLOCKER_LOCAL_SELECTION.md
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_conflict_set_solver.json
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/EXACT_BLOCKER_CONFLICT_SET_SOLVER.md
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_conflict_sensitivity.json
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/EXACT_BLOCKER_CONFLICT_SENSITIVITY.md
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_corridor_coverage.json
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/EXACT_BLOCKER_CORRIDOR_COVERAGE.md
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_quotient_corridor_audit.json
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/EXACT_BLOCKER_QUOTIENT_CORRIDOR_AUDIT.md
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/exact_blocker_uncovered_corridor_comparison.json
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/F1 Accuracy Validation Renders 2026-04-25 YSherman Exact Blocker/candidate_ranking/EXACT_BLOCKER_UNCOVERED_CORRIDOR_COMPARISON.md
```

The ranking is explicitly not a promotion rule. It is only meant to identify which overcomplete fragments are plausible test subjects for a formal local add/split move.

Top candidates:

```text
OB1   score=0.911922  atoms=21  reasons=[non_duplicate_candidate]
OB6   score=0.900406  atoms=10  reasons=[non_duplicate_candidate]
OB7   score=0.888762  atoms=10  reasons=[non_duplicate_candidate]
OB10  score=0.856069  atoms=6   reasons=[non_duplicate_candidate]
OB11  score=0.832980  atoms=6   reasons=[non_duplicate_candidate]
OB4   score=0.822081  atoms=13  reasons=[non_duplicate_candidate]
```

Visual read of the contact sheet:

- the ranking does surface plausible extra supports, so the overcomplete 17-solution is useful diagnostically;
- several high-rank candidates still appear to sit along the same repeated vertical/near-90 interface family;
- therefore a naive "take the top candidate" rule is unsafe;
- the missing formal piece is a conflict-aware local model comparison that can add one support while rejecting duplicate subspan fragments on the same corridor.

This reinforces the concrete question below: should the next move be formulated as a local one-birth-vs-baseline test, a candidate subset selection problem over the subspan pool, or a split test on ambiguous baseline supports?

We then added a first diagnostic local-selection probe:

```text
target count: 12
status: hold_exact_promotion
recommended candidate if exactly one birth is forced: OB1
OB1 score: 0.911922
runner-up: OB6
score gap: 0.011516
admissible candidates: 8
rejected candidates: 9
hold reasons:
  - top_candidate_not_unique_enough
  - too_many_plausible_candidates
  - plausible_candidates_conflict_on_corridors
```

Most important candidate conflicts:

```text
OB1 vs OB6   conflict=0.70  distance=16.21mm  reasons=[near_centroid, parallel_axis, span_overlap]
OB6 vs OB11  conflict=1.00  distance=10.70mm  reasons=[same_incident_pair, near_centroid, parallel_axis, span_overlap]
OB10 vs OB12 conflict=0.50  reasons=[same_incident_pair, parallel_axis]
OB4 vs OB5   conflict=0.50  reasons=[same_incident_pair, parallel_axis]
```

This is the strongest evidence so far that exact promotion should remain blocked: the overcomplete pool does contain plausible extra supports, but the top two are too close in score and physically/corridor-conflicting. The next solver change should probably resolve a local conflict set, not simply add the highest-scoring candidate.

Finally, we added a small diagnostic conflict-set subset solver over the admissible candidates. It enumerates subsets up to size 3 with a label cost and pairwise conflict penalties.

Result:

```text
status: hold_exact_promotion
admissible candidates: 8
enumerated subsets: 92
hold reasons:
  - one_birth_subset_not_dominant
  - multi_birth_subset_competes_with_one_birth
  - conflict_graph_non_empty
  - too_many_admissible_birth_candidates
```

Best subset by added count:

```text
+1: [OB1]              score=0.491922
+2: [OB1, OB7]         score=0.960684
+3: [OB1, OB7, OB10]   score=1.396753
```

This confirms that our current local candidate score is not yet an exact-12 acceptance objective. It can rank plausible missing-support fragments, but it does not distinguish "the single missing physical bend" from "multiple plausible subspan births." We need a stronger object-level model-selection term before exact promotion.

We also swept label cost and conflict weight:

```text
label_costs:      [0.30, 0.42, 0.55, 0.70, 0.85, 1.00]
conflict_weights: [0.50, 0.85, 1.20, 1.60]
dominant +1 runs: 0 / 24
interpretation: no_reasonable_sweep_promotes_exact_12
```

The sweep confirms this is not just a constant-tuning issue. Low/moderate label costs favor multi-birth subsets; high label costs suppress births without providing a uniquely defensible 12th object. The missing term is likely a better object likelihood/corridor coverage model that explains why exactly one local corridor remains unowned, not a stronger generic label cost.

We then grouped admissible candidates into provisional corridor families using the conflict graph. Result:

```text
status: hold_exact_promotion
corridors: 4
strong corridors: 4
admissible clean corridors: 1
hold reasons:
  - multiple_strong_corridor_gaps
  - corridor_fragmentation_unresolved
  - internal_corridor_conflicts
```

Corridor summary:

```text
C1: [OB1, OB6, OB11]   best=OB1   score=0.911922  ambiguous/conflicting fragments
C2: [OB7]              best=OB7   score=0.888762  clean isolated candidate
C3: [OB10, OB12]       best=OB10  score=0.856069  ambiguous/conflicting fragments
C4: [OB4, OB5]         best=OB4   score=0.822081  ambiguous/conflicting fragments
```

This is the clearest current formulation gap: the overcomplete pool does not indicate "one missing bend"; it indicates several strong unresolved corridor families. The next mathematical term should probably score **corridor coverage relative to the baseline flange/bend quotient graph**, not just candidate support quality. In other words, which flange-interface corridor is actually missing an owned bend component, versus which corridors are birth-generator artifacts?

We added a first quotient-graph audit against the baseline assignment. It checks whether each overcomplete corridor's selected flange pair is already represented by an owned baseline bend or by a bend label touching the same two flanges.

Result:

```text
status: hold_exact_promotion
candidate corridors: 4
uncovered exact pairs: 2
exact-covered corridors: 2
hold reasons:
  - multiple_uncovered_exact_pairs
  - corridor_coverage_not_single_gap
```

Corridor audit:

```text
C1 pair=[F10,F19] best=OB1   uncovered exact pair=true
C2 pair=[F1,F19]  best=OB7   uncovered exact pair=true
C3 pair=[F13,F6]  best=OB10  quotient graph already has bend on pair
C4 pair=[F11,F19] best=OB4   exact pair already has owned region
```

This narrows the unresolved case from four strong candidate corridors to two exact-uncovered flange-pair corridors. It still does not justify exact 12 because two uncovered exact-pair hypotheses remain, and flange IDs may be fragmented. The next useful formula should compare these two uncovered corridors using a stronger physical criterion: interface-line coverage, alternating flange-bend-flange topology, residual transition mass, and whether either uncovered pair is just a split of an already represented physical flange family.

We then compared the two exact-uncovered corridors geometrically using pair-axis agreement, candidate/pair angle agreement, support strength, bendness, baseline separation, span, and distance to the selected flange-pair interface line.

Result:

```text
status: hold_exact_promotion
top: OB1, pair=[F10,F19], score=0.836488
runner-up: OB7, pair=[F1,F19], score=0.782131
gap: 0.054357
hold reasons:
  - multiple_uncovered_corridors
  - uncovered_corridor_comparison_not_dominant
  - top_uncovered_corridor_geometry_weak
```

Both candidates have near-perfect pair-axis and angle agreement, but both have weak interface-line fit:

```text
OB1 line_fit=0.179647, axis=1.0, angle=1.0, atoms=21
OB7 line_fit=0.0,      axis=1.0, angle≈1.0, atoms=10
```

This suggests the selected flange IDs are probably not enough to define the physical corridor line. The likely next mathematical issue is flange-component fragmentation or plane-pair line instability. A correct test may need to compare a candidate support to a **local flange-interface corridor inferred from adjacent observed atoms**, not the infinite intersection line of two global flange hypotheses.

We then added a PCA local-support corridor fit, independent of global flange plane intersection. This did **not** resolve the ambiguity:

```text
status: hold_exact_promotion
top: OB1, pair=[F10,F19], score=0.806248
runner-up: OB7, pair=[F1,F19], score=0.771384
gap: 0.034864
```

Both exact-uncovered candidates are also weak under the local support-line fit:

```text
OB1 pair_line=0.179647, local_line=0.340895, pair_axis=1.0, local_axis=0.0, atoms=21
OB7 pair_line=0.0,      local_line=0.351808, pair_axis=1.0, local_axis=0.0, atoms=10
```

So the current evidence says neither remaining candidate is a clean line-like support corridor. They may be curved/fragmented transition clusters, or they may be artifacts from the subspan birth generator. The next mathematical test likely needs **boundary-contact topology and local residual ownership**, not only line/corridor geometry.

We added that boundary/residual ownership probe. It checks what the candidate atoms were labeled as in the baseline, and whether the overcomplete candidate has boundary contact to its designated flange pair.

Result:

```text
status: hold_exact_promotion
top by ownership/topology: OB7, pair=[F1,F19], score=0.598788
runner-up: OB1, pair=[F10,F19], score=0.230469
```

Key difference:

```text
OB7 baseline atoms: residual=1.0
OB1 baseline atoms: flange=0.952381, bend=0.047619
```

So `OB7` is the more credible missing-support candidate under residual ownership: it explains atoms the baseline left residual. `OB1` mostly steals atoms already owned by baseline flanges, so it is more likely a flange-fragment/birth artifact.

However, neither candidate has clean designated flange boundary contact in the current graph:

```text
OB7 designated_contact_ratio=0.0, leakage=0.0
OB1 designated_contact_ratio=0.0, leakage=0.307692
```

This means `OB7` is now the best hypothesis for the missing 12th, but still not exact-promotable. The next model issue is probably that boundary contact is not visible in the atom graph at the candidate scale, or the selected flange pair is still wrong/incomplete. A promotion rule would need a way to accept a residual transition support with weak explicit contact, perhaps by using nearby flange proximity/corridor distance rather than adjacency-only contact.

We added a distance-based flange proximity incidence probe to replace adjacency-only contact. It measures candidate support distances to the selected flange planes and compares designated-pair proximity against leakage to other flanges.

Result:

```text
status: hold_exact_promotion
top: OB7, pair=[F1,F19], score=0.565174
runner-up: OB1, pair=[F10,F19], score=0.490999
gap: 0.074175
hold reasons:
  - flange_proximity_not_dominant
  - multiple_uncovered_corridors
  - top_flange_proximity_not_clean
```

Details:

```text
OB7 designated_near=0.90, designated_median=8.60mm, best_other_near=1.00, best_other_median=6.09mm
OB1 designated_near=0.81, designated_median=5.08mm, best_other_near=1.00, best_other_median=0.48mm
```

This again favors `OB7` over `OB1`, but not cleanly. Both supports are closer to some non-designated flange than we would like. The repeated finding is now:

```text
OB7 is the best missing-support hypothesis because it owns baseline-residual atoms
and has better selected-flange proximity than OB1,
but the selected flange-pair identity is still contaminated by neighboring/fragmented flange components.
```

So the next formulation probably needs **latent flange-family incidence**, not raw flange-ID incidence: group flange hypotheses into physical flange families/components, then score bend incidence to families while preserving connected component IDs for count.

We added a flange-family incidence diagnostic. Flange hypotheses were grouped by orientation/plane similarity:

```text
raw flanges: 11
families: 6
FF1 = [F1,F2]
FF2 = [F10,F11,F12]
FF3 = [F13]
FF4 = [F19,F20]
FF5 = [F3]
FF6 = [F6,F7]
```

This resolves the `OB1` vs `OB7` ambiguity at the family level:

```text
OB1 raw pair=[F10,F19] -> family pair=[FF2,FF4]
family pair already covered by baseline

OB7 raw pair=[F1,F19] -> family pair=[FF1,FF4]
family pair uncovered
```

Family-incidence status:

```text
single_uncovered_flange_family_pair
```

This is the first diagnostic that produces a clean structural argument for `OB7` as the missing 12th candidate:

```text
OB7 owns baseline-residual atoms
OB7 is the only candidate whose flange-family pair remains uncovered
OB1 is explained as raw flange-ID fragmentation of an already-covered family pair
```

We are still not promoting exact 12 automatically, because this is a diagnostic and not yet an integrated F1 repair rule. But the next implementation target is now precise: add a local repair candidate acceptance rule based on **baseline residual ownership + uncovered flange-family pair incidence**, then test it across the six-scan focus slice before allowing exact promotion.

## Concrete Question

What is the right mathematical local move for this case?

More specifically:

1. Should we formulate a **single-interface local split test** on each large/ambiguous owned support, comparing:

```text
one bend on support S
vs
two bends on S with a small intervening flange/contact gap
```

using BIC/MDL and incidence/contact terms?

2. Or should we formulate a **candidate-subset selection problem** over subspan births:

```text
choose k subspan births from the 17-candidate overcomplete pool
subject to duplicate/conflict penalties and support coverage
```

where we compare 11, 12, ..., 17 active-object decompositions directly?

3. What should the score be for selecting one subspan birth while rejecting nearby fragments?

We currently have:

```text
unary + Potts smoothness + label cost
object incidence penalty
fragmentation penalty
selected-pair duplicate penalty
support-mass admissibility
```

But the missing term seems to be:

```text
local split/birth benefit relative to coverage of an unassigned transition corridor,
while penalizing repeated fragments on the same flange-pair corridor.
```

4. Should this be implemented as:

```text
birth proposal scoring before MAP
```

or:

```text
post-MAP repair move that tests one added birth at a time
```

against the current 11-bend decomposition?

5. What would be a defensible acceptance rule for a 12-bend local move?

For example, should we require:

```text
local energy improvement on previously residual/high-bendness atoms
AND clean incidence to a selected flange pair
AND low overlap/conflict with existing bends
AND no increase in duplicate-like selected-pair clusters
```

or should we use a stricter BIC-style global score even if the added bend has small support?

## What We Need From You

Please translate this into literal next-step formulas / code-level replacement logic for:

- local split/birth candidate scoring
- conflict graph among subspan births and existing bends
- acceptance rule for adding exactly one bend to a stable MAP decomposition
- how to distinguish “true missing neighboring bend” from “spurious subspan fragment”

The specific engineering goal is:

```text
ysherman should move from accepted [11,12] to exact 12 only if the 12th support is mathematically defensible.
```

We do not want to tune a heuristic that also turns clean 4-bend or partial scans into over-fragmented 17-bend outputs.

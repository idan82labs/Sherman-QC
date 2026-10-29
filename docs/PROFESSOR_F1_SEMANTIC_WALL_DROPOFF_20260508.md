# F1 Semantic Wall Dropoff - May 8, 2026

## Executive Summary

We resolved the floating-marker/render-frame issue. The remaining blocker is semantic, not display placement:

Current F1 can often find real transition evidence, but it still confuses these objects:

- countable conventional bend-line instances
- edge/perimeter/hole artifacts
- rounded/rolled transition families
- connected corner/junction support that should not itself be counted
- missing internal/long-arm bends where the evidence exists but is not promoted to a countable object

The best current aggregate on the 9-case semantic known-conventional slice is:

```text
candidate exact accuracy:              0.80 on evaluated exact cases
candidate range contains truth:         0.889
raw F1 exact accuracy:                  0.40
raw F1 range contains truth:            0.556
render-admissible exact accuracy:       0.222
render-admissible range contains truth: 0.222
render-blocked candidate cases:         8 / 9
```

Interpretation: the guarded/count-selection layer is still useful numerically, but raw/render F1 ownership is not ready to promote. The visual-validity gate is doing its job by blocking bad markers. The next mathematical work is not another render fix; it is a stronger object model for conventional bend-line instances.

## What Is Resolved

### Floating markers / CAD+PLY overlay frame mismatch

We added scan-supported display gates to the regular CAD+PLY overlay path and the zero-shot strip overlay path.

Resolved behavior:

- unsupported CAD/local-frame lines are suppressed before rendering;
- unsupported zero-shot strip axis lines are suppressed before rendering;
- F1 owned-region markers use owned atom support/medoids rather than raw strip anchors;
- generated review packs now report zero unsupported rendered lines after the display gate.

Relevant files in the package:

```text
06_resolved_render_gate/FLOATING_MARKER_GATE_STATUS.json
06_resolved_render_gate/README.md
```

Code references:

```text
backend/bend_report_generator.py
backend/zero_shot_bend_overlay.py
backend/patch_graph_latent_decomposition.py
```

This is not the professor question unless you see a geometric reason that the display gate invalidates later modeling. We currently treat it as an engineering fix.

## Current F1 Ontology

Current F1 lane has:

```text
SurfaceAtom
SurfaceAtomGraph
FlangeHypothesis
BendHypothesis
TypedLabelAssignment
DecompositionResult
OwnedBendRegion
```

The current decomposition is patch/atom based:

```text
atoms -> flange / bend / residual / outlier labels
owned bend regions -> marker-quality gate -> render-admissible accepted markers
```

The recent visual gate rejects conventional marker promotion when support is:

```text
too short
too small
edge/perimeter dominated
hole/free-boundary dominated
weakly incident to two flanges
duplicated/clustered
curve-family rather than conventional
```

That prevents false visual success, but it does not recover missed conventional bend lines.

## Current Measured Status

Benchmark files:

```text
03_benchmark_jsons/semantic_known_conventional_summary.json
03_benchmark_jsons/semantic_visual_probe_summary.json
03_benchmark_jsons/f1_semantic_known_conventional_slice.json
```

Key aggregate from `semantic_known_conventional_summary.json`:

```text
scans_processed:                         9
exact_count_evaluated:                   5
candidate_exact_accuracy:                0.8
raw_f1_exact_accuracy:                   0.4
render_admissible_exact_accuracy:        0.222
control_range_contains_truth_rate:       0.778
candidate_range_contains_truth_rate:     0.889
raw_f1_range_contains_truth_rate:        0.556
render_admissible_range_contains_truth:  0.222
tightened_case_count:                    2
broadened_case_count:                    0
render_blocked_candidate_count:          8
recovery_promotion_eligible_parts:       [49024000]
recovery_missing_arm_birth_support:      [49024000]
```

My conclusion:

```text
Do not promote raw/render F1.
Keep F1 offline.
Keep the visual-validity gate.
Use guarded candidate counts only as research signals, not product truth.
The next solver should produce countable maximal bend-line instances, not counted support patches.
```

## Primary Case Study 1: 49024000

Manual truth from Idan:

```text
3 conventional bends
OB2 visually plausible
other raw F1 regions are false/fragment supports
```

Current status:

```text
F1 has real evidence around the connected ridge/corner complex.
Birth arrangement diagnostics look promising.
Current F1 cannot promote because it lacks sparse pair/junction arrangement selection.
```

Relevant files:

```text
02_case_studies/49024000/49024000_birth_arrangement_review_sheet.png
02_case_studies/49024000/49024000_raw_f1_overview.png
02_case_studies/49024000/49024000_missed_line_candidates.png
02_case_studies/49024000/49024000_local_ridge_candidates.json
02_case_studies/49024000/49024000_residual_interface_support_candidates.json
02_case_studies/49024000/49024000_residual_interface_subclusters.json
02_case_studies/49024000/49024000_junction_bend_arrangement.json
02_case_studies/49024000/49024000_split_candidate_diagnostics.json
02_case_studies/49024000/decomposition_result_visual_probe.json
```

Working hypothesis after your previous guidance:

```text
Connected ridge/corner support should become:

uncounted junction/corner complex
  + countable maximal conventional bend-line instance 1
  + countable maximal conventional bend-line instance 2
  + countable maximal conventional bend-line instance 3
```

Likely sparse topology:

```text
F17 -- F11 -- F14 -- F1
```

Likely countable direct transitions:

```text
F11-F17
F11-F14  (accepted OB2)
F14-F1
```

Likely chord/diagonal to reject:

```text
F1-F17
```

The hard modeling question:

```text
How should the local MAP/MDL problem choose a sparse flange-incidence path and reject chord edges when all candidate pairs have overlapping support in one connected ridge/corner domain?
```

## Primary Case Study 2: 48991006

Manual feedback from Idan:

```text
OB4 and OB5 sit on the same bend, so they are duplicate/alias supports.
The part processed poorly.
A whole bend in the middle/internal raised area is unmarked.
Several raw regions sit on exterior edges/perimeters.
```

Current visual read:

```text
Raw F1 clusters on exterior/top/right edge families.
Internal/middle bend structures remain missed.
Side view confirms multiple labels are aliases on one exterior edge family.
This is not a floating-marker bug; the markers are scan-supported but semantically wrong.
```

Relevant files:

```text
02_case_studies/48991006/48991006_raw_f1_overview.png
02_case_studies/48991006/48991006_raw_f1_side.png
02_case_studies/48991006/48991006_missed_line_candidates.png
02_case_studies/48991006/48991006_uncovered_bend_corridors.json
02_case_studies/48991006/48991006_interior_bend_gaps.json
02_case_studies/48991006/48991006_junction_bend_arrangement.json
02_case_studies/48991006/48991006_junction_bend_arrangement_with_interior.json
02_case_studies/48991006/decomposition_result_visual_probe.json
```

The hard modeling question:

```text
What is the correct recovery move for a missing internal/raised conventional bend when edge/perimeter candidates dominate the patch graph?
```

Specific subquestions:

```text
Should internal bend-line birth be driven by uncovered flange-pair corridors even when the local owned-support/ridge signal is weaker than exterior edge supports?

Should edge/perimeter supports compete under a different object class so they do not absorb conventional bend-line budget?

How should duplicate/alias suppression work when multiple F1 regions are scan-supported and near a bend but represent the same physical edge/bend family?
```

## Secondary Case Studies

### 37361005

Manual truth:

```text
4 conventional bends
```

Observed F1 failure:

```text
Raw F1 creates many edge/perimeter markers.
Visual gate correctly blocks promotion.
```

Relevant file:

```text
02_case_studies/37361005/37361005_raw_f1_overview.png
```

Question:

```text
Should free-boundary/perimeter ridge supports be modeled as an explicit non-counted edge-transition class, separate from residual/outlier, to prevent them from competing as bend-line candidates?
```

### 10839000

Manual truth from Idan:

```text
8 conventional bends + 2 raised/form features
```

Prior legacy/control label expected 10, which mixed conventional bends and raised features.

Observed issue:

```text
The validation target itself must distinguish conventional bend-line count from raised/form-feature count.
```

Question:

```text
Should the solver report separate object families:
conventional_bend_line_count
raised/form_feature_transition_count
rounded/rolled_transition_count
instead of one bend count?
```

### 49125000

Manual truth:

```text
rounded/rolled part, 2 bends
```

Observed issue:

```text
F1 can mark curve/rounded support, but the visual gate blocks it as conventional bend-line success.
```

Question:

```text
Should rolled/rounded cases be routed into a separate curve-family lane with different count semantics, instead of being judged by conventional bend-line gates?
```

## Current Code Surface For Replacement

Most relevant file:

```text
04_code_files/patch_graph_latent_decomposition.py
```

Functions/areas to inspect:

```text
run_patch_graph_latent_decomposition(...)
solve_typed_latent_decomposition(...)
build_owned_region_marker_admissibility(...)
build_owned_region_renderable_objects(...)
_bend_data_cost(...)
_label_cost(...)
_bend_component_attachment_evaluation(...)
_bend_component_admissibility(...)
_activate_interface_bends(...)
_apply_object_repair_moves(...)
_build_missed_line_candidates(...)
_build_birth_arrangement_candidates(...)
_build_junction_arrangement_candidates(...)
```

Router:

```text
04_code_files/scan_family_router.py
```

Runner:

```text
04_code_files/run_patch_graph_f1_prototype.py
```

Marker audit:

```text
04_code_files/audit_f1_marker_quality.py
```

## Current Conceptual Wall

The current F1 countable object is still too close to:

```text
connected owned support component
```

But the visual review indicates the countable object should be:

```text
maximal, flange-pair-conditioned, axis-stable conventional bend-line instance
```

inside a larger object:

```text
uncounted corner/junction/edge/curve transition complex
```

The specific hard problem is:

```text
Given one connected high-curvature/ridge domain, infer:

1. countable direct flange-pair bend-line paths,
2. uncounted junction/corner atoms,
3. uncounted perimeter/edge artifacts,
4. curve/rolled support where applicable,
5. residual/uncertain atoms,

while rejecting chord/diagonal candidate pairs and duplicate aliases.
```

## Concrete Request

Please rewrite this as literal formulas and code-level replacement logic for the next local solver tranche.

The target implementation should answer:

```text
For a connected ridge/corner domain R, how do we infer:

F_R = participating flange hypotheses
B_R = countable conventional bend-line instances
J_R = uncounted junction/corner-transition zones
E_R = uncounted free-edge/perimeter ridge support
C_R = curve/rolled transition support where applicable
U_R = residual/uncertain atoms
```

Please provide:

```text
1. The exact local energy/MDL score.
2. The data term for assigning atoms or skeleton segments to a bend-line instance.
3. The junction/corner label cost and acceptance criteria.
4. The edge/perimeter non-counted class criteria.
5. The chord/diagonal rejection criterion for pair candidates like F1-F17 on 49024000.
6. The internal/missing-line birth criterion for cases like 48991006.
7. The merge/split/remove moves and their acceptance tests.
8. The final count rule.
9. The minimum diagnostics we should output so visual review can prove the model is doing the right thing.
```

## Proposed Direction To Confirm Or Correct

My current proposal is:

```text
Build local ridge domains R from high-curvature/support atoms.
Build a ridge skeleton/path graph inside R.
Compute per-skeleton-node posterior over flange pairs.
Generate maximal path candidates for stable pair + stable axis.
Add explicit uncounted labels:
  junction_transition
  edge_perimeter_transition
  curve_or_rolled_transition
Solve a small local MAP/MDL selection:
  bend-line candidates + junctions + edge/curve/residual
Count only selected conventional bend-line instances.
Reject direct chord pair if a sparse path through intermediate flanges explains the same support with lower MDL.
Birth missing internal lines from uncovered stable flange-pair corridors, not from generic edge ridge strength.
```

Please be aggressive in correcting this if the object model is still wrong.


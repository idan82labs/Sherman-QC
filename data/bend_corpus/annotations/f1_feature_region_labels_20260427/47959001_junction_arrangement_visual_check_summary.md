# `47959001` Junction Arrangement Visual Check

Date: 2026-04-29

## Inputs

- `47959001_count_context_region_labels.json`
- `47959001_count_context_summary.json`
- `47959001_residual_interface_support_candidates.json`
- `47959001_local_ridge_candidates.json`
- `47959001_junction_bend_arrangement.json`
- Source decomposition: `data/bend_corpus/evaluation/status_partial_visual_qa_native_20260426/partial/03_47959001_47959001_01PREL_scan1/decomposition_result.json`

## Count Context

This is not a per-region manual-label case.

The validation context is:

```text
expected visible observed bends = 5
raw F1 owned regions = 4
diagnostic recovery target = 1 missing conventional bend-line instance
```

## Diagnostic Chain Result

Residual/interface support:

```text
status = candidate_pool_covers_deficit
recovery_deficit = 1
admissible_candidate_count = 1
top_admissible_candidate = RIC1
```

Local ridge tracing:

```text
status = ridge_candidates_cover_deficit
admissible_candidate_count = 1
```

Junction-aware arrangement:

```text
status = single_bend_arrangement_candidate
best_hypothesis = H1_locked_0_plus_1_new
selected = JBL7 / F1-F20
conflicts = none
```

Selected line instance:

```text
JBL7 F1-F20
atom_count = 11
proposal_atom_count = 32
path_atom_count = 16
span_mm = 46.048976
median_line_residual_mm = 2.132082
axis_delta_deg = 4.231464
mean_normal_arc_cost = 0.000663
touch_counts = F1:35, F20:139
```

## Visual Verdict

The 1080p close-up is promising:

- the selected `F1-F20` line core sits on a visible diagonal bend/ridge zone;
- it is compact compared with the broader yellow residual ridge support;
- it has no selected-candidate overlap conflicts;
- it appears to be a plausible recovery for the one missing visible bend in the partial scan.

This is still diagnostic-only:

- no per-region manual OB labels exist for this scan;
- the target is visible observed bends for a partial scan, not whole-part total;
- the candidate must be validated against the existing four raw owned regions before exact promotion;
- this line-instance object is not wired into F1 solver output yet.

## Decision

Keep as a promising one-missing-bend recovery candidate.

This is the first cross-scan case where the junction-aware line-instance diagnostic appears useful beyond `49024000`.

Next implementation step should be a small aggregator that compares:

```text
current raw owned regions + selected non-conflicting junction line candidates
```

against the visible-observed target, while blocking:

```text
conflicted candidates like 48991006
overfit/chord candidates
whole-part promotion on partial scans
```

## Artifact Paths

```text
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/47959001_junction_arrangement_visual_check/junction_arrangement_overview_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/47959001_junction_arrangement_visual_check/junction_arrangement_closeup_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/47959001_junction_arrangement_visual_check/junction_arrangement_render_sidecar.json
```

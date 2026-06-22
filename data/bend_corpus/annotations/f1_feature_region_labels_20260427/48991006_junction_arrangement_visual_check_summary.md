# `48991006` Junction Arrangement Visual Check

Date: 2026-04-29

## Inputs

- `48991006_feature_region_labels.json`
- `48991006_feature_region_summary.json`
- `48991006_residual_interface_support_candidates.json`
- `48991006_local_ridge_candidates.json`
- `48991006_junction_bend_arrangement.json`
- Source decomposition: `data/bend_corpus/evaluation/f1_visual_qa_after_debug_alias_patch_20260426/known/02_48991006_48991006_A_4/decomposition_result.json`

## Result

The local junction-aware arrangement diagnostic selected:

```text
H3_locked_plus_2_new
```

Selected candidates:

```text
JBL25: F32-F8, locked accepted OB4
JBL2:  F1-F5, proposed line instance
JBL19: F31-F5, proposed line instance
```

The result is conflicted:

```text
status = two_bend_arrangement_conflicted
JBL2/JBL19 atom IoU = 0.636364
```

## Visual Verdict

The 1080p close-up confirms the conflict:

- the locked accepted `OB4` zone remains isolated at the top edge;
- the two proposed new line instances sit mostly on the same lower-edge support;
- yellow junction/ridge atoms remain broad along that same lower edge;
- this does not separate into distinct conventional bend-line objects;
- it matches the earlier manual note that this part did not process well and has same-zone duplicate behavior.

## Decision

Reject as a promotion candidate.

This is still useful as a fail-safe regression case:

- the diagnostic must not promote conflicted two-line arrangements;
- high overlap between selected line instances should block exact-count promotion;
- `48991006` should remain a poor-scan / duplicate-support blocker until either the scan payload improves or a stronger skeleton/path model separates independent support.

## Artifact Paths

```text
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/48991006_junction_arrangement_visual_check/junction_arrangement_overview_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/48991006_junction_arrangement_visual_check/junction_arrangement_closeup_1080p.png
/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/48991006_junction_arrangement_visual_check/junction_arrangement_render_sidecar.json
```

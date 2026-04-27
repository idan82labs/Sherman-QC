# 49024000 Residual / Interface Visual Gate

This records the visual inspection gate for the residual/interface recovery diagnostics.

## Candidate Overlay

ICloud folder:

`/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/49024000_residual_interface_visual_check`

Finding:

- `RIC1`-`RIC4` all sit in the same broad corner/transition area.
- The candidate supports overlap heavily and do not form two distinct missing bend supports.
- This matches `status=ambiguous_overlapping_candidate_pool`.

## Sub-Cluster Overlay

ICloud folder:

`/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/49024000_residual_interface_subcluster_visual_check`

Finding:

- Sub-clustering produced one plausible broad cluster: `cluster_count=1`, `plausible_cluster_count=1`.
- The 1080p overlay still shows one same-zone patch, not two separated bend lines.
- This matches `status=partial_subcluster_signal`.

## Decision

Do not promote residual/interface candidates or sub-clusters into F1 count logic for `49024000`.

The next verifiable method must produce two separated support overlays on actual bend lines, with low overlap and no stealing from the manually accepted `OB2` support.

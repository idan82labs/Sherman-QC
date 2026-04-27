# 49024000 Residual / Interface Candidate Visual Check

Generated 1080p visual overlays in iCloud:

`/Users/idant/Library/Mobile Documents/com~apple~CloudDocs/Sherman QC/Status for Idan to test/feature_region_labels_20260427/49024000_residual_interface_visual_check`

## Visual Finding

- The residual/interface diagnostic is finding real geometric signal near the part corner/transition zone.
- The current candidates `RIC1`-`RIC4` are not visually separated into two missing conventional bend objects.
- They overlap heavily and sit around the same broad support area, matching the diagnostic status `ambiguous_overlapping_candidate_pool`.
- Therefore this diagnostic is not promotion-safe for exact `3` on `49024000`.

## Decision

Do not promote these candidates into accepted F1 count logic.

The next implementation should first add corridor-family sub-clustering / local support partitioning so the algorithm can prove whether the residual/interface support contains two distinct bend objects or one broad false support.

## Verifiable Next Gate

A future candidate-support tranche is only acceptable if the 1080p overlays show:

- two visually separated candidate supports for the two missing conventional bends;
- low overlap between candidates;
- candidate centers and support spans sitting on actual bend lines, not broad flange/corner patches;
- no candidate stealing the manually accepted `OB2` support.

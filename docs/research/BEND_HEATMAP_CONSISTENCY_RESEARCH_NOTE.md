# Bend Heatmap Consistency Research Note

## Thesis
A local bend verdict and the local deviation field should usually correlate on full, well-covered folded parts. They do not correlate reliably enough on partial scans, shallow bends, weakly observed bends, or rolled/hybrid families to be used as a primary detector.

## Practical Position
The heatmap consistency layer is a corroboration/contradiction signal, not a replacement for bend metrology.

## Circularity Caveat
This validator reuses the same signed-distance field that drives the whole-part heatmap. That is acceptable only if:
- it stays advisory in v1
- measurement methods are stratified by dependence on deviations
- `NON_INDEPENDENT` results are never described as independent validation

## Relevant Literature Themes
- industrial scan-vs-CAD inspection commonly uses point-to-surface deviation fields as a global conformity signal
- bend/feature metrology is a local geometric problem, not solved by global heatmaps alone
- partial overlap, occlusion, and anisotropic sampling make local evidence quality highly non-uniform

## Working Hypothesis
- Full scans with observed folded bends: local bend verdict and local heatmap should agree often enough to flag contradictions usefully
- Partial scans: abstention will be common and is expected
- Rolled/hybrid families need separate treatment and are excluded from v1 scorecards

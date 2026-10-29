# V68 BVH-indexed early NCMM: terminal retirement

## Result

V68 moved V67's normal-cycle mode-match decision to the duplicate-rich V16
thresholded candidate stream and added a deterministic atom-AABB BVH. The BVH
changed only access cost; all geometry and decision constants remained those
of V67.

The indexing improvement worked:

- 48 real early-stage candidates across four already exposed LiDAR cases;
- 337 candidate pairs;
- 44 geometry-eligible pairs;
- 3,624,918 exact atom clip visits, below the 8,000,000 cap;
- per-case wall times 27.4--45.7 seconds, below V68's 60-second primary cap;
- zero aborts.

The detector mechanism did not:

- 32 eligible pairs failed `NO_BRIDGE`;
- 5 failed `MULTI_MODE`;
- 4 failed `OPPOSING_TURN`;
- 3 failed `MODE_OUTSIDE_BRIDGE`;
- zero pairs proposed suppression;
- zero candidate sets changed.

V68 is therefore terminally retired for zero actionability. Moving the same
criterion earlier and making it computationally viable did not change its
real-domain conclusion.

## Visual interpretation

This aligns with the native real-part overlays: the close early candidates are
usually responses to distinct visible folds, opposing walls, or curvature
outside the narrow inter-candidate bridge. The synthetic positives looked
different because they were complete, uniformly sampled, rectilinear objects
with idealized bridge curvature. Those synthetic parts remain quarantined and
did not participate in this result.

## Next optimization direction

Do not loosen NCMM or generate more CAD-like synthetic positives. The next
cycle must target candidate validity rather than duplicate ownership: reject
source lines that do not coincide with a persistent, visually continuous real
crease on the scan, while retaining separate nearby, rounded, partial, return,
and opposing folds. It must be developed and visually audited on real-scan
resampling or newly scanned physical parts.

Machine evidence:
`output/research/f1-bvh-indexed-early-ncmm-v68-real-actionability-v1-20260815/summary.json`.

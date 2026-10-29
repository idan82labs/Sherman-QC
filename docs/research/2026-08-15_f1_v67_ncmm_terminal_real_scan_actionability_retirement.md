# V67 NCMM: terminal real-scan actionability retirement

## Decision

Retire V67 `SIMILARITY_INVARIANT_NORMAL_CYCLE_MODE_MATCH_NMS_V1` before any
fresh-lane output commit or pseudo-truth opening.

## Why

After the source-free implementation gate passed with geometric erratum E1,
V67 was run source-only on four already exposed real LiDAR meshes solely as a
fast morphology-matched actionability reconnaissance. These cases are not
fresh and are not claimed as generalization evidence.

Across the four real cases:

- five pairs entered the normal-cycle profile domain;
- three were on one case and two on another;
- all five failed `NO_BRIDGE`;
- zero disjoint suppressions were proposed;
- zero candidate sets changed;
- no labels, CAD, expected counts, gold, or pseudo-truth were opened.

The mandatory real actionability gate requires at least two proposed
suppressions and at least one changed case. V67 therefore cannot reach its own
execution gate on the available real candidate representation.

## Visual consequence

There is no changed overlay to approve: V67 output is byte-for-byte the same
candidate set as its input in all four cases. Rendering a synthetic positive
would be misleading because the generated synthetic morphology has already
been visually rejected as unlike the corpus and false-positive-prone.

## Interpretation

This is a useful wall hit. The normal-cycle measure is deterministic,
similarity-invariant, topology-aware, and conservative, but it is attached too
late: the available V52 candidate artifact has already separated or removed
the duplicate responses that V67 was designed to adjudicate. On the real
parts, the remaining close pairs do not place a dominant curvature mode in the
inter-candidate bridge.

The next optimization cycle should move the same-fold ownership decision to an
earlier candidate stage where duplicate responses still coexist, or improve
the candidate representation itself. It should not loosen V67 thresholds or
rescue V67 with the quarantined synthetic parts.

Machine-readable evidence:
`output/research/f1-ncmm-v67-real-scan-actionability-recon-v1-20260815/summary.json`.

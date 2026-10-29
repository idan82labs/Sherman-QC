# F1 Synthetic LiDAR Stress V2 — V53 ALBC Cycle

Date: 2026-08-15  
Evidence class: engineering-only, non-confirmatory  
Token: `SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY`

## Outcome

`V53_ADJACENCY_LATTICE_BANK_CONSENSUS_V1` is terminally retired.

The frozen append-only selector executed once over all 30 scans and once as a deterministic replay. It emitted no additions. The V52 result therefore remains the retained source-only synthetic prototype.

## Causal result

After the frozen V52 representation test, only one candidate cluster in the complete 30-scan pool contained both `scan_mesh_adjacency` and `plane_lattice` provenance while remaining unrepresented. That cluster was S029/LO16.

Its route axes agreed well:

- undirected axis angle: 0.556 degrees;
- infinite-axis transverse separation: approximately 0.536 mm;
- shorter-route common-span overlap: approximately 0.940.

However, the common interval was only 103.398 mm over a 261.323 mm bounding-box scale, or 0.3957 B. The preregistered V53 requirement was at least 0.50 B. It therefore abstained at the geometry gate before bank reconstruction. No threshold was changed and no rerun was made.

This rules out the frozen adjacency/lattice bank-consensus mechanism as a sufficiently broad next selector layer on this corpus. It does not rule out source-only ownership methods in general.

## Execution integrity

- 30/30 primary scans committed.
- 30/30 replay scans committed.
- 30/30 primary/replay semantic outputs were byte-identical.
- Every V52 prediction prefix hash was unchanged in primary and replay.
- No scan-level exception or abstention occurred.
- P95 per-case wall time: 0.0431 seconds.
- Peak process RSS: 395,526,144 bytes, below 16 GiB.
- No source points, vertices, faces, or support bitsets were serialized.
- Focused regression suite: 7/7 passing.

The preexecution lock is at `output/research/f1-synthetic-lidar-stress-v2-20260814/v53-adjacency-lattice-bank-consensus-v1/preexecution_lock.json` and the execution manifest is at `output/research/f1-synthetic-lidar-stress-v2-20260814/v53-adjacency-lattice-bank-consensus-v1/execution_manifest.json`.

## Visual audit

Ten mandatory source-only multiview sheets were rendered at 5188 pixels wide and reviewed at native resolution:

- all six exact V52 cases: S005, S006, S007, S013, S026, S027;
- required difficult cases: S011, S016, S024, S029.

The six exact cases remain visually unchanged. S011, S016, S024, and S029 retain their inherited V52 weaknesses, but V53 introduced no diagonal, cross-panel, hole-rim, bridge, free-edge, seam, or floating axis. In particular, it did not reinforce the existing S016 diagonals or the existing S029 short floating axis.

The sealed visual record is `output/research/f1-synthetic-lidar-stress-v2-20260814/v53-adjacency-lattice-bank-consensus-v1/source_only_visual_audit.json`.

## Score and retention ruling

V53 exactly reproduces the V52 score:

- TP: 52;
- FP: 27;
- FN: 48;
- precision: 0.658228;
- recall: 0.520000;
- F1: 0.581006;
- exact-count cases: 6/30;
- minimum family recall: 0.0.

The positive retention gate required TP at least 60, FP exactly 27, FN at most 40, F1 at least 0.641711, exact cases at least 7/30, gains across at least three topology families, a paired-capture family gain, and one fewer zero-recall family. V53 satisfies none of the gain conditions and is therefore retired without rescue.

The immutable score is `output/research/f1-synthetic-lidar-stress-v2-20260814/v53-adjacency-lattice-bank-consensus-v1/synthetic_score.json`.

## Next research implication

The remaining gap is not explained by lack of a second long adjacency/lattice consensus layer. The next bounded action should target either:

1. candidate-birth gaps in the zero-recall families; or
2. a different source-only ownership certificate that applies to the already-present non-cylindrical candidate population.

V52 remains frozen. V53 may not be rerun, relaxed, or rescued.

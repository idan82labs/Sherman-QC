# Next Step: 23553000 Recall Recovery

## Why this is next
Current best trusted benchmark:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260330_sequence_continuity_v3/summary.json`
- structured total MAE: `1.515`
- structured partial total MAE: `3.133`

The next largest remaining error mass is concentrated in part family `23553000`, which is undercounted even on the full scan.

## Current profiling
Across `23553000` scans:
- consistently recovered: `CAD_B1`, `CAD_B2`, `CAD_B3`, `CAD_B6`, `CAD_B7`
- full-only recovered: `CAD_B4`, `CAD_B5`, `CAD_B9`, `CAD_B10`
- unstable / contradictory: `CAD_B8`, `CAD_B13`
- never recovered: `CAD_B11`, `CAD_B12`, `CAD_B14`, `CAD_B15`, `CAD_B16`

This means the next ROI is not evaluator continuity. It is detector recall / target assignment on a specific bend family.

## Execution plan
1. Instrument the global-to-CAD assignment path for `23553000`.
   - Record candidate count, candidate kind, null score, and winning reason for `CAD_B4`, `CAD_B5`, `CAD_B9`, `CAD_B10`.
   - Goal: decide whether the bends are missing at detection time or lost during assignment.
2. Compare detector candidates between:
   - `23553000_B_SCAN_2.ply`
   - `23553000_B_FULL_SCAN 1.ply`
   - Goal: identify whether full-only bends are absent on the partial or just unmatched.
3. If candidates exist but lose assignment:
   - tune local/null competition only for this family slice.
   - prefer countable bend targets over null when candidate geometry is plausible and target neighborhood is consistent.
4. If candidates do not exist:
   - improve detector recall for short symmetric bends.
   - target the missing set: `CAD_B4`, `CAD_B5`, `CAD_B9`, `CAD_B10`.
5. Rebenchmark against:
   - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`
   - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260329_observability_v1b/summary.json`
   - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260330_sequence_continuity_v3/summary.json`

## Success target
- recover at least 2 additional `23553000` bends on the full scan without creating new hybrid/partial false positives
- stretch: reduce structured total MAE from `1.515` to `< 1.455`

# Next Step: 23553000 Partial Recall Recovery

## Current state
Best structured benchmark:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260330_partial_to_full_retention_v1/summary.json`
- structured total MAE: `1.485`
- structured completed MAE: `0.238`
- structured partial total MAE: `3.133`
- structured partial completed MAE: `1.0`
- structured full exact completion rate: `0.895`

Remaining nonzero structured completed-bend error is concentrated entirely in `23553000`:
- `23553000_B_SCAN_1.ply`: `+1`
- `23553000_B_SCAN_2.ply`: `-2`
- `23553000_B_FULL_SCAN 1.ply`: `-2` before retention, now `-2` total / completed after retention win became `-2`

## What we now know
1. The evaluator-side retention win was real and safe.
2. The medium-full local-refinement gate changed raw reporting but did not improve the structured benchmark.
3. `23553000_B_SCAN_2.ply` is still not recoverable through the current local-refinement path:
   - direct probe counts: global/local/merged = `3 / 4 / 6`
4. So the next structured-MAE lever is detector recall on the `23553000` partial family, not more evaluator carry logic.

## Hypothesis
The missing stage-2 bends are short/local ear bends that are either:
- not emitted by the scan detector at all, or
- emitted only under fallback settings and then lost before stable CAD assignment

## Execution plan
1. Instrument raw scan detections on `23553000_B_SCAN_2.ply` and `23553000_B_FULL_SCAN 1.ply` side by side.
2. Record for each raw detected bend:
   - centroid
   - bend line length
   - target angle
   - confidence
   - adjacency / local support if available
3. Compare the short-bend population between partial and full scans.
4. Patch the scan fallback path narrowly for short symmetric bends:
   - lower `min_bend_points` only in fallback for this regime, or
   - relax the spatial clustering gate for short, high-curvature regions, or
   - add a second fallback pass specialized for short bends on medium partial scans.
5. Rebenchmark against:
   - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`
   - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260329_observability_v1b/summary.json`
   - `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260330_partial_to_full_retention_v1/summary.json`

## Success target
- Recover at least one additional structured completed bend on `23553000_B_SCAN_2.ply`
- Hold the current retention win on `23553000_B_FULL_SCAN 1.ply`
- Do not create new false positives on `28870000_A_SCAN_1.ply`
- Stretch target: reduce structured completed MAE from `0.238` to `< 0.20`

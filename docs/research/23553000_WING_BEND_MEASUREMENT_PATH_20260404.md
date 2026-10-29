# 23553000 Wing Bend Measurement Path Audit

Date: 2026-04-04

## Scope

Investigate the short wing bends on `23553000` that are called out as `120°` in the drawing, but are currently reported around `124°–128°` in the benchmark JSON.

Reference drawing evidence:
- `/tmp/23553000_pdf/page-2.png`

Reference benchmark file:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/tmp_23553000_surface_recovery_20260404/23553000/23553000_B_FULL_SCAN_1.ply--3e0767d8.json`

## Established facts

1. This is not primarily a scan-quality problem.
   - Aligned local support near the short wing bend lines is strong.
   - The ScanBox data is sufficient for bend-level reasoning in those zones.

2. CAD and drawing agree at the bend-family level.
   - The small wing bends are a `120°` family in the PDF.
   - The CAD-extracted short wing bends also target `120°`.

3. The current benchmark JSON still reports these problematic raw results:
   - `CAD_B6`: `126.87°` via `GLOBAL_DETECTION`
   - `CAD_B7`: `128.59°` via `GLOBAL_DETECTION`
   - `CAD_B9`: `124.66°` via `CAD_LOCAL_NEIGHBORHOOD/profile_section`
   - `CAD_B10`: `126.32°` via `CAD_LOCAL_NEIGHBORHOOD/profile_section`
   - `CAD_B11`: `128.67°` via `CAD_LOCAL_NEIGHBORHOOD/profile_section`
   - `CAD_B14`: `NOT_DETECTED`
   - `CAD_B15`: `NOT_DETECTED`

## New result from direct dense local refinement

Running `refine_dense_scan_bends_via_alignment(...)` directly on:
- CAD: `/Users/idant/Downloads/23553000_B.stp`
- scan: `/Users/idant/Downloads/23553000_B_FULL_SCAN 1.ply`

with the current code produces a materially different local report:

- `CAD_B6`: `111.42°`, `FAIL`, `surface_classification`
- `CAD_B7`: `118.11°`, `WARNING`, `surface_classification`
- `CAD_B9`: `111.29°`, `FAIL`, `surface_classification`
- `CAD_B10`: `122.39°`, `FAIL`, `surface_classification`
- `CAD_B11`: `119.07°`, `PASS`, `surface_classification`
- `CAD_B14`: `90.73°`, `PASS`, `surface_classification`
- `CAD_B15`: `117.87°`, `FAIL`, `surface_classification`

Local refinement warning tail:
- `Recovered-face surface classification replaced local measurement for CAD_B11 (2.58deg -> 0.93deg).`
- `Recovered-face surface classification replaced local measurement for CAD_B15 (3.02deg -> 2.13deg).`

## Interpretation

This proves two important things:

1. The measurement path can already get much closer to the `120°` truth family on real scan data.
2. The persistent `124°–128°` values in the benchmark JSON are not a direct property of the scan.

## Most likely root cause

The remaining mismatch is in the CAD baseline / report selection path, not the raw scan.

Evidence:
- direct local refinement runs on a selected baseline of `17` bends
- benchmark evaluation still applies `legacy_mesh_expected_recovery` and reduces CAD to an expected baseline of `16`
- the full benchmark JSON still serializes the older `profile_section` / `detected_bend_match` values

So the unstable layer is now:
- selected-CAD baseline recovery (`17 -> 16`)
- bend-id / wing-bend identity mapping under that recovery
- merge/report selection between primary and local evidence

## Engineering conclusion

Before trying to "calibrate" the `124°–128°` values down to `120°`, we need to stabilize the `23553000` CAD baseline and wing-bend identity mapping.

Otherwise we are tuning on top of a moving target.

## Next correct step

1. Freeze the `23553000` wing-bend inventory against the drawing.
2. Audit the `legacy_mesh_expected_recovery` reduction from `17` bends to `16`.
3. Ensure the short wing bends keep stable IDs through:
   - selected baseline construction
   - dense local refinement
   - merge/report selection
4. Only then tune the final measurement choice for those bends.

## 2026-04-04 Follow-up: Clean-Branch Evaluator and Narrow Angle Scoring

Two concrete issues were fixed on the clean QC branch:

1. `evaluate_bend_case.py` was originally importing from the dirty enterprise worktree.
   - Fixed by resolving `ROOT` from the script location and adding the clean repo root to `sys.path`.
2. local assignment candidates were still using source CAD angle instead of the candidate's measured angle.
   - Fixed in `_build_local_assignment_candidates(...)`.

Then a narrower `surface_classification` selection policy was added:
- when multiple offset/locality variants are within a tight quality band, prefer the result closer to the CAD angle instead of blindly taking the single highest-quality variant

### Full-scan wing-bend state after the patch

From:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_case_debug_20260404.json`

Short `120°` family now measures:
- `CAD_B6`: `127.88°` via `GLOBAL_DETECTION` with obviously wrong line length (`133.699 mm` vs `53.748 mm`)
- `CAD_B7`: `121.12°` via `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B9`: `122.39°` via `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B10`: `119.07°` via `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B11`: `122.39°` via `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B14`: `117.87°` via `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B15`: `118.46°` via `CAD_LOCAL_NEIGHBORHOOD`

### Interpretation

This narrows the remaining problem substantially:
- `CAD_B6` is still a bad global match, not a scan-completeness issue
- `CAD_B9` and `CAD_B11` are still local measurement errors around `122.39°`
- `CAD_B14` is still low at `117.87°`
- `CAD_B7`, `CAD_B10`, and `CAD_B15` are now close enough to the `120°` family to treat the measurement path as directionally correct

### Next targeted fix

Do not add another broad heuristic.

The next surgical candidates are:
1. override or suppress the bad `GLOBAL_DETECTION` match on `CAD_B6` when line-length mismatch is extreme and a local candidate exists
2. inspect why `surface_classification` still stabilizes at `122.39°` on `CAD_B9` and `CAD_B11`
3. keep `28870000` as the guard family while changing only short `120°` bend logic

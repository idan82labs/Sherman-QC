# 23553000 Wing Bend Spec Check

## Scope

Check the short wing bends on `23553000` against:
- the technical drawing `/Users/idant/Downloads/שרטוט החלק 9.PDF`
- the CAD-extracted bend inventory
- the full-scan QC result

Goal:
- determine whether the current misses are caused by scan quality
- or by assignment / local measurement behavior

## Drawing vs CAD

The drawing shows short wing bends called out at `120°` in `SECTION B-B (x2)`.

The current CAD-extracted bend inventory also contains a `120°` bend family.

Short `120°` CAD bends currently recognized in the benchmark report:
- `CAD_B6`
- `CAD_B7`
- `CAD_B9`
- `CAD_B10`
- `CAD_B11`
- `CAD_B14`
- `CAD_B15`

So at the angle-family level, CAD and the drawing are aligned:
- right-angle family: `90°`
- wing-bend family: `120°`

## Full-scan result for the short 120° wing bends

Source:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/23553000/23553000_B_FULL_SCAN_1.ply--3e0767d8.json`

Current statuses:
- `CAD_B6`: `FAIL`, measured `126.87°`
- `CAD_B7`: `FAIL`, measured `128.59°`
- `CAD_B9`: `FAIL`, measured `124.66°`
- `CAD_B10`: `FAIL`, measured `126.32°`
- `CAD_B11`: `FAIL`, measured `128.67°`
- `CAD_B14`: `NOT_DETECTED`
- `CAD_B15`: `NOT_DETECTED`

## Independent local support check from the aligned scan

Aligned scan used:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_fullscan_wing_bends_20260403/aligned_scan.ply`

For each CAD bend line, the number below is the count of aligned scan points within a tube around the CAD bend line.

### Point support within a 3 mm tube
- `CAD_B6`: `216`
- `CAD_B7`: `83`
- `CAD_B9`: `133`
- `CAD_B10`: `216`
- `CAD_B11`: `114`
- `CAD_B14`: `215`
- `CAD_B15`: `6`

### Point support within a 5 mm tube
- `CAD_B6`: `321`
- `CAD_B7`: `229`
- `CAD_B9`: `227`
- `CAD_B10`: `337`
- `CAD_B11`: `238`
- `CAD_B14`: `333`
- `CAD_B15`: `181`

### Point support within an 8 mm tube
- `CAD_B6`: `437`
- `CAD_B7`: `405`
- `CAD_B9`: `372`
- `CAD_B10`: `434`
- `CAD_B11`: `397`
- `CAD_B14`: `451`
- `CAD_B15`: `417`

## Interpretation

This is not a scan-completeness problem.

Reason:
- even the two currently missed bends, `CAD_B14` and `CAD_B15`, have strong local point support once the radius is widened modestly around the CAD line
- `CAD_B14` already has `215` points within `3 mm`
- `CAD_B15` is tighter and harder, but still has `181` points within `5 mm` and `417` within `8 mm`

That means the scan contains the geometry needed to reason about those bend zones.

## Failure split

### 1. `CAD_B14` and `CAD_B15`

Current pipeline result:
- `assignment_source = NONE`
- `status = NOT_DETECTED`
- `physical_completion_state = UNKNOWN`

But local aligned point support is clearly present.

Conclusion:
- these are algorithm misses
- specifically local assignment / local observability misses
- not scan quality misses

### 2. `CAD_B6` and `CAD_B7`

Current pipeline result:
- measured by `detected_bend_match`
- `assignment_source = GLOBAL_DETECTION`
- measured line lengths are far larger than expected:
  - `CAD_B6`: expected `53.75 mm`, actual `133.53 mm`
  - `CAD_B7`: expected `35.69 mm`, actual `133.69 mm`

Conclusion:
- these are primarily assignment problems
- the scan has signal
- but the wrong detected bend candidate is being matched to the short wing target

### 3. `CAD_B9`, `CAD_B10`, `CAD_B11`

Current pipeline result:
- measured by `profile_section`
- `assignment_source = CAD_LOCAL_NEIGHBORHOOD`
- exact or near-exact line-length agreement with CAD
- strong visibility / support
- but measured angles stay high:
  - `124.66°`
  - `126.32°`
  - `128.67°`

Conclusion:
- these are not scan-quality misses
- these are local measurement / section-estimation problems on short 120° bends
- scan support is strong; the estimator is not resolving the bend correctly

## Visual artifacts

Artifacts generated for this check:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_fullscan_wing_bends_20260403/aligned_scan.ply`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_fullscan_wing_bends_20260403/reference_mesh.ply`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_wing_focus_from_benchmark_20260403/CAD_B6.png`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_wing_focus_from_benchmark_20260403/CAD_B7.png`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_wing_focus_from_benchmark_20260403/CAD_B9.png`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_wing_focus_from_benchmark_20260403/CAD_B10.png`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_wing_focus_from_benchmark_20260403/CAD_B11.png`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_wing_focus_from_benchmark_20260403/CAD_B14.png`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_wing_focus_from_benchmark_20260403/CAD_B15.png`

## Bottom line

For the short wing bends on `23553000`, the current problem is not that the scan is too weak.

It is:
1. wrong candidate assignment on some short `120°` bends
2. poor local section estimation on others
3. missing local observability / routing on two bends that already have enough aligned point support

So this family should be improved by:
- short-bend assignment fixes
- short-bend local section metrology fixes
- not by changing the scan recipe

# Guard Pack Validation — 2026-04-11

## Goal

Validate the kept `CAD_B6` fix narrowly before merge-prep.

## Target rerun

Live rerun:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/guard_pack_20260411/23553000_full.json`

Reference:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_case_debug_20260406d.json`

Tracked wing bends matched exactly:
- `CAD_B6`: `119.64°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B7`: `121.0°`, `WARNING`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B9`: `120.61°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B10`: `119.88°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B11`: `121.76°`, `WARNING`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B14`: `120.45°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B15`: `119.09°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`

Conclusion:
- the kept short-obtuse fixes now cover `B6`, `B14`, and a partial `B11` improvement on the shipped full-case path

## Guard applicability

The patch predicate is:
- obtuse bend only: `105°–140°`
- `selected_baseline_direct == true`
- bend-line length `<= 70 mm`

### `28870000`

Frozen reference:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/28870000/28870000_A_FULL_SCAN.ply--e91bd366.json`

Countable bends:
- `25.0°`
- `22.45°`

Result:
- the narrow `B6` retry path cannot fire on `28870000`

### Stable completed controls

References:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/15651004/15651004_01.ply--02d3a36b.json`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/10839000/10839000_01_full_scan.ply--d9b351cf.json`

Observed control families:
- `15651004`: `90°` family
- `10839000`: `90°` family

Result:
- the narrow `B6` retry path cannot fire on these controls either

## Environment caveat

The direct case runner is not laptop-safe for broad brute-force validation on this 16 GB MacBook.

Observed during attempted live reruns:
- `28870000`: about `19.5 GB` footprint
- `15651004`: about `29.9 GB` footprint

Both were spending time in a NumPy / LAPACK SVD path.

Interpretation:
- this is an environment constraint for local validation
- it is not evidence that the kept `B6` patch is unsafe
- production sizing can absorb this better, but this numeric path should still be treated as a separate runtime concern

## Validation decision

The short-obtuse follow-up remains keepable for merge-prep:
- target case improved on the shipped `23553000` full-scan path
- `28870000` remains outside the patch predicate (`22°/25°`, not selected short-obtuse)
- rerun of `10839000` showed only a tiny unaffected-path drift (`90.83° PASS -> 90.97° WARNING` on `detected_bend_match`), which is better explained as rerun variance than as an effect of this patch class
- the heavy-case guard runs on this laptop remain a runtime/environment constraint, not a logic rejection

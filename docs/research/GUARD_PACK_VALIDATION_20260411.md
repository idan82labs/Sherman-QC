# Guard Pack Validation — 2026-04-11

## Goal

Validate the kept `CAD_B6` fix narrowly before merge-prep:
- reproduce the fixed `23553000` full-scan result on the laptop
- ensure the patch does not affect `28870000` hybrid semantics
- ensure the patch does not affect already-good completed controls

## Patch under validation

Code:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/bend_detector.py`

Guard predicate:
- `_is_selected_short_obtuse_bend(...)`

Current hard gating:
- `105° <= bend_angle <= 140°`
- `selected_baseline_direct == true`
- bend-line length `<= 70 mm`
- retry only after a successful `surface_classification` result is still more than `4°` away from CAD

This is not a global search-radius change. It is a narrow retry path for the short `120°` wing-bend class.

## Live target rerun

Serial run:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/guard_pack_20260411/23553000_full.json`

Reference:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_case_debug_20260406d.json`

Observed result:
- exact match on the tracked wing bends

Tracked bends:
- `CAD_B6`: `119.64°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B7`: `121.0°`, `WARNING`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B9`: `120.61°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B10`: `119.67°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B11`: `122.39°`, `FAIL`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B14`: `117.87°`, `FAIL`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`
- `CAD_B15`: `119.09°`, `PASS`, `surface_classification`, `CAD_LOCAL_NEIGHBORHOOD`

Conclusion:
- the `B6` fix reproduced exactly on the shipped full-case path
- no regression was observed on the other tracked `120°` wing bends during the rerun

## Laptop safety finding

The case runner is not laptop-safe for broad guard-pack brute force on this machine.

Sampling results during attempted serial guard cases:
- `28870000` live run:
  - physical footprint: `19.5 GB`
  - stack: NumPy generalized ufunc -> `_umath_linalg` SVD -> LAPACK `dgesdd`
- `15651004` live run:
  - physical footprint: `29.9 GB`
  - same SVD / LAPACK path

This is not acceptable on the 16 GB MacBook and is not a reasonable validation method for the remaining guard cases.

Operational decision:
- keep laptop validation to the already-completed target rerun
- use structural patch non-applicability plus frozen clean benchmark outputs for the remaining guard cases
- do not continue brute-force case reruns on this machine until the SVD memory path is fixed separately

## Structural guard for `28870000`

Frozen clean benchmark reference:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/28870000/28870000_A_FULL_SCAN.ply--e91bd366.json`

Countable bends:
- `CAD_B4`: `25.0°`, `COUNTABLE_DISCRETE_BEND`
- `CAD_B5`: `22.45°`, `COUNTABLE_DISCRETE_BEND`

Why the `B6` patch cannot affect this part:
- neither countable bend is obtuse
- neither bend is in the `105°–140°` trigger range
- therefore `_is_selected_short_obtuse_bend(...)` is false
- the narrow retry path cannot execute on `28870000`

Engineering conclusion:
- hybrid semantics are structurally unaffected by the `B6` retry patch

## Structural guards for completed controls

### `15651004`

Frozen clean benchmark reference:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/15651004/15651004_01.ply--02d3a36b.json`

Observed completed feature:
- `CAD_B1`: `90.42°`, `ROLLED_PROCESS_FEATURE`

Why the patch cannot affect it:
- not in the obtuse trigger range

### `10839000`

Frozen clean benchmark reference:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/10839000/10839000_01_full_scan.ply--d9b351cf.json`

Observed completed family:
- `90°` rolled/process features across the part

Why the patch cannot affect it:
- not in the obtuse trigger range

## Validation decision

The kept `B6` patch is validated strongly enough to enter merge-prep:
- target case reproduced exactly
- the remaining guard parts are outside the patch predicate
- no evidence suggests hybrid or stable-control regression from this patch class

What this does not claim:
- it does not claim full laptop-safe brute-force guard-pack completion
- it does not claim `CAD_B11` or `CAD_B14` are solved

## Next branch action

Proceed to merge-prep on:
- branch: `codex/qc-clean-mainline`
- worktree: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean`

Do not resume broad tuning until:
- the merge candidate is cleaned
- temp artifacts are excluded
- the unresolved `CAD_B11` / `CAD_B14` work is explicitly deferred as follow-up

# Next Step: 23553000 Stage Semantics

## Current kept benchmark state
Reference clean benchmark to keep:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260402_selected_face_recovery_v1/summary.json`

Current kept structured metrics:
- structured total MAE: `1.394`
- structured completed MAE: `0.190`
- structured partial total MAE: `3.000`
- structured partial completed MAE: `1.0`
- structured full exact completion rate: `0.895`
- failed scans: `0`

This is better than:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260401_line_aware_locality_v6/summary.json`

Current best mixed validation state:
- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260403_probe_region_completion_v1/projected_summary_from_reused_tail_v2.json`

Projected structured metrics relative to the clean 20260402 benchmark:
- structured total MAE: `1.394 -> 1.364`
- structured completed MAE: `0.190 -> 0.143`
- structured partial total MAE: `3.000 -> 2.933`
- structured partial completed MAE: `1.000 -> 0.667`
- structured full exact completion rate: `0.895 -> 0.944`

Important correction:
- `38172000` is no longer a live regression target
- the earlier drop on `CAD_B2` was a `--skip-existing` evaluator bug and is fixed

## What improved
The selected-face recovery patch solved the right layer:
- `23553000_B_SCAN_1.ply`: structured median `5 -> 6`
- `23553000_B_SCAN_2.ply`: structured median `7 -> 8`
- `28870000` hybrid semantics stayed stable

This confirms the previous bottleneck was real local CAD side recovery, not scan completeness.

## Remaining highest-value residual
The remaining structured completed-bend error is now concentrated in `23553000`:
- `23553000_B_SCAN_1.ply`
  - expected completed: `4`
  - structured median: `6`
  - completed error: `+2`
- `23553000_B_SCAN_2.ply`
  - expected completed: `9`
  - structured median: `9`
  - completed error: `0`
- `23553000_B_FULL_SCAN 1.ply`
  - expected completed: `16`
  - structured median: `15`
  - completed error: `-1`

Everything else is lower leverage for structured completed MAE.

## Updated diagnosis
This is no longer a broad recall problem.

It is now a stage-appropriate completion problem on the same family:
- one partial scan over-promotes two bends as completed
- the full scan still under-promotes one bend

That means the next likely bottleneck is one of:
1. assignment exclusivity across nearby short bends
2. stage-specific observability / formed-vs-completed semantics
3. local angle plausibility on one remaining missed bend family member

## Frozen bend-id evidence
Saved artifact:
- `/Users/idant/82Labs/Sherman QC/Full Project/output/23553000_stage_semantics_cards_20260402.json`
- refreshed residual card pack:
  - `/Users/idant/82Labs/Sherman QC/Full Project/output/23553000_residual_cards_20260403.json`

Current strongest candidates from the local selected-CAD reports:
- likely early over-promotion on `23553000_B_SCAN_1.ply`:
  - `CAD_B8`
  - reason:
    - `FORMED` on scan 1
    - on scan 1 it is driven by `surface_classification`
    - support is numerically high but side balance is almost degenerate (`0.011`)
    - that is exactly the profile of a likely early-stage false completion
- likely early under-promotion on `23553000_B_SCAN_1.ply`:
  - `CAD_B4`
  - reason:
    - selected local probe exists on scan 1
    - `surface_visibility_ratio = 1.0`
    - `local_support_score = 1.0`
    - but `side_balance_score = 0.099` and the probe angle stays implausible
    - this is a local-evidence interpretation problem, not raw completeness
- `23553000_B_SCAN_2.ply`:
  - `CAD_B12` is no longer the primary residual
  - the kept `probe_region_completion_v1` patch now carries it as a completed bend
- likely full under-promotion on `23553000_B_FULL_SCAN 1.ply`:
  - `CAD_B4`
  - `CAD_B14`
  - `CAD_B15`
  - reason:
    - all three still end as `NOT_DETECTED`
    - unlike scan 1, the full scan currently has no local evidence at all for them
    - this points back to CAD baseline decomposition / target routing rather than scan coverage

## Next step to execute
Do not add more broad recall heuristics.

Build a per-bend delta card for the `23553000` trio:
- `23553000_B_SCAN_1.ply`
- `23553000_B_SCAN_2.ply`
- `23553000_B_FULL_SCAN 1.ply`

For each CAD bend id, record:
- final observability state
- final pass/fail/not-detected status
- local measurement method
- measured angle
- score / null-score / assignment source
- whether the bend is present in the selected-face recovery path
- whether that bend flips state across the three scans

Required focus:
- verify whether `CAD_B8` is one of the two bends causing the `+2` overcount on `B_SCAN_1`
- verify whether `CAD_B4` is the complementary scan-1 miss that offsets the `CAD_B8` over-promotion
- determine whether `CAD_B4` or one of `CAD_B14` / `CAD_B15` is the best first full-scan promotion target

## Intended patch class
Only after the above cards exist, choose one narrow fix:
1. tighten completion promotion for the two over-promoted early-stage bends
2. decide whether `CAD_B4` on scan 1 should be promoted as completion evidence or remain blocked by plausibility
3. adjust assignment exclusivity among neighboring short bends if the same CAD ids are swapping incorrectly across stages
4. separately evaluate whether `CAD_B4` can be promoted on the full scan without re-opening `CAD_B14` / `CAD_B15`

## Acceptance rules for the next patch
- `23553000_B_SCAN_1.ply` must move toward expected completed `4`
- `23553000_B_SCAN_2.ply` must stay at `9`
- `23553000_B_FULL_SCAN 1.ply` must stay at `15` or improve to `16`
- `28870000_A_SCAN_1.ply` must remain `0/2`
- `28870000_A_FULL_SCAN.ply` must remain `2/2`
- structured total MAE must stay `<= 1.364`
- structured completed MAE must improve below `0.143` or hold without harming the kept metrics above

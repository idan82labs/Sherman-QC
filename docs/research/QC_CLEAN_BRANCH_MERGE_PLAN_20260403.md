# QC Clean Branch Merge Plan

## Current position

The active development branch `codex/enterprise-reorder` is not suitable for direct merge into `main`.

Reasons:
- very large dirty worktree
- unrelated frontend and backend architecture changes mixed with QC work
- temporary capture scripts and broad app-layer additions in the same branch
- difficult to review or present as one coherent change set

## Clean merge path

Use the dedicated clean QC branch created from `main`:
- branch: `codex/qc-clean-mainline`
- worktree: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean`

This branch was validated with:
- targeted QC tests: `122 passed`
- full benchmark:
  - `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/benchmark_stability_core_20260403_qc_clean_v1/summary.json`

Structured benchmark result:
- total MAE: `1.364`
- completed MAE: `0.143`
- partial total MAE: `2.933`
- partial completed MAE: `0.667`
- structured full exact completion rate: `0.895`
- failed scans: `0`

## Minimum file set for the mergeable QC slice

Core code:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/bend_detector.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/bend_inspection_pipeline.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/bend_unary_model.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/cad_bend_extractor.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/feature_detection/bend_detector.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/qc_engine.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/scripts/evaluate_bend_corpus.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/domains/bend/services/runtime_semantics.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/domains/__init__.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/domains/bend/__init__.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/domains/bend/services/__init__.py`

Tests:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/tests/test_bend_detector.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/tests/test_bend_inspection_pipeline.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/tests/test_bend_unary_model.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/tests/test_local_bend_measurement.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/tests/test_evaluate_bend_corpus.py`

## What should not be in the QC PR

Do not mix in:
- enterprise app/router rearchitecture
- unrelated frontend page rewrites
- temporary Playwright or capture scripts
- broad infrastructure folder additions not required by QC
- unrelated corpus cache or evaluation artifacts

## Presentability standard

The QC PR should be reviewable as:
1. bend detection / local metrology improvements
2. evaluator continuity and retention consistency fixes
3. observability/runtime semantics normalization
4. benchmark-backed regression tests

## Current engineering conclusion

The current QC patch set is good enough to merge from the clean branch.

The remaining unresolved problem is not branch hygiene. It is stage-truth certainty on:
- `23553000_B_SCAN_1.ply`

That should be handled as a separate follow-up:
- either obtain bend-id stage labels
- or add a narrowly justified stage-semantics correction after stronger evidence

## Updated confidence after stage audit

Stage audit:
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/23553000_STAGE_TRUTH_AUDIT_20260403.md`

Updated conclusion:
- the unresolved `23553000_B_SCAN_1.ply` residual is now best treated as a truth-mapping problem, not a merge blocker
- the current best-fit stage model explains the user counts exactly:
  - stage 1: `CAD_B1`, `CAD_B2`, `CAD_B3`, `CAD_B5`
  - stage 2 adds: `CAD_B6`, `CAD_B7`, `CAD_B8`, `CAD_B12`, `CAD_B13`
  - final adds: `CAD_B4`, `CAD_B9`, `CAD_B10`, `CAD_B11`, `CAD_B14`, `CAD_B15`, `CAD_B16`
- this makes the current kept patch more trustworthy, not less:
  - the kept patch corrects `CAD_B12` on stage 2
  - it does not depend on the unresolved stage-1 ambiguity

Merge recommendation remains:
- merge from `codex/qc-clean-mainline`
- do not merge from `codex/enterprise-reorder`
- carry `23553000_B_SCAN_1.ply` stage truth as an explicitly documented follow-up

## Updated merge-prep gate after B6 validation

Validation note:
- `/Users/idant/82Labs/Sherman QC/Full Project/docs/research/GUARD_PACK_VALIDATION_20260411.md`

Current conclusion:
- the `CAD_B6` narrow retry patch is validated strongly enough for merge-prep
- the target full-scan rerun reproduced exactly
- `28870000` and the frozen completed controls are outside the patch predicate, so this patch class cannot affect them

Important caveat:
- the direct case runner is not laptop-safe for broad brute-force validation
- attempted serial guard runs for `28870000` and `15651004` entered a NumPy/LAPACK SVD path with `~20–30 GB` physical footprints
- merge-prep should continue, but large local validation runs should not be used again on this machine until that memory path is fixed

Practical branch status:
- `codex/qc-clean-mainline` is the correct merge source
- the branch is ready for cleanup and staging of the QC slice
- unresolved follow-up work should remain explicitly deferred:
  - `CAD_B11`
  - `CAD_B14`
  - broader laptop-safe validation of the heavy case runner

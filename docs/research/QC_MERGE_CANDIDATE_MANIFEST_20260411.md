# QC Merge Candidate Manifest — 2026-04-11

## Merge source

- branch: `codex/qc-clean-mainline`
- worktree: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean`

Do not merge from:
- `codex/enterprise-reorder`

## Merge candidate scope

Core code:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/bend_detector.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/bend_inspection_pipeline.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/bend_unary_model.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/cad_bend_extractor.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/feature_detection/bend_detector.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/backend/qc_engine.py`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/scripts/evaluate_bend_case.py`
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

Essential research:
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_B6_ROOT_CAUSE_20260406.md`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_B14_ROOT_CAUSE_20260411.md`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_B11_FOLLOWUP_20260411.md`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/GUARD_PACK_VALIDATION_20260411.md`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/QC_MERGE_CANDIDATE_MANIFEST_20260411.md`

## Explicitly exclude from the PR

- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/data/bend_corpus/evaluation/`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/`
- `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/tmp_qc_validation_20260403/`
- local one-off debug JSONs and temporary validation folders

## Current merge judgment

Ready for merge-prep:
- `B6` fix is reproduced
- the fix is narrow and evidence-backed
- hybrid guard is structurally unaffected
- stable completed controls are structurally unaffected

Still deferred after merge-prep:
- final `CAD_B11` correction from warning to pass
- wider runtime work on the heavy SVD path

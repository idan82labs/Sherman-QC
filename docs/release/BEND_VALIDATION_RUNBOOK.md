# Bend Validation Runbook

This runbook defines the promotion contract for bend-engine changes.

## Seed Profile

- Deterministic seed: `101`
- Retry seeds: `211, 307`

## Validation Order

1. Fast tests for touched subsystems.
2. `scripts/evaluate_cad_gold_set.py`
3. `scripts/evaluate_zero_shot_bends.py --slice-json data/bend_corpus/evaluation_slices/zero_shot_development.json`
4. `scripts/evaluate_zero_shot_bends.py --slice-json data/bend_corpus/evaluation_slices/zero_shot_holdout.json`
5. `scripts/evaluate_bend_corpus.py`
6. `scripts/compare_bend_corpus_summaries.py`
7. `scripts/analyze_engine_gap_blockers.py`
8. `scripts/analyze_position_holds.py`
9. `scripts/analyze_metrology_holds.py`
10. `scripts/run_bend_corpus_seed_matrix.py` when the change affects ranking, local refinement, stochastic search, ambiguity handling, or zero-shot strip inference.

## Supplemental Zero-Shot Slices

Run these in addition to the permanent order when the touched code affects the named morphology or stage:

- `data/bend_corpus/evaluation_slices/zero_shot_small_tight_bends.json`
  - Required when changing detector support, evidence admission, strip grouping, or observability scoring for compact fixtures with short neighboring bends.
  - Treat this as a required supplemental benchmark for small-bend rescue work even though it is not yet a branch-head hard blocker for unrelated changes.
- `data/bend_corpus/evaluation_slices/zero_shot_known_count_plys.json`
  - Use this as the deterministic-output audit slice for real PLY scans with accepted bend-count truth.
  - Required when changing exact-count promotion, abstention, or selector logic intended to improve deterministic zero-shot output.
  - Read it as a scorecard, not a hard blocker by itself: exact outputs may improve, but label conflicts must be investigated before forcing new exact claims.
- `data/bend_corpus/evaluation_slices/zero_shot_repeated_near_90.json`
  - Use this as the repeated near-90 perturbation benchmark for scans like `10839000`, `49204020_04`, and `ysherman`.
  - Required when changing voxel perturbation handling, detector preprocessing, or repeated-family candidate preservation.
  - Pair it with `scripts/analyze_zero_shot_repeated_near_90.py` and treat any broader range on `ysherman` as a regression even if protected dev/holdout slices stay green.
- `data/bend_corpus/evaluation_slices/zero_shot_partial_total_bends.json`
  - Use this as the partial-scan negative-control audit for scans where only a subset of the part geometry is visible.
  - Required when changing scan-state inference, partial handling, or any exact-count promotion rule that could leak into partial scans.
  - The target on this slice is abstention discipline, not deterministic whole-part total count. Treat any revived exact-count output on partial scans as a regression unless a narrower partial-aware contract is deliberately added.

## Theory-to-Reality Experiment Lane

Use this lane for research-inspired detector/evidence/grouping experiments that should not change runtime behavior by default.

Reference:

- `docs/release/THEORY_TO_REALITY_TEST_PROGRAM.md`
- `scripts/run_zero_shot_theory_experiment.py`

Run order for one isolated family:

1. focused tests for touched code
2. fresh control run on:
   - development
   - holdout
   - known-count PLYs
   - repeated-near-90
   - small-tight-bend slice
   - partial negative-control slice when applicable
3. candidate run on the same slices
4. repeated-near-90 analyzer pass
5. control-vs-candidate comparison
6. keep/reject decision and professor brief generation

Do not promote a theory-experiment family into runtime behavior unless it tightens at least one named hard full scan without regressing protected or supplemental slices.

## Renderable Bend Object Verification

Run these checks when changing scan-only bend rendering, bend cards, or scan overlay interaction:

1. Backend contract smoke:
   - `pytest tests/test_zero_shot_bend_overlay.py -q`
2. Frontend type/build:
   - `npm run typecheck`
   - `npm run build`
3. Live demo smoke:
   - `node frontend/react/scripts/verifyScanOverlayDemo.mjs --base-url http://127.0.0.1:4174`
   - or another explicit local app URL when Vite chooses a different port

The live demo smoke should verify:

- the demo route loads
- bend chip selection updates the selected bend card
- in-view callout selection updates the selected bend card
- overview and selected-bend screenshots are written for visual inspection

## Hard Blockers

- Any CAD gold mismatch.
- Any failed scan in the promoted full-corpus run.
- Any unapproved regression in protected metrics from `data/bend_corpus/gold_sets/validation_baselines.json`.
- Any unexpected change to CAD authority outputs:
  - `profile_mode`
  - `nominal_source_class`
  - expected bend count / nominal topology semantics
- Any new contradiction or invariant failure.

## Refreeze Rule

Refreeze only after:

1. a deliberate improvement changed a protected CAD gold row,
2. the immediate verify run returns exact parity against the new frozen row set,
3. the tranche artifact folder contains completed promotion notes.

## Tranche Artifact Folder

Use one untracked folder per change:

`data/bend_corpus/evaluation/<stamp>_<change>/`

Minimum contents:

- `cad_gold_summary.json`
- `cad_gold_comparison.json`
- `zero_shot_dev_summary.json`
- `zero_shot_holdout_summary.json`
- `full_corpus_summary.json`
- `full_corpus_compare.json`
- `engine_gap_analysis.json`
- `position_hold_analysis.json`
- `metrology_hold_analysis.json`
- `seed_matrix_manifest.json` when applicable
- `PROMOTION_NOTES.md`

## Current Protected Metrics

Read from:

- `data/bend_corpus/gold_sets/validation_baselines.json`

Treat that file as the single promoted baseline contract.

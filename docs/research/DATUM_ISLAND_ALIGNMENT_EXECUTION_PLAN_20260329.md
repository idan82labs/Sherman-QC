# Datum-Island Alignment Execution Plan — 2026-03-29

## Goal
Implement the smallest alignment change that can improve partial-scan bend metrology without destabilizing the current stable core benchmark.

Reference benchmarks:
- baseline: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260328/summary.json`
- post-tranche-1 reference: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260329_observability_v1b/summary.json`

## Current State
### What exists now
Global alignment stack lives mainly in:
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/qc_engine.py`

Current sequence in `ScanQCEngine.align_scan_to_reference` and helpers:
1. PCA pre-alignment
2. flip search
3. fine ICP
4. plane-based datum refinement via `_ransac_datum_alignment`
5. final transform application to aligned scan and raw aligned scan

Relevant functions:
- `_fine_icp`
- `_ransac_datum_alignment`
- `_extract_planes_ransac`
- `_select_datums`
- `_build_datum_point_set`
- `_find_datum_scan_points`
- `_datum_icp`
- `_evaluate_datum_vs_global`
- `_apply_final_transform`

### What the bend pipeline uses now
Local dense-scan refinement in:
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/bend_inspection_pipeline.py`

Current path:
- `refine_dense_scan_bends_via_alignment(...)`
- uses CAD-guided ICP-aligned geometry as the authority for local bend measurement
- trusts or rejects local absence mainly from:
  - local alignment fitness
  - local observability evidence

Current problem:
- local bend metrology still depends on a whole-part alignment result
- the pipeline does not expose which rigid region was trusted as the parent datum island
- no explicit parent-frame metadata exists in bend results

## High-ROI MVP Scope
Do not attempt full hole/slot-first datum-island alignment in one step.
Start with a partial-safe plane-island MVP.

### MVP objective
For partial scans:
- identify one trusted dominant rigid flat region
- mark it as `parent_island_id = PRIMARY_PLANE`
- expose whether the local frame is based on:
  - global ICP only
  - plane datum refinement
- allow downstream bend logic to distinguish:
  - trusted parent-frame alignment
  - generic whole-part alignment

### Why this is the right MVP
- the current engine already has plane datum refinement
- it requires less invasive geometry work than hole extraction / slot fitting
- it is directly compatible with the professor’s spec ordering
- it lets later hole/slot alignment replace the parent-island chooser instead of rewriting the whole pipeline twice

## Concrete Implementation Steps
### Step 1 — Expose alignment provenance from `qc_engine`
Add an alignment metadata object emitted by the alignment path.

Suggested fields:
- `alignment_source`: `GLOBAL_ICP` | `PLANE_DATUM_REFINEMENT`
- `parent_island_id`: `PRIMARY_PLANE` | `NONE`
- `alignment_confidence`: float derived from fitness / primary-plane evaluation
- `alignment_abstained`: bool
- `alignment_conditioning`: dict with:
  - primary plane area
  - secondary plane present
  - scan points near datum count
  - median primary-datum residual
  - p90 primary-datum residual

Suggested seam:
- augment `ScanQCEngine` state or return object without changing viewer compatibility

### Step 2 — Preserve aligned parent-frame metadata in bend local refinement
In `refine_dense_scan_bends_via_alignment(...)`:
- carry the new alignment metadata from the QC engine result into the bend report
- store on report and on each locally measured match context:
  - `parent_island_id`
  - `alignment_source`
  - `alignment_confidence`
  - `alignment_abstained`

This does not yet change the geometry math.
It creates the contract needed for the next tranche.

### Step 3 — Restrict authoritative local absence to trusted parent frames
Current logic already guards on local alignment fitness.
Tighten this further:
- local absence can override a global detection only if:
  - `alignment_source == PLANE_DATUM_REFINEMENT`
  - `alignment_abstained == false`
  - `alignment_confidence >= threshold`
  - canonical observability state is explicit `UNFORMED`

This is a low-risk change and traces directly to the datum-island doctrine.

### Step 4 — Add corpus-visible metadata
Extend output contracts with:
- report-level alignment provenance
- match-level parent-frame provenance in `measurement_context`

Do not change UI behavior yet except making the fields available.

## Files To Change First
Primary:
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/qc_engine.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/bend_inspection_pipeline.py`

Likely contract ripple:
- `/Users/idant/82Labs/Sherman QC/Full Project/contracts/ui/bend_results.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/domains/bend/mappers/ui_contracts.py`
- `/Users/idant/82Labs/Sherman QC/Full Project/frontend/react/src/services/api.ts`

Tests:
- `/Users/idant/82Labs/Sherman QC/Full Project/backend/tests/test_bend_inspection_pipeline.py`
- add new alignment metadata tests near QC alignment behavior if coverage exists

## Acceptance For This MVP
This wave is a success if all hold:
1. no benchmark failures/timeouts added
2. full exact completion rate does not regress below `0.895`
3. structured total MAE does not regress above `1.636`
4. partial authoritative local-absence overrides become stricter and more interpretable
5. benchmark outputs expose parent-frame provenance for later slice-based metrology work

This wave does **not** need to improve MAE by itself.
Its job is to create the correct parent-frame contract for the next wave.

## What Comes Immediately After
Once alignment provenance is exposed, the next wave should use it to change actual measurement:
1. build parent-island anchored section planes
2. compute multi-slice angle/radius estimates in that frame
3. upgrade `UNKNOWN -> FORMED` only when parent-frame and slice evidence both pass

That second wave is where the bigger MAE movement should happen.

## Execution Outcome
Completed in code:
- report-level alignment provenance from `qc_engine`
- per-match `measurement_context.parent_frame`
- authoritative-local-absence restriction to trusted `PLANE_DATUM_REFINEMENT`
- benchmark closure at `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260329_alignment_meta_v1/summary.json`

Observed benchmark effect:
- behavior-neutral on structured MAE and structured full exact completion rate
- safe to keep as the parent-frame contract base for later geometry work

Follow-on implemented immediately after this contract wave:
- first parent-frame local section metrology pass in `/Users/idant/82Labs/Sherman QC/Full Project/backend/bend_detector.py`
- benchmark at `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/benchmark_stability_core_20260329_parent_frame_metrology_v1/summary.json`
- aggregate result remained neutral versus `observability_v1b`

Conclusion:
- the contract work was necessary and is now in place
- the first slice-based implementation did not yet move MAE
- next work should target the slice estimator and per-target routing, not the alignment metadata surface

## Follow-on Slice Estimator V2 Outcome
- Support-driven slice placement and stricter per-slice validity were implemented on top of the parent-frame contract.
- The resulting corpus benchmark improved structured total MAE and structured partial total MAE while keeping the full exact completion rate flat.
- The patch did not improve structured completed-bends MAE; the only structured-set delta vs `observability_v1b` came from `28870000_A_SCAN_1.ply`.
- Next correction target: preserve the visible-span estimator gains while recovering completed-bend calibration on hybrid partials and any other scans where positive promotion is now too eager.

## Hybrid Partial Correction Outcome
- The first slice-estimator gain came from a hybrid partial false positive (`28870000_A_SCAN_1.ply`) promoted by plain `profile_section` under `GLOBAL_ICP`.
- A correction was added so weak non-parent-frame `profile_section` evidence on partial scans downgrades to `UNKNOWN` and stays non-detected through match construction.
- The corrected benchmark (`20260330_parent_frame_metrology_v4`) matches `observability_v1b` exactly on structured metrics while keeping the safer estimator and provenance plumbing.
- Next target: find a real trusted parent-frame or sequence-based gain on partials, because the recovered v2 MAE win was not robust.

# Theory-to-Reality Test Program for Zero-Shot Bend Inference

This document defines the isolated experiment lane for research-inspired zero-shot bend-inference methods.

## Purpose

Use this lane to test theory-derived methods against the current validated zero-shot engine without changing runtime behavior by default.

Required outcomes:

1. rank experiment families against the frozen control
2. record a keep/reject decision for each family
3. identify the smallest winning tranche worth promoting after review
4. package a compact professor-facing brief showing what helped and what failed

## Frozen Control

Treat the current validated zero-shot engine as the control, not a moving target.

The control contract is the combination of:

- `data/bend_corpus/gold_sets/validation_baselines.json`
- protected zero-shot development and holdout slices
- supplemental slices:
  - `zero_shot_known_count_plys`
  - `zero_shot_repeated_near_90`
  - `zero_shot_small_tight_bends`
  - `zero_shot_partial_total_bends`

## Experiment Families

Run families independently before combining them:

1. `fold_contour_diagnostics`
   - adds non-decision fold/trace diagnostics only
   - intended to prove where upstream candidate loss occurs
2. `trace_informed_evidence_atoms`
   - extends evidence atoms with trace/fold support fields
   - does not change selector policy
3. `trace_aware_strip_grouping`
   - requires real along-axis overlap, or a very small gap with strong fold support, before grouping atoms into the same strip
4. `deterministic_candidate_graph_pruning`
   - preserves a richer overcomplete candidate set longer, then prunes deterministically with continuity and compactness penalties

## Execution Rules

- Each family must write to its own artifact folder under `data/bend_corpus/evaluation/`.
- Each family must compare:
  - fresh control run
  - isolated candidate run
- Do not mix two new families in the same first-pass experiment.
- Reject losers immediately instead of leaving them in the runtime path.
- Only promote the smallest winning tranche after a full comparison pass.

## Winning Rule

A family is worth keeping only if all of the following are true:

- zero-shot development stays green
- zero-shot holdout stays green
- known-count PLY audit does not regress
- repeated-near-90 slice does not regress
- small-tight-bend slice does not regress
- at least one hard full-scan case materially improves

Material improvement means one of:

- `ysherman` tightens beyond the validated band
- `49204020_04` tightens beyond the validated band
- another named hard full scan gains a new correct exact output without harming calibration

Do not treat minor aggregate drift as sufficient by itself.

## Automatic Rejection Rule

Reject a family if any of the following occur:

- protected slice regression
- broader range on `ysherman` or `49204020_04`
- loss of exact anchors such as `10839000` or `47959001`
- visual-only improvement with flat count determinism
- benefit that depends on broad preprocess loosening without stable canary wins

## Required Watch Cases

- `10839000` must keep exact `10`
- `47959001` must keep exact `6`
- `ysherman` must not broaden
- `49204020_04` must not broaden
- `48963008` and `48991006` should be tracked as secondary hard full scans

## Artifact Contract

Each experiment folder should include:

- control run outputs
- candidate run outputs
- `experiment_manifest.json`
- `decision.json`
- `DECISION.md`
- repeated-near-90 analysis output
- professor brief
- per-slice comparison JSONs
- per-slice run logs

The manifest should record:

- hypothesis id
- method family
- exact code path touched
- slices run
- artifact folder
- keep/reject decision
- rejection reason when rejected

## Professor Handoff

Each family should produce a short brief that answers:

- what changed conceptually
- what improved
- what regressed
- which real scans illustrate the result
- whether the remaining failure appears upstream, grouping-level, or decomposition-level

Use the brief to decide whether the next jump should remain deterministic and heuristic, or move to explicit latent support assignment and decomposition-level confidence.

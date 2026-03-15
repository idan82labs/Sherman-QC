# Bend Regression Testing Framework

This document defines the current regression-testing framework for bend
inspection and progression analysis in the Sherman QC project.

## Purpose

The regression framework exists to answer four questions reliably:

1. Does a code change improve bend detection on known real parts?
2. Does it preserve behavior on parts that already work?
3. Does it improve partial-process progression reporting, not only final scans?
4. Does it stay runnable on the current development machine constraints
   (`MacBook`, `16 GB RAM`) while remaining faithful to the production path?

The framework is deliberately based on real industrial assets:

- CAD references (`STEP/STP`, some `STL`)
- real scans (`PLY`, `STL`, `OBJ`)
- PDF drawings
- XLSX specification sheets

## Corpus

Ordered corpus root:

- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus`

Core files:

- inventory: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/dataset_inventory.json`
- ready set: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/ready_regression.json`
- confirmation queue: `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/confirmation_queue.json`

Each part folder contains:

- `cad/`
- `scans/`
- `drawings/`
- `specs/`
- `metadata.json`
- `labels.template.json`

## Evaluation path

Primary evaluator:

- `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_corpus.py`

Isolated per-scan worker:

- `/Users/idant/82Labs/Sherman QC/Full Project/scripts/evaluate_bend_case.py`

Runtime inspection entrypoint:

- `/Users/idant/82Labs/Sherman QC/Full Project/backend/bend_inspection_pipeline.py`

Execution model:

- serial
- thread-limited
- one scan per subprocess
- timeout-guarded

This is not a mock route. It uses the real progressive bend inspection pipeline.

## What is scored

Per scan:

- completed bends
- completed bends in spec
- completed bends out of spec
- remaining bends
- observability breakdown:
  - `OBSERVED_FORMED`
  - `OBSERVED_NOT_FORMED`
  - `PARTIALLY_OBSERVED`
  - `UNOBSERVED`
- CAD strategy selected
- expected bend count used

When labeled truth exists, also score:

- error versus expected total bend count
- error versus expected completed bends
- exact completion match

## Current corpus baseline

Latest full run:

- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/full_ready_regression_20260310_absence_fix/summary.json`

Aggregate:

- scans processed: `29`
- successful scans: `29`
- failed scans: `0`
- scans with expected total: `28`
- scans with expected completed: `16`
- MAE vs expected total completed bends: `1.143`
- MAE vs expected completed bends: `0.062`
- full-scan exact completion rate: `0.882`
- continuity-inferred scans: `1`
- continuity-inferred bends: `2`

Staged metrics now tracked in the evaluator:

- monotonic partial-stage pair rate
- retained bend rate between confident staged partial scans
- continuity carry count per scan and per corpus run

Most relevant prior comparison:

- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus/evaluation/full_ready_regression_20260310_continuity/summary.json`
- same `29/29` successful scans
- MAE vs expected total completed bends: `1.286`
- full-scan exact completion rate: `0.882`

Interpretation:

- the framework is now a stable non-regression gate on a broader labeled corpus
- recent gains came from fixing staged-part observability semantics, not from loosening thresholds
- hard partial scans remain the dominant weakness, but the worst staged tray family is no longer collapsing as badly as before

## What improved recently

Four important upgrades are now part of the regression framework:

1. Per-bend observability
2. CAD-local bend neighborhood measurement context
3. Spec-schedule-aware CAD baseline selection
4. Sequence-aware partial continuity plus observability-aware merge semantics

This means the framework is no longer only asking "how many bends were found."
It now also records whether a bend was actually observable and how much trust to
place in spreadsheet-derived bend-angle hints.

The newest regression finding is specific and important:

- a globally well-aligned scan can still have locally unobserved bend neighborhoods
- therefore, local `UNOBSERVED` cannot be treated as authoritative bend absence
- after fixing that merge rule, `38172000` improved on the real single-scan path from:
  - stage 1: `4/10` -> `6/10`
  - stage 2: `5/10` -> `6/10`
- then the staged evaluator lifted stage 2 further to `8/10` using confident continuity carry from stage 1

## Spec-schedule trust model

Spec metadata is classified into:

- `complete_or_near_complete`
- `partial_angle_schedule`
- `mixed_section_callouts`
- `mixed_unknown`

Behavior:

- `complete_or_near_complete` may influence CAD baseline selection strongly
- `partial_angle_schedule` acts only as a compactness hint
- `mixed_section_callouts` must not be used as hard bend-count truth

This matters because many XLSX sheets are not full bend schedules. They often
mix section callouts, local angles, and manufacturing notes.

## Current readiness picture

Stable or mostly stable families in the current run:

- `15651004`
- `47266000`
- `48991006`
- `49125000`
- `11979003` final scans

Families still needing work:

- `38172000` single-scan local identity still over-relies on weak local matches for some bends
- `40500000` partial scans
- `10839000` early partial stage
- `11979003` partial pre-bend scan
- tray-style partials with incomplete observability and one-sided local evidence

## Failure classes we now measure

1. CAD over-segmentation
2. Partial-scan timeout
3. Native-process termination on heavy partials
4. Under-recovery on partial progression
5. Mixed-spec misinterpretation

This is the main reason the regression corpus is valuable: it covers multiple
failure modes, not only success cases.

## How to use this framework

Before promoting an algorithm change:

1. Run backend tests.
2. Run targeted part replays on known weak families.
3. Run the corpus evaluator on `ready_regression`.
4. Compare aggregate results and changed per-part behavior.
5. Reject any change that improves one family by breaking stable families.

## Immediate next step

The next regression goal is to tighten local bend formation semantics:

- one-sided `probe_region` matches with `very_low` confidence should not be promoted as easily
- `PARTIALLY_OBSERVED` must be handled as incomplete local evidence, not as equivalent to a formed bend
- the main hard cases remain:
  - `38172000`
  - `40500000`
  - `10839000` early partial stage

Those are the most informative staged-process cases for the next accuracy pass.

## Rejected assignment experiments

Two mathematically plausible assignment refinements were tested after the staged
metrics baseline:

1. Geometry-weighted local assignment:
   - add bend-line direction and flange-normal pair consistency to the local
     CAD-to-measurement score
2. Global optimal assignment:
   - replace greedy local matching with a cost-matrix solve across all dense
     local bend candidates

Outcome:

- neither change improved the hard staged target slice
- the global assignment variant regressed the broader corpus, most visibly on
  `49024000_A.ply`
- the safer promotion decision was to reject both changes and keep the previous
  promoted matcher

This is part of the value of the regression framework: it prevents mathematically
clean but operationally weaker changes from silently becoming runtime behavior.

# Bend Corpus Data Card

This document describes the current bend corpus as an engineering and
research-grade dataset in progress.

## Dataset purpose

The corpus is used for:

- bend-count regression testing
- progression detection on partially bent parts
- in-spec / out-of-spec classification
- observability analysis
- CAD baseline strategy validation
- future model training or benchmark publication

## Data sources

Source location used to build the current corpus:

- `/Users/idant/Downloads`

Structured corpus root:

- `/Users/idant/82Labs/Sherman QC/Full Project/data/bend_corpus`

Source modalities:

- CAD: `STEP/STP`, `STL`
- scans: `PLY`, `STL`, `OBJ`
- drawings: `PDF`
- specifications: `XLSX`, `XLS`

## Data organization

Each part is normalized into a dedicated folder with:

- asset links grouped by modality
- machine-readable metadata
- label template for ground-truth augmentation

This organization matters because future data will continue arriving and must be
folded into the same evaluation structure without changing benchmark semantics.

## Label state

The corpus contains a mix of:

- fully labeled parts with expected bend count
- partially labeled parts with known process stage
- unlabeled but structurally useful parts
- ambiguous cases waiting for operator confirmation

Two distinct truths are tracked:

1. expected total bends for the finished part
2. expected completed bends for a specific partial scan

This distinction is critical. A partial scan can be "correct so far" even if the
part is incomplete.

## Example high-value cases

### `38172000`

- known expected total bends: `10`
- two partial scans from different process stages
- strongest current progression benchmark

### `47266000`

- rounded / rolled geometry
- strong example of CAD over-segmentation risk
- useful for validating rounded-profile strategy selection

### `47479000`

- rounded family with mixed XLSX callouts
- useful for testing spec-sheet trust logic

### `40500000`

- hard partial family
- useful for runtime-bounded evaluation and recovery work

## Known dataset limitations

1. Label incompleteness

Some parts still lack confirmed expected bend counts or exact completed counts.

2. Family imbalance

The corpus currently contains more sheet-metal and rolled-profile families than
fully characterized, multi-stage fold-progress families.

3. Runtime skew

Heavy partial scans are overrepresented in the failure bucket because they are
also the most computationally expensive.

4. Scan observability variation

Some scans are not wrong; they are simply incomplete from a coverage standpoint.

## Why this dataset is valuable

This is not a generic academic point-cloud dataset. It has properties that are
hard to obtain:

- direct CAD-to-scan linkage
- partial manufacturing stages
- defective and out-of-tolerance outcomes
- associated drawings and spreadsheets
- operator-relevant metrology targets in millimeters and degrees

That makes it useful for:

- industrial QC benchmarking
- partial-process verification
- geometry-aware inspection research
- confidence and observability calibration research

## Research potential

As it stands, the corpus is best described as:

- strong thesis-grade engineering dataset
- pre-publication benchmark candidate

It is not yet publication-ready as a standalone research dataset because it
still needs:

- stronger labeling completeness
- clearer licensing / sharing boundaries
- a frozen benchmark split
- a formal annotation protocol
- reproducibility packaging

## Recommended next upgrades

1. Freeze benchmark splits:
   - stable full-scan set
   - hard partial set
   - rounded-profile set

2. Add annotation fields:
   - expected total bends
   - expected completed bends
   - completed bend IDs
   - known good / known bad final state

3. Add benchmark policy:
   - timeout budget
   - memory budget
   - allowed fallbacks
   - acceptance metrics

4. Add dataset provenance notes:
   - source date
   - file origin
   - drawing/spec association confidence

With those additions, this becomes a credible internal benchmark and a serious
candidate for thesis-level research support material.

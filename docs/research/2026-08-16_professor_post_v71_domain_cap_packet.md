# Post-V71 architecture-selection packet

## Decision requested

Select exactly one bounded next action for the unchanged Sherman QC F1
generalization objective, or issue a formal capability-opening STOP. V71 is
consumed and terminal. Do not rescue it by raising caps, shrinking its domain,
changing constants, rerunning, or substituting an input.

## What passed

- V71 source-free lock suite: 34/34 passed in one run.
- Exact cutoff-shell ordering, dense/sparse closure equality, source and
  endpoint invariance, domain-exit abstention, atomic caps, unchanged V70
  comparators/formulas, duplicate ownership, and refinement were covered.
- All primary/replay aborts were byte-identical.
- Every execution was fast (2.33-2.96 s) and below 3.1 GB peak RSS.

## What failed

The frozen 48-candidate U01-U04 stream produced zero complete scan decisions:

- U01 and U02: primary and replay `GRAPH_EDGE_VISIT_CAP`.
- U03 and U04: primary and replay `Q0_CAP`.

This occurred before sparse-field evaluation could make candidate decisions.
The full evidence is in
`docs/research/2026-08-16_f1_v71_clec_mvt_terminal_retirement.md` and
`output/research/f1-v71-clec-mvt-u01-u04-actionability-v1-20260816`.

## Frozen constraints

- Preserve the same F1/generalization objective.
- Source-only unless you explicitly argue the smallest scientifically
  necessary capability opening.
- Gold, labels, CAD, expected counts, filename/identity, runtime/main,
  production, and benchmark feedback remain closed.
- V52 remains the retained detector baseline.
- V70 and V71 are terminal and cannot be rerun or rescued.
- The quarantined synthetic morphology cannot influence architecture or tuning.
- The reserved new physical scan remains sealed until unchanged mechanical and
  native-resolution visual gates pass.
- One selected architecture, one primary/replay tranche, no post-hoc tuning.

## Required response

Return either:

1. `EXECUTE_ONE` with one fully specified, bounded architecture that does not
   depend on conservative graph-distance halos of the V71 kind; include exact
   inputs, formulas, comparators, caps, atomic aborts, source-free fixtures,
   U01-U04 mechanical gates, and native-resolution visual gates; or
2. `CAPABILITY_OPENING_STOP` with the smallest necessary new information or
   measurement and a falsifiable protocol for using it.

Favor a mechanism whose computational dependency domain is intrinsically
bounded before seeing real-scan topology, rather than another cap adjustment
to candidate-local graph closure.

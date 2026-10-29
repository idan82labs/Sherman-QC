# F1 professor hard-case adjudication packet — 2026-07-15

## Status

The deterministic F1 lane now reaches render-admissible count equality on 11
of 13 verified scans after suppressing a contained same-flange-pair marker alias
in case 07. Hard visual acceptance remains 10 of 13: count equality alone does
not make case 07 correct because a second cross-pair collinear ambiguity remains.

The machine-readable request is
`output/experiments/f1-10of13/professor-hard-cases-20260715/adjudication_packet.json`.
It pins every scan and visual artifact by SHA-256 and states the exact response
fields needed to turn guidance into typed `reference_mesh_raw` instance truth.

## Case 06

Four plane-lattice transitions are visually defensible against evaluation count
six. Candidate additions found by ridge and junction probes lie on face interiors.
The required answer is not a count: it is two axis endpoint pairs, their physical
flange incidences, and the property that separates each from its closest rejected
ridge.

![Case 06 four-line evidence](/Users/idantal/ShermanQC/FullProject/output/experiments/f1-10of13/line-audits/06_38172000_plane_lattice.png)

## Case 07

OB8 is a contained alias of OB4: the two regions select the same `F1/F8` pair,
their axes differ by only `0.000035°`, the perpendicular offset is `2.063 mm`,
and the shorter span is completely contained. The dominant scan-supported strip
has 6.91 times the support mass and 2.38 times the atoms. Suppressing OB8 changes
the render-admissible count from nine to eight without using truth.

The remaining OB1/OB3 pair is almost equally alias-like geometrically but has
different inferred second flanges. If merged, the result has seven objects and a
different bend is missing. The professor must decide whether a real intermediate
flange makes OB1/OB3 two bends, or whether flange segmentation drift duplicated
one bend, then localize all eight instances.

![Case 07 exact-eight but ambiguous evidence](/Users/idantal/ShermanQC/FullProject/output/experiments/f1-11of13/case07-same-pair-span-alias/07_38172000_render_admissible8.png)

## Case 08

Six distinct surface-attached plane intersections are defensible against
evaluation count seven. No seventh line survives two-sided support and hard
surface-attachment review. The required answer is either a seventh axis with
incident flanges or an explicit instruction to split one of the six maximal
instances across a support gap.

![Case 08 six-line evidence](/Users/idantal/ShermanQC/FullProject/output/experiments/f1-10of13/line-audits/08_40500000_plane_lattice.png)

## Leakage boundary

Counts and professor annotations are evaluation-only. They may score or frame a
review question, but they may not enter detector configuration, route selection,
candidate generation, cache identity, or final marker selection. Any returned
per-instance annotations must be joined after inference by scan content hash.

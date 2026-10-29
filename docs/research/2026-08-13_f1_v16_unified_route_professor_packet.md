# GPT-5.6 Pro packet — V16 unified mesh route and next recall primitive

## Requested ruling

Select exactly one smallest, falsifiable, source-only next intervention for the
full-resolution mesh lane, or issue STOP. Ground the reasoning in relevant
published research where possible and cite the specific ideas being adapted.
Do not propose a coupled architecture sweep or post-hoc threshold tuning.

The concrete target is to recover true bend axes that already exist as
scan-mesh-adjacency candidates but are rejected by the current V11 full-mesh
selector. Specify an inference-time predicate using only already-emitted source
features, the exact insertion point, deterministic tie/NMS behavior, and hard
retain/reject gates.

## Frozen constraints

- Blind scan-only inference; no CAD, drawing, labels, gold, expected count,
  filename/identity, production feedback, or benchmark feedback.
- Deterministic CPU path.
- Approximately 200,000-point bounded detector input; no source-point
  serialization in caches.
- Peak RSS below 16 GiB; P95 latency no more than 300 seconds.
- Research-only; runtime/main and production remain closed.
- No part-specific threshold or family identity logic.
- One bounded implementation followed by one development regression and fresh
  held-out confirmation. No rescue tuning after seeing locked outcomes.

## What changed

V15 was a bounded-complex selector. It was accidentally invoked directly on
one native full-resolution held-out pool (`U02`, 50552000), producing two
aliases. V16 makes route ownership explicit:

1. native full-resolution mesh -> existing V11 full-mesh lane;
2. bounded small pool -> existing bounded recall and consensus repair;
3. bounded complex pool -> retained V15 paired-plane-fold lane;
4. unavailable mesh -> existing V6 non-mesh lane.

No generator, feature schema, model, or runtime path changed.

## Development evidence

Frozen 35-scan / 25-part-family corpus, same candidate pools and gold:

| Metric | V12 unified | V16 unified |
|---|---:|---:|
| TP | 132 | 132 |
| FP | 98 | 60 |
| FN | 44 | 44 |
| Precision | 0.573913 | 0.687500 |
| Recall | 0.750000 | 0.750000 |
| F1 | 0.650246 | 0.717391 |
| Exact cases | 3/35 | 4/35 |

V16 removes 38 false positives with zero true-positive loss. Route composition:
15 native full-resolution, 18 bounded-complex, 2 bounded-small. Minimum-family
F1 remains 0.0, so V16 is retained but not promotable.

## Fresh held-out confirmation

Four manually precommitted scans were visually selected before inference:

| Opaque scan | Route | TP | FP | FN | Visual ruling |
|---|---|---:|---:|---:|---|
| U02 | full V11 | 4 | 0 | 0 | four wall/base creases; aliases gone |
| U03 | bounded V15 | 3 | 0 | 0 | three structural folds; no window/free edge |
| U04 | bounded V15 | 3 | 0 | 0 | all three closely spaced return folds preserved |
| U05 | bounded V15 | 5 | 0 | 0 | paired long/upper folds plus short closing fold |

Aggregate: 15 TP / 0 FP / 0 FN. Native-resolution four-view overlays were
manually inspected, not merely scored. U02's largest angle residual is 4.69
degrees but lies on the visible crease. U05's largest offset is 3.99 mm; the
short junction fold is visibly correct and does not capture the adjacent
opening.

## Operational evidence

- U02--U05 independent V16 serializations are byte-identical.
- Selector-only U05 execution: 1.02 s wall, 392,773,632-byte maximum RSS,
  356,648,040-byte peak memory footprint.
- Upstream bounded candidate-generation and compact serialization contracts
  are unchanged.

## Residual diagnosis requiring the ruling

The only zero-F1 family is development case C017 / part family 48333006, one
native full-resolution scan with two trusted bends. V11 emits one prediction,
which matches neither. However the frozen pool already contains two
source-derived adjacency axes that match the trusted axes essentially exactly:

| Candidate | Gold | angle | transverse offset | gold overlap | pred overlap | source features |
|---|---|---:|---:|---:|---:|---|
| SMA1 | B01 | 0.0007° | 0.0004 mm | 0.999995 | 0.999994 | adjacency-only; span 32.59 mm; finite/longest 0.96; endpoint 0.833; p90 distance 1.787 mm; transition 0.606; normal angle 75.93°; balance 0.984; min-side 0.445; coherence 0.635 |
| SMA3 | B02 | 0.0006° | 0.0003 mm | 1.000000 | 0.999992 | adjacency-only; span 32.60 mm; finite/longest/endpoint 1.0; p90 distance 1.389 mm; transition 0.312; normal angle 49.72°; balance 0.749; min-side 0.307; coherence 0.545 |

These gold comparisons are diagnostic only and cannot become inference inputs.
The inference problem is to distinguish trustworthy adjacency-only bend axes
from adjacency aliases across all native full-mesh cases using source evidence.

## Required response format

1. `RULING`: one intervention name, or `STOP`.
2. `RESEARCH BASIS`: relevant established methods/ideas with citations and a
   short explanation of how the proposed primitive adapts them.
3. `EXACT PREDICATE`: feature names, inequalities or deterministic ranking,
   NMS/duplicate relation, and insertion point in V11/V16.
4. `WHY IT MAY GENERALIZE`: evidence-based mechanism, not family-specific
   intuition.
5. `FALSIFICATION GATES`: at minimum zero TP loss on all 35 development scans,
   improved pooled F1 and minimum-family recall, no regression on U02--U05,
   byte-identical replay, and unchanged resource/serialization contracts.
6. `EXPECTED FAILURE MODE`: the most likely source of new false positives and
   the single visual audit that would expose it.

Evidence paths:

- `scripts/select_f1_mesh_route_v16.py`
- `scripts/validate_f1_mesh_route_v16.py`
- `output/research/f1-new-parts-20260813/v16_unified_development_regression.json`
- `output/research/f1-new-parts-20260813/v16/overlays/`
- `docs/research/2026-08-13_f1_v12_mesh_route_consensus_midpoint.md`

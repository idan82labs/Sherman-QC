# GPT-5.6 Sol Pro packet — post-V17 endpoint/ownership architecture ruling

## Requested ruling

V17 is terminally stopped. Select exactly one smallest, falsifiable,
source-only next intervention for the native full-resolution mesh lane, or
issue STOP for this research direction.

The next intervention must address the observed causal gap: distinguish an
independent interior material fold from a terminal silhouette, opening edge,
thickness face, welded/formed corner seam, or cycle-closing boundary. Do not
rescue V17, lower its thresholds, add a scan-level cap, or propose a coupled
architecture sweep.

Ground the proposal in relevant published research where possible. Specify the
exact source-only primitive, feature schema, deterministic predicate/ranking,
insertion point, duplicate behavior, and hard retain/reject gates.

## Frozen constraints

- Blind scan-only inference; no CAD, drawing, labels, gold, expected count,
  filename/identity, production feedback, or benchmark feedback.
- Deterministic CPU path.
- Approximately 200,000-point bounded detector input.
- Compact caches; no source-point serialization or new unbounded arrays.
- Peak process-tree RSS below 16 GiB; P95 latency at most 300 seconds.
- Research-only; runtime/main and production remain closed.
- No part-specific thresholds, identity logic, post-hoc rescue, or sweep.
- One bounded implementation followed by preregistered source-only and gold
  gates. Failure is terminal for that architecture identity.

## Retained baseline

V16 explicitly routes native full-resolution meshes to V11, bounded-small pools
to bounded recall, bounded-complex pools to retained V15, and unavailable mesh
to V6.

Frozen 35-scan / 25-family development result:

- 132 TP / 60 FP / 44 FN;
- precision 0.6875;
- recall 0.75;
- F1 0.717391;
- exact cases 4/35;
- minimum-family F1 0.0.

Recent frozen U02--U05 confirmation is 15 TP / 0 FP / 0 FN with byte-identical
replay. V16 is retained but not promotable.

## V17 intervention and source-only falsification

V17 implemented the authorized append-only axial-hysteresis rescue exactly:

- rejected `scan_mesh_adjacency` only;
- finite support and longest run >= 0.90;
- endpoint support >= 2/3;
- `0 < p90 <= 3 mm`;
- `span >= 10 * p90`;
- transition >= 0.25;
- two-sided normal angle >= 45 degrees;
- balance >= 0.50;
- minimum-side support >= 0.20;
- coherence >= 0.50;
- incumbent-first frozen V11 NMS;
- no top-k or rescue cap.

Implementation tests and independent reconstruction passed 29/29.

Before development gold was opened, the mandatory label-closed U02--U05 check
found:

| Scan | V17 rescues | Frozen result |
|---|---|---|
| U02 | SMA1, SMA2 | V16 exact 4/4; V17 becomes 6 lines; bytes differ |
| U03 | none | byte-identical |
| U04 | none | byte-identical |
| U05 | none | byte-identical |

This violates the explicit requirement that U02 remain byte-identical. No
development-gold regression or tuning was performed.

## Visual causal diagnosis

Native-resolution four-view artifact:
`output/research/f1-new-parts-20260813/v17_preflight/U02_v17_rescue_native_audit.png`

- SMA1 is a terminal/perimeter corner segment.
- SMA2 is a short exterior/thickness-edge segment.
- Neither is a distinct interior material fold.
- Both satisfy the V17 continuity/localization/two-sided certificate.

Therefore the missing information is endpoint/ownership topology, not stronger
axial evidence. A candidate can be long, localized, continuous, and two-sided
while still tracing an exterior boundary or closing an already-explained corner.

## Available source-only structures

The frozen pool contains line geometry, mesh-adjacency source features,
two-sided support summaries, compact transition atoms/parents/modes and their
bounded identifier graph, plane modes, source-family membership, and existing
V11 axes. Raw source points are not serialized. Any proposal must state whether
these existing compact structures are sufficient or whether one smallest
bounded generator/schema addition is scientifically necessary.

## Required response format

1. RULING: one intervention name, or STOP.
2. RESEARCH BASIS: cited established methods/ideas and the exact adaptation.
3. CAUSAL FEATURE: how the primitive distinguishes interior fold ownership from
   silhouette/opening/thickness/corner-cycle evidence using scan-only data.
4. EXACT SPECIFICATION: fields, deterministic construction or inequalities,
   insertion point, ranking/NMS, and serialization/resource bounds.
5. FALSIFICATION GATES: source-only U02 must remain a no-op; intended recall
   activation must occur on C017 source evidence; then one-shot development and
   fresh held-out gates, determinism, memory, latency, and visual endpoint audit.
6. EXPECTED FAILURE MODE: the single most likely new false-positive mechanism
   and the native-resolution audit that exposes it.

Evidence paths:

- `docs/research/2026-08-14_f1_v17_axial_hysteresis_terminal_stop.md`
- `docs/research/2026-08-14_gpt56_sol_pro_v17_adjacency_hysteresis_ruling_verbatim.md`
- `scripts/select_f1_full_mesh_adjacency_hysteresis_v17.py`
- `scripts/select_f1_mesh_route_v17.py`
- `scripts/validate_f1_mesh_route_v17.py`
- `output/research/f1-new-parts-20260813/v17_preflight/`
- `output/research/f1-new-parts-20260813/v16_unified_development_regression.json`

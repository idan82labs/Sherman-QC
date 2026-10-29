# Sherman QC F1 — post-V66 terminal next-architecture request

You are the professor/architect for a zero-shot, source-only bend-line detector for sheet-metal 3D scan geometry. Select exactly one bounded, action-oriented next architecture, or issue the smallest scientifically necessary capability-opening STOP. Ground the ruling in relevant primary research where possible, but make the contract self-contained and executable.

Return exactly one leading token:

- `EXECUTE_ONE` — followed by one fully frozen architecture and one source-free fixture suite; or
- `CAPABILITY_OPENING_STOP` — followed by the smallest necessary new information/capability and a falsifiable reason source geometry alone is insufficient.

Do not propose alternatives, tuning menus, or additional exploratory analysis. We will implement the one selection immediately, run it prospectively once, visually inspect every changed native-resolution result, and retire it quickly if it fails.

## Objective

Improve zero-shot bend-line generalization and F1 on genuinely unseen physical parts, specifically by reducing false positive bend responses without sacrificing nearby real bends. The system must remain deterministic, CPU-bounded, compact, scan-only, and auditable.

## Retained baseline

- V52 remains retained and frozen.
- On the consumed R001–R008 real-scan affine/resampling lane, V52 scored TP 18, FP 22, FN 14; precision 0.45, recall 0.5625, F1 0.50; exact cases 0/8.
- Dominant visual defect: multiple close, parallel predictions on what appears to be one physical fold, plus junction/perimeter/flat-face clutter.
- The fresh U01–U04 source-only lane used below remains pseudo-truth sealed; no labels, gold, expected counts, identities, or benchmark feedback were opened.

## V66 that was executed

The prior Pro ruling selected `V66_TRANSVERSE_FACE_TURN_OWNERSHIP_CERTIFICATE_V1` (`TFTOC_V1`), a pre-NMS suppression-only certificate based on exact transverse mesh sections, face-local normals/turn events, and all-pairs-positive cliques.

V66 passed all 11 exact source-free fixtures plus point-only, resource-cap, and incomplete-clique safety fixtures. It was then frozen by byte hash and executed once on four bounded real corpus meshes with incoming candidate counts 4, 5, 9, and 3. Every identity/replay, locked similarity, order/winding, exact PL-refinement, and ordinary final NMS variant was computed in the frozen order.

Observed real-scan result:

- V66 eligible pairs: 0 across all four cases
- V66 positive pairs: 0
- V66 suppressions: 0
- changed cases: 0
- three cases passed the complete route contract
- the nine-candidate case failed the route contract
- identity replay: byte-identical
- V66 mutation semantics: invariant under the locked similarity, order/winding, and exact PL refinement
- ordinary final NMS output/audit: not invariant under the locked 0.8x similarity transform
- pseudo-truth remained sealed
- no scoring was performed
- V66 is terminally retired; no rerun or rescue is authorized

Scientific interpretation: the source-free fixtures proved the predicate implementation, but the real candidate geometry never entered the mechanism's pair-eligibility domain. Separately, the retained route's ordinary final NMS is scale-sensitive. V66 therefore delivered no visual delta and no benefit, and did not compose with the required similarity contract.

## Synthetic-data correction that must constrain the ruling

The recent synthetic morphology lanes are quarantined. Manual comparison against real corpus meshes showed that generated parts had clean full-span rectilinear seams, rectangular windows, zero-radius joints, repeated parallel lips, and complete uniform sampling. Real scans instead show finite-radius/stamped/asymmetric bends, partial views, irregular boundaries, occlusion and dropout. The synthetic lanes produced many artificial false positives; e.g. one audited case produced 0 TP and 3 FP on shoulders/perimeter structures. These artifacts are excluded from tuning, architecture selection, benchmark counts, and generalization claims. Do not use them as success evidence or request a synthetic rescue.

## Required ruling

Select the single smallest architecture that can materially reduce the observed real-scan false positives while remaining source-only. The intervention may replace the scale-sensitive ordinary final NMS, add a new candidate-validity certificate, or make another sharply bounded change, but it must not rescue V66 or alter V52 itself.

The frozen contract must specify:

1. exact insertion/replacement point relative to V52 composition and final NMS;
2. exact inputs, source population, normalization and deterministic scale;
3. exact candidate/pair/cluster eligibility with proof that realistic V52 real-scan candidates can enter the domain;
4. exact positive and negative evidence and fail-closed behavior;
5. exact mutation, representative ranking and protection of nearby distinct bends;
6. strict similarity, rigid-transform, ordering, winding and exact PL-refinement invariance, including all downstream deduplication;
7. deterministic compact audit and primary/replay semantic identity;
8. CPU, memory, graph, pair, candidate and wall-time caps with atomic no-op;
9. internally consistent source-free fixtures including close duplicates, nearby distinct bends, U-channel/opposing folds, return/hem double skin, rounded folds, small creases, flat-face/perimeter distractors, partial support and exact decision boundaries;
10. a prospective, truth-unopened real-scan protocol using fresh deterministic variants without opening identities or labels;
11. native-resolution visual audit of every changed case before pseudo-truth scoring;
12. hard retain/retire gates that require a real visual delta and prevent false suppression;
13. whether the newly received physical part should be reserved as later unseen evidence or is the smallest necessary capability opening now.

## Frozen constraints

- Allowed decision inputs: scan coordinates, mesh adjacency when present, candidate segment geometry and opaque hashes only.
- Closed: gold, labels, CAD, drawings, expected counts, filenames, part/parent/family identity, good/bad status, transformed pseudo-truth, benchmark feedback, production outcomes and post-hoc tuning.
- V52 is retained and frozen.
- V65 and V66 are terminal and may not be rescued.
- Consumed R001–R008 and the V66 one-shot cases cannot be used for selection or threshold fitting.
- Quarantined synthetic parts cannot support architecture selection or generalization claims.
- Every changed real case must be inspected at native resolution before pseudo-truth is opened.
- No rerun rescue, replacement input, or threshold tuning after prospective execution begins.
- Gold and runtime/main/production remain closed unless you explicitly prove the smallest scientifically necessary opening.

Please check mathematical consistency between every mandatory positive fixture and every confidence/validity threshold before issuing `EXECUTE_ONE`. Prefer a mechanism whose eligibility has an explicit preflight coverage gate so we do not again spend a one-shot on an architecture that cannot act on real candidate geometry.

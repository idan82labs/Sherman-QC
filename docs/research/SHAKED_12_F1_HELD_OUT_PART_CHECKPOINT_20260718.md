# Shaked 12 F1 held-out-part checkpoint — 2026-07-18

## Claim boundary

This checkpoint measures count-free prediction on each held-out part with nested
leave-one-part-out fitting and threshold selection. The held-out part's manual
labels are excluded from its fit and inner threshold selection.

This is **not** a fresh blind benchmark and it is **not strict model-zero-shot**:
the selector learns from the other labeled Shaked parts, and the full corpus had
already been inspected during development. The 59 assistant-reviewed axes are
also user-review-pending and 49 were derived from or snapped to generated mesh
candidates. They are development labels, not independent promotion ground truth.

The enterprise promotion claim therefore remains blocked. No F1 code from this
research lane is promoted to runtime/main by this result.

## Canonical result

- Report: `output/experiments/shaked-mesh-nested-lopo-v18-surface-graph-outline-pruning/nested_lopo_line_accuracy.json`
- Predictions and six-view overlays: `output/experiments/shaked-mesh-nested-lopo-v18-surface-graph-outline-pruning/`
- TP / FP / FN: **59 / 4 / 0**
- Precision: **0.936508**
- Recall: **1.000000**
- Line F1: **0.967213**
- Exact geometry: **9 / 12 parts**
- Exact count: **9 / 12 parts**
- Repeated-run metrics and all prediction JSON SHA-256 hashes: identical
  (report path fields differ only by the repeat output root)

For comparison, the corrected original verified-13 result is still only
**0.480620 line F1 and 0/13 exact geometry**. The Shaked result must not be
substituted for the original-13 promotion target.

The SHA-frozen four-part Shaked slice scores **16 TP / 1 FP / 0 FN**, precision
**0.941176**, recall **1.000000**, and F1 **0.969697**, with 3/4 exact geometry
and 3/4 exact count. Its report is
`output/experiments/shaked-selector-evaluation-v15-v8-surface-graph/mesh_adjacency_holdout/locked_score.json`.
Because the slice had already been inspected before these methods were designed,
this is post-development holdout evidence rather than a fresh blind claim.

## Six-view manual audit

Colors in the canonical renders are: matched gold green, missed gold orange,
matched prediction cyan dashed, and false prediction red dashed.

| Part | Score | Hard visual assessment |
| --- | --- | --- |
| 01_49191023 | 8 TP, 0 FP, 0 FN | Clean exact overlay. A finite-junction surface-graph contradiction removes the multiply over-closed parallel chord while preserving all eight reviewed folds. Acceptable. |
| 02_24694000 | 6 TP, 0 FP, 0 FN | Clean exact overlay. The short broad-face triangle chord and isolated dominated parallel perimeter interface are removed while retaining all six physical folds. Acceptable. |
| 03_49191054 | 6 TP, 0 FP, 0 FN | Clean exact overlay across all six views. The recovered axes sit on the visible fold interfaces rather than the exterior silhouette. Acceptable. |
| 04_48340008 | 3 TP, 0 FP, 0 FN | Clean exact overlay after weak-parallel outline pruning removes the broad-face diagonal while retaining all three physical folds. Acceptable. |
| 05_49191032 | 3 TP, 2 FP, 0 FN | All three reviewed axes now align: the two short side returns are recovered only when their fully supported cylindrical refits retain independently supported detector-seed provenance. Two long broad-face/outline axes remain. Complete recall, unsafe precision. |
| 06_49191061 | 5 TP, 0 FP, 0 FN | Clean exact result. Three close dominant folds plus both short terminal bends align after topology completion. Acceptable. |
| 07_49191038 | 8 TP, 0 FP, 0 FN | Clean exact overlay after parallel surface-sibling completion recovers the final long/local interface. Every cyan axis lies on its reviewed fold in all six views. Acceptable. |
| 08_48333018 | 2 TP, 0 FP, 0 FN | Clean exact overlay. Pairwise shared-surface evidence removes both weaker parallel perimeter branches while retaining the two physical folds. Acceptable. |
| 09_49191037 | 5 TP, 1 FP, 0 FN | All five real bends align after parallel surface-sibling completion. One end-tab/protrusion outline remains visibly falsely marked. Complete recall, not exact. |
| 10_48333006 | 2 TP, 0 FP, 0 FN | Both parallel folds align cleanly in every view with no outline substitution. Acceptable. |
| 11_48340007 | 4 TP, 1 FP, 0 FN | All four real bends align. One unmistakable diagonal traverses a broad face and must be suppressed. Not exact. |
| 12_49191030 | 7 TP, 0 FP, 0 FN | Clean exact overlay across a long multi-flange part, including short terminal bends. Acceptable. |

## Remaining error families

1. **Broad-face and silhouette substitution** — four false axes remain in
   parts 05, 09, and 11. Surface adjacency still treats two case-05 branches,
   one protrusion boundary, and one broad-face diagonal as bend evidence.
2. **Bend recall is complete on this development corpus** — all 59 reviewed
   axes are recovered. Short scan-mesh, surface-wedge, parallel-sibling, and
   supported-seed completion add nine true axes without adding false positives.
   This is not an independent blind-recall claim because the labels and corpus
   were already inspected during development.
3. **Part-05 candidate suppression is resolved, precision is not** — exact
   combined-pool axes `LO95` and `LO100` are now admitted only through supported
   detector-seed provenance, while the long `LO4` interface comes from guarded
   mesh consensus. The remaining part-05 failure is two false outline axes.
4. **Evidence independence** — the assistant labels need human adjudication or
   independently authored ground truth before any blind/promotion claim.

## Promotion decision

The development target of at least 0.80 line F1 is met, but the visual target of
near-perfect whole-part alignment is not: 9/12 parts are exact and three parts
still contain four visibly false outline/broad-face axes. The independent original-13
corpus remains 0/13 exact. Keep F1 in the research lane and continue with
nonlocal ownership/developability semantics plus topology-aware false-line
suppression before running or claiming the enterprise promotion benchmarks.

## Professor decision request

The next useful GPT-style professor review should answer these narrowly scoped
questions from the raw six-view evidence, without access to part IDs or expected
bend counts:

1. What axis-local geometric invariant distinguishes a countable forming bend
   from an intersection with a small attachment, cut feature, or segmentation
   fragment when both show two-sided normals and continuous line support?
2. Should a scan-mesh candidate with less than 15 degrees of local normal turn
   ever be retained as a physical bend when the fitted adjacent surface normals
   imply a much larger angle? If yes, what independent evidence is required?
3. For part 05, what nonlocal ownership/developability topology rejects the two
   remaining long scan-mesh outline axes while retaining recovered `LO4`,
   `LO95`, and `LO100`, without using the known bend count or manual labels?
4. What oriented physical-sheet representation distinguishes the case-11 false
   branch from its nearly identical true branch, when gap, scan support,
   region-graph path, pair-contact occupancy, and plane-incidence evidence all
   favor or tie the false branch? The same representation must also reject the
   case-09 protrusion boundary without losing its six-hop true fold.

The tested alternatives do not answer these questions: new mesh births,
cross-normal axes, global combined-source selection, normal-angle gates,
area/symmetry features, interaction features, local cross-section descriptors,
raw region-graph ownership/path features, developability proxies, and tree-model
substitutions all regressed or failed strict nested isolation.

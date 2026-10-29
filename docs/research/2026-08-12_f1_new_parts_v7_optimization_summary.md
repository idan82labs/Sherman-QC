# F1 new-parts optimization summary — V7

## Outcome

The new scans exposed two distinct failures and produced three bounded selector iterations:

- V4 returned zero on both 389 scans despite six visible bend candidates.
- V5 added a source-only dead-zone route and recovered all six visually precommitted 389 bends.
- V6 recovered the two long plane-lattice bends on each 472 scan but could not see the terminal head because mesh adjacency was disabled above the 200k-vertex budget.
- The generator now applies deterministic quadric decimation before mesh adjacency. It recovered 18/17 raw adjacency candidates and 16/13 canonical line objects.
- V7 uses bounded-mesh coverage plus scale-normalized long-alias suppression. It matches all 12 visually selected H01 bends and all 10 visually selected H02 bends, while retaining the six 389 selections.

The manual-fit metric is 28 TP, 0 FP, 0 FN over the 28 visually localized lines. This is an optimization result, not independent generalization evidence. If the sender's 12-bend count for each 472 scan is used only at evaluation time, H02 has two unlocalized false negatives and the four-new-scan count-conservative score is precision 1.0, recall 0.933333, F1 0.965517.

## Visual defect check

Rigid scan-to-scan registration does not validate a large physical defect. At full resolution the defective-to-good nearest-surface distance is 0.170 mm median, 0.415 mm p99, 0.906 mm maximum, and no point exceeds 1 mm. Therefore the 12-versus-10 candidate difference must be treated as extraction instability, not proof that the defective part has two missing bends.

## Verification

- Candidate-pool deterministic replay hash matched exactly.
- 32 targeted tests passed.
- The old 12-case development result is unchanged: TP 29, FP 23, FN 23, F1 0.557692.
- CAD/PDF remained sealed; no expected count, role, identity, or CAD entered inference.

## Next independent gate

Freeze V7 and evaluate it unchanged on a third unseen part with bend-line annotations precommitted before candidate generation. For defect detection, add a tolerance-aware signed/normal deviation route; nearest-surface distance alone cannot distinguish the supplied good/bad pair at the observed sub-millimeter scale.

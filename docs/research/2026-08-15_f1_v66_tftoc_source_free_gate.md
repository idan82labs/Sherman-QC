# V66 TFTOC — Source-Free Gate

Date: 2026-08-15

## Outcome

`V66_TRANSVERSE_FACE_TURN_OWNERSHIP_CERTIFICATE_V1` passes the locked SF01–SF11 suite exactly. The implementation is deletion-only, operates on the frozen V52 pre-NMS candidate list, retains face-local normals from exact piecewise-linear transverse sections, and performs no internal ordinary NMS.

The old V65 contradiction remains preserved as an expected failure: an equal-area 70° dihedral cannot satisfy V65's averaged-projector confidence threshold. V66 does not compute that projector. Its sharp 70° fixture produces three positive pairs, nine SAME_EVENT stations per pair, a complete three-candidate clique, and retains the +1h route while suppressing the −3h and +4h duplicates.

## Exact locked results

- SF01 sharp 70° duplicate ownership: pass.
- SF02 rounded 70°, radius 3h: pass.
- SF03 two +55° folds 6h apart: two positive cliques; every cross-fold pair is SPLIT_EVENT at all nine stations; pass.
- SF04 +80°/−80° U-channel: two positive cliques; every cross-fold pair is SPLIT_EVENT at all nine stations; pass.
- SF05 170° return/hem: rejected by the 145° ceiling; no mutation; pass.
- SF06 20h×0.8h plate perimeter: no positive ownership and no mutation; pass.
- SF07 true fold plus flat route: true +0.25h route retained; +3h route suppressed; pass.
- SF08 missing first three stations: 6 SAME, 3 INDETERMINATE, UNKNOWN, no mutation; pass.
- SF09 missing stations 1 and 9: exactly 7 SAME, positive, one suppression; pass.
- SF10 90° crossing line: ineligible at the 3° parallelism gate; pass.
- SF11 replay/order/winding: complete semantic audit byte-identical under replay, reversed winding, face permutation, and vertex permutation with remapped incidence; pass.

Additional safety tests prove that a positive transitive chain without an all-pairs certificate cannot mutate, a point-only source is a global `NO_TRIANGLE_ADJACENCY` no-op, and a candidate-cap violation preserves the complete incoming list through `RESOURCE_ATOMIC_NOOP`.

## Evidence boundary

These fixtures validate only the locked geometry, accounting, determinism, and fail-closed mutation rules. They are not corpus morphology, detector accuracy, or generalization evidence. The previously generated O001–O011 synthetic parts remain quarantined and were not used to choose or adjust any V66 constant.

## Next gate

The source-free result authorizes only a fresh U01–U04 preflight: hash and freeze the V52 executable, V66 executable, fixture source, renderer, pseudo-scorer, and source bytes; keep pseudo-truth sealed; generate identity/replay, similarity, order/winding, and exact PL-refinement variants; then execute once. No rescue tuning or replacement input is authorized.

Machine-readable evidence: `output/research/f1-tftoc-v66-source-free-fixtures-v1-20260815/summary.json`.

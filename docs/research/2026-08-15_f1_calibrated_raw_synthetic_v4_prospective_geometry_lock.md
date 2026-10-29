# F1 calibrated raw-range synthetic V4 prospective geometry lock

Date: 2026-08-15

Status: frozen before full-corpus generation and before any detector invocation.

## Purpose

V4 is a fresh, prospective, non-confirmatory corpus for evaluating the already-frozen V61 and V62 source-only routes. It is not a recapture of V3. It contains fifteen new cross-section families selected to stress small creases, 2.0–5.5 mm bends, irregular spacing, tight returns, compound shapes, broad rounded facets, and mixed-radius transitions.

## Frozen identifiers

- token: `SYNTHETIC_CALIBRATED_RAW_RANGE_V4_PROSPECTIVE_COMPLEX_NON_CONFIRMATORY`
- corpus kind: `sherman_qc_f1_calibrated_raw_range_prospective_complex_v4`
- capture indices: `8, 9`
- family count: 15
- planned case count: 30
- generator SHA-256: `a31a582961293b719fb7afd412d5606bfe8a758d1fab639f3f68eb08ddf36520`
- generator test SHA-256: `9bffa8feb0198689129de5f5536220c9b134064f4c045a899daccc28063c3134`

## Frozen families

1. PF01 triple micro crown
2. PF02 rolled Z return
3. PF03 multi-radius omega
4. PF04 inverted nested hat
5. PF05 faceted low crown
6. PF06 staggered shelf joggle
7. PF07 broad bulb channel
8. PF08 sub-three-millimetre step ladder
9. PF09 asymmetric W return
10. PF10 rolled box lip
11. PF11 paired micro-bead shelf
12. PF12 deep multi-facet trough
13. PF13 alternating micro/macro ribs
14. PF14 hooked hat with double return
15. PF15 long mixed-radius labyrinth

## Pilot visual ruling

The four-case PF01/PF02 pilot was inspected in fused 3D and in all eight native calibrated depth views per case. Both captures of both families passed construction observability. The micro-crown retained multiple shallow, closely spaced transitions without relying on missing-data boundaries. The rolled Z return retained continuous rounded shoulders, a tight return, realistic occlusion, and different capture geometry. No detector was invoked during this audit.

Any change to the generator, family table, capture indices, sampling implementation, or token after this lock creates a new corpus version and invalidates V4 prospective status.

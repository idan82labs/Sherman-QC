# F1 calibrated raw-range synthetic V4 construction audit

Date: 2026-08-15

Ruling: **PASS — prospective execution authorized**

V4 was generated only after the geometry specification and generator hash were frozen. No bend detector was invoked before this construction audit was completed.

## Structural checks

- 30 opaque cases across 15 new profile families
- 318 intended bend centers
- 8 calibrated depth/validity frames per case
- 240/240 depth frames unique within their cases
- 240/240 validity frames unique within their cases
- source point count range: 33,418–131,874
- simulated sampling spacing: 0.4636–0.7076 mm
- native valid-fraction range: 0.0170–0.2776
- every intended bend represented in all eight calibrated views
- generator regression tests: 2 passed

## Visual audit

Every one of the 30 fused 3D construction overlays and every one of the 30 native eight-view depth sheets was manually inspected. The audit explicitly checked:

- intended centers lie on sampled geometry rather than empty space;
- small creases and 2.0–5.5 mm structures remain visually separable in multiple native views;
- broad bends appear as continuous rounded/multi-facet curvature rather than disconnected artificial folds;
- deep returns and hooks remain observable despite realistic self-occlusion;
- structured scan holes do not create or define the intended bend locations;
- the two captures of each family vary in pose, validity, occlusion, and noise while preserving the same physical profile;
- no family duplicates a V3 profile.

All fifteen families passed. PF05 (faceted low crown), PF08 (sub-three-millimetre ladder), and PF11 (paired micro-beads) deliberately approach the scanner observability limit: they compress at grazing angles but remain resolved in multiple other native views. PF02, PF03, PF07, PF10, PF12, PF14, and PF15 provide the requested rounded-facet and compound-shape stress.

## Frozen artifacts

- public manifest SHA-256: `d1fa7e1435d101092e1c33ebb1177ba8c2356f42f3dd09a78f6f64ccf0f80f2f`
- private manifest SHA-256: `d08f67856508170dfbeaedb3a52a00affb856b36f5075c9817d5db283a630a55`
- generation metadata SHA-256: `757aac237303b5ffb4850083f35b828240618ce0e98ad71ea187349ce0c30a99`
- public corpus tree digest: `a120d0af3f1ab588d60a57ea6c46095e990f28c24e1277aac1e551276ca9420f`
- canonical per-case observability tree digest: `927d4ff19788e301f95b4fc5b37f1a8bdc169648429dc4880084abd3b7c75a3b`

The canonical audit directory is `geometry-observability-by-case`. A noncanonical shared-directory render attempt stopped after a renderer safety collision and has no evidentiary role.

Private construction truth must now remain closed until V61 and V62 have completed primary/replay execution and the truth-free changed-case visual audit has been frozen.

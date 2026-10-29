# Calibrated raw-range synthetic V3 complex-geometry lock

Date: 2026-08-15

## Purpose

V3 is a new, non-confirmatory geometry stress corpus for the retained V61 synthetic research candidate. It is intentionally harder than V1/V2 in shape, bend scale, bend spacing, and radius—not just in sensor noise.

## Frozen corpus

- 15 new cross-section families, two independent calibrated raw-range captures per family: 30 cases total.
- native frame size: 384x288
- calibrated views per case: 8
- capture indices: 6 and 7
- acquisition noise, dropout, camera coverage, fusion density, and source point caps remain identical to V1/V2.
- evaluator meshes and bend centers remain private from inference.
- synthetic token: `SYNTHETIC_CALIBRATED_RAW_RANGE_V3_COMPLEX_GEOMETRY_NON_CONFIRMATORY`
- generator SHA-256: `42cfbe2f9cf588d99abc12978cc6241fa63df0c89dbf0041bddbe795c791e0cb`

## Geometry coverage

The family set includes:

- 2.2–4 mm micro-steps and 2.5–7 mm shallow creases;
- close parallel crease pairs and dense miniature joggle trains;
- mixed small and large bends in the same part;
- compound S-joggles, nested returns, asymmetric hats, channels, and labyrinth profiles;
- tight nominal fillets down to 0.55 mm;
- broad rolled/rounded transitions up to 9 mm nominal radius;
- multi-facet rounded crowns and pans rather than only sharp single hinges;
- profiles with up to 11 evaluator bend targets.

## Pilot gate

A two-case HF01 pilot was generated before locking the full corpus.

- both bundles passed the unchanged raw-range preflight;
- fused truth overlays showed all eight intended bend centers on coherent source geometry;
- all eight native views were inspected for both captures;
- the 3 mm micro-creases remain visible in multiple native views at scanner-like density;
- holes, occlusion, and grazing views make the task meaningfully harder without erasing the intended geometry.

Pilot artifacts:

- `output/research/f1-calibrated-raw-range-synthetic-v3-pilot2-20260815`
- pilot public manifest SHA-256: `74b3cc11e01480ed7a1e003a34bf8d33822326d7c48d3d5586505c77b03e9c5d`
- O001 preflight SHA-256: `85ea8c3e7b919fa702360a2741a82746352e4b5a9c30e68aca20286bc38d117d`
- O002 preflight SHA-256: `99bfc9d5be38d51abe2163259514c0e3826c891d60802d0181081266c76e8b5a`

## Execution discipline

1. Generate the 30-case corpus once from the locked generator.
2. Preflight every public bundle without opening evaluator truth.
3. Perform the construction-stage geometry observability audit for every case before inference. This audit may view synthetic truth only to reject invalid corpus construction; it may not change V61, thresholds, or scoring rules.
4. Freeze the observability ruling and generator/corpus hashes.
5. Execute V52 and V61 primary/replay routes without tuning or rescue reruns.
6. Audit every changed case visually in native frames before scoring.
7. Score once and retain/retire V61 based on the frozen aggregate and visual evidence.

## Evidence boundary

V3 can test whether the source-only architecture survives harder simulated geometry. It cannot establish performance on the newly received physical parts until their scanner export includes organized depth frames, intrinsics, and poses.


## 2026-04-11 — `CAD_B11` locality audit (no patch kept)
- audited `/Users/idant/Downloads/23553000_B_FULL_SCAN 1.ply` against `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/23553000_case_debug_20260411_b11b.json`
- confirmed recovered-face contamination grows materially at larger radii for `CAD_B11`
- confirmed narrower locality only moves `surface_classification` from roughly `121.76°` to `121.5°–121.6°`
- confirmed `profile_section` is materially worse on the same wing cluster (`CAD_B11 -> 126.29°`)
- decision: no new `CAD_B11` heuristic patch kept from this audit
- current best shipped result remains `CAD_B11 = 121.76°`, `WARNING`
- follow-up note: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_B11_LOCALITY_AUDIT_20260411.md`

## 2026-04-11 — `CAD_B11` explicit-face audit (no patch kept)
- confirmed shipped `CAD_B11` currently runs on recovered-face fallback because selected wing-bend payloads carry `surface1_id = null`, `surface2_id = null`
- reconstructed the exact legacy recovery config underlying the selected 16-bend baseline
- tested explicit legacy surface IDs on `CAD_B10/B11/B14/B15`
- result: explicit legacy-face measurement regressed `CAD_B11`, `CAD_B14`, and `CAD_B15`
- decision: do not patch the shipped path to force explicit legacy face IDs for the wing cluster
- follow-up note: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_B11_EXPLICIT_FACE_AUDIT_20260411.md`

## 2026-04-12 — `CAD_B10` / `CAD_B11` overlay audit (retry-margin patch rejected)
- generated bend-local fitted overlays for `CAD_B10` and `CAD_B11`
- confirmed `CAD_B10` benefits strongly from the short-obtuse retry path
- confirmed `CAD_B11` only improves marginally under narrower recovered-face retries
- tested a narrow retry-margin patch to accept smaller angle improvements in the short-obtuse loop
- rejected that patch after real rerun because `CAD_B11` regressed from `121.76°` to `122.38°`
- decision: keep shipped `CAD_B11 = 121.76°`, `WARNING`
- note: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_B10_B11_OVERLAY_AUDIT_20260412.md`

## 2026-04-12 — `CAD_B10` / `CAD_B11` 3D ROI audit
- rendered 3D ROI overlays with CAD bend line, CAD normals, and shipped scan-fit normals
- `CAD_B10` visually confirms an algorithmically corrected case near nominal `120°`
- `CAD_B11` visually supports a coherent but slightly steeper local geometry around the shipped `121.76°`
- current decision strengthened: keep `CAD_B11 = 121.76°`, `WARNING`
- note: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_B10_B11_ROI3D_AUDIT_20260412.md`

## 2026-04-12 — `23553000` wing bend ID map from drawing + CAD + shipped full scan
- reopened `/Users/idant/Downloads/שרטוט החלק 9.PDF` and confirmed `SECTION B-B` remains the nominal `120°` source for the wing family
- mapped the current shipped `120°` CAD bends into structural classes using CAD line locality and fitted face normals from `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/output/guard_pack_20260411/23553000_full.json`
- confirmed `CAD_B6` and `CAD_B10` are crown/apex folds, not the same structural class as `CAD_B11`
- confirmed `CAD_B11` is the rear left wall-to-upper-wing fold; its proper comparators are `CAD_B9` and secondarily `CAD_B15`, not `CAD_B10`
- decision: keep `CAD_B11` as a structurally correct assignment with `WARNING` status; do not force it toward the crown/apex result
- note: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/23553000_WING_BEND_ID_MAP_20260412.md`

## 2026-04-12 — production QC decision policy implemented
- split bend evaluation into explicit `completion_state`, `metrology_state`, and `positional_state`
- release logic now treats positional invalidity and unformed bends as hard failures
- out-of-tolerance / unmeasurable / unknown states degrade to warning when no hard blockers exist
- current live positional source is bend-line center deviation; per-bend heatmap contradiction remains a future override layer
- note: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/PRODUCTION_QC_DECISION_POLICY_20260412.md`

## 2026-04-12 — conformity decision model phase 1
- added compatibility-safe `release_decision` with `AUTO_PASS / HOLD / AUTO_FAIL`
- added first-class debug states for `correspondence_state` and `observability_state_internal`
- benchmark now emits separate completion / metrology / position / abstention scoreboards
- contradiction cases are exported explicitly instead of being buried inside blended MAE
- note: `/Users/idant/82Labs/Sherman QC/Full Project-qc-clean/docs/research/CONFORMITY_DECISION_MODEL_PHASE1_20260412.md`

## 2026-04-12 — consumer/report migration follow-up
- bend inspection PDFs now show `release_decision`, trust flags, and hold/block reasons
- bend-level exported report tables now expose explicit conformity states and position evidence
- legacy job/detail consumers now surface `release_decision` without dropping legacy `overall_result`
- benchmark output now writes dedicated `scoreboards.json` and `decision_contradiction_cases.md`

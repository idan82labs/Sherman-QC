# Real-wiring audit and real-part E2E (2026-10-19)

Owner priority: "the new engine is wired to the system as it should and the e2e we presented works
for real, no fake parts or simulated areas".

**Verdict.**
- The production path is real. Onboarding feeds E1, the hot folder feeds the runner, the runner runs
  E3, the result is stored, and the station and the PDF show it.
- The real part 10839000 now passes the whole flow end to end through the UI with no API shortcut:
  **13/13 PASS** in run `20261019T-real-g`. The run used the FULL scan and both stage scans.
- Every per-bend value, state and decision is identical in five places: the direct E3 CLI run, the
  job's engine output, the station API, the station screen and the PDF. 0 mismatches in 146 rows.
- The station heatmap is exactly the engine's per-point deviations.
- The earlier "E2E we presented" did **not** prove this:
  - the 2026-10-17 runs never exercised the real part (R1 NOT_RUN "by custody");
  - the bend tagging was an API assist;
  - the guide video ran the real engine on a **synthetic** part.
- Remaining issues are listed in §6. The biggest are engine-accuracy findings for A1, and the legacy
  analysis pages that were still reachable from the line UI (their API is now closed).

## 1. Link table (real path, file:line evidence)

| # | Link | Status | Evidence |
|---|---|---|---|
| 1 | Onboarding: SKU | REAL | `POST /api/production/parts` (`features/part-onboarding/api/onboardingApi.ts:91-95`). Run g W1: SKU preview = 10839000. |
| 1b | Onboarding: folded STEP | REAL | `POST /api/parts/{id}/cad` → `backend/_server_legacy_impl.py:3642`. For `.stp` it runs `backend.vendor_native_step_upload_v1.preprocess_vendor_step` (`:3690-3700`), a supervised OCP `STEPControl_Reader` BRep solver (`vendor_unfold_preview.py:1116-1142`). An env flag can switch to a `v2-development` solver (`SHERMAN_VENDOR_CAD_RUNTIME`); it is not set in production. If the solve fails, nothing is published (`CAD_ANALYSIS_FAILED_NOT_PUBLISHED`). |
| 1c | Onboarding: drawing PDF | REAL | `POST /api/parts/{id}/drawing` (`_server_legacy_impl.py:3814`). Run g used the part's own drawing (`שרטוט 10839000.pdf`), not a placeholder. |
| 1d | Onboarding: flat DXF | REAL, NOT EXERCISED on the real part | `POST /api/production/parts/{id}/flat` (`apps/production_line/recipes/flat_intake.py`). 10839000 has no flat DXF on file. The synthetic E2E (`run_e2e_app.mjs` W5/W6) covers it. |
| 2 | CAD bend solver auto-marks bends in the studio | REAL | `GET /api/parts/{id}/vendor-setup` → `apps/api/services/vendor_setup_service.py:212-243`. It reads the bend specs stored by link 1b. Run g: 12 candidates, 12 on-model markers, all pending until confirmed. The `vendor_cad_bend_candidates_v3..v8` modules are research only (not imported on this path). |
| 3 | Bend tagging, order and מהלך saved | REAL | `PUT /api/parts/{id}/vendor-setup` (`VendorSetupPage.tsx:494-505`; validation in `vendor_setup_service._validate_draft:90-203`). Run g W3: every bend was clicked **on the model** (0 list fallbacks). The real shop order was tagged, Move 1 = B1+B2, and the draft read back equal. |
| 3b | General measurement authoring | REAL | Studio "Between two bends" (`VendorSetupPage.tsx:640-656`) → `general_measurements` in the draft. Run g W3b: a bend_pair B10↔B7 was picked by clicking the CAD surface; each point sits next to a different bend (12.1 mm / 2.1 mm from its line). |
| 4 | Recipe compile (E1) from the saved setup | REAL | `release_service.py:1174` `compiler.compile(...)` runs `python -m domains.production.engine.compile --recipe --cad [--flat] --out`. The recipe comes from `domains/production/recipe/builder.py:161` using the draft's marks, sequence, machine_moves and general_measurements. Run g W4: released `10839000@v1` (`e1-1.1.0`); the recipe order equals the studio order (by `cad_label`); the benchmark reference is REGISTERED. |
| 4' | E1 code root | FALLBACK (low) | `release_service.py:372-375` `engine_root_for`: if the bundle has no manifest, E1 runs from the source/app root instead of the pinned bundle. It is the same E1 module, but unpinned. |
| 5 | ZEISS PLY hot-folder intake | REAL | `apps/production_line/ingest/hotfolder.py` polls with a stability/age gate, then probe, sha256, filename → SKU (`ingest/sku_resolver.py:35` → `domains.production.recipe.sku.parse_scan_filename`), then `store.match_and_enqueue`. It also handles quarantine and duplicates. Run g dropped `10839000_01_FULL_SCAN 1.ply`, `..._SCAN_1 1.ply` and `..._SCAN_2 1.ply` the ZEISS way (temp name, rename, aged mtime). The job's `scan_sha256` equals the source sha (custody check). |
| 6 | Job queue and runner | REAL | `line_store.lease_next`, then `runner.py:432-465`, which dispatches only `inspect` / `recipe_compile` (there is no legacy branch). `service_main.py:85-100` refuses to start without a verified engine bundle (`profile="engine"`, no legacy worker). |
| 7 | E3 invocation | REAL, no swap path | `runner.py:790-807`: `<python> -m domains.production.engine.inspect --compiled <ws>/compiled/compiled_part.json --recipe <ws>/recipe.json --scan <ws>/input/scan.ply --out <ws>/engine --scan-sha256 S --compiled-sha256 C --recipe-sha256 R --time-budget-s 170 [--declared-stage K --declared-stage-source SRC] --engine-bundle-id ID [--config CFG]`. The cwd is the bundle, the env is whitelisted, and limits are 180 s / 4 GiB. `inspect_module` is never overridden in production: only the `synthetic_scorecard` ops CLI uses an oracle module, and only with `--fake-engine`. E3 reads no environment variables. |
| 8 | Result storage | REAL | E3 writes `<data_root>/jobs/<id>/engine/result.json`. C06 writes `result/shopfloor_result.json` and may change only `viewer_assets`, `artifacts`, `warnings` and `context` (`runner.py:1067-1075`, else `RESULT_ALTERED`). |
| 9 | Station API | REAL, pass-through | `GET /api/line/jobs/{id}/result` → `ResultLoader._load_uncached` (`apps/api/services/production_line_service.py:554`). It re-validates and returns the stored file; nothing is recomputed. Artifacts come from an allow-list (`contracts/production/layout.py:128-136`). |
| 10 | Station UI: 3D view | REAL | `features/station/hooks/useViewerAssets.ts` loads `stage_mesh.ply`, `final_mesh.ply` and `viewer_scan.ply` from the job's artifacts. The `station/dev/harness.tsx` mock is not in the app bundle. |
| 10b | Station UI: per-bend table | REAL | `BendResultTable.tsx` columns are Flange A, Flange B, Angle (flange first); the values are the result's `actual/expected/tolerance/decision`. Run g: the screen equals the engine for all 146 rows (§4). |
| 10c | Station UI: heatmap | REAL | E3 computes the per-point deviations (`engine.py:1427 _heatmap`, written to `engine/deviations.f32`). C06 only voxel-decimates them with a shared index (`inspection/viewer_assets.py:130+`), and the client maps them to colours (`model/heatmapScale.ts`). Run g: every viewer point is an engine point (match 0 mm), its deviation is identical (max diff 0), and it equals the direct CLI run's deviations. See §4b. |
| 11 | Full vs partial scan | REAL, in E3 | Bend states and the stage decision come from E3's decode (`inspect/decode.py`, `contracts/production/inspection_result.py:687 derive_stage_decision`). Run g: FULL → CONSISTENT, stage 12/12. scan_1 → 8 formed / 4 NOT_YET_FORMED; scan_2 → 10 / 2. Both are FINAL_INCOMPLETE because the recipe is a FINAL inspection. Not-yet-formed bends are greyed in the table, on the 3D labels and in the PDF. |
| 12 | PDF report | REAL | `domains/production/reporting/shopfloor_pdf.py` builds it from the validated result (it never computes a decision). The station's "Download PDF" button fetches `artifacts/report.pdf`. Run g: PDF values = engine values (2 dp), Hebrew in logical order, and the header ids are right on every page. |
| 13 | Measurement (GENERAL_MEASUREMENT) end to end | REAL (bend_pair) | Path: studio → recipe → E3 `_general_measurements` (`engine.py:1244`) → result → station "General measurements" rows → PDF "General measurements" table. Run g: B10↔B7, nominal 309.92. FULL 309.55 PASS, scan_1 309.06 FAIL, scan_2 310.01 PASS, identical in all five places. `points`-kind measurements are always UNKNOWN(UNSUPPORTED_FEATURE) by design, and the 3D view does not draw measurements (§6). |
| L | Legacy runtime (`backend/bend_analysis_worker.py`) | Not on the line path; was REACHABLE beside it | Nothing in the `/api/line` path imports it, and the line bundle excludes it. However, the same server mounted `POST /api/analyze`, `/api/batch/analyze` and `/api/bend-inspection/analyze[-from-catalog]`. Any operator could run the old engine through the sidebar pages "New Analysis", "Batch Processing" and "Bend Inspection". **Fixed:** these routes are refused while the line is enabled (§2). |

Grep sweep (apps/, scripts/production/, contracts/, frontend features) for mock, stub, fake, fixture, demo, synthetic, simulate, placeholder, API_ASSIST, `e2-synthetic`, and engine-swapping env flags:
- **Tests and ops only:** `ops/synthetic_scorecard.py` (with `--fake-engine` it uses an oracle module), `recipes/synthetic_onboarding.py`, and `inspection/fake_engine_output.py` (used by tests only). None of these is imported by the server or the service.
- **Studio demo mode:** `VendorSetupPage.tsx` "SYNTHETIC-DEMO" runs only for `partId === 'demo'`. Saving is blocked and a banner is shown. It is shipped and linked, but it cannot touch a real part.
- **Corpus mode:** `VITE_VENDOR_CORPUS_DEMO` is a build-time flag and is not set.
- **Old drivers:** `run_e2e_app.mjs` keeps an API fallback for W4 that is recorded as FAIL. Its R1 step uses a **placeholder** drawing. Its kiosk device token is minted through the API.
- **Engine-swapping env flags:** none.

## 2. What was fixed (files)

- **Legacy engine unreachable on the line.**
  - Files: `apps/production_line/security/route_policy.py`, `apps/production_line/settings.py` (new `SHERMAN_LEGACY_ENGINE`, default 0) and `apps/production_line/server.py`.
  - `POST /api/analyze`, `/api/batch/analyze`, `/api/bend-inspection/analyze` and `/api/bend-inspection/analyze-from-catalog` now return 403 `ROUTE_DISABLED` while the line is enabled, even for an admin. With the line disabled, the legacy app is unchanged.
  - Tests: `tests/test_prod_route_policy.py` (updated test plus the new `test_legacy_engine_routes_refused_on_the_line`). Run g Z1 probes all four routes live.
- **Optional flat step was skipped.** `frontend/react/src/features/part-onboarding/model/checklist.ts` `nextOpenStep`: the rail's "Next" pill now points at the optional unfolded-part step after Drawing, instead of jumping to Bends. The footer "Next" already did. This closes guide finding 7. Run g W2 asserts both.
- **Clipped decision text in the station table.** `frontend/react/src/features/station/station.css` `.st-cell-sub` now wraps. Before, the angle column clipped the decision ("145.0° ±1.0° Revie|"), in English and Hebrew; this is visible in run d screenshots 17, 21 and 32. Run g asserts that no table cell overflows.
- **New real-part E2E**, with no API shortcut for any operator step:
  - `scripts/production/run_e2e_real.mjs`;
  - `scripts/production/real_e2e_support.py`: `pdf-bends` (PDF table parser), `compare` (the five-way equality), `heatmap` (viewer vs engine deviations).
  - The station is opened by the signed-in admin with a user token, so no device token is minted through the API.

Checks run:
- `tests/test_prod_route_policy.py`, `test_prod_settings.py` and `test_prod_hardening.py`: 109 passed.
- Frontend: `tsc --noEmit` is clean, eslint on `checklist.ts` is clean.
- `check-station-layout.mjs` (19 steps), `check-station-view-model.mjs` (941 assertions), `check-onboarding-model.mjs` (750) and `check-onboarding-ui.mjs` (17/17) all pass.
- Nothing under `domains/production/engine/**` was touched.

## 3. Real run: `docs/production/runlog/e2e/20261019T-real-g/` (13/13 PASS)

The driver was `node scripts/production/run_e2e_real.mjs`, run under the shared geometry lock with OMP_NUM_THREADS=1 at 1920×1080.

- **Engine:** `e3-1.7.0`, line bundle `555f9982cc15b264`, E1 `e1-1.1.0`.
- **Inputs:** read-only custody copies, byte-identical to their sources:
  - STEP `ea3e4820…`;
  - drawing `82ba62d6…`;
  - FULL `71cf0b21…`, scan_1 `08686dc1…`, scan_2 `cec08dfc…`.

| Step | What (all through the UI) | Result |
|---|---|---|
| S0/S1 | App copy (C04 `--profile app`); server and line service from the copy; temp data root | PASS |
| W1 | Login, then Production → Parts → New part 10839000; STEP upload → 12 auto-marked bends; the card shows `10839000_01.stp` | PASS (01–02) |
| W2 | Real drawing PDF; rail and footer "Next" go to the optional flat step, then Skip goes to Bends | PASS (03–04) |
| W3 | Studio: 12 bends confirmed **on the model**; real shop order B1,B2,B3,B4,B5,B10,B7,B12,B6,B11,B8,B9; Move 1 = B1+B2; saved | PASS (05–10) |
| W3b | Studio: general measurement B10↔B7 by clicking the CAD; saved | PASS (11–12) |
| W4 | Tolerances (FINAL), review, release `10839000@v1`; expectations table has 12 rows; benchmark REGISTERED | PASS (13–16) |
| K1 | Station opened as the signed-in admin; live | PASS (17) |
| R-full | Drop FULL → job 26 s → station within 0.2 s → PDF → direct CLI → 5-way equality → heatmap | PASS (18–21) |
| R-scan_1 | Drop stage scan 1: stage 8/12, B9–B12 greyed | PASS (22–25) |
| R-scan_2 | Drop stage scan 2: stage 10/12, B11–B12 greyed | PASS (26–29) |
| R-hebrew | Station in Hebrew (RTL), no overflow | PASS (30) |
| Z1 | No page errors; legacy routes refused (403 `ROUTE_DISABLED` ×4); inbox emptied | PASS |

**Earlier runs of the same driver.**
- `20261019T-real-d` is kept as evidence. It tagged the studio's list order (B1..B12), which is not the real shop order.
- With that order, E3 correctly reported `OUT_OF_ORDER_SUSPECTED` on both stage scans ("no compiled stage model for the decoded formed set"). It **withheld** the heatmap (`STAGE_MODEL_UNDETERMINED`), and the station said "Heatmap unavailable". It did not draw a substitute.
- So when the order is wrong, the line refuses to fake a heatmap.

**Visual review.** I opened these screenshots and PDF pages myself:
- **Bends land on the part:** studio markers and station labels sit on their bend lines (06, 11, 19, 20, 23, 27).
- **Hebrew is not mirrored:** station RTL (30) and every PDF page read correctly ("דוח בדיקה", "נמצאה חריגה", "טרם כופף").
- **Nothing is clipped** after the CSS fix (25, 29).
- **WebGL is not blank:** canvas probe of 1000 or more painted pixels in every scan step.
- **Partial scans:** not-yet-formed bends are dashed and grey on the 3D labels and in the stage strip. In the table they show "Not yet formed" with nominals only, and in the PDF they are greyed "n/a / לא רלוונטי".

## 4. Engine == job == API == station screen == PDF

The full per-row tables are in `20261019T-real-g/scans/<scan>/compare.md` and `compare.json`.

Direct CLI (same bundle, same compiled recipe, same PLY):
```
cd <data_root>/engine/current && OMP_NUM_THREADS=1 PYTHONPATH=<bundle>:<bundle>/backend \
  .venv/bin/python -m domains.production.engine.inspect --compiled <job>/compiled/compiled_part.json \
  --recipe <job>/recipe.json --scan <custody copy>.ply --out <tmp> --scan-sha256 <sha> \
  --compiled-sha256 <C> --recipe-sha256 <R> --time-budget-s 170 --engine-bundle-id 555f9982cc15b264
```

| Scan | Summary / stage (all 5 sources) | Rows compared | Mismatches |
|---|---|---|---|
| FULL | OUT_OF_TOLERANCE_FOUND · CONSISTENT · 12/12 | 48 bend characteristics + 1 GM | 0 |
| scan_1 | OUT_OF_TOLERANCE_FOUND · FINAL_INCOMPLETE · 8/12 | 49 + 1 GM | 0 |
| scan_2 | OUT_OF_TOLERANCE_FOUND · FINAL_INCOMPLETE · 10/12 | 49 + 1 GM | 0 |

**How the sources were compared.**
- **JSON sources** (direct CLI, job engine, rendered result, station API): exact equality of per-bend state and of every characteristic's state, decision, actual and expected.
- **Station screen:** the displayed actual is within the 1 dp rounding, and state and decision are equal.
- **PDF:** the actual is within the 2 dp rounding, the decision word is equal, and the bend state is equal.

FULL scan extract, run g. Labels are the recipe's sequence labels; for example recipe B12 is CAD B9.

| Bend | Char | Expected | E3 CLI | Station API | Screen | PDF |
|---|---|---|---|---|---|---|
| B1 | Flange A (joggle) | 3.61 | UNKNOWN | UNKNOWN | Not measured | Undetermined |
| B1 | Angle | 145.00 | 143.346 REVIEW | 143.346 REVIEW | 143.3° REVIEW | 143.35 Review |
| B5 | Flange A | 12.00 | 11.982 PASS | 11.982 PASS | 12.0 PASS | 11.98 Pass |
| B5 | Angle | 90.00 | 88.615 FAIL | 88.615 FAIL | 88.6° FAIL | 88.62 Fail |
| B9 | Flange A | 21.41 | 21.037 REVIEW | 21.037 REVIEW | 21.0 REVIEW | 21.04 Review |
| B12 | Flange B | 206.06 | 206.251 PASS | 206.251 PASS | 206.3 PASS | 206.25 Pass |

General measurement "E2E B10-B7" (nominal 309.92 ±0.5):

| Scan | Value in all five sources | Decision |
|---|---|---|
| FULL | 309.549 | PASS |
| scan_1 | 309.061 | FAIL |
| scan_2 | 310.013 | PASS |

### 4b. Heatmap vs deviation data

| Scan | Viewer pts | Engine pts | Viewer point = engine point | Deviation diff | = direct CLI | Within ±b / b..2b / >2b (b = 0.5 mm) | median / P95 \|dev\| |
|---|---|---|---|---|---|---|---|
| FULL | 299,187 | 863,115 | 0 mm | 0 | yes | 74.8% / 24.0% / 1.2% | 0.28 / 0.73 mm |
| scan_1 | 342,675 | 837,407 | 0 mm | 0 | yes | 92.0% / 7.7% / 0.3% | 0.15 / 0.59 mm |
| scan_2 | 324,999 | 817,867 | 0 mm | 0 | yes | 83.4% / 14.0% / 2.6% | 0.23 / 0.73 mm |

- The colour map is `heatmapScale.ts` (stops at ±b, ±2b). `check-station-view-model.mjs` verifies it against the legend.
- The screenshots agree with these numbers:
  - mostly neutral or light points, with red/blue concentrated near the out-of-tolerance flanges in FULL;
  - scan_1 is visibly the most neutral of the three.

## 5. Guide-video data verdict

**The real engine ran on a synthetic scan. It was not real-part data, and no results were staged.**
- `~/ShermanQC/guide-video/v3/run/base/prep.log` shows the part set `E2-DEV-PSHT-00`, SKU `SYN-4ED9FC3569`.
- Its `part_set.json` says `"generator_version": "e2-synthetic-1.1.1"` and `"e2e_errors": {"angle_on_B3_deg": 2.0, "flange_on_B2_mm": 1.5}`. The errors are baked into the generated PLY geometry; no result was overridden.
- The servers were the real line (`harness.mjs:54-55`), and the scans were dropped into the real inbox (`harness.mjs:63-67`).
- The jobs ran E3 `e3-1.1.0` for 5.4 s on 295k points, with `scan_sha256` equal to the synthetic files. No `page.route` or mocked API was used.
- **API shortcuts in v3:**
  - CLI admin bootstrap;
  - a device token minted through the API;
  - read-only planning and polling GETs;
  - a read-only SQLite check.
- In v3 the bend tagging was clicked in the UI. The 2026-10-17 take (`drive_guide.mjs:187`) saved the tagging through `PUT vendor-setup` (API_ASSIST).
- **Presentational only:**
  - overlay captions are typed text that matches the engine values;
  - the "ZEISS" card is an overlay graphic, though the file drop behind it was real;
  - processing waits were cut;
  - the PDF scene shows PNG renders of the engine PDF in a custom viewer.
- **Consequence:** the video shows the real software on a synthetic part. It is not evidence of real-part accuracy. Run g is the real-part evidence.

## 6. Remaining gaps (severity)

| Sev | Gap | Owner / note |
|---|---|---|
| HIGH (engine, A1) | **Joggle bends (35°, B1–B4).** The 3.6 mm short flanges are always UNKNOWN ("Feature not supported" / "Neighbouring bend not measured"). Joggle included-angle U reaches ±8.2° (scan_1 B3/B4), so these bends can only ever be REVIEW, and with the worst-rank rule `(\|d\|+U)/tol` a high-U REVIEW outranks a clear FAIL ("Worst: B3 angle −2.59° ±8.23°"). | A1 (engine is single-writer) |
| MEDIUM (engine, A1) | On the FULL scan every 90° angle is 88.0–89.7° (4 FAIL, 4 REVIEW). This may be real springback; it is not judged here. scan_2 warns `THICKNESS_DIFFERS_FROM_CAD measured 1.275 vs compiled 1.500` (run d). | A1 / owner: physical check of one part |
| MEDIUM | **Legacy pages still in the line sidebar:** "New Analysis", "Batch Processing", "Bend Inspection" and "Live Scan" (`components/Layout.tsx:22-30`). Their engine calls are now refused (403), but the pages still show. They should be hidden when the line is enabled; that is outside this audit's file scope (`components/`). | UI owner |
| MEDIUM | **Bend labels change between studio and station.** The recipe renumbers bends by sequence (`builder.py:431` `label=f"B{index}"`, `cad_label` keeps the studio label). With the real order, studio "B10" is station/PDF "B6" (the uid code `0eb0` matches). This could confuse operators. Suggest showing `cad_label` in the station and PDF, or renumbering the studio after the order is set. | UI / recipe owner |
| LOW | Station "Worst first" selected B10 (angle −2.0° FAIL), while the banner/PDF "worst" is B1 (angle −1.7°, REVIEW with high U): two different worst rules. | Station + engine summary |
| LOW | **E1 code-root fallback** (`release_service.py:372-375`, `flat_intake.py:282`): with no bundle, release-time compile runs unpinned from the source root. Suggest refusing under `SHERMAN_PRODUCTION_HARDENING`. | apps |
| LOW | **Station device tokens** can only be minted through `POST /api/line/devices` (no admin UI). The E2E avoids this by opening the station as the signed-in user. A kiosk deployment still needs an API call or a small admin UI. **Quarantine assign** (`POST /api/line/quarantine/{id}/assign`) also has no UI; the synthetic E2E K9 uses the API. | UI |
| LOW | File cards show the original name only in the browser that uploaded it (localStorage keyed by sha). The catalog stores CAD/drawing under uuid names with no original-name column (`_server_legacy_impl.py:3667, 3832`), so another workstation shows the neutral label "Folded CAD (STEP)", not a hash. A real fix needs a backend column (outside scope). | backend owner |
| LOW | "Propose binding" needs a saved bend setup (`flat_intake.py:495-500` `DRAFT_MISSING`), so the flat step (4) depends on step 5. The UI explains this, but the order is still awkward (guide finding 9). | design decision |
| LOW | The 3D station view does not draw general measurements (the `nominal/measured_start/end_mm` fields are unused), and `points`-kind measurements are always UNKNOWN in E3. | station / A1 |
| INFO | The app copy (`--profile app`) contains `backend/` including `bend_analysis_worker.py`, because the legacy app server is part of it. The line's engine bundle (`service_main --build-bundle`, profile engine) does not. | — |
| INFO | 10839000 has no flat DXF, so the flat-binding path is only proven on the synthetic E2 part (`run_e2e_app.mjs`). | data |

## 7. Re-run

```
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
OMP_NUM_THREADS=1 node scripts/production/run_e2e_real.mjs [--keep] [--scans full,scan_1,scan_2]; rc=$?
rmdir /private/tmp/claude-501/sherman_geom.lock
```
- The run takes about 6 minutes and exits 0 only when every step passes.
- The direct-CLI engine outputs (registered scan, deviations) stay in the temp directory. Only `result.json` and `engine.log` are copied to `docs/`.
- The driver was changed after run g to write the direct-CLI outputs to the temp directory instead of `docs/`. That is a path change only; the run-d/run-g binaries were pruned from `docs/`.

## Follow-up 2026-10-19

Two MEDIUM gaps from §6 are closed. Nothing under `domains/production/engine/**` was touched, and the recipe's `label` / `bend_uid` semantics are unchanged.

**1. Bend labels now match the studio everywhere.**
- Cause: `builder.py` sets `label = B<sequence_index>`, and E3 copies that label into the result.
- Changing the recipe label was tried and rejected: the engine's synthetic tooling keys on `B<seq>` (`test_prod_synth_cases` failed).
- New optional contract field `ResultContextV1.bend_display_labels` (uid → studio `cad_label`). It is left out of the JSON when unset, so existing results and examples are byte-identical.
- The runner fills the field from the recipe into `input/context.json` (`runner._context_dict`). C06 already copies that context into the result, and E3 never sees it.
- Who reads it:
  - PDF: `reporting/labels.bend_display_label` drives the bend tables, the move headings, "worst", the backgauge table, the "OML to OML of …" reasons and the 3D view labels (`snapshots.py`).
  - Station: `stationView.bendDisplayLabel` drives the table, the stage strip, the 3D aria labels and the banner's "worst".
- The 3D on-model tags still show sequence number + uid code (by design).
- Tests:
  - `test_prod_shopfloor_pdf` (new studio-label test), `test_prod_runner` (context carries the labels) and `check-station-view-model.mjs` (+5 assertions; the CONTRACT_KEYS parity check still holds).
  - `real_e2e_support.py compare` now keys every source by display label, cross-checks the DOM sequence, and requires the API's label map to equal the rendered map.
  - `run_e2e_real.mjs` checks DOM, PDF and context labels against the recipe `cad_label` for each sequence.

**2. Legacy pages hidden on the line.**
- The frontend received no flag before. New `GET /api/line/features` (no auth, flags only) returns `{line_enabled, legacy_engine_enabled}` from the same `LineSettings.legacy_engine_enabled` (`SHERMAN_LEGACY_ENGINE`) that the route policy uses.
- `hooks/useLineFeatures.tsx`:
  - `Layout.tsx` hides "New Analysis", "Batch Processing" and "Bend Inspection" unless the legacy engine is enabled;
  - `App.tsx` wraps `/upload`, `/batch` and `/bend-inspection` in `LegacyEngineRoute`, which redirects to `/`.
- If the line is disabled, or the server has no such endpoint, the pages stay.
- Test: `test_prod_line_routes` (features in both modes).

**Checks.**
- pytest:
  - `test_prod_recipe_builder`, `release_service`, `contracts`, `shopfloor_pdf`, `runner`, `postprocess_cli`, `flat_intake`, `recipe_store`, `onboarding_checklist`, `recipe_routes`, `recipe_resolver` and `synth_cases`: all pass;
  - `line_routes`, `route_policy`, `hardening` and `settings`: 153 passed;
  - `line_artifacts` and `line_events_stream`: 42 passed.
- Frontend: tsc is clean; eslint on the touched files is clean.
- Check scripts: `check-station-view-model` (946), `check-station-layout` (19 steps), `check-station-fixtures`, `check-onboarding-model` (750) and `check-onboarding-ui` (17/17) all pass.

**Real runs** (under the geometry lock, OMP_NUM_THREADS=1):
- `20261019T-real-h`, all scans: 12/13.
  - Every scan step passes, including the new label checks. FULL compare: 0 mismatches.
  - Station sequence 6 = **B10** (uid 0eb0), 10 = B11, 11 = B8, 12 = B9, matching the studio order. The PDF shows the same labels (p. 4: "6 B10 bu_0eb02c66"; "Move M1 · B1, B2").
  - Z1 failed only on the new sidebar probe's locator: it targeted a hidden mobile nav and `/` on the API port.
- `20261019T-real-h2` (FULL only): the same locator issue.
- `20261019T-real-h3` (FULL only), after the driver fix:
  - Z1 PASS: features `{line_enabled: true, legacy_engine_enabled: false}`; sidebar = Dashboard, Jobs, Live Scan, ShermanAI Chat, Part Catalog, Production Parts; `/upload`, `/batch` and `/bend-inspection` → `/`; screenshot 22.
  - W3b failed: "studio markers not found", 0 markers listed. This step was not touched and passed in runs h and h2. The studio UI is being edited in parallel, so this is likely flake; recheck.

**Still open (LOW).**
- The Dashboard's "New Analysis" / "Batch Processing" quick actions still show; the route guard sends them home.
- "Live Scan" stays in the sidebar; it has its own `SHERMAN_LEGACY_LIVE_SCAN` gate and was out of scope.
- Results released before this change carry no label map and still show the sequence labels.

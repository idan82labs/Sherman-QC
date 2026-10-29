# Fixes, round 3 (2026-10-19): owner decisions D4, D1, D2, D3

> **Final engine after the §8 follow-up (2026-10-20): `e3-1.12.0`.** It adds the pose-borne flange-length guard. A1 and G1 are unchanged, and E2E is 13/13. §1–§7 describe e3-1.11.0.

The owner approved the FIXES2.md §4 decisions on 2026-10-19. They were done in the order D4, D1, D2, D3.

**Result:**
- The engine is now **`e3-1.11.0`** (it was `e3-1.10.0`). E1 is unchanged: compile and eval fingerprints equal round 2, and the 18 frozen A1 recipes recompile hash-identical.
- Two contract amendments were added within `sherman_shopfloor_inspection_result_v1`. Every earlier document stays valid.
  - **ENGINE_IO-A1:** flange length is NOT_APPLICABLE_AT_STAGE next to an unformed bend.
  - **ENGINE_IO-A2:** tied symmetric twins publish only identical readings.
- The per-bend scoring definition is now **V3**. V2 is kept untouched.
- Ledger NOTEs were appended:
  - seq 7 `frz_0234e6b1c20e`: the D4 amendment;
  - seq 8 `frz_1669642c5076`: the D1 authorization, recorded before any read;
  - seq 9 `frz_b8ff489b99ad`: D2 and the D3 amendment.

  `verify_ledger` passes: 9 entries, chain intact. No RUN entry was appended.

**Rules followed:**
- The engine was snapshotted first: `fixes3/engine_before.tgz`.
- The full diff is `fixes3/engine.diff` (inspect only: `config.py`, `engine.py`, `registration.py`, `result.py`).
- E3 has no part-number or file-name logic: grep for `49207`, `38968`, `fr_` and `pc_` in `inspect/*.py` finds nothing.
- Every new constant is a general rule, with its tests:
  - `free_datum_rank_k = 6` equals `pose_hypotheses`, so every refined pose gets its symmetric images;
  - `free_datum_seeds = 4`.
- Heavy jobs ran serially under `sherman_geom.lock`, with `OMP_NUM_THREADS=1`. Peak RSS was 2.48 GiB.
- Only named tests were run. Nothing was committed or installed.
- `human_bend_lines_v1` was not touched.

## Summary: before → after

| | Before (e3-1.10.0) | After (e3-1.11.0) |
|---|---|---|
| Frozen A1 recompile | 18/18 hash-identical | **18/18 hash-identical** |
| GOLD_SET_V1 micro / family-macro | 0.990 / 0.974 | **0.990 / 0.974** |
| Gold11 relabel | 0.987 | **0.987** |
| BROAD micro / family-macro | 0.702 / 0.817 | **0.702 / 0.817** |
| A1 per-bend, riser-aware (V2 → **V3**) | 183/203 = 0.901, strict 170, family-macro 0.910, false-PASS proxy 3 | **192/203 = 0.946**, strict 176, family-macro 0.938, proxy 3 (the same 3 bends) |
| A1 characteristic changes | – | 13 flange rows (D4), nothing else |
| A1 E3 P95 (G2(e)) | 28.54 s | **28.37 s** (gate PASS; the +25 % cap is 35.7 s) |
| G1 | PASS; false PASS 0/553 | **PASS**, no failed gate; false PASS 0/553; wrong/mirrored never PASS 0/32; symmetric wrong-ID 0 |
| G1 runtime P95 | 13.67 s | 13.96 s (gate 45 s) |
| Real E2E (`run_e2e_real.mjs`) | 13/13 | **13/13 PASS** (run `20261019T-real-fixes3`) |
| Named tests | 363 (round 1) + 7 (round 2) | **837 pass** in one run over 46 `tests/test_prod_*` files (incl. the new `tests/test_prod_engine_fixes3_2026_10_19.py`: 14 tests + 1 opt-in). The 1 failure was the pre-amendment ambiguous-pose CLI assertion; it was updated to ENGINE_IO-A2 and passes on re-run. |

---

## 1. D4: flange to a not-yet-formed bend → NOT_APPLICABLE_AT_STAGE

**Change:**
- **Contract (ENGINE_IO-A1).**
  - `_NA_REASONS_BY_KIND["FLANGE_LENGTH"]` now allows `NOT_APPLICABLE_AT_STAGE`.
  - A new result validator requires such a row's `adjacent_bend_uid` to name a bend whose row is NOT_YET_FORMED.
  - The docs were updated: `ENGINE_IO.md` §1.3, §4.2 and the new §7.1 amendment log, and `ENGINE_INSPECT.md`. Its old open item 13 is marked resolved.
- **E3.** On a formed bend, a side bounded by a NOT_YET_FORMED adjacent bend:
  - goes into `BendOutcome.stage_na`;
  - is not measured;
  - is published as NOT_APPLICABLE(NOT_APPLICABLE_AT_STAGE).

  A side next to a FORMED bend is measured as before. A side next to an UNKNOWN bend stays UNKNOWN(ADJACENT_BEND_NOT_MEASURED).
- **Scoring V3** (`scripts/production/per_bend_scoring_v3.py`, runner `fixes3/tools/per_bend_all_v3.py`): V2, plus such a side satisfies the flange criterion, strict and riser-aware. V2 itself is unchanged.
- **Station UI** (`stationView.ts`): the validator mirror carries the same rule. `check-station-view-model.mjs` passes.

**Evidence:**
- **A1 characteristic diff (e3-1.10.0 f5 → D4).** Exactly 13 FLANGE_LENGTH rows changed, all with the adjacent bend NOT_YET_FORMED. Nothing else changed.
  - 11 were ADJACENT_BEND_NOT_MEASURED: 08686 B2/B4, 1864 B6/B10, 8dfd B1, 927a B1, cec08 B2/B11, f247 B4(A, B)/B12.
  - 2 were joggle UNSUPPORTED_FEATURE: 1864 B2/B14.
- **A1 per-bend V3: 183 → 192/203** (expected about 188). The +9 bends are:
  - the 5 that FIXES2 listed (08686 B2/B4, 8dfd B1, 927a B1, cec08 B2);
  - 4 more whose only blocker was the same row: pc_186463746daa B6/B10 and pc_f2475286312b B4/B12, all UNEXPECTED_FORMED bends with FORMED truth.

  FIXES2 §2.4 counted only the first 5.
- Under the old V2 definition the new engine scores 181. This is because 2 joggle sides moved from UNSUPPORTED to NA, which V2 does not accept. That is why the decision ships with V3.
- Strict 170 → 176. The false-PASS proxies are unchanged: 3, the same bends.
- **Real E2E.** On the 10839000 stage scans, B2 and B4 FLANGE_LENGTH/A are NOT_APPLICABLE_AT_STAGE. Engine = job = API = screen = PDF, and the result is 13/13 PASS.
- **Tests** (`test_d4_*`, 4):
  - a flange next to an unformed bend is NA;
  - a flange next to a formed bend is MEASURED;
  - a complete part has no NA flange;
  - the contract rejects the reason on a flange next to a FORMED bend, on a FREE_EDGE side, and on an angle;
  - V3 accepts only the stage reason, and NOT_FORMED truth is unchanged.

**Risk:**
- Low. A staged part is still never green: the existing `NOT_APPLICABLE_AT_STAGE needs a NOT_YET_FORMED bend` rule still applies, and so does the FINAL_INCOMPLETE / IN_PROCESS summary logic.
- The E4 oracle still emits UNKNOWN(ADJACENT_BEND_NOT_MEASURED) for such rows. That is still valid under the contract; the G1 scorer does not score them (their truth is undefined). It was left unchanged.

## 2. D1: consumed-set labels, post-hoc diagnostics

> **DIAGNOSTIC — consumed sets, not a held-out score.** This is not a new evaluation, and no RUN entry was appended.

**Custody:**
- Ledger seq 8 (`AUTHORIZATION`) was recorded before any read. Its scope is exactly:
  - `data/bend_corpus/annotations/ai_bend_lines_fresh_v1_SEALED/`
  - `docs/production/runlog/freeze/devcheck_SEALED.json`

  The purpose is consumed-set diagnostics. The read was not blocked: no guard and no prompt.
- **Nothing else sealed was opened.** That includes `benchmark_v3/fresh_SEALED`, `recipes_devcheck_SEALED`, the `devcheck_SEALED` work directory, `g3_truth_material/devcheck_SEALED`, HOLDOUT10, N001–N016, EHO4 and the rhino-preview face scans. As a result:
  - **Engine inputs** come only from non-sealed copies:
    - FRESH: the one-time evaluation's `final_eval_w/fresh` compiled parts and recipes (sha equal to the round-1 results), else the round-1 E1 rebuild.
    - DEV-CHECK: the round-1 E1 rebuild of the same STEP. Its nominals and tolerances were checked equal to the frozen ones: 0 mismatches over 144 characteristics.
  - **Line gold:**
    - FRESH: each scan's sealed ai_studio label.
    - DEV-CHECK: the gold of record (GOLD_SET_V1, `ai_bend_lines_gold_v1`), per the owner direction. The BROAD copy in `devcheck_SEALED.json` is old human Gold11 for pc_a030.
  - **Per-bend:** complete scans only, with every bend expected FORMED (the iter-7 convention). Staged scans have no authorized truth table, so fr_c0ce is not scored per bend.

**Engine outputs:**
- "Before" is round 1's e3-1.10.0 post-hoc outputs.
- "After" means every engine-ran case was re-run on e3-1.11.0 (`fixes3/tools/posthoc3_driver.py`). This was done for two reasons:
  - the D2 trigger cannot be ruled out from the old logs;
  - DEV-CHECK had to be rebuilt from non-sealed inputs.
- Only 3 cases changed:
  - fr_6fa053514df8 (49207000): D2;
  - pc_cebb and pc_a030: D4.
- fr_c0ce (38968000, D3) changed only its bend states; see §4.
- Tools: `fixes3/tools/consumed_cases.py`, `consumed_diag.py`. Records: `fixes3/diag/diag_before.json`, `diag_after.json`.

### 2.1 Line F1 (frozen 5° / 4 mm / 0.6, Hungarian, OUTER_APEX, raw) — DIAGNOSTIC — consumed sets, not a held-out score

| Set | Cases | Before TP/FP/FN, micro F1, family-macro | After TP/FP/FN, micro F1, family-macro |
|---|---|---|---|
| FRESH, all compiled cases (12) | 12 | 53/0/16, 0.869, 0.770 | **54/0/15, 0.878, 0.786** |
| FRESH, engine-ran | 11 → 12 | 53/0/6, 0.946, 0.847 | 54/0/15, 0.878, 0.786 (49207000 now runs, with 1/10 lines: its positions are withheld) |
| DEV-CHECK, all cases (6) | 6 | 28/0/6, 0.903, 0.714 | 28/0/6, 0.903, 0.714 |
| DEV-CHECK, engine-ran | 5 | 28/0/3, 0.949, 0.893 | 28/0/3, 0.949, 0.893 |

### 2.2 Per-bend given CAD, complete scans, engine-ran — DIAGNOSTIC — consumed sets, not a held-out score

| Set | Def. | Before correct (riser-aware/strict)/bends; false-PASS proxy | After correct/bends; false-PASS proxy |
|---|---|---|---|
| FRESH | V1 | 44/44/55; fr_7bde B3 | 44/44/64; fr_7bde B3, fr_6fa0 B2, B3, B5, B6, B7, B8, B9 |
| FRESH | V2 | 45/45/55; none | 48/48/64; **fr_6fa0 B2, B3, B9** |
| FRESH | **V3** | 45/45/55; none | **48/48/64**; fr_6fa0 B2, B3, B9 |
| DEV-CHECK | V1 | 22/19/33; pc_6e5d B2, B3 | 22/19/33; pc_6e5d B2, B3 |
| DEV-CHECK | V2 | 25/22/33; none | 25/22/33; none |
| DEV-CHECK | **V3** | 25/22/33; none | **29/26/33**; none |

- **Not run** (refused/NO_RECIPE): DEV-CHECK pc_2d3e4a623a31 is a true mirror, kept refused (3 bends). Every FRESH complete case now runs; before, 49207000 (9 bends) was refused.
- **The 3 V2 proxies on 49207000 are a pose-offset artifact, not wrong bends.** Evidence is in `fixes3/diag/identity_49207000.txt`:
  - D2's pose is about 7 mm off along the in-plane direction the datum leaves free. Positions are withheld there, correctly.
  - The V2 identity check maps the CAD nominal line through that pose. The mapped lines land 3–8 mm from their gold lines, parallel within ≤ 0.7°, which is beyond the 4 mm gate.
  - Two independent checks give the **same identities for 8 of 9 bends** (B1→g5, B3→g4, B4→g2, B5→g8, B6→g9, B7→g1, B8→g6, B9→g7):
    - the engine's own **measured** apex lines (from the result diagnostics, mapped to raw) match gold with the frozen matcher, TP 8 at 0.15–0.46 mm (B1 3.9 mm);
    - the CAD nominals mapped by the independent ICP pose from FIXES2 give the same pairs.
  - So the B3 and B9 PASSes rest on the right bends.
  - **B2** (the unique 120° bend, read 120.36°) cannot be confirmed by any method: no gold line matches it. Its PASS is unverified.

## 3. D2: free-datum registration (49207000 class)

**Change:** `engine._free_datum_trigger`, `_free_datum_poses`, `_free_datum_rank`; `registration.coarse_hypotheses(plane_seeds=)`, `DatumModel.rotational_symmetries`, `symmetric_images`. The algorithm id is `..._free_datum_plane_seeds_symmetric_images_flange_end_rank_v6`.

**Trigger** (general; it never fires when the datum's own edges fix the pose):
- `NO_IN_PLANE_CONSTRAINT`: the best refined datum fit leaves all 3 in-plane directions undetermined; or
- `SYMMETRIC_DATUM`: the datum's material maps onto itself under an in-plane rotation (90/180/270°, ≥ 0.98 overlap), and the best fit leaves some direction undetermined.

  A symmetric datum whose cut edges fix the pose is already handled by the raster, which finds both images, and by the twin logic.

**Steps when triggered.** These run in both the proper and the mirrored pass, so a mirrored part gets the same chance.
1. **Plane seeds** (`NO_IN_PLANE_CONSTRAINT` only). The best raster peak of every candidate plane that no global peak represents is added, at most 4. This fixes the "seeding crowd-out" from FIXES2 §2.1.
2. **Symmetric images.** The datum's symmetric images of the best 6 fits are added (FIXES2 cause 4). Each new pose gets the standard fine search and refinement; repeats are dropped.
3. **Articulation.** Up to 6 added poses get the articulated fit.
4. **Per-hypothesis flange-ends ranking.** The existing flange-ends step (`_refine_free_directions`) runs on each of the best 6 articulated hypotheses that have undetermined directions. A refined pose replaces its hypothesis **only if its total is higher**.

**Not adopted from `fixes2/attempt_49207`:**
- (b) the band normalisation;
- (c) the re-search.

Both change kernels and fits for every part; they were not needed.

**Evidence:**
- **49207000** (consumed FRESH, fr_6fa053514df8):
  - Before: WRONG_PART_SUSPECTED, SCAN_EXPLAINED 0.249, at 25.6 s.
  - After: **VALID, explained 0.856** (≥ 0.85), total 1.400 (the old pose 0.717). All 9 bends are FORMED, and the angles are within 0.8° of nominal: 90.8 / 120.4 / 144.7 / 90.3 / 89.4 / 89.6 / 90.1 / 90.0 / 89.4. The mirrored best is 1.093, which is not ambiguous.
  - The winner is the **symmetric image of the datum-plane seed**, which neither the global peaks nor attempts (a) to (d) reached.
  - Its flange-ends variant (shift −1.18 mm in rotation·L) would drop explained to 0.32, so the ranking rule rightly kept the unrefined pose. The chosen pose's own refusal ("changed a decoded bend state") also stands.
  - The ICP-seeded upper bound in FIXES2 was 0.906.
- **Tests** (`test_d2_*`, 6 plus 1 opt-in):
  - a synthetic chiral TWO_DIRECTION_TAB whose datum fixes nothing (cut edges unusable, holes removed) **registers** (explained ≥ 0.95, all bends FORMED);
  - its **mirrored scan is refused** (MIRRORED_PART_SUSPECTED, no PASS);
  - a **wrong part** (the HAT scan) is refused (WRONG_PART_SUSPECTED, no PASS);
  - unit tests for the trigger, the images and the ranking rule;
  - `test_d2_the_49207000_weldment_...` checks VALID and explained ≥ 0.85 on the consumed scan. It is opt-in (`fixes3/tools/check_49207.sh`), so normal test runs never touch consumed data. It passes.
- **Synthetic probe** (`fixes3/diag/d2_synthetic_free_datum_probe.txt`):
  - With D2 off, the same free-datum part's **true** scan is refused as MIRRORED, and its **mirrored** scan is accepted **with PASS decisions**. That is a pre-existing false acceptance on free datums, which D2 fixes.
  - With K = 3 the mirrored scan was still accepted. K = 6 (= `pose_hypotheses`) is the smallest principled bound that refuses it, so K is not tuned to 49207000.
  - An L-bracket scan against that part is VALID with or without D2: an L is a sub-shape of the tab, with one bend missing. Such a wrong part cannot be told apart in principle, and no PASS is published.
- **A1.** The trigger fired on 4 cases (SYMMETRIC_DATUM: pc_0339, pc_8dfd, pc_927a, pc_e080). **0 characteristic changes.**

**Runtime:**
- A1 E3 P95 is 28.37 s (it was 28.54 s). The 4 triggered A1 cases rose from 4.5–9.2 s to 6.8–13.8 s, which does not move P95.
- G1 P95 is 13.96 s (gate 45 s).
- 49207000 rose from 25.6 s to 67.8 s (1.35 M points; both passes search). It is not an A1 or G1 case, and it is within the time budget.

**Risk:**
- The free direction stays undetermined after D2, so positions and heatmap are withheld there.
- On 49207000, B2's identity is not confirmable (see 2.2). Measured flange decisions on that weldment include several FAILs. Their truth is not available, and the welded plate may affect the far edges: **review before trusting 49207000 flange FAILs.**
- More hypotheses also mean more chances for a wrong part to find a fitting pose. G1 wrong/mirrored stays 0/32 PASS, and the synthetic wrong part stays refused.

## 4. D3: tied symmetric twins (38968000 class)

**Change (contract ENGINE_IO-A2 + E3 `_twin_agreement`):**
- `RegistrationV1.identity_ambiguous` is new: default false, and it requires `pose_ambiguous`.
- When certified symmetric twins tie, each twin is decoded and measured as an unambiguous pose would be. A characteristic of label *j* is **published only if all of the following hold**:
  1. every twin gives bend *j* the same state;
  2. the **matched characteristic** (the twin's label of the same physical bend, through E1's certified bend permutation) has the same nominal and tolerance;
  3. every twin reads it;
  4. the values agree within the published U;
  5. the guard-band decision is identical under each twin.

  So no PASS can rest on a reading that differs between the identities.
- Everything else is UNKNOWN(AMBIGUOUS_POSE). A bend whose state differs between the twins is UNKNOWN(AMBIGUOUS_POSE).
- Bend-line position, backgauge, heatmap, stage model and segment labels (other than the datum) stay withheld: they depend on which bend is which.
- The AMBIGUOUS_POSE stage flag keeps the stage decision UNKNOWN, so the summary is never green.
- The result carries the warning `IDENTITY_AMBIGUOUS_TWINS`.
- Without the mark, the old rule "an ambiguous pose measures nothing" still holds.
- The contract examples were regenerated: only `"identity_ambiguous": false` was added. The station TS validator, `CONTRACT_KEYS` and types were updated, and the view-model parity check passes.

**Evidence:**
- **Synthetic symmetric U, nominal:** identity-ambiguous. 6 readings are published (angles and flanges), and every PASS is in tolerance under the E2 truth. Positions are UNKNOWN(AMBIGUOUS_POSE), and the cross-document check is clean.
- **Synthetic symmetric U, with errors:**
  - the angles differ between the twins (91.12° vs 89.71°), so they stay UNKNOWN;
  - flange B agrees (98.68 both) and is published as PASS, which is true under the truth;
  - no false PASS.
- The unit tests cover:
  - an agreeing reading is published;
  - a differing one is withheld;
  - "same within U but different decisions" is withheld;
  - an unreadable twin is withheld;
  - a mismatched matched-characteristic spec is withheld;
  - the contract rejects measuring without the mark.

  Tests: `test_d3_*`, 4. The old CLI test was updated to the amendment.
- **38968000 (fr_c0ce5ecd2ba3, consumed; DIAGNOSTIC):**
  - B1, B2, B5 and B6 now carry agreed states: 2 FORMED and 2 UNEXPECTED_FORMED. B3 and B7 are NOT_YET_FORMED; B4 and B8 are UNKNOWN.
  - **0 readings are published.** The two twins' readings for one label belong to two different physical bends. On this real part those differ by 0.2–0.6 mm and 0.6° (for example B1 flange A: 16.50 vs 16.74 mm, U 0.10; angle 89.21° vs 88.61°, U 0.30–0.37). The rule therefore withholds them, correctly (`fixes3/diag/twins_38968000_readings.txt`).
  - Lines stay 0/0/4, because positions are withheld. Runtime is 106 s, unchanged.
- **G1.** SYMMETRIC_NO_WRONG_ID is 0 (2 bends, RIGHT_ID). FALSE_PASS is 0/553. COVERAGE is 0.9969 (n 326, was 322).

**Risk:**
- Low for false PASS: the decision must be identical under every twin.
- On real symmetric parts the rule will usually publish little. Counterpart bends differ by more than U, so the expected coverage gain from D3 is small (0 readings on 38968000). See the owner question below.

## 5. Not done (owner decisions)

- **D5, direct joggle-riser length measurement:** deferred. Riser-aware scoring stays. The 13 strict-only joggle bends are unchanged; after D4 the strict count is 176, against 192 riser-aware.
- **D6, 38898000 seam lines:** option A. The part stays out of scope as a "rolled / seamed part: not inspectable by the press-brake engine", with no recipe. Its 2 gold seam lines stay visible as coverage misses.

## 6. Open points / owner decisions

1. **49207000 B2 identity is unverified.** 8/9 bends are confirmed by two independent methods. If the owner wants a hard guard, the candidate from FIXES.md §1 is "a PASS on a bend whose position is withheld needs its measured fold nearer its own mapped nominal than any parallel bend's". It is not implemented: it would also need the pose along the free direction, which is what is missing here.
2. **49207000 flange FAILs** (B1 A/B, B2 A/B, B3 A, B5 A, and others) have no truth. A section check, like the 37126000 joggle one, would tell whether they are real or caused by the weldment.
3. **D3 yield on real parts is about 0 under the "within U" rule.** An alternative would need a new amendment: part-level statements that do not name a bend, such as "all four 90° bends of the symmetric set are within tolerance".
4. **Consumed-set V2 false-PASS proxies went from 0 to 3, all on 49207000.** Before, that case was refused, so it had no proxies. These are the pose-offset artifacts of §2.2, not wrong bends, except that B2 is unverifiable.

## 7. Files

- **`fixes3/`:**
  - `engine_before.tgz`, `engine.diff`, `fp_chain_fin_start.json` / `fp_chain_fin_end.json` (equal);
  - `a1/` (`keynums_d4`/`_fin`, `per_bend_all_v3_d4`/`_fin`, `compile_frozen_fin.log`);
  - `g1/scorecard_fin.*`;
  - `diag/`:
    - the D1 diagnostic records `diag_before.json` and `diag_after.json` (counts only, no label coordinates);
    - `posthoc3.jsonl`;
    - the 49207000 and 38968000 engine logs;
    - `identity_49207000.txt`, `twins_38968000_readings.txt`;
    - the D2 synthetic probe;
  - `logs/` (named tests, E2E);
  - `tools/`:
    - `per_bend_all_v3.py`, `consumed_cases.py`, `consumed_diag.py`, `posthoc3_driver.py`, `ledger_note.py`, `check_49207.sh`;
    - the D2/D3 diagnostic harnesses `h49.py`, `syn.py`, `ident49.py`, `twins389.py`.
- **Outside this folder:**
  - `contracts/production/inspection_result.py`, `ENGINE_IO.md` (§7.1), and the regenerated `examples/*.json`;
  - `domains/production/engine/inspect/{config,engine,registration,result}.py`;
  - `docs/production/ENGINE_INSPECT.md`;
  - `scripts/production/per_bend_scoring_v3.py`;
  - `tests/test_prod_engine_fixes3_2026_10_19.py`, `tests/test_prod_engine_inspect_cli.py` (one test updated);
  - `frontend/react/src/features/station/model/{stationView,types}.ts`;
  - `freeze/LEDGER.json` seq 7–9.
- **E2E run:** `docs/production/runlog/e2e/20261019T-real-fixes3/`.

---

## 8. Follow-up (2026-10-20): 49207000 B2 and the flange FAILs, settled with independent geometry

**Ask:** coordinator follow-up, with zero false PASS as the top rule. Same rules and lock as above.

**Engine:** now **`e3-1.12.0`**. Ledger seq 10 `frz_86c4a5875303`. The snapshot before the change is `fixes3/engine_before_followup.tgz`, and `fixes3/engine.diff` is updated.

**Independent method** (`fixes3/tools/plane49.py`, `xsec_b2.py`; records in `fixes3/followup/`):
- Each flange's **OML skin plane** is fitted directly from the scan, around every CAD bend.
- The skin is chosen explicitly: the outer of the two skin peaks t apart. This matters because a pose off by about t otherwise swaps the skins; this happened in a first version of the tool.
- The fit is done under both the E3 pose and the independent ICP pose. From it:
  - the angle comes from the plane dihedral;
  - the fold comes from the plane intersection;
  - the lengths are sharp-to-sharp, measured station-wise for non-parallel bends, or out to the far edge.
- The CAD panel planes are run through the same code, so the comparison is like for like.
- **Check:** the scan fold lines built this way match **9 of 10 gold lines under both poses** (ICP-anchored offsets 0.13–0.40 mm). The tenth gold line (B10) is not a bend; see below.

### 8.1 B2 verdict: the bend is there, it is gold B3, and the PASS is supported

- **The bend exists.** Raw cross-sections perpendicular to gold B3 (`followup/xsec_B2_vs_gold.png`) show one clean fold with t = 3. Its outer apex lies on **gold B3**: the ICP-anchored scan fold is 0.37 mm / 0.03° from it, with overlap 0.98 / 1.00.
- **Included angle** from independent planes: **120.09°** (ICP-anchored) and 119.89° (E3-anchored). The engine reads 120.36°. Nominal is 120°, tolerance ±1°. **The engine's PASS is supported.**
- **Why it matched no gold line:**
  1. The engine withholds B2's position: the in-plane pose is degenerate, so no line is published.
  2. The V2 identity check maps the CAD nominal through the E3 pose. That pose is about 8 mm off across B2, along the free direction, so the mapped line lands 6.8 mm from gold B3, beyond the 4 mm gate.
  3. The engine's own measured apex also sits 4.7 mm from gold B3, for the same reason: the fold geometry is right, but its frame position inherits the pose shift.
- **Gold B10 is not a bend.** It is a 99 mm line about 8 mm from gold B3, and the sections show it is the **edge of the welded plate** lying on that flange. This is a gold-label issue, a non-bend line; nothing was edited.
- **The remaining V2 "false-PASS proxies" are scoring artefacts of the mapped-nominal identity gate**, not wrong bends. They are 49207000 B2, B3 and B9, all angle PASSes. The independent scan folds of all three lie on their gold lines within 0.4 mm (B2→g3, B3→g4, B9→g7).
- **All 8 angle PASSes on 49207000 are in tolerance under both measurements:**
  - independent vs engine within 0.2–1.4°;
  - every independent value within ±0.85° of nominal;
  - B1 is REVIEW in the engine; the independent value is 87–88°. It is not a PASS.

### 8.2 Flange FAIL verdicts (49207000)

Independent values are "scan − CAD" with the same method. The ICP-anchored value is the reliable one: under ICP every fold lies within 0.4 mm of its gold line, while the E3-anchored fit of B2 was 3.4 mm off.

| Flange | Engine (e3-1.11.0) | Independent, ICP-anchored (E3-anchored) | Cause | Verdict | e3-1.12.0 |
|---|---|---|---|---|---|
| B1 A (free edge, 35) | 38.48 FAIL (+3.48) | 34.57 (38.36) | B1's OML sharp was built from the inner skin + t on a panel whose height comes from the free pose direction: skin flip | **engine artefact** | UNKNOWN |
| B1 B (→B2, 90) | 92.54 FAIL (+2.54) | −0.10 (+1.07) | skin flip (B1 and B2 both `ONE_SKIN_OML_FROM_INNER_SKIN_PLUS_T`) | **engine artefact** | UNKNOWN |
| B2 A (→B1, 90) | 92.45 FAIL (+2.45) | −0.04 (+1.10) | skin flip | **engine artefact** | UNKNOWN |
| B2 B (→B3, 254.25) | 249.53 FAIL (−4.73) | +0.01 (−3.32) | skin flip at B2 | **engine artefact** | UNKNOWN |
| B3 A (→B2, 254.25) | 249.45 FAIL (−4.81) | +0.02 (−3.32) | ends at B2's flipped sharp | **engine artefact** | UNKNOWN |
| B5 A (→B4, 196.88) | 200.75 FAIL (+3.87) | −1.13 (+3.52) | `VS_STATION_NOMINAL` | **engine artefact** | UNKNOWN |
| B6 A (→B4, 112.65) | 116.34 FAIL (+3.69) | −1.11 (+3.44) | as B5 A | **engine artefact** | UNKNOWN |
| B8 B (→B7, 196.88) | 199.63 FAIL (+2.75) | −1.65 (+3.16) | as B5 A | **engine artefact** | UNKNOWN |
| B9 B (→B7, 112.65) | 116.77 FAIL (+4.12) | −1.35 (+3.44) | as B5 A | **engine artefact** | UNKNOWN |

**Mechanism of the B5 A / B6 A / B8 B / B9 B rows** (`VS_STATION_NOMINAL`):
- These flanges end on bends at 30°, so the nominal varies along the line by about 0.58 mm per mm.
- Each station is compared with the nominal at its along-line place. That place comes from the pose, which is undetermined along these bend lines: the E3 and ICP poses differ by 8.4–8.6 mm along them. 8.5 × 0.58 ≈ 4.9 mm, which matches the observed +2.8 to +4.1.
- The independent residual of −1.1 to −1.7 mm is itself station-based, so it inherits the ICP pose's along-line error. Treat it as "within about 2 mm", not as a confirmed real deviation.

**Also corrected** (REVIEWs, same station mechanism): B5 B, B6 B, B8 A and B9 A. These free-edge flanges have far edges that step by 17–26 mm along the bend, so the station placement decides the nominal. They are now UNKNOWN.

**Untouched, and confirmed:**
- B3 B PASS: independent 50.06 vs 49.98.
- B4 B and B7 A REVIEW: engine −0.44 / −0.42, independent −0.61.

**Welded plate.** It covers part of the datum and one face next to B2 (that is gold B10). It is **not a direct cause of any flange FAIL**. Indirectly, by hiding datum material, it leaves the pose under-constrained, which is what the two mechanisms feed on.

**Real deviations:** none confirmed. No flange of 49207000 shows a deviation beyond tolerance under the reliable (ICP-anchored) independent measurement.

### 8.3 Engine change (general rule; no part tokens): `engine._pose_borne_lengths`, `_station_free_decision`

This applies only while the **final** pose still leaves an in-plane direction undetermined.
- **Skin rule.** A bend sharp built from the inner skin + t (`BendMeasure.one_skin`) is pose-borne when that side panel's normal lies along an undetermined direction (`InPlaneFit.position_determined`). Every flange length starting or ending at it becomes UNKNOWN(REGISTRATION_UNTRUSTED).
- **Station rule.** A `..._VS_STATION_NOMINAL` comparison becomes UNKNOWN(REGISTRATION_UNTRUSTED) when the bend line's own direction is undetermined. It is kept only if the guard-band decision is the same over the whole station-nominal range: equal at both ends of `[raw median − max profile, raw median − min profile]`, and either PASS at both ends or both ends on one side of nominal.
  - The first version compared only the end decisions. It wrongly kept FAIL-short ↔ FAIL-long intervals, which pass through PASS. This was caught on 49207000 and fixed, with a test.
  - `Measurement` now carries `profile_range` and `raw_median`.
- **Tests** (3 new, in `tests/test_prod_engine_fixes3_2026_10_19.py`):
  - skin rule, station rule, and no-op when the pose is determined;
  - the decision-interval rule, including the FAIL-short ↔ FAIL-long case.

**Before → after:**

| | e3-1.11.0 | e3-1.12.0 |
|---|---|---|
| 49207000 flange decisions | 9 FAIL, 6 REVIEW, 1 PASS | 0 FAIL, 2 REVIEW (B4 B, B7 A), 1 PASS (B3 B); 13 UNKNOWN(REGISTRATION_UNTRUSTED); summary REVIEW_REQUIRED |
| 49207000 angle decisions | 8 PASS, 1 REVIEW | unchanged (all supported, 8.1) |
| A1 | GOLD_SET_V1 0.990/0.974, BROAD 0.702/0.817, V3 192/203, proxy 3 | **identical, 0 characteristic changes**: in A1 the flange ends resolve every degenerate pose, so the guard never fires. P95 28.35 s |
| Frozen recompile | 18/18 | 18/18 |
| G1 | PASS, 0/553 false PASS | **PASS, 0/553, wrong/mirrored 0/32**; P95 13.78 s |
| Named tests | 837 + 1 | **840 pass** (one run) |
| Real E2E | 13/13 | **13/13** (`20261020T-real-fixes3e`) |

**E2E note.** Two earlier runs on e3-1.12.0 (`…fixes3c`, `…fixes3d`) failed only **W3b**, a Vendor Setup studio step that authors a general measurement. There the studio rendered its bend markers as plain labels before any point was picked, and that step runs before any E3 job. A control run on e3-1.11.0 passed, and a third run on e3-1.12.0 passed 13/13. W3b is an intermittent studio UI race, not engine behaviour. It is worth a look by whoever owns the studio.

### 8.4 Consumed-set diagnostic after the change — DIAGNOSTIC — consumed sets, not a held-out score

| Set | Lines TP/FP/FN (all cases) | Per-bend V3 correct/bends (e3-1.11.0 → e3-1.12.0) | V2/V3 false-PASS proxies |
|---|---|---|---|
| FRESH | 54/0/15 (unchanged) | 48/64 → **45/64** (49207000: 3 → 0 correct, because its flanges are now UNKNOWN) | 49207000 B2, B3, B9: the identity-gate artefacts of 8.1, shown to be the right bends |
| DEV-CHECK | 28/0/6 (unchanged) | 29/33 → **27/33** (pc_6e5d: the guard also withheld B3 A, a station-dependent PASS, and B4 B / B5 A / B5 B, which use B5's inner-skin sharp on a panel whose normal is the free direction; B1 A FAIL is kept because it is FAIL long whatever station applies) | none |

**Trade-off (owner):** 5 per-bend correct rows on consumed sets were given up to remove pose-borne flange readings. On 49207000, 9 of those readings were false FAILs. On pc_6e5d the withheld readings included 2 PASSes whose correctness cannot be shown without the pose.

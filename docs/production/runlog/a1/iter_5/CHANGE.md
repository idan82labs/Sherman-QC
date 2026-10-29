# A1 iteration 5: engine change record (2026-10-18)

All physical numbers below come from consumed development scans (engineering ledger). None of them is
unseen-family accuracy. Synthetic numbers are E2 DEV seeds, or E1 CAD stage models seen through the E2 scanner;
none of them is physical truth.

## Summary

- **Owner direction for iteration 5.** Raise BROAD line recall toward 0.80. Find out which BROAD gold lines the engine
  misses and why, and separate engine misses from BROAD-label problems without tuning to the labels.
- **Answer: 36 of the 39 BROAD misses are label problems.** They are lines that the gold of record on the same scan
  does not have at that place. Only 3 are engine misses:
  - 2 hooks bent the wrong way (23553000 SCAN_2);
  - 1 fold that the recipe excludes (28870000 FULL).
  - Without fitting label errors, recall on the frozen BROAD gold can reach at most (105 + 3) / 144 = 0.75.
  - Against the gold of record on the same 24 scans, recall is 0.981 and precision 0.987 (best F1 lane: 0.475 /
    0.679).
  - Details: "BROAD recall audit" below and `diag/broad_fn_fp_audit_fixAB.json`.
- **Three general engine rules.**
  - **Fixes A and B** (hinge search; first start of this iteration) repair the iteration-4 regression on
    50757000_02_X (pc_752e2bd54aad).
    - Only that scan changes, of all 27 A1 cases.
    - Per-bend correctness given CAD: 177 → 181 of 203 (0.872 → 0.892, riser-aware).
    - GOLD_SET_V1: micro 0.977 → 0.990, family-macro 0.966 → 0.974.
    - BROAD (frozen): family-macro 0.806 → 0.814, recall 0.708 → 0.729.
  - **Fix C** (signed sweep; second start): a bend formed the wrong way can no longer read its nominal angle. It
    closes a latent false-PASS path that the audit's wrong-way hooks led to.
    - It changes nothing on any A1 scan, the G1 tier or the CAD-stage probes (see Results).
    - On synthetic wrong-way probes within the hinge search's reach (θ ≥ -10°), the angle now always FAILs, and no
      length or position is reported against the wrong skin. Further wrong-way bends stay UNKNOWN as before; reading
      them needs a mirrored search (Next).
- **G1.** The regression tier passes with every accuracy gate identical to iteration 4.
- **CAD-stage probes.** This is the first full re-sweep since iteration 4 (113 probes, 1006 bend states). It finds
  0 wrong states, 0 invalid, 0 angle false PASS and 0 angle false FAIL, with and without fix C. The second start of
  iteration 4 had 4 wrong states.
- **Gates still failing**: the same as in iteration 4, none of them engine-limited on this data.
  - G2(a): the human Gold11 gate of record.
  - G2(f): 15651004's three "repeats" are three different stages.
  - G3: the disputed 10839000 scan_2 truth rows, the rolled 28870000 SCAN_1, and the known-good angle MAE.

## Provenance

| Item | Value |
|---|---|
| Snapshot before editing | `engine_before.tgz`, sha256 `bee17fa9d270eadf17e6318d0ede7b11981e377c6e32e8b23703e88e660f0056`. Written at 16:48 local by the first start of this iteration, before any edit. Its engine fingerprint is `dd4f039d…`, which is iteration 4's final engine (`iter_4/validation/code_fingerprints_final.json`). |
| Why the snapshot was not re-taken | This iteration was started twice. The first start (16:48-17:46) diagnosed the 50757000_02_X regression and made fixes A and B (`articulated.py`, `config.py`). It wrote 7 tests, ran the physical set (fix A alone, then A+B) and the G1 tier, and stopped at its session limit before writing this record. This second start (from 19:51) kept the 16:48 snapshot, because a new one would have captured the edited engine. |
| State found at the second start | The engine fingerprint (`32f0dfe4…`) equalled the one the first start's final physical and G1 runs had used, so those runs are this record's "fixes A+B" evidence. The 7 tests passed. |
| How fix C was made | Developed and unit-tested in a scratch copy of `domains/` while the A+B probe sweep was still reading the repository engine. It was copied in only after that sweep ended. |
| Diff against the snapshot | `engine.diff`: `inspect/articulated.py` and `config.py` (A, B), and `inspect/geom.py`, `decode.py`, `metrology.py`, `engine.py` and `config.py` (C). No change to `eval/`, `compile/`, `synthetic/` or `contracts/`. |
| Engine | `e3-1.7.0` (was `e3-1.5.0`). Algorithm ids: `articulated_fit` `…_joggle_step_one_skin_reread_astray_wide_v7`; `state_decode` `signed_dihedral_fraction_…_hole_evidence_v4`; `metrology` `…_reseat_signed_angle_v7`. |
| Code fingerprints | Engine `98feb9dc…` (A+B: `32f0dfe4…`; iteration 4: `dd4f039d…`). Compile `6147b43c…`, eval `8aa6860c…` and contracts `5f7e6a8c…` are unchanged (frozen). `validation/code_fingerprints_*.json`. |
| Ledger | `check OK`, anchor `cf01e709…` (recorded in every scorecard below). |
| Previous generality check | Clean, so nothing to remove. |
| Previous iteration | Iteration 4 regressed on one case (pc_752e2bd54aad, top failure 1 of its scorecard) but improved every gate-level number. This iteration fixed that case instead of reverting iteration 4. |
| Tests touched outside the engine | `tests/test_prod_engine_inspect_a1_iter5.py` (new: 7 tests for A and B, 4 for C). No existing test was changed. |

## The regression and fixes A and B

**What failed** (`diag/real_50757000_X_*`).
- In iteration 4, 50757000_02_X (pc_752e2bd54aad, t = 3 mm) lost 19 characteristic readings that iteration 3 had:
  - B4's angle read -16.4° (FAIL) and its line matched no gold line;
  - B5 and B6 were UNKNOWN (OCCLUDED);
  - B7 and B8 read +15.1° / +13.8° (FAIL).
- **Cause.** B1's line on this part sits 4.85 mm off nominal. The wide search finds it, and fix 5 of iteration 4 then
  turns P5 onto its fitted plane. B4's hinge frame is carried by that pose, so its window is placed a few millimetres
  off.
  - The B4 flange's near skin fell inside the window and its far skin outside. Hough profile at 90°: inner skin
    10954 points, outer 273.
  - The window took the near skin for the inner one. That one-skin reading lost to a pair of lines crossing the
    flange's two skins obliquely at 107°.
  - B4's child grew 492 voxels on that pair, and its grandchildren B5-B8 were searched from a wrong pose.

**Fix A. A one-skin line is also read the other way** (`articulated._one_skin_reread`, inside `_hough`).
- Applies to a formed-like line (|θ| ≥ `one_skin_reread_min_deg` = 20°) that the window reads on one skin only. "One
  skin" means the weaker skin holds at most `one_skin_fraction` of the stronger one.
- The visible skin is re-read as the *other* skin. Its partner is then one thickness over. It may lie up to one
  thickness past the window, but never past the grandchild limit (the side toward a parallel bend the child
  carries).
- The two-skin pair replaces the one-skin reading when the partner skin holds at least `one_skin_reread_min_share`
  (0.3) of the visible skin's support and the pair scores higher.
- The re-reading starts from the centroid of the visible skin's raw counts, so the smoothing offset is not carried
  twice.
- With the partner skin unseen, nothing changes.

**Fix B. A formed reading far from the nominal sweep gets the wide search too** (`articulated._wide_search_needed`).
- Before, iteration 2's wide (displaced-line) search ran only when the narrow line read flat or partial.
- Now it also runs when the narrow line reads formed but more than `displaced_check_dev_deg` (8°) off the nominal
  sweep (`narrow_astray`). Only bends of at least `displaced_min_sweep_deg` (45°) qualify, as before.
- The wide line still replaces the narrow one only with `displaced_support_ratio` (1.5×) its support.
  - A truly mis-formed flange keeps its reading: the wide search finds that same line, which cannot beat itself
    1.5 times.
  - Clutter (lines crossing a displaced sheet's two skins obliquely) loses to the sheet.

**Effect** (fixes A+B vs iteration 4's final engine; all 27 A1 cases; characteristic-level diff with a 0.05
tolerance).
- **Only pc_752e2bd54aad changes**, in 19 characteristics:
  - B4 angle -16.36° FAIL → +0.06° PASS;
  - B7/B8 angles +15.09° / +13.78° FAIL → +0.25° / +0.29° PASS;
  - B5/B6 UNKNOWN (OCCLUDED) → measured: angles -0.78° / -0.50°, flanges within 0.34 mm;
  - B7/B8 side A UNKNOWN → +0.35 / +0.25 mm PASS;
  - B7/B8 side B -1.78 / -1.41 mm FAIL → -0.28 / -0.00 mm PASS;
  - B4 side A -0.54 mm REVIEW → -4.97 mm FAIL;
  - B4, B5 and B6 positions → +4.82 / +4.86 / +4.21 mm FAIL.
- **The new FAILs fit a part made off-nominal, not an engine error.**
  - B1's line was already measured 4.85 mm off in iterations 3 and 4.
  - B4-B8 now read about 4.8 mm off, with their own flange lengths and angles in tolerance. P4 and its four
    flanges sit about 4.8 mm off nominal, like a backgauge error on one bend.
  - All 9 published lines on this scan match GOLD_SET_V1 and the BROAD gold.
  - The other scan of this SKU (50757000_02_V) reads every position within 0.4 mm with the same engine.

## Fix C. A bend formed the wrong way never reads its nominal angle

**Found during the audit.**
- The audit's two engine misses are hooks bent the wrong way. The G3 truth table says those must FAIL any angle check.
- Following that up showed the engine's bend angle is unsigned in three places:
  - the decode evidence (`engine._articulate`: `acos(n_parent · n_child)` of the OML normals);
  - the metrology's region-wide sweep;
  - its station angles (`180 - acos(...)`).
- So a flange on the wrong side of its parent's plane reads the magnitude of its sweep as if it were folded the CAD
  way.
- Synthetic wrong-way probes on the A1 recipes (`diag/synthetic_wrong_way_probes.txt`; CAD stage models through E2's
  scanner, no physical data) confirm it:
  - 38172000's 30° hook folded to -10°: before, PARTIALLY_FORMED with formed fraction **+0.43** (read as 43% formed
    the CAD way). Its flange A read -1.73 mm FAIL, flange B +0.85 mm REVIEW and position +0.85 mm REVIEW, all
    measured from a sharp on the wrong skin. The angle read +17.2° FAIL.
  - The angle FAILed only because 10° is far from 30°. A bend of nominal sweep ≤ ~11° folded fully the wrong way,
    within the hinge search's reach (θ ≥ -10°), would read its nominal angle and PASS.
  - No A1 recipe has such a shallow bend; the shallowest is 25°. Parts with 10-15° folds exist in GOLD_SET_V2.

**Rule** (`geom.signed_sweep_deg`, `engine._articulate`, `decode._raw_state`, `metrology`).
- **Signed sweep.** The angle between the parent's and the child's OML normals is signed about the hinge axis. The
  axis is oriented so that the CAD fold turns the child from the parent's continuation `u` toward `v = axis × u`.
  - The magnitude is exactly the old unsigned value. Only its sign is new: negative when the child turned the other
    way (it lies on the other side of its parent's plane).
  - So every CAD-direction reading is bit-identical to before.
  - A reading within `reversed_wrap_margin_deg` (30°) of 180° keeps its positive sign whatever its sense of turn: a
    hem's 180° is no wrong-way fold.
- **Decode.** A negative sweep within the flat limits (`flat_sweep_max_deg`, `flat_fraction_max`) is a flat sheet:
  NOT_YET_FORMED, as before. Beyond them, the bend was formed the wrong way: FORMED or PARTIALLY_FORMED by its
  magnitude, with a negative formed fraction.
- **Metrology.**
  - Such a bend's station angles are `180 - signed sweep`, above 180°, so its INCLUDED_ANGLE always FAILs the guard
    band.
  - Its OML sharp lies on the CAD's inner skin, which is not the characteristic's locus. So these are UNKNOWN
    (UNSUPPORTED_FEATURE): its flange lengths, position and backgauge, and bend-pair general measurements on it.
  - A neighbour's ADJACENT_BEND length ending at it is UNKNOWN (ADJACENT_BEND_NOT_MEASURED).
  - The bend row carries the note `BEND_FORMED_WRONG_WAY`. `state_evidence.measured_included_deg` stays empty,
    because the contract bounds it to (0, 180].
- **Evidence after the change** (same probe scans):
  - The -10° hook reads PARTIALLY_FORMED, formed fraction -0.43, angle FAIL +42.8° (192.8° vs 150°). Flanges and
    position are UNKNOWN, with the note.
  - 28870000's 25° bend at -10° reads angle FAIL +36.4° (was +13.6°) and position UNKNOWN (was a -1.69 mm FAIL).
  - Every other bend of these probes reads identical values.
- **What it does not do.** It does not find a wrong-way flange outside the hinge search.
  - A wrong-way bend beyond -10° stays UNKNOWN, NOT_YET_FORMED or WRONG_PART_SUSPECTED, as before (probes 3 and 4 in
    the diag file). That is never a PASS, but the defect is not named.
  - Reading the 23553000 hooks needs the mirrored hinge search (Next).

## BROAD recall audit (owner direction, 2026-10-18)

The audit was made on the fixes A+B engine and re-run on the final engine with identical results
(`diag/broad_fn_fp_audit_final.json`, `diag/broad_three_golds_final.json`): fix C changes no published line.

**Method.**
- Inputs: the frozen BROAD gold (`cases_A1.json`), the engine's apex lines (raw frame) and the frozen matcher
  (5° / 4 mm / 0.6 / Hungarian).
- Each unmatched BROAD gold line (FN) and each unmatched engine line (FP) was compared, with the same matcher, with
  the gold of record on the same scan:
  - Gold11 relabel v1 on the 9 Gold11 scans (owner decision 2026-10-17);
  - GOLD_SET_V1 where the scan is in it;
  - otherwise none.
- For every FN, the nearest engine line and the nearest gold-of-record line were recorded: angle, offset and both
  overlaps.
- Tool: `tools/fn_audit.py`. Output: `diag/broad_fn_fp_audit_fixAB.json`.

**The 39 BROAD false negatives**

| Class | Lines | Scans | Gold of record there | Engine or label |
|---|---|---|---|---|
| Human Gold11 line parallel to a fold but 4.2-14.7 mm off its apex, beyond the 4 mm gate (the 14 mm ones are the drafting script's "+14 mm copies") | 20 | 10839000 ×3, 37361005, 38172000 SCAN_1/SCAN_2 | the relabel has the fold at the apex, and the engine matches it | label |
| Human Gold11 line spanning several separate folds and gaps (overlap ≤ 0.3 with each) | 6 | 38172000 SCAN_1, SCAN_2 | the relabel has the separate folds, and the engine matches each | label |
| Human Gold11 line where no fold exists (a 15 mm line perpendicular to the folds, or 60-70 mm from any fold) | 6 | 10839000 scan_1, 38172000 V, X, SCAN_2 | the relabel has no line there | label |
| Pre-audit `bench_lines` splits each of two window-interrupted folds into two segments (147 + 123 mm) | 4 | 11979003_C (STL) | GOLD_SET_V1 (GOLD_FIXED) has one 285 mm line per fold, by the owner's convention; the engine publishes one 282 mm line each | label (convention; flagged GOLD_CONVENTION_CONFLICT at freeze) |
| Hook bent the wrong way (truth table: FORMED, REVERSE_DIRECTION_VS_CAD) | 2 | 23553000 SCAN_2 | not in GOLD_SET_V1 | **engine** |
| Fold the recipe excludes (recipe 1 bend, gold 2 lines) | 1 | 28870000 FULL | GOLD_SET_V1 has it too | **engine (recipe ceiling)** |

- The 32 human-Gold11 lines fall into the classes the Gold11 audit found independently (`GOLD11_AUDIT.md`). Of the
  71 human lines it judged, 20 are convention offsets, 23 real errors and 15 not bends. 45 are bit-identical to the
  output of an untracked drafting script.
- On these 9 scans, the engine matches every relabel line: 77/77, with FP 2.

**The 54 BROAD false positives.**
- 52 match a gold-of-record line: one that the old BROAD gold lacks, or places off the apex.
- The other 2 are 38172000 SCAN_1's 40 mm tab bends, B1 and B9 (`pc_07062eee17aa`).
  - At that stage B4 and B11 are not yet formed, so the tabs are coplanar and collinear with the 214 mm folds B2/B10.
  - The relabel draws each pair as one 305 mm line, and the engine publishes the two recipe bends separately. The
    305 mm line matches B2/B10.
  - On V, X and SCAN_2, the relabel draws the same tabs as separate 43 mm lines, which the engine matches.
  - The gold confirms both lines geometrically. Each lies wholly inside the relabel's 305 mm line (prediction
    overlap 1.0): B1 is 0.17° and 0.30 mm off it, B9 0.55° and 1.85 mm. Only the one-to-one matching leaves them
    unmatched.
  - So this is a stage-level convention question, not a measurement error. It is also the source of the
    "false-PASS proxy" pc_07062eee17aa:B1 (B1 flange A +0.21 mm PASS on a line the gold contains).

**BROAD scans scored three ways** (24 line cases, 14 families; same scans for both lanes;
`diag/broad_three_golds_fixAB.json`)

| Gold | Engine P / R / micro / family-macro | Best F1 lane P / R / micro / family-macro |
|---|---|---|
| Frozen BROAD gold | 0.660 / 0.729 / 0.693 / **0.814** | 0.580 / 0.451 / 0.508 / 0.543 |
| Relabel substituted on the 9 Gold11 scans (owner decision; BROAD elsewhere) | 0.975 / 0.957 / 0.966 / **0.945** | 0.661 / 0.457 / 0.540 / 0.560 |
| Gold of record per scan (relabel; GOLD_SET_V1; else BROAD) | 0.987 / 0.981 / 0.984 / **0.972** | 0.679 / 0.475 / 0.559 / 0.581 |

**Conclusion.**
- BROAD recall ≥ 0.80 cannot be reached on the frozen BROAD gold without fitting label errors.
- On the owner's gold of record for the same scans, recall is already 0.98.
- The two misses a better engine could recover are the wrong-way hooks (Next). The recipe-excluded fold needs a
  recipe bend.
- The label problems are reported here only. Nothing in the engine was tuned to them.

## Per-bend correctness given CAD: the 22 bends not counted correct

Definition as in iteration 4 (`tools/per_bend_all.py`): 203 bends, the G3 truth where it exists, else FORMED.

| Why not correct | Bends | Engine-limited? |
|---|---|---|
| A flange side ends at an adjacent bend that is NOT formed in this staged scan (truth table). The side is then UNKNOWN(ADJACENT_BEND_NOT_MEASURED), as the contract and the E4 oracle prescribe | 8 (10839000 scan_1 B2/B4, 23553000 SCAN_2 B6/B10, 15651004 SECOND/3RD B1, 38172000 SCAN_2 B4/B12) | no: not measurable at that stage |
| E1 gives no nominal for a flange side (NOMINAL_UNAVAILABLE, frozen compile) | 4 (50757000 X and V, B1/B4) | no |
| Disputed G3 truth rows on 10839000 scan_2 (iteration 4's hole evidence: 10/0 vs 1/7) | 5 (B2, B6, B8, B9, B11) | no (re-adjudication needed) |
| Stage-level line convention (above) | 2 (38172000 SCAN_1 B1/B9) | no |
| Rolled part read WRONG_PART_SUSPECTED (its roll differs from CAD S0) | 1 (28870000 SCAN_1) | partly (left open) |
| Hooks bent the wrong way, outside the hinge search: UNEXPECTED_FORMED with UNKNOWN characteristics (LOW_DENSITY) | 2 (23553000 SCAN_2 B2/B14) | **yes** |

- Once measured, the wrong-way hooks would still miss the per-bend criterion: each also has a side ending at an
  unformed adjacent bend.
- So 0.90 per-bend correctness cannot be reached on this population by measurement changes.
- A report-only "stage-aware" variant shows how much of the gap is definitional. It also accepts a side that is
  UNKNOWN(ADJACENT_BEND_NOT_MEASURED) when the truth table and the engine both say that adjacent bend is not formed
  yet. It scores 189/203 (0.931), family-macro 0.933. It does not replace the definition of record.
- The 3 false-PASS proxies are all explained:
  - 38172000 SCAN_1 B1: the stage-level line convention above;
  - the two disputed 10839000 scan_2 rows (B6, B11), whose truth the hole evidence contradicts.

## Results

**How the physical runs were made.**
- The frozen `prefreeze run-physical` / `score-physical` on all 27 A1 cases, run serially under the shared lock with
  `OMP_NUM_THREADS=1`. The ledger check was OK on every scorecard.
- "Iteration 4" is iteration 4's final engine (its `validation/`). "A+B" is this iteration's first-start engine
  (`validation/*_fixAB*`). "Final" is this record's engine (A+B+C, `validation/*_final*`).

**Fix C on the physical set: no change at all.**
- The final engine's results equal the A+B run's on all 26 runnable cases: every bend state, state evidence,
  displayed line, note, and every characteristic's value, U, decision and reason. Only the engine version string
  differs.
- `pc_b4719db3dd69` stays compile BLOCKED in both runs.
- So every physical number below holds for both A+B and the final engine.

**G2 (physical dev; frozen thresholds).** Identical to iteration 4: pc_752e2bd54aad is in no G2 population.

| Sub-gate | Iteration 4 | Final (A+B+C) |
|---|---|---|
| (a) line metric (gate of record: human Gold11) | FAIL, F1 0.391 | FAIL, F1 0.391 (P 0.342, R 0.458) |
| (b) angle / flange coverage | PASS 84/85, 160/166 | PASS 84/85, 160/166 |
| (c) known-good angle | PASS, MAE 0.795° (26 bends); common bends 0.910° vs legacy 1.291° | same |
| (d) known-good flange | PASS, P95 0.392 mm, max 0.455 mm | same |
| (e) runtime and failures | PASS, P95 26.8 s, max 30.6 s | PASS, P95 26.9 s, max 30.7 s, 0 crashes |
| (f) repeatability | FAIL, P95 0.426° / 0.254 mm | same |
| compile rate | PASS | PASS |

**G3 (frozen truth table, 13 cases)**: identical to iteration 4.
- State match 0.959 (118/123): FAIL.
- Known-good angle MAE 1.186°: FAIL. It covers 31 bends, 28 of them on 10839000:
  - 12 joggle step-equivalent angles (U 7-8°, MAE 1.53°);
  - 16 90° bends (MAE 1.01°), which read up to 2° overbent on all three scans (legacy agrees: common-bend MAE 1.22°).
  - Even without the joggles, the gate (1.0°) would sit on the edge.
- Unformed reported as defect: 2, FAIL (the disputed rows).
- These PASS: formed-bend angle coverage 77/81, flange coverage 130/160, and known-good flange P95 0.368 mm.

**Line localisation** (apex locus, frozen matcher). Engine vs the best F1 lane on the same scans.

| Gold | Scans | Iteration 4 | **Final** | Best F1 lane |
|---|---|---|---|---|
| GOLD_SET_V1 (A1 scans with STEP) | 23, 14 families | micro 0.977, family-macro 0.966 (TP 147, FP 3, FN 4) | **micro 0.990, family-macro 0.974** (TP 150, FP 2, FN 1) | micro 0.566, family-macro 0.583 |
| Gold11 relabel v1 (gold of record) | 9 | F1 0.987 (P 0.975, R 1.000) | **F1 0.987** (P 0.975, R 1.000) | F1 0.547 (P 0.686, R 0.455) |
| Human Gold11 (historical; G2(a) gate of record) | 9 | F1 0.391 | F1 0.391 (P 0.342, R 0.458) | F1 0.473 |
| BROAD gold (frozen, older labels) | 24, 14 families | micro 0.678, family-macro 0.806 (P 0.650, R 0.708) | **micro 0.693, family-macro 0.814** (P 0.660, R 0.729) | micro 0.508, family-macro 0.543 |
| BROAD scans, gold of record per scan (report-only, audit above) | 24, 14 families | – | micro 0.984, family-macro 0.972 (P 0.987, R 0.981) | micro 0.559, family-macro 0.581 |

- GOLD_SET_V1's remaining FN is the recipe-excluded 28870000 fold.
- Its 2 FPs are the SCAN_1 tab lines (above).

**Per-bend correctness given CAD** (all A1 cases the engine ran; report-only)

| | Iteration 4 | **Final** |
|---|---|---|
| bends scored | 203 | 203 |
| correct, strict | 166 (0.818) | **170 (0.837)** |
| correct, riser-aware | 177 (0.872) | **181 (0.892)** |
| family-macro, riser-aware | 0.891 | **0.907** |
| report-only "stage-aware" variant (above) | – | 189 (0.931), family-macro 0.933 |
| false-PASS proxy | 3 | 3 (the same bends, all explained above) |

**G1 regression tier** (DEV seeds, 50 cases, 7 families; `validation/g1_regression_scorecard_final.md`).
- regression_verdict **PASS**, with 0 failed accuracy gates. The overall verdict is INCOMPLETE and the run is
  non-qualifying by construction (TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL).
- Every accuracy value equals iteration 4's and the A+B run's:
  - flange bias 0.009375 mm, P95 0.1034 mm;
  - angle bias 0.0015°, P95 0.019°;
  - position P95 0.073 mm;
  - coverage 0.997, false PASS 0/553, false FAIL 0, U calibration 0.984, decisiveness 0.983, stage state 1.0;
  - slopes 1.008 / 1.000 / 0.993.
- Runtime P95 12.9 s, RSS 0.89 GB.

**CAD-stage probes** (E1 stage meshes of all 17 A1 recipes, every stage, springback ±2° on formed bends and ±1°
kinks on unformed ones, through E2's scanner; seed 23, the cached probe scans of iteration 4; `tools/sbsweep.sh`,
`validation/probe_sweep_*`).

| Engine | Probes | Bend states | Not VALID | Wrong states | Angle false PASS / false FAIL | Angles within U | Other FAILs (position, springback-driven) |
|---|---|---|---|---|---|---|---|
| Iteration 4, second start (fixes 1-8, before fix 12) | 113 | 1006 | 0 | 4 (the fix-4 regression on 38172000 S1/S3) | 0 / 0 | 497/499 | 96 |
| A+B (iteration 4 final + A + B) | 113 | 1006 | 0 | **0** | 0 / 0 | 501/503 | 96 |
| Final (A+B+C) | 113 | 1006 | 0 | **0** | 0 / 0 | 501/503 | 96 |

- This is the first full re-sweep with iteration 4's final engine; iteration 4 had left it open.
- Final vs A+B: no probe differs in its bend states, validity, angle decisions or other FAILs.

## Other checks made in this iteration

- **23553000 FULL hook flanges (iteration 4, risk 7).** An independent 2D section supports the current FAILs on the
  18.18 mm hook panels (for example B6 side B, +1.16 mm).
  - At y = -197 mm through the B7-B6-B5 chain, the hook panel measures 19.41 mm OML sharp to OML sharp. The engine
    reads 19.34 mm; nominal is 18.18 mm.
  - The wall's B6 sharp sits about 4-5 mm below nominal, consistent with the engine's B6 position of +4.18 mm FAIL.
  - So the earlier PASS readings, before iteration 4's fix 12, were probably the wrong ones.
  - Only one station had enough lip points, so this is supporting evidence, not proof (`tools/hook_section.py`).
- **15651004 repeatability.** B1's station angles are tight within each scan (0.1-0.2° range): PLY 90.2-90.5°, STL
  90.7-90.8°. So the 0.4° between scans is systematic, not station noise; it may be real change between stages.
- **50757000 NOMINAL_UNAVAILABLE** comes from E1's `nominals.py` (frozen compile), not the engine.

## Generality

- No SKU, family, part-id, file-name or metadata-count logic was added, and E3's inputs are unchanged.
- What each rule uses:
  - Fix A: the hinge band's two-skin histogram, the sheet thickness and the grandchild limit.
  - Fix B: the narrow line's angle against the nominal sweep.
  - Fix C: the fitted OML normals and the hinge axis's orientation.
- New thresholds are engine config fields with range checks:
  - `one_skin_reread_min_deg` (20°) and `one_skin_reread_min_share` (0.3);
  - `displaced_check_dev_deg` (8°);
  - `reversed_wrap_margin_deg` (30°).
- The frozen gate thresholds are untouched.
- Checked on all 27 physical A1 cases, the G1 regression tier, the 113 CAD-stage probes and the synthetic wrong-way
  probes.

## Risks

1. **Fix A can move a one-skin line by one thickness** where a second sheet lies one thickness past the window (a hem,
   a doubled flange).
   - It needs a formed-like angle, a partner skin with at least 30% of the visible skin's support, and a higher
     two-skin score.
   - It never reaches past the grandchild limit.
   - Not seen on the physical set, the G1 tier or the probes.
2. **Fix B lets the wide search replace a formed reading that strays more than 8° from nominal.**
   - A truly mis-bent flange (say 80° instead of 90°) keeps its reading unless a displaced formed line with 1.5× its
     support lies within the wide slack (10 mm).
   - If one does, the bend is measured on that line: a different sheet near the hinge, read as this bend displaced.
   - E2's angle-defect cases (G1: false PASS 0/553), the physical set and the probes show no such case.
3. **Fix C relies on the skin orientation of the fitted child plane.**
   - A child fitted with its skin-0 normal flipped would now read as formed the wrong way: a false FAIL. Before fix
     C such a plane read `180 - sweep`, which is harmless at 90° and wrong at any other angle.
   - Nothing on the physical set, the G1 tier or the 113 probes changed with fix C. So no fitted child there reads
     a negative sweep.
   - The wrap margin keeps 150-180° readings positive, so a hem's sign noise cannot flip it.
   - A bend formed the wrong way now loses its flange lengths and position (UNKNOWN). Its angle, the
     characteristic that shows the defect, FAILs.
4. **The audit's label classes rest on the gold of record.** Where the relabel or GOLD_SET_V1 is itself wrong, a
   "label problem" could hide an engine miss. Both were audited by blind vision verifiers. The few FPs and FNs that
   remain against them are listed above.

## Next (not done in this iteration)

- **Read hooks bent the wrong way** (`diag/real_23553000_scan2_B2_hook_band.png`).
  - The real lip curls about 115-120° toward -v, while CAD folds it toward +v. The hinge search covers θ from -10°
    to sweep + 25° and finds only clutter there (146°, 313 points).
  - Fix C supplies the signed angle and the FAIL semantics. Still missing:
    1. a mirrored hinge search: the axis reflected across the parent's mid-plane, v → -v, and the parent's other
       skin as the inner one;
    2. mirrored fold kinematics for the child's pose;
    3. the line of such a bend at its own outer apex (on the CAD's inner skin).
  - This would recover the 2 BROAD lines and name a common press-brake error instead of reporting UNKNOWN. It needs
    synthetic wrong-way probes at -90° to -120° (`tools/reversed_probe.py`) as its test bed.
- **28870000 SCAN_1 (rolled).** The datum fits (0.91-0.97), but the roll differs from CAD S0, so the scan is
  "explained" only 0.21-0.24. E1 excludes the roll as a ROLLED feature.
  - A wrong-part test on the recipe's supported panels only could read B1 NOT_YET_FORMED, as the truth table says:
    the second-best pose, which explains more of the scan, does.
  - It must keep G1's WRONG_OR_MIRRORED_NEVER_PASS.

## Tests

| Test | Result |
|---|---|
| `tests/test_prod_engine_inspect_a1_iter5.py` (new). 7 tests for A and B: one-skin re-reading, the grandchild limit, the astray trigger, the wide search beating oblique clutter, config. 4 for C: the signed sweep (CAD way, wrong way, 180° wrap, twisted plane), decode of negative sweeps, the signed fraction, config. Pure numpy; synthetic hinge bands and normals; no part named. | 11 passed |
| The final engine: `test_prod_engine_inspect_cli.py` (E3's 32 E2E tests, E2 → E1 → E3), `_units.py`, `_a1.py`, `_a1_iter2.py`, `_a1_iter3.py`, `_a1_iter4.py` and `_a1_iter5.py` | **142 passed** (208 s, under the lock) |
| Runs | The tests, the G1 tier, the physical set, both probe sweeps and every probe and debug run were done one at a time under `/private/tmp/claude-501/sherman_geom.lock`, with `OMP_NUM_THREADS=1`. |

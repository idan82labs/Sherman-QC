# A1 iteration 4: engine change record (2026-10-18)

All physical numbers below come from consumed development scans (engineering ledger). None of them is
unseen-family accuracy. Synthetic numbers are E2 DEV seeds, or E1 CAD stage models seen through the E2 scanner;
none of them is physical truth.

## Summary

- **Starting point.** The G3 truth table froze after iteration 3's measurement, so all 27 A1 cases ran through E3
  for the first time.
- **Kept.** The second start's eight registration and fit fixes (1-8).
- **Added.** Four general fixes (9-12):
  - a joggle's riser taken from its far flange's step;
  - near twins told apart by their holes;
  - bowed panels re-seated near their bends;
  - a window keeps its wider hole-core clearance, which corrects a probe regression of fix 4.
- **Final engine on all 27 cases.**
  - G2(b), (c), (d) and (e) pass: known-good angle MAE 3.49° → 0.795°, flange P95 16.0 → 0.392 mm.
  - G3 state match 0.927 → 0.959; formed-bend angle coverage and flange P95 now pass.
  - GOLD_SET_V1 line-F1 0.924 → 0.977 micro, 0.942 → 0.966 family-macro (best F1 lane 0.566 / 0.583).
  - Gold11 relabel F1 0.919 → 0.987 (best lane 0.547). BROAD family-macro 0.801 → 0.806 (best lane 0.543).
  - Per-bend correctness given CAD 0.798 → 0.872 (riser-aware; 0.788 → 0.818 strict).
- **G1.** The regression tier passes with every accuracy gate identical.
- **Still failing.**
  - G2(a), on the human Gold11 gate of record.
  - G2(f), on 15651004's three different-stage "repeats".
  - G3: 4 rows on 10839000 scan_2 that the scan's hole evidence says are mislabelled (below), 28870000 SCAN_1,
    and 10839000's real 1-2° overbends in the known-good MAE.

## Provenance

| Item | Value |
|---|---|
| Snapshot before editing | `engine_before.tgz`, sha256 `ca108c4394c8c38e9fad49d70545fddd00cd1b5eff3fc7b07c0ce99814ae8067`. It was written at 09:33 local by the first start of this iteration, before any edit, and equals iteration 3's final engine. |
| Why the snapshot was not re-taken | This iteration was started three times. The first start stopped at the 10:00 account pause before any edit. The second start (10:07-12:18) made fixes 1-8 below, ran tests, G1 and the physical set, and hit its session limit before writing this record. Re-taking the snapshot at the third start would have captured that half-finished engine, so the 09:33 snapshot stays the iteration's "before". The second start's edits, as found, are in `diag/second_start_inflight_edits.diff`. |
| Diff against the snapshot | `engine.diff`: five files under `domains/production/engine/inspect/` (`articulated`, `config`, `engine`, `metrology`, `registration`). No change to `eval/`, `compile/` or `synthetic/`. |
| Engine | `e3-1.5.0` (was `e3-1.4.0`). Algorithm ids: `registration` `…_hole_pairs_weak_dirs_walk_distinct_v4`; `articulated_fit` `…_pose_on_plane_joggle_step_v6`; `state_decode` `…_near_twins_hole_evidence_v3`; `metrology` `…_riser_model_reseat_v6`; `uncertainty` `…_station_spread_riser_length_v4`. |
| Code fingerprints (freeze-style) | Freeze-style: engine `dd4f039d…` (was `e642ea98…` with fixes 1-8 only), compile `6147b43c…` (= freeze), eval `8aa6860c…` (frozen), contracts `5f7e6a8c…`; unchanged from the start to the end of the final validation (`validation/code_fingerprints_final.json`). Default config `687eb815…`. |
| Ledger | `check OK`, anchor `cf01e709…` (every scorecard below records the ledger check). |
| Previous generality check | Clean, so nothing to remove. Iteration 3 did not regress (its fixes targeted then-refused cases), so nothing to revert. |
| Tests touched outside the engine | `tests/test_prod_engine_inspect_a1_iter4.py` (new in this iteration, 19 tests). No existing test was changed. |

## What changed since the last scorecard: the G3 truth table froze

The G3 truth table froze at 07:10Z (ledger `frz_7344c93bcfe8`), after iteration 3's measure step. The amendment-7
gate now accepts all 13 truth-relevant cases, so every A1 case runs through E3: 27 cases, 26 with a compiled recipe
(`pc_b4719db3dd69` stays compile BLOCKED). Thirteen of them had never been seen by the engine.

The first run of all 27 cases, with the second start's fixes 1-8, showed where the engine still fails:

| Gate | First full run (fixes 1-8) | What failed |
|---|---|---|
| G2(c) known-good angle MAE | 3.49° FAIL | 10839000 FULL: joggle bends B1/B2 read +19.6° / +33.2° (PARTIALLY_FORMED), B3/B4 -9.2° / -12.1° |
| G2(d) known-good flange P95 | 16.0 mm FAIL | 10839000 FULL: ±16-17 mm on B1, B2 and B8 (B2's line 16.8 mm off) |
| G3 state match | 0.927 (114/123) | 10839000 scan_1/scan_2: B1/B2 UNKNOWN; scan_2: four long-wall bends; 28870000 SCAN_1 INVALID |
| G3 formed-bend angle coverage | 64/81 FAIL | the joggles above, and 23553000 FULL (5 bends UNKNOWN) |
| GOLD_SET_V1 line recall | 133/151 | 9 joggle lines (10839000), 5 lines of 23553000 FULL |

## Diagnosis (renders in `diag/`)

I opened the sections, hinge bands and wall renders of the failing bends myself.

1. **Joggles on real parts have no flat riser** (`real_10839000_full_joggle_B1B2_sections.png`,
   `real_10839000_full_B2_band_candidate_lines.png`, `real_10839000_scan1_B2_band_riser_not_grown.png`).
   - CAD models each joggle as two 35° bends with a 0.1 mm inner radius around a 2.1 mm flat riser, step 1.80 mm.
     The real part has a smooth S: step 1.66-1.83 mm (matching CAD), run about 2.5 mm, no flat material anywhere.
   - **FULL B2.** The concave skin of the S is occluded, so the convex skin offers a tangent line at any angle from
     15° to 45°. A shallow pair across the far flange's first millimetres (13.2°, 11957 points) outscored the riser
     (35°, 8416 points; `real_10839000_full_B2_riser_profile.txt`). The riser was then grown onto the far flange
     (5.4° from the parent); B2 read 178° (PARTIALLY_FORMED) and its sharp, the intersection of two nearly parallel
     planes, landed 16.8 mm off. That one wrong riser explains every 10839000 FULL G2(c)/(d) failure: B1 and B2,
     and B8's side A, which ends at B2's sharp.
   - **scan_1 / scan_2 B2.** The line was right (31-35°), but the S's voxel normals leave every growth cone: the
     riser grew no region (AMBIGUOUS_SEGMENTATION), so B1 below it was BEND_STATE_UNKNOWN.
   - **B3/B4.** The riser grew 231 voxels on the S's steepest part (39° plane vs a 28° line); the metrology then
     found the stations 10° off the region planes (AMBIGUOUS_SEGMENTATION), or measured the S's steepest slope
     (FULL B3/B4: 135.8° / 132.9°, FAIL on a known-good part).
2. **The near-symmetric twin is settled by its walls' holes, not by its totals**
   (`real_10839000_scan2_wall_P1_holes_engine_pose.png`, `..._P2_...`, `real_10839000_scan2_twin_hole_evidence.txt`).
   - A 180° rotation maps 10839000 onto itself within 0.33 mm. The two images' totals differ by 0.006-0.007 on all
     three scans: noise.
   - The short walls carry four 2.4 mm holes, 13.9/28.9 mm from one end and 19.7/36.4 mm from the other, and one
     wall has a 30 x 18 mm cut-out. Under the twin rotation these land 6.8-21.7 mm from any hole.
   - Counting confirmed and contradicted holes of every fitted panel gives 10/0 vs 1/7 (scan_2), 12/3 vs 7/9
     (FULL), 10/0 vs 1/9 (scan_1). The engine's chosen image wins on all three, decisively.
   - See "A G3 truth-table dispute" below: on scan_2 this physical evidence contradicts the frozen truth table.
3. **A long datum bowed by 1-2 mm at its ends** (23553000 FULL).
   - The 460 x 96 mm datum (t = 0.92 mm measured) sits 0.6-2 mm outside its region-wide plane near the bends at its
     ends.
   - The station fits look for a skin within 0.37 mm of that plane, and the selection slab is ±1.06 mm, so B1, B4,
     B7, B12 and B16 had no parent material in their strips: OCCLUDED / LOW_DENSITY. Every characteristic of those
     five bends was UNKNOWN, and their lines were not published (5 GOLD_SET misses).
4. **A regression of fix 4 on CAD-stage probes** (the second start's final probe sweep, which finished after it
   stopped).
   - 38172000 S1 and S3 springback probes: B1 (and B2/B3) read NOT_YET_FORMED (4 wrong states). The base engine
     read them right.
   - A config bisection (each of fixes 1-5 switched off) pins it on fix 4's tighter hole-core clearance. In the big
     windows that hold tabs flat at S0, the tab's kerf, edge flare and kink put scan points 0.75-1.5 mm beside its
     compiled footprint. The tighter clearance made those points penalised hole core, so the true pose scored below
     a 180° image.
5. **Left open** (no engine change this iteration):
   - 28870000 SCAN_1 stays WRONG_PART_SUSPECTED: a rolled part whose curvature differs from CAD S0, explained 0.24
     (`real_28870000_scan1_registration.png`).
   - 15651004 repeatability: the three "repeats" are three different stages (S3, B1+B3, S1). B1 reads 90.36° /
     90.71° / 90.79°, so the 0.43° P95 may include real change between stages.
   - BROAD-gold convention lines.

## What changed, and why

Every fix is a geometric rule. None names a part, family or file, and E3's inputs are unchanged. Each new rule is
unit-tested in `tests/test_prod_engine_inspect_a1_iter4.py` on synthetic geometry that names no part.

### Fixes 1-8 (second start; reviewed, kept, validated here)

Their causes are in that start's probe diagnosis; their validation is the runs below.

| Fix | File | Rule |
|---|---|---|
| 1 | `registration.refine_datum` | A weakly determined in-plane direction (1 to 3 effective observations) is moved to its least-squares optimum instead of staying at the fine raster's window edge. It is still reported undetermined. |
| 2 | `registration.fine_search`, `engine._register` | A fine search whose best angle lies on its ±2° window's edge re-centres there (at most 2 walks, for the 3 best such searches). |
| 3 | `registration.hole_pair_hypotheses` | A datum with 2-8 round holes also gets poses from pairs of its holes found in the scan (an enclosed empty disc, matched by radius and pair distance). |
| 4 | `registration.DatumModel.kernel` | Hole cores count 0.75 mm (was 1.5 mm) from compiled stage material inside the hole, so a countersunk bore is visible. **Narrowed by fix 12** to round holes. |
| 5 | `articulated.pose_on_plane` | A fitted child's kinematic pose is turned about its hinge onto its fitted plane (within 10°), except a riser (its next parallel bend within 8 mm), so grandchild hinges stay on their sheets. |
| 6 | `metrology.station_spread` | The 10-90% range of the station angles a median stands for joins the angle's U, so a systematic trend along a bend cannot hide behind the median. |
| 7 | `engine._register`, `_same_cluster` | Refinement slots go to distinct poses (clustered within 3°/2 mm/same skin height first), so six images of one wrong pose cannot crowd out the true one. |
| 8 | `engine.inspect_scan` | Before a scan is called WRONG_PART, every refined datum pose (and the mirrored scan's) gets its articulated fit. |

### Fix 9. A joggle's riser comes from its far flange's step (`articulated._joggle_step`, `_model_riser`; `metrology`)

- **Applies to** a short child (riser) whose parallel bend turns the sheet back to the parent's orientation: the far
  flange's skin-0 normal equals the parent's at the final pose (a joggle). A hook or U-turn is excluded.
- **Step.** In the parent bend's hinge frame, the far flange is read directly: `_sheet_line` over 6 mm along `u`,
  starting 0.5 mm past the far bend's child-side tangent lines at the nominal pose. Its skin levels give its
  mid-line. It must lie within 6° of the parent's orientation with at least 60 points on its denser skin. The
  parent's sheet is read the same way just before the hinge, so a bow or twist of the parent's region-wide plane
  does not enter. The step is the offset between the two mid-lines at the riser's middle. A step below half the
  nominal one (or of the other sign) is not a formed joggle.
- **Angle.** The fold kinematics, with both bends turned alike so the far flange keeps its orientation, give the
  riser angle that puts the far flange at that step (`_joggle_offset`). The crossing nearest the nominal sweep is
  taken.
- **When the step decides.** The step's angle decides when the riser's own evidence strays from it by more than
  `joggle_theta_tol_deg` (8°): the Hough line, the grown region's plane, or the line vs that plane. A riser that
  grew no region also falls back to it.
  - The riser is then placed on its line at the step's angle (`_model_riser`, marked `model_plane`). The plane
    passes through the S's middle: each skin is half way up the step there, and below a step of 2t only that
    skin's S reaches that level. The far bend is found from that pose as usual.
  - On E2's flat risers the line, the grown plane and the step agree, so nothing changes there (G1 below).
- **Metrology.**
  - A side whose panel is a model riser keeps the model plane at every station: there is no flat material to fit.
  - The angle is the one the step implies under the riser's nominal length (method `…_JOGGLE_STEP_RISER_…`). Its U
    adds `tan(theta) * joggle_riser_length_u_fraction` (0.1) rad as a standard term, about 3.5°, so a joggle angle
    is REVIEW unless it is grossly wrong.
  - The same angle uncertainty moves both bends' sharps by `lever * angle / sin(sweep)`, and that enters the
    position U and the U of every ADJACENT_BEND length that ends at them.
  - The riser's own two flange lengths are UNKNOWN(UNSUPPORTED_FEATURE): with its angle taken from the step,
    the distance between its two sharps is the nominal riser length by construction, not a measurement.
- **Why the step and not the S's steepest slope.** The real S is steeper (45-60° at its inflection) than the
  CAD's 35° flat riser, and the steepest slope depends on where a plane is fitted. That slope read 132.9-135.8° on
  a known-good part (false FAILs). The step is the joggle's controlled dimension: here it agrees with CAD within
  0.15 mm on all three scans. The step-equivalent angle is a stated convention, not a plane fit, and its U says so.

### Fix 10. Near twins are told apart by their holes (`engine._hole_evidence`, `_ambiguity`)

- Rivals within `ambiguity_margin` that are certified twins or near twins (E1's rotation within 1.0 mm) are
  first compared by the holes of every fitted panel. Each hole loop is carried by the panel's pose:
  - **confirmed** when its core (the loop shrunk by `max(0.75 mm, 2 spacings)`) holds at most 10% of the points its
    3 mm annulus' density predicts;
  - **contradicted** at 40% or more.
  - Holes in unobserved areas, too small to judge, or filled by another fitted panel (a tab still flat in its
    window) give no evidence.
- A rival whose confirmed-minus-contradicted beats the best total's by `hole_evidence_min_diff` (2) or more is
  chosen, ahead of the order prior.
- Certified-symmetry ambiguity is decided afterwards exactly as before, so a truly symmetric part still reads
  AMBIGUOUS_POSE.
- On 10839000 this confirms the engine's existing choice on all three scans, by 11-18 in (confirmed -
  contradicted) instead of a 0.007 total. No decision changed on the physical set; the choice is now robust.

### Fix 11. A bowed panel is re-seated near its bend (`metrology._reseat_plane`)

- Before the stations, each fitted side's region-wide OML plane is re-seated on its own material in the station
  strip:
  - the skin levels within 3 mm outside and `t + 3 mm` inside (two skins: the outer is the OML; one: the reading
    nearer the plane's own levels);
  - when the OML level is more than 0.3 mm off, the plane is refitted three times on the band the last refit
    selects.
- It is refused unless it holds at least 25% of the strip's points and stays within 6° of the region-wide plane.
- A FREE_EDGE far edge that the local plane cannot follow over a long run is retried on the region-wide plane.
  Example: a 280 mm datum side, whose far end is bowed away from the plane re-seated at its bend.
- Flat E2 parts sit within 0.3 mm of their region-wide planes, so this never triggers there.

### Fix 12. A window keeps its wide core clearance (`registration.DatumModel.kernel`)

- Fix 4's 0.75 mm hole-core clearance now applies to round holes only (a stepped or countersunk bore).
- A non-round hole (a window or slot whose material is a tab cut on its own contour) keeps
  `window_core_clearance_mm` (1.5 mm, iteration 3's value).
- Re-checked: the two 38172000 springback probes read B1 (S1) and B1-B3 (S3) FORMED again. The 28870000 probes fix
  4 was made for still read right: S0 seed 7 B1 NOT_YET_FORMED, S1 seed 23 B1 FORMED, and the real FULL scan.

## Results

All physical runs: the frozen `prefreeze run-physical` / `score-physical` on all 27 A1 cases, serial under the
shared lock with `OMP_NUM_THREADS=1`; ledger check OK on every scorecard. "First full run" is the engine with fixes 1-8
(the second start's final state), run on all 27 cases after the truth table froze. "Final engine" is this
record's engine. Both scorecards are in `validation/`.

**G2 (physical dev; frozen thresholds)**

| Sub-gate | First full run (fixes 1-8) | **Final engine** |
|---|---|---|
| (a) line metric (gate of record: human Gold11) | FAIL, F1 0.415 | FAIL, F1 0.391 |
| (b) angle coverage | PASS 79/85 | PASS 84/85 |
| (b) flange coverage | PASS 150/166 | PASS 160/166 |
| (c) known-good angle | FAIL, MAE 3.490° on 26 bends | PASS, MAE 0.795° on 26 bends |
| (d) known-good flange | FAIL, P95 16.009 mm, max 17.290 mm | PASS, P95 0.392 mm, max 0.455 mm |
| (e) runtime and failures | PASS, P95 26.1 s, max 30.0 s, 0 crashes | PASS, P95 26.8 s, max 30.6 s, 0 crashes |
| (f) repeatability | FAIL, P95 0.426° / 0.254 mm | FAIL, P95 0.426° / 0.254 mm |
| compile rate | PASS | PASS |

**G3 (frozen truth table, 13 cases)**

| Gate | First full run (fixes 1-8) | **Final engine** |
|---|---|---|
| blank_stage0 | PASS | PASS |
| formed_bends_angle_coverage | FAIL 64/81 | PASS 77/81 |
| formed_bends_flange_coverage | PASS 115/160 | PASS 130/160 |
| formed_bends_known_good_angle_mae | FAIL 4.022° | FAIL 1.186° |
| formed_bends_known_good_flange_p95 | FAIL 16.054 mm | PASS 0.368 mm |
| formed_count_equals_expected_completed | PASS | PASS |
| state_match_rate | FAIL 0.927 (114/123) | FAIL 0.959 (118/123) |
| truth_complete | PASS | PASS |
| unformed_reported_as_defect | FAIL (2) | FAIL (2) |

**Line localisation (apex locus, frozen 5°/4 mm/0.6 matcher), engine vs the best F1 lane on the same scans**

| Gold | Scans | First full run (fixes 1-8) | **Final engine** | Best F1 lane |
|---|---|---|---|---|
| GOLD_SET_V1 (A1 scans with STEP) | 23 scans, 14 families | micro 0.924, family-macro 0.942 (TP 133, FP 4, FN 18) | **micro 0.977, family-macro 0.966** (TP 147, FP 3, FN 4) | micro 0.566, family-macro 0.583 |
| Gold11 relabel v1 (gold of record) | 9 | F1 0.919 (P 0.958, R 0.883) | **F1 0.987 (P 0.975, R 1.000)** | F1 0.547 (P 0.686, R 0.455) |
| Human Gold11 (historical diagnostic; G2(a) gate of record in the frozen scorer) | 9 | F1 0.415 | F1 0.391 (P 0.342, R 0.458) | F1 0.473 |
| BROAD gold (older labels) | 24 scans, 14 families | micro 0.674, family-macro 0.801 | **micro 0.678, family-macro 0.806** (P 0.650, R 0.708) | micro 0.508, family-macro 0.543 |

**Per-bend correctness given CAD (all A1 cases the engine ran; report-only)**

| | First full run (fixes 1-8) | **Final engine** |
|---|---|---|
| bends scored (G3 truth where it exists, else complete-scan FORMED) | 203 | 203 |
| correct, strict (angle + both sides MEASURED, line confirmed) | 160 (0.788) | **166 (0.818)** |
| correct, riser-aware (a joggle riser's own length UNSUPPORTED_FEATURE allowed) | 162 (0.798) | **177 (0.872)** |
| family-macro, riser-aware | 0.866 | **0.891** |
| false-PASS proxy | 3 ['pc_07062eee17aa:B1', 'pc_cec08dfca9ab:B6', 'pc_cec08dfca9ab:B11'] | 3 ['pc_07062eee17aa:B1', 'pc_cec08dfca9ab:B6', 'pc_cec08dfca9ab:B11'] |


The best F1 lane is the old research lane's frozen predictions on the same scans (`runs/best_f1`). The shipped
runtime engine is not used as a comparator (owner direction). On the known-good legacy control arm (informational):
G2(c) common bends E3 MAE 0.910° vs legacy 1.291° (16 bends), and coverage 26/26 vs legacy 16/26.

**What moved, case by case** (first full run → final engine):

- **10839000 (3 scans)**
  - All 12 joggle bend states (4 per scan) now read FORMED, with lines published. Before: UNKNOWN on B1/B2 of scan_1 and
    scan_2, and PARTIALLY_FORMED B1/B2 on FULL.
  - Joggle angles read 142.4-148.7° against 145° nominal (step-equivalent, all REVIEW).
  - FULL: B1 side A and B2 side B went from +16.07 / +15.96 mm FAILs to UNKNOWN (riser lengths,
    UNSUPPORTED_FEATURE). B1 side B and B2 side A now read -0.06 / -0.04 mm, and B8 side A went from -17.28 mm to
    -0.12 mm. B3/B4 angles went from -9.2° / -12.1° FAILs to -0.9° / +1.1° REVIEWs.
  - GOLD_SET_V1 lines on these scans: 21 → 30 of 30.
- **23553000 FULL**
  - The five bends whose datum strips sat 0.6-2 mm off the bowed datum's plane are measured (18 more
    characteristics), and their 5 lines are published.
  - Fix 12 also moved this scan's datum pose to a better-scoring one (total 1.447 vs 1.428, 0.65 mm away). That
    moved the short hook flanges of B5-B7 by about 1.2 mm (risk 7).
- **38172000 X and 37361005**: values moved by 0.01-0.03 mm through fix 12's kernel. One 38172000 X flange
  decision went PASS → REVIEW (-0.16 mm, U 0.35), and one 37361005 angle U went 0.72 → 0.54°.
- **Every other case** (38172000 V/SCAN_1/SCAN_2, 28870000, 15651004, 11979003, 34499000, 44211000, 47266000,
  48811000, 49024000, 49204020, 50757000): identical characteristic values.

**G2 summary.** (b), (c), (d) and (e) and the compile rate pass. (a) fails only on the human Gold11 gate of record;
on the relabel (gold of record per the owner) F1 is 0.987, P 0.975, R 1.000. (f) fails at 0.426° on 15651004,
whose three "repeats" are three different stages.

**G3 summary.** Every remaining state mismatch is either the disputed scan_2 identity (4 rows, below) or 28870000
SCAN_1 (INVALID: a rolled part whose curvature differs from CAD S0; 1 row). The known-good formed-bend angle MAE
(1.186°) is dominated by 10839000's 90° bends: they read 1.0-2.0° overbent on all three scans, as the legacy arm
also sees (common-bend legacy MAE 1.22°). That looks real, not engine bias: no other family shows it (11979003
0.37°, 15651004 0.64°, 37361005 0.37°).

**CAD-stage probes** (E1 stage meshes through E2's scanner; nominal geometry, so every FAIL is false).
- The second start's sweep over every compiled stage of all 17 A1 recipes at a new pose (springback seed 23, 84
  probes, 802 bend states) went 8 → 4 wrong bend states and 3 → 0 INVALID. Its remaining 4 wrong states were the
  38172000 S1/S3 regression of fix 4.
- Re-checked with the final engine (`diag/probe_38172000_S1_clearance_bisection.txt`, `tools/probe_check.sh`):
  - 38172000 S1: B1 FORMED again; S3: B1-B3 FORMED again.
  - 28870000 S1 (seed 23) and S0 (seed 7), fix 4's own target, still read right, as does the real 28870000 FULL.
- A full re-sweep of all 84 probes with the final engine was not run in this iteration (time).

**G1 regression tier** (DEV seeds, 50 cases, 7 families; `validation/g1_regression_scorecard.md`).
- regression_verdict **PASS**, 0 failed accuracy gates. The overall verdict is INCOMPLETE and the run is
  non-qualifying by construction (TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL; G1.SUITE needs 200 cases).
- Every accuracy gate value is identical to the second start's G1 run:
  - flange bias 0.009375 mm, P95 0.1034 mm;
  - angle bias 0.0015°, P95 0.019°;
  - position P95 0.073 mm;
  - coverage 0.997, false PASS 0/553, false FAIL 0, U calibration 0.984, decisiveness 0.983, stage state 1.0;
  - slopes 1.008 / 1.000 / 0.993.
- Fixes 9-12 change nothing on E2's flat, unbowed parts, as designed. Runtime P95 13.1 s (was 12.9 s).

## A G3 truth-table dispute (10839000 scan_2, `pc_cec08dfca9ab`)

The frozen table says that on scan_2 the long wall **B9** and its lip **B8** are formed, and **B11** and **B6** are
not. The engine reads the opposite: B11 and B6 formed, B9 and B8 not yet formed. Every other bend agrees. The
disagreement is exactly the part's 180° near-symmetry (E1 rotation 0.33 mm RMS), which swaps B9↔B11 and B8↔B6.
The physical evidence sides with the engine's identity:

- **Hole evidence** (fix 10): under the engine's pose, 10 of the fitted panels' holes are confirmed open and none
  is contradicted. Under the 180° image that would give the table's states, 1 is confirmed and 7 are contradicted
  (`diag/real_10839000_scan2_twin_hole_evidence.txt`).
- **Renders.** Under the engine's pose, both short walls' 2.4 mm holes and the 30 x 18 mm cut-out coincide with
  the scan's openings (`diag/real_10839000_scan2_wall_P1_holes_engine_pose.png`, `..._P2_...`).
  - Under the twin image, the same scan holes sit 7-9 mm from every CAD hole and the cut-out lands on solid sheet
    (`..._P1_holes_twin_image.png`, `..._P2_...`).
  - The short walls' holes are 13.9/28.9 mm from one end and 19.7/36.4 mm from the other, so they fix which end is
    which.
- **Which wall is formed** (`diag/real_10839000_scan2_section_formed_wall.png`): in the engine's frame the formed
  long wall stands at the far end from that 13.9/28.9 mm hole pair. By the CAD that is P3 (B11), with its lip P11
  (B6).

So the table's identity for these four rows looks swapped: as if its reviewers' registration were the 180° image,
whose totals differ from the engine's by only 0.0065. I did not change the engine to match it.

**Recommendation.** Re-adjudicate `pc_cec08dfca9ab` B6/B8/B9/B11 with the hole renders above, under the G3
freeze procedure; that is not A1's to do. Until then these rows give G3 four state mismatches and two "unformed
reported as defect" (B6 and B11, whose angles FAIL at -2.5° / -1.8° on bends the table calls unformed).


## Generality

- No SKU, family, part-id, file-name or metadata-count logic was added, and E3's inputs are unchanged.
- Every rule is geometric. Fix by fix:

  | Fix | What the rule uses |
  |---|---|
  | 9 | The flange tree's parallel bends and their final-pose normals (a joggle), the fold kinematics, and the scan's sheet levels in the hinge frame |
  | 10 | The compiled panels' hole loops, carried by each hypothesis' fitted poses, and the scan's point density around them |
  | 11 | The scan's skin levels in each side's own station strip |
  | 12 | Whether a datum hole is round |

- New thresholds are engine config fields with ranges: `joggle_*` (6), `hole_evidence_*` (6), `local_reseat_*` (4)
  and `window_core_clearance_mm`. The frozen gate thresholds are untouched; `eval/` is untouched (fingerprint
  `8aa6860c…` as frozen).
- Fixes 9-12 were found on the real A1 scans and on the E1/E2 probes. Each was checked on every A1 case: 27
  physical runs, with no change outside 10839000 and 23553000 beyond 0.01-0.03 mm on 38172000 X and 37361005.
  They were also checked on the G1 regression tier.

## Risks

1. **Fix 9 replaces a joggle riser's measured angle by a convention.** The step-equivalent angle is exact only for
   a riser of nominal length.
   - A real riser formed shorter and steeper but with the right step reads in tolerance (REVIEW, since U is about
     7°).
   - A wrong step reads as a wrong angle: 22° per mm here.
   - A joggle whose far flange is occluded, or tilted more than 6°, keeps the old line/plane path.
   - The riser's own length is now UNKNOWN(UNSUPPORTED_FEATURE): 2 flange sides per formed joggle leave the
     coverage counts.
2. **Fix 9's trigger is a 8° disagreement.** A partially formed joggle (riser half way, far flange tilted) fails the
   parallel test and keeps the old path. A riser whose plane is right but whose Hough line strays by 8° is
   replaced by the step model, which is still a measurement of the same part.
3. **Fix 10 trusts hole loops of fitted panels.** A wrongly fitted panel could carry its holes onto sheet and
   count against a correct pose. It only reorders twins within `ambiguity_margin`, needs a 2-hole difference, and
   never removes a certified-symmetry ambiguity. It adds about 0.2-0.5 s per twin pair compared.
4. **Fix 11 can move a bend's sharp by up to 3 mm** on a bowed panel. On 23553000 FULL this turned five UNKNOWN
   bends into measured ones, among them position FAILs of -1.50 mm (B1) and -2.99 mm (B7) and flange FAILs of
   -0.70 to -1.24 mm (B3, B4, B16) on the bowed ends.
   - The bow is real (1-2 mm, seen in the scan), so these positions are honest measurements in the datum frame.
   - But the part is not known-good: whether they are defects or a fixture-induced bow is not known.
5. **Fix 12 restores iteration 3's window clearance.** A countersunk non-round hole (unusual) would again show no
   core.
6. **The G3 truth-table dispute** (below) means the engine is scored against identities that the hole evidence
   contradicts. Until the table is re-adjudicated, 4 state mismatches and 2 "unformed reported as defect" on scan_2
   count against G3.
7. **23553000 FULL's short hook flanges are pose-sensitive** (open, the top item for the next iteration).
   - Fix 12 changed that scan's datum pose to a better-scoring one: total 1.447 vs 1.428, 0.65 mm and 0.07° away,
     both poses VALID.
   - The hook chain's 6-13 mm panels then moved by about 1.2 mm: B5 side B, B6 sides A/B and B7 side A went from
     PASS (-0.11…+0.03 mm) to FAIL (±1.15-1.19 mm, U 0.10-0.15).
   - A flange length should not depend on the datum pose that much. At least one of the two readings lies outside
     its U, so one of them is a false FAIL or a false PASS.
   - 23553000 is not a known-good part, so no gate sees it; G2/G3 and the line metrics are unchanged.
   - The likely cause: these short panels are found inside a kinematic window placed from the datum pose. A
     0.65 mm move of that pose can switch which skin or feature the window locks onto.
   - Next: fit such panels on their own material and widen their U by the window sensitivity.
8. **No full CAD-stage probe re-sweep with the final engine.** Fixes 9-12 were checked on the regressed and the
   targeted probes only (above).

## Tests

| Test | Result |
|---|---|
| `tests/test_prod_engine_inspect_a1_iter4.py` (new: 19 tests; pure numpy, synthetic strips, joggle bands, holed sheets and poses, no part named) | 19 passed |
| The final engine: `test_prod_engine_inspect_cli.py` (E3's 32 E2E tests, E2 → E1 → E3), `_units.py`, `_a1.py`, `_a1_iter2.py`, `_a1_iter3.py`, `_a1_iter4.py` | **131 passed** (209 s, under the lock) |
| Runs | Tests, G1 tier, the physical set, the probes and every debug run were done one at a time under `/private/tmp/claude-501/sherman_geom.lock`, with `OMP_NUM_THREADS=1`. E3 peak RSS on the physical set 0.95 GB at most. |

What the new tests cover (fixes 9-12; the second start's 9 tests cover fixes 1-8):
- **Fix 9.**
  - A curved-S joggle (no flat riser) gives its step and the riser angle that step implies; one visible skin
    gives the same step.
  - The riser's mid-plane passes through the S's middle.
  - No step model without a parallel far flange at a formed step (flat, too small, tilted 15°).
  - A joggle is a parallel far flange, a U-turn is not.
  - A model riser's skin planes contain its two lines.
- **Fix 10.**
  - An open hole is confirmed and a filled one contradicted.
  - A hole filled by another fitted panel, or in an unobserved area, gives no evidence.
  - A twin whose holes land on sheet loses to one whose holes are open, ahead of the order prior.
  - A 1-hole difference does not decide, and a rival outside the ambiguity margin is never chosen.
- **Fix 11.** A strip 1.3 mm off its region-wide plane is re-seated; one within 0.15 mm, one tilted 12° and one 5 mm
  away are not.
- **Fix 12.** A round bore's core reaches 0.75 mm from its step, while a window's core stays 1.5 mm from its tab.
- **Config.** All new fields, with their range checks.

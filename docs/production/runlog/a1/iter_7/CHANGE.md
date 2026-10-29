# A1 iteration 7: engine change record (2026-10-19)

All physical numbers below come from consumed development scans (engineering ledger). None of them is
unseen-family accuracy. Synthetic numbers are E2 DEV seeds, or E1 CAD stage models seen through E2's scanner;
none of them is physical truth. Thresholds are the frozen ledger's; none changed.

## Summary

- **Owner direction (2026-10-19)** after iteration 6: wrong-way bends (done in iteration 6), recipe-excluded folds,
  the 38172000 SCAN_1 tab-merge FPs, per-bend correctness given CAD toward 0.90+ with zero false PASS; and from the
  real-part E2E audit: joggle bends that can only ever be REVIEW (HIGH), and "every 90 deg angle on the 10839000
  FULL scan reads 88.0-89.7 deg: real springback or a systematic angle bias?" (MEDIUM).
- **Answer to the MEDIUM question: real geometry, not an engine bias** (section "Diagnosis 1"). On the 10839000
  FULL scan the two 12 mm lips are flat and stand at 88.2-88.7 deg; the four base-edge bends read 88.0-88.8 deg
  locally because the 313 x 206 mm base is tilted by 0.7-1.8 deg near the middle of its long edges, while the
  walls are flat. Against the base's mean plane the walls stand at 89.1-89.6 deg. The engine measures the local
  bend angle, as the plan specifies; it is not changed to hide the bow.
- **An engine bias was found elsewhere and fixed (the one change, engine `e3-1.9.0`).** The station windows
  started 0.5 mm past the tangent line of the CAD radius. Physical bends often keep bending past that line: a
  real inner radius larger than the CAD's (15651004: about 2.6 mm real vs 0.8 mm CAD; 50757000: 3 mm sheet drawn
  with a 1 mm radius) or a long springback transition. That tail, bending toward the bend's centre, tilted the
  station planes toward an underbent reading and scattered the stations. Each side's window now starts on its own
  flat material (`metrology._flat_start`, rules below).
- **Effect on the physical set** (all 26 runnable A1 cases; 159 measured angles, the two wrong-way hooks excluded):
  angle MAE vs nominal 0.946 -> **0.898 deg**, mean angle U 1.018 -> **0.956 deg**; 15651004 0.698 -> **0.364**,
  23553000 1.002 -> **0.788** (its 60 deg bends: mean +1.58 -> +1.08 deg); 15651004 B1 now reads -0.26 / +0.06 /
  -0.03 deg on its three scans (was +0.36 / +0.71 / +0.79).
  - G2(f) repeatability P95 angle 0.426 -> **0.306 deg** (gate 0.3; still FAIL, and its three "repeats" are three
    stages). G3 known-good angle MAE 1.186 -> **1.147 deg** (still FAIL: 10839000, Diagnosis 1). G2(c) 0.795 ->
    0.798 deg (PASS).
  - Line localisation, GOLD_SET_V1, the Gold11 relabel and per-bend correctness given CAD (183/203 = 0.901,
    false-PASS proxies 3) are unchanged.
- **Not changed, with reasons** (section "What was not changed"): the joggle angle (its riser has no flat
  material, see the section image), the part summary's "worst" ranking (contract code, not A1's), a far-end
  version of the same search (evaluated, not kept), the tab-merge FPs (owner convention), the recipe-excluded
  fold, the rolled 28870000 SCAN_1, the scan-to-scan thickness spread.

## Provenance

| Item | Value |
|---|---|
| Snapshot before editing | `engine_before.tgz`, sha256 `160a0e795c28ffee8e83f74c597344f75f943e8263a2a3d10386f146c16ec0da`, already present at this start (written 2026-10-19 03:29 local, 11 minutes earlier). The engine tree was byte-identical to it (`diff -rq`) and no engine file was newer, so it was kept rather than re-taken. Its inspect fingerprint is `349b80a9…` (the `fp.py` tool), iteration 6's final engine `e3-1.8.0`. |
| Previous iteration | Iteration 6 improved and did not regress; its generality check was clean. Nothing was reverted or removed. |
| Diff against the snapshot | `engine.diff`: `inspect/metrology.py`, `inspect/config.py` only. No change to `eval/`, `compile/`, `synthetic/` or `contracts/`. |
| Engine | `e3-1.9.0` (was `e3-1.8.0`). Algorithm id `metrology` `…_reversed_position_mirrored_flat_start_v9` (was `…_v8`). Config sha256 `cb14d5ca…` (was `bf0c93bf…`). |
| Code fingerprints (`fp.py`, `validation/code_fingerprints_final.json`) | Engine (`inspect/`) `349b80a9…` -> `6297d8e3…`. Compile `6147b43c…`, eval `8aa6860c…` and contracts `78e31787…` unchanged (contracts still differ from the freeze: the external 2026-10-19 edit noted in iteration 6). |
| New config fields | `flat_start_step_mm` 0.25, `flat_start_test_mm` 0.5, `flat_start_gap_mm` 1.0, `flat_start_ref_mm` 3.0, `flat_start_dev_mm` 0.02, `flat_start_max_shift_mm` 4.0 (0 turns the search off), `flat_start_chunk_mm` 25. Range-checked. Frozen gate thresholds untouched. |
| Tests touched outside the engine | `tests/test_prod_engine_inspect_a1_iter7.py` (new, 19 tests). No existing test changed. |
| Runs | Every engine run, test run, tier and probe sweep ran one at a time under `/private/tmp/claude-501/sherman_geom.lock` with `OMP_NUM_THREADS=1` (tools in `tools/`, scratch `a1_iter7/`). Run `v2` (physical set, G1 tier, tests, probe sweep) was the change as described; a last guard then keeps the reference parabola on at least 10 points (the robust line's last rejection round could leave fewer; `tools/guard_patch.py`). Run `v3` on that final engine repeated the physical set, the G1 tier and the tests: every bend result, score and gate record equals `v2`'s (only timings and timestamps differ), so the probe sweep (`v2`) stands for the final engine. |

## Diagnosis 1: the 10839000 FULL angles are the part's geometry (owner's MEDIUM question)

Method: cross-section profiles of each side's skins against that side's region-wide plane, in 0.5 mm bins of
distance from the bend's OML sharp line, per axial third (`tools/profile.py`; `diag/profile_10839000_full_*.txt`).

- **The 12 mm lips (B5, B7) are flat and overbent.** Past the arc (which ends about 3.2 mm from the sharp, so the
  real inner radius is about 1.8 mm against the CAD's 0.78 mm) the lip's outer skin is a straight line: windowed
  slopes -1.51 / -1.52 / -1.68 deg (B5) and -1.99 / -2.01 / -2.21 deg (B7) against the lip's region plane, at 2 mm
  steps out to the free edge. The parent walls are flat (slopes under 0.1 deg). The lip angle from its flat part
  is 88.4-88.7 deg (B5) and 88.2-88.5 deg (B7); the engine reads 88.62 and 88.49 deg. The region-wide lip planes
  (90.2 / 90.4 deg) are tilted by the arc points they include; the stations exclude those.
- **The base-edge bends (B9-B12) read 88.0-88.8 deg because the base is bowed near its long edges.** Both skins of
  the 313 x 206 mm base fall linearly away from its mean plane over the first 30 mm from the bend: -0.7 / -1.4 /
  -1.0 deg (B9, by axial third) and -0.8 / -1.8 / -1.1 deg (B11). The walls are flat (under 0.1 deg). The station
  angles follow the tilt: about 89.2-89.5 deg near the corners, 87.7-88.4 deg mid-edge. Against the base's mean plane the walls
  stand at 89.1-89.6 deg.
- **The same bends read nominal on another scan.** On scan_2 (same recipe, same engine) B5 / B7 / B10 / B12 read
  -0.03 / -0.43 / -0.17 / -0.27 deg against the FULL scan's -1.38 / -1.51 / -1.23 / -1.68 deg. A systematic angle
  bias of the engine would show on both.
- **Consequence.** The plan defines the angle as the one between the fitted flange planes at the bend, and the
  engine measures it there. Measuring against the base's mean plane would pass G3's known-good gate on this part
  by hiding a 1-2 deg local deviation, so it is not done. G3's known-good angle MAE stays above 1.0 deg on
  10839000 with any honest local measurement: whether this part is "known-good" at +-1 deg, or the gate should use
  the base's mean plane, is the owner's call.
- **The joggle angles (B1-B4).** The section through the joggle (`diag/joggle_section_10839000_full_y205.png`)
  shows its riser as a short S, about 2.5 mm along the sheet, with no flat material: the arcs' tangent zones take
  the whole riser. Its angle is therefore placed by the far flange's step (iteration 4's riser model), whose U is
  tan(35 deg) x 10 % of the riser length, about 4 deg standard. The 12 joggle readings on the three scans have MAE
  1.53 deg within U of 7.0-8.2 deg, all REVIEW. A direct slope of the S at its inflection would need validation on
  more than this one joggle family; not attempted.

## Diagnosis 2: an angle bias from the arc's tail (the change)

- Every station window started at `d_t + 0.5 mm`, where `d_t = (Ri + t) tan(sweep / 2)` uses the **CAD** inner
  radius. The profiles show where physical bends actually reach their flat material:
  - 15651004 (t 1.92 measured, CAD Ri 0.8 mm): both skins flatten about 4.5 mm from the sharp (real Ri about
    2.6 mm), and the window started at 3.3 mm. Windowed slopes over [3.3, 5.3] mm read +3.2 deg against -0.1 deg
    beyond; the station angle read 90.35 deg where the flat parts give about 89.6-89.8 deg
    (`diag/profile_15651004_full_*.txt`).
  - The transition is gentle, not a circular arc: on 15651004 B1 the skin sits 0.115 mm below its flat 1.8 mm
    before the flat begins, an effective radius near 14 mm. Most of that tail lies inside the stations' 0.15 mm skin
    band, so the station planes fitted it.
- The tail always bends toward the bend's centre, so it tilts each side's plane toward an underbent reading, and
  where it varies along the bend it scatters the stations (inflated U: 23553000 SCAN_2 B5 read +4.99 deg with U
  7.17 deg).

**The rule** (`metrology._flat_start`, applied per side in `measure_bend_geometry` before the stations are cut;
joggle risers placed by their step are skipped):
1. Take the side's slab points from the CAD start to the side's fit depth, and its denser skin (OML near 0 or inner
   near -t; both skins' arcs end on the same line), in axial chunks of 25 mm.
2. From the CAD start, step out by 0.25 mm. At each step, in each chunk, fit a robust line on up to 3 mm of skin
   that starts 1 mm beyond a 0.5 mm test strip (the gap keeps a slowly departing tail out of the line), and pool
   the strip's residuals over the chunks.
3. The strip is judged only when the line is straight (the sagitta of a parabola fitted on the points the robust
   line kept, at least 10 of them, within 0.02 mm, chunks' median) and the strip kept its skin (at least half the
   points the reference density predicts). A judged strip whose median residual is within
   0.02 mm toward the centre is flat: the window starts there. Otherwise the window steps on.
4. With no judged flat strip within 4 mm (and the window keeping 1.5 mm of reference) the CAD start stays: a short
   tab curved all along has no flat material to move to (unit-tested).
5. The window's fit depth (and the U term `tilt_deg = rms / depth`) counts from the new start. The shift per side
   is reported in the angle's evidence notes (`flat_shift_parent_mm`, `flat_shift_child_mm`).

Why it is general: it reads only the side's own skin profile in the scan; no part, family, file or recipe value
beyond the CAD bend geometry already used. A bow (radius of hundreds of mm) or a tilted panel is a straight line
over 3 mm, so neither moves the window (unit-tested); a gap in the scan at the window start drops no scanned point
(unit-tested).

Evaluated and not kept:
- A 0.01 mm departure (instead of 0.02 mm): more and larger shifts, overall MAE 0.902 deg against 0.898 deg, U up on
  several bends (`diag/angles_flat_start_dev0.01_all_cases.txt`).
- The same search at a side's far end next to an adjacent bend (`tools/flat_end_patch.py`): overall MAE 0.879 deg and
  a large U drop on 38172000 SCAN_1 B8 (4.19 -> 0.29 deg), but on 49024000 B2 it read a local dip of the outer skin
  (0.15 mm over 1 mm, the inner skin flat there) as a tail and moved +0.80 -> +1.18 deg (`diag/angles_flat_start_plus_far_end_all_cases_NOT_KEPT.txt`,
  `diag/profile_49024000_B2_far_end.txt`). Short windows between two bends leave too little skin to tell a tail from a
  local feature; it is a candidate for iteration 8 with a stricter test.

## Results

**Physical set** (frozen `prefreeze run-physical` / `score-physical`, all 27 A1 cases, serial under the shared lock,
`OMP_NUM_THREADS=1`; scratch `a1_iter7/phys`, run `v3` = the final engine, identical to `v2`; `validation/keynums_*.json`).
- Characteristic-level diff against iteration 6's final run (0.05 tolerance, `validation/physical_diff_vs_iter6.txt`):
  18 of 26 cases change; 40 angles, 25 flange sides, 11 positions and 1 backgauge move. No bend state changes.
  pc_b4719db3dd69 stays compile BLOCKED.
- Decision changes:

| Characteristic | Change | Bends |
|---|---|---|
| Angle | REVIEW -> PASS (5) | 15651004 FULL B2 (+0.57 -> -0.14), SECOND B1 (+0.79 -> -0.03), 3RD B1 (+0.71 -> +0.06); 23553000 FULL B5 (+0.75 -> -0.09); 50757000 V B9 (+0.53 -> -0.64, U 0.52 -> 0.25) |
| Angle | REVIEW -> FAIL (3) | 23553000 SCAN_2 B5 (+4.99, U 7.17 -> +1.69, U 0.55); 23553000 SCAN_1 B9 (-1.34, U 1.10 -> -1.37, U 0.22); 38172000 SCAN_1 B7 (30 deg tab, -1.71, U 0.78 -> -2.15, U 1.04) |
| Angle | PASS -> REVIEW (3) | 10839000 scan_1 B5 (-0.49 -> -0.67); 23553000 FULL B8 (-0.67 -> -0.69); 50757000 V B3 (-0.61 -> -0.63) |
| Flange | PASS -> REVIEW (4), REVIEW -> PASS (2) | sides of the bends above, moved by 0.01-0.2 mm with their OML sharps |
| Position | FAIL -> REVIEW (1), REVIEW -> PASS (1) | 23553000 SCAN_2 B11 (0.67 -> 0.63 mm), 38172000 X B7 (0.21 -> -0.02 mm) |

  - The three new angle FAILs are readings the stations now agree on (U down on two of them, 7.17 -> 0.55 and
    1.10 -> 0.22 deg). The traces of the search (`diag/flat_start_trace_*.txt`) show a clean tail on each: the test
    strip's median residual climbs monotonically to zero (23553000 SCAN_2 B5 -0.166 -> -0.007 mm over 1 mm,
    SCAN_1 B9 -0.095 -> -0.008 mm, 38172000 SCAN_1 B7 -0.078 -> -0.006 mm over 2 mm) against references whose
    sagitta stays within 0.006 mm. The 38172000 B7 tab is overbent by 2.15 deg on the flat part beyond its
    transition, not a tab curved all along.

| Angles vs nominal (159 measured, the two wrong-way hooks excluded) | Iteration 6 | **Iteration 7** |
|---|---|---|
| MAE, all | 0.946 deg | **0.898 deg** |
| mean U, all | 1.018 deg | **0.956 deg** |
| MAE / mean U, without the 14 readings whose U was above 3 deg (12 joggle risers, 23553000 SCAN_2 B5, 38172000 SCAN_1 B8) | 0.853 / 0.408 deg | **0.823 / 0.385 deg** |
| 15651004 (6 bends) | 0.698 deg | **0.364 deg** |
| 23553000 (27 bends) MAE / mean U | 1.002 / 0.659 deg | **0.788 / 0.348 deg** |
| 60 deg included bends (5): mean signed deviation | +1.58 deg | **+1.08 deg** |
| 11979003 (7, STL mesh) MAE / mean U | 0.374 / 0.799 deg | 0.470 / 0.525 deg |
| 10839000 (30) | 1.293 deg | 1.301 deg |
| Angles moved by more than 0.05 deg | | 37: 24 toward nominal, 13 away; mean signed change -0.38 deg |

**Gates** (frozen thresholds; physical scorecard of run `v3`, `validation/physical_scorecard_final.json`):

| Gate item | Iteration 6 | **Iteration 7** |
|---|---|---|
| G2(a) line metric (human Gold11 gate of record) | FAIL, F1 0.391 | FAIL, 0.391 (relabel 0.987) |
| G2(b) coverage angle / flange | 0.988 / 0.964 PASS | same |
| G2(c) known-good angle MAE (26 bends; legacy common 1.291) | 0.795 deg PASS | **0.798 deg PASS** (10839000 1.225, 11979003 0.470, 15651004 **0.453** (was 0.643), 37361005 0.352) |
| G2(d) known-good flange P95 | 0.392 mm PASS | 0.393 mm PASS |
| G2(e) runtime P95 / max | 27.9 / 31.8 s PASS | 27.7 / 31.5 s PASS |
| G2(f) repeatability P95 angle (gate 0.3 deg) | 0.426 deg FAIL | **0.306 deg** FAIL (the three 15651004 "repeats" are three stages) |
| G3 state match | 118/123 FAIL | 118/123 FAIL (same 4 disputed rows + the rolled part) |
| G3 known-good angle MAE (31 bends) | 1.186 deg FAIL | **1.147 deg** FAIL (10839000 1.241, Diagnosis 1; 15651004 **0.274**, was 0.753) |
| G3 known-good common bends vs legacy | 1.323 vs 1.222 deg | 1.324 vs 1.222 deg |
| G3 unformed reported as defect | 2 FAIL | 2 FAIL (the disputed rows) |
| G3 coverage, flange P95, blank, counts | PASS | PASS (flange P95 0.368 mm) |

**Line localisation and per-bend correctness**: unchanged to the last digit.
- BROAD (frozen gold, 24 scans): P 0.665, R 0.743, micro 0.702, family-macro 0.817 (best F1 lane 0.580 / 0.451 /
  0.508 / 0.543).
- GOLD_SET_V1 (23 scans): micro 0.990, family-macro 0.974 (best F1 lane 0.566 / 0.583). Gold11 relabel F1 0.987 (best
  F1 lane 0.547); human Gold11 0.391 (best F1 lane 0.473).
- Per-bend correctness given CAD (riser-aware): 183/203 = 0.901, family-macro 0.910; false-PASS proxies 3 (the same
  bends: 38172000 SCAN_1 tab B1 and the two disputed 10839000 scan_2 rows).

**G1 regression tier** (DEV seeds, 50 cases, 7 families; scratch `a1_iter7/g1_v3`, identical to `g1_v2` apart from timings;
`validation/g1_regression_scorecard_final.json`).
- regression_verdict **PASS**, no failed gate. 29 of the 32 gate records are byte-identical to iteration 6's; the
  other 3 are the informational runtime, RSS and tier-budget records (P95 13.66 -> 13.54 s). Coverage 0.997, false
  PASS 0/553, false FAIL 0, U calibration 0.984 (as before).
- Non-qualifying as before: TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL, FREEZE_LEDGER_MISMATCH (the external
  contracts edit noted in iteration 6). The synthetic parts are built with their CAD radius, so their arcs end where
  the CAD says and the search leaves their windows where they were.

**CAD-stage probes** (E1 stage meshes of all A1 recipes, every stage, springback +-2 deg on formed bends and +-1 deg
kinks on unformed ones, through E2's scanner; iteration 4's cached seed-23 probe scans; `tools/sbsweep7.sh`,
`validation/probe_sweep_final.log`, `validation/probe_sweep_vs_iter6.txt`).
- 113 probes, 1006 bend states: 0 wrong states, 0 not VALID, 0 angle false PASS / false FAIL, 501/503 angles within
  U, 96 other FAILs (position, springback-driven): every aggregate equals iteration 6's, and so does every probe's
  state, validity and FAIL list.
- 3 of the 503 angle errors moved, each by at most 0.05 deg and toward zero or within 0.003 deg of it (23553000 S16
  B14 -0.001 -> +0.002, 48811000 S2 B1 0.070 -> 0.061, S3 B3 0.222 -> 0.172 deg). The probes are built with their CAD
  radius, so their arcs end where the CAD says.


## What was not changed, and why

- **G3's known-good angle gate on 10839000**: Diagnosis 1. Passing it would need the angle against the base's mean
  plane, which hides a real 1-2 deg local deviation. Owner decision.
- **The joggle angles (HIGH in the E2E audit)**: Diagnosis 1, last item. They stay REVIEW with U 7-8 deg, which is
  honest for a riser without flat material. The joggle's step itself is measured directly from both flanges; a
  step characteristic would be decidable, but the recipe contract has none.
- **"A high-U REVIEW outranks a clear FAIL as the part's worst characteristic"**: the ranking is
  `contracts/production/inspection_result.py` (summary `worst`, key `exceedance_ratio = (|d| + U) / tol`), not
  engine code. Suggestion for the contract owner: rank by decision first (FAIL, then REVIEW, then PASS), then by
  the ratio.
- **The 2 tab-merge FPs on 38172000 SCAN_1** (B1/B9): a labelling convention and a contract question (iteration 6);
  unchanged.
- **The recipe-excluded fold (28870000 FULL)**: no recipe row to report it on (iteration 6); unchanged.
- **The rolled 28870000 SCAN_1** (UNKNOWN vs truth NOT_FORMED): relaxing the wrong-part test risks G1's
  wrong-or-mirrored-never-PASS; unchanged.
- **The measured thickness spread between scans** (10839000: 1.444 / 1.468 / 1.275 mm against 1.5 CAD; 38172000:
  1.375-1.619 mm): most likely the fusion offset between the two skins' scan setups (the scanner model card's
  duplicate-skin offset). It enters U through `u_t`; changing the thickness rule was out of this iteration's
  scope.
- **The far-end version of the search**: evaluated, not kept (Diagnosis 2).

## Generality

- No SKU, family, part-id, file-name or metadata-count logic was added, and E3's inputs are unchanged.
- The rule reads each side's own skin profile in the scan, with the CAD bend geometry already used (radius,
  thickness, sweep, fit depth). Its thresholds are the seven engine config fields above, with range checks.
- Checked on all 26 runnable physical A1 cases, the G1 regression tier, the CAD-stage probes and 19 unit tests.

## Risks

1. **A side whose near flat part is genuinely short and curved.** If the 3 mm reference beyond the test strip is
   straight while the strip dips toward the centre for a reason other than the bend (a local dent, a scan
   artefact), the window moves out and the plane is fitted further from the bend. Guards: pooled over 25 mm
   chunks, the strip must keep its skin, the reference must be straight, at most 4 mm and never below 1.5 mm of
   reference. The far-end variant showed this failure (a local dip) and was not kept.
2. **Fewer points per station on short sides**, so a larger station scatter in U on some bends (11979003 B2: U
   0.64 -> 0.89 deg). U stays a valid bound; decisions can move from PASS to REVIEW (3 angles here).
3. **New FAILs where the old reading hid a deviation in its scatter or its tail** (3 angles). They are as right as
   the flat-flange definition of the angle; the search traces show a clean tail on each, and no physical truth
   exists for these bends.
4. **Runtime**: negligible (physical P95 27.9 -> 27.7 s).

## Tests

| Test | Result |
|---|---|
| `tests/test_prod_engine_inspect_a1_iter7.py` (new): the window moves past a real tangent line beyond the CAD one (circular arc; a 120 deg sweep; the inner skin alone; a deep tail that leaves the skin band), the CAD radius keeps the CAD window, a tilted side and a panel bowed either way (400 mm radius) do not move it, a short tab curved all along keeps the CAD start, a gap in the scan at the window start drops no scanned point, the cap and the switch-off, too few points, the station plane on the moved window loses most of the tail's tilt (circular arc and gentle springback transition), config range checks. Pure numpy; no part named. | 19 passed |
| The final engine: `test_prod_engine_inspect_cli.py` (E3's E2E tests, E2 -> E1 -> E3), `_units.py`, `_a1.py`, `_a1_iter2.py` ... `_a1_iter7.py` | **169 passed** (211 s, under the lock, on the final engine; iteration 6: 150) |

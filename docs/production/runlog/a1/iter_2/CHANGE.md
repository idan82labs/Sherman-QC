# A1 iteration 2: engine change record (2026-10-18)

All physical numbers below come from consumed development scans (engineering ledger). None of them is
unseen-family accuracy. Synthetic numbers are E2 DEV seeds only.

## Provenance

| Item | Value |
|---|---|
| Snapshot before editing | `engine_before.tgz`, sha256 `36933708fe2d6c21d8ad9b3e559c154ad5f24a9c956439883480df3b9ac2b487`, taken before the first edit. It equals iteration 1's final engine (iteration 1 snapshot + `iter_1/engine.diff` + its version and docstring edits; checked by reconstruction). |
| Diff against the snapshot | `engine.diff`: five files under `domains/production/engine/inspect/` (`registration`, `engine`, `articulated`, `metrology`, `config`). No change to `eval/` or `compile/`. |
| Engine | `e3-1.3.0` (was `e3-1.2.0`), default config `417f580b2d9369040411137a153b96458c973a3f50401d77e20be4b01a8b9d46` (was `74c03050…`) |
| Algorithm ids bumped | `registration` `…_polygon_flange_ends_v3`; `articulated_fit` `…_displaced_fold_window_short_child_v3`; `metrology` `…_station_profile_short_side_consistency_v4` |
| Code fingerprints | freeze-style: engine `4b4bdb0f…`, compile `6147b43c…` (= freeze), eval `8aa6860c…` (frozen), contracts `5f7e6a8c…`; identical at the start and end of the measurement runs. G1 runner: engine `f9879fab…`, compile `d4e870e8…` (unchanged), eval `8aa6860c…` (unchanged). |
| Ledger | check OK, anchor `cf01e709…` checked (`frz_d57683acfd83`) |
| Previous generality check | clean, so nothing to remove. Iteration 1 improved on iteration 0, so nothing to revert. |
| Tests touched outside the engine | new `tests/test_prod_engine_inspect_a1_iter2.py`; one existing test adapted in `tests/test_prod_engine_inspect_cli.py` (see Tests). |

## Diagnosis

From the iteration-1 scorecard, its per-bend rows and the overlays (opened and checked):

1. **Six bends had no position and no line** (`REGISTRATION_UNTRUSTED`): 49204020 B3/B4/B6/B12 and 15651004
   B1/B2. They cost 6 BROAD TPs and 6 per-bend rows. Their angles and flanges were measured, which
   the per-bend proxy counted as six "PASS without a published line".
   - 49204020: the datum is bounded by bends on both of its ±y sides and has no hole. Its free edges carry
     zero information along y (`diag/pc_e080ae094f35_datum.png`).
   - 15651004: two Ø6 holes 25 mm apart on a 155 × 28 mm strip leave a rotation-plus-y direction at 2.03
     effective observations, below the threshold of 3 (`diag/pc_03395097cd96_datum.png`).
   - A CAD-only check of all 17 compiled A1 parts (`diag/cad_datum_dof.txt`) finds that these are the only
     two whose datum leaves an in-plane direction undetermined.
2. **The 38172000 angle FAILs are real geometry, not engine error.**
   - What was checked: independent robust plane fits on each flange's OML skin of the registered raw
     scan, at three depth windows from the measured sharp (`diag/angle_check_38172000.txt`).
   - Result: they reproduce every engine value within about 0.2°. Examples: V B9 −2.81° vs −2.78/−2.89/−3.02;
     X B5 +4.82° vs +6.56/+5.39; X B7 +5.44° vs +5.61/+4.12.
   - Reading: V is overbent by about 1–2° on most 90° bends. X's two 30° tabs are underbent by about 5°.
   - Several V flanges bow (the far window reads up to 1° more), so the angle depends on where it is taken.
     The engine's station convention, flat material next to the bend, is the near window.
   - No change was made.
3. **Not engine-fixable here**:
   - The 13 cases refused by the amendment-7 gate (process: the G3 truth table is not frozen).
   - The BROAD gold-convention FPs/FNs (all 12 FP lines are confirmed by the relabel or GOLD_SET_V1).
   - Five bends with a compile-side unavailable flange nominal: 50757000 B1/B4 side B on both scans, and
     47266000 B1 (`UNSUPPORTED_FEATURE`). E1's nominals are hash-bound, so the engine may not invent them.
4. **Forward-looking (the hidden staged cases).** A CAD-only risk scan of every A1 part
   (`diag/cad_bend_risks.txt`) found two joggles in 10839000.
   - 10839000 is the Gold11 family with three truth-relevant scans, and it is the §6 E2E real part.
   - Each joggle is 35° UP + 35° DOWN, 3.6 mm apart OML, with a 2.6 mm straight riser and Ri 0.1 mm. Both
     joggles are formed first in the sequence (B1–B4).
   - An E2 → E1 → E3 synthetic reproduction of that geometry (`tools/probe_profile.py`) shows the
     iteration-1 engine reads a joggle as not yet formed. It also gives confident false FAILs on the
     same-direction variant (table below).
   - Cause: the riser supports its own line over about 1.7 mm only. The joggle's far flange is parallel to
     the parent and about one thickness off it. So it reads as the parent's flat continuation (a ±3 mm
     window around nominal at every angle), or shallow line pairs crossing its two skins outscore the
     riser (`diag/joggle_B2.png`).

## What changed, and why

### 1. Cut flange ends fix a datum direction the datum leaves free (`registration.py`, `engine.py`)

**Principle.** Take a flange reached from the datum only through bends whose axes are parallel to one
direction `a` of the datum plane.
- Whatever those bends' angles and bend-line positions, they move the flange only across `a`.
- Its blank's cut edges perpendicular to `a` (the flange's ends) therefore stay where the blank put them
  along `a`, whether formed or still flat (stage independent).
- Their scanned positions observe the datum frame along `a` exactly as the datum's own free edges do.
- No reported characteristic can move them: not a bend-line position, not an angle, not a flange length.
  This is amendment 9's "articulated ICP over the datum and all fitted flanges", restricted to the
  observations that cannot absorb a reported error.

**Implementation.**
- `child_edge_rows`: every usable free edge of every fitted (FITTED or FLAT) child panel whose
  outward normal lies within 5° of the path's common axis.
  - Each edge is placed by the panel's fitted pose and binned exactly like a datum free edge: 3 mm bins,
    observed at the cut wall when it is seen, else at the 3rd extreme skin point with weight 0.15.
  - Only points owned by that panel, or by no panel, are used.
  - Articulated-model material within the band beyond the edge shortens the band. This covers a slot's
    far side, a neighbour across a relief, and the window a flat tab sits in. If the band would be under
    1 mm, the edge is not used.
- `resolve_free_directions`: moves **only** `fit.undetermined`.
  - The directions the datum's own edges and holes fix are never touched; a test checks this.
  - Edges must be consistent along their length (bin MAD-σ ≤ 0.25 mm). A Huber solve drops the worst edge
    while its median residual exceeds 0.5 mm.
  - At least 2 consistent edges and at least 3 effective observations are needed per direction, and the
    frame may move by at most 3 mm. Otherwise the direction stays withheld, as before.
  - Covariance along the resolved directions = the edges' sandwich covariance + their disagreement (the
    blank's cut tolerance). Unwalled rows add their erosion gain to the pose-bias bound.
- `engine._refine_free_directions`: runs on the chosen pose when it has an undetermined direction.
  - The refined pose is articulated again and observed a second time.
  - It is refused if a decoded state changes, if the model explains more than 0.02 less of the scan, or if
    the budget runs out.
  - New warning: `IN_PLANE_POSE_FROM_FLANGE_ENDS`. The heatmap becomes available when nothing stays free.

**Effect on the two real scans** (engine log):
- 49204020: 4 flange ends (P3 and P0 lateral ends at ±40 mm) move the frame by −0.326 mm in y. Edge scatter
  0.037 mm, σ 0.020 mm. The second pass moves it by 0.000.
- 15651004: the two ends of the far flange (P3, 77 mm from the datum centre) fix the rotation-plus-y
  direction (shift (rotation·L, tx, ty) = (−0.054, −0.001, −0.048) mm, 41 effective observations).
- All 6 positions are now measured and PASS:
  - 49204020: B3 +0.22, B4 −0.08, B6 −0.02, B12 −0.14 mm.
  - 15651004: B1 +0.10, B2 +0.10 mm.
  - U is 0.10–0.12 mm.
- 49204020 becomes ALL_MEASURED_IN_TOLERANCE.
- Overlays `01`–`06` show all six lines on their bends.
- Cross-check on a scan whose datum is fully determined (37361005): the same flange ends agree with the
  datum frame within 0.05 mm (`tools/proto_lateral.py`).

### 2. Joggles and short steps in the hinge search (`articulated.py`)

- **The line window follows the fold kinematics and is tight at flat (`rho_window`).**
  - The child's inner-skin line is searched around the offset the fold kinematics give at each angle
    (`HingeFrame.rho_iml_at`). A partially formed bend has not consumed its strip.
  - The half-width is `min(slack, 0.4 mm + slack·|sin θ|)`. A bend line placed `d` off moves the child's
    line by `d·sin θ` only, so a flat continuation must be the parent's own plane.
  - Before this change, ±3 mm was searched around the final-pose offset at every angle.
  - The displaced-flange search uses the same window rule.
- **Hinge continuity for short children (`_hough(continuity=True)`).**
  - Applies when the child carries a bend parallel to its own within the fine reach (a joggle's riser, a
    short step).
  - Within 10° of flat, a candidate line counts only when its material is continuous from the hinge. Every
    0.5 mm sub-band of the first 2 mm must hold at least a fifth of the line's mean support.
  - A parallel sheet one thickness off starts millimetres out, and a crossing line pair touches it far from
    the hinge.
- **A short child grows only up to the parallel bend's tangent lines + 1 mm**, within that bend's axial span
  only (`_short_child_limit`). Beside a tab, the panel runs on to its own far edge; without the span limit,
  50757000's tab carrier was cut short.
  - Without this limit, a 2.6 mm riser's region (normals smeared at a 2.7 mm normal scale) grew onto the
    flange beyond it. On the same-direction variant this measured the riser at 70° and gave false FAILs.

### 3. Short sides and stations that leave their panel (`metrology.py`)

- **The station plane fits never run past an adjacent bend's tangent line** (`_flat_depth`). The old 4 mm
  depth floor reached the next bend's arc and flange when a side is shorter than that. FREE_EDGE sides are
  unchanged.
- **Safety net.** A bend whose median station sweep differs by more than 10° from its region-wide panel
  planes' sweep is not measured: UNKNOWN(AMBIGUOUS_SEGMENTATION). Its ADJACENT_BEND neighbours then read
  ADJACENT_BEND_NOT_MEASURED.
  - Such stations settled on another sheet, so every value from them would be confidently wrong.
  - On the one-side-scanned joggle, consistent station fits on the far flange had produced FAILs of up to
    32° and 25 mm with small U.
  - The check never triggers on the 13 real scans or on G1.

### Synthetic joggle probes (E2 DEV, custom profile 80 / 21.4 / 3.6 / 13 mm, t 1.5, sweeps 90/35/35)

The same five cases, generated once, were run through the iteration-1 engine (`diag/joggle_probes_baseline.txt`)
and through iteration 2 (`diag/joggle_probes_final.txt`):

| Case | e3-1.2.0 (iteration 1) | e3-1.3.0 (iteration 2) |
|---|---|---|
| Joggle UP/DOWN, FULL, nominal | B2, B3 NOT_YET_FORMED | 3/3 FORMED. Positions ≤ 0.005 mm, flanges ≤ 0.038 mm. Angles 0.05° / 0.29° (U 0.8°, one REVIEW). |
| Joggle, FULL, injected errors | B1, B2 NOT_YET_FORMED; B3 UNEXPECTED_FORMED | 3/3 FORMED, every measured error inside U |
| Same direction 35+35, FULL | B3 NOT_YET_FORMED; **4 false FAILs** (B2 −34.7°, flanges +2.2 mm) | 3/3 FORMED, ALL_MEASURED_IN_TOLERANCE, errors ≤ 0.056 mm / 0.05° |
| Joggle, TOP_ONLY (one skin) | B2, B3 NOT_YET_FORMED | 3/3 FORMED. B2 UNKNOWN(AMBIGUOUS_SEGMENTATION); no FAIL. |
| Joggle at a mid stage, second bend shifted 1.47 mm (riser straight about 1.1 mm) | B1, B2 NOT_YET_FORMED | **unchanged (open)**: NOT_YET_FORMED, no FAIL |

States correct: 5/15 → 13/15. False FAILs: 4 → 0.

## Results: engine vs the best F1 lane on the same scans

Frozen scorer: 5° / 4 mm / 0.6 matcher, apex locus, raw frame. 13 A1 cases are still refused by the
amendment-7 gate (process) and count as misses.

| Population | iteration 1 | **iteration 2** | Best F1 lane |
|---|---|---|---|
| BROAD, all 26 A1 cases, family-macro / micro | 0.626 / 0.569 (P 0.84) | **0.667 / 0.607** (TP 68, FP 12, FN 76; P 0.85) | 0.543 / 0.508 |
| BROAD, the 13 scans E3 runs, family-macro / micro | 0.832 / 0.816 | **0.896 / 0.861** (TP 68, FP 12, FN 10; P 0.85, R 0.87) | 0.649 / 0.620 |
| G2(a) Gold11 relabel (gold of record), the 4 cases E3 runs / all 9 | 1.000 / 0.574 | 1.000 / 0.574 (unchanged; the fixed parts are not Gold11) | 0.561 / 0.547 |
| G2(a) human Gold11 (historical), 4 / 9 | 0.724 / 0.467 | 0.724 / 0.467 | 0.717 / 0.473 |
| GOLD_SET_V1, 13 A1 scans with STEP that E3 ran, micro / family-macro | 0.961 / 0.936 (TP 74, FN 6) | **1.000 / 1.000** (TP 80, FP 0, FN 0) | 0.597 / 0.653 |

The legacy arm stays informational only (0.088 human / 0.140 relabel on G2(a)) and is not a bar.

Per case, BROAD gold TP / FP (only changes):
- 15651004 `pc_03395097cd96`: 1/0 → **3/0** (best lane 3/0). Family F1 0.286 → 0.500; two of the family's
  three scans are refused.
- 49204020 `pc_e080ae094f35`: 8/0 → **12/0** (best lane 6/0). Family F1 0.80 → 1.00.

All 12 remaining FPs are the documented gold-convention lines (38172000 X/V, 37361005, 11979003). Every one
is confirmed by the relabel or GOLD_SET_V1; the proxy `line_confirmed_any_rate` is 80/80.

**Per bend given CAD** (report-only proxy, 80 recipe bends on complete scans E3 ran):

| Measure | iteration 1 | iteration 2 |
|---|---|---|
| Correct vs BROAD gold | 57/80 = 0.713 | **63/80 = 0.788** |
| Correct vs any gold of record | 69/80 = 0.863 | **75/80 = 0.938** |
| PASS on a bend without a published line | 6 | **0** |
| Strict false-PASS proxy (PASS on a line no gold confirms) | 0 | 0 |
| Decisions PASS / REVIEW / FAIL / UNKNOWN | 236 / 45 / 27 / 12 | 241 / 46 / 27 / 6 |

The 5 bends still not "correct" vs any gold are the compile-side unavailable flange nominals listed above.
No physical dimensional truth exists, so this proxy does not measure real false PASS.

**Other real-scan changes** (all 13 runs compared characteristic by characteristic with iteration 1):
- Apart from the two fixed scans, values move by at most 0.05 mm and 0.02°. The exception is 38172000 X
  B5 side B, which moves 0.08 mm; it is a REVIEW side with U 30 mm.
- The only decision changes are the six recovered positions (UNKNOWN → PASS) and 11979003 B2 angle
  PASS → REVIEW (|dev| + U = 1.007 vs 1.0).
- No state changed. No AMBIGUOUS_SEGMENTATION appeared.

## Gates

| Gate | Result |
|---|---|
| **G1, synthetic `regression` DEV tier** (50 cases, 7 families, scanner model card, final code) | `regression_verdict` **PASS**, 0 failed accuracy gates. Gate for gate identical to iteration 1, except flange bias 0.009377 → 0.009379 mm and the slopes in the 6th decimal. Values: flange P95 0.103 mm, angle bias 0.0015° and P95 0.019°, position P95 0.073 mm, coverage 0.997, false PASS 0/553, false FAIL 0, U calibration 0.984, median U 0.106 mm / 0.116°, decisiveness 0.983, stage state 1.0, 0 engine failures. Non-qualifying by construction (TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL; G1.SUITE fails below 200 cases). |
| **G2** | FAIL, `final: false` (no truth table). Sub-gates in the table below. |
| **G3** | NOT_EVALUABLE: the frozen truth table does not exist. |
| **G5** | Not re-run here. No `backend/`, `domains/bend/` or `scripts/` file was touched. The known G5 failure (`tests/test_vendor_unfold_api_contract.py`, another workflow's `vendor_setup_service.py` edit) is unrelated. |

| G2 sub-gate | Result |
|---|---|
| (a) line metric | FAIL. Same as iteration 1: refused cases count as misses. |
| (b) coverage | FAIL only on refused and blocked cases: angle 55/91, flange 108/178. 100 % on every runnable case. |
| (c) known-good angle | FAIL on coverage only (14/32 vs legacy 16/32). MAE 0.437°; on the 8 common bends 0.506° vs legacy 1.360°. |
| (d) known-good flange | PASS: P95 0.522 mm, max 0.860 mm |
| (e) runtime | P95 19.7 s, max 20.5 s, 0 crashes; `all_cases_run` FAIL (refused cases) |
| (f) repeatability | NOT_EVALUABLE (2 of the 3 15651004 scans refused) |
| (4d) compile rate | PASS (24/26 = 0.923) |

## Tests

| Test | Result |
|---|---|
| `tests/test_prod_engine_inspect_a1_iter2.py` (new). Pure numpy: flange-end rows and resolution (free ty, only the free direction moves, one end refused, an inconsistent end dropped, two disagreeing ends refused, owned and blocking material, crossing-bend paths and 10° edges excluded), config ranges and hash, the angle-dependent window, hinge continuity on a joggle section and on a true flat continuation. E2 → E1 → E3 end to end: a FULL joggle (3 FORMED, no FAIL, positions and flanges within U of truth), a TOP_ONLY joggle (no FAIL), the same-direction step (3 FORMED, angles within U), and the HAT with its datum edges unusable (flange ends fix the pose; every position within U of truth). | 13 passed (about 23 s, under the lock) |
| `tests/test_prod_engine_inspect_cli.py` (E3's 32 E2E tests, under the lock) | 32 passed. One test adapted: `test_a_degenerate_in_plane_pose_withholds_only_the_position_and_the_heatmap` now marks every panel's free edges unusable, not only the datum's. Otherwise the flange ends would legitimately fix the pose this test needs to stay degenerate. Its intent and assertions are unchanged. |
| `test_prod_engine_inspect_units.py` + `test_prod_engine_inspect_a1.py` | 55 passed |
| Runs | G1 regression tier, `compile-frozen`, `run-physical`, `score-physical`, analysis, overlays and probes ran one at a time under `/private/tmp/claude-501/sherman_geom.lock` with `OMP_NUM_THREADS=1`. Peak RSS 0.92 GiB for an engine run on a real scan. |

## Generality

- There is no SKU, family, part-id, file-name or metadata-count logic. E3's inputs are unchanged.
- Every new rule is geometric: parallel bend axes, edge normals, model material in a band, the fold
  kinematics, the angle-dependent window, continuity from the hinge, a parallel bend's tangent lines, and
  station-vs-region sweep agreement.
- The parts named above were found by CAD-wide scans over all A1 parts (`diag/cad_*.txt`). The fixes were
  verified on synthetic geometry before any real scan was re-run.
- All new thresholds are engine config fields with ranges: `lateral_*`, `hinge_rho_slack_flat_mm`,
  `flat_continuity_*`, `short_child_margin_mm` and `station_region_max_deg`. The frozen gate thresholds are
  untouched.

## Risks

1. **Flange ends assume the blank's cut is right along the bend axis.**
   - A flange whose bend line is skewed (not parallel to nominal) by α moves its ends along the axis by
     about `h·α` (h = the end's height on the flange).
   - The defences: at least 2 ends must agree within 0.5 mm, their disagreement enters U, and the datum's
     own determined directions are never moved.
   - A real skew shared by every flange end would still bias the free direction.
   - Wider than the probes: only the two A1 parts with a free datum direction (and the HAT test) exercise it.
2. **The flat window is now 0.4 mm at 0°** (was 3 mm).
   - An unformed bend whose continuation sits more than 0.4 mm off the parent plane, through a kink of
     ≤ 4° or a strongly bowed parent near the bend, can lose its flat line.
   - G1's staged cases are unchanged (stage state 1.0), but real staged scans (G3) have not been seen.
   - The window widens with `3 mm·|sin θ|`, so a kink of a few degrees is still covered.
3. **Hinge continuity and the growth limit apply only to children carrying a parallel bend within the fine
   reach.**
   - An unformed bend whose hinge zone is occluded (a clamp over the first 2 mm) would lose its flat
     candidate there.
   - 23553000's 18.2 mm hooks (reach 18.75 mm) fall under the rule and are unseen.
4. **The station/region consistency check** turns a bend into UNKNOWN when its stations and region planes
   disagree by more than 10°. A strongly twisted long flange could trigger it. It never fired on the real
   scans or on G1.
5. **Open: a joggle shortened by a bend-line defect** (riser straight about 1 mm) is still read as not yet
   formed. Fixing it needs a chain model that fits both bends' angles and the riser length jointly.
   - A joint cross-section search over (θ1, θ2) was tried and removed: without a riser-length DOF it traded
     angle for riser length (24° instead of 35° on the shifted joggle).
   - The bends are then NOT_YET_FORMED with NOT_APPLICABLE characteristics, never FAIL. In G3 terms it is
     a state error on such defective parts.

## Candidates for iteration 3

- Chain fit for short children: angles plus the riser length (the shifted-joggle probe).
- Once the G3 table is frozen, review the staged behaviour of fixes 2 and 3 on 10839000 (joggles) and
  23553000 (hooks), plus the 0.4 mm flat window on real unformed bends.
- The parametrisation of `undetermined` (the eigen test uses the outline's vertex mean as the rotation
  centre, so a short-lever rotation can couple with a translation). Not needed by any A1 part now.
- `docs/production/ENGINE_INSPECT.md` still names the e3-1.1.0 algorithm ids; that document has its own
  owner (A1 only writes `engine/**`).
- **Process remains the biggest lever**: freeze the G3 visual truth table (amendment 7) so the 13
  truth-relevant A1 cases run.

## Files

| Path | Contents |
|---|---|
| `engine_before.tgz`, `engine.diff` | Snapshot and diff |
| `raw/` | Frozen `physical_scorecard.json`, `subset_scores.json`, `goldset_scores.json`, `per_bend.json`, `per_bend_ext.json`, and the final G1 regression scorecard (json and md) |
| `overlays/` | The six recovered bends: frozen G4 close-up and raw-frame context (`INDEX.json`) |
| `diag/` | Datum renders, joggle cross-sections (`joggle_B2.png`: iteration-1 window; `joggle35top.png`; `joggle_stage_B1.png`: the open shifted joggle), `joggle_probes_{baseline,final}.txt`, `angle_check_38172000.txt`, `cad_datum_dof.txt`, `cad_bend_risks.txt` |
| `tools/` | Run, compare, probe, debug and measurement scripts. Work directory: `…/scratchpad/a1_iter2` (outside the repository). |

Custody: only `cases_A1.json` cases were run, read or scored (13 runnable plus the compile-blocked one).
GOLD_SET_V1 rows were scored for A1 scans only. Nothing `*SEALED*`, FRESH, HOLDOUT10, N001–N016 or EHO4
was read. No engine output exists for the 13 truth-relevant cases. The CAD-only risk scans read compiled A1
recipes only, never their scans. No credentials were touched.

# A1 iteration 3: engine change record (2026-10-18)

All physical numbers below come from consumed development scans (engineering ledger). None of them is
unseen-family accuracy. Synthetic numbers are E2 DEV seeds or E1 CAD stage models seen through the E2 scanner;
none of them is physical truth.

## Provenance

| Item | Value |
|---|---|
| Snapshot before editing | `engine_before.tgz`, sha256 `2e595827aa02c074b9e01cd95330cef28121dfcb5bb12df36d85ef739ecf6c12`, taken before the first edit. It equals iteration 2's final engine: iteration 2's snapshot plus `iter_2/engine.diff` reproduces it exactly. |
| Diff against the snapshot | `engine.diff`: five files under `domains/production/engine/inspect/` (`articulated`, `config`, `engine`, `metrology`, `registration`). No change to `eval/` or `compile/`. |
| Engine | `e3-1.4.0` (was `e3-1.3.0`). Default config `9ec3bb56…` (was `417f580b…`). |
| Algorithm ids bumped | `articulated_fit` `…_short_child_window_support_tabs_one_skin_v4`; `state_decode` `…_symmetric_near_twins_v2`; `metrology` `…_footprint_edge_v5` |
| Code fingerprints | Freeze-style: engine `ab513015…`, compile `6147b43c…` (= freeze), eval `8aa6860c…` (frozen), contracts `5f7e6a8c…`. The G1 run manifest records engine `9525cfff…`. |
| Ledger | check OK, anchor `cf01e709…` |
| Previous generality check | Clean, so nothing to remove. Iteration 2 improved on iteration 1, so nothing to revert. |
| Tests touched outside the engine | New `tests/test_prod_engine_inspect_a1_iter3.py`. No existing test was changed. |

## Diagnosis

1. **The last scorecard and its worst bends.**
   - I opened the worst-bend overlays myself: 38172000 X B7 and V B9/B5, 50757000 X B5, 11979003 B1.
   - They show what iteration 2 found: real over- and underbend on 38172000, the 50757000 sc_752e defect chain, and
     gold-convention lines.
   - On the 13 runnable scans the engine is at the reachable ceiling: GOLD_SET_V1 1.000, and the BROAD
     false-positive lines are convention lines.
   - Nothing there is engine-fixable without part-specific logic.
2. **Where the score is still open.**
   - The 13 truth-relevant A1 cases are half of BROAD, 5 of 9 G2(a) Gold11 cases and all of G3. They stay refused
     until the G3 truth table is frozen (amendment 7), and the engine has never seen them:
     - 10839000 scan_1, scan_2 and FULL (joggles; the §6 E2E real part);
     - 23553000 FULL, SCAN_1 and SCAN_2;
     - 38172000 SCAN_1 and SCAN_2;
     - 28870000 FULL and SCAN_1;
     - the 11979003 blank;
     - 15651004 SECOND and 3RD.
   - Amendment 7 forbids engine output on them before the truth table, so they were not run, rendered or read.
3. **A CAD-stage probe harness** (`tools/stage_probe.py`, `tools/springback_probe.py`). Every compiled A1 recipe
   has E1 stage models S_0…S_N. For a recipe and a stage:
   - The stage's display mesh is posed randomly and scanned with E2's virtual ZEISS cell: the amendment-4b model
     card, FULL plan, merge voxel 0.57 mm, about 0.40 mm median spacing, as on real exports.
   - E3 then runs on that virtual scan through its CLI.
   - The truth is the stage's formed set. The geometry is nominal, so every FAIL is false.
   - The springback variant re-poses every panel by E3's fold kinematics: ±2° on formed bends, ±1° kinks on unformed
     ones. Its self-check reproduces S_k to 0.0002 mm.
   - A baseline engine tree (the snapshot) and the new engine run on identical scans.
   - CAD only: no physical scan of any case was read, rendered or run by these probes.
4. **The baseline sweep** covered every compiled stage of the six recipes whose families have truth-relevant scans:
   10839000, 23553000, 38172000, 11979003, 28870000 and 15651004, 57 probes. It found eight general failure classes,
   all of them on those families. The unbounded crease surfaced only once fix 2 made the joggle sides measurable:

   | Class | Where on the probes (baseline e3-1.3.0) | Effect |
   |---|---|---|
   | Tab lying flat in its window | 38172000 B5–B8 at S0–S7; 23553000 B8/B9/B15 at S0–S14 | AMBIGUOUS_SEGMENTATION, subtree UNKNOWN: a G3 state error at every early stage |
   | Joggle half made | 10839000 S1 | B1 NOT_YET_FORMED, B2 UNEXPECTED_FORMED, position FAIL −2.85 mm |
   | Near-symmetric pose, true pose not articulated | 10839000: the 180° image was chosen at S0, S1, S2 and S4 | S2: the other joggle decoded formed. S0/S4: right states, every bend identity swapped (G4 TAG_OK) |
   | Both joggle bends formed: line drift beyond a 2.6 mm riser | 10839000 S9–S12 | 8–16 characteristics UNKNOWN(AMBIGUOUS_SEGMENTATION) per stage, 16 on the FULL-part stage S12 |
   | One skin: mid-plane drift and skin identity | 10839000 S10 | FAIL +1.50 mm on two P1 lengths, U 0.11 and 0.24 |
   | FREE_EDGE region running through coplanar panels | 10839000 S12 (the FULL part) | B5/B7 side A FAIL +300.4 mm |
   | Unbounded unplanned crease | 10839000 S11 (exposed once fix 2 made the side measurable) | FAIL −6.15 mm on a 13 mm tab |
   | Stepped datum hole read as filled | 28870000 S0 and S1 | WRONG_PART_SUSPECTED: INVALID, every bend UNKNOWN |

   The springback sweep of the bound-only engine then found a ninth: a shallow line through the sheet beyond a short
   riser outscoring the riser's own line (10839000 S5 and S7; fix 2's support cut).

## What changed, and why

Every fix below is a geometric rule found on the CAD-stage probes. None names a part, family or file, and E3's
inputs are unchanged. Each rule is tested in `tests/test_prod_engine_inspect_a1_iter3.py`.

### 1. A tab lying flat in its window belongs to the tab (`articulated.py`)

- **Cause.** A panel's footprint was its outer outline only. While a tab cut out of a window was still flat, the
  window's interior lay inside the datum's outline, so the datum owned the tab's voxels. The tab could not grow
  (AMBIGUOUS_SEGMENTATION), and everything below it became BEND_STATE_UNKNOWN. On the probes this hit 38172000
  B5–B8 at S0–S7 and 23553000 B8/B9/B15 at S0–S14. Every bend whose child lies in a parent's hole at S0 is affected;
  the CAD scan `diag/cad_child_in_hole.txt` lists the four.
- **Fix.** `in_footprint`: a panel's footprint is its outline grown by the margin, minus the core of every hole
  (the hole shrunk by the same margin). It is used for the datum's own voxels and for every child's owned voxels.
  Small round holes have no core at the 3 mm child margin.

### 2. A short child's line stays on the short child (`articulated.py`)

- **Cause.** Take a child that carries a bend parallel to its own within the fine reach, such as a joggle's riser
  or a hook's web.
  - **First bend flat, next one formed.** The next bend's tab line lay inside the first bend's ±2.1 mm window at
    35°, offset 1.48 mm. The first bend read it as itself, formed and displaced, and the next bend then read flat.
    On 10839000 S1 this gave B1 NOT_YET_FORMED and B2 UNEXPECTED_FORMED, with a false −2.85 mm position FAIL.
  - **Both formed.** The riser is only 2.6 mm straight. The best line in the full window drifted toward the
    material beyond the riser, and the riser's slab and station fits then disagreed with the region planes. On
    10839000 S9–S12 every characteristic of both joggles was UNKNOWN(AMBIGUOUS_SEGMENTATION): 8–16 per stage, 16
    at S12, the FULL-part stage.
  - **Both formed, with springback.** The riser's own line has little support. A shallow line, about 11°,
    crosses the sheet beyond the next bend over several millimetres and outscored it. With the window bound alone,
    this became a false FAIL on the springback probe of 10839000 at S7: B3/B4 PARTIALLY_FORMED, B3 angle FAIL
    +15.3° (truth +0.5°, U 4.6°). The baseline engine and the dropped variant read that probe right. In the variant,
    B4's line was a different spurious one (10.2°, on the other side of the window), and the growth happened to
    recover the riser from it (`diag/probe_10839000_S7_springback_B4_*`).
- **Fix.**
  - `_grand_offset` gives `grand_u`: the parallel bend's axis offset along the parent's flat continuation, by the
    fold kinematics, with the child flat.
  - On the side toward that bend, the line window reaches at most `max(0.4 mm, grand_u/2 · |sin θ|)` (`_hough`,
    and the wide displaced-flange search). A line whose hinge sits nearer the next bend's axis than this bend's is
    the next bend's line, never this one displaced.
  - The other side keeps the full window, so a bend line displaced toward the parent is still measured where it is
    (a test checks −1 mm).
  - The line's support stops where the short child ends (`s_cut`): per angle, at the next bend's tangent lines
    placed by the fold kinematics, plus 1 mm. This is the limit `_short_child_limit` already put on the child's
    growth. It applies only within that bend's axial span, and on the narrow, fine and wide searches alike. On
    10839000's joggles the cut lies 3.6–4.0 mm out; on every other short child in the compiled A1 recipes it lies
    8.6–35.9 mm out (`diag/cad_short_child_cut.txt`).
- **A variant tried and dropped.** A variant replaced the bound with "revert to flat when the parent's sheet
  continues flat from the hinge"; its source is kept in the work directory as `articulated_conditional_version.py`.
  It fixed S1 but brought back the S9–S12 UNKNOWNs, so the bound stays.

### 3. One visible skin keeps its identity (`articulated.py`)

- **Cause.**
  - **Line search.** A line seen on one skin scores the same read as the inner or the outer skin, one thickness
    apart, and `argmax` took whichever came first in rho.
  - **Region growth.** The growth loop moved the mid-plane to the mean of the medians of the slab's two halves.
    Wall and flare points between the skins pulled one half's median, so the mid-plane walked onto one skin in two
    iterations and the other skin dropped out of the slab (`diag/trace_10839000_S10_B5_growth_baseline.txt`).
  - **Last skin named wrongly.** The remaining skin was then named by which side of the drifted mid-plane it lay
    on.
  - **Result on 10839000 S10.** The lip P5's plane was 1.19 mm off, and the two ADJACENT_BEND lengths on P1 FAILed
    by +1.50 mm with U 0.11 and 0.24.
  - **Same failure on the E2 joggle seen from one side** (TOP_ONLY probe). Iteration 2 left B2
    UNKNOWN(AMBIGUOUS_SEGMENTATION) there.
- **Fix.**
  - `_one_skin_tie`: when neither reading's weaker skin holds more than 5 % of the stronger one's support, and the
    two scores agree within 10 %, the reading nearer the nominal kinematics wins. A real second skin still decides.
  - The growth loop follows the slab's skin height levels (`skin_peaks`) at every step.
  - `_one_skin_mid`: a slab seen on one skin is named by the hinge line pair it grew from. The comparison is made
    at the hinge point, so a tilt of the region-wide fit does not move it. `_refine_skin_levels` uses the same rule.

### 4. A free edge stays on its own panel (`metrology.py`)

- **Cause.** A FREE_EDGE far edge took the farthest point of the side panel's grown region at the station. The
  grown region follows coplanar material over the voxel graph. On the box rim of 10839000 at S12, the FULL-part
  stage, a 12 mm lip's region ran through the touching coplanar joggle tabs into the opposite lip, 300 mm away and
  at the same height. That gave B5/B7 side A +300.4 mm FAIL.
- **Fix.** `_panel_footprint_region`: the region is kept within the panel's fitted outline grown by
  `free_edge_footprint_margin_mm` (5 mm, hole cores excluded). The walk past the region over contiguous material
  is unchanged, so a flange cut long is still measured long (a test checks 8 mm past the outline).

### 5. An unplanned crease ends a flange only where it runs (`metrology.py`, `engine.py`)

- **Cause.** A crease running along a bend ended that bend's FREE_EDGE side at every station: its OML sharp was
  intersected with each station plane as an unbounded line. On 10839000 S11, a spurious 26 mm crease patch ended a
  306 mm tab at all 11 stations: −6.15 mm, U 3.4, FAIL. Fix 2 had made that side measurable.
- **Fix.** The crease tuple now carries its polyline (`engine.py`). `_crease_covers` keeps only creases whose
  extent overlaps the stations, and only at the stations they span. As before, a side needs at least 60 % of its
  stations crossing a crease to end at it. A crease without an extent keeps the old reading.

### 6. A near-symmetric part half made: the order decides which half (`engine.py`)

- **Cause.**
  - E1 certifies a symmetry at 0.3 mm RMS. Three 180° rotations of 10839000 map the part onto itself to
    0.32–0.54 mm, so their poses were not twins: four datum hypotheses scored within 0.001 of each other, and they
    competed on raw total.
  - Only the top 3 hypotheses, plus certified twins, get the articulated fit. At S2 the true pose was the 4th and
    was never articulated, so the image with the other joggle formed won (B1/B2 NOT_YET_FORMED, B3/B4
    UNEXPECTED_FORMED).
  - Articulated, the true pose wins on total by 0.0046 (1.7607 vs 1.7561 with the final engine).
- **Fix.**
  - A rival whose motion matches a ROTATION candidate of any compiled stage within `near_twin_rms_mm` (1.0 mm) is
    a *near twin*.
  - Near twins are articulated like certified twins.
  - Within `ambiguity_margin` of the best total, a near twin also joins the order-prior choice: an order-consistent
    decode wins over an out-of-order one. This is the documented soft prior; it guards the case where the images
    tie.
  - A near twin never makes the pose ambiguous; only E1's certified symmetry can.

### 7. A stepped or countersunk datum hole is not a filled hole (`registration.py`)

- **Cause.** The round-hole fill test counted scan points inside the disc `r − 1 mm` of the hole loop on skin 0.
  On a stepped or countersunk hole the CAD itself has material there: the smaller bore on skin 1. So the nominal
  28870000 at S1, its final stage, read WRONG_PART_SUSPECTED (DATUM_HOLES_FILLED:2), INVALID, with every bend
  UNKNOWN. Both truth-relevant scans of that family are at risk.
- **Fix.**
  - The fill test now uses the per-hole compiled stage material that the kernels and the non-round test already
    use (`DatumModel.hole_blocked`, S_0 and S_N in the datum slab).
  - Scan points within the clearance of that material do not count, and the threshold scales with the open share
    of the disc.
  - A disc that is less than 25 % open is not judged.
  - A through-hole has no such material, so wrong-part detection on it is unchanged.

## Results

### CAD-stage probes, nominal geometry: baseline vs iteration 3 on identical virtual scans

Every compiled stage of the six recipes whose families have truth-relevant scans: 57 probes, 654 bend states.
The truth is the stage's formed set. The geometry is nominal, so every FAIL is false. Cells read baseline
(e3-1.3.0) → **final engine**.

| Family (recipe) | Probes (stages) | Bend states | Probes with a wrong state | Wrong bend states | False FAILs | AMBIGUOUS_SEGMENTATION on formed bends | INVALID |
|---|---|---|---|---|---|---|---|
| 10839000 | 13 | 156 | 2 → **0** | 6 → **0** | 5 → **0** | 48 → **0** | 0 → **0** |
| 23553000 | 17 | 272 | 15 → **0** | 33 → **0** | 0 → **0** | 0 → **0** | 0 → **0** |
| 38172000 | 13 | 156 | 8 → **0** | 28 → **0** | 0 → **0** | 0 → **0** | 0 → **0** |
| 11979003 | 8 | 56 | 0 → **0** | 0 → **0** | 0 → **0** | 0 → **0** | 0 → **0** |
| 28870000 | 2 | 2 | 2 → **1** | 2 → **1** | 0 → **0** | 0 → **0** | 2 → **1** |
| 15651004 | 4 | 12 | 0 → **0** | 0 → **0** | 0 → **0** | 0 → **0** | 0 → **0** |
| **All** | 57 | 654 | 27 → **1** | 69 → **1** | 5 → **0** | 48 → **0** | 2 → **1** |

- The one probe still wrong is 28870000 S0, the blank (open items).
- Over all 57 probes:
  - UNKNOWN characteristics on formed bends: 159 → 94.
  - Measured deviations beyond U: 25 → 21.
  - REVIEW: 14 → 31, all on backgauge lengths with a large U, of bends now measured.
  - E3 time: 871 → 932 s in all (+7 %).
- Per probe: `diag/sweep_nominal_base_vs_final.txt`; json in `raw/`.

### CAD-stage probes with springback (±2° formed, ±1° kinks), truth-relevant families

49 probes, final engine. Every panel is re-posed by E3's fold kinematics: each formed bend off by up to ±2°, each
unformed one kinked by up to ±1°. The angle truth is the applied offset. The position truth is the intersection of
the perturbed parent and child OML planes, measured as E3 defines it. No flange-length truth is computed.

| Family | Probes | Bend states | Probes with a wrong state | Wrong bend states | Angles within U | False angle FAIL / PASS | Positions within U | Position FAILs (all true) | False position FAIL / PASS |
|---|---|---|---|---|---|---|---|---|---|
| 10839000 | 13 | 156 | 0 | 0 | 78/78 | 0 / 0 | 71/78 | 16 | 0 / 0 |
| 23553000 | 9 | 144 | 0 | 0 | 72/72 | 0 / 0 | 72/72 | 25 | 0 / 0 |
| 38172000 | 13 | 156 | 0 | 0 | 77/78 | 0 / 0 | 78/78 | 2 | 0 / 0 |
| 11979003 | 8 | 56 | 1 | 1 | 26/28 | 0 / **1** | 26/28 | 8 | 0 / 0 |
| 28870000 | 2 | 2 | 0 | 0 | 1/1 | 0 / 0 | 0/1 | 0 | 0 / 0 |
| 15651004 | 4 | 12 | 1 | 3 | 5/5 | 0 / 0 | 5/5 | 0 | 0 / 0 |
| **All** | 49 | 526 | 2 | 4 | 259/262 | 0 / **1** | 252/262 | 51 | 0 / 0 |

- **The remaining misreads are the baseline engine's, unchanged.** On the three probes involved, e3-1.3.0 gives
  identical results to the 6th decimal:
  - 11979003 S4, B4: a false angle PASS. Measured −0.73°, truth −1.06° (0.06° past the ±1° tolerance), error
    0.32° against U 0.26°. B6 at the same stage reads UNKNOWN where it is not yet formed.
  - 11979003 S5, B5: angle error −0.26° against U 0.22°; the decision is right.
  - 15651004 S1: MIRRORED_PART_SUSPECTED, so no bend is evaluated.
- **What iteration 3 changed here.** The window bound alone made two 10839000 probes wrong:
  - S7: B3/B4 PARTIALLY_FORMED and a false angle FAIL (fix 2, cause).
  - S5: B1/B2 NOT_YET_FORMED, B3–B5 UNEXPECTED_FORMED. The dropped variant also got S5 wrong.
  - With the support cut, both decode right. On the 34 probes both engines ran (10839000, 38172000 and 8 of
    23553000), bound-only → final: wrong bend states 7 → 0, false angle FAILs 1 → 0, angles within U 207/209 →
    211/212. Only S5 and S7 differ.
- Per probe: `diag/sweep_springback_final.txt`.

### E2 synthetic joggle probes: iteration 2's five cases, exact E2 truth

Profile 80 / 21.4 / 3.6 / 13 mm, t 1.5, sweeps 90/35/35. Output: `diag/joggle_probes_iter3.txt`.

| Case | iteration 2 (e3-1.3.0) | iteration 3 (e3-1.4.0) |
|---|---|---|
| Joggle UP/DOWN, FULL, nominal | 3/3 FORMED, REVIEW_REQUIRED (B3 angle +0.29°, U 0.80) | 3/3 FORMED, **ALL_MEASURED_IN_TOLERANCE** (B3 angle +0.20°) |
| Joggle, FULL, injected errors | 3/3 FORMED, every error inside U | unchanged |
| Same direction 35 + 35, FULL | 3/3 FORMED, ALL_MEASURED_IN_TOLERANCE | unchanged |
| Joggle, TOP_ONLY (one skin) | 3/3 FORMED. B2 UNKNOWN(AMBIGUOUS_SEGMENTATION). B1/B3 side A ADJACENT_BEND_NOT_MEASURED. B3 side B −0.69 mm, angle +1.62° | 3/3 FORMED, **every characteristic measured**. All deviations at most 0.22 mm / 0.22° from the zero truth and inside U; no FAIL. |
| Joggle at a mid stage, second bend shifted 1.47 mm | B1, B2 NOT_YET_FORMED, no FAIL (open) | unchanged (still open) |

### Physical: the engine vs the best F1 lane on the same scans

These use the frozen scorer: matcher 5° / 4 mm / 0.6, apex locus, raw frame. The 13 truth-relevant A1 cases are
still refused by the amendment-7 gate (a process block) and count as misses. These are the final-code runs.

| Population | iteration 2 | **iteration 3** | Best F1 lane |
|---|---|---|---|
| BROAD, all 26 A1 cases, family-macro / micro | 0.667 / 0.607 | **0.667 / 0.607** (TP 68, FP 12, FN 76; P 0.85) | 0.543 / 0.508 |
| BROAD, the 13 scans E3 runs, family-macro / micro | 0.896 / 0.861 | **0.896 / 0.861** (P 0.85, R 0.87) | 0.649 / 0.620 |
| G2(a) Gold11 relabel (gold of record), 4 cases E3 runs / all 9 | 1.000 / 0.574 | **1.000 / 0.574** (P 1.00) | 0.561 / 0.547 |
| G2(a) human Gold11 (historical), 4 / 9 | 0.724 / 0.467 | 0.724 / 0.467 | 0.717 / 0.473 |
| GOLD_SET_V1, 13 A1 scans with STEP that E3 ran, micro / family-macro | 1.000 / 1.000 | **1.000 / 1.000** (TP 80, FP 0, FN 0) | 0.597 / 0.653 |

The runnable scans were already at the reachable ceiling; this iteration targets the 13 refused ones. The legacy
arm is informational only and is not a bar.

**Per bend given CAD** (report-only proxy, 80 recipe bends on the complete scans E3 ran):

| Measure | iteration 2 | iteration 3 |
|---|---|---|
| Correct vs BROAD gold | 63/80 | 63/80 |
| Correct vs any gold of record | 75/80 | 75/80 |
| Strict false-PASS proxy | 0 | 0 |
| PASS on a bend without a published line | 0 | 0 |

The 5 bends still not correct against any gold are the compile-side unavailable flange nominals: 50757000 B1/B4
side B on both scans, and 47266000 B1.

**Per characteristic, against iteration 2's measured run** (all 13 scans, 314 measured characteristics):
- No state changed, and no FAIL appeared or went.
- One decision changed: 11979003 B7 side B, PASS → REVIEW (+0.355 → +0.386 mm, U 0.117, tolerance ±0.5). The
  support cut (fix 2) moved it: 11979003 has three short-child bends (B3–B5), and the bound-only engine kept B7
  side B at PASS.
- On 15651004 the summary's "worst" pointer moved from B2's angle to B3's angle (REVIEW, +1.00°, moved less than
  0.05°), because B2's exceedance dropped below it.
- 307 characteristics moved by at most 0.05 mm / 0.05°. The 7 that moved more are listed below. They come from the
  changed child segmentation (fixes 1 and 3):

| Scan | Characteristic | iteration 2 | iteration 3 | Note |
|---|---|---|---|---|
| 15651004 | B1 side A | −0.289 mm | −0.135 mm | toward nominal |
| 15651004 | B2 angle | +0.64° | +0.57° | |
| 38172000 V | B5 angle | +1.92° | +1.85° | stays FAIL |
| 38172000 X | B5 angle | +4.82° | +4.76° | stays FAIL |
| 38172000 X | B5 side A | +0.46 mm | +0.40 mm | |
| 38172000 X | B5 side B | −9.1 mm | −8.3 mm | the REVIEW side with U 31 mm |
| 11979003 | B7 angle | −0.44° | −0.32° | |

### Gates

| Gate | Result |
|---|---|
| **G1, synthetic `regression` DEV tier** (50 cases, 7 families, scanner model card, final code) | `regression_verdict` **PASS**, 0 failed accuracy gates. Identical to iteration 2 gate for gate, except flange bias 0.009379 → 0.009369 mm, flange P95 0.103252 → 0.103377 mm, the flange slope in the 6th decimal, and runtime noise (P95 11.31 s). Values: angle bias 0.0015° and P95 0.019°, position P95 0.073 mm, coverage 0.997, false PASS 0/553, false FAIL 0, U calibration 0.984, median U 0.106 mm / 0.116°, decisiveness 0.983, stage state 1.0, 0 engine failures. Non-qualifying by construction (TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL; G1.SUITE fails below 200 cases). |
| **G2** | FAIL, `final: false` (no truth table). Sub-gates in the table below. |
| **G3** | NOT_EVALUABLE: the frozen truth table does not exist. |
| **G5** | Not re-run here. No `backend/`, `domains/bend/` or `scripts/` file was touched. The known G5 failure (`tests/test_vendor_unfold_api_contract.py`, another workflow's edit) is unrelated. |

| G2 sub-gate | Result |
|---|---|
| (a) line metric | FAIL: refused cases count as misses |
| (b) coverage | FAIL only on refused and blocked cases: angle 55/91, flange 108/178. 100 % on every runnable case. |
| (c) known-good angle | FAIL on coverage only (14/32 vs legacy 16/32). MAE 0.422° (was 0.437°); on the 8 common bends 0.503° vs legacy 1.360°. |
| (d) known-good flange | PASS: P95 0.537 mm, max 0.901 mm (iteration 2: 0.522 / 0.860). The max is 11979003 B1 side B, +0.86 → +0.90 mm, moved by the support cut (fix 2). |
| (e) runtime | P95 20.0 s, max 20.6 s, 0 crashes; `all_cases_run` FAIL (refused cases) |
| (f) repeatability | NOT_EVALUABLE |
| (4d) compile rate | PASS (24/26) |

The ledger check is OK, with anchor `cf01e709…` checked. The compile code fingerprint equals the freeze, and
eval and contracts are unchanged.

## Generality

- No SKU, family, part-id, file-name or metadata-count logic was added, and E3's inputs are unchanged.
- Every rule is geometric. Fix by fix:

  | Fix | What the rule uses |
  |---|---|
  | 1 | Hole cores of a panel's own outline |
  | 2 | The parallel bend's axis offset and tangent lines from the fold kinematics |
  | 3 | The two readings of one skin and their distance to the nominal kinematics; skin height levels; the hinge line pair |
  | 4 | A panel's fitted outline |
  | 5 | A crease's own polyline |
  | 6 | E1's own symmetry candidates and the planned order |
  | 7 | Compiled stage material inside a hole |

- The failing parts were found by a sweep over every compiled A1 recipe and every compiled stage. The fixes were
  then checked over the same population. Each rule is unit-tested on synthetic sections or stubs that name no part.
- New thresholds are engine config fields with ranges: `free_edge_footprint_margin_mm` 5, `one_skin_fraction`
  0.05, `one_skin_tie_fraction` 0.1, `near_twin_rms_mm` 1.0. There is also one module constant,
  `_HOLE_FILL_MIN_OPEN_SHARE` 0.25. The frozen gate thresholds are untouched.

## Risks

1. **Fix 2 (window bound and support cut) can misread a real defect, and it depends on the frame.** It applies
   only to children carrying a parallel bend within the fine reach.
   - A bend formed with its line displaced more than half the axis spacing toward the next bend is read as that
     bend's line, so this bend reads flat or partial. That is a state error, never a FAIL. On a 3.6 mm joggle riser
     the threshold is about 1.9 mm of hinge shift, so such a part is out of tolerance either way.
   - A frame error of about that size toward the next bend could likewise cut off a true formed riser. The one probe
     with a large frame error (0.26°, see the open items) decodes right with the support cut. It failed with the
     bound alone and with the dropped variant.
   - The support cut trusts the next bend's nominal tangent lines. A riser left longer than nominal (the next bend's
     line displaced away from it) is read from the length before the cut. A riser left shorter lets the start of
     the next bend's arc into its support, as before the cut. A flat riser needs material up to the end of the
     flat-continuity span (2.9 mm out on 10839000); the nearest cut at flat in the compiled A1 recipes is 4.0 mm.
2. **The one-skin tie (fix 3) prefers the nominal reading.** A one-sided flange whose line is off by more than half
   a thickness along its normal is read one thickness off. Before, the reading was arbitrary. With a second skin
   seen, the two-skin rule decides as before.
3. **The skin-level growth (fix 3) applies to every child, and the support cut (fix 2) to every short child.** On
   the 13 real runnable scans one decision changed (11979003 B7 side B, PASS → REVIEW by +0.03 mm, from the cut),
   and 307 of 314 values moved by at most 0.05 mm / 0.05°. The largest moves:
   - 15651004 B1 side A, +0.15 mm, toward nominal;
   - 11979003 B7 angle, +0.13°;
   - 38172000 X B5 side B, +0.8 mm, a REVIEW with U 31 mm.
   A slab whose two skins are both weaker than the clutter between them is now named by the hinge lines. On the
   known-good 11979003 scan the cut moved B1 side B from +0.86 to +0.90 mm, the G2(d) maximum.
4. **The FREE_EDGE footprint (fix 4)** cuts a grown region 5 mm past the fitted outline. Suppose a flange's material
   beyond that point is interrupted, for example by a slot crossing the station near its far edge. The side would
   then end at the interruption, where the unrestricted region could have bridged it.
5. **Crease extent (fix 5).** A real crease that the detector finds over only part of its length now ends the side
   only at the stations it spans. If fewer than 60 % of the stations are covered, the side is measured to its free
   edge, and the far-edge truncation rule drops the stations that cross the crease.
6. **Near twins (fix 6) let the order prior choose between poses that E1 does not call symmetric.** A part really
   formed out of order on its near-symmetric half would be decoded as the in-order image: wrong bend identity, no
   FAIL. A near twin is never declared ambiguous, and a pose more than 0.03 better on total always wins. On the
   probes, articulating the near twins was enough; the prior was not needed.
7. **The hole fill (fix 7)** does not judge the part of a hole's disc where the CAD itself has material. Suppose a
   wrong part differs only in a stepped hole. The fill check would pass it; explained share and the mirror check
   still apply.
8. **The probes are E1's own stage geometry,** seen through the E2 scanner model (4b card, FULL plan). They test
   topology, stage decode and scanner effects, not real defects. The springback sweep adds ±2° per formed bend and
   ±1° kinks; neither is physical truth. The real truth-relevant scans remain unseen until the G3 table is frozen.

## Tests

| Test | Result |
|---|---|
| `tests/test_prod_engine_inspect_a1_iter3.py` (new; pure numpy, no part named) | 12 passed |
| The final engine: `test_prod_engine_inspect_cli.py` (E3's 32 E2E tests, E2 → E1 → E3), `_units.py`, `_a1.py`, `_a1_iter2.py`, `_a1_iter3.py` | **112 passed** (175 s, under the lock) |
| Runs | G1 regression tier, physical runs, probes, sweeps and debugging all ran one at a time under `/private/tmp/claude-501/sherman_geom.lock` with `OMP_NUM_THREADS=1`. Peak RSS: 0.91 GiB (physical run), 0.82 GiB (G1 runner). |

What the new tests cover:
- fix 1, a window's core;
- fix 4, a lip region running 300 mm into a coplanar lip, and a flange cut 8 mm long still followed;
- fix 2, a formed tab beyond a flat riser, a whole joggle, the bound applied only on the side toward the next
  bend (a flange displaced toward the parent is measured where it is), and the support cut (a shallow line through
  the sheet beyond a short riser no longer outscores the riser; no cut beside the next bend's span);
- fix 3, one visible skin read nearest the nominal lines, a weak second skin, and a one-skin slab named by its hinge
  lines;
- fix 6, the near-twin order prior: chosen, never ambiguous, a clearly worse twin not chosen, an unrelated rotation
  ignored;
- fix 7, a stepped hole, a filled through-hole, and a sector of stage material;
- fix 5, a short crease;
- the config fields and their ranges.

## Open items and candidates for iteration 4

- **Process is still the biggest lever.** Freeze the G3 visual truth table (amendment 7) so the 13 truth-relevant
  cases run. All seven fixes target their geometry, but none has been seen on their real scans.
- **The in-plane rotation of a datum bounded by bends on every side.** 10839000's datum (the box's back face) has one
  hole and eight free edges of 0.9 mm each, at its corners (`diag/cad_datum_free_edges.txt`). While the walls are
  still flat, four of those corner edges have a flat neighbour's material 0.4–0.6 mm beyond them. Their edge rows
  can then read the neighbour's edge, and the rotation rests on few short rows.
  - In one springback probe at S5 the chosen pose is 0.26° / 0.66 mm off; its datum score is 0.75.
  - Before the support cut, the 306 mm joggle then smeared about 1.4 mm in its cross-section and read flat: B1/B2
    NOT_YET_FORMED, B3–B5 UNEXPECTED_FORMED, no FAIL.
  - With the cut it decodes right, but narrowly. B2's line is read at 14.6°, and the growth recovers the riser
    from it. The true pose beats the 180° image by 0.002 in total, and the order prior agrees.
  - The nominal probes at the same stage register within 0.08° / 0.3 mm.
  - Candidate: give each datum free edge a band reach that stops short of compiled stage material beyond it, as the
    non-round hole runs already do.
- **28870000 at S0** (its single bend flat): no datum hypothesis is near the true pose.
  - The datum is a 74 × 11 mm strip; at S0 its coplanar neighbour and rolled feature dominate the plane segment.
  - Result: WRONG_PART_SUSPECTED (explained 0.20) on the nominal stage.
  - S1, the formed stage, is now VALID and correct.
- **Unplanned-crease false positives on nominal CAD scans** (10839000 S11/S12: 72–132 mm² patches at the joggle and
  corner regions). Fix 5 keeps them from ending a flange, but they still put REVIEW_REQUIRED on a nominal part.
- **Iteration 2's open item:** a joggle whose riser is shortened by a bend-line defect still reads not yet formed.
- **A short riser's hinge line is still weak.** With the support cut, the springback probes decode right, but the
  line that seeds the growth can sit well off the riser's angle (B4 at S7: 27.2° for 36.4°; B2 at S5: 14.6°). A
  riser line fitted from the grown riser itself, and not only from the hinge search, would make the reading robust.
- **Pre-existing springback misreads** (identical on e3-1.3.0): 11979003 S4, a marginal false angle PASS (U 0.26°
  against an error of 0.32°) and B6 UNKNOWN; 15651004 S1, MIRRORED_PART_SUSPECTED on a springback-perturbed part.

## Files

| Path | Contents |
|---|---|
| `engine_before.tgz`, `engine.diff` | Snapshot and diff |
| `raw/` | Final G1 regression scorecard (json, md); frozen physical scorecard, subset / gold-set / per-bend analyses; sweep comparison tables (json) |
| `diag/` | Cross-sections: 10839000 S1 B2 (the half-made joggle, baseline and final engine, with the final hinge trace `probe_10839000_S1_B2_hinge_iter3.txt`); 10839000 springback S7 B4 (bound only, and final, with the angle profiles and the dropped variant's reading); S10 B5 (the one-sided lip); 38172000 S0 B6 (the tab in its window). The S10 growth trace. CAD-only scans (`cad_child_in_hole.txt`, `cad_near_symmetry.txt`, `cad_datum_free_edges.txt`, `cad_short_child_cut.txt`). Joggle probes (`joggle_probes_iter3.txt`). Sweep tables (`sweep_nominal_base_vs_final.txt`, `sweep_springback_final.txt`). |
| `tools/` | Probe, sweep, comparison, debug, patch and run scripts. The work directory is `…/scratchpad/a1_iter3`, outside the repository: base, bound-only, dropped-variant, prototype and final engine trees, probes, compiled parts, physical runs, G1. |

**Custody.**
- Only `cases_A1.json` cases were run, read or scored: 13 runnable, plus the compile-blocked one.
- The CAD-stage probes read compiled A1 recipes only (E1 stage meshes and samples). No physical scan of any
  truth-relevant case was read, rendered or run, so no engine output exists for them.
- GOLD_SET_V1 rows were scored for A1 scans only.
- Nothing `*SEALED*`, FRESH, HOLDOUT10, N001–N016 or EHO4 was opened.
  - One slip: a directory-listing glob over `g3_truth_material/*` printed the entry names of
    `g3_truth_material/devcheck_SEALED/` (case-id directory names only). No file there was opened, and the names
    were not used.
- No credentials were touched.
- Process slip: stopping the superseded sweeps mid-probe left the shared `sherman_geom.lock` directory behind for
  about a minute. I released it after checking that no process held it and no heavy job was running.

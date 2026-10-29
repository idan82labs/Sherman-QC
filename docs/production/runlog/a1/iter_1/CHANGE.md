# A1 iteration 1: engine change record (2026-10-18)

All numbers below come from consumed development scans (engineering ledger). None of them is
unseen-family accuracy.

## Provenance

| Item | Value |
|---|---|
| Snapshot before editing | `engine_before.tgz`, sha256 `c0210de0c0883ddbcd8539ee7793d7d1e6946cbd7b9bc7f054725c616f61640b` (taken before the first edit) |
| Diff against the snapshot | `engine.diff`: five files under `domains/production/engine/inspect/` (registration, engine, metrology, articulated, config). No change to `eval/` or `compile/`. |
| Engine | `e3-1.2.0` (was `e3-1.1.0`), default config `74c0305091bf0fb900ce1e137603ec1b3176361ca61f6d94a29c72fed044dcc7` |
| Algorithm ids bumped | `registration` `..._polygon_v2`; `articulated_fit` `..._displaced_v2`; `metrology` `..._station_profile_v3` |
| Code fingerprints (G1 run) | engine `4e63a064…`, compile `d4e870e8…` (unchanged), eval `8aa6860c…` (unchanged, frozen) |
| Ledger | check OK, anchor `cf01e709…` checked |
| Previous generality check | null, so there was nothing to remove. Iteration 0 was the baseline, so there was nothing to revert. |

## What changed, and why

### 1. Non-round datum holes in the registration (`registration.py`)

**Diagnosis.** Five of the 13 runnable BROAD scans were INVALID with `WRONG_PART_SUSPECTED (DATUM_HOLES_FILLED)`:
38172000 X and V, 37361005, 49024000 and 34499000. Together they cost 30 BROAD gold lines and every G2(a) case
that ran. The cause was one mistake in two places: holes were treated as circles about their vertex
centroid, and "circular" was tested only by the scatter of the vertex radii.

- The fill test counted scan points inside a disc of radius (mean radius − 1 mm). On a 50 × 80 mm
  rectangular window (four equidistant corners, so it was taken as a circle of r = 47 mm), a rounded
  trapezoid window (r = 45 mm) and an obround slot, that disc covers solid material. The scans were
  perfectly registered. See `diag/pc_d1732de18023_hole2.png`, `diag/pc_828bc34d9c8b_hole0.png` and
  `diag/pc_70b39f530f00_hole2.png`.
- The same centroid ring produced the in-plane hole rows, so the rows of a non-round hole were garbage.
  On 38172000, whose datum's only free edges are two 1.5 mm notches, this pulled the pose about 3 mm and
  1.5° off (datum score 0.10). See `diag/pc_2714fb7e0c0b_datum.png`.

**Fix.**
- **Round test.** A hole is round only if it has at least 8 vertices, its radius scatter is below 8 %,
  and its area is within 12 % of πr². Round holes keep exactly the old radial rows and fill test. Every
  one of the 207 holes on the synthetic DEV tier is a round 72-gon, so G1 is unaffected.
- **Straight runs as cut edges.** Every other hole contributes its straight boundary runs as cut edges
  in the existing edge machinery. The normal points into the hole, and the band stops short of the
  hole's opposite side.
- **Stage independence (amendment 9).** Compiled stage material (S_0 and S_N in the datum slab) inside a
  hole blocks that part of it. For example, a tab lies flat in its window at S_0. Blocked areas are
  dropped from the hole runs, the fill test and the forbidden hole cores of the coarse and fine kernels.
- **Fill test for non-round holes.** The hole counts as filled when the scan covers at least 30 % of its
  open core (cells inset one cell from the boundary) and at least half of the coverage of the 4 mm
  material ring around it. The hole is not judged when the ring is unseen or the core is a narrow slot.

**Effect.**
- All five scans are now VALID.
- 38172000 X and V register with datum scores 0.87 and 0.88, against 0.10 and 0.44 before.
- The four Gold11 cases among them now score 31 of 31 lines on the relabel, with 0 FP.

### 2. Per-direction trust of bend-line position (`engine.py`, `registration.py`)

**Diagnosis.** On 49204020 and 15651004, `REGISTRATION_UNTRUSTED` withheld every bend-line position, and
with it every line: 15 gold lines. On 49204020 the datum is bounded by bends on both of its ±y sides and
has no holes, so only ty is free. The rotation is fixed to 4e-5 rad.

**Fix.**
- The fit now keeps its undetermined eigen-directions: those with information below
  `degenerate_min_information`, which the solver never updates.
- A bend's position, measured along side A's outward direction, is MEASURED when that direction's
  sensitivity to every undetermined direction is at most 0.05 mm per mm and the projected sigma of the
  determined directions is at most `degenerate_max_sigma_mm`. Its U uses that projected sigma plus the
  usual pose-bias bound.
- Every other position stays UNKNOWN(REGISTRATION_UNTRUSTED), and the heatmap is still withheld.

**Contract semantics, to note for R1.** `registration.in_plane_degenerate` is now set only when no
position at all can be determined, because the frozen contract forbids any MEASURED position under that
flag. A partially determined pose leaves the flag false, withholds each affected position individually
(UNKNOWN, REGISTRATION_UNTRUSTED), withholds the heatmap, and adds the warning `IN_PLANE_POSE_PARTIAL`.

**Effect.**
- 49204020: 8 of 12 positions and lines are recovered. The 4 left (B3, B4, B6 and B12) are measured
  along the free ty, so their absolute position is genuinely undefined for this datum.
- 15651004: 1 of 3 is recovered. B1 and B2 depend on a weak rotation and ty mix with 2.03 effective
  observations, below the threshold of 3, which was left unchanged.

### 3. Flange length against the nominal station profile (`metrology.py`)

**Diagnosis.** Twelve of the 145 measured flange sides deviated by more than 0.5 mm. All of them were far
boundaries that are stepped, notched or tapered, or adjacent bends that are not parallel.

- E1's nominal is one number for the whole corridor: a dominant level, a taper median, or a median
  crossing. The scan's visible stations, each taken as a far-edge statistic over the station width,
  differ from it even on a good part.
- The results were a false FAIL on 15651004 B3 side B (+1.45 mm, U 0.12; the stations sit on a 1.5 mm tab
  that E1's majority ignores), REVIEWs with U of 5 to 22 mm, and the G2(d) known-good P95 of 1.22 mm.

**Fix.** `SideProfile` evaluates E1's own definition at any axial station position: the farthest crossing
of the side panel's boundary loops, or the crossing of the adjacent bend's OML sharp.
- Where the CAD far edge varies across a station's width, the station's points are flattened by the CAD
  offset before the existing far-edge statistic. The edge is then taken at the station centre, so the
  wall, rank and truncation logic are unchanged.
- The side's value is the compiled nominal plus the robust location of the per-station deviations.
- It is applied only when the compiled nominal lies inside the profile's range and the profile actually
  varies by more than 0.05 mm. A straight far edge parallel to the bend keeps the old statistic
  bit-for-bit.

**Effect, on the 7 scans that had such sides** (before fix 4):
- |dev| above 0.5 mm: 12 sides → 3.
- FAIL 1 → 0, PASS 65 → 74.

**Final run, all 13 scans.** Of 154 measured flange sides, 10 deviate by more than 0.5 mm:
- seven sides of the 50757000 sc_752e defect chain found by fix 4 (P6 +4.9 mm, P5 −2.0 mm, four tabs
  about −3 mm; the sum stays near zero, as for bend lines placed off position on a correct blank);
- 38172000 V, B5 side B (REVIEW, U 30 mm);
- two 11979003 sides (REVIEW).

**Synthetic.** Six PN-family flanges move by at most 0.004 mm. Flange bias goes from 0.00933 to
0.00938 mm.

### 4. A formed flange displaced beyond the hinge slack (`articulated.py`, `config.py`)

**Diagnosis.** On 50757000 sc_752e, B1 decoded as PARTIALLY_FORMED at 16.7°, and the whole chain below it
(B4 to B8) was UNKNOWN. The Hough searches the inner-skin line only within ±3 mm
(`hinge_rho_slack_mm`) of nominal. The real flange stands about 5 mm away, which is visible in the
section: a flange 5 mm long. Clutter from the parent's continuation then won at 16.7°.

**Fix.** For bends of at least 45° that the narrow search leaves flat or partial:
- The formed-like angles (half the sweep up to the overbend limit) are searched again with a ±10 mm
  inner-skin window over the fine reach.
- The displaced line replaces the narrow winner only if its two-skin support is at least 1.5 × the
  narrow winner's.
- It is skipped if the displaced line lies inside the narrow window.
- New config fields: `hinge_rho_slack_wide_mm` 10, `displaced_support_ratio` 1.5,
  `displaced_min_sweep_deg` 45.

This removes a distance-to-nominal gate that plan §3.2 forbids: defective flanges must be measured, not
dropped.

**Effect.**
- sc_752e decodes 9 of 9 FORMED. B1's position shows a −4.9 mm FAIL and the P6 flange +4.8 mm FAIL: the
  real defect, shown on the close-up `overlays/16_*`.
- BROAD goes from 4 of 9 to 9 of 9.
- No synthetic case changed.

## Results: engine vs the best F1 lane on the same scans

13 A1 BROAD cases are still refused by the amendment-7 gate, which is a process block: the G3 visual truth
table is not frozen. The frozen scorer counts them as misses.

| Population (frozen matcher 5° / 4 mm / 0.6, apex locus, raw frame) | iter 0 | **iter 1** | Best F1 lane |
|---|---|---|---|
| BROAD, all 26 A1 cases, family-macro / micro | 0.319 / 0.282 | **0.626 / 0.569** (P 0.84) | 0.543 / 0.508 |
| BROAD, the 13 scans E3 runs (11 families), family-macro / micro | 0.406 / 0.462 | **0.832 / 0.816** (TP 62, FP 12, FN 16) | 0.649 / 0.620 |
| G2(a) Gold11 relabel (gold of record), the 4 cases E3 runs | 0.000 | **1.000** (31 / 31, FP 0) | 0.561 |
| G2(a) Gold11 relabel, all 9 A1 cases | 0.000 | **0.574** (P 1.00) | 0.547 |
| G2(a) human Gold11 (historical), 4 cases / all 9 | 0.000 / 0.000 | 0.724 / 0.467 | 0.717 / 0.473 |
| GOLD_SET_V1, 13 A1 scans with STEP that E3 ran, micro / family-macro | 0.491 / 0.440 | **0.961 / 0.936** (TP 74, FP 0, FN 6) | 0.597 / 0.653 |

The legacy arm scores 0.088 on G2(a). It is informational only and is not a bar.

### Per case: BROAD gold TP / FP

| Case | Family | Gold lines | iter 0 | iter 1 | Best lane |
|---|---|---|---|---|---|
| pc_03395097cd96 | 15651004 | 3 | 0/0 | 1/0 | 3/0 |
| pc_2714fb7e0c0b | 38172000 X | 10 | 0/0 | 8/4 | 7/2 |
| pc_361fc1049e39 | 44211000 | 1 | 1/0 | 1/0 | 1/0 |
| pc_4f8188b4d1a8 | 38172000 V | 10 | 0/0 | 8/4 | 8/3 |
| pc_70b39f530f00 | 49024000 | 3 | 0/0 | 3/0 | 2/1 |
| pc_752e2bd54aad | 50757000 | 9 | 4/0 | 9/0 | 4/4 |
| pc_777c64c88967 | 50757000 | 9 | 9/0 | 9/0 | 4/2 |
| pc_828bc34d9c8b | 34499000 | 3 | 0/0 | 3/0 | 3/0 |
| pc_9b9fd762f361 | 47266000 | 1 | 1/0 | 1/0 | 0/1 |
| pc_d1732de18023 | 37361005 | 4 | 0/0 | 2/2 | 2/1 |
| pc_d45186653eed | 11979003 | 9 | 5/2 | 5/2 | 1/6 |
| pc_e080ae094f35 | 49204020 | 12 | 0/0 | 8/0 | 6/0 |
| pc_ff1667a2a80a | 48811000 | 4 | 4/0 | 4/0 | 3/0 |

**All 12 remaining FPs are gold-convention differences.** On the Gold11 relabel the same 38172000 X/V and
37361005 scans are 12/12, 12/12 and 4/4 with 0 FP. The BROAD gold of 38172000 is the human Gold11 set
with 10 lines. 11979003 has the documented window-split conflict.

**The 16 FNs:**

| Scans | Lines | Cause |
|---|---|---|
| 11979003 | 4 | 9 gold lines but 7 recipe bends (convention) |
| 49204020 | 4 | positions along its free ty |
| 15651004 | 2 | weak rotation information |
| 38172000 X/V and 37361005 | 6 | matched on the relabel |

### Per bend given CAD

Report-only proxy over 80 recipe bends on the complete BROAD scans E3 ran:

| Measure | iter 0 | iter 1 |
|---|---|---|
| Correct | 20 / 80 = 0.25 | **57 / 80 = 0.713** |
| FORMED | 0.51 | 1.00 |
| Angle measured | 0.51 | 1.00 |
| Line matches BROAD gold | 0.30 | 0.775 |

The false-PASS proxy is 17, the same total as iter 0 but a different split:
- 11 are PASS decisions on bends whose line misses the BROAD gold. These are the convention cases above;
  all of them match the relabel.
- 6 are PASS on bends whose position is withheld (49204020 ×4, 15651004 ×2). Their angle and flanges are
  measured.

No physical dimensional truth exists, so this proxy is no measure of real false PASS.

## Gates

| Gate | Result |
|---|---|
| **G1, synthetic `regression` DEV tier** (50 cases, 7 families, scanner model card) | `regression_verdict` **PASS**, gate-for-gate identical to iter 0 except:<br>- flange bias 0.009329 → 0.009377 mm<br>- runtime noise<br>Values: flange P95 0.103 mm, angle P95 0.019°, position P95 0.073 mm, coverage 0.997, false PASS 0 of 553, false FAIL 0, U calibration 0.984, decisiveness 0.983, stage state 1.0, 0 engine failures. `G1.SUITE` FAILs by construction (fewer than 200 cases); the run is non-qualifying (TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL). |
| **G2** | FAIL, `final: false` (no truth table). Per sub-gate below. |
| **G3** | NOT_EVALUABLE: no frozen truth table. |
| **G5** | Not re-run here. The engine change touches no `backend/`, `domains/bend/` or `scripts/` file. The known G5 failure is `tests/test_vendor_unfold_api_contract.py` after another workflow's `vendor_setup_service.py` edit, and is unrelated. |

G2 sub-gates:

| Sub-gate | Result | Note |
|---|---|---|
| (a) line metric | FAIL | Refused cases count as misses. |
| (b) coverage | angle 55/91 = 0.604, flange 108/178 = 0.607 | **100 % on every runnable G2 case.** The gap is the 5 refused cases and 1 compile-BLOCKED case. |
| (c) known-good angle | MAE 0.435°; on 8 common bends 0.503° vs legacy 1.360° | FAIL only on coverage, 14/32 vs legacy 16/32. Three of the four known-good parts are refused. |
| **(d) known-good flange** | **PASS**: P95 0.518 mm, max 0.847 mm, no deviation above 1 mm | Was FAIL at 1.22 mm. |
| (e) runtime | P95 19.6 s, 0 crashes | `all_cases_run` FAIL: refused cases. |
| (f) repeatability | NOT_EVALUABLE | Two of three 15651004 scans are refused. |
| (4d) compile rate | PASS | |

**The single biggest remaining lever is process, not engine:** freeze and record the G3 visual truth table
(amendment 7) so that the 13 truth-relevant A1 cases can run.

## Tests

| Test | Result |
|---|---|
| `tests/test_prod_engine_inspect_a1.py` (new, pure numpy): round-hole test, straight runs, hole runs with a stage-0 tab, the fill test (window, slot, unseen ring), window rows, per-direction pose trust, `SideProfile` (step, window, taper), flattening on a slanted edge, the displaced-flange search | 10 passed |
| `test_prod_engine_inspect_units.py` | 45 passed |
| `test_prod_engine_inspect_cli.py` (E2 + E1 + E3 CLI, under the lock) | 32 passed |
| `test_prod_engine_freeze_physical.py` + `test_prod_contract_parity.py` (with the unit files) | 102 passed |
| G1 regression tier, frozen `run-physical` / `score-physical`, analysis | all serial under `/private/tmp/claude-501/sherman_geom.lock`, `OMP_NUM_THREADS=1`; peak RSS 0.94 GiB (engine) / 1.0 GiB (runner) |

## Generality

- No SKU, family, part-id, file-name or metadata-count logic anywhere.
- Every new rule is geometric: hole shape and area, stage-material occupancy from the compiled part,
  eigen-directions of the registration information, the CAD far-boundary profile, and hinge support
  ratios.
- E3's inputs are unchanged: no file names or counts. All thresholds are engine config, not gate
  thresholds. The frozen gate thresholds are untouched.

## Risks

1. **The `in_plane_degenerate` semantics changed** (fix 2): the flag now means "no position can be
   determined". UI and PDF consumers that read the flag as "the pose is degenerate in some direction" will
   now see false on partially determined poses. The per-bend UNKNOWN(REGISTRATION_UNTRUSTED), the
   withheld heatmap and the `IN_PLANE_POSE_PARTIAL` warning carry the information. R1 should review this.
2. **The station profile assumes E3's station mapping to the CAD axial coordinate is right to well under
   a step's width.** A misregistered child or a twisted flange shows up as station scatter (larger U,
   REVIEW), not as a hidden error. The profile is used only when the compiled nominal lies inside the CAD
   range. A wrong replica (the panel's loops differing from E1's OML-side loops) could shift a deviation
   on side panels whose OML skin is skin 1 at a bend-type boundary. No such case appeared.
3. **The displaced-flange search can, in principle, take a strong parallel feature 3 to 10 mm away for
   the child of a bend that is really unformed.** It is limited to sweeps of at least 45°, formed-like
   angles, a 1.5 × support margin, and only when the narrow result is flat or partial. No synthetic case
   changed (G1 identical), but staged real scans (G3) have not been seen.
4. **The non-round fill test is less sensitive than a disc count**: it needs 30 % core coverage. A wrong
   part whose window is only partly closed could pass the hole check. Explained share and the mirror
   check still apply.
5. **One unchanged threshold costs lines.** `degenerate_min_information` 3 still withholds 15651004 B1/B2
   (2.03 effective observations). It was left unchanged on purpose.

## Candidates for iteration 2

- 15651004: use datum hole walls more strongly, or a principled information test, so a well-conditioned
  2-observation direction is not discarded.
- 38172000: two to three angle FAILs per scan at −1.5° to −2.8°. They are consistent across X and V, so
  likely real, but they need a visual check of the section.
- 38172000 V, B5 side B: far-edge station scatter of 15 mm (REVIEW, U 30 mm).
- Once the G3 table is frozen: the staged-scan behaviour of fixes 1 and 4 (tabs flat in windows; unformed
  bends next to formed flanges).
- `docs/production/ENGINE_INSPECT.md` lists the old algorithm ids. That document belongs to its own
  owner; A1 only writes `engine/**`.

## Files

| Path | Contents |
|---|---|
| `engine_before.tgz`, `engine.diff` | Snapshot and diff |
| `raw/` | Frozen `physical_scorecard.json`, `subset_scores.json`, `goldset_scores.json`, `per_bend.json`, and the G1 regression scorecard (json and md) |
| `overlays/` | The 20 worst bends after the fix, as close-up plus raw-frame context (`INDEX.json`) |
| `diag/` | Iteration-0 hole and datum renders, and the 752e segment view |
| `tools/` | Diagnostic and run scripts. Work directory: `…/scratchpad/a1_iter1` (outside the repository). |

Custody: only `cases_A1.json` cases were run or scored. Nothing `*SEALED*`, FRESH, HOLDOUT10, N001–N016
or EHO4 was read.

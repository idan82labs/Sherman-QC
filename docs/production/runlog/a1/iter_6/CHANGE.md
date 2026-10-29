# A1 iteration 6: engine change record (2026-10-19)

All physical numbers below come from consumed development scans (engineering ledger). None of them is
unseen-family accuracy. Synthetic numbers are E2 DEV seeds, or E1 CAD stage models seen through E2's scanner;
none of them is physical truth.

## Summary

- **Owner direction (2026-10-19).** Measure recall against the gold of record and never tune to the frozen BROAD
  labels. Remaining engine work: wrong-way bends (mirrored hinge search), recipe-excluded folds, the 2 tab-merge FPs
  on 38172000 SCAN_1, and per-bend correctness given CAD toward 0.90 with zero false PASS.
- **One general change: folds formed the wrong way are found and measured** (engine `e3-1.8.0`).
  - A mirrored hinge search reads a child that turned to the other side of its parent's plane. Its hinge is the CAD
    axis reflected across the parent's mid-plane.
  - The bend then reads FORMED the wrong way: its angle FAILs (fix C of iteration 5), and its bend line is measured
    by mirror symmetry and published at the fold's own outer sharp.
  - The search runs on the chosen pose only, so the pose decisions are exactly iteration 5's.
- **Effect on the physical set.** Of the 26 runnable A1 cases only 23553000 SCAN_2 changes: its two hooks bent
  about 119° the wrong way (B2, B14) go from UNKNOWN (LOW_DENSITY) to measured.
  - BROAD (frozen gold): recall 0.729 → **0.743**, micro F1 0.693 → **0.702**, family-macro 0.814 → **0.817**
    (best F1 lane on the same scans: 0.451 / 0.508 / 0.543).
  - BROAD scans against the gold of record: recall 0.981 → **0.994**, precision 0.987 → 0.988, family-macro
    0.972 → **0.974** (best F1 lane 0.475 / 0.679 / 0.581).
  - Per-bend correctness given CAD (riser-aware): 181 → **183 of 203 (0.901)**, family-macro 0.907 → 0.910. Strict
    stays 170. False-PASS proxies stay the same 3 bends.
  - GOLD_SET_V1 (0.990 / 0.974) and the Gold11 relabel (0.987) are unchanged.
- **Not changed, with reasons** (details below): the tab-merge FPs (a convention and contract question for the
  owner), the recipe-excluded fold (no recipe bend to report it on), the rolled 28870000 SCAN_1 (wrong-part test).
- **External change that breaks the ledger check.** `contracts/production/inspection_result.py` was edited by
  another workflow at 2026-10-19 01:44 (contracts fingerprint `5f7e6a8c…` → `78e31787…`). Every scorecard of this
  iteration reports `CONTRACTS_CODE_CHANGED_SINCE_FREEZE` (ledger `ok: false`; G1 adds `FREEZE_LEDGER_MISMATCH`).
  A1 does not own that file and did not touch it.

## Provenance

| Item | Value |
|---|---|
| Snapshot before editing | `engine_before.tgz`, sha256 `e34794922f7dfff151cf490c62bb1303abb1f911c0ec00bc890ce4e351691e52`, written 2026-10-18 22:07 by the first start of this iteration. Its inspect fingerprint is `98feb9dc…`, iteration 5's final engine. |
| Why it was not re-taken | This is the second start. The first start (22:07-22:17) only ran section-angle diagnostics on 10839000 (scratch `a1_iter6/logs/secang_*`, `dbgang_*`) and made no edit: at this start the engine tree was byte-identical to the snapshot (`diff -rq`). |
| Previous iteration | Iteration 5 improved every gate-level number and did not regress, so nothing was reverted. The previous generality check was clean, so nothing had to be removed. |
| Diff against the snapshot | `engine.diff`: `inspect/articulated.py`, `metrology.py`, `engine.py`, `config.py`. No change to `eval/`, `compile/`, `synthetic/` or `contracts/`. |
| Engine | `e3-1.8.0` (was `e3-1.7.0`). Algorithm ids: `articulated_fit` `…_astray_wide_reversed_mirrored_hinge_v8`; `metrology` `…_reseat_signed_angle_reversed_position_mirrored_v8`. |
| Code fingerprints | Engine (`inspect/`) `b9951fec…` (iteration 5: `98feb9dc…`). Compile `6147b43c…` and eval `8aa6860c…` unchanged. Contracts `78e31787…`: changed by another workflow (Summary). |
| New config fields | `reversed_support_ratio` 1.5 (≥ 1), `reversed_two_skin_share` 0.2 (in (0, 1]). Range-checked. Frozen gate thresholds untouched. |
| Tests touched outside the engine | `tests/test_prod_engine_inspect_a1_iter6.py` (new, 8 tests). No existing test changed. |

## The change: folds formed the wrong way

**What was missing.**
- Iteration 5's audit found two BROAD misses a better engine could recover: the hooks of 23553000 SCAN_2
  (pc_186463746daa B2, B14). Both reviewers of the G3 truth table see them bent about 115-120° the wrong way
  (`REVERSE_DIRECTION_VS_CAD`); any angle check must FAIL them.
- The CAD-direction hinge search covers θ from -10° to sweep + 25°. A wrong-way child lies outside it, so the engine
  read clutter (146°, 313 points) and reported every characteristic UNKNOWN (LOW_DENSITY).
- Fix C (iteration 5) already gives a negative signed sweep and FAILs the angle once such a child is fitted. What it
  lacked was the search and the child's pose.

**Rule 1. The mirrored hinge** (`articulated.MirroredBend`, `mirrored_bend`, `mirrored_hinge_frame`).
- A wrong-way fold starts from the same flat blank and consumes the same neutral strip. It turns the child toward
  `-v` about the CAD axis reflected across the parent's mid-plane.
- Skin 0 still continues across the bend, so the fold's outer skin is the CAD's inner one (`oml_skin` flipped).
- `fold_transform` takes a `MirroredBend`. At the nominal sweep it gives the CAD child reflected across the parent's
  mid-plane with its own skins kept, a proper rigid motion (unit tests).
- In the mirrored hinge frame (`o'` reflected, `a' = -a`, `u' = u`, `v' = -v`) the wrong-way fold has the CAD fold's
  cross-section exactly. So the existing two-skin Hough search, its rho windows and the child growth apply unchanged.

**Rule 2. The search and its guards** (`articulated._reversed_reading`, `_sheet_skins`).
- Angles from past the flat limits and the CAD search's reach (`-theta_min_deg`) up to `180 - reversed_wrap_margin_deg`.
  A reading nearer 180° would keep a positive sign in the signed sweep, so it is not searched.
- Only material no fitted panel owns is used.
- The reading replaces the CAD-direction one only when:
  - its support is at least `reversed_support_ratio` (1.5) times the CAD reading's (0 when the CAD search found no
    child);
  - it shows a sheet: a level parallel to its visible skin, 0.4 to 1.6 thicknesses away, holds at least
    `reversed_two_skin_share` (0.2) of that skin's support. On the physical hooks the two skins read 0.6 mm apart
    for a 0.935 mm sheet (a thin lip's skins blur toward each other), so a fixed one-thickness pairing rejected them;
  - its line meets the usual child-line minimums, and the child grows from it.
- The child then grows with the mirrored kinematics; its grandchildren are searched from that pose.

**Rule 3. Chosen pose only** (`engine.inspect_scan`, `fit_articulated(reversed_search=...)`).
- Every candidate pose gets the CAD-direction fit, as in iteration 5. The pose decisions (ranking, twins, ambiguity,
  free directions) are therefore unchanged.
- The chosen pose is then fitted again with the mirrored search. The new fit is used only when it reads at least one
  bend the wrong way and did not run out of time.
- Why: with the mirrored search on every candidate, a datum turned upside down could read each of its bends as
  formed the wrong way and explain the scan as well as the true pose. A first version did that and is not kept.
- Cost: one more articulated fit per scan. Physical runtime P95 26.9 → 27.9 s, max 31.8 s (gate 45 s).

**Rule 4. Bend-line position by mirror symmetry** (`metrology.reversed_position_measurement`, `engine._outcomes`).
- The fold's own outer sharp is where the CAD's inner skins meet (the OML planes moved by `t`), per station and
  region-wide (`BendMeasure.rev_sharp`, `rev_line_point`).
- Reflected across the parent's mid-plane, the wrong-way fold is a fold of the same bend line turned the CAD way, and
  those sharps become its OML sharps. They are compared with the stage nominal like any bend's position. The
  reflection keeps every offset within the parent's plane; it removes only the side the fold turned to.
- The position is MEASURED with method `STATION_REVERSED_FOLD_SHARP_MIRRORED_VS_STAGE_NOMINAL_…` and the note
  `BEND_FORMED_WRONG_WAY: own outer sharp mirrored across the parent mid-plane`.
- The display line is the fold's own outer sharp, so the frozen line metric publishes it.
- Flange lengths and the backgauge stay UNKNOWN (UNSUPPORTED_FEATURE) as in fix C. The angle FAILs (`180 + sweep`).
  The bend therefore always FAILs; its position may read PASS when the bend line itself sits where it should.

## Results

**Physical set** (frozen `prefreeze run-physical` / `score-physical`, all 27 A1 cases, serial under the shared lock,
`OMP_NUM_THREADS=1`; scratch `a1_iter6/phys`).
- Characteristic-level diff against iteration 5's final run (0.05 tolerance): **1 of 26 cases changes**,
  pc_186463746daa (23553000 SCAN_2):
  - B2: angle UNKNOWN → FAIL (299.0° vs 60° included), position UNKNOWN → FAIL +1.11 mm (U 0.17);
  - B14: angle UNKNOWN → FAIL (298.8°), position UNKNOWN → REVIEW +0.59 mm (U 0.15);
  - both: flanges UNKNOWN (LOW_DENSITY) → UNKNOWN (UNSUPPORTED_FEATURE), note `BEND_FORMED_WRONG_WAY`, state still
    UNEXPECTED_FORMED (G3 truth FORMED: match), explained share of the scan 0.990 → 0.999.
- pc_b4719db3dd69 stays compile BLOCKED.
- The engine log names every wrong-way reading it adopts (`formed the wrong way: …`). Physical set: only
  pc_186463746daa (B2, B14). G1 tier: none of 50 cases.

**Line localisation** (apex locus, frozen matcher). Engine vs the best F1 lane on the same scans.

| Gold | Scans | Iteration 5 | **Iteration 6** | Best F1 lane |
|---|---|---|---|---|
| BROAD gold (frozen, older labels) | 24, 14 families | P 0.660, R 0.729, micro 0.693, family-macro 0.814 (TP 105, FP 54, FN 39) | **P 0.665, R 0.743, micro 0.702, family-macro 0.817** (TP 107, FP 54, FN 37) | P 0.580, R 0.451, micro 0.508, family-macro 0.543 |
| BROAD scans, gold of record per scan (report-only) | 24, 14 families | P 0.987, R 0.981, micro 0.984, family-macro 0.972 | **P 0.988, R 0.994, micro 0.991, family-macro 0.974** (TP 159, FP 2, FN 1) | P 0.679, R 0.475, micro 0.559, family-macro 0.581 |
| BROAD scans, relabel substituted on Gold11 | 24 | micro 0.966, family-macro 0.945 | micro 0.972, family-macro 0.948 | micro 0.540, family-macro 0.560 |
| GOLD_SET_V1 (A1 scans with STEP) | 23, 14 families | micro 0.990, family-macro 0.974 (TP 150, FP 2, FN 1) | same | micro 0.566, family-macro 0.583 |
| Gold11 relabel v1 (gold of record) | 9 | F1 0.987 (P 0.975, R 1.000) | same | F1 0.547 |
| Human Gold11 (historical; G2(a) gate of record) | 9 | F1 0.391 | same | F1 0.473 |

- 23553000 SCAN_2 now matches all 9 of its BROAD gold lines (was 7).
- The 37 remaining BROAD FNs: 36 have no line of the gold of record at that place (iteration 5's label classes) and
  1 is the recipe-excluded 28870000 fold. Recall on the frozen BROAD gold can reach at most (107 + 1) / 144 = 0.75
  without fitting label errors.
- The 2 FPs against the gold of record are the 38172000 SCAN_1 tab lines (below).

**Per-bend correctness given CAD** (all A1 cases; iteration 4's definition, `tools/per_bend_all.py`; report-only)

| | Iteration 5 | **Iteration 6** |
|---|---|---|
| bends scored | 203 | 203 |
| correct, strict | 170 (0.837) | 170 (0.837) |
| correct, riser-aware | 181 (0.892) | **183 (0.901)** |
| family-macro, riser-aware | 0.907 | **0.910** |
| false-PASS proxy | 3 | 3 (the same bends: the SCAN_1 tab B1 and the two disputed 10839000 scan_2 rows) |

- The two hooks now count under the riser-aware definition (state formed, angle measured, sides UNKNOWN
  UNSUPPORTED_FEATURE, no gold of record on that scan). They do not count as strict: their sides are not measured,
  and each also has a side ending at an adjacent bend that is not formed at that stage.
- The other 20 bends not counted correct are iteration 5's classes, none engine-limited on this definition: 8 sides
  ending at an unformed adjacent bend, 4 NOMINAL_UNAVAILABLE from the frozen compile, 5 disputed truth rows, 2 tab
  lines (below), 1 rolled part.

**G2 / G3** (frozen thresholds): every verdict equals iteration 5's.
- G2: FAIL on (a) (human Gold11 gate of record, F1 0.391) and (f) (15651004's "repeats" are three stages). (b)-(e) and
  compile rate PASS; runtime P95 27.9 s, max 31.8 s, 0 crashes.
- G3: state match 118/123 = 0.959 FAIL (the 4 disputed 10839000 scan_2 rows and the rolled 28870000 SCAN_1),
  known-good angle MAE 1.186° FAIL (10839000, as in iteration 5), unformed reported as defect 2 FAIL (the disputed
  rows). Formed-bend angle coverage 77/81 → **79/81** (the hooks); flange coverage 130/160; known-good flange P95
  PASS.
- Ledger: `CONTRACTS_CODE_CHANGED_SINCE_FREEZE` (external edit, Summary).

**G1 regression tier** (DEV seeds, 50 cases, 7 families; scratch `a1_iter6/g1_rev3`).
- regression_verdict **PASS**, 0 failed gates. Every accuracy gate is identical to iteration 5: 29 of the 32 gate
  records are byte-identical, and the other 3 are the informational runtime, RSS and tier-budget records (max wall
  15.7 → 16.1 s). The case counts, scenarios and summary states are identical too.
- Non-qualifying as before (TIER_REGRESSION, SEEDSET_DEV, SCANNER_MODEL_PARTIAL), plus FREEZE_LEDGER_MISMATCH from
  the external contracts edit.

**CAD-stage probes** (E1 stage meshes of all A1 recipes, every stage, springback ±2° on formed bends and ±1° kinks on
unformed ones, through E2's scanner; iteration 4's cached seed-23 probe scans; `tools/sbsweep6.sh`,
`validation/probe_sweep_final.log`).
- 113 probes, 1006 bend states: 0 wrong states, 0 not VALID, 0 angle false PASS / false FAIL, 501/503 angles within
  U, 96 other FAILs (position, springback-driven): the same as iteration 5.
- Every characteristic of every probe (state, decision, value to 1e-6) equals iteration 5's. No probe adopted a
  wrong-way reading.

**Wrong-way probes** (new; E1 stage meshes with chosen bends folded the wrong way by the mirrored kinematics,
through E2's scanner; `tools/reversed_probe6.py`, `tools/revsweep.sh`, `tools/revsweep_base.sh`; scratch
`a1_iter6/revprobes/v3` and `v3base`). Both engines read the same scan files (same sha256).

| Recipe (family), stage, bend folded the wrong way | Iteration 5 engine | **Iteration 6 engine** |
|---|---|---|
| 38172000 S12, B5 (30° hook) | PARTIALLY_FORMED, angle FAIL +35.5° (wrong value), position UNKNOWN | FORMED wrong way, angle FAIL **+59.9°** (true +60), position PASS +0.02 mm |
| 15651004 S3, B1 (90°) | PARTIALLY_FORMED, every characteristic UNKNOWN | angle FAIL **+180.0°**, position +0.02 mm |
| 10839000 S12, B5 (90°) | UNKNOWN | angle FAIL **+180.0°**, position -0.02 mm |
| 23553000 S16, B2 + B14 (120° hooks) | UNKNOWN, UNKNOWN | angles FAIL **+240.0° / +240.0°**, positions +0.01 / -0.01 mm |
| 28870000 S1, B1 (25°, rolled part) | WRONG_PART_SUSPECTED (explained 0.221) | VALID, angle FAIL **+49.9°**, position REVIEW -0.87 mm |
| 38172000 S12, B1 + B9 (90° tabs) | **NOT_YET_FORMED** (hidden; every later bend then UNEXPECTED_FORMED) | angles FAIL **+180.0°**, positions -0.01 / +0.04 mm; the other bends FORMED |
| 15651004 S3, B1 at 15 % (-13.5°; arc faces dropped) | UNKNOWN | PARTIALLY_FORMED wrong way, characteristics UNKNOWN (LOW_DENSITY) |
| 50757000 S9, B2 (90°) | WRONG_PART_SUSPECTED (the flange covers two datum holes) | same: never PASS |
| 44211000 S1, B1 (55°, single-bend part) | read as the CAD part turned over, every characteristic PASS | same (Risks 2) |

- Apart from the folded bends (and, on the B1 + B9 probe, the later bends' UNEXPECTED_FORMED → FORMED), no bend of
  these probes reads differently between the two engines.
- The probe mesh keeps the material's skins through the reversed arc (`r → 2 Ri + t - r` about the mirrored axis). A
  first probe version reflected the arc without that swap; its arcs joined the wrong skins and gave distorted
  readings (8-19° off), which were probe artifacts and are not quoted.
- On the 23553000 S16 probe, B3 and B13 read -3.9° FAIL with both engines: the reversed lip next to the short panel
  disturbs that panel's fit. This is not introduced by the change.

## What was not changed, and why

- **The 2 tab-merge FPs on 38172000 SCAN_1** (pc_07062eee17aa B1/B9).
  - At that stage B4 and B11 are not formed, so the tabs B1/B9 fold along the same lines as the 214 mm folds
    B2/B10. The relabel and GOLD_SET_V1 draw one 305 mm line for each pair; the engine publishes the two recipe
    bends.
  - The parent side is one flat plate at that stage (P8, P7 and P5 are joined through unformed bends). The child side
    is two separate panels in the CAD: the tab P11 (child of B1 only) and the flange P12 (child of B2 only). The
    convention says "a fold interrupted by windows or reliefs that fold the same two plates is ONE line", and here
    the children are two plates. Whether such coplanar flanges count as one plate is the owner's call.
  - A merge also needs a contract decision. The contract requires `measured_polyline_mm` exactly when
    BEND_LINE_POSITION is MEASURED, so one line per pair means withdrawing the tab's measured position (a recipe
    characteristic) to UNKNOWN. Per-bend correctness would not change either way: the frozen per-bend tool gives the
    one gold line to one bend.
- **The recipe-excluded fold** (28870000 FULL, gold B2, 9.7 mm from the recipe's B1 and parallel to it).
  - E1 compiles one bend for this part and excludes the rest as ROLLED (`bend cylinders meet 1 tangent ends`). The
    frozen line metric publishes recipe bend rows only, so there is no row to report the fold on.
  - The engine's unplanned-crease search reports one crease on this scan (`C1`, EXTRA_BEND_SUSPECTED, 127°), but it
    lies 33 mm from that gold line (`tools/crease_check.py`). It is a different feature, not the missed fold.
- **The rolled 28870000 SCAN_1** (WRONG_PART_SUSPECTED, truth: B1 not formed). The roll differs from CAD S0, so only
  21-24 % of the scan is explained. Relaxing the wrong-part test for it would risk G1's WRONG_OR_MIRRORED_NEVER_PASS;
  left open as in iteration 5.

## Generality

- No SKU, family, part-id, file-name or metadata-count logic was added, and E3's inputs are unchanged.
- What the rules use: the parent's fitted pose and CAD mid-plane, the bend's CAD axis, sweep, inner radius and
  neutral strip, the scan's points in the hinge band, and point ownership.
- The new thresholds are the two engine config fields above, with range checks.
- Checked on all 27 physical A1 cases, the G1 regression tier, the CAD-stage probes and the wrong-way probes.

## Risks

1. **A false wrong-way reading.** A bend whose CAD reading is weak (occluded) could be read the wrong way on
   material below its parent.
   - Guards: material owned by no fitted panel, a parallel second skin, 1.5 times the CAD reading's support, the
     child-line minimums and a grown child; chosen pose only.
   - Seen nowhere: the only adoptions on the physical set are the two hooks, none on the 50 G1 cases, none on the
     113 CAD-stage probes, whose every value is unchanged.
2. **A part whose every bend is formed the wrong way** reads as the CAD part turned over when its datum's cut
   features are (nearly) symmetric under that flip: the pose choice is iteration 5's and prefers the reading that
   needs no wrong-way bend.
   - The single-bend 44211000 probe shows it: its datum fits that pose at 0.977 against 0.988 for the true pose.
   - If most bends of such a part are reversed, the minority would be named as reversed instead.
   - A version with the mirrored search on every candidate pose found the 44211000 probe, but risks turning
     symmetric good parts upside down; it is not kept.
3. **The mirrored position assumes the CAD inner radius and strip** for the wrong-way fold (the same tooling, part
   turned over). On the probes it reads within 0.04 mm of nominal, except on the rolled 28870000 part (-0.87 mm,
   REVIEW). The physical hooks read +1.11 / +0.59 mm, with no physical truth for them.
4. **The frozen line metric places a wrong-way bend's apex on the wrong side** of the published sharp (it rotates the
   CAD bisector by the measured angle). For the 119° hooks that is about 2.9 mm from the real apex; both still match
   their BROAD gold lines within the 4 mm gate. A wrong-way bend with a large outer radius could be missed by the
   metric while the engine's line is right.
5. **Runtime**: one more articulated fit per scan, about +1 s.

## Tests

| Test | Result |
|---|---|
| `tests/test_prod_engine_inspect_a1_iter6.py` (new): mirrored kinematics (same flat blank; the CAD fold reflected with its skins kept; proper motion; the mirrored frame's cross-section), the two-skin search finds a wrong-way flange about the mirrored hinge and not about the CAD one, the sheet test at 0.6-1.4 t and against one surface, position by mirror symmetry (0 at nominal, -0.7 mm for a line 0.7 mm toward the child), its minimum stations, config range checks. Pure numpy; no part named. | 8 passed |
| The final engine: `test_prod_engine_inspect_cli.py` (E3's E2E tests, E2 → E1 → E3), `_units.py`, `_a1.py`, `_a1_iter2.py` … `_a1_iter6.py` | **150 passed** (216 s, under the lock; iteration 5: 142) |
| Runs | The tests, the physical set, the G1 tier, the probe sweeps and every debug run were made one at a time under `/private/tmp/claude-501/sherman_geom.lock`, with `OMP_NUM_THREADS=1`. |

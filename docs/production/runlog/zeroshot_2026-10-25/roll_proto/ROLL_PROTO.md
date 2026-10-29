# Roll measurement prototype (no CAD): ROLL_PROTO

2026-10-25. Research prototype, owner-approved. Code: `research/zeroshot/roll_measure/`. Tests: `tests/test_prod_roll_measure.py` (7 pass, about 5 s). Machine-readable output: `results.json` in this folder. Images: `png/`.

**Custody.** The three new customer scans (41018000_H, 45351000_B, 51101001_01) are *development scans for this feature*: they are consumed for the N1 bend test. The DEV scans used are DEV_CONSUMED (INVENTORY.json). No TEST-B, DEV-CHECK, FRESH, sealed or forbidden data was read. Gold bend lines were not read. The markers' notes were read only after the numbers below were final.

Known rolled parts in the consumed FRESH set were **not used**: 38539000, 49566000, 44001000, 49565000, 43798000, 50278001, 46465000.

## 1. Method (short)

1. **Candidate axes.** A cylinder puts its normals on a great circle. Smoothed normals that are not on flat-flange peaks (z3 `gauss_samples`, `normal_peaks`) vote on a hemisphere grid, greedily, with each accepted axis explaining away its band.
2. **Sections and arcs.** Each axis is sliced with z3 `slice_family`. Each section chain gets a signed curvature (convex vs concave from the outward normals). Constant-sign runs are kept as arcs only if all of the following hold on the **raw** points:
   - R ≥ max(12 mm, 5 t);
   - sweep ≥ 15°;
   - circle rms < 6 % of the arc sagitta;
   - the curvature is uniform.

   The tangent points are found by a 1-D circle-then-tangent-line model search, which is v2's sweep idea applied along the arc. Free edges extend along on-circle points. Co-circular pieces and gaps in the scan are bridged.
3. **Regions.** Arcs are tracked across stations. Convex tracks (outer skin) and concave tracks (inner skin) are paired per station when they share a centre and sit t apart. Coaxial pieces are merged, and regions found from several candidate axes are de-duplicated after refinement.
4. **Axis.** The axis is a TLS line through the station centres of a common angular window, iterated by re-slicing until it moves less than 0.05°.
5. **Measurement.** Every number is a per-station series with a bootstrap over stations. Thickness comes from the two skins where both are scanned, otherwise from D1 (`d1_two_skin.estimate_thickness`). The two agree to within 0.005 mm on 7 of 9 scans. The exceptions are 28870000 FULL (0.005 mm) and 48565022 (two-skin 1.936 vs D1 2.052: D1's whole-part mode is pulled by the plate, while two-skin is measured on the curl).
6. **Disjoint from the bend lane.** Sharp corners fail the radius and rms/sagitta tests. Curved regions with Ri ≤ 10 t are labelled `LARGE_RADIUS_BEND`, following the CAD solver's rule (`research/cad_roll_solver/SCHEMA.md`), so they go to the bend lane rather than being reported as rolls.

**Uncertainty reported.**
- `u_station_boot` is the bootstrap over stations.
- `u_total = sqrt((2·u_boot)² + floor²)`. Here *floor* is the synthetic RMS error of the method for the nearest sweep class. The factor 2 comes from calibration: on the synthetic grid, |error|/u_boot has a median of 1.2–1.3 and a 90th percentile of 2.1–2.5.
- Neither number covers real-part form errors such as a multi-radius profile, faceting or holes. Those are listed per scan.

## 2. Synthetic accuracy (truth known)

**Set-up.**
- Cylinders and cones, sampled as two-skin triangle meshes (0.6–0.8 mm) with edge walls, in a random pose.
- Gaussian normal noise of 0.07 mm, which is the median per-station circle rms of the real rolled scans (0.011–0.13 mm).
- Grid: R ∈ {30, 60, 120, 200, 300} mm × sweep ∈ {30, 60, 90, 135, 180}°, t = 2 mm, L = 150 mm, 30 mm tangent flats.
- Plus 26 special cases. Detection: 51/51.

**Grid, all 25 cases.** Bias ± sd, then RMS and max |error|.

| Quantity | Bias ± sd | RMS | max \|e\| |
|---|---|---|---|
| R_inner (mm) | −0.001 ± 0.021 | 0.021 | 0.083 |
| R_outer (mm) | −0.002 ± 0.020 | 0.021 | 0.079 |
| t (mm) | 0.000 ± 0.001 | 0.001 | 0.003 |
| sweep (°) | +0.07 ± 0.18 | 0.19 | 0.92 |
| chord at run-outs (mm) | +0.04 ± 0.08 | 0.08 | 0.23 |
| run-out line position (mm, ⊥ distance to truth) | 0.06 ± 0.03 | 0.06 | 0.11 |
| axis direction (°) | mean 0.027 | | 0.18 |
| taper (mm/m, truth 0) | +0.17 ± 0.68 | 0.70 | 3.2 |
| twist (°/m, truth 0) | 0.00 ± 0.36 | 0.36 | 0.96 |
| axis straightness rms (mm, truth 0) | 0.17 ± 0.28 | | 1.1 |
| coverage (truth 1) | 0.000 | | 0 |

**By sweep class** (RMS error). Small sweeps are the weak case.

| Sweep | R_in (mm) | Sweep (°) | Taper (mm/m) | Run-out (mm) | Chord (mm) |
|---|---|---|---|---|---|
| 30° | 0.046 | 0.41 | 1.55 | 0.07 | 0.13 |
| 60° | 0.007 | 0.07 | 0.25 | 0.08 | 0.11 |
| 90° | 0.005 | 0.05 | 0.12 | 0.06 | 0.07 |
| 135° | 0.002 | 0.05 | 0.03 | 0.05 | 0.05 |
| 180° | 0.001 | 0.08 | 0.02 | 0.07 | 0.004 |

**Special cases** (errors).

| Case | Result |
|---|---|
| Cones, taper 10–30 mm/m (4 cases) | Taper error ≤ 0.05 mm/m, R ≤ 0.02 mm, sweep ≤ 0.08° |
| One skin only (inner or outer, 4 cases) | The scanned skin's R ≤ 0.008 mm. The other skin is reported as not measurable when no thickness is known. |
| Hole across 15–20 % of the arc (2 cases) | Sweep ≤ 0.06°, R ≤ 0.005 mm |
| Noise ×2.5 (0.175 mm) | R ≤ 0.002 mm, but sweep −2.3° at R 200 |
| t = 4 mm and t = 0.6 mm | R ≤ 0.003 mm, sweep ≤ 0.02° |
| **Free edges** (no flat beyond the arc, 4 cases) | Sweep −0.2° to −1.7°, run-out 0.4–0.5 mm short |
| **Short flats** before a free edge: 10 mm (R 60 / R 200) and 5 mm | Sweep +2.6° / +1.0° / +7.0°, run-out 1.3 / 1.8 / 3.8 mm |
| **Injected flat spots**, 0.5–1.5 mm deep, 20° wide, over half the length (5 cases) | 4/5 detected, depth −0.05 to −0.17 mm, location ±1–5°. The one at 10 % of the sweep (next to the run-out) was absorbed into the run-out: sweep −6°, run-out moved 22 mm. A flat spot on part of the length also biases taper by 2–9 mm/m. No false flat spots on the 25 clean cylinders. |

Full rows: `results.json → synthetic.rows`.

## 3. Real rolled scans: measurements

Uncertainties are `u_total`. "sd" is the station-to-station spread, which measures consistency along the length. All values in mm unless marked °. Axis points and directions and the 3-D run-out lines are in `results.json`.

| Scan | Class (Ri/t) | t | R_inner | R_outer | Sweep ° (stations) | Chord | Taper mm/m | Twist °/m | Fit rms | Arc coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| **51101001_01** (N1) | ROLL (48) | 3.921 | 190.35 ± 0.01 (sd 0.07) | 194.27 ± 0.01 | 177.1 ± 0.3 (87) | 380.5 ± 0.0 | +0.56 ± 0.09 | −7.8 ± 0.6 | 0.091 | 0.99 |
| **45351000_B** (N1) | ROLL (139) | 0.538 | 74.77 ± 0.04 (sd 0.20) | 75.30 ± 0.04 | 175.4 ± 1.9 (60 clean) | 148.7 ± 0.4 | −0.1 ± 0.8 | 5.3 ± 0.8 | 0.071 | 0.88 (holes) |
| **41018000_H** (N1) | ROLL (215) | 1.227 | see note | | | | **−33.9 ± 0.5** (cone) | | 0.076 | 1.00 |
| 49125000_A00 | ROLL (25) | 1.469 | 37.23 ± 0.02 (sd 0.06) | 38.70 ± 0.02 | 146.7 ± 0.3 (46) | 70.6 ± 0.6 | +1.5 ± 0.5 | 3 ± 5 | 0.011 | 1.00 |
| 28870000 FULL | ROLL (20) | 1.456 | 29.55 ± 0.01 (sd 0.03) | 31.00 ± 0.02 | 140.5 ± 0.3 (47) | 55.3 ± 0.4 | +0.5 ± 0.3 | 11 ± 7 | 0.013 | 1.00 |
| 28870000 SCAN_1 | ROLL (21) | 1.454 | 29.85 ± 0.02 (sd 0.05) | 31.30 ± 0.02 | 136.7 ± 0.8 (44) | 55.1 ± 0.5 | +1.7 ± 0.5 | 5 ± 3 | 0.030 | 0.99 |
| 47479000_C | ROLL (70) | 1.016 | 70.99 ± 0.07 (sd 0.34) | 72.00 ± 0.08 | 177.7 ± 3.2 (90) | 140.8 ± 1.9 | +2.8 ± 0.2 | 11.5 ± 1.7 | 0.129 | 0.94 |
| 48565022_A | LARGE_RADIUS_BEND (6.8) | 1.936 | 13.07 ± 0.03 | 15.01 ± 0.04 | 131.7 ± 1.3 | 23.0 ± 0.6 | 0.2 ± 1.3 | | 0.025 | 1.00 |
| 49188000_A | LARGE_RADIUS_BEND (8.3) | 2.944 | 24.46 ± 0.01 | 27.41 ± 0.01 | 161.1 ± 0.3 | 48.3 ± 0.0 | 1.3 ± 0.6 | | 0.008 | 1.00 |

**41018000_H** is a stepped, two-width roll, so one R and one sweep do not describe it.
- **Narrow portion** (190 mm long): arc 67.6° (P10–P90 66–70°). It ends at free edges and short tips.
  - Full-arc R_in 269.6 (sd 3.2), because it is a cone.
- **Wide portion** (125 mm): arc 130.9°, both ends tangent into flats about 40 mm wide.
  - Full-arc R_in 255.2 (sd 3.7).
  - The last ~20° at each end is tighter: zone radius 256 (start) and 228 (end), against 267 in the middle.
- **Taper.** Measured on a common 67° angular window (the same part of the profile at every station), R_in falls linearly from 273.5 to 262.7 over 330 mm, i.e. **−33.9 ± 0.5 mm/m**. This is a cone with a semi-angle of about 1.9°.
  - Independent RANSAC on three hand-cropped slabs of the narrow portion gives R_out 273.2, 271.6 and 269.5, i.e. −31 mm/m. The wide-portion slab (265.8) lies on the same line.
- Both markers saw a constant-radius roll (A: R ≈ 220; B: R ≈ 250–270). Neither noticed the cone or the tighter ends.
- Run-outs: the wide portion's lines are tangent (end line: rms 0.74 mm, u ± 0.4 mm). The narrow portion ends at edges.

**Consistency along the length** (`straightness`, `zone_radius`, `flat_spots`, heat maps).
- Axis straightness (centre rms / max): 0.015 / 0.04 (51101001), 0.025 / 0.08 (49125000), 0.30 / 0.86 (45351000; bow 0.78 mm), 0.25 / 1.6 (47479000).
- 41018000 shows 3.7 / 22 mm, which reflects its stepped multi-radius profile, not a bent axis.
- Flat spots: none passed the detector threshold, max(0.3 mm, 5 × fit rms), except on 41018000. There, all 10 flagged zones are the multi-radius ends, not dents.
- The heat maps show the following below the threshold:
  - 51101001: a ~15°-wide flattened lead-in at the start edge (−0.3 mm).
  - 45351000, 47479000 and 28870000: regular ±0.2–0.5 mm facet bands, the signature of step (bump) rolling.
  - 28870000 SCAN_1: a start zone R of 29.2 against 29.7 in the middle.
  - 47479000: start zone R 71.9 against 70.2 in the middle.
- Flat unrolled tips at free edges: 14 mm on 48565022 (start) and 3.8–5.6 mm on 28870000. Elsewhere they are ≤ 1 mm.

## 4. Independent cross-check (RANSAC + direct least-squares cylinder)

The cross-check is `crosscheck.py`. It runs RANSAC on raw points and normals, then a nonlinear least-squares cylinder fit on the inliers, with optional hand crops. It shares nothing with the slicing, arcs or tracking. There are three seeds per scan; the table shows the first, and the other two agree to within 0.03 mm except on 41018000, where they agree to within 1 mm.

| Scan | Main R_in / R_out | Cross-check R_in / R_out | Δ |
|---|---|---|---|
| 51101001 | 190.35 / 194.27 | 190.53 / 194.30 (whole); halves 190.27→190.38, 194.23→194.37 | ≤ 0.18 / 0.03. Taper of the halves: +0.7–0.8 mm/m vs main +0.56 |
| 49125000 | 37.23 / 38.70 | 37.35 / 38.72 | 0.12 / 0.02 |
| 28870000 FULL | 29.55 / 31.00 | 29.61 / 31.05 | 0.06 / 0.05 |
| 28870000 SCAN_1 | 29.85 / 31.30 | 30.03 / 31.36 | 0.18 / 0.06 |
| 49188000 | 24.46 / 27.41 | 24.52 / 27.40 | 0.06 / 0.01 |
| 48565022 (R ≤ 100 hand limit) | 13.07 / 15.01 | 13.19 / 15.02 | 0.12 / 0.01 |
| 45351000 | 74.77 / 75.30 | 74.77 / 75.08 | 0.00 / **0.22** |
| 47479000 | 70.99 / 72.00 | 71.59 / 72.39 | **0.60 / 0.39** (faceted shell, RANSAC rms 0.21) |
| 41018000 | cone; see §3 | whole part 266.0–267.0 (cylinder averages the cone); slabs −31 mm/m | taper agrees within 3 mm/m |

**Agreement.**
- On smooth rolls: within 0.2 mm, typically 0.05–0.12 mm.
- On faceted or bump-rolled or thin (0.54 mm) shells: 0.2–0.6 mm. A single cylinder is ill-posed there, so this sets the honest real-world R precision.
- The RANSAC inlier sweep (150–186°) overestimates by including the tangent overshoot within its 0.3 mm tolerance, so it is not a sweep check.

## 5. No false rolls

The screen ran on **all 95 usable DEV_CONSUMED scans** in INVENTORY.json, at full resolution, 0.1–7 s per scan, with 0 errors.
- **Hits (8):** 28870000 ×2, 47479000, 49125000, 48565022, 49188000, plus two UNKNOWN-part scans (`uploads/98072c85/scan.ply` and `02_CAD_QC_3D/part 1.ply`) whose R (29.5) and length match 28870000. These are likely copies of 28870000 and were not measured further.
- **Every hit was checked by eye** (`png/`). All are genuine curved regions:
  - 48565022 is a curled plate edge (Ri 13, 6.8 t);
  - 49188000 is a pipe-clamp semicircle (Ri 24.5, 8.3 t).
  - Both are `LARGE_RADIUS_BEND` under the CAD rule (Ri ≤ 10 t), so they are not rolls for reporting.
- **87 scans gave 0 rolled regions.** 86 of them have gold bend lines, i.e. press-brake parts. A random sample of 12 non-hits checked by eye (`png/screen_nonhit_sample.png`) shows only flats and sharp bends.
- **Explicit 10–15 flat-and-sharp-bend set** (one per part, most gold lines first): all 0 rolls.
  - 23553000 (16 lines), 47251002 (12), 38172000 (12), ysherman (12), 10839000 (12), 49204020 (12), 49191038 (10), 49191030 (10), 50757000 (9), 47247004 (9), 45796000 (8), 49191023 (8), 11979003 (7), 50566000 (7), 50552000 (7).
- 23553000, whose marker notes mention a roll, has no rolled region on any of its 3 DEV scans, and the CAD roll solver finds no roll in 23553000_B.stp. 11979003 (zero-bend control) also has no roll in CAD or on the scan.
- **No missed rolls among STEP-bearing DEV parts.** The CAD roll solver over the 23 DEV STEPs finds rolls only in 28870000 and 47479000, and the screen detects both.
- An earlier screen version reported 50546000 (×2) as rolls. Those are large-radius press bends (R_out ≈ 11 mm < 5 t) that leaked through a missing radius re-check after the end refinement. This was fixed, and the screen was re-run.

## 6. CAD comparison (measured − nominal)

Nominals are from `research/cad_roll_solver`. 28870000 and 47479000 use the solver's own outputs in `docs/production/runlog/cad_roll_solver/out/`. For 49125000 I ran the same solver in scratch on `uploads/05d7cd09/reference.stp`; that upload's `scan.ply` is byte-identical to 49125000_A00.ply, and the drawing title is cat 9353735-01 SPS FLEX GUIDE. The inventory had it as STEP_NOT_LOCATED.

**No registration was done.** All quantities below are pose-invariant. Overlays place the CAD on the fitted axis and the axial centre. Angularly, the CAD is anchored at its tangent (bend) run-out on the scan's non-edge end where that is unambiguous, otherwise centred. This is stated on every overlay.

| Scan | ΔR_in | ΔR_out | Δt | Δsweep ° | Δextent to edge ° | Δchord | ΔL |
|---|---|---|---|---|---|---|---|
| 49125000 | **+0.23 ± 0.08** | +0.20 ± 0.08 | −0.03 | **−7.6 ± 0.4** | −9.9 ± 1.1 | −1.5 ± 0.6 | −1.5 |
| 28870000 FULL | **−0.45 ± 0.08** | −0.50 ± 0.08 | −0.04 | **−13.6 ± 0.4** | −14.7 ± 0.8 | −3.2 ± 0.5 | −1.5 |
| 28870000 SCAN_1 | −0.15 ± 0.08 | −0.20 ± 0.08 | −0.05 | **−17.4 ± 0.6** | −16.5 ± 1.4 | −3.4 ± 0.6 | −1.5 |
| 47479000 | **−3.01 ± 0.09** | −3.00 ± 0.09 | +0.02 | −0.3 ± 1.6 | +0.3 ± 1.7 | −7.1 ± 1.1 | −2.9 |

ΔL is a lower bound: the scan length spans the measured stations, about one station step short. Taper: CAD 0; scan 0.5–2.8 mm/m (u ≈ 1).

**Readings.**
- **47479000's shell is about 3 mm tighter than CAD**: R 71/72 against 74/75. Sweep and t match. The RANSAC cross-check agrees (71.6/72.4).
- **The flex guides (49125000, 28870000) carry 8–17° less arc than CAD.** At R ≈ 30–38 mm, this is about 4–9 mm less rolled arc between the joggle and the free edge. A free-edge bias of −1.5° is known from synthetic and covers only a small part of it.
  - The scan arcs reach the sheet edge (verified in the sections), so the parts are shorter or less rolled than the CAD.
  - 49125000 is also 0.2 mm larger in R; 28870000 is 0.15–0.45 mm smaller.
  - The two 28870000 scans differ from each other by 0.3 mm in R and 4° in sweep. They are probably different physical parts, or the R difference reflects scan or part variation; it is far above the method's precision.
- The solver's sweep excludes edge rounds ("at a free end with edge rounds, the round is not counted"), so the free-end comparison is like-for-like.

**Not compared:** axis position and run-out positions. Both need the E3 registration pose.

## 7. Drawings found (Sherman roll dimensions and tolerances)

The customer drawing for 51101001 (02.001.6301-01.pdf) is **not on disk**: there is no PDF named 6301 or 51101001 anywhere under ~/ShermanQC. Two drawings of rolled parts were found, both from Spectrum Dynamics.

**`uploads/821462a4/drawing.pdf` = 28870000, cat 9351220-01 rev A, "DETECTION UNIT-FLEX GUIDE, COLUMN 8X1".**
- **Ø63** on the roll; the leader is on the outer surface, so R_out 31.5.
- 1.5 THK, Al-5053-H32.
- The arc is given by angles: **30°** above the horizontal centreline at the joggle end, and **35°** past the bottom centreline at the free end. That makes a nominal arc of 155°, which matches the CAD's 154.1°.
- Joggle legs 9.38 and 11.26, at 62° and 87°.
- R0.5 (×4) at the joggle bends and edge corners.
- Section A-A: holes at 10°, 60° and 103°.
- **Tolerances: general ±0.1, holes ±0.1, angular ±0.3°.** There is no roll-specific tolerance.

**`uploads/05d7cd09/drawing.pdf` = 49125000, cat 9353735-01 rev a00, "SPS FLEX GUIDE".**
- **Ø77** (leader on the outer surface, so R_out 38.5).
- The same 30° / 35° arc definition (nominal 155°), 1.5 THK, R0.5 (×4).
- Same tolerance block: ±0.1 general, ±0.3° angular.

**What Sherman specifies for rolls.**
- Diameter Ø, which reads as outer, ±0.1 (i.e. R ±0.05). On these drawings the arc is defined only by the angular positions of its ends relative to the centrelines, ±0.3°.
- Measured against these (registration-free), the scans are out of tolerance:
  - 49125000: Ø_out 77.40 against 77 ± 0.1;
  - 28870000: Ø_out 62.0 / 62.6 against 63 ± 0.1;
  - arc ends: 8–17° short against ±0.3°.
- No other drawing on disk shows a roll. The `test_files` and other `uploads` drawings, and the e-mail PDFs, are press-brake parts.

## 8. Eye checks and blind review (per region)

I checked every overlay (`png/*_roll0_overlay.png`) and every per-scan figure (`png/<scan>.png`).

A fresh subagent then reviewed the 9 overlays **blind**: titles were cropped off, it got only the images and a legend, and none of my numbers. It agreed with my verdict on 7/9.

| Region | My verdict | Blind verdict | Notes |
|---|---|---|---|
| 51101001_01 | OK | OK | Arc on both skins; run-outs at both free edges; flattened lead-in visible at the start edge. |
| 49125000_A00 | OK | OK | Start at the free edge, end at the joggle start. CAD R +0.23 and 7.6° longer. |
| 28870000 FULL | OK | OK | Start where the arc turns into the joggle (the heat map turns there); end 1–4° short of the free edge (known bias). |
| 28870000 SCAN_1 | OK | OK | Start at the tangent into the flange. |
| 49188000_A | OK | OK | Run-outs at both reverse bends. LARGE_RADIUS_BEND. |
| 41018000_H | **PROBLEM** | **PROBLEM** | The run-outs sit where the curvature ends in each portion. But the stepped two-width part, the cone and the tighter wide-portion ends mean a single R/sweep is not valid. Reported per portion (§3). |
| 45351000_B | **PROBLEM** | **PROBLEM** | The primary start line (71 stations) is right. In the perforated half (s > 60 mm), holes cut the arc into undetectable pieces, and a 7-station segment sits ~25° early (the arc continues past it). Clean-station sweep 175.4°. The stripe at s = 0 is the welded D-bulkhead. |
| 47479000_C | OK | PROBLEM | **Disagreement, resolved.** The reviewer saw a purple tick mid-arc at s = 0. That was the raw station run-out of a station whose arc a large hole cuts; the robust line fit excludes it. The section panel now draws the fitted line (thick) and shows the raw station value as an open circle, and was re-rendered. The ~3 mm CAD radius gap is real (§6). |
| 48565022_A | OK | PROBLEM (marginal) | **Disagreement, minor.** The start may be 1–3° early (< 0.7 mm at R 13). Stray end dots at the curl tip. LARGE_RADIUS_BEND, so not a roll for reporting. |

Methods that were changed because a check failed:
- run-out lines are fitted robustly in segments;
- the arc is extended across scan gaps;
- run-outs use the union of the two skins at edges;
- the half-maximum core must be contiguous (it had stretched arcs into flats);
- the radius and rms are re-checked after the end refinement (this removed the 50546000 false rolls);
- the CAD is anchored at its bend end;
- section ticks are drawn from the fitted line.

Every change was re-run on synthetic data. The final table above is from the last code.

## 9. Readiness recommendation

**Ready to report as review-only** (no pass/fail), on smooth single-radius rolls with sweep ≥ 60° and both skins or a D1 thickness:

| Measurement | Report tolerance (honest) | Basis |
|---|---|---|
| R_inner, R_outer | ±0.15 mm smooth rolls; ±0.6 mm faceted / bump-rolled / t < 0.6 mm | synthetic ≤ 0.02 mm; cross-check 0.01–0.18 mm (smooth), 0.2–0.6 mm (faceted) |
| Thickness (two-skin) | ±0.05 mm | two-skin vs D1 within 0.005 mm on 7/9 scans (0.12 mm on 48565022); CAD Δ −0.05 to +0.02 |
| Sweep between run-outs | ±1.5° where ends are tangent flats ≥ 10 mm or free edges; flag otherwise | synthetic 0.05–0.4° RMS, free edge −1.5°, short flats +2.6 to +7° |
| Chord / opening at the run-outs | ±0.6 mm | synthetic 0.08 mm; real u_total 0.04–1.9 |
| Taper / cone flag | ±2 mm/m (as a flag ≥ 5 mm/m) | synthetic 0.7 mm/m RMS (1.5 at 30° sweep); slab cross-check within 3 mm/m |
| Axis direction | ±0.2° | synthetic max 0.18° |
| Coverage | as is | |

Each of these suits the CAD drawings' ±0.1 mm on Ø only as a coarse review: ±0.15 mm on R is ±0.3 mm on Ø. The 3 mm and 8–17° deviations found are far outside both. To judge ±0.1 on Ø, the method floor needs to come down on real parts, for example with per-skin facet modelling, and be shown on a gauge-measured part.

**Not ready** (research only):
- Run-out line positions against CAD. They need the E3 registration pose. The intrinsic line quality is good on clean ends (0.1–0.5 mm), but the lines are 1–5 mm rms where the arc ends at holes or in a stepped outline.
- Twist: there is no independent check, and the real values (3–11 °/m) have u of 0.6–7.
- Flat-spot classification: the threshold detector found none of the real lead-in flats (≤ 0.3 mm) that the heat maps show. The heat map is the useful review artefact today.
- Multi-portion or multi-radius parts like 41018000: the tool can report per-portion values but does not yet segment portions automatically.
- Arcs on perforated regions (45351000).

**Next.**
1. **CAD comparison with registration.** Feed the E3 pose into `compare.py` to compare axis position and run-out lines, using the solver's cylindrical B-rep faces for the nominal R and sweep. Run 28870000, 47479000 and 49125000 first.
   - Add the 49125000 STEP from uploads/05d7cd09 to the solver's DEV list; it is currently missing from its out/.
   - Get the 51101001 drawing (02.001.6301-01.pdf) and the STEP files for 41018000, 45351000 and 51101001.
2. **Contract amendment.** Add a `ROLL` characteristic as **review-only**, with these fields:
   - R_inner and R_outer, t, sweep, chord;
   - run-out lines with end type;
   - taper and the cone flag;
   - coverage and a per-portion list;
   - classification Ri > 10 t, otherwise a LARGE_RADIUS_BEND handed to the bend lane.

   No pass/fail until a gauge-measured validation exists.
3. **Method items.**
   - Segment along the length into portions (41018000).
   - Measure per-zone curvature for pre-bent or flat lead-in ends, reported as a zone radius with a threshold tied to the drawing tolerance.
   - Improve gap bridging on perforated shells.
   - Correct the free-edge −1.5° bias.

## 10. Files

**Code:** `research/zeroshot/roll_measure/`

| File | Contents |
|---|---|
| `geom.py` | circle, concentric and cylinder fits |
| `measure.py` | detection and measurement |
| `run.py` | per-scan job; `classify` |
| `viz.py` | per-scan figure |
| `overlay.py` | eye-check overlay sheet |
| `compare.py` | scan against `cad_roll_nominal.json` |
| `synth.py`, `synth_eval.py` | synthetic generator and accuracy evaluation |
| `crosscheck.py` | RANSAC + least-squares cylinder |
| `screen.py` | DEV pool screen |

**Tests:** `tests/test_prod_roll_measure.py`.

**PNGs in `png/`** (≈1200 px wide, ≤ 410 KB):
- per-scan figures `<scan>.png`: scan with axis and run-outs, radius along the length, one section with its fitted circle;
- overlay sheets `<scan>_roll0_overlay.png`: iso, end-on, three sections, and heat maps against the cylinder, the cone and the CAD;
- `screen_nonhit_sample.png`.

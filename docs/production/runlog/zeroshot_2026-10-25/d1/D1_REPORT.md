# D1 two-skin diagnostic and prototype (2026-10-25)

This is consumed development evidence on GOLD_SET_V1, which includes the 11 Gold11 relabel scans. None of it is an unseen-family claim.

**Bottom line**
- The two-skin hypothesis is true as geometry: the A0 lane does put its lines on the inner side of the sheet.
- It costs **zero** line-F1 on this dev set. Every inner-side offset is under the 4 mm gate.
- Snapping lines to the outer skin cuts the position error of matched lines from 1.17 mm to 0.05 mm, but F1 stays at 0.402.
- Feeding the mid-surface into A0 makes F1 worse: 0.402 → 0.332.
- D1 is not where the missing F1 is. 86 % of A0 false negatives are bends with no candidate line at all.

## Scope, custody and method
- **Scans.** All 58 GOLD_SET_V1 scans (`gold_set_v1/case_list_gold_v1.json`); 50 of them are line-scored with existing predictions.
  - Before any scan was opened, each one was checked with `research/zeroshot/d1_two_skin/custody.py`. That guard wraps `bench_labeling/custody.check_path` and adds N001–N016, the 10 EHO4 part numbers, `*_SEALED*`/HOLDOUT10/rhino-preview, and the INVENTORY custody class.
  - 48 scans are DEV_CONSUMED and 10 are CONSUMED_DEVCHECK. None is in TEST-B (`SPLIT_FROZEN.json` was checked). For the 5 STL twins, only their prediction JSON was read; the identical PLY mesh was used for geometry.
  - Note for the planner (OD-3): `sc_eaa89c6d1f1f`, `sc_152f1b1aacb2`, `sc_c623d2e45bff`, `sc_9ef2cf4e52cb` and `sc_7a1773eca9fa` are in GOLD_SET_V1. D1 geometry ran on them, A0 ran on their mid-surface, and their STEP files were parsed for thickness. No E3/CAD-path engine was run on them.
- **Matcher.** The frozen `research/broad_baseline/score.score_case` → `scripts/evaluate_f1_against_manual_gold.match_lines` (5° / 4 mm / 0.6 of both spans, Hungarian, apex locus, raw frame), with frame records from `broad_baseline/broad_frame_records.json`. The baseline reproduces exactly: A0 TP/FP/FN 81/26/215, F1 0.4020. best_f1 120/79/176, F1 0.4848.
- **Baseline no-CAD lane.** A0 strict-blind typed union (`predictions_a0.json` from `broad_baseline/runs/best_f1`). The alternative no-CAD file `f1_zero_shot_decomposition` scores 0.168, so A0 is the best no-CAD lane. best_f1 (A0 ∪ CAD) is reported alongside as a secondary.
- **Code.** All of it is in `research/zeroshot/d1_two_skin/`. No engine or backend file was edited.
  - `two_skin.py`: the module.
  - `cad_thickness.py`: reference thickness from STEP via OCP.
  - `run_d1.py` / `run_all.sh`: the per-scan job, run through `sherman_slot.sh`.
  - `analyze.py`: classes, locus and the snap A/B.
  - `run_a0_mid.sh` / `ab_mid.py`: the mid-surface A/B.
- **Tests.** `tests/test_prod_zeroshot_d1_two_skin.py`: 11 passed. They cover a synthetic bent sheet (thickness mode, loci, t/sin(θ/2) separation, snap, side labels, mid-surface) and the custody refusals.
- **Resources.** Each D1 job takes about 1–2 s and stays under 2 GiB. A0 on the mid-surface took about 2–5 min per scan.

## 1. Error-class counts (A0, no-CAD; 53 scans incl. 3 zero-bend controls; 296+ gold)

Classification uses the frozen match. For each FN, the nearest *unmatched* prediction within 15° / 15 mm is placed in the gold line's measured cross-section. "Inner-skin offset" means it is parallel (≤ 5°), sits inside the part on the bisector, and is no deeper than t/sin(θ/2) + max(1.5 mm, t/2), or lies within that tolerance of an inner / mid locus.

| Class | A0 FN (215) | A0 FP (26) | best_f1 FN (176) | best_f1 FP (84) |
|---|---|---|---|---|
| inner-skin offset (≈ t/sin(θ/2) along the bisector) | **0** | **0** | **0** | **0** |
| duplicate skin line | n/a | **0** | n/a | **0** |
| other | 30 | 26 | 46 | 84 |
| missed entirely (no prediction within 15° / 15 mm) | **185** | n/a | **130** | n/a |

Breakdown of "other":
- **A0 FN:**
  - 25 have offset ≤ 4 mm but fail span or angle;
  - 4 are offset > 4 mm but *lateral*, not along the bisector (`sc_8935b8a8e8aa` B7–B9, lines about 10 mm sideways on ~11° lips; `sc_6eba02723d4b` B4, 2.1 mm *outside* the apex plus 1.5 mm lateral);
  - 1 fails both angle and offset (`sc_3c93a0bf6f97` B1, 4.7 mm lateral).
- **A0 FP:** 22 are near a gold line (span failures) and 4 are spurious.
- **best_f1:**
  - FN: 39 span/angle, 5 offset-not-skin, 2 angle+offset;
  - FP: 38 near gold, 46 spurious (5 of them on zero-bend controls).

The hypothesis is still true geometrically. 100 of the 107 A0 lines get a valid cross-section profile. For 85 of those 100, the nearest locus is the inner virtual sharp (48) or the inner arc apex (37). The median A0 line sits **1.05 mm (0.63 t) inside the outer apex** along the bisector (p10–p90: 0.34–1.65 mm).

Because stock in this set is 0.9–3.2 mm (median about 1.5 mm), that offset never reaches 4 mm. The RESEARCH.md failure regime (t ≥ 2.9 mm at 90°, or acute bends) is not present here. The 4-mm OFFSET failures in this lane are lateral or adjacent-bend misplacements, not skin swaps.

## 2. Gold apex convention

The GOLD_SET_V1.md rule is "on the outer-skin apex of a real formed fold". It is the outer-arc apex, not the OML virtual sharp: the audit moved lines *from* the virtual sharp and the inner skin *onto* the outer-arc apex.

This was measured on the scans. Of 328 gold lines, 322 have a valid profile, and the nearest locus is:

| Nearest locus | Lines |
|---|---|
| outer arc apex | **301** |
| outer virtual sharp | 11 |
| mid sharp | 5 |
| inner apex | 3 |
| inner sharp | 2 |

- Distance from the gold line to the outer apex: median **0.024 mm**; 96.2 % are within 1 mm.
- Bisector offset from the outer apex: median 0.008 mm (p10 0.00, p90 0.02).
- Distance to the outer virtual sharp: median 1.27 mm.
- The 22 lines more than 1 mm from the outer apex are profiler failures, not gold errors. In narrow channels a neighbouring bend about 8 mm away enters the window (`sc_1059a9192687` / `sc_9bd69bc6535c`), plus a few near-flat (~172°) and hem-like (~7°) folds.

**Convention confirmed: outer-skin arc apex.**

## 3. Thickness estimation accuracy

Method:
1. Sample 120k area-weighted surface points.
2. Cast a ray along −n (`RaycastingScene`), accepting a hit only where it leaves the material through an antiparallel face (cos < −0.85), at 0.25–12 mm.
3. Take the smoothed histogram mode, then the median within ±20 %.
4. Confidence = peak share × two-sided coverage × tightness.

The mesh winding was outward on every scan (exit fraction about 1.0).

Results against CAD thickness (STEP B-rep: antiparallel planar-face pairs, cross-checked against coaxial cylinder pairs; the two CAD methods agree on every part checked):

| Measure | Value |
|---|---|
| Scans with a usable STEP | **41** |
| Median absolute error | **0.055 mm** (median relative 3.1 %) |
| p90 absolute error | 0.147 mm |
| Max absolute error | 0.235 mm (15.7 %) |
| Signed bias | −0.046 mm (scan reads thin, −3.4 % mean) |
| Within 0.2 mm | 39 / 41 |
| Within 10 % | 37 / 41 |
| Confidence ≥ 0.8 (31 scans) | MAE 0.055 mm, max 0.19 mm |
| Confidence < 0.8 (10 scans) | MAE 0.104 mm |

The worst scans are `sc_32099fb52638` 1.26 vs 1.5 (confidence 0.52), `sc_cec08dfca9ab` 1.29 vs 1.5 (0.66), `sc_ef0abac61ae7` 1.34 vs 1.5 and `sc_70b39f530f00` 2.81 vs 3.0. Per-scan values are in `d1_analysis.json` → `thickness.per_scan`.

Skin pairing: two-sided coverage is typically 60–80 % of vertices. The A/B side 2-colouring has pair-link consistency 0.9999 on the checked scan, and the synthetic test asserts > 0.95.

## 4. A/B line-F1 (same 50 line-scored scans, frozen matcher)

| Arm | TP / FP / FN | micro F1 | family-macro F1 | matched offset median |
|---|---|---|---|---|
| **A0 baseline** | 81 / 26 / 215 | **0.4020** | 0.3498 | 1.17 mm |
| A0 + outer-apex snap, gated (shift ≤ t/sin(θ/2) + 1 mm) | 81 / 26 / 215 | **0.4020** | 0.3498 | **0.049 mm** |
| A0 + ungated apex snap (≤ 8 mm) | 79 / 28 / 217 | 0.3921 | 0.3464 | 0.048 mm |
| A0 + ungated snap + two-skin dedupe | 79 / 25 / 217 | 0.3950 | 0.3471 | 0.048 mm |
| A0 + outer virtual-sharp snap | 79 / 28 / 217 | 0.3921 | 0.3464 | 1.32 mm |
| A0 on the D1 **mid-surface** cloud | 65 / 31 / 231 | **0.3316** | 0.2440 | 1.34 mm |
| A0 mid-surface + gated snap | 65 / 31 / 231 | 0.3316 | 0.2440 | 0.040 mm |
| best_f1 baseline | 120 / 79 / 176 | 0.4848 | 0.4694 | 1.10 mm |
| best_f1 + gated snap | 120 / 79 / 176 | 0.4848 | 0.4694 | 0.038 mm |

- **Determinism control.** A0 was re-run on 5 original scans and was identical (11/3/27 both times). The mid-surface delta is not run noise.
- **Zero-bend controls.** Every arm has 0 FP lines.
- **Mid-surface.** The cloud kept the two-sided vertices collapsed by d/2 and the one-sided ones moved by t/2, and dropped the edge-band points. A0 on it:
  - returned NO_PREDICTIONS on 21 of 53 scans (the scan baseline: 13, plus 1 LANE_FAILED);
  - gained TPs on 10 scans (+15) and lost them on 18 (−31).
  - A0's surfel and atom thresholds are tuned to raw two-skin density, so the mid-surface is not a drop-in input.
- **Ungated snap.** It lost 2 TPs, on `sc_1059a9192687` and `sc_9bd69bc6535c`. There the profiler latched onto a neighbouring bend about 8 mm away. The gate removes that failure.

## 5. Recommendation

1. **Do not spend more on D1 as an F1 lever for this dev set.** It explains 0 of 215 A0 FNs and 0 of 26 FPs. The F1 budget is MISSED (185 / 215) and span (25). Spend on recall instead: D2 slicing, or the PLAN Z2 flange-graph lane. Do not adopt the mid-surface as A0 input (−0.070 F1).
2. **Keep D1 as infrastructure, and adopt the gated outer-apex snap as a free final placement step.** It is F1-neutral today, cuts the matched position error about 24× (1.17 → 0.05 mm), and puts every lane on the gold convention. That protects F1 once thicker stock (≥ 2.9 mm at 90°) or acute bends appear, where the inner-side offset (median 0.63 t, up to t/sin(θ/2)) would cross 4 mm. Use the gate (shift ≤ t/sin(θ/2) + 1 mm): ungated, it damaged 2 TPs.
3. **Give Z2 / E3-style metrology the module's API:** t with confidence (median error 0.055 mm vs CAD, conf ≥ 0.8 is reliable), per-vertex two-sided / side labels, and `bend_profile` loci.
   - Known limit: `bend_profile` aggregates the whole span inside a 14 mm / 5t window. Two bends less than about 10 mm apart (narrow channels, hems) can capture the wrong arm pair, so it should be seeded with Z2's flange pair instead of the histogram.

## Files
- Per-scan records: `d1/scans/sc_*.json` (thickness, skins, gold-line profiles, lane-line profiles).
- `d1/d1_analysis.json`: classes per case, gold locus per line, thickness per scan, snap A/B.
- `d1/ab_mid.json`: mid-surface A/B and the determinism control.
- `d1/a0_mid_runs/`: A0 predictions and run records on the mid-surface and control scans.
- `d1/baseline/`: frozen-scorer rescore of the lanes.
- Reproduce:

```
research/zeroshot/d1_two_skin/run_all.sh 2 d1/scans
.venv/bin/python -B research/zeroshot/d1_two_skin/analyze.py --scans d1/scans --out d1
sherman_slot.sh .venv/bin/python -B research/zeroshot/d1_two_skin/ab_mid.py --a0mid d1/a0_mid_runs --scans d1/scans --out d1/ab_mid.json
```

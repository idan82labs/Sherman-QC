# Angle check: is the zs_lane_v1 bend angle true? (DEV, CAD-path cohort, 2026-10-25)

> **Ledger.** DEV only, consumed data. Every scan went through `gold_guard.require_runnable(cad_path=True)` (the 2 OD-3 DEV-CHECK cases had already been refused when the cohort was built). Gold was read by the frozen scorer, plus the `incl` value in each gold line's notes. No engine run and no engine edit: everything comes from stored outputs. The only new geometry is the cross-sections of the worst bends, cut from the raw DEV scans.

## Answer

**Yes. On ordinary bends the zs angle is true to about a tenth of a degree.**
- Against E3 (the metrology authority), median |zs − E3| is **0.06°** and P90 is **0.28°**, with no bias (+0.02°).
- On the bends where E3 publishes a firm PASS or FAIL, 99.5 % agree within 1°.
- A visible 90° bend reads near 90°. Example, sc_03395097cd96 B1 (CAD 90°): **zs 89.71°, E3 89.74°, gold 90.11°**. Its sibling bend B2 reads zs 90.26°, E3 89.86°, gold 90.17°.

**Two exceptions:**
1. **Joggles** (10839000, 22 bends). These are not measurable at this scale, and zs reads them 4–14° more open than E3.
2. **Six bends where E3 itself is wrong.** zs, gold and an independent cross-section agree with each other there. These are false FAIL/REVIEW readings from E3, not false PASSes.

## Method

- **Cohort.** The 51 E3-eligible DEV scans. The data comes from:
  - `cadpath_cohort_base.json`, built by `cad_path/tools/build_cohort.py` on the e3-1.12.0 re-run;
  - the E3 `result.json` from `cadpath_runs/base`.
- **Exclusions.** 2 E3 refusals were excluded:
  - mirrored sc_2d3e4a623a31;
  - unformed sc_3bde9996f14d.
  - There is no registration on these, so no CAD bend can be tied to a line.
  - The zero-bend control sc_f99d883f2754 has no lines.
- **zs.** The frozen CLI output, `integration/dev_cli/zs/<scan>/eval_lines.json`, field `incl`.
  - The out-of-fold selection (p ≥ 0.5) gives the same numbers to 0.01° (it is in the JSON, key `oof`). The angle is a geometric output; it is not learned.
- **Matching zs to gold.** The frozen matcher (5° / 4 mm / 0.6, apex, raw). It gives **294 zs↔gold matches on 48 scans and 32 families.**
- **Matching gold to the CAD bend.** One-to-one, using in order:
  1. the E3 published OUTER_APEX line (frozen match): 273;
  2. E3's decoded-stage CAD locus mapped to raw through E3's registration (5° / 6 mm / 0.5): 15.
  - That gives **288 bends with a CAD bend.** 6 matched zs lines have no CAD bend, and 4 bends have E3 INCLUDED_ANGLE NOT_MEASURED (sc_c0ce5ecd2ba3, twin-ambiguous). So **284 bends have an E3 angle.**
- **Angle sources.**
  - (a) CAD = `nominal_included_deg`.
  - (b) E3 = INCLUDED_ANGLE `actual`.
  - (c) gold = `incl X deg` in the label notes. This is the labelling tool's own patch-fit angle, so it is not fully independent of the scan.
- **Classes.**
  - ≈90 means |nominal − 90| ≤ 5°.
  - acute is below that, obtuse above.
  - joggle means two bends that share a panel, with opposite directions, equal sweep ≤ 60° and axes ≤ 10 t apart. Only 10839000 (145° nominal) and 37126000 (135°) qualify.
- **Error.** Error = zs − reference. A positive bias means zs reads more open, toward 180°.

## Error table (degrees)

| Slice | vs | n | median \|err\| | P90 | max | bias (mean / median) | within 1° |
|---|---|---|---|---|---|---|---|
| **All 288** | **(b) E3** | 284 | **0.07** | 0.52 | 237.4 | −1.43 / +0.02 | 92.6 % |
| | (a) CAD | 288 | 0.45 | 1.99 | 23.1 | +0.18 / −0.12 | 73.6 % |
| | (c) gold | 288 | 0.09 | 0.47 | 26.0 | +0.78 / −0.02 | 94.1 % |
| | E3 − CAD (the real bend deviation) | 284 | 0.49 | 1.95 | 239.0 | | 74.6 % |
| **Excl. 6 E3-wrong** | **(b) E3** | 278 | **0.06** | 0.44 | 14.0 | +0.38 / +0.02 | 94.6 % |
| | (a) CAD | 282 | 0.45 | 2.00 | 23.1 | +0.17 / −0.13 | 73.8 % |
| | (c) gold | 282 | 0.09 | 0.46 | 26.0 | +0.79 / −0.02 | 94.0 % |
| **Excl. E3-wrong and joggles** | **(b) E3** | 256 | **0.06** | **0.28** | 2.85 | +0.04 / +0.02 | **98.8 %** |
| | (a) CAD | 260 | 0.44 | 1.68 | 23.1 | −0.24 / −0.17 | 76.9 % |
| | (c) gold | 260 | 0.09 | 0.40 | 13.2 | −0.12 / −0.02 | 98.1 % |
| E3 PASS/FAIL bends only (excl. E3-wrong) | (b) E3 | 209 | 0.05 | 0.22 | 1.67 | +0.04 / +0.02 | 99.5 % |

**Reading the table:**
- zs agrees with E3 and with gold an order of magnitude more closely than either agrees with CAD.
- The CAD column measures the part, not zs. E3 − CAD has the same size (median 0.47°, P90 1.8°), because the real parts are over- or under-bent.
- The 23.1° CAD maximum is sc_6e5d5a2f60f5 B1. It is a real mis-bend (90° nominal): zs 66.9°, E3 68.6° FAIL.

### By nominal class (excluding the 6 E3-wrong bends; the "all" values are in the JSON)

| Class | n | vs E3: median / P90 / max / bias | vs CAD: median / P90 / max / bias | vs gold: median / P90 / max | E3 − CAD median / P90 |
|---|---|---|---|---|---|
| ≈90° | 223 (219 E3) | 0.05 / 0.28 / 2.85 / +0.04 | 0.40 / 1.53 / 23.1 / −0.41 | 0.08 / 0.32 / 13.2 | 0.39 / 1.69 |
| acute (60°) | 8 | 0.06 / 0.36 / 0.50 / +0.10 | 1.03 / 1.77 / 1.98 / +1.08 | 0.05 / 0.17 / 0.24 | 1.02 / 1.77 |
| obtuse (100–150°) | 29 | 0.12 / 0.33 / 0.89 / +0.04 | 0.79 / 1.97 / 5.26 / +0.67 | 0.18 / 0.53 / 0.90 | 0.79 / 1.91 |
| **joggle** | 22 | **4.04 / 10.6 / 14.0 / +4.37** | 5.32 / 11.3 / 12.7 / +5.10 | **17.7 / 23.8 / 26.0** | 0.87 / 3.20 |

Notes on the classes:
- The acute class has 10 bends in all. 2 of them are E3's 299° readings, see worst cases.
- In the joggle class, 37126000 alone (2 bends) gives zs − E3 of +0.8 and +0.6. The large errors all come from 10839000.

### By sheet thickness (excluding E3-wrong)

| t (mm) | n vs E3 | families | vs E3: median / P90 / max / bias | vs CAD: median / P90 / max | vs gold: median / P90 |
|---|---|---|---|---|---|
| 1.0 | 29 | 23553000, 48555000 | 0.11 / 0.31 / 0.50 / +0.03 | 0.75 / 1.56 / 1.98 | 0.08 / 0.26 |
| **1.5** | 135 | 11 families incl. 10839000 (joggles), 42288005 | 0.06 / **1.52** / 14.0 / +0.75 | 0.71 / 3.27 / 12.7 | 0.10 / 0.65 |
| 1.6 | 1 | 44211000 | 0.12 | 0.59 | 0.14 |
| 2.0 | 78 | 9 families (15651004 … 48991006, 50597000) | 0.04 / 0.21 / 1.67 / 0.00 | 0.22 / 0.83 / 23.1 | 0.06 / 0.35 |
| 2.032 | 2 | 37126000 (joggle) | 0.67 / 0.75 / 0.77 | 4.44 / 4.78 / 4.87 | 0.36 / 0.46 |
| 3.0 | 26 | 49024000, 49207000, 50757000 | 0.08 / 0.48 / 0.89 / +0.07 | 0.37 / 1.05 / 1.69 | 0.13 / 0.52 |
| 3.175 | 1 | 47266000 | 0.07 | 0.45 | 0.03 |
| 4.0 | 2 | 23317000, 49783000 | 0.04 | 0.59 | 0.21 |
| 6.0 | 4 | 46127000, 46128000 | 0.06 / 0.41 / 0.53 | 0.20 / 0.31 | 0.25 / 0.32 |

**Thickness by itself does not degrade the angle.** The 1.5 mm tail is the 10839000 joggles; without them 1.5 mm behaves like the rest.

## Same side of a ±1° tolerance as E3 (|angle − nominal| ≤ 1°)

The recipe tolerance on every INCLUDED_ANGLE here is ±1°.

| Slice | n | same side | both in | both out | E3 in / zs out | E3 out / zs in |
|---|---|---|---|---|---|---|
| All | 284 | **265 (93.3 %)** | 201 | 64 | 11 | 8 |
| Excl. E3-wrong | 278 | 263 (94.6 %) | 201 | 62 | 11 | 4 |
| Excl. E3-wrong and joggles | 256 | **246 (96.1 %)** | 193 | 53 | 6 | 4 |
| ≈90° / acute / obtuse / joggle (excl. E3-wrong) | 219 / 8 / 29 / 22 | 95.9 / 100 / 96.6 / 77.3 % | | | | |

The 15 disagreements that remain after excluding the E3-wrong bends:
- **8 are boundary cases.** Both readings sit within 0.35° of the ±1° edge, and E3 itself said REVIEW on 7 of them. Examples: zs 91.15 vs E3 90.96; zs 89.17 vs E3 88.52 FAIL on sc_1f1d06c24e9f B5.
- **5 are joggles.**
- **2 are on sc_07062eee17aa** (bends E3 calls UNEXPECTED_FORMED):
  - B8: E3 86.7° vs zs 89.5°, gold 89.8°, section 89.9°. E3 looks wrong here.
  - B10: E3 89.2° vs zs 91.3°, gold 91.75°, section ≈93°.
- **What the 6 "E3 in, zs out" cases mean.** zs would never turn an E3 PASS into a firmer verdict: zs is display-only. They show that a zs angle near the edge is not a substitute for E3's uncertainty band.

## The worst disagreements (zs vs E3), with a cross-section

**How the sections were cut** (`tools/section.py`):
- The raw scan is sliced perpendicular to the zs line at 5 stations, with a ±0.75 mm slab.
- Sequential RANSAC fits two flange lines through the apex, within 3–20 mm of it.
- The tool reports the median angle over the stations.
- It is a crude, independent check, good to about ±1.5°. The control sc_03395097cd96 B1 gives 91.1° against zs 89.7° and E3 89.7°.
- It cannot resolve the joggle webs.
- The plots are in `sections/`.

| # | Bend (CAD) | zs | E3 (decision) | gold | section | Verdict |
|---|---|---|---|---|---|---|
| 1 | sc_186463746daa B2 / B14 (23553000, 60°, t 1.0) | 61.6 / 61.8 | **299.0 / 298.8** (FAIL) | 61.35 / 61.18 | 60.4 / 61.3 | **E3 wrong.** 299° = 360 − 61: E3 measured the reflex side. E3 state UNEXPECTED_FORMED. False FAIL. |
| 2 | sc_6e5d5a2f60f5 B4 (48991006, 90°, t 2.0) | 90.0 | **152.9** (FAIL) | 90.1 | 90.5 (all 5 stations 90.4–90.6) | **E3 wrong.** E3 state PARTIALLY_FORMED; the bend is fully formed at 90°. False FAIL. This is the same bend the seeded `add` arm flipped to PASS at 90.1° (CAD_PATH §4). |
| 3 | 10839000 joggles: sc_08686dc1d5d4 B3/B4, sc_cec08dfca9ab B1/B2, sc_71cf0b216f43 B1–B4 (145°, t 1.5) | 150–158 | 142–149 (REVIEW) | **130–135** | n/a | **Not measurable.** The section shows an S-shaped offset of about 2.5 mm with no straight web: the "web" is all bend radius. Every method returns its own number (zs +6 to +14°, gold −10 to −15° vs E3). E3 correctly withholds a verdict (REVIEW). |
| 4 | sc_2233372b8df7 B2 / B3 / B8 (42288005, 90°, t 1.5) | 89.9 / 89.0 / 90.1 | **81.7 / 80.4 / 81.1** (REVIEW) | 90.2 / 89.0 / 89.6 | 89.9 / 88.1 / 90.1 | **E3 wrong.** These are 90° bends on a stepped profile with a short leg of about 12 mm between bends (see `sections/sc_2233372b8df7_B5.png`). E3's station planes are probably reaching into the next bend. |
| 5 | sc_6e5d5a2f60f5 B1 (48991006, 90°) | 66.9 | 68.6 (FAIL) | **80.1** | 68–83 along the length (median 80) | **The bend is genuinely bad and the flange is bowed** (section plot). zs and E3 agree on a mis-bend, about 21–23° off nominal. Gold, which fitted a different flange extent, reads 80. The angle is ill-defined because the flange is not flat. |

**The 6 E3-wrong bends** are cases 1, 2 and 4 (keys in the JSON, `slices.e3_wrong_keys`). There:
- zs, gold and the section agree within about 1.5°;
- E3 is 8–237° off.

All 6 are E3 FAIL or REVIEW, so none is a false PASS. They are, however, false rejections, and they are worth an E3 ticket:
- an included-angle convention flip (reflex side);
- a fully formed bend decoded as PARTIALLY_FORMED;
- station planes on short legs.

## Takeaways

- **zs tracks E3 closely.**
  - On ordinary bends (≈90°, acute, obtuse; all thicknesses 1–6 mm), the zs included angle tracks E3 to a median of 0.06° and P90 of 0.3°, with no bias.
  - It is on the same side of ±1° as E3 96 % of the time. The misses sit on the tolerance edge.
- **CAD error is the part's own deviation.** zs − CAD (median 0.45°, P90 2°) is almost entirely real over- or under-bend, the same as E3 − CAD. The one-sided under-bend (+7 to +13°) seen on the 145° bends in CAD_PATH is the joggle class here: E3 reads those bends near nominal, and the excess is zs's reading.
- **Do not show zs joggle angles as numbers.** On 10839000 zs reads about 5° (up to 14°) more open than E3, and the gold disagrees with both. A joggle should be displayed as "joggle, angle not measured".
- **Independent check on E3.** Where zs and E3 differ by more than about 5° on a non-joggle bend, E3 was the one in error in every case seen here (6 of 6, plus a probable seventh, sc_07062eee17aa B8). A zs–E3 angle disagreement could be used as a REVIEW trigger on E3 FAILs (display/REVIEW only; it never creates a PASS).

## Files

- `ANGLE_CHECK.json`:
  - `cli` (primary) and `oof` (secondary): per-bend records (`records`), summaries, skipped scans;
  - `slices`: the tables above;
  - `sections`: the cross-section angles.
- `tools/angle_check.py`: matching and statistics. Usage: `COHORT.json E3_DIR OUT.json`.
- `tools/summarize.py`: the slices above.
- `tools/section.py`: the cross-sections. Plots are in `sections/*.png`.

# FE1 adjudication: final held-out evaluation, 2026-10-28

The adjudicator (a third agent, neither marker) wrote this file before any engine run on these scans. No engine, lane, pool, E3, CAD or baseline output was read; none exists for FE1.

- **Protocol:** `PREREG.md` §4 (sha `c50f58e0…`). The model is the N1 adjudication (`zeroshot_2026-10-25/tools/adjudicate_n1.py`).
- **Matcher:** the frozen `research/broad_baseline/score.py score_case` (5° / 4 mm / 60 % of both spans, one-to-one).
- **Gold rule:** GOLD_SET_V2, as quoted in `packets/MARKER_INSTRUCTIONS.md`.
- **Records:**

| What | Path |
|---|---|
| Tool | `tools/adjudicate_fe1.py` (`main`, `labels-and-freeze`, `ledger-note`) |
| Gold-ledger FROZEN appends | `tools/gold_freeze_fe1.py` |
| Batch summary | `adjudication/ADJUDICATION_FE1.json` |
| Per item | `adjudication/<gid>/adjudication.json`, `adjudication/<gid>/final/lines/lines.json` |
| Adjudicator re-runs | `adjudication/<gid>/{pairs.json,lines/}` |
| Raw-scan sections and station checks | `adjudication/evidence/` |
| Gold-root manifest | `adjudication/GOLD_ROOT_MANIFEST.json` |

## 1. Canary QC (PREREG 4.4)

The hidden canary is `gb_9528f8`, a consumed DEV scan. Its gold of record is GOLD_SET_V2 `ai_bend_lines_gold_v1/3af35cca6ebedaec495ab663.json`, with 4 lines.

| Marker | Line-F1 vs gold of record | Matches (offset, angle) |
|---|---|---|
| A | **1.000** (4/4) | all ≤ 0.01 mm, 0.00° |
| B | **1.000** (4/4) | all ≤ 0.01 mm, 0.00° |

Both markers pass the ≥ 0.95 rule, so no batch is redone. No canary label was written, and the canary was not frozen.

## 2. A vs B agreement, before adjudication

Each item was scored with `score_case`, taking marker A as gold and marker B as prediction.

- No matched pair lies outside 1.5 mm / 2°: 165 of 165 pairs are inside.
- B's lines are a subset of A's on every line scan.
- The one exception is `gb_437879`, where A drew 0 lines and B drew 2.

| Set | TP / FP / FN | Micro line-F1 |
|---|---|---|
| 27 scored scans, line scans only | 161 / 0 / 15 | **0.9555** |
| 27 scored scans, B's lines on the zero-bend (by A) scan counted as FP | 161 / 2 / 15 | **0.9499** |
| All 28 items (canary included) | 165 / 2 / 15 | 0.9510 |

- **Macro item-F1 (27 scored):** 0.923. Each zero-bend item scores 1 if both markers drew no lines and 0 otherwise.
- **Zero-bend items:** the markers agree on `gb_61b6da` and `gb_cdecd9`, and disagree on `gb_437879`.
- **Hems are not in this agreement.** The `lines` tool refuses parallel planes, so neither marker could put a hem line in `lines.json`. Both markers did flag the hems on `gb_1baa22`, `gb_d11e33`, `gb_231669` and `gb_ac2239` in their notes.

## 3. Summary table

The final gold has **186 lines over 27 scans**.

| gid | Sub-set | A | B | A/B F1 | Final | Source | Class | Decisions |
|---|---|---|---|---|---|---|---|---|
| gb_04738b | N | 6 | 6 | 1.000 | 6 | A | bent | stepped end tabs kept separate (both) |
| gb_138090 | N | 4 | 2 | 0.667 | **4** | A | bent | 2 short ears ACCEPT_A |
| gb_1baa22 | N | 2 | 2 | 1.000 | **3** | ADJ | bent | open hem MISSING_ADD |
| gb_1f3fe0 | N | 4 | 2 | 0.667 | **4** | A | bent | 2 short ears ACCEPT_A |
| gb_231669 | EHO | 12 | 9 | 0.857 | **14** | ADJ | bent | 3 lanced tabs ACCEPT_A; box corners not bends; 2 hems MISSING_ADD |
| gb_42dd99 | H10 | 4 | 4 | 1.000 | 4 | A | mixed | rolled facets not bends |
| gb_437879 | H10 | 0 | 2 | 0.000 | **2** | B | bent | joggle, 2 folds of ~14° (ACCEPT_B) |
| gb_44ddfb | N | 4 | 4 | 1.000 | 4 | A | bent | |
| gb_47b48f | H10 | 2 | 2 | 1.000 | 2 | A | bent | |
| gb_61b6da | N | 0 | 0 | 1.000 | 0 | A | rolled | zero-bend |
| gb_68cd06 | N | 18 | 17 | 0.971 | **20** | ADJ | bent | 14 lanced tabs separate (both); end tab B18 ACCEPT_A; 2 end hems MISSING_ADD |
| gb_8ccb40 | H10 | 2 | 2 | 1.000 | 2 | A | bent | |
| gb_8df897 | H10 | 4 | 4 | 1.000 | 4 | A | bent | |
| gb_959f4a | N | 17 | 17 | 1.000 | **19** | ADJ | bent | 14 lanced tabs separate (both); 2 end hems MISSING_ADD |
| gb_a0538b | N | 4 | 4 | 1.000 | 4 | A | bent | |
| gb_a2dd83 | EHO | 21 | 19 | 0.950 | **21** | A | bent | inset end-flange pieces B12/B13 are formed: ACCEPT_A |
| gb_a64e51 | H10 | 4 | 4 | 1.000 | 4 | A | bent | |
| gb_ac2239 | EHO | 12 | 9 | 0.857 | **14** | ADJ | bent | 3 lanced tabs ACCEPT_A; box corners not bends; 2 hems MISSING_ADD |
| gb_b21bad | N | 6 | 6 | 1.000 | 6 | A | bent | stepped end tabs kept separate (both) |
| gb_b3ae40 | N | 9 | 9 | 1.000 | 9 | A | bent | ladder web: 1 line per side; apex note NC2 |
| gb_b97c8e | N | 9 | 9 | 1.000 | 9 | A | bent | ladder web: 1 line per side; apex note NC2 |
| gb_bbc3a8 | H10 | 4 | 4 | 1.000 | 4 | A | bent | louvres not bends (confirmed) |
| gb_c6e707 | H10 | 2 | 2 | 1.000 | 2 | A | bent | |
| gb_cdecd9 | N | 0 | 0 | 1.000 | 0 | A | rolled | zero-bend |
| gb_d1093e | EHO | 22 | 20 | 0.952 | **20** | B | bent | inset pieces B21/B22 are sharp corners: REJECT |
| gb_d11e33 | N | 2 | 2 | 1.000 | **3** | ADJ | bent | open hem MISSING_ADD |
| gb_f63c89 | H10 | 2 | 2 | 1.000 | 2 | A | bent | |
| *gb_9528f8* | *canary* | 4 | 4 | 1.000 | *(4, no label)* | | | |

Notes on the table:
- **Sub-set codes:** N = N001–N014, EHO = EHO4, H10 = HOLDOUT10.
- **Source codes:**
  - **A** and **B** mean the final gold is that marker's tool output, used unchanged.
  - **ADJ** means the adjudicator re-ran `lines` on the shared prep with marker A's ok pairs plus the hem constructions. The A lines came out bit-identical, with a maximum endpoint difference of 0.
- **Item-level fields, for every item:**
  - `label_complete` is true.
  - The scan is complete, not partial: both markers saw the whole part in every view.
  - No item is excluded.
- **Zero-bend scans:** `zero_bend` is true for `gb_61b6da` and `gb_cdecd9`, both rolled.

## 4. Decisions, with the rule applied

Each decision applies an existing frozen convention, cited in brackets. Section 5 lists the decisions where no existing convention settled the case.

### 180° hems: count, as 10 lines on 6 scans

**Hems count, as one line each.**
- [D0 `gb_44b168`: "Both count the closed 180 deg hem; a formed fold >=10 deg counts under GOLD_SET_V2".]
- A hem is a formed fold of ~180°. It is not an edge, an emboss or a rolled kink.

**How the line is built.** The tool refuses parallel planes, so each hem line uses the **D0 tangent-band construction**:
- Two `plane_from_points` sides are seeded on the outer hem arc, at ±45° about the crown. Their apex therefore lands on the crown.
- The extent is the crown extent measured on the raw scan.
- **No manual line was used.** The gold is tool output, as D0's gold was.

**How the crown was measured.**
- The adjudicator took raw-mesh sections at 1 mm stations along a crown axis, refined until the axis was robust (`evidence/hem2.py` and `hem2_out.json`).
- Section plots are in `evidence/sections/*_v2.png`.
- Every label line was checked against the measured crown (`evidence/hemcheck.json`):
  - offset along the outward direction: −0.01 to −0.20 mm;
  - distance from the crown line, end to end: ≤ 0.80 mm;
  - axis angle: ≤ 0.77°.

| gid | Hem | Stack / crown length | Marker view |
|---|---|---|---|
| gb_1baa22 | open hem at the short end | 5.0 mm / 19 mm | A (manual) and B flagged it |
| gb_d11e33 | open hem at the short end | 4.9 mm / 19 mm | A (manual) and B flagged it |
| gb_231669 | 2 tongue-edge open hems | 5.4–5.5 mm / 105–106 mm | A (manual) and B flagged them |
| gb_ac2239 | 2 tongue-edge open hems | 5.4–5.5 mm / 106 mm | A (manual) and B flagged them |
| gb_959f4a | 2 closed end hems | 2.1–2.4 mm (t 1.0) / 26 mm | A (manual) flagged them; B called them "end fragments" |
| gb_68cd06 | 2 closed end hems | 1.9–2.0 mm (= 2t) / 24–25 mm | A flagged the patch-10 end as "possible"; nobody marked the patch-9 end |

On `gb_68cd06`, both end sections show the strip folding round a crown and back under itself for ~8 mm.

### Lanced or coplanar tabs separated by bridges: many lines

**`gb_68cd06` and `gb_959f4a`: one line per tab** (14 each, plus `gb_68cd06` B18).
- [Separate tabs make separate lines: T1 (canary-confirmed), T4 `gb_efbb1e`, T5 `gb_0f8d41`, and GOLD_SET_V2 `sc_b76452ea6fa7`, where "two separate collinear folds (different tabs) = 2 bends".]
- Each tab is bent out of its own window and is joined to the strip only along its own hinge. The flat ~6 mm bridges do not fold.
- The windows rule merges one plate interrupted by windows. It does not merge tabs that are disconnected from each other.
- Both markers marked the tabs separately.

**`gb_68cd06` B18: ACCEPT_A.** This is the short tab in the end window.
- B identified the same tab, patches 39/40, as a real bent tab.
- B left it out only because its pairings came out on the wrong skin.

**`gb_231669` and `gb_ac2239` B10–B12: ACCEPT_A (A only).**
- These are lanced tabs in the end wall: ~14–15 mm tabs and a ~38–40 mm slot tab.
- Each stands at ~90° to the wall, hinged on the edge of its own opening (A sweep g7/g8; adjudicator sections `evidence/sections/*A_B1*.png`). Each is its own plate. [T1: "tab 16 bent in the wall window is its own fold".]
- The tab undersides were barely scanned. The tool apex therefore sits ~1.4 mm from the expected outer arc, which is inside tolerance, and the lines keep confidence `uncertain`.
- B did not mention these tabs.

### Box wall-to-wall corners: seams, not bends

**`gb_231669` and `gb_ac2239`: not bends.**
- [T1: a fold is formed only when the section shows a concentric arc; a sharp corner or slit is not a fold.]
- Both markers rejected the corners. B measured their apex offsets at 0.18–0.87 mm, against ~1.6 mm for the formed folds.
- The 12-vs-9 difference between the markers is the three lanced tabs, not corners.

### Short corner pieces of ~7 mm on stepped trays

**`gb_a2dd83` B12/B13: ACCEPT_A, giving 21 lines.**
- The station check on the raw scan (`evidence/st_out.json`) puts the outer envelope at the formed-apex offset:
  - B12: 1.21–1.39 mm over 0–19 mm of its 23 mm, against an expected 1.23 mm;
  - B13: 0.64–1.53 mm.
- A's close-ups show concentric arcs on both skins.
- They are ~7 mm tall, ~23 mm wide flanges, inset ~10 mm from the main end flange. That makes them separate, non-collinear folds.
- B's "0.4 mm" was the tool's envelope apex, pulled down by the last ~3 mm of the line where it meets the corner.

**`gb_d1093e` B21/B22: REJECT (B right), giving 20 lines.**
- The outer envelope is 0.10–0.48 mm from the plane sharp at every station.
- A formed 90° fold of t 1.48 needs at least 0.61 mm even at Ri = 0, and the expected value here is 1.25 mm.
- A's close-up shows the inner skin ending square, with no arc.
- This is a sharp or abutting corner, not a formed fold. [GOLD_SET_V2: only formed folds count.]

**Stepped end tabs, `gb_04738b` and `gb_b21bad`: kept separate.**
- Both markers marked them separately.
- Their fold lines are offset ~7 mm, so they are not collinear folds.

### Short ears

**`gb_138090` and `gb_1f3fe0` B3/B4: ACCEPT_A, giving 4 lines each.**
- [GOLD_SET_V2: a formed fold ≥ 10° between two flat plates. Precedents for short folds: T5 counted 6.5 mm feet; N1 `gb_4dd5ba` counted 5 mm hook lips.]
- The flange overhang turns ~90–100° into an ear, and the bend radius is visible.
- The station check gives an outer envelope of:
  - 1.83–1.95 mm on `gb_138090`, against an expected 2.0 mm;
  - 2.65–4.6 mm on `gb_1f3fe0`, against an expected 2.5–2.8 mm. The ear there is slightly twisted.
- B classed the ear strips as end faces and took no section.

### Windowed ladder web

**`gb_b3ae40` and `gb_b97c8e`: one line per side.**
- [GOLD_SET_V2 windows rule; T2 `gb_d2aa20`, T3 `gb_630bfb`.]
- The web is one connected frame plate: the rungs are joined through both rails, and the windows are closed openings. It is folded from the same rail plate along one straight fold.
- This differs from T4/T5, where the slit tabs were joined only along their own fold.

**Apex.** The station check puts the formed arc at the rungs 1.14–1.25 mm from the plane sharp. Between rungs, the outermost material is the rail-edge corners, at ~0.2 mm. The tool's envelope apex follows that outer material, so the lines are 0.85 mm (`gb_b3ae40`) and 0.93–1.25 mm (`gb_b97c8e`) outside the rung arc apex.
- Both markers' lines are identical.
- The lines were kept, with confidence `uncertain`.
- This choice is new convention NC2.

### Louvres: not bends (confirmed)

**`gb_bbc3a8`: the ~30 louvres are not bends.**
- [T3 `gb_c61b1b`: its frozen gold excludes lanced louvres as emboss-type features. GOLD_SET_V2: embosses are not bends.]
- They are lanced hoods formed to ~24–25°, attached at both ends.
- Both markers rejected them.

### gb_437879: a joggle, 2 lines (ACCEPT_B)

The adjudicator profiled the raw mesh itself (`evidence/k437.log`), with t ~0.55 mm and 0.5 mm stations on both skins.

- The profile has three straight segments.
  - Normal tilt is −5.0 / +9.0 / −5.3° on the top skin and +5.2 / −9.1 / +5.4° on the bottom skin.
  - Within each segment, the heights are linear to ±0.01 mm.
- The transitions are short, ~1–1.5 mm, at u ≈ −2.4 mm and +1.9 mm.
- The end segments are parallel and offset, a Z shape.

So there are **two distinct folds of 14.0–14.5° between flat plates**. The tool's plane fits give 11.9° and 12.3°. Both values are above the 10° floor, and a 10–15° fold here is a distinct fold between flat plates, as GOLD_SET_V2 requires. A joggle counts as 2.

A smooth warp would not produce piecewise-flat skins. A's own reading (−4 / +8 / −5°) also gives ~13° folds.

B's two lines put the apex on the convex skin at each kink, and they are the gold.

## 5. New convention decisions (owner sign-off requested)

| ID | Items | Decision | Why it is new |
|---|---|---|---|
| NC1 | gb_1baa22, gb_d11e33, gb_231669, gb_ac2239 | Open (gapped) 180° hems count as 1 line on the outer crown, built with the D0 tangent-band construction. | D0 ruled only on a closed hem. An open hem is the same formed ~180° fold, so GOLD_SET_V2 counts it. Effect: +6 lines. |
| NC2 | gb_b3ae40, gb_b97c8e | On a fold that exists only at short rungs between large windows, keep the tool's envelope apex (the outermost material, mostly rail-edge corners). Do not use the rung arc apex. The resulting offset is 0.85–1.25 mm. | T4/T6 preferred a station-checked apex where a better full-length line existed. Here no tool construction gives one. The offset is well inside the 4 mm tolerance. |
| NC3 | gb_d1093e, gb_a2dd83 | A formed-fold test with a number. If the outer material is closer to the plane sharp than any formed fold allows ((√2−1)·t at 90°, Ri = 0) at every station, the junction is a corner, not a bend. If it sits at the formed-apex offset over ≥ 80 % of the line, it is a bend. | It turns T1's "concentric arc vs sharp corner" ruling into a number. It rejects d1093e B21/B22 and accepts a2dd83 B12/B13. |
| NC4 | gb_68cd06 | The adjudicator added the patch-9 end hem from its own raw sections, with no third marking. | The earlier MISSING_ADDs (T1, T4) came from a FRESH third marking. The owner may want a confirming third marking. |

The other decision items are settled by the existing conventions cited in section 4:
- whether hems count at all;
- separate tabs;
- box corners;
- short ears;
- ladder line count;
- louvres;
- the 10° floor.

## 6. Gold written and frozen (PREREG 4.6)

**Labels root:** `data/bend_corpus/annotations/ai_bend_lines_final_eval_2026_10_28_v1`.
- It holds 27 labels, written by `write_label` and annotated `claude-vision-ai-v1`.
- It has `provenance/` sidecars carrying both markings, the adjudication record and the render hashes.
- Every asset matched its lines by sha256.
- The manifest is `adjudication/GOLD_ROOT_MANIFEST.json`.

**Assets.**
- For the 9 HOLDOUT10 items, the asset is the original scan, which the bench_labeling custody guard admits.
- For the 18 N/EHO items, the guard refuses the original by name. These use a byte-identical, guard-clean copy, as PREREG 4.6 allows. The copies were staged with `run_allowlist.stage`, are logged in `RUN_ALLOWLIST_ADMISSIONS.jsonl`, have mode 0444 and are 470 MB in total. They live at `output/research/fe1_gold_assets_2026-10-28/<gid>.ply`.

**Gold ledger:** `zeroshot_2026-10-25/gold/GOLD_LEDGER.jsonl`.
- 27 FROZEN appends, seqs 166–192, through `tools/gold_freeze_fe1.py`.
- `gold_guard.verify_ledger` passes with 192 entries.
- The FE1 status is 27 FROZEN.
- `gold_guard check` now admits the staged scans, with basis GOLD_LEDGER_FROZEN.

**Not changed:** `gold_guard.py`, the canary, every past ledger entry, all existing gold, all engine code and all pins.

**Freeze ledger:** a NOTE appended after this file binds this file's sha256, the gold-root manifest, the gold-ledger sha256, the counts, the canary scores, the A/B agreement and NC1–NC4. It comes before any FE1 engine RUN entry. No engine was run.

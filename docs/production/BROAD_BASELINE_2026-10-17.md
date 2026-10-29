# Broad baseline, 2026-10-17: the current engine on Gold11 and on the broadened benchmark

**Owner question (2026-10-17):** "We optimized the engine for those 11 (Gold11), they should
come back green. Broader benchmark is key to understanding how to proceed."

**Short answer:** Gold11 does not come back green, and no lane in the record has ever passed the
Gold11 gate. The best lane reproduces its historical 0.458 exactly. The shipped runtime scores
0.074. On the broadened benchmark (57 scans, 41 families, 305 bend lines) the best lane scores
0.445 and the shipped runtime scores 0.114. Visual triage also found that Gold11's own reference
lines are wrong on all 6 Gold11 scans it examined. If those verdicts hold, the errors are large
enough that a perfect engine would score only about 0.61 against today's Gold11 labels. The AI
gold on the new scans held up.

> **Ledger: consumed development.** Every number here comes from Gold11 development cases or
> BENCHMARK_V2 CONSUMED scans. None of it is unseen-family accuracy or promotion evidence. No
> FRESH, sealed, HOLDOUT10, N0xx or EHO4 data was opened. FRESH is scored once, at the end, by
> another process.

**Scoring used everywhere:** the frozen F1 matcher
(`scripts/evaluate_f1_against_manual_gold.match_lines`). A predicted line matches a gold line when
the axes are within 5°, the transverse offset is within 4 mm, and each line covers at least 60 % of
the other's span. Assignment is one-to-one Hungarian, counts are micro-aggregated, and everything
is in the raw-scan frame. The gate is G2(a) in `PRODUCTION_LINE_PLAN.md`: line-F1 ≥ 0.80 **and**
precision ≥ 0.90 on Gold11.

**The two lanes measured:**

- **Best lane** (`best_f1`): the recipe behind the historical 0.458. It takes the union of two
  line sets:
  - the scan-only A0 typed-union lane;
  - the CAD-guided detections of `run_progressive_bend_inspection` (run without a shared
    registration), cropped by the July raw-scan support gate.

  It is research code, with a median of 194 s per scan. Without CAD it runs as A0 only.
- **Runtime:** the shipped worker, `backend/bend_analysis_worker.py` →
  `run_progressive_bend_inspection`, invoked exactly as the API invokes it. It needs CAD. Lines
  tagged `CAD_REFERENCE_PLACEHOLDER` are nominal CAD axes, not detections, so they are not counted.

## 1. Bottom line

1. **Gold11 is not green, and no lane in the record has reached the gate.**
   - The best lane reproduces 0.4580152672 exactly. Re-scoring the historical artifacts and
     re-running the lane fresh give identical TP/FP/FN on 11 of 11 scans.
   - That is 0.342 below the F1 gate, and precision is 0.500 against 0.90.
   - The shipped runtime scores 0.074.
   - "Should come back green" was an expectation written in `BENCHMARK_V2.md`, not a past
     result. The best audited Gold11 score in the record is 0.458.
2. **The broad set gives the same answer, with error bars less than half as wide.**
   - Best lane: 0.445 on 57 scans (95 % CI 0.375–0.512, resampling scans).
   - Runtime: 0.114 (CI 0.053–0.181).
   - Gold11 is not special: the best lane scores 0.457 on the 44 scans from families outside
     Gold11.
   - Gold11 alone cannot resolve much: the best lane's CI there is 0.277–0.623.
3. **The shipped runtime does not work as a bend-line localizer yet.** It misses 284 gold lines:
   - 97 on scans without CAD, which it refuses to run;
   - 48 on scans where the worker raised an unhandled exception during registration (10 of 36
     CAD scans, plus 2 of the 3 zero-bend controls);
   - 42 on scans where it ran and emitted no line;
   - 97 on scans where it did emit lines. Those lines are 33 % precise.

   Its P95 wall time is 57 s and its peak RSS 12.96 GiB, against the plan's 45 s (G2(e)) and
   4 GiB (G1).
4. **Engine failure modes that dominate (best lane):**
   - **Short bends are rarely found.** Over all 305 gold lines, recall is 0.17 on bends shorter
     than 60 mm and 0.50 on longer ones. About half of the missed short bends have no line at
     all. The rest sit under a merged or over-long line.
   - **The right fold drawn with the wrong extent.** Collinear bends are merged across gaps, and
     lines are stretched along free edges. Each case costs a false negative *and* a false
     positive. Together these are about half of the errors on the triaged AI-gold scans, where
     the labels held up.
   - **False lines on box-corner seams and cut edges.** Most come from the CAD-guided half,
     whose lines are 0.42 precise against 0.64 for scan-only lines.
5. **Gold11's own labels cause a large share of its low score.**
   - Triage of 6 Gold11 scans found reference lines off the bend on all 6. That is 27 of their
     49 lines, and 11 real bends are missing from the references.
   - Label errors account for 34 of the best lane's 54 errors on those scans.
   - If these verdicts hold, an engine that drew every real bend perfectly would score about 0.61.
     **G2(a) cannot be passed until Gold11 is relabelled.**
   - With the triage-corrected labels, the best lane would score about 0.63 (an estimate, not a
     rescore). That still fails.
6. **The AI gold held up.**
   - Triage of 16 AI-gold scans (139 lines) found no label error that changes a score.
   - The STEP cross-check found one real AI miss: two 15° joggle folds on 42288005, which matter
     only if joggles count as gold. It also found two wrong stage tags.
7. **How to proceed** (details in §6):
   - Relabel Gold11, with owner sign-off, before G2(a) is used for any decision.
   - Settle four labeling conventions.
   - Judge the recipe-guided production engine on this frozen case list and matcher. On the 36
     CAD scans it must beat F1 0.474 / precision 0.517 (best lane) and F1 0.154 (runtime), with
     zero crashes, P95 ≤ 45 s and ≤ 4 GiB.

## 2. Gold11: reproduction and the "green" question

| | Lane | TP / FP / FN | Precision | Recall | Line-F1 | G2(a) |
|---|---|---|---|---|---|---|
| (a) | Historical prediction artifacts, re-scored | 30 / 30 / 41 | 0.500 | 0.423 | **0.4580152672** | fail |
| (b) | Best lane re-run fresh, one scan at a time | 30 / 30 / 41 | 0.500 | 0.423 | **0.4580152672** | fail |
| (c) | Shipped runtime | 3 / 7 / 68 | 0.300 | 0.042 | **0.0740740741** | fail |

**(a) Historical re-score.** Three independent routes give exactly 0.4580152672:

- `research/broad_baseline/score.py`;
- the original `scripts/diagnose_f1_cad_scan_fusion_gold11.py` CLI;
- the frozen evaluator CLI.

The components also match history: A0 alone scores 19/14/52 (0.365) and the gated CAD half alone
scores 20/24/51 (0.348).

The frame recovery is the identity on all 11 scans. The 9 "reference mesh" label sets were drawn
on a display cloud that is a vertex subset of the raw scan, with 0.0 mm residual. There is one
caveat. On CG08506AE97D, the display cloud was rewritten about 4 h after the labels were saved, so
a frame change there cannot be fully excluded. Even so, no single transform repairs all 6 of its
lines (§5.4).

**(b) Fresh re-run reproduces.**

- Per-scan TP/FP/FN, the CAD half, the decomposition and the candidate pools are identical on
  11 of 11 scans. Compute took 74–460 s per scan, with a peak RSS of 1.64 GiB.
- One component differs. On CG84C8FA12D2 (10839000 full scan), fresh A0 gives 4 lines / 0 TP
  against 5 / 1 historically. The cause is confirmed:
  - The historical A0 for that scan was copied from a smoke run built on a probe pool.
  - That pool's 6 extra candidates were built from all 863,115 mesh vertices, which breaks the
    200k-point contract.
  - Re-solving that probe pool reproduces the historical bytes.
- So 1 of the 30 historical true positives came from a contract-violating pool. The headline
  survives only because a CAD line (B5) covers the same gold line (M04).

**(c) Why the runtime scores 0.074.**

- **Registration refused on 3 staged scans (16 gold lines).** Two failed with "Invalid raw scan
  to reference transform" and one with fitness below 0.55.
- **No scan-fitted line on 4 well-registered scans (35 lines; fitness 0.87–0.93).** Every bend
  came back NOT_DETECTED or as a CAD-axis placeholder. Counting placeholders anyway gives 0.154.
- **The other 4 scans emit 10 lines, 3 of them correct.**
- **Mechanism, probed.** With a shared registration, the pipeline forces strict spatial matching:
  a detection's midpoint must lie within 40 mm of the CAD bend's midpoint.
  - On partial views of long bends the midpoints end up 68–73 mm apart, so real detections are
    dropped.
  - 38172000_V keeps 1 of 8 detections, 38172000_X keeps 1 of 5, and 40500000 keeps 0 of 1.
  - The best lane's CAD half calls the same function without a shared registration and emits 44
    gated lines, 20 of them correct.
- **Repeatability and memory.** A second full pass was identical on 11 of 11 scans. Peak RSS was
  10.6 GiB on the 357,851-point 40500000 scan.

**Per scan, with what triage found in the gold.** In this table, "exact" means every gold line is
matched and no line is false. Neither lane is exact on any Gold11 scan.

| Gold11 | Scan id | Scan | Gold | Best lane TP/FP/FN | F1 | Runtime TP/FP/FN | F1 | Gold quality (triage, §5.4) |
|---|---|---|---|---|---|---|---|---|
| CG08506AE97D | sc_cec08dfca9ab | 10839000 partial 2 | 6 | 0/4/6 | 0.000 | crashed: transform | 0.000 | all 6 lines in free air, 4–6 mm inside the bends |
| CG191DEC7F81 | sc_7a1773eca9fa | 49001000 PRELIM | 5 | 1/2/4 | 0.250 | 1/0/4 | 0.333 | not triaged |
| CG1E0BBA2933 | sc_4f8188b4d1a8 | 38172000 X (Codex gold) | 10 | 8/3/2 | 0.762 | 0/0/10 | 0.000 | B07/B08 sit on butt seams; 2 of the 8 TPs are engine false lines matching them |
| CG4BDD1371B6 | sc_d1732de18023 | 37361005 | 4 | 2/1/2 | 0.571 | 1/0/3 | 0.400 | not triaged |
| CG4FAF273093 | sc_70b39f530f00 | 49024000 | 3 | 2/1/1 | 0.667 | 1/2/2 | 0.333 | not triaged |
| CG64C412CC73 | sc_152f1b1aacb2 | 40500000 partial 2 | 7 | 3/3/4 | 0.462 | 0/0/7 | 0.000 | M03–M06 are wall free ends, not bends; 4 short bends missing |
| CG65CD575B1C | sc_07062eee17aa | 38172000 partial 1 | 6 | 2/0/4 | 0.500 | crashed: transform | 0.000 | not triaged |
| CG84C8FA12D2 | sc_71cf0b216f43 | 10839000 full | 8 | 2/9/6 | 0.211 | 0/5/8 | 0.000 | 6 of 8 lines off the bend, 4 of them on the flat wall face |
| CGE4B0549E41 | sc_08686dc1d5d4 | 10839000 partial 1 | 4 | 2/0/2 | 0.667 | refused: fitness | 0.000 | not triaged |
| CGEBA9A869AC | sc_f2475286312b | 38172000 partial 2 | 8 | 1/5/7 | 0.143 | 0/0/8 | 0.000 | 7 of 8 lines wrong (3 are free edges); 3 bends missing |
| CGF41F49066A | sc_2714fb7e0c0b | 38172000 V (Codex gold) | 10 | 7/2/3 | 0.737 | 0/0/10 | 0.000 | B06/B08 on seam corners; 2 end-wall bends missing |
| **All 11** | | | 71 | **30/30/41** | **0.458** | **3/7/68** | **0.074** | |

On matched lines, the mean axis-angle error is 0.69° for the best lane and 0.72° for the runtime.
Bend angle against the CAD nominal is nominal-relative only, because Gold11 has no angle truth. It
is 1.27° MAE over 64 bends for the best lane's CAD half and 2.05° over 33 bends for the runtime.

## 3. Broad benchmark

### 3.1 Case list

- **File:** `runlog/broad_baseline/case_list_broad_v3.json`, with case_list_sha256 `c25a49c1…`
  (file sha `c3592162…`). It was built from `BENCHMARK_V2.json` (sha `0db0ff1f…`).
- **Cases:** 60 in total: 11 Gold11, 4 human-gold outside Gold11, and 45 AI_GOLD_ACCEPTED (there
  are no AI_GOLD_PARTIAL).
  - 57 are scored for line-F1, with 305 gold lines over 41 families. 36 of these 57 have a STEP
    file.
  - The other 3 are zero-bend controls.
- **Excluded by construction:**
  - FRESH and sealed scans;
  - 3 possible CAD reference meshes;
  - the 7 synthetic 47266000 LIDAR_DEMO captures (`EXCLUSIONS_NOTE.md`).
- **Label version:** sc_9bd69bc6535c is scored on its round-3 label, which has 12 lines.
- **Left in:** two source meshes of the demo generator (sc_9b9fd762f361, sc_d45186653eed). The
  sensitivity check in §3.7 drops them.

### 3.2 Headline by cohort

Macro F1 is the mean of per-scan F1. "–" means the lane emitted no lines in that cohort.

| Cohort | Scans | Gold | Best TP/FP/FN | P | R | **F1** | Macro | Runtime TP/FP/FN | P | R | **F1** | Macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **All** | 57 | 305 | 115/97/190 | 0.542 | 0.377 | **0.445** | 0.426 | 21/43/284 | 0.328 | 0.069 | **0.114** | 0.101 |
| Gold11 | 11 | 71 | 30/30/41 | 0.500 | 0.423 | **0.458** | 0.452 | 3/7/68 | 0.300 | 0.042 | **0.074** | 0.097 |
| Not Gold11 | 46 | 234 | 85/67/149 | 0.559 | 0.363 | **0.440** | 0.420 | 18/36/216 | 0.333 | 0.077 | **0.125** | 0.102 |
| Families not in Gold11 | 44 | 220 | 82/57/138 | 0.590 | 0.373 | **0.457** | 0.432 | 15/32/205 | 0.319 | 0.068 | **0.112** | 0.099 |
| AI gold | 42 | 214 | 80/63/134 | 0.559 | 0.374 | **0.448** | 0.422 | 18/36/196 | 0.333 | 0.084 | **0.134** | 0.112 |
| Human gold outside Gold11 | 4 | 20 | 5/4/15 | 0.556 | 0.250 | **0.345** | 0.401 | 0/0/20 | – | 0.000 | **0.000** | 0.000 |
| Full scans | 44 | 228 | 93/64/135 | 0.592 | 0.408 | **0.483** | 0.455 | 16/35/212 | 0.314 | 0.070 | **0.115** | 0.111 |
| Staged scans | 13 | 77 | 22/33/55 | 0.400 | 0.286 | **0.333** | 0.327 | 5/8/72 | 0.385 | 0.065 | **0.111** | 0.070 |
| CAD available | 36 | 208 | 91/85/117 | 0.517 | 0.438 | **0.474** | 0.486 | 21/43/187 | 0.328 | 0.101 | **0.154** | 0.161 |
| No CAD | 21 | 97 | 24/12/73 | 0.667 | 0.247 | **0.361** | 0.323 | 0/0/97 | – | 0.000 | **0.000** | 0.000 |
| AI gold with CAD | 23 | 131 | 59/53/72 | 0.527 | 0.450 | **0.486** | 0.490 | 18/36/113 | 0.333 | 0.137 | **0.195** | 0.205 |

**95 % CIs from resampling scans (10,000 resamples):**

| Cohort | Best lane F1 CI | Runtime F1 CI |
|---|---|---|
| All | 0.375–0.512 | 0.053–0.181 |
| Gold11 | 0.277–0.623 | 0.000–0.169 |
| CAD available | 0.390–0.557 | 0.077–0.241 |
| Staged | 0.224–0.443 | 0.000–0.212 |

The gap between the two lanes on all scans is 0.25–0.41.

**Other aggregates:**

| Measure | Best lane | Runtime |
|---|---|---|
| Family macro F1 (41 families) | 0.416 | 0.120 |
| Scans scored exactly (all gold matched, no false line) | 6 / 57 | 2 / 57 |
| Scans with F1 ≥ 0.8 | 8 | 2 |
| Scans with zero TP | 14 | 45 |
| Scans with no line at all | 9 | 38 |
| Axis-angle error on matched lines, mean | 0.56° | 0.41° |
| Transverse offset on matched lines, mean / median | 1.27 / 1.08 mm | 0.91 / 0.58 mm |
| Zero-bend controls (3): false lines | 5 lines on 2 scans | 3 lines on 1 scan (the worker crashed on the other 2) |

- **Exact scans are all small.** Every exactly scored scan has 1–3 bends. No scan with more than 3
  bends is exact in either lane.
- **Geometry is not the problem.** Where a line matches, it is accurate in both lanes: well under
  1° and about 1 mm. The failures are about which lines exist and how long they are.

### 3.3 Recall by bend length (all 305 gold lines)

| Gold line length | Gold lines | Best lane recall | Runtime recall |
|---|---|---|---|
| < 30 mm | 60 | 0.233 | 0.050 |
| 30–60 mm | 55 | 0.109 | 0.000 |
| 60–120 mm | 77 | 0.429 | 0.117 |
| 120–250 mm | 61 | 0.607 | 0.115 |
| ≥ 250 mm | 52 | 0.481 | 0.038 |

- **Short bends are mostly missed.** Bends under 60 mm are 115 of the 305 gold lines (38 %). The
  best lane finds 20 of them (17 %), against 95 of the 190 longer bends (50 %).

Why the best lane misses lines, by gold length. The causes are the scorer's nearest-line
diagnostic, not the metric.

| Gold length | Missed | No line nearby | Covered only by a longer or merged line (span < 60 %) | Nearest line > 4 mm off | Angle > 5° |
|---|---|---|---|---|---|
| < 60 mm | 95 | 54 | 40 | – | 1 |
| 60–250 mm | 68 | 48 | 14 | 6 | – |
| ≥ 250 mm | 27 | 14 | – | 13 | – |

- **Short bends fail in two ways.** 54 have no line nearby at all. Another 40 lie under a line that
  is too long: a merged or over-extended line covers them but fails the span rule.
- **The dip at ≥ 250 mm is mostly Gold11.** 22 of the 27 misses are on Gold11 scans.
  - 13 of the 27 have a line more than 4 mm away. 10 of those 13 are on scans where triage found
    the gold off the bend (§5.4).
  - Another 5 are the 541–564 mm bends of sc_db1839480738, a no-CAD scan on which the scan-only
    half drew nothing.

### 3.4 Where the best lane's lines come from

| Source | TP | FP | Precision |
|---|---|---|---|
| Scan-only A0 lines | 75 | 42 | 0.641 |
| CAD-guided lines (`CADF_*`) | 40 | 55 | 0.421 |

The CAD-guided half adds 40 true positives but brings 55 of the 97 false lines. A0 lines alone
would score F1 0.355 on the same 305 gold lines.

### 3.5 Runtime: what happened on each scan

| Outcome (57 scored scans) | Scans | Gold lines missed | TP | FP |
|---|---|---|---|---|
| No CAD, so `RUNTIME_REQUIRES_CAD` | 21 | 97 | 0 | 0 |
| Worker exception "Invalid raw scan to reference transform" | 7 | 38 | 0 | 0 |
| Worker exception "Alignment unavailable: insufficient shared registration fitness" | 3 | 10 | 0 | 0 |
| Ran, emitted no line | 7 | 42 | 0 | 0 |
| Ran, emitted lines | 19 | 97 | 21 | 43 |
| **Total** | 57 | 284 | 21 | 43 |

- **The exceptions are crashes, not abstentions.** Each one is an unhandled `ValueError`: exit
  code 1 and no report. The plan's G2(e) wants each such failure to be an explicit UNKNOWN or
  INVALID.
- **Staged scans fail most.** 7 of the 13 staged scans hit an exception. So do 3 of the 23 full
  scans that have CAD.
- **One of the crashes is a full scan that registers fine elsewhere.** sc_6e5d5a2f60f5 registers
  cleanly in an independent CAD registration (90 % of points within 2 mm), so the crash there is an
  engine alignment bug.

Two diagnostic variants of the runtime were rescored on the same case list. Neither is shipped.

| Variant | TP/FP/FN | P | R | F1 |
|---|---|---|---|---|
| Shipped runtime | 21/43/284 | 0.328 | 0.069 | 0.114 |
| Plus the research-side span gate (crops lines to scan support) | 27/35/278 | 0.435 | 0.089 | 0.147 |
| Plus CAD-axis placeholders (nominal axes, not detections) | 26/64/279 | 0.289 | 0.085 | 0.132 |

### 3.6 Run time and memory

| | Best lane (60 runs) | Runtime worker (39 runs with CAD) |
|---|---|---|
| Wall time P50 / P95 / max | 194 / 459 / 600 s | 25.0 / 57.3 / 79.1 s |
| Runs over 45 s | 51 | 5 |
| Peak RSS P50 / P95 / max | 1.09 / 1.55 / 1.85 GiB | 0.65 / 10.6 / 12.96 GiB |
| Runs over 4 GiB | 0 | 3: sc_152f1b1aacb2 10.58, sc_c623d2e45bff 10.74, sc_9ef2cf4e52cb 12.96 |
| Timeouts / crashes | 1 timeout at 600 s (sc_85a7f8cca6bc, no CAD, in the CP-SAT step) | 12 exceptions (10 scored scans + 2 controls) |

- **Against the plan's gates.** G2(e) asks for a per-scan P95 ≤ 45 s and zero crashes; G1 asks
  for ≤ 4 GiB at about 1M points. The best lane misses the speed gate by about 10×. The runtime
  misses all three.
- **How these were measured.**
  - Runs were serial, with `OMP_NUM_THREADS=1`.
  - Wall time includes the wrapper's file copies.
  - The 21 no-CAD runtime runs return in under 0.11 s and are left out of the runtime column.

### 3.7 Sensitivity checks

| Check | Scans | Best TP/FP/FN | Best F1 | Runtime TP/FP/FN | Runtime F1 |
|---|---|---|---|---|---|
| Headline | 57 | 115/97/190 | 0.445 | 21/43/284 | 0.114 |
| Without the 2 demo-generator source meshes | 55 | 114/90/181 | 0.457 | 21/38/274 | 0.119 |
| Without the 8 scans whose gold triage found wrong | 49 | 90/67/149 | 0.455 | 21/38/218 | 0.141 |
| Gold11 without its 6 label-defect scans | 5 | 9/4/13 | 0.514 | 3/2/19 | 0.222 |
| Full scans, stage tags corrected (§4) | 44 | 94/71/141 | 0.470 | 19/39/216 | 0.130 |
| Staged scans, stage tags corrected | 13 | 21/26/49 | 0.359 | 2/4/68 | 0.053 |

None of these moves either lane anywhere near the gate.

### 3.8 Worst 15 scans for the best lane

| # | Case | Family | Scan | Gold | Best lane | Runtime |
|---|---|---|---|---|---|---|
| 1 | sc_cec08dfca9ab | 10839000 | Gold11, staged | 6 | 4 lines, 0 TP (gold is in free air, §5.4) | crashed: transform |
| 2 | sc_8379cebef9e3 | 49001000 | OBJ, staged PRELIM | 5 | 2 FP | crashed: fitness |
| 3 | sc_38091ac91d9d | 28870000 | full | 2 | 3 FP | 3 FP |
| 4 | sc_822f083e3dfc | 37126000 | full | 2 | 3 FP | 3 FP |
| 5 | sc_db1839480738 | 38169000 | no CAD | 5 | no lines | needs CAD |
| 6 | sc_9ef2cf4e52cb | 45796000 | staged | 4 | no lines | no lines, 12.96 GiB |
| 7 | sc_6101a12648a9 | 38965002 V | no CAD | 3 | no lines | needs CAD |
| 8 | sc_85a7f8cca6bc | 50539000 | no CAD | 3 | timed out at 600 s | needs CAD |
| 9 | sc_c60c74f5ae01 | 38965002 X | no CAD | 3 | no lines | needs CAD |
| 10 | sc_9b930fe58a1b | unknown (`scan.ply`) | no CAD | 2 | no lines | needs CAD |
| 11 | sc_9b9fd762f361 | 47266000 | demo source STL | 1 | 1 FP (span too short) | crashed: fitness |
| 12 | sc_9c890e76934f | unknown (`part 2.stl`) | no CAD | 2 | no lines | needs CAD |
| 13 | sc_abe194f04ade | unknown (`part 1.ply`) | no CAD | 2 | no lines | needs CAD |
| 14 | sc_fee685eee4db | 49125000 | human gold, no CAD | 2 | no lines | needs CAD |
| 15 | sc_d45186653eed | 11979003 | demo source STL | 9 | 7 lines, 1 TP (F1 0.125) | 5 FP |

- **The scan-only half produced nothing on 7 small no-CAD parts** (2–5 bends each, 19 gold lines)
  and timed out on an eighth. These 8 scans were not triaged, so the cause is unknown.
- **Per-family table:** Appendix A.

## 4. CAD cross-check of the AI gold

- **Coverage.** 25 BENCHMARK_V2 parts have a STEP file. 23 accepted AI labels sit on FULL scans of
  those parts.
- **Why not compare raw V10 counts.** The cross-check registered each scan to its CAD and measured
  the fold angle in the scan at every CAD bend. It did not trust V10 counts alone, for two
  reasons:
  - V10 under-counts some parts: 28870000 (1 found vs 2 real), 49001000 (3 vs 5; the human gold
    also has 5) and 37126000 (0 vs 2).
  - V10 counts a fold interrupted by reliefs as several segments.
- **Collinear merge.** Collinear segments that fold the same two plates were merged into one CAD
  "bend line". That reproduces every merge the AI labels made, and it is also the Gold11 humans'
  convention (40500000).
- **Result.** 17 of the 23 FULL scans match CAD by count, and every AI line matched a CAD bend after
  registration.
- **The 6 flagged scans:**

| Part | Scan | Stage tag | AI | V10 | CAD bend lines | Verdict |
|---|---|---|---|---|---|---|
| 42288005 | sc_2233372b8df7 | FULL | 6 | 8 | 8 | **AI miss.** The two 15° joggle folds (C5/C6, 45 mm, 11.7 mm apart) are formed (14.1° and 15.1° measured) but unlabelled. Counts only if joggles are gold. |
| 15651004 | sc_927a2fa13067 | REPEAT_3RD_SCAN | 2 | 3 | 3 | **Stage tag wrong.** The second 155 mm flange is not bent yet (0.2°); AI's 2 is right. |
| 28870000 | sc_38091ac91d9d | FULL | 2 | 1 | 2 | **V10 wrong.** It rejected a real 27° bend; AI is right. |
| 11979003 | sc_d45186653eed | STL, no tag | 9 | 11 | 9 | Neither. Three collinear segments form one 287 mm AI line. |
| 50757000 | sc_777c64c88967 | GOOD_V | 9 | 12 | 9 | Neither. Collinear groups are one AI line each. |
| 50757000 | sc_752e2bd54aad | DEFECTIVE_X | 9 | 12 | 9 | Neither. Same merges; the measured deviations fit the DEFECTIVE tag. |

**Staged scans:**

- No accepted label has more bends than its CAD.
- 40500000 PARTIAL_SCAN_1 (sc_eaa89c6d1f1f) has all 14 CAD segments formed, so it is really a
  full scan.
- The 28870000 SCAN_1 zero-bend label (sc_3bde9996f14d) is inconclusive: a rolled R30 section sits
  next to its end tab. Both lanes draw 3 lines on this "zero-bend control", so its false-line count
  is weak evidence until someone checks the tab.

## 5. Failure taxonomy from visual triage

### 5.1 What was triaged

- **Selection.** 24 scans: the ones with the most FN+FP across both lanes. 16 have AI gold, 6 are
  Gold11 and 2 have human gold outside Gold11.
- **Coverage.** They hold most of the errors:

  | Lane | False negatives covered | False positives covered |
  |---|---|---|
  | Best lane | 130 of 190 | 77 of 97 |
  | Runtime | 194 of 284 | 35 of 43 |

- **Method.** Each scan was triaged by one agent, working from its own cross-sections of the raw
  points; most agents also checked against the STEP. There was no second, blind verifier.
- **Bias.** The selection is biased toward bad scans. The shares below describe where the bulk of
  the errors come from, not a random sample.
- **Mapping.** Each of the 436 scored errors on these scans was mapped to the triage verdict that
  names it (`research/broad_baseline/report/triage_map.py`; 0 left unmapped). The raw verdicts are
  in `runlog/broad_baseline/triage/triage_results_24.json`.

### 5.2 Cause counts, mapped to scored errors

**Best lane: 207 errors on the 24 triaged scans.**

| Cause | FN | FP | Total | Share | On AI gold | On Gold11 | On human gold outside Gold11 |
|---|---|---|---|---|---|---|---|
| ENGINE_MISS: real bend, no line | 63 | – | 63 | 30 % | 49 | 9 | 5 |
| ENGINE_SPLIT_OR_MERGE: collinear merge or wrong extent | 33 | 17 | 50 | 24 % | 48 | 1 | 1 |
| LABEL_ERROR: the gold is wrong | 27 | 18 | 45 | 22 % | 0 | 34 | 11 |
| ENGINE_FALSE_LINE: no bend there | – | 32 | 32 | 15 % | 22 | 10 | 0 |
| ENGINE_WRONG_LOCUS: off the fold, or overrun | 7 | 10 | 17 | 8 % | 17 | 0 | 0 |
| **Total** | 130 | 77 | 207 | | 136 | 54 | 17 |

- **On AI gold (136 errors), the gold was never the cause.**
  - Wrong extent (merge, overrun or off-locus): 65 errors (48 %).
  - Missed bends: 49 (36 %).
  - False lines: 22 (16 %).
- **On the 6 triaged Gold11 scans, label errors are 34 of 54 errors (63 %).**

**Runtime: 229 errors on the 24 triaged scans, by what the worker did.**

| Worker outcome on triaged scans | Scans | Errors | What triage found |
|---|---|---|---|
| No CAD | 5 | 49 FN | the lane abstained |
| Worker exception | 5 | 30 FN | registration refused or crashed. One full scan (sc_6e5d5a2f60f5) registers fine in an independent CAD registration, so its crash is an engine bug. |
| Ran, no line | 4 | 35 FN | 24 real bends not emitted (strict spatial gate, or placeholder-only geometry); 11 label errors |
| Ran, emitted lines | 10 | 80 FN + 35 FP | 53 misses (46 %), 20 merge/extent (17 %), 17 wrong locus (15 %), 17 false lines (15 %), 8 label errors (7 %) |

### 5.3 Mechanisms

This table counts how many of the 24 triaged scans show each mechanism. The tags are this report's
reading of the 24 triage summaries; per-scan tags are in `SUMMARY.json` → `triage.scan_mechanisms`.

| Mechanism | Scans | Lanes | Example |
|---|---|---|---|
| **Bend miss:** a real formed bend gets no line (mostly tabs, lips, feet, return flanges) | 20 | both | [50757000 tab/foot bends](runlog/broad_baseline/overlays/sc_777c64c88967/triage/zoom_B8_B9.png), [38172000 slot tabs](runlog/broad_baseline/overlays/sc_4f8188b4d1a8/triage/slices_slot_tabs.png) |
| **Over-extension:** the right fold, with the line stretched along a free edge or past the part (1 FN + 1 FP per bend) | 13 | both | [45796000 plate-to-wall lines, 175–191 mm on a ~98 mm bend](runlog/broad_baseline/triage/sc_c623d2e45bff/plate_wall_bend_zone_extent.png) |
| **Seam or edge false line:** a line on a box-corner butt or open seam, a cut edge or flat plate | 13 | both, mostly the CAD-guided half | [45796000 butt-seam corners](runlog/broad_baseline/overlays/sc_1268c72dcc30/triage_corner_closeups.png), [10839000 open corner seams](runlog/broad_baseline/overlays/sc_71cf0b216f43/triage_corners.png), [40500000 closed corners called "rolled R14"](runlog/broad_baseline/triage/sc_eaa89c6d1f1f/corners.png) |
| **Collinear merge:** one line across collinear bends separated by a gap, relief, window or slot | 11 | both | [47247004 three tabs merged across relief slots](runlog/broad_baseline/triage/sc_8935b8a8e8aa/slot_and_endflange_closeups.png), [11979003 B7+B8 across an 11 mm relief](runlog/broad_baseline/overlays/sc_d45186653eed/triage/b7_b8_gap.png), [23553000 leg roots fused into 395–404 mm lines](runlog/broad_baseline/overlays/sc_6eba02723d4b/triage_long_lines_profile.png) |
| **Detection discarded:** the runtime measured or detected the bend, then emitted nothing (no CAD slot, placeholder-only geometry, or the strict spatial gate) | 10 | runtime, and the best lane's CAD half, which calls the same function | `runlog/broad_baseline/gold11_repro/spatial_matching_probe/*.json` |
| **CAD target list wrong:** the lane's own CAD bend list has phantom rolled or corner bends, collapses collinear bends, or omits bends (e.g. 4 of 16 bends on 23553000, 6 of 12 on 49204020, 3 phantom 151–162° bends on 50757000) | 9 | runtime and the CAD-guided half | [49204020 sections](runlog/broad_baseline/overlays/sc_e080ae094f35/triage_sections.png) |
| **Label error** (§5.4) | 8 | – | [10839000 gold on the wall face](runlog/broad_baseline/overlays/sc_71cf0b216f43/triage_xsections.png), [10839000 gold in free air](runlog/broad_baseline/overlays/sc_cec08dfca9ab/triage_zoom_midspan.png) |
| **Selection dropped a good candidate:** a correct candidate existed (decomposition, pool or CAD-gated) but the typed union or the fusion dropped it | 7 | best lane | triage notes for sc_8935b8a8e8aa, sc_8379cebef9e3, sc_6e5d5a2f60f5, sc_36e5d6998b9b |
| **Registration failure:** an exception, or a shared alignment that slid about 8.5 mm so nothing matched | 6 | runtime | §3.5 |
| **No CAD:** the runtime abstains | 5 | runtime | §3.5 |

Three example images:

![Gold11 CG08506AE97D: gold in free air inside each bend; engine lines on the apex just outside the 4 mm circle](runlog/broad_baseline/overlays/sc_cec08dfca9ab/triage_zoom_midspan.png)

![45796000: the runtime's "rolled" corner bends are butt seams with no bend arc](runlog/broad_baseline/overlays/sc_1268c72dcc30/triage_corner_closeups.png)

![23553000: thick green lines are missed gold; the dashed red lines are 395–404 mm best-lane lines that fuse leg roots with free plate edge](runlog/broad_baseline/overlays/sc_6eba02723d4b/overlay_4views.png)

### 5.4 Label quality

**Gold11.** All 6 triaged Gold11 scans have gold that changes the score. Together:

- 27 of their 49 lines are judged wrong;
- 11 real bends are missing;
- the 2 Codex-labelled scans are included.

| Gold11 | Scan | What is wrong | Evidence |
|---|---|---|---|
| CG84C8FA12D2 | sc_71cf0b216f43 | M05–M08 were drawn on the flat 40 mm wall face, slanted, 1.6–16.4 mm from the corner. M01/M02 sit 3.5–5 mm out on the short flange. Two 35° joggles are unlabelled (a policy question). | [triage_xsections.png](runlog/broad_baseline/overlays/sc_71cf0b216f43/triage_xsections.png) |
| CGEBA9A869AC | sc_f2475286312b | M01/M03/M07 sit 4.7–5.7 mm off the corner. M05 is on the wrong side of a cutout (14.6 mm away). M04/M06/M08 are free edges. The +PC1 lip and 2 ear bends are missing. M02 is also off the corner, but it still matches, so it is not counted. | [triage_zoom_M05M06.png](runlog/broad_baseline/overlays/sc_f2475286312b/triage_zoom_M05M06.png) |
| CG08506AE97D | sc_cec08dfca9ab | All 6 lines are in free air, 2.6–3.6 mm (median) from the scan and 4–6 mm inside the concave side. The display cloud was rewritten about 4 h after labelling, but no single transform (a 1.032 scale fit lines up only 3 of the 6) repairs them. | [triage_zoom_midspan.png](runlog/broad_baseline/overlays/sc_cec08dfca9ab/triage_zoom_midspan.png) |
| CG64C412CC73 | sc_152f1b1aacb2 | M03–M06 are wall free ends or straight wall, not bends. M01/M02/M07 carry over-long extents from the engine seed. Two 20 mm lips and two 14 mm tabs are missing. | [closeups_M03_M06.png](runlog/broad_baseline/triage/sc_152f1b1aacb2/closeups_M03_M06.png) |
| CGF41F49066A (Codex) | sc_2714fb7e0c0b | B06/B08 mark seam corners, 90° off in direction. The two 44 mm end-wall bends are missing. | [endwall_pc1_slices.png](runlog/broad_baseline/triage/sc_2714fb7e0c0b/endwall_pc1_slices.png) |
| CG1E0BBA2933 (Codex) | sc_4f8188b4d1a8 | B07/B08 sit on butt seams; the two end-flange hinges (STEP C1/C9) are missing. The best lane's 2 TPs there are false lines that match the mislabel. | [zoom_endpiece_bend_vs_seam.png](runlog/broad_baseline/overlays/sc_4f8188b4d1a8/triage/zoom_endpiece_bend_vs_seam.png) |

- **Not yet triaged:** CG191DEC7F81, CG4BDD1371B6, CG4FAF273093, CG65CD575B1C and CGE4B0549E41.
- **A related observation from the Gold11 reproduction.** It independently found the gold off the
  scanned surface on CG08506AE97D and CGEBA9A869AC.
- **Circularity.** The references are partly derived from engine output:
  - CG64C412CC73's kept extents came from the engine seed.
  - On sc_36e5d6998b9b (below), M01–M11 are bit-identical to the engine's own zero-shot
    decomposition.
  - "Optimizing for Gold11" therefore partly optimized toward engine-derived and misplaced
    references.

**Human gold outside Gold11.** Both triaged scans have gold that changes the score:

| Scan | What is wrong |
|---|---|
| sc_36e5d6998b9b (ysherman, "human" tier) | M08–M11 are free edges or flat feet; M02/M04 merge bends of opposite direction, about 5 mm off every crease; 4 ear-tab bends are missing. M01–M11 are bit-identical to the engine's zero-shot decomposition, so this label is engine output that was accepted. |
| sc_a0305a6cf787 (47959001) | HB3, HB4 and HB5 sit 4.6, ~6 and 8 mm off the fold; a sixth formed fold (STEP has 6) is unlabelled. |

sc_fee685eee4db and sc_8dfdab44605d were not triaged.

**AI gold.** 16 scans with 139 lines were triaged, and none had a label error that changes a score.

- **Minor defects:**
  - sc_752e2bd54aad: B2 is truncated at 284 mm; the real bend is about 394 mm.
  - sc_c623d2e45bff: B2 is about 17 mm short.
  - Neither changes a score.
  - sc_2233372b8df7: the two 15° joggle folds are unlabelled (§4).
  - sc_186463746daa: the labeler's notes swap bend names; the coordinates are right.
- **Caveat.** The AI labelers, the AI verifiers and the AI triage may share blind spots. The STEP
  cross-check (§4) is the independent check.

**What correct labels would mean for Gold11.** These are estimates, not rescores; per-scan
assumptions are in `SUMMARY.json` → `estimates_not_rescored`.

| Gold11 estimate | TP/FP/FN | P | R | F1 |
|---|---|---|---|---|
| Best lane, as scored today | 30/30/41 | 0.500 | 0.423 | 0.458 |
| Best lane against the triage-corrected labels | 41/19/30 | 0.683 | 0.577 | **~0.63** |
| Runtime against the triage-corrected labels | 7/3/64 | 0.700 | 0.099 | ~0.17 |
| A perfect engine (one apex line per real bend) against **today's** labels | 43/28/28 | 0.606 | 0.606 | **~0.61** |

The last row assumes the 5 untriaged Gold11 scans have perfect gold, which is optimistic. If the
triage verdicts hold, today's Gold11 references cap any engine at about 0.6, which is below the
0.80 gate.

## 6. What this says about how to proceed

### 6.1 Which failure modes dominate

- **Best lane.** It has 287 errors in all.
  - On AI-gold scans, where the labels held, the order is:
    1. wrong extent: collinear merges across gaps and lines overrunning along free edges (about
       half);
    2. missed bends. Half of all misses are bends under 60 mm, although those are only 38 % of
       the gold;
    3. false lines on seams and cut edges, mostly from the CAD-guided half.
  - Matched-line geometry is already good: 0.56° and 1.1 mm median.
- **Runtime.**
  - 66 % of its missed lines (187 of 284) are lost before any line is drawn: no CAD, an
    exception, or nothing emitted.
  - Where it does emit lines, the main causes are a wrong CAD bend list and detections it threw
    away for lack of a CAD slot.
  - Only 21 of its 64 lines are correct.
- **Gold11.** Label error is the largest single cause of its low score.

### 6.2 The bar the recipe-guided production engine must beat

Use the same case list (sha `c25a49c1…`), the same matcher and the raw-scan frame. A recipe-guided
engine needs CAD, so the 21 no-CAD scans (97 lines) are outside its scope. Report them as out of
scope; do not drop them silently. Per the plan, label the comparison "recipe-guided vs blind union,
not like-for-like".

| Measure | Best lane today | Runtime today | Plan gate |
|---|---|---|---|
| Gold11 line-F1 / precision (current labels) | 0.458 / 0.500 | 0.074 / 0.300 | ≥ 0.80 / ≥ 0.90. Not reachable on current labels (§5.4): relabel, then re-score both legacy lanes for the real baseline. |
| CAD-available, 36 scans / 208 lines: F1 (P / R) | **0.474** (0.517 / 0.438) | **0.154** (0.328 / 0.101) | must beat |
| AI gold with CAD, 23 scans / 131 lines: F1 (P / R) | 0.486 (0.527 / 0.450) | 0.195 (0.333 / 0.137) | must beat |
| Staged scans, 13 / 77 lines: F1 | 0.333 | 0.111 | G3 |
| Exact scans: all 57 / with CAD (36) | 6 / 4 | 2 / 2 | must beat, per G2(a) |
| Zero-bend controls: false lines | 5 (2 of 3 scans) | 3 (1 of 3) | 0 |
| Run time P95 per scan | 459 s | 57 s | ≤ 45 s |
| Peak RSS | 1.85 GiB | 12.96 GiB | ≤ 4 GiB |
| Crashes | 1 timeout | 12 exceptions in 39 CAD runs | 0, with explicit UNKNOWN or INVALID |
| Matched-line geometry (axis / median offset) | 0.56° / 1.08 mm | 0.41° / 0.58 mm | must not regress |

- **What counts as a real improvement.** The scan-resampled CI half-width is about ±0.08 on the
  CAD cohort and ±0.17 on Gold11. A claimed gain smaller than that on Gold11 alone is not
  evidence.
- **What this report does not measure.** The new production engine (`domains/production`) is
  being built by another workflow.

### 6.3 Labels that need fixing, and the decisions they need

**Gold11 and human gold. These need the owner's re-verification and a relabel.**

- The 6 triaged Gold11 scans: CG84C8FA12D2, CGEBA9A869AC, CG08506AE97D, CG64C412CC73 and the
  Codex-labelled CGF41F49066A and CG1E0BBA2933.
- sc_36e5d6998b9b: demote it from the human tier now, because it is circular.
- sc_a0305a6cf787.
- Check the 5 untriaged Gold11 scans and the 2 untriaged human scans.
- Each scan's triage verdict lists the corrected lines; proposed hinge lines for CG1E0BBA2933 are in
  `overlays/sc_4f8188b4d1a8/triage/triage_verdicts.json`.
- Use the AI-gold protocol: apex locus, two blind labelers, a verifier and the STEP cross-check,
  then owner sign-off.

**AI gold. These are small.**

- **Joggles.** Add the two 15° joggle folds to sc_2233372b8df7 if joggles count.
- **Extents.** Extend sc_752e2bd54aad B2 to about 394 mm, and sc_c623d2e45bff B2 by about 17 mm.
- **Stage tags.**
  - sc_eaa89c6d1f1f: change PARTIAL_SCAN_1 to full.
  - sc_927a2fa13067: change REPEAT_3RD_SCAN to staged.
- **Zero-bend control.** Look at the end tab of sc_3bde9996f14d before treating it as a control.

**Conventions to decide once, and apply to every tier, before the next scoring run:**

1. **Joggles.** Are joggle folds gold lines? 42288005 has 15° joggles and 10839000 has 3 mm,
   35° micro-joggles.
2. **Shallow lanced tabs.** Do they count? 47251002 B12 is a 15° tab.
3. **Collinear pieces split by a full-width window or relief.** Example: 11979003 B2/B3 and B4/B5.
   - One line, as in the `ENGINE_COMPILE.md` "Windows" rule.
   - Or two lines, as in the gold and in V10.

   The choice flips FN+FP pairs either way.
4. **Locus.** The gold is the visible outer apex. Engines that output the virtual sharp or the
   cylinder axis sit about 0.41 × (Ri + t) or more away, and that offset eats into the 4 mm
   tolerance.

### 6.4 Recommended order of work

1. **Fix the yardstick first.**
   - Relabel Gold11 and re-score both legacy lanes on the corrected labels; that sets the true
     Gold11 baseline.
   - Until then, treat G2(a) as unmeasurable rather than failed-by-the-engine. An accuracy loop
     tuned against today's Gold11 would chase label noise.
2. **Use the broad CAD-available cohort as the main yardstick.** Use it together with the
   relabelled Gold11, freeze it (it already has a sha), and score FRESH once at the end.
3. **Engine work, in order of error mass.** These are design inputs for the recipe-guided engine.
   The legacy lanes are baselines and are not to be tuned.
   - **Extent.** Bound each line to where both flanges are present, and split at material gaps
     according to convention 3. This turns FN+FP pairs into TPs, and it is the largest single
     lever on the clean AI-gold scans.
   - **Short-bend recall.** Bends under 60 mm have 17 % recall. The recipe says where they are,
     but each one must still be measured from the scan.
   - **Seam rejection.** Require a bend arc, meaning material that wraps around the corner,
     rather than two planes meeting. Box parts (10839000, 38172000, 40500000, 45796000) are
     full of butt and open seams.
   - **CAD target hygiene.** The recipe's bend list should come from STEP topology with the
     same-plate collinear merge. It must not contain corner or "rolled" pseudo-bends.
   - **Never discard a measured bend** for lack of a CAD slot. Never crash on registration;
     return an explicit INVALID or UNKNOWN instead (G2(e)).
   - **Memory and time.** Keep peak RSS ≤ 4 GiB and P95 ≤ 45 s.

## 7. Caveats

- **Ledger.** These are consumed development numbers. They are not unseen accuracy.
- **Gold provenance.** 42 of the 57 scored scans use AI gold, which is agent-adjudicated. The
  STEP cross-check supports it, but it is not owner-verified.
- **Triage.** Verdicts come from one agent per scan. The triage sample is the 24 worst scans. For
  the other 33 scans, the only cause information is the scorer's nearest-counterpart diagnostic.
- **Estimates.** The label-corrected and perfect-engine Gold11 figures are estimates, not
  rescores.
- **Gold sources and family labels.** 5 scans have an unknown family. Each of them, and
  `ysherman`, is counted as its own family, so "41 families" may overstate the number of distinct
  parts. Five AI-gold scans exist only
  as STL or OBJ (4 STL, 1 OBJ) and have no studio label file. They are scored against the bench
  label lines, with an apex locus.
- **Historical recipe kept as is.** The best lane's CAD half mixes frames: primary detections are
  in the raw frame and refinement lines in the CAD frame. The gate drops the CAD-frame lines. This
  is part of the historical recipe and was kept.
- **Stale files.** `scores_runtime__runtime_gated.json` and
  `scores_runtime__runtime_with_placeholders.json` in the run log are on the older 67-case v2 list.
  The v3 rescores are in `report_rescore_v3/`.
- **Custody.**
  - Only Gold11 cases and gold, `../source-assets/02_CAD_QC_3D`, and BENCHMARK_V2 CONSUMED scans
    with their AI gold were used.
  - Nothing FRESH or sealed, nothing from HOLDOUT10, N0xx or EHO4, and nothing under
    `incoming_codex_fetch_2026-10-17` was opened.
  - Nothing under `backend/`, `scripts/` or `domains/bend/` was edited.
  - Heavy engine runs went one at a time behind the RSS gate. The geometry lock was not taken, and
    nothing was committed.
  - Writing this report involved no engine run. The only computation was rescoring existing
    predictions (light, under 0.1 GiB).

## 8. Files

All paths are under `docs/production/runlog/broad_baseline/` unless they start with `research/`.

| File | Contents |
|---|---|
| `SUMMARY.json` | Machine-readable version of this report: cohorts, CIs, family table, resources, the taxonomy mapped to scored errors, label findings, estimates, and the bar to beat. |
| `gold11_repro.json`, `gold11_repro/` | Gold11 (a)/(b)/(c), the C001 replays, the spatial-matching probe, and the determinism check. |
| `case_list_broad_v3.json` | The scored case list. |
| `scores_best_f1.json`, `scores_runtime.json` | Per-case scores (matches, unmatched ids, diagnostics). |
| `report_rescore_v3/` | Runtime and its two diagnostic variants rescored on v3. |
| `BROAD_BASELINE_SUMMARY.json`, `runs/` | Run-step summary and the per-run records and predictions. |
| `cad_crosscheck.json`, `cad_crosscheck_*` | STEP cross-check of the AI gold. |
| `triage/triage_results_24.json` | The 24 triage verdict sets, verbatim. |
| `overlays/<scan_id>/`, `triage/<scan_id>/` | Four-view overlays for all 60 cases, plus triage renders. |
| `research/broad_baseline/report/` | Code for this report (`build_summary.py`, `triage_map.py`, `length_bins.py`, `rescore_runtime_variants_v3.py`). Rebuild with `.venv/bin/python -B research/broad_baseline/report/build_summary.py`. |

## Appendix A. Per-family results (57 scored scans)

Column G11 marks a Gold11 family; CAD says whether the family has a STEP file. TP/FP/FN are
micro-summed over the family's scans.

| Family | G11 | CAD | Scans | Gold | Best TP/FP/FN | Best F1 | Runtime TP/FP/FN | Runtime F1 |
|---|---|---|---|---|---|---|---|---|
| 10839000 | G11 | yes | 3 | 18 | 4/13/14 | 0.229 | 0/5/18 | 0.000 |
| 11979003 | | yes | 1 | 9 | 1/6/8 | 0.125 | 0/5/9 | 0.000 |
| 15651004 | | yes | 3 | 6 | 6/1/0 | 0.923 | 1/0/5 | 0.286 |
| 23553000 | | yes | 3 | 29 | 11/5/18 | 0.489 | 0/3/29 | 0.000 |
| 28870000 | | yes | 1 | 2 | 0/3/2 | 0.000 | 0/3/2 | 0.000 |
| 34499000 | | yes | 1 | 3 | 3/0/0 | 1.000 | 3/0/0 | 1.000 |
| 37126000 | | yes | 1 | 2 | 0/3/2 | 0.000 | 0/3/2 | 0.000 |
| 37361005 | G11 | yes | 1 | 4 | 2/1/2 | 0.571 | 1/0/3 | 0.400 |
| 38169000 | | no | 1 | 5 | 0/0/5 | 0.000 | 0/0/5 | 0.000 |
| 38172000 | G11 | yes | 4 | 34 | 18/10/16 | 0.581 | 0/0/34 | 0.000 |
| 38965002 | | no | 2 | 6 | 0/0/6 | 0.000 | 0/0/6 | 0.000 |
| 40500000 | G11 | yes | 2 | 16 | 6/11/10 | 0.364 | 3/4/13 | 0.261 |
| 42288005 | | yes | 1 | 6 | 4/3/2 | 0.615 | 1/1/5 | 0.250 |
| 44211000 | | yes | 1 | 1 | 1/0/0 | 1.000 | 1/0/0 | 1.000 |
| 45796000 | | yes | 3 | 17 | 8/8/9 | 0.485 | 5/8/12 | 0.333 |
| 47247004 | | no | 1 | 6 | 1/1/5 | 0.250 | 0/0/6 | 0.000 |
| 47251002 | | no | 2 | 24 | 6/2/18 | 0.375 | 0/0/24 | 0.000 |
| 47266000 | | yes | 1 | 1 | 0/1/1 | 0.000 | 0/0/1 | 0.000 |
| 47555020 | | no | 1 | 1 | 1/0/0 | 1.000 | 0/0/1 | 0.000 |
| 47959001 | | yes | 1 | 5 | 1/2/4 | 0.250 | 0/0/5 | 0.000 |
| 48811000 | | yes | 1 | 4 | 3/0/1 | 0.857 | 2/0/2 | 0.667 |
| 48963008 | | yes | 1 | 3 | 2/1/1 | 0.667 | 0/0/3 | 0.000 |
| 48991006 | | yes | 1 | 5 | 4/6/1 | 0.533 | 0/0/5 | 0.000 |
| 49001000 | G11 | yes | 2 | 10 | 1/4/9 | 0.133 | 1/0/9 | 0.182 |
| 49024000 | G11 | yes | 1 | 3 | 2/1/1 | 0.667 | 1/2/2 | 0.333 |
| 49125000 | | no | 1 | 2 | 0/0/2 | 0.000 | 0/0/2 | 0.000 |
| 49204020 | | yes | 1 | 12 | 6/0/6 | 0.667 | 2/4/10 | 0.222 |
| 50535000 | | no | 1 | 2 | 2/0/0 | 1.000 | 0/0/2 | 0.000 |
| 50537000 | | no | 1 | 2 | 1/0/1 | 0.667 | 0/0/2 | 0.000 |
| 50539000 | | no | 1 | 3 | 0/0/3 | 0.000 | 0/0/3 | 0.000 |
| 50552000 | | no | 1 | 7 | 5/2/2 | 0.714 | 0/0/7 | 0.000 |
| 50553000 | | no | 1 | 2 | 1/0/1 | 0.667 | 0/0/2 | 0.000 |
| 50566000 | | no | 1 | 7 | 1/1/6 | 0.222 | 0/0/7 | 0.000 |
| 50567000 | | no | 1 | 2 | 1/0/1 | 0.667 | 0/0/2 | 0.000 |
| 50757000 | | yes | 2 | 18 | 8/6/10 | 0.500 | 0/5/18 | 0.000 |
| unknown (sc_9b930fe58a1b) | | no | 1 | 2 | 0/0/2 | 0.000 | 0/0/2 | 0.000 |
| unknown (sc_9c890e76934f) | | no | 1 | 2 | 0/0/2 | 0.000 | 0/0/2 | 0.000 |
| unknown (sc_a5e534e6097c) | | no | 1 | 5 | 1/2/4 | 0.250 | 0/0/5 | 0.000 |
| unknown (sc_abe194f04ade) | | no | 1 | 2 | 0/0/2 | 0.000 | 0/0/2 | 0.000 |
| unknown (sc_f362d8de507c) | | no | 1 | 5 | 1/2/4 | 0.250 | 0/0/5 | 0.000 |
| ysherman | | no | 1 | 12 | 3/2/9 | 0.353 | 0/0/12 | 0.000 |

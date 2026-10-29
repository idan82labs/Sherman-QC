# Baseline: current zero-shot detector (A0) and E3, every gold scan (2026-10-25)

> **Ledger: consumed development**, plus one sealed aggregate. Every DEV number is on consumed scans. TEST-B is reported as a single aggregate of the frozen baseline: its per-case rows are sealed (`baseline/scores_TESTB_SEALED.json`), and no development may read them.

**Matcher.** Frozen: `research/broad_baseline/score.py score_case`, which is 5° / 4 mm / 0.6 of both spans, one-to-one Hungarian, apex locus, raw scan frame.

**Aggregates.**
- Micro = summed TP/FP/FN over complete-label line scans.
- Family-macro = mean over part families of family-micro F1.
- 95 % CIs: percentile bootstrap over families, 10 000 reps, seed 20261018.
- Zero-bend controls are reported separately as FP lines.

**Tools.** `tools/score_baseline.py` (scoring), `tools/a0_driver.sh`, `tools/e3_driver.py`. Results are in `BASELINE.json` and per-case DEV rows in `baseline/scores_dev_per_case.json`.

## Arms

| Arm | What runs | Code state | Outputs |
|---|---|---|---|
| **A0** (zero-shot) | `run_lane.py --lane best_f1` without `--cad`, i.e. the strict-blind scan-only typed union. On no-CAD scans this is exactly the "best_f1" lane. | Unchanged since the GOLD_SET_V2 runs: no file under `backend/`, `domains/bend` or `config` changed after 2026-10-17 12:00. A determinism replay on 2 stored scans gave **identical** lines (`baseline/a0_replay/`). | Reused 103 stored GOLD_SET_V2 A0 outputs (PLY twins scored under their STL prediction id, vertex sets identical). **New runs: 18 DEV + 61 TEST-B**, with the frozen 600 s budget. |
| **best_f1** (A0 ∪ gated CAD half) | Stored GOLD_SET_V2 runs | as above | GOLD_SET_V2 scans only. This reproduces GOLD_SET_V2's blind-baseline 0.413 / 0.338 exactly. |
| **E3** | `e3-1.12.0`, current engine | Recipes: the frozen A1 recipe where one exists (16 SKUs, recompiled), else an `E1-DEV-POLICY-2` rebuild (20 SKUs). Of 36, 33 compile; 3 rolled parts give NO_RECIPE. | **Re-run on all 53 E3-eligible DEV scans** (`baseline/e3_runs/`). Peak RSS 2.48 GiB; wall time up to 116 s. |
| **E3-else-A0** | E3 lines when E3 returns a VALID inspection, otherwise A0 | | |

**Not run.** E3 was not run on the 5 DEV-CHECK cases that have never had E3 output (`pc_eaa89…`, `pc_152f…`, `pc_c623…`, `pc_9ef2…`, `pc_7a17…`). This preserves their one-time use (PLAN OD-3). E3 was not run on TEST-B either, which has 1 STEP scan.

## Line-F1

| Cohort | Scans / fam | Gold lines | A0 P / R | **A0 micro [CI]** | **A0 fam-macro [CI]** | E3 micro [CI] (scans) | E3 fam-macro [CI] | best_f1 micro (scans) |
|---|---|---|---|---|---|---|---|---|
| **DEV, all** | 116 / 91 | 586 | 0.597 / 0.220 | **0.322** [0.262, 0.378] | **0.267** [0.207, 0.329] | – | – | 0.413 (99) |
| DEV: A1 + other consumed | 89 / 71 | 435 | 0.631 / 0.228 | 0.334 [0.262, 0.398] | 0.274 [0.202, 0.349] | 0.987 [0.969, 1.0] (31) | 0.932 | 0.414 (89) |
| DEV: consumed DEV-CHECK | 10 / 7 | 59 | 0.615 / 0.271 | 0.376 [0.231, 0.474] | 0.302 [0.143, 0.455] | 0.885 [0.60, 1.0] (5) | 0.714 | 0.407 (10) |
| DEV: consumed FRESH | 17 / 16 | 92 | 0.424 / 0.152 | 0.224 [0.127, 0.319] | 0.235 [0.113, 0.361] | 0.864 [0.605, 0.981] (13) | 0.721 | – |
| DEV with STEP | 55 / 37 | 327 | 0.661 / 0.239 | 0.351 [0.272, 0.417] | 0.277 | – | – | 0.504 (42) |
| DEV E3-eligible (same scans for both arms) | 49 / 34 | 295 | 0.683 / 0.241 | 0.356 [0.269, 0.433] | 0.287 [0.199, 0.379] | **0.951** [0.890, 0.982] (49): P 0.993, R 0.912 | **0.820** [0.697, 0.926] | – |
| DEV no CAD | 61 / 54 | 259 | 0.520 / 0.197 | 0.286 [0.199, 0.372] | 0.260 [0.175, 0.349] | n/a | n/a | 0.285 (57) |
| DEV full / partial | 102 / 14 | 503 / 83 | | 0.327 / 0.291 | 0.272 / 0.351 | 0.941 (39) / 0.984 (10) | | |
| **TEST-B (sealed, aggregate only)** | 61 / 50 | 329 | 0.722 / 0.198 | **0.310** [0.236, 0.376] | **0.252** [0.170, 0.339] | not run | | |

- **Zero-bend controls** (5 DEV): A0 puts 0 FP lines on them, E3 puts 0, and best_f1 puts 5. E3 refused or failed to compile 3 of the 4 E3-eligible controls (WRONG_PART on `sc_3bde…`, NOT_COMPILED on `sc_4d62…`, a BLOCKED rolled recipe on `sc_b471…`), so none of them received a line.
- **E3-else-A0 equals E3** on the E3-eligible cohort: 0.951 / 0.820. A0 emits nothing on the 3 scans E3 could not serve: the mirrored `sc_2d3e…` and the rolled/seamed `sc_ead8…`.

## Error budget on DEV

The cause is diagnostic only: it is the nearest counterpart, from `score_broad.diagnose`.

| Arm | FN: MISSED | FN: SPAN < 60 % | FN: OFFSET > 4 mm | FN: ANGLE > 5° | FP: SPURIOUS | FP: SPAN | FP: OFFSET | FP: ANGLE |
|---|---|---|---|---|---|---|---|---|
| A0, DEV all (458 FN, 87 FP) | **373 (81 %)** | 52 | 26 | 6 | 39 | 37 | 4 | 6 |
| E3, E3-eligible (26 FN, 2 FP) | 25 | 0 | 1 | 0 | 0 | 2 | 0 | 0 |

- **A0 emits zero lines on 36 of 116 DEV scans:** 33 NO_PREDICTIONS and 3 lane failures (600 s timeouts). That accounts for **121 of the 586 gold lines (21 %)**.
- **Recall is the whole gap.** A0 precision is 0.60, but recall is 0.22.
- OFFSET is a minor class for A0: 26 FN and 4 FP. So the two-skin hypothesis can explain only a small part of the A0 gap on this set. Its main value is as the base for a new segmentation (PLAN Z1 → Z2).
- **E3 misses are coverage.**
  - 49207000: positions withheld along the free datum direction (9 lines).
  - 2d3e: a true mirror, refused.
  - 38898000: seams on a rolled part, no recipe.
  - Plus a few DEV-CHECK and FRESH bends.

## TEST-B aggregate (frozen A0, one look)

Scored once, on 2026-10-25 at about 01:45Z, after all 61 runs finished.

| | Value |
|---|---|
| Scans / families / gold lines | 61 / 50 / 329 |
| TP / FP / FN | 65 / 25 / 264 |
| Precision / recall | 0.722 / 0.198 |
| Micro line-F1 | **0.310**, CI [0.236, 0.376] |
| Family-macro line-F1 | **0.252**, CI [0.170, 0.339] |
| Lane failures | 1 timeout at 600 s |

- This is consistent with DEV (0.322 / 0.267). A0 transfers at the same low level, with recall about 0.20.
- No per-case data is in this document. Runs and per-case scores are under `baseline/a0_runs_TESTB_SEALED/` and `baseline/scores_TESTB_SEALED.json`, and the pools are under `/private/tmp/claude-501/sherman_zeroshot_work/testb_SEALED/`.
- The error budget was not computed for TEST-B, on purpose.

**What a pass would take.** TEST-B is a pass at ≥ 0.85 micro, and family-macro should be reported alongside it. Getting there from 0.310 needs recall to go from about 0.20 to ≥ 0.85 while precision rises from 0.72 to ≥ 0.85.

## Runtime of the new A0 runs

- 79 runs, 2 workers sharing the 3-slot semaphore with other agents.
- Lane wall time: median 229 s, P95 432 s, max 600 s (the timeout).
- Peak RSS at most 1.93 GiB.
- E3 runs took up to 116 s each, with peak RSS 2.48 GiB.

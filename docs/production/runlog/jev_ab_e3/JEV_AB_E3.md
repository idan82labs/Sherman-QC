# New production engine (E3) and Jev: A/B on the frozen A1 cases

Date: 2026-10-18. Follow-up to `docs/production/runlog/jev_ab_v2/JEV_INPUT_OPT.md` (v2) and the blocked
new-engine A/B in `docs/production/runlog/a1/iter_0/jev_ab_e3/`. Code: `research/jev/e3_ab_v1.py`, tests
`tests/test_research_jev_e3_ab_v1.py`. Machine-readable: `results.json` here (v2 completion summary plus
every E3 arm, cohort, CI and per-case row); per iteration under `iter_N/`. Nothing under
`domains/production/**`, `contracts/**`, `apps/**` or `frontend/**` was edited. Engine code was imported
read-only (`domains.production.engine.eval.freeze.physical`, `eval.line_metric`).

**Ledger: exploratory, not unseen.** Every scan is a CONSUMED development scan. Nothing here is promotion
evidence. DEV-CHECK, `*_SEALED*`, FRESH, HOLDOUT10, N001-N016 and EHO4 were never read.

## Headline

1. **v2 test step finished.** With the restored credits, all three Jev designs now have an answer for
   every candidate on all 37 test scans / 27 families. Nothing chosen on calibration changed (each
   threshold reproduces exactly). The v2 verdict holds on the full test families. The frozen composite
   filter is flat (ΔF1 [-0.040, +0.037]). Pool + Jev composite is worse than the lane (0.307 vs 0.429).
   Pool + Jev + GBM gives 0.442, below pool + logistic regression at 0.478. Jev ranks below the plain
   features it reads (pool AUC 0.878 vs 0.930).
2. **Jev does not help the new engine.** On the scans E3 ran, the engine's own lines are already
   P 0.85 / R 0.87 against the frozen BROAD gold and 80/80 against GOLD_SET_V1.
   - **Filters.** By GOLD_SET_V1, every line a filter removes is a true line. The Jev composite filter
     drops 6 lines, all 6 also true against BROAD. The single-feature filter drops 8: 4 are BROAD-gold
     convention FPs and 4 are true. On the engine's lines, Jev's AUC is 0.51 (chance level).
   - **Rescue.** Pool rescue recovers 1 of the engine's 10 BROAD misses. All 10 are in the pool, so the
     limit is selection, not supply. Every rescue arm adds false lines: on GOLD_SET_V1 the engine falls
     from 1.000 to 0.899 (LR) or 0.952 (Jev+GBM), and recall cannot rise there.
   - **Using Jev as an extra ranker signal** helps precision a little against LR alone. Jev+GBM on the
     engine-ran BROAD scans gives 0.831 vs 0.784, but both stay below the engine alone (0.861).
3. **Where rescue does raise recall:** only on the 13 BROAD scans the engine *refused* in iteration 3.
   There the engine has no lines, so the arm is simply the scan-only pool. On all 26 cases, recall rises
   0.472 → 0.597 (ΔR CI [+0.041, +0.221]) but micro F1 does not move (0.607 → 0.589, CI includes 0).
   These refused cases run from iteration 4 on (the G3 truth table froze at 07:10Z), so this effect
   should vanish.

## Engine scored

| item | value |
|---|---|
| A1 iteration | **iter_3** (highest `docs/production/runlog/a1/iter_N` with a `scorecard.json`; `iter_4` holds only `engine_before.tgz`, still in its CHANGE phase at 10:56 IDT, no physical run yet) |
| engine | `e3-1.4.0`, default config sha256 `9ec3bb560b617e6d3ea73ad453eccfe10189b1796d2c0238002fad3780ecea12` (read from every OK `result.json`; `iter_3/engine_manifest.json`) |
| fingerprints (scorecard) | engine `ab513015…` (freeze-style), G1 runner engine `9525cfff…`, eval `8aa6860c…`, compile `6147b43c…` |
| per-scan outputs | `scratchpad/a1_iter3m/phys/physical/E3/<case>/` (13 OK of 26 BROAD cases; the other 13 truth-relevant cases were refused by the amendment-7 gate in iteration 3) |
| lines | the engine's OUTER_APEX loci in the raw scan frame, `physical._engine_line_predictions` (= the A1 scorecard's lines; 80 lines; engine alone reproduces the scorecard exactly: BROAD all-26 0.607 / 0.667, engine-ran 0.861 / 0.896, GOLD_SET_V1 1.000) |

## Arms (all thresholds frozen from `jev_ab_v2/design_choice.json`; none fitted here)

- **(a) engine alone.**
- **(b) engine + Jev composite filter.** Five atomic Nouls, combined by the calibration-fitted logistic
  regression (`jev_composite_combined`). An engine line is kept if the score is ≥ 0.3066.
- **(c) engine + v2 single-feature filter.** Pool two-sided transition score ≥ 0.2577. A line with no
  matching pool cluster has no value and is dropped, as frozen. (c′) is the sensitivity run that keeps
  those lines.
- **(d) engine + liberal pool** (the `apply_v2 --add-from-pool` candidate set: the v2 clustered scan-only
  pool of each scan).
  - **d1, rescue with logistic regression.** Engine lines are always kept. Pool lines with LR ≥ 0.9786
    are added if they duplicate no kept line (NMS 5°/4 mm/0.6).
  - **d2, Jev as an extra gate.** As d1, but an added pool line must also pass the Jev composite
    filter (≥ 0.3066).
  - **d3, Jev as a ranker.** As d1, but ranked by the frozen Jev+classifier score: the mean of Jev
    composite and the GBM, ≥ 0.7150.
  - **d4, `apply_v2` literally.** Engine lines and pool lines are ranked together by LR, then NMS. This
    can drop engine lines.
  - Reference: pool + LR alone, with no engine.
- **Scores.** Models are fitted on the v2 calibration-family pool rows, leave-one-family-out when the
  target scan's family is a v2 calibration family. Four A1 families are v2 calibration families
  (10839000, 15651004, 48811000, 49204020). The `…_v2test` cohort excludes them.
- **Truth and scorer.**
  - BROAD: the frozen BROAD A1 gold (`cases_A1.json` `broad_gold_cases`), scored with the frozen
    physical scorer `broad_line_scores` (5°/4 mm/0.6, one-to-one, raw frame). Complete-label cases give
    P/R/F1; recall-only and zero-bend controls are reported apart.
  - GOLD_SET_V1, A1 view: the selection of `a1/iter_3/measure/tools/goldset.py` (A1 cases with STEP
    and an OK engine run, including STL→PLY twins), scored with `research/broad_baseline/score.score_case`.
  - CIs: family-level bootstrap, 2,000 rounds. Deltas are paired with engine alone on the same
    resampled families.
- **Jev calls.** The composite design, on the 80 engine lines only (states from `states_v2`:
  coordinate-free, identity-free). Pool-candidate answers come from the v2 cache. Cost $0.013.

## New-engine arms (iter_3, e3-1.4.0)

Column 2 counts lines over all 26 A1 scans; the TP/FP/FN columns are the cohort. Bold CIs exclude 0.

### BROAD, the 13 scans the engine ran (11 families) — the rescue test

| arm | lines kept, all 26 A1 scans (engine / pool added) | TP / FP / FN | P | R | micro F1 [95% CI] | ΔF1 [95% CI] | ΔR [95% CI] | fam-macro F1 | Δ fam-macro [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| (a) engine alone | 80 / 0 | 68 / 12 / 10 | 0.850 | 0.872 | **0.861** [0.747, 0.982] | - | - | 0.896 | - |
| (b) engine + Jev composite filter | 74 / 0 | 62 / 12 / 16 | 0.838 | 0.795 | **0.816** [0.721, 0.933] | [-0.088, +0.000] | [-0.152, +0.000] | 0.863 | [-0.076, +0.000] |
| (c) engine + v2 single-feature filter | 72 / 0 | 64 / 8 / 14 | 0.889 | 0.821 | **0.853** [0.761, 0.947] | [-0.055, +0.031] | [-0.106, +0.000] | 0.891 | [-0.025, +0.011] |
| (c') same, lines with no pool match kept (sensitivity) | 80 / 0 | 68 / 12 / 10 | 0.850 | 0.872 | **0.861** [0.747, 0.982] | [+0.000, +0.000] | [+0.000, +0.000] | 0.896 | [+0.000, +0.000] |
| (d4) apply_v2 literal (engine + pool re-ranked by LR) | 25 / 107 | 46 / 35 / 32 | 0.568 | 0.590 | **0.579** [0.487, 0.708] | **[-0.444, -0.140]** | **[-0.456, -0.114]** | 0.513 | **[-0.636, -0.142]** |
| (d3) engine + pool rescue, Jev+GBM ranker | 80 / 43 | 69 / 19 / 9 | 0.784 | 0.885 | **0.831** [0.722, 0.952] | [-0.048, +0.002] | [+0.000, +0.056] | 0.883 | [-0.048, +0.019] |
| (d1) engine + pool rescue, logistic regression | 80 / 69 | 69 / 29 / 9 | 0.704 | 0.885 | **0.784** [0.702, 0.906] | **[-0.148, -0.018]** | [+0.000, +0.056] | 0.868 | [-0.071, +0.000] |
| (d2) engine + pool rescue, LR and Jev composite gate | 80 / 69 | 69 / 29 / 9 | 0.704 | 0.885 | **0.784** [0.711, 0.908] | **[-0.156, -0.011]** | [+0.000, +0.056] | 0.871 | [-0.072, +0.005] |
| ref: pool + LR alone (no engine) | 0 / 134 | 44 / 39 / 34 | 0.530 | 0.564 | **0.547** [0.463, 0.689] | **[-0.464, -0.174]** | **[-0.483, -0.130]** | 0.500 | **[-0.640, -0.160]** |

### GOLD_SET_V1 in the A1 view (13 scans / 11 families, engine ran)

| arm | lines kept, all 26 A1 scans (engine / pool added) | TP / FP / FN | P | R | micro F1 [95% CI] | ΔF1 [95% CI] | ΔR [95% CI] | fam-macro F1 | Δ fam-macro [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| (a) engine alone | 80 / 0 | 80 / 0 / 0 | 1.000 | 1.000 | **1.000** [1.000, 1.000] | - | - | 1.000 | - |
| (b) engine + Jev composite filter | 74 / 0 | 74 / 0 / 6 | 1.000 | 0.925 | **0.961** [0.918, 1.000] | [-0.082, +0.000] | [-0.152, +0.000] | 0.969 | [-0.075, +0.000] |
| (c) engine + v2 single-feature filter | 72 / 0 | 72 / 0 / 8 | 1.000 | 0.900 | **0.947** [0.912, 1.000] | [-0.087, +0.000] | [-0.161, +0.000] | 0.981 | [-0.046, +0.000] |
| (c') same, lines with no pool match kept (sensitivity) | 80 / 0 | 80 / 0 / 0 | 1.000 | 1.000 | **1.000** [1.000, 1.000] | [+0.000, +0.000] | [+0.000, +0.000] | 1.000 | [+0.000, +0.000] |
| (d4) apply_v2 literal (engine + pool re-ranked by LR) | 25 / 107 | 51 / 30 / 29 | 0.630 | 0.637 | **0.633** [0.515, 0.755] | **[-0.485, -0.245]** | **[-0.510, -0.232]** | 0.544 | **[-0.682, -0.231]** |
| (d3) engine + pool rescue, Jev+GBM ranker | 80 / 43 | 80 / 8 / 0 | 0.909 | 1.000 | **0.952** [0.920, 0.983] | **[-0.080, -0.018]** | [+0.000, +0.000] | 0.959 | **[-0.086, -0.010]** |
| (d1) engine + pool rescue, logistic regression | 80 / 69 | 80 / 18 / 0 | 0.816 | 1.000 | **0.899** [0.825, 0.978] | **[-0.175, -0.022]** | [+0.000, +0.000] | 0.941 | **[-0.130, -0.005]** |
| (d2) engine + pool rescue, LR and Jev composite gate | 80 / 69 | 80 / 18 / 0 | 0.816 | 1.000 | **0.899** [0.823, 0.978] | **[-0.177, -0.022]** | [+0.000, +0.000] | 0.945 | **[-0.120, -0.005]** |
| ref: pool + LR alone (no engine) | 0 / 134 | 48 / 35 / 32 | 0.578 | 0.600 | **0.589** [0.492, 0.727] | **[-0.508, -0.273]** | **[-0.531, -0.260]** | 0.527 | **[-0.700, -0.253]** |

### BROAD, engine ran, v2 test families only (no family used for any v2 threshold)

| arm | lines kept, all 26 A1 scans (engine / pool added) | TP / FP / FN | P | R | micro F1 [95% CI] | ΔF1 [95% CI] | ΔR [95% CI] | fam-macro F1 | Δ fam-macro [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| (a) engine alone | 80 / 0 | 49 / 12 / 10 | 0.803 | 0.831 | **0.817** [0.686, 0.970] | - | - | 0.857 | - |
| (b) engine + Jev composite filter | 74 / 0 | 43 / 12 / 16 | 0.782 | 0.729 | **0.754** [0.667, 0.866] | **[-0.111, -0.012]** | **[-0.189, -0.018]** | 0.812 | **[-0.097, -0.004]** |
| (c) engine + v2 single-feature filter | 72 / 0 | 45 / 8 / 14 | 0.849 | 0.763 | **0.804** [0.696, 0.900] | [-0.072, +0.034] | [-0.125, +0.000] | 0.850 | [-0.034, +0.016] |
| (c') same, lines with no pool match kept (sensitivity) | 80 / 0 | 49 / 12 / 10 | 0.803 | 0.831 | **0.817** [0.686, 0.970] | [+0.000, +0.000] | [+0.000, +0.000] | 0.857 | [+0.000, +0.000] |
| (d4) apply_v2 literal (engine + pool re-ranked by LR) | 25 / 107 | 32 / 33 / 27 | 0.492 | 0.542 | **0.516** [0.400, 0.600] | **[-0.500, -0.142]** | **[-0.531, -0.097]** | 0.372 | **[-0.778, -0.203]** |
| (d3) engine + pool rescue, Jev+GBM ranker | 80 / 43 | 50 / 19 / 9 | 0.725 | 0.848 | **0.781** [0.676, 0.922] | [-0.057, +0.008] | [+0.000, +0.079] | 0.839 | [-0.061, +0.025] |
| (d1) engine + pool rescue, logistic regression | 80 / 69 | 50 / 28 / 9 | 0.641 | 0.848 | **0.730** [0.653, 0.837] | **[-0.172, -0.010]** | [+0.000, +0.079] | 0.824 | [-0.087, +0.000] |
| (d2) engine + pool rescue, LR and Jev composite gate | 80 / 69 | 50 / 28 / 9 | 0.641 | 0.848 | **0.730** [0.667, 0.828] | [-0.180, +0.000] | [+0.000, +0.079] | 0.827 | [-0.094, +0.011] |
| ref: pool + LR alone (no engine) | 0 / 134 | 30 / 37 / 29 | 0.448 | 0.508 | **0.476** [0.383, 0.559] | **[-0.525, -0.191]** | **[-0.551, -0.122]** | 0.354 | **[-0.794, -0.231]** |

### BROAD, all 26 frozen A1 cases (24 complete-label line cases / 14 families; refused cases count as misses)

| arm | lines kept, all 26 A1 scans (engine / pool added) | TP / FP / FN | P | R | micro F1 [95% CI] | ΔF1 [95% CI] | ΔR [95% CI] | fam-macro F1 | Δ fam-macro [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| (a) engine alone | 80 / 0 | 68 / 12 / 76 | 0.850 | 0.472 | **0.607** [0.355, 0.866] | - | - | 0.667 | - |
| (b) engine + Jev composite filter | 74 / 0 | 62 / 12 / 82 | 0.838 | 0.431 | **0.569** [0.337, 0.822] | [-0.078, +0.000] | [-0.104, +0.000] | 0.642 | [-0.061, +0.000] |
| (c) engine + v2 single-feature filter | 72 / 0 | 64 / 8 / 80 | 0.889 | 0.444 | **0.593** [0.349, 0.844] | [-0.051, +0.010] | [-0.070, +0.000] | 0.662 | [-0.019, +0.003] |
| (c') same, lines with no pool match kept (sensitivity) | 80 / 0 | 68 / 12 / 76 | 0.850 | 0.472 | **0.607** [0.355, 0.866] | [+0.000, +0.000] | [+0.000, +0.000] | 0.667 | [+0.000, +0.000] |
| (d4) apply_v2 literal (engine + pool re-ranked by LR) | 25 / 107 | 63 / 68 / 81 | 0.481 | 0.438 | **0.458** [0.366, 0.621] | [-0.332, +0.060] | [-0.278, +0.141] | 0.432 | **[-0.492, -0.011]** |
| (d3) engine + pool rescue, Jev+GBM ranker | 80 / 43 | 79 / 44 / 65 | 0.642 | 0.549 | **0.592** [0.397, 0.865] | [-0.089, +0.101] | **[+0.029, +0.125]** | 0.700 | [-0.020, +0.088] |
| (d1) engine + pool rescue, logistic regression | 80 / 69 | 86 / 62 / 58 | 0.581 | 0.597 | **0.589** [0.447, 0.811] | [-0.109, +0.121] | **[+0.041, +0.221]** | 0.708 | [-0.029, +0.119] |
| (d2) engine + pool rescue, LR and Jev composite gate | 80 / 69 | 86 / 62 / 58 | 0.581 | 0.597 | **0.589** [0.447, 0.813] | [-0.113, +0.123] | **[+0.041, +0.221]** | 0.710 | [-0.031, +0.120] |
| ref: pool + LR alone (no engine) | 0 / 134 | 61 / 72 / 83 | 0.459 | 0.424 | **0.440** [0.356, 0.609] | [-0.349, +0.053] | [-0.294, +0.138] | 0.424 | **[-0.498, -0.017]** |

**Rescue diagnostic (engine-ran scans).**
- The engine misses 10 BROAD gold lines on the 13 scans it ran, and every one is matched by some pool
  candidate (`rescue_ceiling_engine_ran`). The misses are 2 on each of 38172000 X and V, 2 on 37361005,
  and 4 on 11979003.
- d1–d3 recover 1 of them (37361005). They add 17 (d1/d2) or 7 (d3) BROAD false lines on those scans.
- GOLD_SET_V1, the vision-checked set, already gives the engine 80/80 on these scans, so the 10 BROAD
  misses are gold-convention conflicts, not engine misses (iter_3 scorecard: "BROAD-gold convention
  conflicts"). Rescue cannot help there, and it costs precision (GOLD_SET_V1 P 1.000 → 0.816 / 0.909).
- AUC on the engine's 80 lines (label: matches a BROAD gold line; 68 positive):
  - Jev composite 0.511;
  - Jev+GBM 0.560;
  - GBM 0.556;
  - logistic regression 0.614;
  - single feature 0.676 (n = 72 with a pool match).

  Nothing separates the engine's few BROAD-FP lines from its true lines. Those FPs are themselves real
  bends drawn to a different convention.

## v2 test step, completed (all 37 test scans / 27 families, truth = AI gold + Gold11 relabel)

The frozen design choice (`design_choice.json`, sha256 `26fb8880…`) and every calibration threshold are
unchanged: re-scoring reproduces each threshold exactly (`results.json` →
`v2_test_completed.arms.*.threshold_unchanged_vs_9scan_run`, all true). Only the test-family Jev answers
were added: 15,715 composite, 15,454 typed and 15,571 typed+reading bodies. Cached answers replayed.
`JEV_INPUT_OPT.md` and `jev_ab_v2/results.json` are updated. The 9-scan versions are kept in `logs/`.

| arm | keep t | TP / FP / FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs lane [95% CI] | family-macro F1 [95% CI] | AUC (any-match) | control FP lines |
|---|---|---|---|---|---|---|---|---|---|
| lane alone (best_f1) | - | 71 / 63 / 126 | 0.530 | 0.360 | **0.429** [0.324, 0.496] | - | 0.348 [0.221, 0.486] | - | 5 |
| old Jev filter (v1 question + v1 state) | 0.35 | 67 / 44 / 130 | 0.604 | 0.340 | **0.435** [0.333, 0.500] | [-0.018, +0.028] | 0.314 [0.193, 0.440] | 0.718 [0.61, 0.84] | 1 |
| new-input Jev filter: typed | 0.196 | 69 / 53 / 128 | 0.566 | 0.350 | **0.433** [0.327, 0.507] | [-0.016, +0.024] | 0.343 [0.220, 0.476] | 0.550 [0.46, 0.70] | 5 |
| new-input Jev filter: typed+reading | 0.244 | 63 / 48 / 134 | 0.568 | 0.320 | **0.409** [0.314, 0.482] | [-0.046, +0.011] | 0.335 [0.215, 0.467] | 0.574 [0.49, 0.71] | 5 |
| new-input Jev filter: composite (frozen choice) | 0.307 | 66 / 47 / 131 | 0.584 | 0.335 | **0.426** [0.324, 0.507] | [-0.040, +0.037] | 0.354 [0.228, 0.496] | 0.716 [0.58, 0.89] | 5 |
| Jev(composite)+classifier filter | 0.212 | 67 / 46 / 130 | 0.593 | 0.340 | **0.432** [0.327, 0.506] | [-0.019, +0.031] | 0.352 [0.226, 0.489] | 0.813 [0.67, 0.94] | 5 |
| classifier filter (GBM) | 0.131 | 61 / 28 / 136 | 0.685 | 0.310 | **0.427** [0.292, 0.518] | [-0.059, +0.044] | 0.315 [0.187, 0.444] | 0.831 [0.67, 0.94] | 3 |
| single best feature filter | 0.258 | 71 / 48 / 126 | 0.597 | 0.360 | **0.449** [0.338, 0.523] | **[+0.006, +0.042]** | 0.357 [0.227, 0.498] | 0.925 [0.84, 0.98] | 3 |
| liberal pool + pool generator prior | 6.45 | 45 / 83 / 152 | 0.352 | 0.228 | **0.277** [0.202, 0.344] | **[-0.230, -0.053]** | 0.223 [0.150, 0.308] | 0.848 [0.79, 0.89] | 0 |
| liberal pool + Jev typed | 0.72 | 57 / 146 / 140 | 0.281 | 0.289 | **0.285** [0.191, 0.365] | **[-0.250, -0.031]** | 0.215 [0.114, 0.320] | 0.782 [0.75, 0.81] | 0 |
| liberal pool + Jev typed+reading | 0.74 | 55 / 107 / 142 | 0.340 | 0.279 | **0.306** [0.227, 0.359] | **[-0.207, -0.023]** | 0.195 [0.119, 0.270] | 0.792 [0.76, 0.82] | 0 |
| liberal pool + Jev composite (frozen choice) | 0.941 | 56 / 112 / 141 | 0.333 | 0.284 | **0.307** [0.196, 0.408] | [-0.254, +0.009] | 0.205 [0.110, 0.311] | 0.878 [0.85, 0.91] | 0 |
| liberal pool + Jev(composite)+classifier | 0.715 | 65 / 32 / 132 | 0.670 | 0.330 | **0.442** [0.294, 0.561] | [-0.115, +0.133] | 0.310 [0.191, 0.445] | 0.901 [0.87, 0.93] | 9 |
| liberal pool + logistic regression | 0.979 | 80 / 58 / 117 | 0.580 | 0.406 | **0.478** [0.359, 0.567] | [-0.047, +0.138] | 0.361 [0.246, 0.496] | 0.930 [0.89, 0.96] | 2 |
| liberal pool + GBM | 0.533 | 74 / 62 / 123 | 0.544 | 0.376 | **0.444** [0.344, 0.531] | [-0.075, +0.128] | 0.390 [0.279, 0.505] | 0.921 [0.88, 0.95] | 30 |
| liberal pool + single best feature | 0.908 | 61 / 262 / 136 | 0.189 | 0.310 | **0.235** [0.129, 0.340] | **[-0.308, -0.081]** | 0.136 [0.080, 0.199] | 0.851 [0.79, 0.89] | 0 |

**What changed vs the 9-scan subset (Jev arms):**
- Every Jev arm moved toward the lane once all 27 test families counted:
  - composite filter 0.370 → 0.426;
  - pool + composite 0.214 → 0.307;
  - pool + composite + GBM 0.338 → 0.442.
- The CIs narrowed. The composite filter's ΔF1 went from [-0.086, +0.084] to [-0.040, +0.037].
- The ranking did not change. Pool + Jev composite is now clearly worse than the lane (ΔF1 [-0.254,
  +0.009]). Pool + typed and pool + typed+reading have CIs that exclude 0 on the negative side.
- Jev pool AUC rose slightly: composite 0.861 → 0.878, typed 0.762 → 0.782, typed+reading 0.779 → 0.792.
  It is still below logistic regression (0.930) and the GBM (0.921), and Jev+GBM (0.901) is still below
  the GBM alone.
- The non-Jev arms are unchanged. They never depended on the answers.
- The only significant gain is still the scan-only single-feature filter over the lane, ΔF1
  [+0.006, +0.042].
- With the second truth (`ai_bend_lines_gold_v1`, now present), these arms give the same values to three
  decimals.

## Recommendation

Do not put Jev into the production line, as a filter or as a rescue ranker. The full v2 test now
confirms it ranks below the plain scan features it is given. On the new engine it has nothing to
work with. E3's lines on the scans it runs are already at the GOLD_SET_V1 ceiling (80/80), and Jev
scores them at chance (AUC 0.51). The frozen composite filter only deletes true bends (−6 TP, 0 FP
removed). Pool rescue with or without Jev recovers 1 of 10 BROAD misses, and those misses are gold
convention conflicts. Rescue costs 0.03–0.08 F1 on BROAD and 0.05–0.10 on GOLD_SET_V1. The one place
rescue raised recall (all-26, R 0.47 → 0.60) is the scans the engine refused in iteration 3, which now
run from iteration 4 on. Re-run `research.jev.e3_ab_v1` on iter_4 once its scorecard exists; the
commands are below and the Jev cost is about $0.02. If rescue still adds nothing on the formerly refused
families, close the Jev track. Spend the effort on engine coverage and the BROAD-gold convention
(apex vs inner skin), not on line filters. If a cheap scan-only filter is wanted for the old research
lane, use the v2 single-feature filter, not Jev. It is not useful on E3 either: it drops 4 true lines.

## Spend and keys

- `TYPESAFE_API_KEY` was read from `.env.local` into the calling Python process only, through
  `input_opt_v2._key_from_env_file`. It was never printed or written.
- Dry-run estimate first: $5.73 for the v2 completion (composite $2.53, typed $1.52, typed+reading
  $1.68) and $0.013 for the engine lines.
- Actual: composite $2.47, typed $1.65, typed+reading $1.84, E3 lines $0.013, so **$5.97 this run**,
  under the $30 cap. No HTTP 402.
- The v2 study total is now $12.14 over all cached responses.
- Every answer is cached by body sha: `jev_ab_v2/answers/<design>/cache/` and `jev_ab_e3/answers/composite/cache/`.

## Caveats

- **The engine ran on only 13 of 26 BROAD cases in iter_3.** The rescue test on the formerly refused
  families (10839000, 23553000, 38172000 staged, 28870000, 11979003 blank, 15651004 SECOND/3RD) is
  pending iter_4.
- **The CIs are small-sample:** 11 families on the engine-ran cohort, 8 on the v2-test subset. Several
  CIs touch 0 exactly because most families do not change.
- **Calibration-family scans.** Three engine-ran scans (15651004, 49204020, 48811000) belong to v2
  calibration families. Their thresholds were fitted there, though their scores are
  leave-one-family-out. The `BROAD_engine_ran_v2test` table excludes them, and the conclusions do not
  change.
- **BROAD gold for Gold11 cases is the human catalog,** as frozen for A1, and its conventions differ
  from the engine's apex locus. GOLD_SET_V1 is the cleaner truth here.
- **The pool is the v2 one:** the old lane's scan-only pool stage, clustered. A pool built for E3 could
  differ.
- **Egress:** the states are coordinate-free and identity-free (tested). Whether ZDR/enterprise terms
  are in place with TypeSafe was not checked.

## Reproduce

```bash
cd ~/ShermanQC/FullProject; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1
O=docs/production/runlog/jev_ab_e3; N=4; W=<A1 iter_N physical work dir, e.g. scratchpad/a1_iter4m/phys>
# 1. engine lines + v2 features (heavy; under the lock)
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
.venv/bin/python -m research.jev.e3_ab_v1 build --a1-work $W --out $O/iter_$N; rc=$?
rmdir /private/tmp/claude-501/sherman_geom.lock
# 2. Jev composite on the engine lines: dry run, then live (cached)
.venv/bin/python -m research.jev.e3_ab_v1 jev --build $O/iter_$N --answers $O/answers
.venv/bin/python -m research.jev.e3_ab_v1 jev --build $O/iter_$N --answers $O/answers --live --env-file .env.local --budget 0.5
# 3. score (offline)
.venv/bin/python -m research.jev.e3_ab_v1 score --build $O/iter_$N --answers $O/answers --out $O/iter_$N/results_iter$N.json
```

Files:
- `iter_3/e3_candidates.jsonl`: the engine lines, with v2 features, pool context, split and engine case.
- `iter_3/engine_manifest.json`: per case, the engine version, config sha, result sha and status.
- `iter_3/results_iter3.json`, `iter_3/build.log`.
- `answers/composite/`: the Jev answers.
- `logs/`: the v2 live log, the final score log, and the 9-scan copies of the v2 results and report.

## Iteration 4 re-check (2026-10-18), track closed

Engine e3-1.5.0, all 26 BROAD A1 cases plus the GOLD_SET_V1 A1 view, frozen v2 designs, live Jev $0.017.
Engine alone: BROAD 0.678 micro / 0.806 family-macro; GOLD_SET_V1 0.977 / 0.966. Every Jev arm is lower
on GOLD_SET_V1 (best: d_jevclf 0.926; composite filter 0.923) and on BROAD family-macro (best: d_lr_jev 0.804).
Jev composite AUC on engine lines 0.590 (n 157). The single-feature filter raises BROAD micro to 0.701
but drops GOLD_SET_V1 to 0.883 and is not Jev. Decision: no Jev in production; the Jev track is closed.
Results: `iter_4/results_iter4.json`.

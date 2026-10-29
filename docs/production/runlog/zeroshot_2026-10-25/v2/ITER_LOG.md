# zs_lane_v2: iteration log (DEV only, out-of-fold)

> **Ledger.** Every number here is consumed development, DEV out-of-fold (125 scans: 120 line scans / 94 families /
> 619 gold lines, plus 5 zero-bend controls; folds from `dev_folds.json`). TEST-B is CONSUMED (zs_lane_v1, 0.855);
> nothing below was tuned on it. TEST-B per-case error records were read only to name the failure classes.

Code: `research/zeroshot/zs_lane_v2/` (v1 copied, v1 never edited; v1 fingerprint `af87df2f…1441c` re-verified).
Tests: `tests/test_prod_zeroshot_lane_v2_contract.py`.

Method:
- Same matcher (frozen `score_case`, 5° / 4 mm / 0.6), same GBM, same fixed threshold 0.5 as v1.
- Post-processing parameters are chosen by **nested** grouped CV: per outer fold, inner out-of-fold GBM
  probabilities on the 4 training folds, best pooled micro-F1 over the grid (ties go to the simplest config).
- Paired family bootstrap (10 000 resamples, seed 20261025) of v2 − v1 against v1's frozen DEV OOF per-case
  records (`integration/oof_v1/DEV_OOF_per_case.json`, arm `zs|fixed|snap`). Exploratory runs used 2 000 resamples.
- Gain rule: the paired CI excludes 0, or a clear class is fixed with no family regression.

## Iteration 0: reproduce v1 from the v2 code path

- v2 = v1 + vendored Z3 (`z3s.py`, default Params identical to v1) + an evidence stage on the Z3 mesh (no effect on
  selection unless used). DEV dump `dump0` (125/125 scans).
- `base|none`: **0.9444**, TP/FP/FN 586/36/33, controls 5 FP, d = 0.0000 vs v1 on every family. Exact reproduction.
- v1 error budget: FN MISSED 13 / SPAN 13 / OFFSET 5 / TAKEN 2; FP SPAN 22 / SPURIOUS 8 / OFFSET 5 / TAKEN 1.

## Iteration 1: class 1 "same two plates?" — flange-continuity test (`evidence.py`, `postproc.py`)

Evidence (computed on the mesh for every pooled candidate, no labels):
- `plate_connected`: for one flank (in-plane direction f, plate normal d × f), vertices on that plate beyond the
  bend zone (a ≥ 2t + 2, |h| ≤ t + 1.5 + 0.03a, normal within ~25°) are rasterised in (s, a) at 1.25 × mesh spacing
  (0.4–1.5 mm), dilated one cell, 8-connected labelling; the plate is continuous when one component reaches both
  sides of the gap. a reaches 1.25 × the larger flank length + 3 mm (12–150 mm), so a plate continuing around a
  window is found, while separate tabs are never joined.
- `collinear_pairs`: each candidate's nearest collinear compatible neighbour beyond its end (≤ 3°, offset
  ≤ 1.5 + 0.4t, |Δincl| ≤ 10°, gap ≤ 250 mm).

Debug on DEV (50757000 B4, 38172000 B1, 48381001): Z3's own `same_plates` had rejected window gaps because a short
flank length on one fragment switched it to the "short flank" band (b1 = 1.1 × 9.6 mm), cell 1.5 mm. The new test
joins 50757000 B4 (both plates connected) and keeps the 138–150 mm-gap tab pairs apart (one plate not connected).
38172000 B1 (a lip cut away for 42 mm, gold = one line) is *not* joined: the lip plate is genuinely interrupted —
the gold labeller treated it as a relief. That case is left as is (gold convention, not geometry).

| Arm (nested CV) | micro F1 [CI] | macro | TP/FP/FN | Δ micro vs v1 [paired CI] | families +/− |
|---|---|---|---|---|---|
| join grid (both / both+low 0.2 / notch 0.8 ± low) | **0.9562** [0.936, 0.975] | 0.9497 | 590/25/29 | **+0.0118 [+0.0008, +0.0263]** | 4 / 0 |

- Inner CV picked `join both + join_low 0.2` in 4 folds, `join both` in 1; never the notch mode.
- SPAN FP 22 → 11, SPAN FN 13 → 9. Families fixed: 50757000 (0.87 → 1.0), 40500000 (0.75 → 0.93), 11979003
  (0.71 → 0.92), 47247004 (0.53 → 0.56). No family worse.

Split, first version (interruption = no bend-zone material **or** a flank missing in the near band): inner CV
rejected it in every fold (0.85–0.92 vs 0.94). 150 of 181 non-connected runs were on correct lines: bend material
present, a flank missing locally (holes, short flanges). Redesigned for iteration 3 (split only where the apex line
itself leaves the surface, tight radius).

## Iteration 2: class 4 — per-station normal sweep across the bisector

- Per 1 mm station, the bend-zone normals are mapped to the angle from flank A's normal (0) to flank B's (θ); the
  station sweeps when ≥ 3 of the 4 inner bins of [0.1θ, 0.9θ] hold a normal (inner-skin normals folded).
  Six per-candidate summaries (`SWEEP_FEATS`).
- Discriminative overall (positives: 5th pct sweep 0.92, both-flank 0.72; negatives: median 0.67 / 0.0).
- As extra GBM features: 0.9421 (Δ −0.0023 [−0.0067, +0.0017], 2 better / 4 worse). **Not adopted** as features.
- Why: the remaining DEV SPURIOUS FPs are not seams. 45141000 P009/P010, 47959001, 49191061 are full 90°
  sweeps (likely real geometry the gold does not count); 38169000 P003/P005 sweep 0.7. Only the 47479000
  control FPs (7° bends, θ < 8° → no sweep) would be removed by a sweep gate; tested as a gate in iteration 3.

## Class 2 — tiny bends next to long folds (diagnosis on DEV)

- Birth is not the problem on DEV: candidate-stream recall is 26/26 for gold < 10 mm and 76/77 for < 20 mm (the
  miss is 48926001 B11, swallowed by a 53 mm over-merged line — a split case).
- Selection is: 10 short gold lines get OOF probability < 0.5 (0.005–0.49).
- An evidence rescue (sweep ≥ 0.8, both flanks ≥ 0.6, bend ≥ 10°, isolated from selected lines) is not
  separable on DEV: 7 positives vs ~370 negatives among p < 0.5 short candidates. **Rejected** (would add
  hundreds of FPs).
- Short-bend birth changes to Z3 (finer step, lower floors) were not attempted: DEV birth recall for short gold is
  already 26/26, so there is nothing on DEV to validate them against (Z3 it5 showed such rule changes lose born
  lines elsewhere).

## Iteration 3: class 3 — small / coarse parts (dump1: `peak_min_area_rel` 0.004 + Z2 fallback)

- Scale-relative Z3 floor min(20, 0.004 × area): changes families on 5 small DEV parts (area < 2 200 mm²).
  `base|none` 0.9436, Δ −0.0008 [−0.0059, +0.0035], 3 better / 3 worse; 37126000 1.0 → 0.67 (new candidates on
  a 758 mm² part give an FP). **Rejected.**
- Replaced by a true fallback (`fallback_frac_only`): only when the default floors give **zero** axis families,
  retry with the fractional floor alone (0.0005 × area, no mm² floor). On DEV it fires on 2 scans (both zero-bend
  controls, 46465000 and 11979003) and finds nothing; every other scan is bit-identical.
- Z2 fallback (`core.z2_fallback`): only when the it08 graph finds no line, rerun once with scale-relative floors
  (region area ≥ max(4 s², min(12, 1e-3 D²)), span ≥ max(3 s, min(8, 0.08 D)); s = point spacing, D = part
  diagonal). Fires on 12 DEV scans, adds 0 lines.
- Both fallbacks are **inert on DEV** (no line changes) — no DEV evidence either way. Kept in the frozen config as
  zero-output safety nets (TEST-B 44001000-type: 0 families, 0 Z2 lines); flagged as unvalidated.

## Iteration 4: split redesign, short-flank join band, join of low-probability pieces, post-join dedupe

Split v2 (dump2): split only where the apex line itself leaves the surface (0.5 mm stations, radius
0.35 mm + 0.5 spacing) and a flank plate is present on both sides, absent inside the run (raw coverage ≤ 0.34)
and not connected. Result on DEV: 9 runs qualify, **6 on correct lines, 0 true splits**. The DEV gold has slots
inside single lines (48381001: a 3 mm full-depth slot at mid-span of a 303 mm gold line; 47251002), and the one
DEV over-merge (48926001 B11, 1.5 mm slot) is bridged at 0.6 mm cells. Inner CV rejects split in every fold.
**Split rejected** — the TEST-B over-merge class (26687021, 15658000) cannot be validated on DEV: DEV contains
almost no over-merge and does contain gold-convention counterexamples.

Short-flank band (`flank_band`): a flank shorter than 2t + 6 mm (taken as the larger of the two pieces' lengths
on that side) is tested on the flank itself (0.4–1.1 × its length). No DEV change by itself.

Join of non-selected pieces (`join_low`): a collinear piece below the threshold may join a selected piece when
both plates are connected. DEV pieces at lattice-window ends have p ≈ 0.001–0.02 (47247004).

| Arm (nested CV, dump2/dump3) | micro F1 [CI] | macro | TP/FP/FN | ctrl FP | Δ micro vs v1 [paired CI] | Δ macro [CI] | +/− |
|---|---|---|---|---|---|---|---|
| join, join_low ∈ {1, 0.2, 0.1, 0.05, 0} | 0.9579 [0.937, 0.976] | 0.9513 | 591/24/28 | 5 | +0.0135 [+0.0020, +0.0279] | +0.0075 [+0.0014, +0.0156] | 5 / 0 |
| join_low fixed 0 (not CV-chosen; diagnostic) | 0.9627 | 0.9549 | 594/21/25 | 5 | +0.0183 [+0.0039, +0.0368] | +0.0110 | 5 / 0 |
| final grid: {no join, join × low {1, 0.2, 0} × dedupe {off, on}}, sweep gate 0.25 | **0.9579** [0.937, 0.976] | **0.9513** | 591/24/28 | **1** | **+0.0135 [+0.0020, +0.0279]** | **+0.0075 [+0.0014, +0.0156]** | 5 / 0 |

- Inner CV picks join_low 0 in 4 folds; fold 3 ties 0.2 = 0 and the declared tie rule takes the first (0.2),
  which leaves 47247004 B1 at 162 of 419 mm. The fixed-0 row shows what that tie costs; it is not the claim.
- Sweep gate (reject a selected Z3 line whose station sweep fraction < 0.25): identical on all 120 line scans for
  0.25 and 0.5 (inner CV ties, as controls are outside micro-F1); removes 4 of 5 control FP lines (47479000, 7°
  kinks). Adopted at 0.25 a priori (the most conservative non-zero value); a clear class fixed, no regression.
- Post-join dedupe (a joined line absorbs a shorter duplicate): picked in 3 folds, ties elsewhere; neutral.

## Freeze (final)

- Final post-processing setting, scored on all 125 DEV scans with the outer OOF probabilities (grid `final`):
  no join 0.9444; join low 1 / 0.2 / 0 = 0.9562 / 0.9579 / 0.9627; with dedupe 0.9562 / 0.9586 / **0.9635**.
  Frozen: `join=True, join_mode=both, join_low=0.0, dedupe_join=True, sweep_min=0.25`; Z3 `fallback_frac_only`,
  Z2 fallback on; selector features `base` (21 Z3 + 5 pool); threshold 0.5.
- **Honest DEV OOF of the procedure** (nested grouped CV over that grid, 10 000 family resamples): micro
  **0.9579 [0.9366, 0.9751]**, family-macro **0.9513 [0.9218, 0.9744]**, TP/FP/FN 591/24/28, controls 1 FP.
  Paired vs v1: **Δ micro +0.0135 [+0.0023, +0.0277]**, **Δ macro +0.0075 [+0.0016, +0.0152]**, 5 better / 0 worse.
  (0.9635 is the score of the chosen setting on the same probabilities — selection-biased, not the claim.)
- Final GBM fit on all DEV: `v2/models/zs_lane_v2_zs_gbm.pkl` (sha `91b36466…f6f6`).
- `config_v2.json` sha `65502f24…0694`; **code fingerprint `39c0efb92c3c3962605168fb2cda977edff5840a2da1cd4518659e7544209097`**
  (63 files, incl. `z3s.py`, `evidence.py`, `postproc.py`).
- DEV CLI consistency (`v2/dev_cli`, frozen CLI, guard + slot, `--strict-pins`): **125/125 identical line sets**
  vs the dump pipeline with the final model, all pins_ok. Wall per process (incl. start-up, one thread): median
  6.5 s, P95 21.3 s, max 31.9 s; peak RSS 3.3 GiB (v1: 6.5 / 17.7 s).
- v1 files and fingerprint `af87df2f…1441c` unchanged (hash diff empty). Tests: v2 contract 7/7, v1 contract 7/7.

## Post-hoc TEST-B re-score — POSTHOC_CONSUMED (not a held-out result)

TEST-B was consumed by zs_lane_v1. Run once after the freeze (`v2/testb_posthoc_CONSUMED/`), nothing changed after.

| | micro [CI] | macro [CI] | TP/FP/FN | Δ micro vs v1 [paired CI] |
|---|---|---|---|---|
| v1 one-time (record) | 0.855 [0.764, 0.926] | 0.855 | 266/39/51 | |
| **v2, POSTHOC_CONSUMED** | 0.8553 [0.7645, 0.9264] | 0.862 [0.791, 0.923] | 266/39/51 | 0.000 [−0.009, +0.010]; macro +0.007 [−0.009, +0.030] |

- 3 joins on TEST-B: 12851000 0.5 → 1.0, 49461000 1.0 → 0.857; error classes unchanged in total
  (FN SPAN 28 / MISSED 16 / OFFSET 7; FP SPAN 24 / SPURIOUS 13 / TAKEN 2).
- 50278001 (window-split 433 mm fold) was not joined, 26687021 over-merge untouched (split rejected on DEV).
- 44001000: the Z3 zero-family fallback fired and the Z2 fallback found 2 lines, but none was selected — still empty.
- Runtime on TEST-B: median 8.1 s, P95 21.9 s, max 30.0 s.
- Reading: the DEV gain (+0.0135) does not transfer to TEST-B; the span class on TEST-B is mostly not window
  gaps the plate test sees as continuous. Diagnosing why (post-hoc) is future work, not a tuning input.

## Status / resume notes (historical)

- DEV dumps (scratchpad, not in repo): dump0 (v1-equivalent + evidence v1), dump1 (rejected floor change),
  dump2 (final evidence), dump3 = dump2 + the 12 fallback scans re-dumped with the fallbacks on (bit-identical
  candidates). Evaluator: `python -m research.zeroshot.zs_lane_v2.train --dump DIR --out DIR --feats base --grid final`.
- Remaining steps: final post-processing choice on all-DEV OOF probabilities; final GBM fit (`--fit-final`);
  `freeze_config.py` → `config_v2.json`; DEV CLI consistency run (`run_cohort.sh`); 10 000-resample DEV report;
  post-hoc consumed TEST-B re-score (`run_cohort.sh … --posthoc-consumed-testb`, `score_posthoc.py`).

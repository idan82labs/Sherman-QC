# Jev input optimisation on the broad benchmark (v2)

Date: 2026-10-18. Follow-up to `docs/production/runlog/jev_ab/JEV_AB.md`. Code: `research/jev/`
(`pool_v2.py`, `features_v2.py`, `questions_v2.py`, `states_v2.py`, `runner_v2.py`, `arms_v2.py`,
`input_opt_v2.py`, `score_v2.py`, `feature_report_v2.py`, `apply_v2.py`), tests
`tests/test_research_jev_input_opt_v2.py`. Machine-readable results: `results.json` in this folder.
No file under `domains/production/**`, `contracts/production/**`, `apps/**` or `frontend/**` was edited.

**Ledger: exploratory, not unseen.** Every scan is a CONSUMED development scan (the 57 scored
scans and 3 zero-bend controls of `case_list_broad_v3.json`). Nothing here is promotion evidence.

## Headline

Better input made Jev a little better, but it still ranks bend candidates worse than the plain
scan measurements it is given. On the same candidates, a logistic regression over those
measurements reaches AUC 0.93, while the best Jev design reaches 0.88 and Jev alone 0.78 to 0.79 (all 37 test scans, after the 2026-10-18 completion run).
A single scan feature (the pool's two-sided transition score, AUC 0.85 to 0.94) also beats every
Jev design. Averaging Jev with the GBM lowered its AUC (0.901 vs 0.921 on the same test scans).
**Plain features beat Jev.**

The liberal pool fixes the structural limit of v1. The old lane's own scan-only candidate pool
holds 83 % of the gold lines (recall 0.858 on test families), against 36 % for the lane's final
lines. A perfect selector over the pool would reach F1 0.92. The best selector found here, a
logistic regression with one-to-one NMS dedupe, reaches F1 0.478 on the 37 test scans, against
0.429 for the lane: recall rises from 0.36 to 0.41. The delta CI is [-0.047, +0.138], so the gain
is not significant. Selecting from the pool is now the bottleneck, not proposing lines.

| arm (test families; truth = accepted AI gold + Gold11 vision relabel) | scans / fam | P | R | micro F1 [95% CI] | ΔF1 vs lane [95% CI] | fam-macro F1 | AUC [95% CI] |
|---|---|---|---|---|---|---|---|
| lane alone (best_f1) | 37 / 27 | 0.530 | 0.360 | **0.429** [0.324, 0.496] | - | 0.348 | - |
| old Jev filter (v1 question, v1 state, t = 0.35 frozen) | 37 / 27 | 0.604 | 0.340 | **0.435** [0.333, 0.500] | [-0.018, +0.028] | 0.314 | 0.718 [0.61, 0.84] |
| new-input Jev filter, composite (frozen choice) | 37 / 27 | 0.584 | 0.335 | **0.426** [0.324, 0.507] | [-0.040, +0.037] | 0.354 | 0.716 [0.58, 0.89] |
| single best feature filter (scan-only) | 37 / 27 | 0.597 | 0.360 | **0.449** [0.338, 0.523] | **[+0.006, +0.042]** | 0.357 | 0.925 [0.84, 0.98] |
| classifier filter (GBM) | 37 / 27 | 0.685 | 0.310 | **0.427** [0.292, 0.518] | [-0.059, +0.044] | 0.315 | 0.831 [0.67, 0.94] |
| liberal pool + Jev composite (frozen choice) | 37 / 27 | 0.333 | 0.284 | **0.307** [0.196, 0.408] | [-0.254, +0.009] | 0.205 | 0.878 [0.85, 0.91] |
| liberal pool + Jev composite + classifier (mean prob.) | 37 / 27 | 0.670 | 0.330 | **0.442** [0.294, 0.561] | [-0.115, +0.133] | 0.310 | 0.901 [0.87, 0.93] |
| liberal pool + logistic regression | 37 / 27 | 0.580 | 0.406 | **0.478** [0.359, 0.567] | [-0.047, +0.138] | 0.361 | 0.930 [0.89, 0.96] |
| liberal pool + GBM | 37 / 27 | 0.544 | 0.376 | **0.444** [0.344, 0.531] | [-0.075, +0.128] | 0.390 | 0.921 [0.88, 0.95] |

Pool ceiling (every pool candidate kept): recall 0.858 on test, 0.797 on calibration, 0.834 on all
57 scans; a perfect selector would give F1 0.92 / 0.89 / 0.91. The lane's final lines cap at
recall 0.360 (perfect filter F1 0.53).

**Update 2026-10-18 (test step completed; see `docs/production/runlog/jev_ab_e3/JEV_AB_E3.md`).**
After the owner added TypeSafe credits, the test-family Jev calls were finished from the cache with the
frozen design choice (`design_choice.json`, sha `26fb8880…`, unchanged). All three designs now have an
answer for every candidate on all 37 scored test scans / 27 families (plus the 3 controls). Nothing
chosen on calibration changed: every keep threshold, the classifier choice (GBM), the composite combiner
(fitted on calibration rows only) and the best feature reproduce exactly (checked arm by arm). The
earlier 9-scan subset version of `results.json` and of this report is kept in
`docs/production/runlog/jev_ab_e3/logs/`. What changed vs the 9-scan subset (test families):

| arm | 9 scans / 7 fam: F1 (ΔF1 CI) | 37 scans / 27 fam: F1 (ΔF1 CI) | AUC on pool, 9 → 37 |
|---|---|---|---|
| lane alone | 0.409 | 0.429 | - |
| Jev filter, composite (frozen) | 0.370 [-0.086, +0.084] | 0.426 [-0.040, +0.037] | 0.861 → 0.878 |
| Jev filter, typed | 0.400 [-0.028, +0.043] | 0.433 [-0.016, +0.024] | 0.762 → 0.782 |
| Jev filter, typed+reading | 0.380 [-0.070, +0.084] | 0.409 [-0.046, +0.011] | 0.779 → 0.792 |
| Jev(composite)+classifier filter | 0.420 [-0.015, +0.084] | 0.432 [-0.019, +0.031] | - |
| pool + Jev composite (frozen) | 0.214 [-0.454, +0.252] | 0.307 [-0.254, +0.009] | - |
| pool + Jev composite + classifier | 0.338 [-0.193, +0.136] | 0.442 [-0.115, +0.133] | 0.889 → 0.901 |
| pool + logistic regression (same cohort) | 0.373 [-0.193, +0.239] | 0.478 [-0.047, +0.138] | 0.931 → 0.930 |

The verdict is unchanged and now rests on the full test families: no Jev arm beats the lane (every
ΔF1 CI contains 0 or is negative), every Jev design ranks below the scan features it reads (pool AUC:
composite 0.878 vs logistic regression 0.930; on lane lines composite 0.716 vs the single feature 0.925),
and averaging Jev into the GBM lowers its AUC (0.901 vs 0.921). The second truth (`gold_set_v1`, which now
exists) gives the same numbers to three decimals on these arms. Total TypeSafe spend for the study is
**$12.14** (all cached responses; $5.96 of it in this completion run, no HTTP 402).

## Recommendation

- **Jev does not earn a place, as a filter or as a ranker, with any input design tried here.**
  Every Jev variant ranks below the scan features it reads. Combining it with the classifier did
  not help. The one place it looked useful (calibration-family F1 of pool + Jev + classifier, 0.542
  vs 0.527 for GBM alone) did not hold on the test families (0.442 vs 0.478 for logistic regression
  alone, all 37 test scans). If Jev is revisited, use the `composite` design (five atomic Nouls combined in code). It
  was the only design that clearly beat the old question.
- **What to use instead:**
  1. **Ranker:** the liberal pool plus a small logistic regression over the typed scan
     measurements, with NMS dedupe. It is the only arm whose recall rises; its F1 gain is not yet
     significant.
  2. **Filter:** if a cheap filter is wanted over a lane's final lines, use the single scan
     feature (pool two-sided transition score ≥ 0.258). It is the only arm whose ΔF1 CI excludes
     0 on test families ([+0.006, +0.042]; all scans [+0.009, +0.036]).
  3. **Priority:** the real headroom is selection quality over the pool. F1 is 0.48 against a
     0.92 ceiling.
- For the new production engine, run `apply_v2` (below) in filter mode on its lines. Use pool mode
  only where the scan-only pool exists for that scan.

## What changed in the input

### 1. The question (`questions_v2.py`, set sha `f63ed466…` for `v2`, `977282ac…` for `composite_v2`)

The primary Noul `is_real_bend` now asks: *"Is the candidate line in `state.candidate` the
outer-skin apex of a real formed press-brake bend?"* Its criteria follow the owner's gold rules
(`gold_set_v1/candidates.json` `gold_rules`):

- **true:** two flat flanges meeting at a formed fold, at any formed angle. Small formed bends
  (about 15 to 30°) count. A joggle (offset step) is two bends, so each of its folds counts. A fold
  interrupted by a window or relief is still the same bend.
- **false:** hole/slot/cutout walls; sheet-thickness edges and the part perimeter; embosses, beads
  and ribs; gentle kinks on rolled or curved surfaces; scan noise, artifacts and seams; the inner
  skin of a fold, or a copy offset by about one sheet thickness.

v1's "joggle / offset step" confuser is gone (a test checks this). Following the TypeSafe docs
(Noul page, advanced structure, jev-1.13 jaggedness), the instructions and criteria are aligned,
the boundary cases are written out, and state fields are referenced by backticked path.

### 2. The evidence (`states_v2.py`)

Each measurement is a typed field `{"value", "unit", "meaning"}`, grouped by what it describes:
cross-section, fold angle, flanges, apex, support, line, and `surroundings`. Each group carries a
one-line `about_this_group`. This follows the docs' structured-field pattern (`name/type/unit/
description`) and keeps only the fields the question needs. The state is fail-closed:

- keys come from a fixed vocabulary, and identity or geometry tokens are denied;
- numbers are finite and rounded to 3 significant digits;
- every string must be one the module wrote, so no text from the data can enter.

So no coordinates, part ids, filenames or SKUs can appear (tested).

Measurements (all scan-only). From a new per-line cross-section pass (`features_v2.py`: 15 slices,
4–40 mm ring):

- two-, one-, many-flange, closed-ring and empty slice shares;
- the dihedral between fitted flange directions, with its IQR and the share of slices within 5°;
- flange fit RMS and flange width on each side, and flange balance;
- line-to-corner distance, corner gap, apex side (+ = convex/outer skin), distance to surface, and
  line support.

From the pool generator: the two-sided transition score and normal angle, corridor line distance
and coverage, and bend radius from cylinder votes. Neighbour context against the scan's pool:
near-identical proposals, parallel lines within 6 mm (other skin) and 6–40 mm (joggle partner, hem),
the nearest parallel offset, and collinear partners across gaps of up to 80 mm (window/relief).

Which measurements separate true from false lines (calibration families, any-match label;
`feature_auc_calib.json`):

| measurement | AUC (pool, 11,306 cand. / 492 true) | true lines | median true | median false |
|---|---|---|---|---|
| pool two-sided transition score | 0.873 | higher | 0.86 | 0.30 |
| parallel candidates within 6 mm (other skin present) | 0.800 | higher | 13 | 8 |
| pool two-sided support balance | 0.799 | higher | 0.98 | 0.86 |
| bend radius (cylinder votes) | 0.794 | lower | 2.0 mm | 4.0 mm |
| near-identical proposals (consensus) | 0.783 | higher | 7 | 4 |
| apex side (+ = outer skin) | 0.774 | higher | +0.85 mm | +0.25 mm |
| corridor points' distance from the line | 0.774 | lower | 1.3 mm | 3.7 mm |
| distance to surface | 0.772 | lower | 1.1 mm | 3.6 mm |
| line-to-corner distance | 0.757 | lower | 3.0 mm | 10.4 mm |
| pool normal-switch angle | 0.748 | higher | 81° | 56° |
| two-flange slice share | 0.735 | higher | 1.00 | 0.80 |
| rougher-flange fit RMS | 0.729 | lower | 1.0 mm | 2.1 mm |
| line length | 0.693 | higher | 97 mm | 50 mm |
| angle stability (share within 5°) | 0.688 | higher | 1.00 | 0.83 |
| one-flange (edge) slice share | 0.672 | lower | 0.00 | 0.07 |

Weak or uninformative (AUC ≤ 0.56): flange widths (capped at 40 mm), corner gap, flatter-flange
RMS, dihedral IQR, collinear-partner counts and the closed-ring / empty slice shares. On the old
lane's final lines, the strongest separators are the transition score (0.805), normal coherence
(0.803), one-flange share (0.790), two-flange share (0.782) and length (0.777).

The first draft of the state lacked the top pool-generator separators. It was revised after this
calibration-only analysis and before any design comparison. About 600 calls made with the draft
state were discarded; their cost is included below.

### 3. Three input designs (all with the v2 question)

| design | state | questions | calib pool AUC | calib lane AUC | test pool AUC, 37 scans [95% CI] |
|---|---|---|---|---|---|
| `typed` | value + unit + meaning | primary Noul | 0.789 | 0.769 | 0.782 [0.75, 0.81] |
| `typed_reading` | plus a `reading` (value in words, fixed physical bands) | primary Noul | 0.776 | 0.703 | 0.792 [0.76, 0.82] |
| **`composite`** (frozen choice) | as `typed_reading` | primary + 4 atomic Nouls: two flat flanges / cut edge or hole / inner or duplicate skin / gentle curvature or artifact; combined by a leave-one-family-out logistic regression on their logits | **0.885** | 0.774 | **0.878** [0.85, 0.91] |
| old v1 (reference) | v1 scalar dump | v1 Noul | - | 0.756 | - |

Turning numbers into words (`reading`), as the jaggedness page suggests, did not help: AUC 0.776
vs 0.789. Splitting the judgment into atomic Nouls and composing them in code (the
composite-scoring pattern) did help, by +0.10 AUC. The design was chosen by calibration pool AUC
and frozen in `design_choice.json` (sha256 `26fb8880…`, file `design_choice.sha256`) before any
test-family call.

### 4. The liberal candidate pool (`pool_v2.py`)

Source: the old lane's scan-only pool stage output (`f1_uniform_scan_only_candidate_pool_v3`;
contract: no part id, no CAD, no truth), still on disk at
`/private/tmp/claude-501/sherman_broad_baseline_work/broad_v2/best_f1/<scan>/pool/BB*.json`. Per
scan it has about 1,000 `candidates` and 700 `line_objects`, about 1,700 rows in all. Rows are
merged into one candidate when near-identical (2°, 1.5 mm, 0.8 mutual overlap). That gives 31,410
candidates on the 60 scans (about 530 per scan), and the merge loses no recall. At selection time
a one-to-one NMS drops a candidate that duplicates a kept, higher-scored one (matcher thresholds:
5°, 4 mm, 0.6 overlap of the shorter span).

Seven STL/OBJ scans were converted to PLY by the pool stage, so the pool sha differs from the
case sha. The coordinates are unchanged: these pools hit gold as well as the others. This is
recorded per row as `pool_scan_converted_from_mesh`. Four scans have tiny or empty pools (the pool
stage failed there).

## Protocol

- **Truth:** case-list gold (accepted AI gold `ai_bend_lines_v1`, 5 bench-label AI scans, and 4
  human non-Gold11 scans), with the 11 Gold11 scans replaced by the vision relabel
  `ai_bend_lines_gold11_relabel_v1` (matched by scan sha256). That is 325 gold lines on 57 scans.
  Under this truth, lane-alone best_f1 is F1 0.458 overall (JEV_AB reported 0.443 under human
  Gold11).
- **Gold set v1:** `ai_bend_lines_gold_v1` and `GOLD_SET_V1.md` did not exist when this ran (only
  `gold_set_v1/candidates.json`), so it is not reported. `score_v2` scores it automatically as a
  second truth once the directory exists, with no new Jev calls.
- **Matching:** the frozen matcher `scripts/evaluate_f1_against_manual_gold.match_lines` (5°,
  4 mm, 0.6/0.6 overlap, one-to-one Hungarian) scores every selected set. AUC uses an
  "any-match" label: the candidate matches some gold line under the same criteria.
- **Split:** the family-disjoint `jev_ab` split (`jev_ab/export/splits.json`): 14 calibration
  families (20 scored scans) and 27 test families (37 scored scans, plus the 3 controls).
- **Calibration-only choices:**
  - input design, classifier type, best single feature and its sign, keep thresholds (maximising
    calibration micro F1), and the composite combiner;
  - calibration scores of trained models are leave-one-family-out; test families use a model
    fitted on all calibration families;
  - the old Jev filter keeps its frozen v1 threshold (0.35).
- **CIs:** family-level bootstrap with 2,000 rounds. Deltas are paired with lane-alone on the same
  resampled families. AUC CIs also resample families.
- **Classifiers:**
  - logistic regression: standardised, C = 0.3, balanced weights;
  - GBM: HistGradientBoosting, depth 3, 250 iterations, min leaf 30;
  - both read 39 measurements plus missing indicators;
  - "Jev + classifier" is the mean of the frozen Jev score and the GBM probability.
- **Heavy geometry:** one build under the shared lock with OMP_NUM_THREADS=1: 31,627 lines
  (pool plus 217 lane lines), 816 s, 2.2 GB peak.
- **Jev calls:**
  - model `jev-1.13.0`, 8 concurrent workers, cached by body sha256 under
    `answers/<design>/cache/`;
  - dry runs first (calibration estimate $3.6; actual usage ran about 12 % above the estimate);
  - the key was read from `.env.local` into the client process only, never printed or written.
- **Spend** (sum of reported usage over every cached response, after the completion run): composite $4.97,
  typed $3.43, typed_reading $3.74, **$12.14 in total**. Before the completion run (the figures below): composite $2.50, typed $1.78
  (including the discarded draft-state calls), typed_reading $1.90, for **$6.18 in total**, under
  the $20 cap.
- **Credits ran out:** the test-family run stopped on HTTP 402 when the organisation ran out of
  credits. Running the two transparency designs in parallel with the frozen design shared the
  remaining credit. The frozen design alone would have covered about three times as many test
  scans. That was a scheduling mistake.


## Caveats

- **(Superseded 2026-10-18: the Jev test evidence is now complete on 37 scans / 27 families.)** Before the completion run it was 9 scans in 7 families, because credits ran out. The Jev-subset CIs are
  very wide. The cross-validated calibration comparison is complete and agrees with it.
- **The lane-filter Jev arms are almost indistinguishable** from each other and from the lane
  (few lines, small deltas). `typed_reading` and `composite` give identical primary answers on
  lane lines: same state, same primary question.
- **Calibration-family F1 values are optimistic** for every fitted arm, because the threshold was
  fitted there. The scores themselves are out-of-family.
- **GBM over the pool flags 30 lines on the 3 zero-bend controls**; logistic regression flags 2.
  The headline ranker is logistic regression partly for this reason. It was chosen by
  calibration micro F1 under the arms rule, not because of the controls.
- **Some new measurements are weak as built:**
  - near-line support uses a 3 × spacing radius on the full-density cloud, which is too tight
    (median 0 for true and false lines alike);
  - flange widths saturate at the 40 mm ring;
  - the corner gap does not separate.
  A second feature pass could fix these. It would not change the Jev verdict: Jev sees the same
  measurements as the classifier.
- **The any-match AUC label** counts several near-duplicate candidates of one gold line as
  positives. F1 always uses the one-to-one matcher.
- **Egress:** the states are coordinate-free and identity-free. As in JEV_AB, whether
  ZDR/enterprise terms with TypeSafe are in place was **not checked**.
- **Nothing here is wired into a runtime decision.**

## Reusable command: apply the winning design to the new production engine

`research/jev/apply_v2.py` takes one scan at a time.

- **Input:** either an E3 output directory or any eval-lines file.
  - E3: `--engine-dir/--compiled/--recipe`. Lines are the engine's measured outer-apex loci in the
    raw scan frame, built by the engine's own `domains.production.engine.eval.line_metric.
    engine_lines(result, compiled, locus="OUTER_APEX", frame="raw")`. This is a read-only import;
    nothing in `domains/` is edited.
  - Eval lines: `--lines` with a `sherman.broad_baseline.eval_lines.v1` file, `frame: scan_raw`.
- **What it does:**
  - computes the same measurements (under the lock), with neighbour context against the scan's
    pool when `--pool` is given;
  - applies `frozen.json`: filter mode = single-feature filter; `--add-from-pool` = pool mode,
    logistic regression + NMS;
  - writes per-line scores and decisions, plus a scorable eval-lines file of the kept lines.
- **Optional Jev:** `--jev-design composite --live --env-file .env.local` adds Jev's answer per
  line as a diagnostic only. It never changes the keep decision.

```bash
cd ~/ShermanQC/FullProject
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
.venv/bin/python -m research.jev.apply_v2 --scan SCAN.ply \
    --engine-dir E3_OUT_DIR --compiled COMPILED.json --recipe RECIPE.json \
    --pool POOL.json [--add-from-pool] --out OUT.json            # or: --lines EVAL_LINES.json
rc=$?; rmdir /private/tmp/claude-501/sherman_geom.lock
# score the kept lines with the frozen matcher
.venv/bin/python research/broad_baseline/score.py single --gold-kind ai_studio --gold GOLD.json \
    --pred OUT.lines.json --out SCORE.json
```

Smoke test (consumed scan `sc_c623d2e45bff`, best_f1 lines plus its pool): filter mode kept 5 of
7 lines (F1 0.50 → 0.60). Pool mode ranked 517 candidates and kept 5 (F1 0.60). Without a pool
the context is measured against the input lines and the pool descriptors are missing. That is a
distribution shift; the output records it as `context_reference`.

To evaluate a whole new lane the way this study does, write one
`predictions_<lane>.json` per scan (eval-lines, `scan_raw`). Then point `LANE`/`LANE_FILE` in
`input_opt_v2.py` at it (or pass the file through `apply_v2` per scan) and re-run `build` and
`score_v2`. The pool and features caches are reused.

## Reproduce

```bash
cd ~/ShermanQC/FullProject; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1
J=docs/production/runlog/jev_ab_v2
# 1. candidates + features + context (heavy; under the lock)
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
.venv/bin/python -m research.jev.input_opt_v2 build --out $J/build; rc=$?; rmdir /private/tmp/claude-501/sherman_geom.lock
# 2. dry run (no network), then live calibration calls per design
for d in typed typed_reading composite; do
  .venv/bin/python -m research.jev.input_opt_v2 jev --build $J/build --answers $J/answers --design $d --split calib
  .venv/bin/python -m research.jev.input_opt_v2 jev --build $J/build --answers $J/answers --design $d --split calib \
      --live --env-file .env.local --budget 2.5; done
# 3. freeze the design choice (design_choice.json + .sha256), then test families (refused without the file)
.venv/bin/python -m research.jev.input_opt_v2 jev --build $J/build --answers $J/answers --design composite \
    --split eval --design-choice $J/design_choice.json --live --env-file .env.local --budget 4
# 4. score (offline; cached answers only)
.venv/bin/python -m research.jev.score_v2 --build $J/build --answers $J/answers \
    --design-choice $J/design_choice.json --out $J/results.json
.venv/bin/python -m research.jev.feature_report_v2 --build $J/build --out $J/feature_auc_calib.json
```

The Jev test evidence was finished this way on 2026-10-18 (re-run of step 3 for each design, then step 4); the 4,789 answered
bodies come from the cache. Then re-run step 4. The design choice stays frozen.

## Files

- `build/candidates.jsonl`: one row per candidate (pool and lane), with scan-frame endpoints (a
  local sidecar, never sent), measurements, context and split. `build/features_cache/`: per-scan
  feature cache. `build/build.log`.
- `answers/<design>/cache/`: raw responses keyed by body sha. `answers/<design>/dry_run_*/`:
  sample bodies and cost estimates. `answers/<design>/live_log.jsonl`.
- `design_choice.json` and `.sha256`: frozen before any test call. `calib_only_results.json`: the
  calibration-only scoring that produced the choice.
- `results.json`: every arm × truth × cohort (TEST, TEST_JEV_COMPLETE, CALIB, ALL), CIs, AUCs,
  ceilings, thresholds and spend. `feature_auc_calib.json`. `frozen.json`: the operating points
  used by `apply_v2`.

## Full tables (truth: AI gold + Gold11 relabel)

#### Test families, all 37 scored test scans / 27 families (all Jev answers complete since the 2026-10-18 completion run)

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

#### Calibration families (20 scans / 14 families). Trained scores are leave-one-family-out; thresholds were fitted here

| arm | keep t | TP / FP / FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs lane [95% CI] | family-macro F1 [95% CI] | AUC (any-match) | control FP lines |
|---|---|---|---|---|---|---|---|---|---|
| lane alone (best_f1) | - | 52 / 26 / 76 | 0.667 | 0.406 | **0.505** | - | 0.554 | - | 0 |
| old Jev filter (v1 question + v1 state) | 0.35 | 48 / 11 / 80 | 0.814 | 0.375 | **0.513** | - | 0.547 | 0.756 | 0 |
| new-input Jev filter: typed | 0.196 | 52 / 19 / 76 | 0.732 | 0.406 | **0.523** | - | 0.568 | 0.769 | 0 |
| new-input Jev filter: typed+reading | 0.244 | 50 / 17 / 78 | 0.746 | 0.391 | **0.513** | - | 0.560 | 0.703 | 0 |
| new-input Jev filter: composite (frozen choice) | 0.307 | 52 / 17 / 76 | 0.754 | 0.406 | **0.528** | - | 0.572 | 0.774 | 0 |
| Jev(composite)+classifier filter | 0.212 | 52 / 16 / 76 | 0.765 | 0.406 | **0.531** | - | 0.577 | 0.801 | 0 |
| classifier filter (GBM) | 0.131 | 51 / 14 / 77 | 0.785 | 0.398 | **0.528** | - | 0.588 | 0.825 | 0 |
| single best feature filter | 0.258 | 52 / 19 / 76 | 0.732 | 0.406 | **0.523** | - | 0.570 | 0.805 | 0 |
| liberal pool + pool generator prior | 6.45 | 33 / 41 / 95 | 0.446 | 0.258 | **0.327** | - | 0.293 | 0.880 | 0 |
| liberal pool + Jev typed | 0.72 | 58 / 115 / 70 | 0.335 | 0.453 | **0.385** | - | 0.420 | 0.789 | 0 |
| liberal pool + Jev typed+reading | 0.74 | 53 / 111 / 75 | 0.323 | 0.414 | **0.363** | - | 0.406 | 0.776 | 0 |
| liberal pool + Jev composite (frozen choice) | 0.941 | 61 / 117 / 67 | 0.343 | 0.477 | **0.399** | - | 0.437 | 0.885 | 0 |
| liberal pool + Jev(composite)+classifier | 0.715 | 61 / 36 / 67 | 0.629 | 0.477 | **0.542** | - | 0.590 | 0.913 | 0 |
| liberal pool + logistic regression | 0.979 | 56 / 42 / 72 | 0.571 | 0.438 | **0.496** | - | 0.532 | 0.917 | 0 |
| liberal pool + GBM | 0.533 | 63 / 48 / 65 | 0.568 | 0.492 | **0.527** | - | 0.577 | 0.943 | 0 |
| liberal pool + single best feature | 0.908 | 40 / 124 / 88 | 0.244 | 0.312 | **0.274** | - | 0.296 | 0.873 | 0 |

#### All 57 scored scans (calibration + test; calibration rows are leave-one-family-out, thresholds fitted there)

| arm | keep t | TP / FP / FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs lane [95% CI] | family-macro F1 [95% CI] | AUC (any-match) | control FP lines |
|---|---|---|---|---|---|---|---|---|---|
| lane alone (best_f1) | - | 123 / 89 / 202 | 0.580 | 0.379 | **0.458** [0.388, 0.519] | - | 0.418 [0.318, 0.522] | - | 5 |
| old Jev filter (v1 question + v1 state) | 0.35 | 115 / 55 / 210 | 0.676 | 0.354 | **0.465** [0.390, 0.527] | [-0.011, +0.023] | 0.394 [0.301, 0.494] | - | 1 |
| new-input Jev filter: typed | 0.196 | 121 / 72 / 204 | 0.627 | 0.372 | **0.467** [0.395, 0.529] | [-0.004, +0.022] | 0.420 [0.319, 0.522] | - | 5 |
| new-input Jev filter: typed+reading | 0.244 | 113 / 65 / 212 | 0.635 | 0.348 | **0.449** [0.381, 0.512] | [-0.029, +0.011] | 0.412 [0.311, 0.513] | - | 5 |
| new-input Jev filter: composite (frozen choice) | 0.307 | 118 / 64 / 207 | 0.648 | 0.363 | **0.466** [0.392, 0.534] | [-0.018, +0.034] | 0.428 [0.326, 0.533] | - | 5 |
| Jev(composite)+classifier filter | 0.212 | 119 / 62 / 206 | 0.657 | 0.366 | **0.470** [0.397, 0.534] | [-0.004, +0.031] | 0.429 [0.328, 0.534] | - | 5 |
| classifier filter (GBM) | 0.131 | 112 / 42 / 213 | 0.727 | 0.345 | **0.468** [0.377, 0.544] | [-0.025, +0.038] | 0.408 [0.307, 0.515] | - | 3 |
| single best feature filter | 0.258 | 123 / 67 / 202 | 0.647 | 0.379 | **0.478** [0.406, 0.540] | **[+0.009, +0.036]** | 0.430 [0.329, 0.534] | - | 3 |
| liberal pool + pool generator prior | 6.45 | 78 / 124 / 247 | 0.386 | 0.240 | **0.296** [0.217, 0.359] | **[-0.231, -0.092]** | 0.246 [0.176, 0.312] | - | 0 |
| liberal pool + Jev typed | 0.72 | 115 / 261 / 210 | 0.306 | 0.354 | **0.328** [0.254, 0.397] | **[-0.209, -0.045]** | 0.285 [0.201, 0.372] | - | 0 |
| liberal pool + Jev typed+reading | 0.74 | 108 / 218 / 217 | 0.331 | 0.332 | **0.332** [0.273, 0.382] | **[-0.193, -0.058]** | 0.267 [0.196, 0.337] | - | 0 |
| liberal pool + Jev composite (frozen choice) | 0.941 | 117 / 229 / 208 | 0.338 | 0.360 | **0.349** [0.263, 0.426] | **[-0.202, -0.009]** | 0.284 [0.201, 0.369] | - | 0 |
| liberal pool + Jev(composite)+classifier | 0.715 | 126 / 68 / 199 | 0.649 | 0.388 | **0.485** [0.376, 0.576] | [-0.073, +0.125] | 0.406 [0.302, 0.509] | - | 9 |
| liberal pool + logistic regression | 0.979 | 136 / 100 / 189 | 0.576 | 0.418 | **0.485** [0.387, 0.566] | [-0.043, +0.088] | 0.419 [0.317, 0.528] | - | 2 |
| liberal pool + GBM | 0.533 | 137 / 110 / 188 | 0.555 | 0.421 | **0.479** [0.396, 0.551] | [-0.059, +0.111] | 0.454 [0.365, 0.539] | - | 30 |
| liberal pool + single best feature | 0.908 | 101 / 386 / 224 | 0.207 | 0.311 | **0.249** [0.177, 0.318] | **[-0.300, -0.118]** | 0.191 [0.136, 0.249] | - | 0 |

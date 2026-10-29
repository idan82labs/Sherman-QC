# Jev A/B on the broad benchmark (wave 4)

Date: 2026-10-17. Code: `research/jev/broad_ab_v1.py` (new) plus the existing harness
`research/jev/cli.py` (dry-run and live). Inputs: `case_list_broad_v3.json` (57 scored scans
in 41 families, plus 3 zero-bend controls) and the frozen lane predictions in
`docs/production/runlog/broad_baseline/runs/{best_f1,runtime}/`. No engine code was touched.
Machine-readable results are in `results.json` (same folder).

**Ledger: exploratory, not unseen.** Every scan is a CONSUMED development scan. These numbers
are not promotion evidence and say nothing about FRESH-family accuracy.

## Headline

A Jev filter does **not** move line-F1 on either lane. It removes about a third of the false
lines and roughly 8 % of the true lines, so precision rises by about 8 points, recall falls
by about 3 points, and F1 stays flat. Every F1 delta CI contains 0.

| lane | arm | keep t (from calib) | all 57: P / R / micro F1 | all: family-macro F1 | non-calib families (37 scans / 27 fam): micro F1 | non-calib: family-macro F1 | Jev AUC, all [95% CI] | Jev AUC, non-calib [95% CI] |
|---|---|---|---|---|---|---|---|---|
| best_f1 | (a) lane alone | - | 0.542 / 0.375 / **0.443** | 0.414 | **0.443** | 0.354 | - | - |
| best_f1 | (b) + Jev | 0.35 | 0.624 / 0.345 / **0.444** (Δ CI [-0.016, +0.017]) | 0.381 (Δ CI [-0.090, +0.004]) | **0.444** (Δ CI [-0.028, +0.021]) | 0.308 (Δ CI [-0.134, +0.005]) | 0.694 [0.619, 0.769] | 0.709 [0.611, 0.809] |
| best_f1 | (c) + Jev CAD-free | 0.61 | 0.620 / 0.345 / **0.444** (Δ CI [-0.024, +0.028]) | 0.425 (Δ CI [-0.005, +0.027]) | **0.421** (Δ CI [-0.051, +0.013]) | 0.350 (Δ CI [-0.021, +0.011]) | 0.721 [0.649, 0.814] | 0.675 [0.583, 0.787] |
| runtime | (a) lane alone | - | 0.328 / 0.068 / **0.113** | 0.119 | **0.118** | 0.123 | - | - |
| runtime | (b) + Jev | 0.57 | 0.400 / 0.065 / **0.112** (Δ CI [-0.010, +0.005]) | 0.123 | **0.116** | 0.128 | 0.709 [0.555, 0.864] | 0.797 [0.678, 0.973] |
| runtime | (c) + Jev CAD-free | 0.57 | 0.392 / 0.065 / **0.112** (Δ CI [-0.011, +0.005]) | 0.123 | **0.115** | 0.128 | 0.743 [0.581, 0.881] | 0.804 [0.622, 0.944] |

How to read it:

- **A filter can only remove lines, so recall can never rise.** The best possible filter
  (one that removes all false lines and keeps every true line) would bring best_f1 to
  P = 1.0, R = 0.375, F1 = 0.545 on all 57 scans. The runtime lane would reach F1 = 0.128.
  The 0.80 target cannot be reached by filtering: about 63 % of the gold lines are never
  proposed (192 of 307 for best_f1). Recall has to come from the engine.
- On best_f1, Jev removes 37 of 102 false lines and 9 of 115 true lines at t = 0.35. In the
  zero-bend controls it cuts false lines from 5 to 1. The CAD-free variant removes 32 false
  lines and 9 true lines.
- Plain Jev at its native t = 0.5 (no fitted threshold) is clearly harmful on best_f1:
  F1 0.381, Δ CI [-0.123, -0.017], and family-macro F1 0.305. It rejects many true lines
  because its median score is only 0.58 on these states.
- Jev's ranking is only moderate (AUC about 0.69 to 0.72 on best_f1). The single best scan
  descriptor in the same state, `two_arm_slice_fraction`, has AUC 0.739 by itself over the
  best_f1 candidates. That number was picked after the fact from 16 descriptors over all
  candidates, so it is optimistic. Still, Jev does not add a visible signal beyond the
  evidence it is given. This repeats the earlier finding: Jev ranks within a scan reasonably
  well, but it is not a usable bend/not-bend classifier.
- The CAD-assisted (normal) and CAD-free variants are within noise of each other. Adding the
  CAD relation block did not help Jev; its AUC is a little lower with it.
- Gold11 (human labels): best_f1 0.458 → 0.468 (+Jev) and 0.441 (+Jev CAD-free). Both CIs
  contain 0. `gold11_relabel_v1` does not exist yet (the audit workflow has not written
  `ai_bend_lines_gold11_relabel_v1`), so only human Gold11 is reported. Once the relabel lands,
  re-score with `--gold-override` (below). That needs no new Jev calls.

Verdict: Jev-assisted does not beat non-Jev on line-F1. A Jev filter is at best a
precision/REVIEW aid and is not a path to the 0.80 gate. The answers are cached, so re-scoring
costs nothing.

## Protocol

1. **Candidates.** Every predicted line of each lane is one candidate: 217 for best_f1
   (212 on the scored scans and 5 on controls) and 67 for runtime, 284 in total. Each
   candidate is labelled TP or FP with the frozen matcher (`research/broad_baseline/score.py`
   `score_case` → `scripts/evaluate_f1_against_manual_gold.match_lines`: 5 deg, 4 mm,
   0.6/0.6 overlap, one-to-one Hungarian). The export reproduces the lane scores exactly:
   best_f1 has 115 TP and 97 + 5 FP, runtime 21 TP and 43 + 3 FP.
   - Gold drift since the broad baseline: 42288005 (`sc_2233372b8df7`) was relabelled from
     6 to 8 bends (two 15 deg joggles), so gold is now 307 lines, not 305. Lane-alone
     best_f1 is therefore 0.443, not the 0.445 in BROAD_BASELINE.
2. **State.** Built per candidate with `state_v1` and its fail-closed deny-list. It has no
   coordinates, no identities and no filename or part tokens.
   - `candidate`: scan evidence measured directly on the scan around the line, by the same
     procedure for every lane. At 21 slices along the line it records near-line support, the
     2-D cross-section "arms" (flanges) and the included angle between them, arm balance,
     spread (thickness proxy) and reach, distance to the surface and point density.
   - `neighbours`: `neighbour_summary` over the lane's lines of that scan.
   - `context`: lane line count, cloud extent, sampling and size.
   - `cad_relation` (dropped by `--cad-free`): proposal source (scan-only or CAD-guided), CAD
     binding, nominal and measured angle and radius, legacy status, observability and
     assignment, CAD planned-bend count, and corroboration by the other proposal source.
   - Uniform `candidate_source` = `SCAN_LEGACY_REPORT`.
   - Feature extraction ran under the shared geometry lock in 39 s, with peak RSS 3.06 GB.
3. **Split** (`export/splits.json`, `jev_split_manifest_v1`). It is family-disjoint: the
   sha256 order of `jev_ab_v1_calib|<family>` assigns the first 14 families to **calib** and
   the other 28 to **eval**. Calib families: 10839000, 15651004, 42288005, 47247004,
   47251002, 47555020, 48811000, 48963008, 48991006, 49001000, 49204020, 50553000,
   UNKNOWN:sc_f362d8de507c and ysherman.
4. **Calls.** The question set is `is_real_bend`, v1 (sha `335e0dc6…`), model `jev-1.13.0`.
   - Dry-run first. All 284 records in both variants came to an estimated 357,772 + 283,329
     tokens, about $0.027 at $0.042/M.
   - Live, calib only (95 records per variant). Then the thresholds were frozen in
     `operating_points.json` (sha `21fa881f…`, recorded in
     `answers/operating_points_frozen_before_eval.sha256` before any eval call). Only then
     were the eval records sent with `--include-eval`.
   - Actual usage was 415,733 input tokens (normal) + 333,750 (CAD-free), about $0.031 in
     total, with 0 API errors. The key was loaded from `.env.local`
     into the client process environment only and was never printed or written.
   - Answers and raw responses are cached by body sha in `answers/{normal,cad_free}/`.
5. **Threshold.** Per lane and variant, the threshold is the one that maximises calib-family
   micro line-F1 of lane + filter. The grid is the calib scores plus 0, and ties go to the
   lowest threshold. It is fitted on calib only, then applied unchanged everywhere. The score
   step re-derives it from the calib answers and checks that it equals the frozen value
   (`refit_on_calib_equals_frozen` = true for all four).
6. **Metrics.**
   - The filtered lane is re-scored with the same matcher.
   - Micro P/R/F1 sums TP/FP/FN over complete-label scans.
   - Family-macro F1 is the mean of family-micro F1.
   - 95 % CIs come from a family-level bootstrap (2000 rounds). Deltas are paired on the same
     resampled families.
   - Jev AUC uses a family-level bootstrap.
   - Zero-bend controls are reported as FP line counts, never in F1.

## Caveats

- **Question criteria conflict with owner conventions.** The frozen `is_real_bend` v1
  criteria list "a joggle, offset step" as a confuser (`questions_v1.CONFUSER_KINDS["step"]`),
  but the owner's convention is joggle = 2 bends. Jev was therefore told to reject some lines
  that count as true. A new question variant would need its own hash and run, and it was not
  done here.
- The runtime lane fails or refuses on 40 of 60 scans, so its candidate pool is small
  (67 lines and 21 positives), and its CIs are wide.
- 42 of the 57 scored scans have AI gold and 15 have human gold; they are pooled here, as in BROAD_BASELINE.
- Egress: the harness egress policy (`docs/research/JEV_BEND_CANDIDATE_DATASET_V1.md`) asks
  for owner approval of the specific export and for ZDR/enterprise terms with TypeSafe. The
  owner asked for this A/B on the 57 scans. Whether ZDR terms are in place was **not checked**
  in this run. The states sent are coordinate-free and identity-free.
- Jev output stays research data. Nothing here is wired into a runtime decision.

## Re-run: the same A/B on another lane (e.g. the new production engine)

The only input a lane needs is one eval-lines file per scan:
`<RUNS>/<lane>/<case_id>/predictions_<lane>.json`.

- Schema: `sherman.broad_baseline.eval_lines.v1`.
- `frame` must be `scan_raw`.
- `owned_bend_regions[]` rows have `{bend_id, line_start, line_end}` in raw scan coordinates.
  These optional keys are copied into `cad_relation`: `origin`, `status`,
  `observability_state`, `assignment_source`, `detected_line_geometry_source`,
  `measured_angle` and `target_angle`.
- For E3 output, write the measured visible-apex locus (`measured_outer_apex`, mapped with
  `frame_to_raw` from `domains/production/engine/eval/engine_output.py`) as the line.

Then run:

```bash
cd ~/ShermanQC/FullProject
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1
LANE=e3; PF=predictions_e3.json; RUNS=docs/production/runlog/broad_baseline/runs   # or wherever the lane lives
J=docs/production/runlog/jev_ab_$LANE; mkdir -p $J
# 1. export candidates + states + labels + family split (heavy part: scan reads; under the shared lock)
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
.venv/bin/python -m research.jev.broad_ab_v1 export --runs-root $RUNS --lane $LANE:$PF \
    --out $J/export --features-cache $J/features_cache; rc=$?; rmdir /private/tmp/claude-501/sherman_geom.lock
# 2. dry-run cost estimate (no network)
for v in normal cad_free; do f=; [ $v = cad_free ] && f=--cad-free
  .venv/bin/python -m research.jev.cli dry-run --states $J/export/states.jsonl --splits $J/export/splits.json \
      --labels $J/export/labels.jsonl --include-eval $f --out $J/dry_run/$v; done
# 3. live, calib families only (key into this process env only; never echoed)
( export TYPESAFE_API_KEY="$(sed -n 's/^TYPESAFE_API_KEY=//p' .env.local | head -1 | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
  for v in normal cad_free; do f=; [ $v = cad_free ] && f=--cad-free
    .venv/bin/python -m research.jev.cli live --live --states $J/export/states.jsonl --splits $J/export/splits.json \
        --labels $J/export/labels.jsonl $f --out $J/answers/$v; done )
# 4. freeze thresholds on calib
.venv/bin/python -m research.jev.broad_ab_v1 score --runs-root $RUNS --export $J/export \
    --answers-normal $J/answers/normal --answers-cad-free $J/answers/cad_free --freeze-out $J/operating_points.json
# 5. live, eval families (calib answers come from the cache), same subshell pattern as step 3 plus --include-eval
# 6. score
.venv/bin/python -m research.jev.broad_ab_v1 score --runs-root $RUNS --export $J/export \
    --answers-normal $J/answers/normal --answers-cad-free $J/answers/cad_free \
    --operating-points $J/operating_points.json --out $J/results.json
```

To re-score this run against the Gold11 relabel once it exists, use no new calls. Write
`{case_id: {"kind": "ai_studio", "path": "data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1/<asset>.json"}}`
for the 11 Gold11 cases. Run step 4 with that `--gold-override MAP.json` into a new
`operating_points` file, because the calib labels change. Then run step 6 with the same
override.

## Files

- `export/`: `states.jsonl`, `labels.jsonl`, `manifest.json` (sha `1025bc36…`),
  `candidates.jsonl` (sidecar with lane, case, bend id and split; never sent), `splits.json`.
- `features_cache/`: per-scan descriptor cache.
- `dry_run/{normal,cad_free}/`: the exact request bodies and the cost summary.
- `answers/{normal,cad_free}/`: `answers.jsonl`, `summary.json` and `cache/` (raw responses by
  body sha), plus the live logs.
- `operating_points.json`: the calib-frozen thresholds and the threshold traces.
- `results.json`: every arm, cohort, CI and Jev classification block.

## Full tables


#### All 57 scans / 41 families

| lane | arm | keep t | TP | FP | FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs (a) [95% CI] | family-macro F1 [95% CI] | Δ fam-macro [95% CI] | control FP lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_f1 | (a) lane alone | - | 115 | 97 | 192 | 0.542 | 0.375 | 0.443 [0.363, 0.513] | - | 0.414 [0.314, 0.521] | - | 5 |
| best_f1 | (b) lane + Jev filter | 0.35 | 106 | 64 | 201 | 0.624 | 0.345 | 0.444 [0.367, 0.512] | [-0.016, 0.017] | 0.381 [0.290, 0.482] | [-0.090, 0.004] | 1 |
| best_f1 | (c) lane + Jev CAD-free filter | 0.61 | 106 | 65 | 201 | 0.620 | 0.345 | 0.444 [0.367, 0.516] | [-0.024, 0.028] | 0.425 [0.320, 0.536] | [-0.005, 0.027] | 5 |
| best_f1 | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 82 | 41 | 225 | 0.667 | 0.267 | 0.381 [0.283, 0.462] | [-0.123, -0.017] | 0.305 [0.218, 0.404] | [-0.189, -0.040] | 0 |
| best_f1 | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 111 | 85 | 196 | 0.566 | 0.362 | 0.441 [0.362, 0.515] | [-0.021, 0.014] | 0.419 [0.318, 0.525] | [-0.006, 0.015] | 5 |
| runtime | (a) lane alone | - | 21 | 43 | 286 | 0.328 | 0.068 | 0.113 [0.050, 0.189] | - | 0.119 [0.051, 0.200] | - | 3 |
| runtime | (b) lane + Jev filter | 0.57 | 20 | 30 | 287 | 0.400 | 0.065 | 0.112 [0.052, 0.187] | [-0.010, 0.005] | 0.123 [0.053, 0.205] | [-0.002, 0.013] | 3 |
| runtime | (c) lane + Jev CAD-free filter | 0.57 | 20 | 31 | 287 | 0.392 | 0.065 | 0.112 [0.051, 0.187] | [-0.011, 0.005] | 0.123 [0.053, 0.205] | [-0.002, 0.013] | 3 |
| runtime | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 21 | 34 | 286 | 0.382 | 0.068 | 0.116 [0.051, 0.194] | [0.000, 0.007] | 0.122 [0.052, 0.204] | [0.000, 0.006] | 3 |
| runtime | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 20 | 39 | 287 | 0.339 | 0.065 | 0.109 [0.050, 0.185] | [-0.013, 0.001] | 0.120 [0.051, 0.201] | [-0.003, 0.005] | 3 |

#### Non-calibration families only (37 scans / 27 families)

| lane | arm | keep t | TP | FP | FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs (a) [95% CI] | family-macro F1 [95% CI] | Δ fam-macro [95% CI] | control FP lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_f1 | (a) lane alone | - | 72 | 62 | 119 | 0.537 | 0.377 | 0.443 [0.337, 0.512] | - | 0.354 [0.227, 0.493] | - | 5 |
| best_f1 | (b) lane + Jev filter | 0.35 | 67 | 44 | 124 | 0.604 | 0.351 | 0.444 [0.335, 0.511] | [-0.028, 0.021] | 0.308 [0.191, 0.429] | [-0.134, 0.005] | 1 |
| best_f1 | (c) lane + Jev CAD-free filter | 0.61 | 63 | 45 | 128 | 0.583 | 0.330 | 0.421 [0.322, 0.495] | [-0.051, 0.013] | 0.349 [0.223, 0.493] | [-0.021, 0.011] | 5 |
| best_f1 | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 55 | 28 | 136 | 0.663 | 0.288 | 0.402 [0.267, 0.490] | [-0.117, 0.007] | 0.251 [0.140, 0.368] | [-0.219, -0.021] | 0 |
| best_f1 | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 68 | 55 | 123 | 0.553 | 0.356 | 0.433 [0.328, 0.508] | [-0.038, 0.012] | 0.354 [0.229, 0.494] | [-0.013, 0.013] | 5 |
| runtime | (a) lane alone | - | 14 | 33 | 177 | 0.298 | 0.073 | 0.118 [0.030, 0.231] | - | 0.123 [0.034, 0.242] | - | 3 |
| runtime | (b) lane + Jev filter | 0.57 | 13 | 21 | 178 | 0.382 | 0.068 | 0.116 [0.031, 0.224] | [-0.014, 0.010] | 0.128 [0.037, 0.249] | [-0.004, 0.018] | 3 |
| runtime | (c) lane + Jev CAD-free filter | 0.57 | 13 | 23 | 178 | 0.361 | 0.068 | 0.115 [0.031, 0.223] | [-0.015, 0.008] | 0.128 [0.037, 0.249] | [-0.004, 0.018] | 3 |
| runtime | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 14 | 25 | 177 | 0.359 | 0.073 | 0.122 [0.032, 0.239] | [0.000, 0.011] | 0.127 [0.037, 0.245] | [0.000, 0.009] | 3 |
| runtime | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 13 | 29 | 178 | 0.309 | 0.068 | 0.112 [0.030, 0.219] | [-0.018, 0.002] | 0.124 [0.035, 0.243] | [-0.004, 0.007] | 3 |

#### Calibration families (20 scans / 14 families; in-sample, threshold was fitted here)

| lane | arm | keep t | TP | FP | FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs (a) [95% CI] | family-macro F1 [95% CI] | Δ fam-macro [95% CI] | control FP lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_f1 | (a) lane alone | - | 43 | 35 | 73 | 0.551 | 0.371 | 0.443 [0.329, 0.598] | - | 0.531 [0.390, 0.677] | - | 0 |
| best_f1 | (b) lane + Jev filter | 0.35 | 39 | 20 | 77 | 0.661 | 0.336 | 0.446 [0.335, 0.604] | [-0.025, 0.027] | 0.522 [0.388, 0.667] | [-0.049, 0.025] | 0 |
| best_f1 | (c) lane + Jev CAD-free filter | 0.61 | 43 | 20 | 73 | 0.682 | 0.371 | 0.480 [0.359, 0.648] | [0.015, 0.071] | 0.572 [0.426, 0.719] | [0.013, 0.072] | 0 |
| best_f1 | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 27 | 13 | 89 | 0.675 | 0.233 | 0.346 [0.230, 0.496] | [-0.213, -0.017] | 0.410 [0.273, 0.575] | [-0.245, -0.003] | 0 |
| best_f1 | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 43 | 30 | 73 | 0.589 | 0.371 | 0.455 [0.336, 0.618] | [0.002, 0.029] | 0.543 [0.400, 0.693] | [0.003, 0.026] | 0 |
| runtime | (a) lane alone | - | 7 | 10 | 109 | 0.412 | 0.060 | 0.105 [0.031, 0.207] | - | 0.111 [0.029, 0.214] | - | 0 |
| runtime | (b) lane + Jev filter | 0.57 | 7 | 9 | 109 | 0.438 | 0.060 | 0.106 [0.031, 0.210] | [0.000, 0.004] | 0.113 [0.029, 0.215] | [0.000, 0.005] | 0 |
| runtime | (c) lane + Jev CAD-free filter | 0.57 | 7 | 8 | 109 | 0.467 | 0.060 | 0.107 [0.032, 0.210] | [0.000, 0.005] | 0.113 [0.029, 0.215] | [0.000, 0.005] | 0 |
| runtime | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 7 | 9 | 109 | 0.438 | 0.060 | 0.106 [0.031, 0.210] | [0.000, 0.004] | 0.113 [0.029, 0.215] | [0.000, 0.005] | 0 |
| runtime | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 7 | 10 | 109 | 0.412 | 0.060 | 0.105 [0.031, 0.207] | [0.000, 0.000] | 0.111 [0.029, 0.214] | [0.000, 0.000] | 0 |

#### Gold11 (11 scans, human Gold11 labels)

| lane | arm | keep t | TP | FP | FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs (a) [95% CI] | family-macro F1 [95% CI] | Δ fam-macro [95% CI] | control FP lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_f1 | (a) lane alone | - | 30 | 30 | 41 | 0.500 | 0.422 | 0.458 [0.283, 0.577] | - | 0.460 [0.328, 0.588] | - | 0 |
| best_f1 | (b) lane + Jev filter | 0.35 | 29 | 24 | 42 | 0.547 | 0.408 | 0.468 [0.292, 0.571] | [-0.035, 0.028] | 0.448 [0.345, 0.549] | [-0.077, 0.029] | 0 |
| best_f1 | (c) lane + Jev CAD-free filter | 0.61 | 26 | 21 | 45 | 0.553 | 0.366 | 0.441 [0.314, 0.538] | [-0.052, 0.032] | 0.460 [0.344, 0.576] | [-0.026, 0.022] | 0 |
| best_f1 | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 25 | 19 | 46 | 0.568 | 0.352 | 0.435 [0.167, 0.560] | [-0.150, 0.008] | 0.381 [0.242, 0.499] | [-0.171, 0.003] | 0 |
| best_f1 | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 29 | 29 | 42 | 0.500 | 0.408 | 0.450 [0.289, 0.560] | [-0.019, 0.019] | 0.462 [0.334, 0.583] | [-0.012, 0.018] | 0 |
| runtime | (a) lane alone | - | 3 | 7 | 68 | 0.300 | 0.042 | 0.074 [0.015, 0.286] | - | 0.178 [0.056, 0.311] | - | 0 |
| runtime | (b) lane + Jev filter | 0.57 | 3 | 5 | 68 | 0.375 | 0.042 | 0.076 [0.015, 0.313] | [0.000, 0.037] | 0.206 [0.056, 0.372] | [0.000, 0.083] | 0 |
| runtime | (c) lane + Jev CAD-free filter | 0.57 | 3 | 4 | 68 | 0.429 | 0.042 | 0.077 [0.015, 0.313] | [0.000, 0.037] | 0.206 [0.056, 0.372] | [0.000, 0.083] | 0 |
| runtime | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 3 | 6 | 68 | 0.333 | 0.042 | 0.075 [0.015, 0.294] | [0.000, 0.017] | 0.189 [0.056, 0.333] | [0.000, 0.033] | 0 |
| runtime | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 3 | 6 | 68 | 0.333 | 0.042 | 0.075 [0.015, 0.294] | [0.000, 0.017] | 0.189 [0.056, 0.333] | [0.000, 0.033] | 0 |

#### Non-Gold11 (46 scans)

| lane | arm | keep t | TP | FP | FN | micro P | micro R | micro F1 [95% CI] | ΔF1 vs (a) [95% CI] | family-macro F1 [95% CI] | Δ fam-macro [95% CI] | control FP lines |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_f1 | (a) lane alone | - | 85 | 67 | 151 | 0.559 | 0.360 | 0.438 [0.358, 0.512] | - | 0.399 [0.286, 0.509] | - | 5 |
| best_f1 | (b) lane + Jev filter | 0.35 | 77 | 40 | 159 | 0.658 | 0.326 | 0.436 [0.354, 0.509] | [-0.024, 0.019] | 0.365 [0.259, 0.469] | [-0.099, 0.007] | 1 |
| best_f1 | (c) lane + Jev CAD-free filter | 0.61 | 80 | 44 | 156 | 0.645 | 0.339 | 0.444 [0.356, 0.532] | [-0.022, 0.034] | 0.411 [0.289, 0.528] | [-0.005, 0.032] | 5 |
| best_f1 | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 57 | 22 | 179 | 0.722 | 0.241 | 0.362 [0.272, 0.444] | [-0.145, -0.021] | 0.291 [0.193, 0.396] | [-0.198, -0.032] | 0 |
| best_f1 | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 82 | 56 | 154 | 0.594 | 0.347 | 0.439 [0.356, 0.521] | [-0.024, 0.018] | 0.404 [0.285, 0.516] | [-0.006, 0.016] | 5 |
| runtime | (a) lane alone | - | 18 | 36 | 218 | 0.333 | 0.076 | 0.124 [0.046, 0.207] | - | 0.110 [0.038, 0.201] | - | 3 |
| runtime | (b) lane + Jev filter | 0.57 | 17 | 25 | 219 | 0.405 | 0.072 | 0.122 [0.048, 0.204] | [-0.013, 0.006] | 0.110 [0.038, 0.202] | [-0.003, 0.002] | 3 |
| runtime | (c) lane + Jev CAD-free filter | 0.57 | 17 | 27 | 219 | 0.386 | 0.072 | 0.121 [0.047, 0.203] | [-0.013, 0.005] | 0.110 [0.038, 0.202] | [-0.003, 0.002] | 3 |
| runtime | (b0) lane + Jev, t=0.5 (not fitted) | 0.5 | 18 | 28 | 218 | 0.391 | 0.076 | 0.128 [0.047, 0.212] | [0.000, 0.008] | 0.112 [0.038, 0.203] | [0.000, 0.003] | 3 |
| runtime | (c0) lane + Jev CAD-free, t=0.5 (not fitted) | 0.5 | 17 | 33 | 219 | 0.340 | 0.072 | 0.119 [0.046, 0.198] | [-0.015, 0.001] | 0.109 [0.037, 0.201] | [-0.003, 0.000] | 3 |

#### Jev as a candidate classifier (positive = line matched to gold by the frozen matcher)

| lane | variant | cohort | candidates (pos/neg) | AUC [95% CI] | keep t | kept | keep precision | recall of true lines | false lines removed | true lines removed | FP rejection rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| best_f1 | Jev | ALL | 217 (115/102) | 0.6937 [0.619, 0.7694] | 0.35 | 171 | 0.6199 | 0.9217 | 37 | 9 | 0.3627 |
| best_f1 | Jev | EVAL_FAMILIES | 139 (72/67) | 0.7093 [0.6111, 0.8087] | 0.35 | 112 | 0.5982 | 0.9306 | 22 | 5 | 0.3284 |
| best_f1 | Jev | CALIB_FAMILIES_IN_SAMPLE | 78 (43/35) | 0.6744 [0.5597, 0.7909] | 0.35 | 59 | 0.661 | 0.907 | 15 | 4 | 0.4286 |
| best_f1 | Jev CAD-free | ALL | 217 (115/102) | 0.7211 [0.649, 0.8135] | 0.61 | 176 | 0.6023 | 0.9217 | 32 | 9 | 0.3137 |
| best_f1 | Jev CAD-free | EVAL_FAMILIES | 139 (72/67) | 0.6753 [0.5833, 0.7872] | 0.61 | 113 | 0.5575 | 0.875 | 17 | 9 | 0.2537 |
| best_f1 | Jev CAD-free | CALIB_FAMILIES_IN_SAMPLE | 78 (43/35) | 0.806 [0.6924, 0.9729] | 0.61 | 63 | 0.6825 | 1.0 | 15 | 0 | 0.4286 |
| runtime | Jev | ALL | 67 (21/46) | 0.7086 [0.5549, 0.8637] | 0.57 | 53 | 0.3774 | 0.9524 | 13 | 1 | 0.2826 |
| runtime | Jev | EVAL_FAMILIES | 50 (14/36) | 0.7966 [0.6775, 0.9731] | 0.57 | 37 | 0.3514 | 0.9286 | 12 | 1 | 0.3333 |
| runtime | Jev | CALIB_FAMILIES_IN_SAMPLE | 17 (7/10) | 0.4357 [0.1759, 1.0] | 0.57 | 16 | 0.4375 | 1.0 | 1 | 0 | 0.1 |
| runtime | Jev CAD-free | ALL | 67 (21/46) | 0.7433 [0.5805, 0.8807] | 0.57 | 54 | 0.3704 | 0.9524 | 12 | 1 | 0.2609 |
| runtime | Jev CAD-free | EVAL_FAMILIES | 50 (14/36) | 0.8036 [0.6217, 0.9444] | 0.57 | 39 | 0.3333 | 0.9286 | 10 | 1 | 0.2778 |
| runtime | Jev CAD-free | CALIB_FAMILIES_IN_SAMPLE | 17 (7/10) | 0.6357 [0.3379, 1.0] | 0.57 | 15 | 0.4667 | 1.0 | 2 | 0 | 0.2 |

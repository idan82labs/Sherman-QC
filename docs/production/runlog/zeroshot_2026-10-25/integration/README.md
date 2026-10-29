# Integrator: zs_lane_v1, DEV out-of-fold, freeze, TEST-B, CAD-path prototype (2026-10-25)

> **Ledger.** Every DEV number below is consumed development and out-of-fold. TEST-B was run once and is now CONSUMED; see `../testb/RESULTS.md`.

## Code

All code is in `research/zeroshot/zs_lane_v1/`. The tests are in `tests/test_prod_zeroshot_lane_v1_contract.py`: 7 synthetic checks, all passing.

| File | Role |
|---|---|
| `lane.py` | The CLI: scan → eval_lines.json. Variants: `zs` (primary), `z3`, `z2`. Runs the gold guard and the custody path check. With `--final-testb` it refuses on any pinned code/model hash mismatch or on an unpinned loaded repo module. A stage failure that leaves no output writes an empty FAILED payload. |
| `core.py` | The stages. Z3 `slicer.detect`; Z2 `run_scan` (it08 config, internal D1 snap); Z3 `pooled.pool_scan`, where Z2 lines become agreement flags or extra candidates; then the D1 gated outer-apex snap on the Z3 mesh. |
| `scoring.py` | Frozen matcher, aggregates and family bootstrap. A missing or failed output counts as all FN, and an exclusion aborts scoring. |
| `fingerprint.py` | Pins 60 files. |
| `train.py` | Out-of-fold evaluation and final fit. |
| `dev_dump.py` | Runs the DEV dump. |
| `freeze_config.py` | Writes `config_v1.json`. |
| `prereg.py` | Writes the pre-registration. |
| `score_testb.py` | Scores the TEST-B run. |
| `cad_prior.py` | The CAD-path prototype. |

Other inputs:
- Driver: `run_cohort.sh`, one process per (scan, lane), each through `tools/guarded_slot.sh`.
- Models: `models/zs_lane_v1_{zs,z3}_gbm.pkl`, fit on all 125 DEV scans. Their shas are in `config_v1.json` and `../testb/PREREG.json`.

**Audit fixes applied.**
1. **Threshold.** Fixed at 0.5 and documented in `config_v1.json`; it replaces Z3's undocumented 0.55. On DEV, fixed and fold-tuned thresholds are equal within noise.
2. **Fingerprint.** It covers every executed file, including Z3 `lane`, `zscore` and `export`-path dependencies, Z2, D1, the E3 metrology modules Z2 imports, the guard and the eval writer, plus the model shas. The scorer fingerprint is separate. Both are pinned in ledger seq 12.
3. **Failed scans.** They emit an empty FAILED payload and are scored as all-FN.
4. **Report format.** A pre-registered report format and expected range are in `PREREG.json`.

## DEV out-of-fold (all 125 DEV scans)

The cohort is 125 scans: 121 SPLIT_FROZEN plus the 4 newly frozen `ai_bend_lines_zs_v1` scans. That gives 120 line scans, 94 families and 619 gold lines, plus 5 zero-bend controls. The FRESH-DEV gold was read under ledger seq 8. Folds come from `dev_folds.json`.

| Arm (placement = D1 snap, threshold = fixed 0.5) | P | R | micro F1 [95 % CI] | family-macro [CI] | TP / FP / FN | ctrl FP | runtime median / P95 |
|---|---|---|---|---|---|---|---|
| A0 baseline (116 scans) | 0.60 | 0.22 | 0.322 | 0.267 | | 0 | 229 / 432 s |
| Z2 alone (it08) | 0.934 | 0.897 | 0.915 [0.877, 0.947] | 0.867 [0.808, 0.919] | 555 / 39 / 64 | 0 | 3.7 / 7.3 s |
| Z3 alone (GBM) | 0.923 | 0.906 | 0.914 [0.885, 0.940] | 0.905 [0.860, 0.942] | 561 / 47 / 58 | 5 | 1.8 / 9.9 s |
| **zs_lane_v1 (Z3 + Z2 pooled GBM)** | 0.942 | 0.947 | **0.944** [0.919, 0.966] | **0.944** [0.914, 0.969] | 586 / 36 / 33 | 5 | 5.7 / 16.5 s |

Ablations, out-of-fold:

| | zs | z3 |
|---|---|---|
| Fold-tuned threshold | 0.939 | 0.913 |
| Raw placement, no snap | 0.943 | 0.911 |

- The fold-tuned thresholds for zs were 0.775 / 0.35 / 0.5 / 0.55 / 0.65.
- The CLI wall time, including interpreter start-up, was zs median 6.5 s / P95 17.7 s, and z3 2.6 / 10.6 s. Peak RSS was at most 3.2 GiB.

**Subgroups** (`oof_v1/DEV_OOF_subgroups.json`), zs / z3 / z2 micro:

| Subgroup | zs | z3 | z2 | Note |
|---|---|---|---|---|
| FRESH-DEV, 17 line scans | 0.950 | 0.857 | 0.945 | Outside Z2's rule design |
| 4 new zs_v1 scans | 0.841 | 0.724 | 0.817 | |
| The other 99 | 0.951 | 0.936 | 0.917 | |

**Error budget, zs** (diagnose):
- FN 33: MISSED 13, SPAN 13, OFFSET 5, TAKEN 2.
- FP 36: SPAN 22, SPURIOUS 8, OFFSET 5, TAKEN 1.
- Controls: 5 FP lines on 5 zero-bend scans, reported separately.
- Worst families: 23317000 (0.0), 47247004 (0.53), 38898000 (rolled, 0.67), sc_9b930fe58a1b, and 45141000 (0.70).

**Choice: the primary is zs_lane_v1.**
- Paired family bootstrap, zs − Z3: **+0.030 micro [+0.013, +0.050]** and +0.039 macro [+0.015, +0.070].
- 25 families are better and 6 worse.
- zs − Z2: +0.029 [−0.005, +0.067].
- That is a clear margin over Z3, so the integration is kept.
- A0 was dropped as an agreement feature for three reasons:
  - Z3 + A0 pooled was 0.926, below Z3 + Z2.
  - Z4's merged pool scored 0.895.
  - A0 costs 200–600 s per scan, and its TEST-B outputs are sealed.
- Z5 / CP-SAT was not used (Z4 found no gain).
- The DEV CLI consistency check gave the same line sets in 375 of 375 cases: the frozen CLI with the final models reproduced the dump-pipeline selections exactly, for 125 scans × 3 lanes, all pins_ok.

**Improvement pass on the largest DEV error class (SPAN).**
- Most SPAN errors are one fold emitted as collinear fragments (11979003, 50757000, 40500000, 47247004, 48381001).
- **A collinear fragment merge was tried** (≤ 2°, ≤ 1.5 mm, |Δincl| ≤ 5°, gap ≤ G):

  | G (mm) | micro F1 |
  |---|---|
  | 2 | 0.945 |
  | 5 | 0.944 |
  | 10 | 0.939 |
  | 20 | 0.939 |
  | 40 | 0.948 |
  | 80 | 0.901 |
  | ∞ | 0.871 |

  The gains are non-monotone, and each family gains or loses. Merging by gap alone also merges separate tabs. **Rejected as noise.**
- **Parallel-twin NMS** (the coordinator's Z4 point 2):

  | Radius | micro F1 |
  |---|---|
  | none | 0.944 |
  | 4 mm | 0.940 |
  | 6 mm | 0.938 |
  | 8 mm | 0.933 |

  It is monotone worse because it removes real parallel neighbours (joggles), so inner CV would choose no NMS. TAKEN FPs number only 1 here. **Not adopted.**
- No stacking is used: a single-level GBM per fold.
- So the frozen lane is the plain integration. The real SPAN fix needs per-station flank-continuity evidence (see the TEST-B post-hoc analysis).

## TEST-B (once, pre-registered; details in `../testb/RESULTS.md`)

Gold of record is the re-audit: 61 scans / 50 families / 317 lines.

| Lane | Re-audit gold: micro [CI] | macro [CI] | Old FRESH labels: micro | macro |
|---|---|---|---|---|
| **zs (primary)** | **0.855** [0.764, 0.926] | **0.855** [0.782, 0.918] | 0.893 | 0.875 |
| Z3 | 0.850 [0.746, 0.928] | 0.857 [0.787, 0.917] | 0.889 | 0.877 |
| Z2 | 0.804 [0.716, 0.875] | 0.785 [0.690, 0.872] | 0.835 | 0.803 |

There were 0 failed scans and no controls.

## CAD-assisted path

### Prototype on DEV (`cad_prior/CAD_PRIOR_PROTO.json`)

- **Cohort.** 51 E3-eligible DEV scans with a compiled CAD: 49 line scans with 302 gold lines, plus 2 controls. The guard (`cad_path=True`) refused the 2 OD-3 cases present.
- **CAD facts** come from `compiled_part.json` and are rotation-invariant: n_bends, the nominal included angles, t, and K distinct axis directions.
- **Out-of-fold.** zs fold models are used, so no CAD is used in training.

| Arm | micro F1 [CI] | family-macro | TP / FP / FN |
|---|---|---|---|
| E3 alone (stored e3-1.12.0 runs) | 0.939 [0.876, 0.979] | 0.839 | 269 / 2 / 33 |
| e3_else_zs (E3 when VALID, else zs + priors) | 0.955 [0.898, 0.986] | 0.845 | 278 / 2 / 24 |
| zs (no CAD) | 0.966 [0.942, 0.987] | 0.948 | 295 / 14 / 7 |
| zs + angle window ±6° | 0.950 | 0.940 | 286 / 14 / 16 |
| zs + angle window ±6° + count cap ≤ n_bends | 0.960 | 0.937 | 285 / 7 / 17 |
| zs + CAD t + family cap (K + 2) + window + cap (Z3 re-run) | 0.960 | 0.937 | 285 / 7 / 17 |
| zs + count cap only | **0.975** [0.957, 0.990] | 0.945 | 294 / 7 / 8 |
| zs + asymmetric window (−6°, +15°) + cap | 0.973 | 0.941 | 293 / 7 / 9 |

The last two rows were computed afterwards, offline, from the same out-of-fold probabilities. They are diagnostic.

**Search time** (Z3 load + detect, single thread):

| Stage | Default: median / P95 / max | With CAD t + K + 2 family cap: median / P95 / max |
|---|---|---|
| Z3 alone | 2.2 / 8.7 / 20.3 s | 2.0 / 5.7 / 6.8 s |
| Full lane | 6.3 / 14.5 / 26.8 s | 5.8 / 11.7 / 12.8 s |

The family cap is what cuts the worst cases. F1 is unchanged.

**Findings.**
1. **The count cap is the useful prior.** It halves the FPs (14 → 7) for 1 TP.
2. **A symmetric angle window hurts.**
   - Real formed angles deviate from nominal by +7 to +13°: under-bent or partially formed parts at 145° nominal measure 152–157° (10839000). One fold also measured 67° against a 90/106° nominal.
   - The window must be one-sided toward 180°, or skipped at partial stage.
3. **Zero-shot is no longer the fallback.**
   - On this consumed cohort, zs alone beats E3 alone, because E3 refuses or withholds on some scans (49207000, mirrored, rolled).
   - E3 keeps far better precision (2 FP).
   - This argues for **E3 primary where VALID, with zs + count cap filling E3-UNKNOWN or withheld bends and refused scans**, rather than switching the whole scan.

### Plan

1. **E3 stays primary when VALID** for measurement (per-bend metrology and pass/fail). For line detection, use E3 lines, plus zs lines that match no E3 line, on refused scans or bends E3 withholds. The arm to measure next is `e3_union_zs_gap`.
2. **CAD priors, in order of value:**
   - (a) the count cap n_bends, full-stage only;
   - (b) the family cap K + 2 and CAD t for Z3 (P95 search time −35 %, max −66 %);
   - (c) a one-sided angle window [nominal − 6°, nominal + 15°];
   - (d) later, spectral matching of the Z2 flange graph to the CAD flange graph, to choose between merging and splitting at windows and tabs. CAD panels tell you whether two collinear tabs are the same plate. This targets the TEST-B SPAN class.
3. **Evaluation.**
   - Pre-register on DEV E3-eligible.
   - The single honest check is the 5 OD-3 DEV-CHECK cases (never run) plus new deliveries with STEP. TEST-B has only 1 STEP scan and is consumed.
   - The target is the CAD-pruned lane at P95 ≤ 60 s. It is already about 12 s.

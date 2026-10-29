# zs_lane_v3 freeze (2026-10-25)

> **Ledger.** Every accuracy number on this page is consumed-DEV out-of-fold: 125 DEV scans (120 line scans, 94 families, 619 gold lines, plus 5 zero-bend controls), folds from `dev_folds.json`.
> - TEST-B, DEV-CHECK, FRESH and N1 were not run and not scored.
> - No forbidden data was read.
> - Gold and labels are unchanged, and E3 engine code is unchanged.
> - The freeze-ledger entry is NOTE **seq 20**, labelled "FREEZE: zs_lane_v3". It binds this file's sha256.

## What was frozen

**zs_lane_v3 = zs_lane_v2 + EXT + Z3R.**
- Code: `research/zeroshot/zs_lane_v3/`.
- Rules and disclosures: `research/zeroshot/zs_lane_v3/README.md`.
- Tests: `tests/test_prod_zs_lane_v3.py`.

**Base: zs_lane_v2.** The rule was: the better DEV OOF F1, with equal or fewer false lines on the zero-bend controls.

| Base | EXT + Z3R micro | Control FP lines |
|---|---|---|
| v1 | 0.9644 | 5 |
| v2 | 0.9660 | 1 |

v2 is better on both counts. Both rows were reproduced exactly by the packaged code (below).

### Pins

| Item | Value |
|---|---|
| **Code fingerprint** | **`e34e1ba7f396eaf7782aff0250ea33599823e63e466ad2ca7eee4eddca965b4a`** (66 files: the v3 sources, the reused frozen v2 modules, Z3 / Z2 / D1, the E3 metrology modules that Z2 imports, the guard and the eval writer) |
| Scorer fingerprint | `05cc9314f471d22571cf9b108c53d11f384db89318dd4baeb31699d3c8104471` (v2's scorer, unchanged) |
| `config_v3.json` | sha256 `9860f46b56d34d99302cfd1d1396626980e216f9b31aa2c6c05420a5c20f0417` |
| Pooled GBM `v3/models/zs_lane_v3_zs_gbm.pkl` | sha256 `ade6cde44ede85dbf8195d0f6c093a2eea50eb4a481a21ec99fa9f17cd49ffeb`. Fit on all DEV, v2 recipe. Its predictions equal the frozen v2 model on all 3618 DEV rows (max \|Δ\| 0). |
| Z3-alone GBM `v3/models/zs_lane_v3_z3_gbm.pkl` | sha256 `62e99d13ccf0945d4b6932b85f19879e757e09ca7f493041c3ce53503e74e28c`. v1 z3 recipe. Its predictions equal the frozen v1 z3 model on all 3582 rows (max \|Δ\| 0). |
| v1 fingerprint | `af87df2f…1441c`, re-verified, unchanged |
| v2 fingerprint | `39c0efb9…09209`, re-verified, unchanged |

The v1 and v2 directories are byte-identical to their state before this work.

### Configuration

- **Threshold:** fixed 0.5 on the pooled GBM.
- **Features:** `base` (21 Z3 + 5 pool flags).
- **Z3:** `fallback_frac_only`.
- **Z2:** fallback on.
- **v2 post-processing (copied from `config_v2.json`):**
  - `join=True`, `join_mode=both`, `join_low=0.0`;
  - `dedupe_join=True`;
  - `sweep_min=0.25`;
  - `split=False`.
- **v3 fixes (`FixParams`):**
  - EXT: `ext_ratio=1.3`, `ext_min_in=0.6`, `ext_dedupe=True`;
  - Z3R: `z3r_tau=0.9`, `z3r_sweep_min=0.25`.

**Disclosure.**
- The 4 fix values were set on all-DEV out-of-fold prototypes, so the DEV numbers below are selection-aware for them.
- The 0.75 containment used in the N1 diagnosis was N1-aware. v3 uses **0.6**, the middle of the DEV plateau (0.5–0.75), on DEV alone. DEV is identical at 0.5, 0.6 and 0.75: 0.9660, 597 / 20 / 22. At 0.9 it is 0.9644.
- Z3R's τ 0.9 admits N1's B14 by 0.004.
- So N1 is no evidence for either fix.

## DEV out-of-fold of the packaged code

The procedure is the same as v2's, with the v2 steps reused from `zs_lane_v2/train.py`:
- per outer fold, the pooled GBM and the v2 post setting are chosen by inner grouped CV over v2's `final` grid;
- the Z3-alone GBM is fit on the same 4 training folds;
- then EXT and Z3R are applied.

Paired family bootstrap: 10 000 resamples, seed 20261025.

**Run path.**
- The DEV dump was produced by the v3 code through the gold guard and the geometry slot: 3 workers, peak RSS 3.6 GiB.
- It is **bit-identical** to the v2 dump that the prototypes used: same candidates, features, Z2 lines and evidence, on 125 of 125 scans.

**Main results.** CIs are 95 % family bootstraps.

| Arm | P | R | micro F1 [CI] | family-macro [CI] | TP / FP / FN | Control FP lines |
|---|---|---|---|---|---|---|
| v1 (frozen OOF) | 0.942 | 0.947 | 0.9444 [0.919, 0.967] | 0.9439 [0.914, 0.970] | 586 / 36 / 33 | 5 |
| v2 (reproduced; 125 / 125 scans identical to `v2/oof`) | 0.961 | 0.955 | 0.9579 [0.937, 0.975] | 0.9513 [0.922, 0.974] | 591 / 24 / 28 | 1 |
| **v3** | 0.968 | 0.965 | **0.9660 [0.951, 0.979]** | **0.9534 [0.923, 0.977]** | **597 / 20 / 22** | **1** |

**Paired differences.**

| Comparison | Δ micro [CI] | Δ macro [CI] | Families better / worse |
|---|---|---|---|
| v3 − v1 | **+0.0216 [+0.0034, +0.0458]** | +0.0096 [−0.0050, +0.0253] | 6 / 1 |
| v3 − v2 | **+0.0082 [−0.0020, +0.0248]** | +0.0021 [−0.0109, +0.0156] | 3 / 1 |

**Ablations and the containment sweep.** These use 2000 resamples.

| Arm | micro | TP / FP / FN | vs v2 Δ micro | Families better / worse |
|---|---|---|---|---|
| EXT only | 0.9643 | 595 / 20 / 24 | +0.0065 | 2 / 1 |
| Z3R only | 0.9595 | 593 / 24 / 26 | +0.0017 | 1 / 0 |
| containment 0.5 / 0.6 / 0.75 | 0.9660 | 597 / 20 / 22 | | |
| containment 0.9 | 0.9644 | 596 / 21 / 23 | | |
| Frozen v2 post setting in every fold, plus fixes (diagnostic, selection-biased) | 0.9668 | 597 / 19 / 22 | | |

**Reproduction of the prototype (`diag_47085000/evidence/proto_minin.json`).**

| Base | Prototype | Packaged code |
|---|---|---|
| v2 + EXT + Z3R | 0.9660, 597 / 20 / 22, control FP 1 | 0.9660, 597 / 20 / 22, control FP 1 |
| v1 + EXT + Z3R | 0.9644, 596 / 21 / 23, control FP 5 | 0.9644, 596 / 21 / 23, control FP 5 |

Both are exact. The CIs differ only in the bootstrap:

| Paired CI | Prototype | Packaged code |
|---|---|---|
| v2 base | [−0.0019, +0.0251] | [−0.0020, +0.0248] |
| v1 base | [+0.0024, +0.0439] | [+0.0021, +0.0441] |

The prototype used 4000 resamples over the union of families. The packaged code uses 10 000 resamples over the families scored in both, with v2's `paired_bootstrap`.

**Fix events on DEV.**
- EXT fired on 8 lines in 5 scans.
- Z3R fired on 2 lines, both on 40500000.

**Error budget, v2 → v3.**
- FN SPAN: 8 → 4.
- FN MISSED: 13 → 11.
- FP SPAN: 9 → 5.

## False lines on the controls

| Scan | Family | Kind | v1 | v2 | v3 |
|---|---|---|---|---|---|
| sc_3bde9996f14d | 28870000 | zero-bend | 1 | 1 | 1 |
| sc_49e5e7a1c973 | 48565022 | zero-bend | 0 | 0 | 0 |
| sc_4d62195d4e2d | 46465000 | zero-bend, rolled | 0 | 0 | 0 |
| sc_b4719db3dd69 | 47479000 | zero-bend, rolled | 4 | 0 | 0 |
| sc_f99d883f2754 | 11979003 | zero-bend | 0 | 0 | 0 |
| sc_ead8e407c174 | 38898000 | rolled, line scan (FP) | 0 | 0 | 0 |
| **Total** | | | **5** | **1** | **1** |

**No regression against v2.** EXT adds no line by construction, and Z3R's sweep gate keeps 47479000 at 0.

## Per-family wins and losses

**v3 against v2.**

| Family | v2 | v3 | Change | Cause |
|---|---|---|---|---|
| 47247004 | 0.556 | 1.0 | better | EXT |
| 40500000 | 0.933 | 1.0 | better | Z3R |
| 45141000 | 0.696 | 0.783 | better | EXT |
| 48340008 | 0.8 | 0.4 | **worse** | EXT; see below |

**v3 against v1.** Six families are better and one is worse:

| Family | v1 | v3 |
|---|---|---|
| 47247004 | 0.526 | 1.0 |
| 40500000 | 0.75 | 1.0 |
| 50757000 | 0.872 | 1.0 |
| 11979003 | 0.714 | 0.923 |
| 48381001 | 0.769 | 0.923 |
| 45141000 | 0.696 | 0.783 |
| 48340008 | 0.8 | **0.4** (worse) |

**The 48340008 loss.** The fold exists only across an 8 mm neck. Z2's line runs a further 15 mm along a slot edge, and EXT adopts that extent. A flank-presence guard did not separate this case (`diag_47085000/evidence/ext_flank_guard.json`), so the loss is accepted and disclosed.

## Runtime

The DEV CLI runs below were each one process per scan, with start-up included, through guard and slot.

| Lane | Median | P95 | Max | Peak RSS |
|---|---|---|---|---|
| v3 | 6.59 s | 21.7 s | 31.7 s | 3.27 GiB |
| v2 | 6.50 s | 21.3 s | 31.9 s | 3.27 GiB |
| v1 | 6.52 s | 17.7 s | 28.8 s | 3.16 GiB |

The selection, post-processing and fix stage takes a median of 5 ms (v2: 3 ms) and at most 28 ms. **v3 is the same as v2 plus about 2 ms.**

## DEV CLI consistency (`v3/dev_cli/CONSISTENCY.json`)

**Run.**
- The frozen CLI was used, with `run_cohort.sh`, `tools/guarded_slot.sh`, `--strict-pins` and the final models.
- 125 of 125 scans ran with pins OK.

**Result.**
- **125 of 125 line sets are identical** to the dump pipeline with the same final models.
- The in-sample score with the final models is recorded there as a check only. It is not a claim.

## Files

| Path | Contents |
|---|---|
| `v3/oof/OOF_SUMMARY.json` | All arms, bootstraps, controls, fix events, chosen v2 post settings |
| `v3/oof/OOF_{v3,v2_repro,v3_frozen_post,v1base_fixes}_{per_case,lines}.json` | Per-case records and lines |
| `v3/oof/OOF_v3_prob.json` | OOF probabilities |
| `v3/models/` | The 2 final models and `FINAL_MODELS.json` (equivalence checks) |
| `v3/dev_cli/` | CLI outputs, `CONSISTENCY.json` and the worker logs |
| `v3/PREREG_FRESH_TEMPLATE.json` | The template for the one-time fresh-scan test |
| `v3/REPIN_PLAN.md` | The production re-pin plan, with the G1 and rehearsal results |

**Next.** The one-time test on fresh customer scans (template above), with v3 as primary and v1 as comparison, and gold frozen first. The production default stays zs_lane_v1 until it passes.


## Superseded by revision 2 (2026-10-25)

The lane was re-frozen as **zs_lane_v3 revision 2** before any fresh-scan run. The change is runtime and memory only: an exact bounded Z3 merge, Z2 before Z3, and a 3 M-face budget. The new code fingerprint is `ab31d061…` and the config sha is `1c3f8cb7…`; the models and parameters are unchanged. See `rev2/FREEZE_REV2.md`. The revision-1 config and pin are archived in `rev2/rev1_archive/`.

# Final held-out evaluation: engine e3-1.9.0 (A1 iteration 7), 2026-10-19

The owner authorized this run on 2026-10-19. It is one-time: each set was evaluated once, with no engine edits and no tuning.

- **Ledger** (`freeze/LEDGER.json`):
  - seq 3 `frz_30c7c6aa8800`: RUN DEVCHECK_FINAL.
  - seq 4 `frz_3f202bedd738`: RUN FRESH_FINAL.
  - seq 5 `frz_b3658f8120f3`: NOTE binding the `results.json` sha256.
- **FRESH:** the parts are now CONSUMED.
- **Engine identity:** the fingerprints before and after the run equal `a1/iter_7/validation/code_fingerprints_final.json`.
- **Scoring:** the matcher is frozen (5° / 4 mm / 0.6, raw frame, OUTER_APEX). CIs are 95% family bootstraps (n=10000, seed 20261018).

## Headline: failures, refusals and not-run cases count as misses

| Set | Arm | Scans / fam | Line-F1 micro [CI] | Family-macro [CI] | TP/FP/FN |
|---|---|---|---|---|---|
| DEV-CHECK | engine | 10 / 7 | 0.297 [0.02, 0.67] | 0.253 [0.03, 0.54] | 11/4/48 |
| DEV-CHECK | best_f1 | 10 / 7 | 0.441 [0.36, 0.52] | 0.440 [0.34, 0.55] | 26/33/33 |
| FRESH | engine | 13 / 12 | 0.685 [0.16, 0.90] | 0.317 [0.08, 0.58] | 37/0/34 |
| FRESH | best_f1 | 13 / 12 | 0.347 [0.23, 0.48] | 0.378 [0.22, 0.56] | 25/48/46 |
| POOL | engine | 23 / 19 | 0.527 [0.19, 0.77] | 0.293 [0.11, 0.49] | 48/4/82 |
| POOL | best_f1 | 23 / 19 | 0.389 [0.31, 0.47] | 0.401 [0.29, 0.52] | 51/81/79 |

**Per-bend correctness given CAD** (definition: iter_7 `per_bend_all.py`):

| Set | Riser-aware | Strict | False PASS |
|---|---|---|---|
| DEV-CHECK | 14/36 = 0.389 | 0.306 | 2 |
| FRESH | 29/70 = 0.414 | 0.414 | 1 |
| POOL | 43/106 = 0.406 | 0.377 | 3 |

The false PASSes are `pc_6e5d5a2f60f5` B2 and B3, and `fr_7bde16010a96` B3.

**Compile rate:**
- DEV-CHECK: 7/7, byte-identical to the freeze.
- FRESH: 8/13 = 0.615, below the 0.80 floor. The failures:
  - 23317000: FLANGE_GRAPH_NOT_A_TREE
  - 38898000: DATUM_PANEL_UNDETERMINED
  - 46465000: DATUM_PANEL_UNDETERMINED
  - 49207000: CAD_READ_FAILED
  - 49783000: NO_CANDIDATE_BENDS

## Engine-ran view: scans where E3 produced a valid inspection

| Set | Engine micro / macro | best_f1 same scans | Engine TP/FP/FN | Per-bend riser-aware |
|---|---|---|---|---|
| DEV-CHECK (3 scans) | 0.667 / 0.590 | 0.474 / 0.439 | 11/4/7 | 0.560 |
| FRESH (6 scans) | 0.937 / 0.760 | 0.369 / 0.382 | 37/0/5 | 0.763 |
| POOL (9 scans) | 0.857 / 0.696 | 0.408 / 0.403 | 48/4/12 | 0.683 |

## Coverage losses: where the headline is lost

- **FRESH compile failures:** 5/13 (listed above).
- **FRESH refusals:** 3/9 runnable scans refused as WRONG_PART_SUSPECTED (DATUM_HOLES_FILLED): 46127000, 46128000, 48101000.
- **DEV-CHECK refusals:** 2/6 runnable scans refused as MIRRORED_PART_SUSPECTED: pc_1268c72dcc30, pc_2d3e4a623a31.
- **DEV-CHECK not run:** 5/11 cases. The amendment-7 gate blocked them because no frozen DEV-CHECK visual truth table exists: pc_eaa89c6d1f1f, pc_152f1b1aacb2, pc_c623d2e45bff, pc_9ef2cf4e52cb, pc_7a1773eca9fa.
  - The gate was not bypassed.
  - No engine output exists on these cases, so they can still be evaluated once, after a two-reviewer truth table is frozen.

## Reading

When the engine accepts a part, its lines are precise and recall beats the old lane. On FRESH it found 0 false lines, with micro 0.94 against the lane's 0.37. The headline is lost to coverage: CAD compile failures, wrong-part and mirror refusals, and the gate. On family-macro the engine is below the lane on every set. None of the owner aims is met: line-F1 >= 0.80, per-bend >= 0.90 and 0 false PASS.

## Files

- `results.json`
- `code_fingerprints_{pre,post}.json`
- logs
- `tools/`: `fresh_step.py` patches no custody guard.
- Work products: `/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/final_eval_w/`

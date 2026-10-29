# zs_lane_v3

zs_lane_v3 is zs_lane_v2 with two generic line fixes added after v2's span post-processing:

- **EXT** adopts the longer Z2 extent at pooling.
- **Z3R** rescues strong Z3-only lines.

The freeze record, with the DEV results, is `docs/production/runlog/zeroshot_2026-10-25/v3/FREEZE.md`.

## Base: zs_lane_v2

The base was chosen with two rules:

1. It must have the better DEV OOF F1.
2. It must have equal or fewer false lines on the zero-bend controls.

v2 satisfies both rules:

| Base | DEV OOF micro with EXT + Z3R | Zero-bend control FP lines (5 scans) |
|---|---|---|
| v1 | 0.9644 | 5 |
| **v2** | **0.9660** | **1** |

The packaged code reproduces both rows exactly; see the v3 FREEZE.md.

## What runs

1. **zs_lane_v2, unchanged.** The frozen v2 modules are imported read-only: `core`, `z3s`, `evidence` and `postproc`. That covers:
   - Z3 slicing (`fallback_frac_only`);
   - Z2 (with its fallback);
   - pooling;
   - the D1 snap of every pooled candidate;
   - the v2 evidence;
   - the pooled GBM at the fixed threshold 0.5;
   - v2's frozen post-processing: join with `join_low` 0, the post-join dedupe, and the sweep gate 0.25.

   `core.process` mirrors `zs_lane_v2.core.process`. Its only additions are an optional external thickness and a Z3 family cap, which the production CAD-prior path uses. The v3 DEV dump is bit-identical to v2's dump on all 125 DEV scans: same candidates, features, Z2 lines and evidence.
2. **Z3-alone GBM probability.** For each candidate with Z3 features, a Z3-alone GBM gives a probability from the 21 Z3 features. This is the zs_lane_v1 `z3` recipe; the predictions equal v1's z3 models exactly. Only Z3R uses it.
3. **`fixes.apply`: EXT, then Z3R** (rules in `fixes.py`):
   - **EXT.** Take a selected line whose candidate absorbed a Z2 line at pooling (pool rule 4° / 2.5 mm / 50 %, best match). If that Z2 line is **≥ 1.3×** the line's extent and contains **≥ 0.6** of it, along the line's own axis, the line takes the union of the two extents. The line's axis and placement are kept. Lines it now duplicates, and that are not longer, are dropped. EXT never adds a line.
   - **Z3R.** Keep a candidate when all of the following hold:
     - only Z3 found it (no Z2 agreement);
     - it is not part of any output line;
     - its Z3-alone probability is **≥ 0.9**;
     - its v2 station sweep fraction is **≥ 0.25**.

     It is added at its snapped placement unless it duplicates an output line.

There is no part-specific logic: no CAD, labels, file names or family identifiers are inputs. There is no TEST-B mode.

## Parameter disclosure (read before quoting any number)

**Where the four fix values come from.** All four were set on **all-DEV out-of-fold prototypes**, not by nested CV: EXT ratio 1.3, containment 0.6, Z3R τ 0.9 and sweep 0.25. They come from `newscans_N1/diag_47085000`, `evidence/proto*.json`. The DEV numbers in FREEZE.md are therefore selection-aware for these 4 values. The v2 part of the procedure is still chosen by nested CV, as in v2.

**The containment threshold was chosen with N1 in view.**
- In the diagnosis it was first set a priori to 0.9.
- It was revisited after the post-hoc check on the consumed N1 scan 47085000 showed that B1's line is only 88 % contained. The diagnosis then used 0.75. **That 0.75 was N1-aware.**
- DEV is flat from 0.5 to 0.75, and one line worse at 0.9. With the packaged code, DEV micro is identical at 0.50, 0.60 and 0.75 (0.9660, TP/FP/FN 597/20/22) and 0.9644 at 0.90.
- v3 therefore uses **0.6**, the middle of the DEV plateau, which DEV alone supports. 0.75 was not used because of N1.
- N1 cannot serve as evidence for EXT.

**Z3R's τ 0.9 is borderline for N1.** It admits N1's B14 (p_z3 0.904) by a margin of 0.004.
- On DEV, τ 0.9 with the sweep gate is the best of {0.8, 0.9, 0.95}: +1 family, 0 worse.
- A nested choice of τ (0.75 in most folds) was worse on DEV.
- N1 cannot serve as evidence for Z3R either.
- Without the sweep gate, Z3R adds a line on the rolled control 47479000. With the gate it adds 0.

**N1 is consumed.** It was used only for the post-hoc diagnosis and is never a v3 score. TEST-B, DEV-CHECK and FRESH were not touched.

## Files

| File | Role |
|---|---|
| `fixes.py` | EXT and Z3R (pure functions) and `FixParams` (the frozen values) |
| `core.py` | Pipeline: v2 stages, Z3-alone probability, `fixes.apply`, payload |
| `lane.py` | CLI: guard, custody, `--strict-pins` (exit 4 on any code/model mismatch or unpinned module), dump mode |
| `fingerprint.py` | Lane fingerprint: 66 files, including the reused v2 modules. The scorer fingerprint is v2's scorer. |
| `dev_dump.py` | DEV dump (guard + slot) |
| `train.py` | DEV OOF evaluation (nested v2 procedure, Z3-alone fold models, the fixes, arms, bootstraps) and the final fit |
| `freeze_config.py` | Writes `config_v3.json` |
| `consistency.py` | Checks the DEV CLI against the dump pipeline, and runtime |
| `make_pin.py` | Writes the production pin `domains/production/zeroshot/assets/zs_lane_v3.pin.json` and vendors the 2 models |
| `run_cohort.sh` | One process per scan through `tools/guarded_slot.sh`, `--strict-pins` |

Tests: `tests/test_prod_zs_lane_v3.py`.

## Commands (repo root)

```
S=<scratch dir>; P=docs/production/runlog/zeroshot_2026-10-25
# DEV dump, 3 workers, each holding one geometry slot
/private/tmp/claude-501/sherman_slot.sh .venv/bin/python -B -m research.zeroshot.zs_lane_v3.dev_dump --out $S/dump --shard i/3
# DEV OOF + final models
.venv/bin/python -B -m research.zeroshot.zs_lane_v3.train --dump $S/dump --out $P/v3/oof --fit-final $P/v3/models
# freeze
.venv/bin/python -B -m research.zeroshot.zs_lane_v3.freeze_config --model $P/v3/models/zs_lane_v3_zs_gbm.pkl --z3-model $P/v3/models/zs_lane_v3_z3_gbm.pkl
# DEV CLI consistency (guard + slot, --strict-pins), then the comparison
research/zeroshot/zs_lane_v3/run_cohort.sh $P/integration/dev_scans.tsv $P/v3/dev_cli i 3
.venv/bin/python -B -m research.zeroshot.zs_lane_v3.consistency --dump $S/dump --cli $P/v3/dev_cli --out $P/v3/dev_cli/CONSISTENCY.json
```

## Revision 2 (2026-10-25)

Revision 2 changes runtime and memory only:
- `z3_merge.py`: the exact bounded segment merge, with the same output as the frozen `z3s.merge_segments`;
- Z2 runs before Z3;
- `budget.py`: scans with more than 3 M faces run on a normal-aware quadric vertex-clustering of themselves.

The code fingerprint is `ab31d061…`. DEV accuracy is unchanged. See `docs/production/runlog/zeroshot_2026-10-25/v3/rev2/FREEZE_REV2.md`.

## Revision 3 (2026-10-26)

Revision 3 changes only the budget step (`budget.py`, plus `cell_mm` passed through `core.py`):
- A scan with more than 3 M faces is resampled on a **fixed absolute grid**. The cell is 0.30 mm, laddered by ×1.12 only to stay within 3 M faces.
- The resampling places one QEF vertex per (cell, skin) from the scan's grid-line crossings and builds one quad per crossing.
- Revision 2's vertex clustering, whose cell grew with the input density, is gone.

Results:
- The output depends on the surface, not on how densely it was sampled. Densified DEV scans up to 46 M faces reproduce every native-density line.
- DEV out-of-fold with the v3 fold models is unchanged.

The code fingerprint is `dc23a905…`. See `docs/production/runlog/zeroshot_2026-10-25/v3/rev3/FREEZE_REV3.md` and, for the Rust port, `rev3/RUST_NOTE.md`. `rev3_eval.py` is the DEV OOF check.

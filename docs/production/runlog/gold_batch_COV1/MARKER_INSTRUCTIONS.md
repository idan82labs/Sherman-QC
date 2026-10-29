# Gold batch COV1: marker instructions (2026-10-28)

Blind two-marker bend-line gold for the permitted real scans that had no frozen gold when the zs coverage run
(`docs/production/runlog/zs_coverage_2026-10-28/`) was set up. 3 items: 2 to label and 1 hidden canary. You are not told which is which.

The procedure and the gold rule are the ones in
`docs/production/runlog/zeroshot_2026-10-25/gold/MARKER_HOWTO.md` (sections "Per item", "Gold rule (GOLD_SET_V2)").
Only the paths differ:

```
R=/Users/idantal/ShermanQC/FullProject
G=$R/docs/production/runlog/gold_batch_COV1
SLOT=$R/docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh
PY="env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 $R/.venv/bin/python -B -m research.bench_labeling"
ID=gb_xxxxxx; SCAN=$G/scans/$ID.ply
W=$G/$ID/marker_X; mkdir -p $W                        # X = A or B
cd $R && $SLOT $PY prep  --scan "$SCAN" --out $W/prep
cd $R && $SLOT $PY lines --prep $W/prep --pairs $W/pairs.json --out $W/lines --locus apex
```

## What you may read
- `batch.json`
- the files in `scans/`
- this file and the MARKER_HOWTO
- your own `$G/<gid>/marker_<X>/` folder

## What you may not read
- `PRIVATE_KEY.json`
- anything under `data/bend_corpus/annotations/`
- any engine, lane or coverage output, including `zs_coverage_2026-10-28/runs/` and `work/`
- the other marker's folder
- `final_eval_2026-10-28_heldout/`
- anything named `*_SEALED*`

Do not hash files or look up their names or sizes. Do not run `write-label`.

## Adjudicator (coordinator)
1. Use `PRIVATE_KEY.json` to recover the original path and the canary.
2. Canary QC: each marker's line-F1 against the canary's gold (path in the key) must be at least 0.95, or the batch is redone.
3. Write the labels to a new root (suggested: `data/bend_corpus/annotations/ai_bend_lines_cov1_v1/`), then freeze them the way N1 was frozen.

Both label items come from a source folder that also holds a sealed evaluator manifest. That manifest was not opened, and it must not be used as a marking.

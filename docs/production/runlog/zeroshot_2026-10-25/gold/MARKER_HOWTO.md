# Gold marker how-to (zero-shot programme, PLAN.md section 2)

Every command runs from the repo root, `/Users/idantal/ShermanQC/FullProject`. Heavy steps (`prep`, `lines`) always go
through the geometry slot (`/private/tmp/claude-501/sherman_slot.sh`, 3 slots).

```
R=/Users/idantal/ShermanQC/FullProject
G=$R/docs/production/runlog/zeroshot_2026-10-25/gold
SLOT=/private/tmp/claude-501/sherman_slot.sh
PY="env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 $R/.venv/bin/python -B -m research.bench_labeling"
```

## What you may read, and what you may not

You may read:
- `gold/batches.json`, but only your own batch;
- the scan files listed there, under `gold/<batch>/scans/gb_xxxxxx.{ply,stl}`;
- this file;
- your own output folder, `gold/<batch>/<gid>/marker_<A|B>/`.

You may never read any of these:
- `gold/canaries.json`, `gold/GOLD_LEDGER.jsonl`, `INVENTORY.json`, `SPLIT_FROZEN.json`, `BASELINE*`, `baseline/`, `d1/`;
- anything under `data/bend_corpus/annotations/`, which includes all prior labels;
- any engine, lane, A0, pool, E3, best_f1 or CAD output;
- the other marker's folder or notes.

More rules:
- Do not try to work out which scan an item is: no hashing, no looking up file names or sizes.
- Every item gets the same full marking.
- Do not run `write-label`. The adjudicator writes the gold.

## Per item: what one marker does (X = A or B)

```
B=T1; ID=gb_xxxxxx; SCAN=$G/$B/scans/$ID.ply          # an STL item is used as is: $ID.stl
W=$G/$B/$ID/marker_X; mkdir -p $W
```

**1. Prep.** This makes planar patches and numbered renders. It is only a measurement aid and proposes no bends.

```
cd $R && $SLOT $PY prep --scan "$SCAN" --out $W/prep [--thickness T_mm]
```

Open every image in `$W/prep/` (the patch views and the contact sheet). `prep/manifest.json` gives the patch list, t and Ri.

**2. Pairs.** Write `$W/pairs.json`:

```json
{"labeler": "claude-vision-ai-v1", "notes": "free text",
 "bends": [
   {"patches": [4, 9], "note": "why"},
   {"patches": [[3, 12], 7], "extent_t": [-25.9, 34.1], "note": "slot splits flange"},
   {"patches": [7, {"plane_from_points": {"seed_xyz": [x, y, z], "radius_mm": 10}}], "note": "flange not segmented"}
 ]}
```

For a scan with no bends, use `{"labeler": "claude-vision-ai-v1", "no_bends_present": true, "bends": []}`.

The overrides `margin_mm`, `extent_t`, `extent_xyz`, `plane_from_points` and `apex_mode` are documented in
`research/bench_labeling/README_OVERRIDES.md`.

**3. Lines.** Use the apex locus, which is the outer-skin apex.

```
cd $R && $SLOT $PY lines --prep $W/prep --pairs $W/pairs.json --out $W/lines --locus apex
```

Open every `lines/closeup_<bend>.png`. Each one shows the mesh close-up plus a cross-section across the middle of the line. Also open the verification views in `lines/`. For each bend, check:
- the line sits on the outer-skin apex;
- it is parallel to the fold;
- it covers at least 80 % of the visible fold;
- `status` is ok;
- `confidence_reasons`, `fold_support_frac` and `fold_support_gaps_t`.

Fix the pairs and re-run until every line is right.

**4. Missed-bend sweep.** Look at the raw renders and sections again **without the patch list**. Add any fold the segmentation missed with `plane_from_points`, then re-run `lines`.

**5. Notes.** Write `$W/notes.md` with:
- every bend and why it is one;
- what you rejected and why (hole walls, edges, embosses, rolled kinks, weld-plate edges);
- whether the scan is partial;
- any doubts.

The final `$W/lines/lines.json` is your marking.

## Gold rule (GOLD_SET_V2)

- Use the outer-skin apex.
- The line is parallel to the fold.
- It covers at least 80 % of the visible fold.
- Formed folds of 10° or more count.
- A joggle counts as 2 lines.
- A fold split by windows or reliefs between the same two plates is 1 line.
- These are not bends: hole walls, edges, embosses, rolled kinks, weld-plate edges.
- On a partial scan, mark formed bends only.
- A zero-bend scan is a valid result.

## Adjudicator only: writing and freezing the gold (PLAN 2.4-2.5)

These steps need the private key `gold/canaries.json` to know the scan id, the original path and the portion. Markers never do this.

1. Match A against B with the frozen matcher (`research/broad_baseline/score.py score_case`). Then adjudicate every disagreement.
2. For TEST-B scans only, bring in the existing FRESH label as a third marking, and only at adjudication.
3. Build the final `lines` directory from the adjudicated pairs. Write the label with the **original** scan path as the asset path; the copy in `gold/` has the same bytes, so the sha256 matches:

```
# DEV (D0) scan:
cd $R && $PY write-label --lines <final_lines_dir> --asset-path "<original_path>" \
    --labels-root $R/data/bend_corpus/annotations/ai_bend_lines_zs_v1
# TEST-B scan:
cd $R && $PY write-label --lines <final_lines_dir> --asset-path "<original_path>" \
    --labels-root $R/data/bend_corpus/annotations/ai_bend_lines_zs_v1_SEALED
```

4. Add both markings, the adjudication record and the render hashes to the provenance sidecar (`<labels_root>/provenance/<asset_id>.json`).
5. Freeze the scan in the gold ledger. This step is append-only and the only way to make a scan runnable:

```
cd $R && .venv/bin/python -B -m research.zeroshot.common.gold_guard append --scan <sc_id> --status FROZEN \
    --label-path <label_path> --label-sha256 <sha256 of label> --by "<batch>/adjudicator"
```

The intermediate statuses `MARKING` and `ADJUDICATING` can also be appended.

6. Canary QC: score each marker against the canary's GOLD_SET_V2 gold (path in `canaries.json`). The line-F1 must be ≥ 0.95, or the batch is redone. Never write a label for a canary.

**STL item (48389000, D0).** `prep` and `lines` read the STL directly. `write-label` indexes only `.ply` files under a studio asset root (`data/`, `../source-assets`, `uploads/`, `live_scans/`, `output/`). So first export a same-frame PLY:

```
cd $R && .venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/tools/stl_same_frame_ply.py "<original .stl>" \
    $R/data/bend_corpus/derived_zs_v1/48389000_02_same_frame.ply
```

- The script checks that the vertex count is equal and the maximum vertex distance is 0. On 2026-10-25 it gave 263686 = 263686 vertices at a distance of 0.0 mm.
- Then pass that PLY as `--asset-path`. `write-label` accepts it through its vertex-set coordinate check (`coordinate_match: vertex_set`).

## Engine runs after freezing

Any engine or lane run on a scan must pass the guard:
- `docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh <scan|sc_id> -- <cmd>`, or
- `gold_guard.require_runnable()` in Python.

The guard refuses:
- a scan whose ledger status is not FROZEN;
- any TEST-B scan, unless `--final-testb` is given (the integrator's single end run);
- unknown scans;
- with `--cad-path`, the 5 OD-3 cases.

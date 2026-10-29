#!/bin/zsh
# usage: probe_dbg.sh SKU STAGE BENDS FRAC LABELS  -> regenerate the wrong-way probe scan (kept) and debug in-process
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter6
C=$W/phys/compiled/dev; R=docs/production/runlog/freeze/recipes
out=$W/revprobes/dbg/${1}_S${2}_${3//,/_}_$4
.venv/bin/python $W/tools/reversed_probe6.py $C/$1 $R/$1.recipe.json $out $2 --reverse $3 --frac $4 --seed 5 --keep-scan | head -3
.venv/bin/python $W/tools/dbg_scan6.py $C/$1 $R/$1.recipe.json $out/scan.ply $5 --png $out/dbg

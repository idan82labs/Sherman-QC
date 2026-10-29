#!/bin/zsh
# usage: probe_base.sh DBGDIR_NAME SKU STAGE BENDS FRAC -> run the iter-5 (snapshot) engine on the kept probe scan
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter6
C=$W/phys/compiled/dev; R=docs/production/runlog/freeze/recipes
src=$W/revprobes/dbg/$1; out=$W/revprobes/base/$1
mkdir -p $out; cp $src/scan.ply $out/scan.ply
.venv/bin/python $W/tools/reversed_probe6.py $C/$2 $R/$2.recipe.json $out $3 --reverse $4 --frac $5 --seed 5 --engine-root $W/base_root

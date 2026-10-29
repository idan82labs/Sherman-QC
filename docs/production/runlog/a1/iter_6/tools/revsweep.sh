#!/bin/zsh
# wrong-way probes (mirrored kinematics) on A1 recipes, CAD only. usage: revsweep.sh TAG ; runs inside ONE lock hold
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter6
TAG=$1
C=$W/phys/compiled/dev
R=docs/production/runlog/freeze/recipes
P=$W/tools/reversed_probe6.py
for spec in "DEV-4D512B5C56 12 B5 1.0" "DEV-1F88D309A7 3 B1 1.0" "DEV-EA3E4820DF 12 B5 1.0" "DEV-6E71609F74 16 B2,B14 1.0" \
            "DEV-AAB8C67C31 1 B1 1.0" "DEV-1F88D309A7 3 B1 0.15" "DEV-4D512B5C56 12 B1,B9 1.0" "DEV-0199D1289E 9 B2 1.0" \
            "DEV-2139CA09B5 1 B1 1.0"; do
  set -- ${=spec}
  sku=$1; st=$2; bl=$3; fr=$4
  out=$W/revprobes/$TAG/${sku}_S${st}_${bl//,/_}_$fr
  echo "=== $sku S$st reverse $bl frac $fr"
  .venv/bin/python $P $C/$sku $R/$sku.recipe.json $out $st --reverse $bl --frac $fr --seed 5 2>&1 | tail -40
done
echo REVSWEEP_DONE

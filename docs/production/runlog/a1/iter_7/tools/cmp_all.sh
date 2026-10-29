#!/bin/zsh
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter7
S6=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter6
TAG=$1; shift
SETS=${SETS:-}
for C in "$@"; do
  SKU=$(grep "^$C " $W/case_sku.txt | awk '{print $2}')
  zsh $W/tools/lk.sh cmp${TAG}_$C .venv/bin/python $W/tools/cmp_angle.py $S6/phys/compiled/dev/$SKU $S6/phys/physical/E3/$C/input/recipe.json $(ls $S6/phys/physical/E3/$C/input/scan.* | head -1) ${=SETS}
  echo "== $C $(grep "^$C " $W/case_sku.txt | awk '{print $3, $4}')"; grep -E "^B" $W/logs/cmp${TAG}_$C.log
done

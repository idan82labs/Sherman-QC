#!/bin/bash
# usage: dense_chain.sh JOBS OUTROOT CONFIG TIMEOUT  (one slot, serial)
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3pose
C=/private/tmp/claude-501/sherman_zeroshot_work/compile
cd /Users/idantal/ShermanQC/FullProject
while IFS='|' read -r mode tag scan name; do
  case $tag in A) SS=DEV-BDE878D2AF;; B) SS=DEV-6E71609F74;; C) SS=DEV-7A0E66894F;; D) SS=DEV-2720C50997;; esac
  [ -f $2/runs/$name/record.json ] && continue
  E3C_CONFIG=$3 /private/tmp/claude-501/sherman_slot.sh .venv/bin/python -B $W/tools/run_cand.py e3 "$scan" $C/$SS/compiled $C/$SS/dev.recipe.json $2/runs/$name 1 $4
  rm -f $2/runs/$name/engine/*.ply $2/runs/$name/engine/*.f32 $2/runs/$name/engine/*.u16; rm -rf $2/runs/$name/engine/display
done < $1
echo CHAIN_DONE

#!/bin/bash
# FRESH driver: per part build+compile (one lock hold each), then E3 per scan (one lock hold each).
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad; F=$S/final_eval_w/fresh
cd /Users/idantal/ShermanQC/FullProject
PY=.venv/bin/python
J() { $PY -c "import json,sys;d=json.load(open('$F/fresh_cases.json'));print(eval(sys.argv[1]))" "$1"; }
n=$(J "len(d['parts'])")
for i in $(seq 0 $((n-1))); do
  step=$(J "d['parts'][$i]['step_path']"); sha=$(J "d['parts'][$i]['step_sha256']")
  sku=$(PYTHONPATH=.:backend $PY -c "from domains.production.engine.eval.freeze.common import dev_sku_for;print(dev_sku_for('$sha'))")
  echo "part $i $sku" >> $F/logs/driver.log
  $S/locked.sh $F/logs/driver.log $PY $S/fresh_step.py build "$step" $F
  if grep -q RECIPE_BUILT $F/recipes/$sku/build.json 2>/dev/null; then
    $S/locked.sh $F/logs/driver.log $PY $S/fresh_step.py compile $sku $F
  fi
  ns=$(J "len(d['parts'][$i]['scans'])")
  for j in $(seq 0 $((ns-1))); do
    sid=$(J "d['parts'][$i]['scans'][$j]['scan_id']"); sp=$(J "d['parts'][$i]['scans'][$j]['path']")
    if [ -f $F/compiled/dev/$sku/compiled_part.json ] && grep -q '"status": "COMPILED"' $F/recipes/$sku/compile.json; then
      echo "e3 $sid" >> $F/logs/driver.log
      $S/locked.sh $F/logs/driver.log $PY $S/fresh_step.py e3 $sid "$sp" $sku $F
    else
      echo "e3 $sid SKIPPED_NOT_COMPILED" >> $F/logs/driver.log
    fi
  done
done
echo DONE >> $F/logs/driver.log

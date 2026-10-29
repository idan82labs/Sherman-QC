#!/bin/zsh
# candidate regression chain (one slot): frozen A1 recompile, A1 run-physical + score-physical, G1 regression tier
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad; W=$S/e3c/a1phys; TAG=$1; CFG=$2
PY=/Users/idantal/ShermanQC/FullProject/.venv/bin/python
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
cd $S/e3c/repo
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$S/e3c/repo:$S/e3c/repo/backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=$S/e3c/mpl
[ -d $W/compiled ] || $PY -B $S/e3c/tools/phys_cand.py compile-frozen --work $W > $S/e3c/logs/compile_frozen.log 2>&1; echo compile rc=$?
rm -rf $W/physical $W/physical_scorecard.json
EC=(); [ -n "$CFG" ] && EC=(--engine-config $CFG)
$PY -B $S/e3c/tools/phys_cand.py run-physical --work $W --cases /Users/idantal/ShermanQC/FullProject/docs/production/runlog/freeze/cases_A1.json $EC > $S/e3c/logs/run_physical_$TAG.log 2>&1; echo run rc=$?
$PY -B $S/e3c/tools/phys_cand.py score-physical --work $W --cases /Users/idantal/ShermanQC/FullProject/docs/production/runlog/freeze/cases_A1.json --ledger-anchor $ANCHOR --out $W/physical_scorecard.json > $S/e3c/logs/score_physical_$TAG.log 2>&1; echo score rc=$?
rm -rf $W/physical_$TAG; cp -R $W/physical $W/physical_$TAG; cp $W/physical_scorecard.json $W/physical_scorecard_$TAG.json
find $W/physical_$TAG -name "*.ply" -o -name "*.f32" -o -name "*.u16" | xargs rm -f
rm -rf $S/e3c/g1_$TAG
$PY -B $S/e3c/tools/g1_cand.py $S/zsprod/e2_regression_card $S/e3c/g1_$TAG $CFG > $S/e3c/logs/g1_$TAG.log 2>&1; echo g1 rc=$?
echo CHAIN_DONE_$TAG

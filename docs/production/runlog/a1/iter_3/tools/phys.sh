#!/bin/zsh
# A1 iter-3 physical measurement (13 runnable A1 cases): every heavy step serial under the shared lock, OMP_NUM_THREADS=1.
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
rm -rf $M/phys/physical $M/phys/analysis $M/phys/physical_scorecard.json
$M/tools/locked.sh run_physical .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze run-physical --work $M/phys --cases docs/production/runlog/freeze/cases_A1.json
$M/tools/locked.sh score_physical .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze score-physical --work $M/phys --cases docs/production/runlog/freeze/cases_A1.json --ledger-anchor $ANCHOR --out $M/phys/physical_scorecard.json
$M/tools/locked.sh analysis .venv/bin/python $M/tools/analyse.py $M/phys
$M/tools/locked.sh goldset .venv/bin/python $M/tools/goldset.py $M/phys
$M/tools/locked.sh per_bend_ext .venv/bin/python $M/tools/per_bend_ext.py $M/phys
echo PHYS_DONE

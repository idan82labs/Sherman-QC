#!/bin/zsh
# A1 iter-4 physical measurement (all A1 cases): every heavy step serial under the shared lock, OMP_NUM_THREADS=1.
# usage: phys.sh TAG  -> results in phys/physical (moved to phys/physical_TAG afterwards), scorecard phys/physical_scorecard_TAG.json
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1i4
TAG=$1
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
rm -rf $S/phys/physical $S/phys/analysis $S/phys/physical_scorecard.json
zsh $S/tools/locked.sh run_physical_$TAG .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze run-physical --work $S/phys --cases docs/production/runlog/freeze/cases_A1.json
zsh $S/tools/locked.sh score_physical_$TAG .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze score-physical --work $S/phys --cases docs/production/runlog/freeze/cases_A1.json --ledger-anchor $ANCHOR --out $S/phys/physical_scorecard.json
zsh $S/tools/locked.sh analysis_$TAG .venv/bin/python $S/tools/analyse.py $S/phys
zsh $S/tools/locked.sh goldset_$TAG .venv/bin/python $S/tools/goldset.py $S/phys
zsh $S/tools/locked.sh per_bend_ext_$TAG .venv/bin/python $S/tools/per_bend_ext.py $S/phys
rm -rf $S/phys/physical_$TAG $S/phys/analysis_$TAG
mv $S/phys/physical $S/phys/physical_$TAG; mv $S/phys/analysis $S/phys/analysis_$TAG; mv $S/phys/physical_scorecard.json $S/phys/physical_scorecard_$TAG.json
echo PHYS_DONE_$TAG

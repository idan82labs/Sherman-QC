#!/bin/zsh
# A1 iter-5 physical measurement (all A1 cases): every heavy step serial under the shared lock, OMP_NUM_THREADS=1.
# usage: phys.sh TAG  -> results in phys/physical + phys/analysis, then copied to phys/physical_TAG etc.
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter5
TAG=$1
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
rm -rf $W/phys/physical $W/phys/analysis $W/phys/physical_scorecard.json
zsh $W/tools/locked.sh run_physical_$TAG .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze run-physical --work $W/phys --cases docs/production/runlog/freeze/cases_A1.json
zsh $W/tools/locked.sh score_physical_$TAG .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze score-physical --work $W/phys --cases docs/production/runlog/freeze/cases_A1.json --ledger-anchor $ANCHOR --out $W/phys/physical_scorecard.json
zsh $W/tools/locked.sh analysis_$TAG .venv/bin/python $W/tools/analyse.py $W/phys
zsh $W/tools/locked.sh per_bend_ext_$TAG .venv/bin/python $W/tools/per_bend_ext.py $W/phys
zsh $W/tools/locked.sh goldset_$TAG .venv/bin/python $W/tools/goldset.py $W/phys
zsh $W/tools/locked.sh per_bend_all_$TAG .venv/bin/python $W/tools/per_bend_all.py $W/phys
rm -rf $W/phys/physical_$TAG $W/phys/analysis_$TAG
mkdir -p $W/phys/physical_$TAG
for d in $W/phys/physical/E3/*; do c=$(basename $d); mkdir -p $W/phys/physical_$TAG/E3/$c; cp $d/record.json $W/phys/physical_$TAG/E3/$c/ 2>/dev/null; mkdir -p $W/phys/physical_$TAG/E3/$c/engine; cp $d/engine/result.json $d/engine/engine.log $W/phys/physical_$TAG/E3/$c/engine/ 2>/dev/null; done
cp -R $W/phys/analysis $W/phys/analysis_$TAG; cp $W/phys/physical_scorecard.json $W/phys/physical_scorecard_$TAG.json
echo PHYS_DONE_$TAG

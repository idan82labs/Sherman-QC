#!/bin/zsh
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 MPLCONFIGDIR=$S/a1_iter0/mpl .venv/bin/python -m domains.production.engine.eval run --tier-dir $S/a1_iter0/e2_regression --out $S/a1_iter1/g1_regression --ledger docs/production/runlog/freeze/LEDGER.json --ledger-anchor $ANCHOR > $S/a1_iter1/logs/g1_run.log 2>&1
echo "G1 rc=$?"
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze run-physical --work $S/a1_iter1/phys --cases docs/production/runlog/freeze/cases_A1.json > $S/a1_iter1/logs/run_physical.log 2>&1
echo "run-physical rc=$?"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 MPLCONFIGDIR=$S/a1_iter0/mpl .venv/bin/python -m domains.production.engine.eval.freeze.prefreeze score-physical --work $S/a1_iter1/phys --cases docs/production/runlog/freeze/cases_A1.json --ledger-anchor $ANCHOR --out $S/a1_iter1/phys/physical_scorecard.json > $S/a1_iter1/logs/score_physical.log 2>&1
echo "score rc=$?"
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=$S/a1_iter0/mpl .venv/bin/python $S/a1_iter1/tools/analyse_iter1.py $S/a1_iter1/phys > $S/a1_iter1/logs/analysis.log 2>&1
echo "analysis rc=$?"
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python $S/a1_iter1/tools/goldset_iter1.py $S/a1_iter1/phys > $S/a1_iter1/logs/goldset.log 2>&1
echo "goldset rc=$?"
rmdir /private/tmp/claude-501/sherman_geom.lock
echo FINAL_DONE

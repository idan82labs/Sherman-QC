#!/bin/zsh
# A1 iter-2 physical measurement: every heavy step serial, each under the shared geometry lock, OMP_NUM_THREADS=1.
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter2
LOCK=/private/tmp/claude-501/sherman_geom.lock
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
PY=.venv/bin/python
ENVV=(env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$M/mpl)
mkdir -p $M/logs
step() { name=$1; shift
  until mkdir $LOCK 2>/dev/null; do sleep 3; done
  date -u +%FT%TZ > $M/logs/$name.start
  /usr/bin/time -l "$@" > $M/logs/$name.log 2>&1; rc=$?
  rmdir $LOCK
  echo "$name rc=$rc $(date -u +%FT%TZ)"; }
rm -rf $M/phys
step compile_frozen $ENVV $PY -m domains.production.engine.eval.freeze.prefreeze compile-frozen --work $M/phys
step run_physical $ENVV $PY -m domains.production.engine.eval.freeze.prefreeze run-physical --work $M/phys --cases docs/production/runlog/freeze/cases_A1.json
step score_physical $ENVV $PY -m domains.production.engine.eval.freeze.prefreeze score-physical --work $M/phys --cases docs/production/runlog/freeze/cases_A1.json --ledger-anchor $ANCHOR --out $M/phys/physical_scorecard.json
step analysis $ENVV $PY $M/tools/analyse.py $M/phys
step goldset $ENVV $PY $M/tools/goldset.py $M/phys
step per_bend_ext $ENVV $PY $M/tools/per_bend_ext.py $M/phys
echo PHYS_DONE

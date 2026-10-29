#!/bin/zsh
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter2
OUT=$1
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
rm -rf $OUT
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
date -u +%FT%TZ > $OUT.start
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$M/mpl .venv/bin/python -m domains.production.engine.eval run --tier-dir $S/a1_iter0/e2_regression --out $OUT --ledger docs/production/runlog/freeze/LEDGER.json --ledger-anchor $ANCHOR > $OUT.log 2>&1
rc=$?
rmdir /private/tmp/claude-501/sherman_geom.lock
echo "G1 rc=$rc $(date -u +%FT%TZ)"

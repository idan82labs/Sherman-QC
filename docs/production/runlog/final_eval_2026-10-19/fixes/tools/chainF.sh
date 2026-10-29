#!/bin/zsh
# Fixes 2026-10-19 regression chain on the fixed engine: A1 physical set (iter_7 procedure) + G1 regression tier.
SP=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
W=$SP/fixes/a1
TAG=$1
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python $W/tools/fp.py $W/logs/fp_chain_${TAG}_start.json > /dev/null
zsh $W/tools/phys7.sh $TAG
rm -rf $W/g1_$TAG
zsh $W/tools/lk.sh g1_$TAG .venv/bin/python -m domains.production.engine.eval run --tier-dir $SP/a1_iter0/e2_regression --out $W/g1_$TAG --ledger docs/production/runlog/freeze/LEDGER.json --ledger-anchor $ANCHOR
echo G1_DONE_$TAG
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python $W/tools/fp.py $W/logs/fp_chain_${TAG}_end.json > /dev/null
echo CHAIN_DONE_$TAG

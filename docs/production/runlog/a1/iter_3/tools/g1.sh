#!/bin/zsh
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
OUT=$1
cd /Users/idantal/ShermanQC/FullProject
ANCHOR=cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67
rm -rf $OUT
$M/tools/locked.sh g1_$(basename $OUT) .venv/bin/python -m domains.production.engine.eval run --tier-dir $S/a1_iter0/e2_regression --out $OUT --ledger docs/production/runlog/freeze/LEDGER.json --ledger-anchor $ANCHOR

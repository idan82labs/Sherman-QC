#!/bin/bash
# pipeline.sh: scans -> synthetic candidate -> G1 candidate (E4) -> real candidate -> synthetic live -> real live
W=/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work
PY=/Users/idantal/ShermanQC/FullProject/.venv/bin/python
cd $W
xargs -P 2 -L 1 ./tools/gen_one.sh < gen.txt > logs/gen.out 2>&1
$PY tools/pool.py runs_syn.tsv > logs/runs_syn.out 2>&1
(cd /Users/idantal/ShermanQC/FullProject && $W/tools/slot.sh env PYTHONPATH=.:backend .venv/bin/python -m domains.production.engine.eval run \
   --tier-dir $W/g1tier --out $W/g1_cand --rss-limit-mb 16384 --engine-cmd "$PY $W/tools/cand_inspect.py" > $W/logs/g1_cand.log 2>&1)
$PY tools/pool.py runs_real.tsv > logs/runs_real.out 2>&1
$PY tools/pool.py runs_syn_live.tsv --env E3C_REPO=/Users/idantal/ShermanQC/FullProject > logs/runs_syn_live.out 2>&1
$PY tools/pool.py runs_real_live.tsv --env E3C_REPO=/Users/idantal/ShermanQC/FullProject > logs/runs_real_live.out 2>&1
echo PIPELINE_DONE > logs/pipeline.done

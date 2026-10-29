#!/bin/zsh
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
LOCK=/private/tmp/claude-501/sherman_geom.lock
cd /Users/idantal/ShermanQC/FullProject
OUT=$M/logs/jprobes_final.txt
: > $OUT
for c in PROBE-joggle35 PROBE-joggle35err PROBE-bump35 PROBE-joggle35top PROBE-joggle35stage; do
  until mkdir $LOCK 2>/dev/null; do sleep 3; done
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 MPLCONFIGDIR=$M/mpl .venv/bin/python $M/tools/run_jprobe.py $M/jprobe/$c $M/final_tree >> $OUT 2>&1
  rmdir $LOCK; sleep 3
done
echo JPROBES_DONE >> $OUT

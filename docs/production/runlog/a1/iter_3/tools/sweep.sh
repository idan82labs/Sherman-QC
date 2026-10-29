#!/bin/zsh
# usage: sweep.sh TAG PLAN SEED [SKU...]  -- every compiled stage of each SKU, one probe at a time, each under the lock
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
TAG=$1; PLAN=$2; SEED=$3; shift 3
C=$M/phys/compiled/dev
R=docs/production/runlog/freeze/recipes
LOCK=/private/tmp/claude-501/sherman_geom.lock
cd /Users/idantal/ShermanQC/FullProject
ER=()
[ -n "$ENGINE_ROOT" ] && ER=(--engine-root $ENGINE_ROOT)
OUTF=$M/logs/sweep_$TAG.txt
touch $OUTF
K0=${K0:-0}
for sku in "$@"; do
  n=$(ls $C/$sku/stages/*.ply | wc -l | tr -d ' ')
  for ((k=K0; k<n; k++)); do
    until mkdir $LOCK 2>/dev/null; do sleep 3; done
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$M/mpl \
      .venv/bin/python $M/tools/stage_probe.py $C/$sku $R/$sku.recipe.json $M/probes/$TAG/$sku/S${k} $k --plan $PLAN --seed $SEED $ER >> $OUTF 2>&1
    rmdir $LOCK
    sleep 5  # fairness: let other lock waiters (3 s pollers) in between probes
  done
  K0=0
done
echo SWEEP_DONE >> $OUTF

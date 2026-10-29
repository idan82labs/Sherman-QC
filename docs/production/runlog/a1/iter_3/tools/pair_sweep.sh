#!/bin/zsh
# usage: pair_sweep.sh SKU:STAGE ...  -- per item: baseline probe (snapshot engine; makes the scan) unless done, then the
# new engine (new2_tree) on the same scan. Each E3 run under the shared lock, one at a time.
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
C=$M/phys/compiled/dev
R=docs/production/runlog/freeze/recipes
LOCK=/private/tmp/claude-501/sherman_geom.lock
cd /Users/idantal/ShermanQC/FullProject
ENVV=(env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$M/mpl)
for item in "$@"; do
  sku=${item%%:*}; k=${item##*:}
  b=$M/probes/base/$sku/S$k; n=$M/probes/new/$sku/S$k
  if [ ! -f $b/summary.json ]; then
    until mkdir $LOCK 2>/dev/null; do sleep 3; done
    $ENVV .venv/bin/python $M/tools/stage_probe.py $C/$sku $R/$sku.recipe.json $b $k --engine-root $M/base_tree >> $M/logs/sweep_base.txt 2>&1
    rmdir $LOCK; sleep 4
  fi
  if [ ! -f $n/summary.json ] && [ -f $b/scan.ply ]; then
    mkdir -p $n; cp $b/scan.ply $b/pose.npy $b/scan_stats.json $n/
    until mkdir $LOCK 2>/dev/null; do sleep 3; done
    $ENVV .venv/bin/python $M/tools/stage_probe.py $C/$sku $R/$sku.recipe.json $n $k --engine-root $M/new2_tree >> $M/logs/reprobe_new.txt 2>&1
    rmdir $LOCK; sleep 4
  fi
done
echo PAIR_DONE >> $M/logs/reprobe_new.txt

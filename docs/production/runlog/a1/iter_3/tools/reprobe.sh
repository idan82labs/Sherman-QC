#!/bin/zsh
# usage: reprobe.sh NEWTAG SKU:STAGE ...  -- re-run E3 (repo engine, or ENGINE_ROOT) on existing base-tag virtual scans
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
TAG=$1; shift
C=$M/phys/compiled/dev
R=docs/production/runlog/freeze/recipes
LOCK=/private/tmp/claude-501/sherman_geom.lock
cd /Users/idantal/ShermanQC/FullProject
ER=()
[ -n "$ENGINE_ROOT" ] && ER=(--engine-root $ENGINE_ROOT)
OUTF=$M/logs/reprobe_$TAG.txt
touch $OUTF
for item in "$@"; do
  sku=${item%%:*}; k=${item##*:}
  src=$M/probes/base/$sku/S$k; dst=$M/probes/$TAG/$sku/S$k
  mkdir -p $dst; cp $src/scan.ply $src/pose.npy $src/scan_stats.json $dst/
  until mkdir $LOCK 2>/dev/null; do sleep 3; done
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$M/mpl \
    .venv/bin/python $M/tools/stage_probe.py $C/$sku $R/$sku.recipe.json $dst $k $ER >> $OUTF 2>&1
  rmdir $LOCK
  sleep 4
done
echo REPROBE_DONE >> $OUTF

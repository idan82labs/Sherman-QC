#!/bin/zsh
# usage: sbsweep.sh TAG SKU:STAGE ...  (ENGINE_ROOT optional) -- springback probes, one at a time under the lock
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
TAG=$1; shift
C=$M/phys/compiled/dev
R=docs/production/runlog/freeze/recipes
LOCK=/private/tmp/claude-501/sherman_geom.lock
cd /Users/idantal/ShermanQC/FullProject
ER=()
[ -n "$ENGINE_ROOT" ] && ER=(--engine-root $ENGINE_ROOT)
OUTF=$M/logs/sbsweep_$TAG.txt
touch $OUTF
for item in "$@"; do
  sku=${item%%:*}; k=${item##*:}
  dst=$M/sbprobes/$TAG/$sku/S$k
  src=$M/sbprobes/scans/$sku/S$k
  mkdir -p $dst $src
  if [ -f $src/scan.ply ]; then cp $src/scan.ply $src/pose.npy $src/truth.json $dst/; fi
  until mkdir $LOCK 2>/dev/null; do sleep 3; done
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$M/mpl \
    .venv/bin/python $M/tools/springback_probe.py $C/$sku $R/$sku.recipe.json $dst $k $ER >> $OUTF 2>&1
  rmdir $LOCK
  if [ ! -f $src/scan.ply ] && [ -f $dst/scan.ply ]; then cp $dst/scan.ply $dst/pose.npy $dst/truth.json $src/; fi
  sleep 4
done
echo SBSWEEP_DONE >> $OUTF

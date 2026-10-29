#!/bin/zsh
# usage: sbsweep.sh TAG  -- every cached springback-23 CAD-stage probe (a1_iter4/sbprobes/scans_s23), one at a time
# under the shared lock; engine = the repository's current engine.
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
W=$S/a1_iter5
SRC=$S/a1_iter4/sbprobes/scans_s23
TAG=$1
C=$W/phys/compiled/dev
R=docs/production/runlog/freeze/recipes
LOCK=/private/tmp/claude-501/sherman_geom.lock
cd /Users/idantal/ShermanQC/FullProject
OUTF=$W/logs/sbsweep_$TAG.txt
: > $OUTF
for skud in $SRC/*; do
  sku=$(basename $skud)
  for sd in $skud/S*; do
    k=${$(basename $sd)#S}
    dst=$W/sbprobes/$TAG/$sku/S$k
    mkdir -p $dst
    cp $sd/scan.ply $sd/pose.npy $sd/truth.json $dst/
    until mkdir $LOCK 2>/dev/null; do sleep 3; done
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$W/mpl \
      .venv/bin/python $W/tools/springback_probe.py $C/$sku $R/$sku.recipe.json $dst $k --seed 23 >> $OUTF 2>&1
    rmdir $LOCK
    rm -f $dst/scan.ply
    sleep 2
  done
done
echo SBSWEEP_DONE >> $OUTF

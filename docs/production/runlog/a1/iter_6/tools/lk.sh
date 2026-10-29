#!/bin/zsh
# usage: lk.sh NAME CMD...  -> runs CMD in the repo under the shared geometry lock, OMP_NUM_THREADS=1, log to logs/NAME.log
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter6
LOCK=/private/tmp/claude-501/sherman_geom.lock
name=$1; shift
cd /Users/idantal/ShermanQC/FullProject
until mkdir $LOCK 2>/dev/null; do sleep 3; done
date -u +%FT%TZ > $W/logs/$name.start
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=$W/mpl "$@" > $W/logs/$name.log 2>&1
rc=$?
rmdir $LOCK
echo "$name rc=$rc $(date -u +%FT%TZ)"
exit $rc

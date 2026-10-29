#!/bin/bash
# usage: run.sh NAME [--lock] args...   (runs build_g4.py with single-thread env; --lock = under the shared geometry lock)
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/g4w
LOCK=/private/tmp/claude-501/sherman_geom.lock
name=$1; shift
cd /Users/idantal/ShermanQC/FullProject
mkdir -p $W/logs $W/mpl
if [ "$1" = "--lock" ]; then shift; uselock=1; fi
[ -n "$uselock" ] && { until mkdir $LOCK 2>/dev/null; do sleep 3; done; }
date -u +%FT%TZ > $W/logs/$name.start
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$W/mpl .venv/bin/python $W/tools/build_g4.py "$@" > $W/logs/$name.log 2> $W/logs/$name.err
rc=$?
[ -n "$uselock" ] && rmdir $LOCK
echo "$name rc=$rc"; tail -5 $W/logs/$name.log; grep -E "maximum resident|real" $W/logs/$name.err; [ $rc -ne 0 ] && tail -20 $W/logs/$name.err
exit $rc

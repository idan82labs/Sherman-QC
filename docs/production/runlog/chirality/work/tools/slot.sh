#!/bin/bash
# slot.sh CMD... : run CMD holding one of 2 geometry slots (mkdir locks in work/locks), OMP_NUM_THREADS=1, nice 15,
# process-tree RSS watched by tools/rsscap.py (killed above 16 GiB).
W=/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work
while true; do
  for s in a b; do
    if mkdir $W/locks/slot_$s 2>/dev/null; then
      trap "rmdir $W/locks/slot_$s" EXIT
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 \
        /Users/idantal/ShermanQC/FullProject/.venv/bin/python $W/tools/rsscap.py "$@"; exit $?
    fi
  done
  sleep 3
done

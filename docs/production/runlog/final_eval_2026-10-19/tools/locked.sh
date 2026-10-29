#!/bin/bash
# usage: locked.sh <logfile> <cmd...>   (runs in repo, under the shared lock)
cd /Users/idantal/ShermanQC/FullProject
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1
log=$1; shift
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
"$@" >> "$log" 2>&1; rc=$?
rmdir /private/tmp/claude-501/sherman_geom.lock
echo "rc=$rc" >> "$log"; exit $rc

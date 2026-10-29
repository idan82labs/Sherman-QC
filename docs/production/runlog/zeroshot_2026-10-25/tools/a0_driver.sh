#!/bin/bash
# usage: a0_driver.sh WORKER_INDEX N_WORKERS   -- runs every N-th job of baseline/a0_jobs.json, one slot per job
H=/Users/idantal/ShermanQC/FullProject/docs/production/runlog/zeroshot_2026-10-25
cd /Users/idantal/ShermanQC/FullProject
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
LOG=$H/baseline/a0_driver_$1.log
n=$(python3 -c "import json;print(len(json.load(open('$H/baseline/a0_jobs.json'))))")
for ((i=$1; i<n; i+=$2)); do
  read id scan out work < <(python3 -c "import json;j=json.load(open('$H/baseline/a0_jobs.json'))[$i];print(j['id'],j['scan'].replace(' ','\\\\ '),j['out'],j['work'])")
  scan=$(python3 -c "import json;print(json.load(open('$H/baseline/a0_jobs.json'))[$i]['scan'])")
  if [ -f "$out/run_record.json" ]; then echo "$(date -u +%FT%TZ) $id SKIP_DONE" >> $LOG; continue; fi
  mkdir -p "$out"
  t0=$(date +%s)
  /private/tmp/claude-501/sherman_slot.sh .venv/bin/python -B research/broad_baseline/run_lane.py --lane best_f1 \
     --scan "$scan" --out "$out" --work-dir "$work" --lock /private/tmp/claude-501/zs_a0_lane_$1.lock \
     --step-timeout 600 --lane-timeout 600 --rss-limit-gib 12 > "$out/driver_stdout.txt" 2>&1
  rc=$?
  st=$(python3 -c "import json;r=json.load(open('$out/run_record.json'));print(r.get('status'),r.get('line_count'),round((r.get('peak_rss_bytes') or 0)/2**30,2))" 2>/dev/null)
  echo "$(date -u +%FT%TZ) $id rc=$rc $st wall=$(( $(date +%s)-t0 ))s" >> $LOG
done
echo DONE >> $LOG

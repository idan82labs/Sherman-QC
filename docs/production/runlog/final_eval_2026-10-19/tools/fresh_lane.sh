#!/bin/bash
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad; F=$S/final_eval_w/fresh
cd /Users/idantal/ShermanQC/FullProject
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p $F/bb_logs
for id in $(.venv/bin/python -c "import json;print(' '.join(c['case_id'] for c in json.load(open('$F/fresh_case_list_bb.json'))['cases']))"); do
  until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
  /usr/bin/time -l .venv/bin/python -B research/broad_baseline/run_broad.py --case-list $F/fresh_case_list_bb.json \
     --runs-root $F/bb_runs --lanes best_f1 --only "$id" --work-root /private/tmp/claude-501/sherman_broad_baseline_work/fresh_final \
     --step-timeout 600 --lane-timeout 600 --rss-limit-gib 9 > $F/bb_logs/$id.stdout.txt 2> $F/bb_logs/$id.time.txt
  rc=$?; rmdir /private/tmp/claude-501/sherman_geom.lock
  echo "$(date -u +%FT%TZ) $id rc=$rc $(grep -- '->' $F/bb_logs/$id.stdout.txt | tr -s ' ')" >> $F/bb_logs/lane_driver.log
  sleep 2
done
echo DONE >> $F/bb_logs/lane_driver.log

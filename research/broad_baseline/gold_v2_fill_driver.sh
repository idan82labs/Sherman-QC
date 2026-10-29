#!/bin/zsh
# Fill best_f1 predictions for GOLD_SET_V2 scans that lack them, one scan at a time.
# Each scan runs ONLY under the shared geometry lock (mkdir lock), released after every scan.
# run_broad.py --only <id> is resume-safe (skips finished scan/CAD sha pairs).
# Usage: zsh research/broad_baseline/gold_v2_fill_driver.sh <case_list.json> <log_dir>
set -u
ROOT=/Users/idantal/ShermanQC/FullProject
CL=$1
LOGDIR=$2
LOCK=/private/tmp/claude-501/sherman_geom.lock
mkdir -p "$LOGDIR"
cd "$ROOT" || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# no-CAD scans first (blind track), then CAD scans
IDS=($(.venv/bin/python -c "import json,sys;c=json.load(open(sys.argv[1]))['cases'];print(' '.join([x['case_id'] for x in c if not x.get('cad_path')]+[x['case_id'] for x in c if x.get('cad_path')]))" "$CL"))
echo "$(date -u +%FT%TZ) start ${#IDS[@]} cases" >> "$LOGDIR/fill_driver.log"
for id in $IDS; do
  w0=$(date +%s)
  until mkdir "$LOCK" 2>/dev/null; do sleep 3; done
  w1=$(date +%s)
  /usr/bin/time -l .venv/bin/python -B research/broad_baseline/run_broad.py --case-list "$CL" \
     --runs-root docs/production/runlog/broad_baseline/runs --lanes best_f1 --only "$id" \
     --work-root /private/tmp/claude-501/sherman_broad_baseline_work/gold_v2 \
     --step-timeout 600 --lane-timeout 600 --rss-limit-gib 9 \
     > "$LOGDIR/$id.stdout.txt" 2> "$LOGDIR/$id.time.txt"
  rc=$?
  rmdir "$LOCK"
  w2=$(date +%s)
  echo "$(date -u +%FT%TZ) $id rc=$rc lock_wait_s=$((w1-w0)) held_s=$((w2-w1)) $(grep -- '->' "$LOGDIR/$id.stdout.txt" | tr -s ' ')" >> "$LOGDIR/fill_driver.log"
  cat "$LOGDIR/$id.stdout.txt" >> docs/production/runlog/broad_baseline/runs/driver.log
  sleep 2
done
echo "$(date -u +%FT%TZ) done" >> "$LOGDIR/fill_driver.log"

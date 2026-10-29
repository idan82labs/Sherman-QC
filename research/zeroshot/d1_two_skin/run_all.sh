#!/bin/bash
# Run run_d1.py over every GOLD_SET_V1 case, each job through the shared 3-slot semaphore.
# usage: run_all.sh WORKERS OUT_DIR [MID_DIR]
set -u
REPO=/Users/idantal/ShermanQC/FullProject
W=${1:-2}; OUT=$2; MID=${3:-}
cd "$REPO"
IDS=$(.venv/bin/python -c "import json;print(' '.join(c['case_id'] for c in json.load(open('docs/production/runlog/gold_set_v1/case_list_gold_v1.json'))['cases']))")
run_one() {
  id=$1
  [ -s "$OUT/$id.json" ] && { echo "skip $id"; return; }
  extra=""; [ -n "$MID" ] && extra="--no-cad --write-mid $MID"   # the mid pass skips the STEP check
  /private/tmp/claude-501/sherman_slot.sh env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
    research/zeroshot/d1_two_skin/run_d1.py --case-id "$id" --out "$OUT" $extra \
    > "$OUT/logs/$id.log" 2>&1
  echo "$id rc=$? $(tail -1 "$OUT/logs/$id.log")"
}
export -f run_one; export OUT MID
mkdir -p "$OUT/logs"
printf "%s\n" $IDS | xargs -P "$W" -I{} bash -c 'run_one {}'

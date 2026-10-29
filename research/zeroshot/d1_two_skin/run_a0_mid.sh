#!/bin/bash
# A0 (strict-blind scan-only lane, research/broad_baseline/run_lane.py without --cad) on the D1
# mid-surface clouds, plus a determinism control on original scans. Each job holds one shared slot.
# usage: run_a0_mid.sh WORKERS MID_PLY_DIR OUT_DIR [CONTROL_IDS...]
set -u
REPO=/Users/idantal/ShermanQC/FullProject
W=$1; MID=$2; OUT=$3; shift 3; CONTROL="$*"
cd "$REPO"
JOBS=$(.venv/bin/python - "$MID" "$CONTROL" <<'PY'
import json, sys
mid, control = sys.argv[1], sys.argv[2].split()
base = {c["case_id"]: c for c in json.load(open("docs/production/runlog/gold_set_v1/case_list_gold_v1.json"))["cases"]}
for b in base.values():
    pid = b.get("prediction_case_id") or b["case_id"]
    print(f"mid|{pid}|{mid}/mid_{b['case_id']}.ply")
    if pid in control:
        print(f"orig|{pid}|{b['scan_path']}")
PY
)
run_one() {
  IFS='|' read kind pid scan <<< "$1"
  o="$OUT/$kind/$pid"
  [ -s "$o/predictions_a0.json" ] && { echo "skip $kind $pid"; return; }
  [ -f "$REPO/docs/production/runlog/broad_baseline/runs/best_f1/$pid/predictions_a0.json" ] || { echo "nobase $pid"; return; }
  mkdir -p "$o" "$OUT/locks"
  /private/tmp/claude-501/sherman_slot.sh env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B research/broad_baseline/run_lane.py \
    --lane best_f1 --scan "$scan" --out "$o" --work-dir "$OUT/work/$kind/$pid" --lock "$OUT/locks/${kind}_$pid.lock" \
    --rss-limit-gib 12 > "$o/driver.log" 2>&1
  echo "$kind $pid rc=$? $(python3 -c "import json;d=json.load(open('$o/predictions_a0.json'));print(d.get('status'),d.get('line_count'))" 2>&1 | tail -1)"
}
export -f run_one; export OUT REPO
printf "%s\n" $JOBS | xargs -P "$W" -I{} bash -c 'run_one "{}"'

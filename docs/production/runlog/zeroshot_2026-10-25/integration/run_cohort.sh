#!/bin/bash
# usage: run_cohort.sh <scan_list.tsv: sid<TAB>path> <out_root> <variants comma> <worker i> <n workers> [--final-testb]
# Runs each zs_lane_v1 variant on each scan, one process per (scan, variant), each through the harness guard
# (tools/guarded_slot.sh: gold-ledger guard, then one geometry slot, OMP 1). Output: <out_root>/<variant>/<sid>/eval_lines.json
REPO=/Users/idantal/ShermanQC/FullProject
G=$REPO/docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh
mkdir -p "$2"; list="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; out="$(cd "$2" && pwd)"; vars="$3"; wi="$4"; wn="$5"; flag="$6"
cd "$REPO"
n=0
while IFS=$'\t' read -r sid path; do
  n=$((n+1)); [ $(( (n-1) % wn )) -eq "$wi" ] || continue
  for v in ${vars//,/ }; do
    d="$out/$v/$sid"; mkdir -p "$d"
    [ -f "$d/eval_lines.json" ] && continue
    $G "$sid" $flag -- nice .venv/bin/python -B -m research.zeroshot.zs_lane_v1.lane --scan "$path" \
      --out "$d/eval_lines.json" --config research/zeroshot/zs_lane_v1/config_v1.json --variant "$v" \
      --record "$d/run_record.json" $flag > "$d/stdout.log" 2>&1
    echo "$sid $v rc=$? $(tail -1 $d/stdout.log)"
  done
done < "$list"

#!/bin/bash
# usage: run_cohort.sh <scan_list.tsv: sid<TAB>path> <out_root> <worker i> <n workers>
# Runs the frozen zs_lane_v3 (config_v3.json, variant zs, --strict-pins) on each scan, one process per scan, each
# through the harness guard (tools/guarded_slot.sh: gold-ledger guard, then one geometry slot, OMP 1).
# Output: <out_root>/zs/<sid>/eval_lines.json + run_record.json + stdout.log. There is no TEST-B mode.
REPO=/Users/idantal/ShermanQC/FullProject
G=$REPO/docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh
mkdir -p "$2"; list="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; out="$(cd "$2" && pwd)"; wi="$3"; wn="$4"
cd "$REPO"
n=0
while IFS=$'\t' read -r sid path; do
  n=$((n+1)); [ $(( (n-1) % wn )) -eq "$wi" ] || continue
  d="$out/zs/$sid"; mkdir -p "$d"
  [ -f "$d/eval_lines.json" ] && continue
  $G "$sid" -- nice .venv/bin/python -B -m research.zeroshot.zs_lane_v3.lane --scan "$path" \
    --out "$d/eval_lines.json" --config research/zeroshot/zs_lane_v3/config_v3.json --variant zs \
    --record "$d/run_record.json" --strict-pins > "$d/stdout.log" 2>&1
  echo "$sid rc=$? $(tail -1 $d/stdout.log)"
done < "$list"

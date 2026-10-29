#!/bin/bash
# usage: run_cohort.sh <scan_list.tsv: sid<TAB>path> <out_root> <worker i> <n workers> [--posthoc-consumed-testb]
# Runs the frozen zs_lane_v2 (config_v2.json, variant zs) on each scan, one process per scan, each through the harness
# guard (tools/guarded_slot.sh: gold-ledger guard, then one geometry slot, OMP 1). Output: <out_root>/zs/<sid>/eval_lines.json
# --posthoc-consumed-testb: the CONSUMED TEST-B re-score (never a held-out result); the guard is called with its
# one-time flag only inside lane.py after the TEST-B consumption record is checked, and every output is stamped
# POSTHOC_CONSUMED.
REPO=/Users/idantal/ShermanQC/FullProject
G=$REPO/docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh
mkdir -p "$2"; list="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; out="$(cd "$2" && pwd)"; wi="$3"; wn="$4"; flag="$5"
cd "$REPO"
n=0
while IFS=$'\t' read -r sid path; do
  n=$((n+1)); [ $(( (n-1) % wn )) -eq "$wi" ] || continue
  d="$out/zs/$sid"; mkdir -p "$d"
  [ -f "$d/eval_lines.json" ] && continue
  if [ "$flag" = "--posthoc-consumed-testb" ]; then
    # the harness guard seals TEST-B; lane.py performs the post-hoc check itself, so only the slot is taken here
    /private/tmp/claude-501/sherman_slot.sh nice .venv/bin/python -B -m research.zeroshot.zs_lane_v2.lane --scan "$path" \
      --out "$d/eval_lines.json" --config research/zeroshot/zs_lane_v2/config_v2.json --variant zs \
      --record "$d/run_record.json" --strict-pins --posthoc-consumed-testb > "$d/stdout.log" 2>&1
  else
    $G "$sid" -- nice .venv/bin/python -B -m research.zeroshot.zs_lane_v2.lane --scan "$path" \
      --out "$d/eval_lines.json" --config research/zeroshot/zs_lane_v2/config_v2.json --variant zs \
      --record "$d/run_record.json" --strict-pins > "$d/stdout.log" 2>&1
  fi
  echo "$sid rc=$? $(tail -1 $d/stdout.log)"
done < "$list"

#!/bin/bash
# usage: [ZS_RS_CONFIG=research/zeroshot/zs_lane_v3_rs/config_v3r3_rs.json] bench.sh <scan_list.tsv: sid<TAB>path> <out_root> "<thread counts, e.g. 1 4 8 16>"
# zs_lane_v3_rs CLI (the zs_rust binary), one process per scan and thread count, through the harness guard + one
# geometry slot (tools/guarded_slot.sh), --strict-pins, thread counts interleaved per scan.
# Writes <out>/t<N>/<sid>/{eval_lines.json,run_record.json,stdout.log,time.txt}. Timing / equivalence only:
# nothing here reads gold or scores.
REPO=/Users/idantal/ShermanQC/FullProject
G=$REPO/docs/production/runlog/zeroshot_2026-10-25/tools/guarded_slot.sh
list="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; mkdir -p "$2"; out="$(cd "$2" && pwd)"; ths="$3"
cd "$REPO"
CFG=${ZS_RS_CONFIG:-research/zeroshot/zs_lane_v3_rs/config_v3_rs.json}
while IFS=$'\t' read -r sid scanpath; do
  for th in $ths; do
    d="$out/t$th/$sid"; mkdir -p "$d"
    [ -f "$d/run_record.json" ] && continue
    $G "$sid" -- env ZS_RUST_THREADS=$th /usr/bin/time -l .venv/bin/python -B -m research.zeroshot.zs_lane_v3_rs.lane \
      --scan "$scanpath" --out "$d/eval_lines.json" --config "$CFG" --variant zs \
      --record "$d/run_record.json" --strict-pins > "$d/stdout.log" 2> "$d/time.txt"
    echo "$sid t$th rc=$? $(tail -1 $d/stdout.log)"
  done
done < "$list"

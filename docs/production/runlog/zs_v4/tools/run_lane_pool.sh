#!/bin/bash
# Run the frozen zs_lane_v4 lane CLI (--strict-pins) on every pool scan, one job at a time through the geometry slot
# (OMP 1, nice 15, RSS watchdog 16 GiB; 32 GiB above 3M faces), 1800 s per job, resumable. The scan is staged as
# <tmp>/input/scan.<ext> (symlink), so no file name reaches the lane. Outputs: results/lane_runs/<tag>/<case>.json
# (+ .rec.json run record). usage (repo root): nohup .../run_lane_pool.sh <tag> [case_id ...] > log 2>&1 &
REPO=/Users/idantal/ShermanQC/FullProject
cd "$REPO" || exit 1
TAG=${1:?tag}; shift
OUT=docs/production/runlog/zs_v4/results/lane_runs/$TAG
mkdir -p "$OUT"
SLOT=docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh
.venv/bin/python -B - "$@" <<'EOF' > /private/tmp/claude-501/zs_v4_lane_order_$TAG.tsv
import json, sys
p = json.load(open("docs/production/runlog/zs_v4/POOL.json"))["scans"]
want = set(sys.argv[1:])
for s in sorted(p, key=lambda s: (s.get("face_count") or 0)):
    if want and s["case_id"] not in want:
        continue
    ext = s["path"].rsplit(".", 1)[-1].lower()
    print(f"{s['case_id']}\t{s.get('face_count') or 0}\t{s['path']}\t{ext}")
EOF
while IFS=$'\t' read -r cid fc src ext; do
  [ -f "$OUT/$cid.json" ] && continue
  lim=16; [ "$fc" -gt 3000000 ] && lim=32
  st=$(mktemp -d /private/tmp/claude-501/zs_v4_lane_XXXXXX); mkdir -p "$st/input"; ln -s "$src" "$st/input/scan.$ext"
  echo "$(date +%H:%M:%S) start $cid faces=$fc"
  SHERMAN_RSS_LIMIT_GB=$lim $SLOT perl -e 'alarm 1800; exec @ARGV' .venv/bin/python -B -m research.zeroshot.zs_lane_v4.lane \
     --scan "$st/input/scan.$ext" --out "$OUT/$cid.json" --record "$OUT/$cid.rec.json" --strict-pins
  echo "$(date +%H:%M:%S) end $cid rc=$?"
  rm -rf "$st"
done < /private/tmp/claude-501/zs_v4_lane_order_$TAG.tsv
echo "ALL DONE $(date)"

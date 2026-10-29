#!/bin/bash
# Build the v4 candidate cache for every pool scan, one job at a time through the geometry slot
# (OMP 1, nice 15, RSS watchdog 16 GiB; 32 GiB for scans above 3M faces), 1800 s per job, resumable.
# usage (repo root): nohup docs/production/runlog/zs_v4/tools/run_cache.sh [tool] [suffix] > .../cache/run_cache.log 2>&1 &
REPO=/Users/idantal/ShermanQC/FullProject
cd "$REPO" || exit 1
TOOL=${1:-docs/production/runlog/zs_v4/tools/cache_build.py}
SUF=${2:-.json.gz}
SLOT=docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh
.venv/bin/python -B - <<'EOF' > /private/tmp/claude-501/zs_v4_order.tsv
import json
p = json.load(open("docs/production/runlog/zs_v4/POOL.json"))["scans"]
first = ["gb_68cd06", "gb_959f4a", "gb_42dd99"]
rest = sorted((s for s in p if s["case_id"] not in first), key=lambda s: (s.get("face_count") or 0))
for s in [x for x in p if x["case_id"] in first] + rest:
    print(f"{s['case_id']}\t{s.get('face_count') or 0}")
EOF
while IFS=$'\t' read -r cid fc; do
  [ -f "docs/production/runlog/zs_v4/cache/${cid}${SUF}" ] && continue
  lim=16; [ "$fc" -gt 3000000 ] && lim=32
  echo "$(date +%H:%M:%S) start $cid faces=$fc rss_limit=${lim}G"
  SHERMAN_RSS_LIMIT_GB=$lim $SLOT perl -e 'alarm 1800; exec @ARGV' .venv/bin/python -B "$TOOL" "$cid"
  echo "$(date +%H:%M:%S) end $cid rc=$?"
done < /private/tmp/claude-501/zs_v4_order.tsv
echo "ALL DONE $(date)"

#!/bin/bash
# Final eval FE1, Stage 2 step 4: one shared marker prep (patches + renders + points_ds.npz) per packet item.
# Per item: run_allowlist admits the packet copy by sha256 (ledger seq 39 -> seq 35) and confirms the staged byte copy;
# gold_guard check is recorded (it refuses REGISTERED scans: GOLD_NOT_FROZEN is the engine gate, prep is not an engine);
# prep runs on the staged guard-clean copy under the geometry slot (tools/sherman_slot.sh: OMP 1, nice 15, one job,
# RSS watchdog 16 GiB; 32 GiB only for the two dense scans). One attempt per item; failures are recorded, not retried.
# usage (repo root): bash docs/production/runlog/final_eval_2026-10-28_heldout/tools/prep_fe1.sh
R=/Users/idantal/ShermanQC/FullProject
P=$R/docs/production/runlog/final_eval_2026-10-28_heldout/packets
S=$R/docs/production/runlog/fe1_stage_2026-10-28/scans
SLOT=$R/docs/production/runlog/zeroshot_2026-10-25/tools/sherman_slot.sh
ST=$R/docs/production/runlog/final_eval_2026-10-28_heldout/PREP_STATUS_FE1.tsv
cd "$R" || exit 1
mkdir -p "$P/FE1"
[ -f "$ST" ] || printf "gid\tfaces\trss_limit_gb\tgold_guard\texit\twall_s\n" > "$ST"
for f in "$S"/gb_*.ply; do
  g=$(basename "$f" .ply)
  n=$(head -c 2000 "$f" | LC_ALL=C grep -a "^element face" | awk '{print $3}' | tr -d "\r")
  echo "$n $g"
done | sort -n | while read -r n g; do
  W=$P/FE1/$g
  if [ -e "$W/prep" ]; then echo "skip $g (prep exists)"; continue; fi
  mkdir -p "$W"
  .venv/bin/python -B -m research.zeroshot.common.run_allowlist check --scan "$P/scans/$g.ply" --by stage2-prep \
    > "$W/allowlist_check.json" || { printf "%s\t%s\t-\t-\tALLOWLIST_REFUSED\t0\n" "$g" "$n" >> "$ST"; continue; }
  gg=$(PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m research.zeroshot.common.gold_guard check --scan "$S/$g.ply" \
       | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("reason") or d.get("basis"))')
  lim=16; [ "${n:-0}" -gt 5000000 ] && lim=32
  t0=$(date +%s)
  SHERMAN_RSS_LIMIT_GB=$lim "$SLOT" env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    .venv/bin/python -B -m research.bench_labeling prep --scan "$S/$g.ply" --out "$W/prep" > "$W/prep.log" 2>&1
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$g" "$n" "$lim" "$gg" "$rc" "$(( $(date +%s) - t0 ))" >> "$ST"
  echo "$g faces=$n rc=$rc"
done
echo DONE

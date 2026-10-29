#!/bin/bash
# gen_one.sh KEY CDIR STAGE MAP SEED VARIANTS : synthetic scans of one part-stage (skips when meta.json exists)
W=/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work
[ -f $W/scans/$1/meta.json ] && exit 0
mkdir -p $W/scans
cd /Users/idantal/ShermanQC/FullProject
$W/tools/slot.sh env PYTHONPATH=.:backend .venv/bin/python -m research.chirality.synth_scans --compiled $2 --stage $3 --map $4 --seed $5 --variants $6 --out $W/scans/$1 > $W/scans/$1.log 2>&1
echo "$1 rc=$?"

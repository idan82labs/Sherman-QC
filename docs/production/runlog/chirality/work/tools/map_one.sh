#!/bin/bash
# map_one.sh ID CDIR : chirality map of one compiled part into work/maps/ID (skips when done)
W=/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work
[ -f $W/maps/$1/chirality_map_v0.json ] && exit 0
mkdir -p $W/maps
cd /Users/idantal/ShermanQC/FullProject
$W/tools/slot.sh env PYTHONPATH=.:backend .venv/bin/python -m research.chirality.chirality_map $2 --out $W/maps/$1 > $W/maps/$1.log 2>&1
echo "$1 rc=$?"

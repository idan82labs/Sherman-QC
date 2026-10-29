#!/bin/zsh
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter6
cd /Users/idantal/ShermanQC/FullProject
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python $W/tools/fp.py $W/logs/fp_chain3_start.json > /dev/null
rm -rf $W/revprobes/v3
zsh $W/tools/lk.sh revsweep_v3 zsh $W/tools/revsweep.sh v3
zsh $W/tools/phys6.sh rev3
zsh $W/tools/g1_6.sh $W/g1_rev3
zsh $W/tools/sbsweep6.sh rev3
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python $W/tools/fp.py $W/logs/fp_chain3_end.json > /dev/null
echo CHAIN3_DONE

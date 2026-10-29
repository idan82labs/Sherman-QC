#!/bin/zsh
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1i4
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter4
cd /Users/idantal/ShermanQC/FullProject
LOCK=/private/tmp/claude-501/sherman_geom.lock
until mkdir $LOCK 2>/dev/null; do sleep 3; done
P=$W/sbprobes/scans_s23/DEV-4D512B5C56/S1/scan.ply
C=$W/phys/compiled/dev/DEV-4D512B5C56
R=docs/production/runlog/freeze/recipes/DEV-4D512B5C56.recipe.json
run() { name=$1; shift; echo "### $name"; env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 MPLCONFIGDIR=$S/mpl .venv/bin/python $S/tools/dbg_scan.py $C $R $P "$@" 2>&1 | grep -E "^hyp|ambig|hole-pair|planes=" | cut -c1-220; }
run current
run clearance15 --cfg hole_core_clearance_mm=1.5
run walks0 --cfg fine_search_walks=0
run refine3 --cfg refine_min_information=3.0
run holepairs0 --cfg hole_pair_hypotheses=0
run orient0 --cfg child_orient_max_deg=0.0
rmdir $LOCK
echo TOGGLE_DONE

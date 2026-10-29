#!/bin/zsh
# re-check the regressed 38172000 springback probes (S1, S3) and the 28870000 probes (S0 seed 7, S1 seed 23) with the current engine
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1i4
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter4
cd /Users/idantal/ShermanQC/FullProject
LOCK=/private/tmp/claude-501/sherman_geom.lock
until mkdir $LOCK 2>/dev/null; do sleep 3; done
run() { name=$1; c=$2; p=$3; echo "### $name"; env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 MPLCONFIGDIR=$S/mpl .venv/bin/python $S/tools/dbg_scan.py $W/phys/compiled/dev/$c docs/production/runlog/freeze/recipes/$c.recipe.json $p 2>&1 | grep -E "^hyp0|ambig|hole-pair" | cut -c1-220; }
run 38172000_S1_sb23 DEV-4D512B5C56 $W/sbprobes/scans_s23/DEV-4D512B5C56/S1/scan.ply
run 38172000_S3_sb23 DEV-4D512B5C56 $W/sbprobes/scans_s23/DEV-4D512B5C56/S3/scan.ply
run 28870000_S1_sb23 DEV-AAB8C67C31 $W/sbprobes/scans_s23/DEV-AAB8C67C31/S1/scan.ply
run 28870000_S0_s7 DEV-AAB8C67C31 $W/dev/aab8_s0_s7/scan.ply
run 28870000_FULL_real DEV-AAB8C67C31 /Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/28870000_A_FULL_SCAN.ply
rmdir $LOCK
echo PROBE_DONE

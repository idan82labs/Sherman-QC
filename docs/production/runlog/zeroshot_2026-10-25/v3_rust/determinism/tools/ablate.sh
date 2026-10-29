#!/bin/zsh
# Per-operation ablation of the portable build vs the Python reference (DEV rev 2, 16 threads), serial.
# usage: ablate.sh SCRATCH_OUT
cd /Users/idantal/ShermanQC/FullProject
O=$1; B=research/zeroshot/zs_rust_det/target/release/zs_rust; T=docs/production/runlog/zeroshot_2026-10-25/v3_rust
M=(sin cos tan acos atan2 exp ln hypot pow); A=(xtx gemm eigh solve det lstsq)
run() { # label hostmath accel
  [[ -f $O/$1.cmp.json ]] && return
  .venv/bin/python -B $T/determinism/tools/det_run.py --bin $B --out $O/$1 --env ZS_HOST_MATH=$2 --env ZS_ACCEL=$3 >/dev/null
  .venv/bin/python -B $T/tools/cmp_eval.py $O/$1 --out $O/$1.cmp.json | sed -n 3p | sed "s/^/$1 /"
  rm -rf $O/$1
}
join() { local IFS=,; echo "$*"; }
run portable "" ""
run accel_only_native "" all
run math_only_native all ""
for x in $M; do run "loo_math_$x" $(join ${M:#$x}) all; done
for x in $A; do run "loo_op_$x" all $(join ${A:#$x}); done
for x in $M; do run "add_math_$x" $x ""; done
for x in $A; do run "add_op_$x" "" $x; done
echo DONE

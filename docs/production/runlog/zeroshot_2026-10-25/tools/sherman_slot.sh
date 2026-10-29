#!/bin/bash
# One geometry slot: single-threaded BLAS/OMP, nice 15, one heavy job at a time (mkdir lock),
# RSS watchdog on the process tree (default 16 GiB; SHERMAN_RSS_LIMIT_GB=32 for an approved dense job).
LIMIT_KB=$(( ${SHERMAN_RSS_LIMIT_GB:-16} * 1024 * 1024 ))
LOCK=/private/tmp/claude-501/sherman_slot.lock
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p /private/tmp/claude-501
release() { [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"; }   # only the owner releases
until mkdir "$LOCK" 2>/dev/null; do
  p=$(cat "$LOCK/pid" 2>/dev/null); [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null && rm -rf "$LOCK"; sleep 2
done
echo $$ > "$LOCK/pid"; trap release EXIT
nice -n 15 "$@" & child=$!
tree_rss() { local t=0 q=($1); while [ ${#q[@]} -gt 0 ]; do local p=${q[0]}; q=("${q[@]:1}")
  t=$(( t + $(ps -o rss= -p $p 2>/dev/null || echo 0) )); q+=($(pgrep -P $p)); done; echo $t; }
while kill -0 $child 2>/dev/null; do
  r=$(tree_rss $child); if [ "$r" -gt "$LIMIT_KB" ]; then echo "sherman_slot: RSS ${r}KB > limit, killing" >&2
    pkill -TERM -P $child; kill -TERM $child; wait $child; exit 137; fi; sleep 2
done
wait $child

#!/bin/zsh
cd /Users/idantal/ShermanQC/FullProject
while IFS= read -r line; do
  h=${line%%  *}; p=${line#*  }
  until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
  OMP_NUM_THREADS=1 /usr/bin/time -l ./.venv/bin/python /private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/step_one.py "$p" "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/benchmark_v3/step_v10/${h[1,12]}.json" > "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/benchmark_v3/step_v10/${h[1,12]}.log" 2>&1; rc=$?
  rmdir /private/tmp/claude-501/sherman_geom.lock
  echo "$h\t$rc\t$p"
done < /private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/steps.txt > /Users/idantal/ShermanQC/FullProject/docs/production/runlog/benchmark_v3/step_v10/_rc.tsv
echo DONE

#!/bin/zsh
# usage: run_cases.sh TAG case1 case2 ...   (runs E3 CLI on iter-0 staged inputs, serially, under the shared lock)
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
P0=$S/a1_iter0/phys
TAG=$1; shift
cd /Users/idantal/ShermanQC/FullProject
for c in "$@"; do
  sku=$(python3 -c "import json;print(json.load(open('$P0/physical/E3/$c/engine/result.json'))['sku'])")
  sha=$(python3 -c "import json;print(json.load(open('$P0/physical/E3/$c/record.json'))['scan_sha256'])")
  out=$S/a1_iter1/runs/$TAG/$c
  rm -rf $out; mkdir -p $(dirname $out)
  until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
  /usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python -m domains.production.engine.inspect \
     --compiled $P0/compiled/dev/$sku --recipe $P0/physical/E3/$c/input/recipe.json --scan $(ls $P0/physical/E3/$c/input/scan.* | head -1) \
     --out $out --scan-sha256 $sha > $out.stdout 2> $out.stderr
  rc=$?
  rmdir /private/tmp/claude-501/sherman_geom.lock
  echo "== $c rc=$rc $(grep RESULT $out.stdout | cut -c1-200) rss=$(grep 'maximum resident' $out.stderr | awk '{print $1/1073741824 " GiB"}')"
done

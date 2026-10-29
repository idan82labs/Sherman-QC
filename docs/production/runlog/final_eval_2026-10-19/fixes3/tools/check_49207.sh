#!/bin/zsh
# D2 mandatory check on the consumed FRESH 49207000 weldment scan (diagnostic; never a RUN): E3 under the shared lock,
# then the named test that asserts VALID and explained >= 0.85. usage: check_49207.sh OUT_DIR
SP=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
C=$SP/fixes/compile_after1/DEV-7052AE8973
OUT=$1
cd /Users/idantal/ShermanQC/FullProject
until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python $SP/fixes/run_e3.py \
  "data/bend_corpus/incoming_codex_fetch_2026-10-17/20260901_e8bfd4f6/49207000_B.ply" $C/compiled $C/dev.recipe.json $OUT | head -1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend SHERMAN_D2_CONSUMED_49207=$OUT/engine .venv/bin/python -m pytest -c /dev/null \
  --noconftest -p no:cacheprovider tests/test_prod_engine_fixes3_2026_10_19.py -q -k 49207
rc=$?
rmdir /private/tmp/claude-501/sherman_geom.lock
exit $rc

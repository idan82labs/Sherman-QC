#!/bin/zsh
# A1 iter-4 final validation of the final engine: tests, G1 regression tier, physical run; each step under the lock.
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1i4
cd /Users/idantal/ShermanQC/FullProject
FP='from domains.production.engine.eval.freeze import current_code_fingerprints as f; import json; print(json.dumps(f()))'
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -c "$FP" > $S/logs/fingerprints_final_start.json
zsh $S/tools/locked.sh tests_final env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_prod_engine_inspect_cli.py tests/test_prod_engine_inspect_units.py tests/test_prod_engine_inspect_a1.py tests/test_prod_engine_inspect_a1_iter2.py tests/test_prod_engine_inspect_a1_iter3.py tests/test_prod_engine_inspect_a1_iter4.py -q
echo STEP_TESTS_DONE
zsh $S/tools/g1.sh $S/g1_final
echo STEP_G1_DONE
zsh $S/tools/phys.sh v4
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -c "$FP" > $S/logs/fingerprints_final_end.json
echo ALL_DONE

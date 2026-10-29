#!/bin/zsh
# A1 iter-5 (second start) validation of fix C: tests, G1 regression tier, physical set, CAD-stage probe sweep.
# Every heavy step runs under the shared lock (locked.sh / sbsweep.sh), one at a time, OMP_NUM_THREADS=1.
W=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter5
cd /Users/idantal/ShermanQC/FullProject
FP='from domains.production.engine.eval.freeze import current_code_fingerprints as f; import json; print(json.dumps(f()))'
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -c "$FP" > $W/logs/fingerprints_fixC_start.json
zsh $W/tools/locked.sh tests_fixC env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_prod_engine_inspect_cli.py tests/test_prod_engine_inspect_units.py tests/test_prod_engine_inspect_a1.py tests/test_prod_engine_inspect_a1_iter2.py tests/test_prod_engine_inspect_a1_iter3.py tests/test_prod_engine_inspect_a1_iter4.py tests/test_prod_engine_inspect_a1_iter5.py -q
echo STEP_TESTS_DONE
zsh $W/tools/g1.sh $W/g1_fixC
echo STEP_G1_DONE
zsh $W/tools/phys.sh vC
echo STEP_PHYS_DONE
zsh $W/tools/sbsweep.sh vC
echo STEP_SWEEP_DONE
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -c "$FP" > $W/logs/fingerprints_fixC_end.json
echo ALL_DONE

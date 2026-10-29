#!/bin/zsh
# A1 iter-3 final re-validation after the short-child support cut: every heavy step one at a time under the shared lock.
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
M=$S/a1_iter3
cd /Users/idantal/ShermanQC/FullProject
FP='from domains.production.engine.eval.freeze import current_code_fingerprints as f; import json; print(json.dumps(f()))'
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -c "$FP" > $M/logs/fingerprints_final3_start.json
$M/tools/g1.sh $M/g1_final3
$M/tools/locked.sh t_cli_final3 env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_prod_engine_inspect_cli.py tests/test_prod_engine_inspect_units.py tests/test_prod_engine_inspect_a1.py tests/test_prod_engine_inspect_a1_iter2.py tests/test_prod_engine_inspect_a1_iter3.py -q
$M/tools/phys.sh > $M/logs/phys_final3.out 2>&1
echo STEP_PHYS_DONE
$M/tools/pair_final.sh $(cat $M/logs/truthrel_items.txt) > $M/logs/pair_final3.stderr 2>&1
echo STEP_NOMINAL_DONE
ENGINE_ROOT=$M/final_tree $M/tools/sbsweep.sh final $(cat $M/logs/sb_items.txt) > $M/logs/sbsweep_final3.stderr 2>&1
echo STEP_SPRINGBACK_DONE
$M/tools/jprobes.sh > /dev/null 2>&1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -c "$FP" > $M/logs/fingerprints_final3_end.json
echo ALL_DONE

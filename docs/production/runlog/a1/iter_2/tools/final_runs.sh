#!/bin/zsh
S=/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad
$S/a1_iter2/tools/g1.sh $S/a1_iter2/g1_final
$S/a1_iter2/tools/phys.sh
cd /Users/idantal/ShermanQC/FullProject && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:backend .venv/bin/python -c "
from domains.production.engine.eval.freeze import current_code_fingerprints as f; import json; print(json.dumps(f()))" > $S/a1_iter2/logs/fingerprints_end.json
echo FINAL_ALL_DONE

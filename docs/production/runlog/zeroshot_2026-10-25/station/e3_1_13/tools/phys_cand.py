"""prefreeze (compile-frozen / run-physical / score-physical) with E3 run from the e3-1.13 candidate shadow tree.
The eval package is the production one (symlinked, unmodified); only run_e3_case's repo_root (the engine's cwd and
PYTHONPATH) points at the candidate tree. usage: phys_cand.py <prefreeze args...>"""
import functools, os, sys
from pathlib import Path
C = Path(os.environ.get("E3C_REPO", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3c/repo"))
sys.path[:0] = [str(C), str(C / "backend")]
from domains.production.engine.eval.freeze import physical, prefreeze  # noqa: E402
physical.run_e3_case = functools.partial(physical.run_e3_case, repo_root=C)
sys.exit(prefreeze.main(sys.argv[1:]))

"""G1 regression tier (eval run_tier) with E1/E3 run from the e3-1.13 candidate shadow tree (repo_root = tree).
usage: g1_cand.py TIER_DIR OUT [ENGINE_CONFIG]"""
import json, os, sys
from pathlib import Path
C = Path(os.environ.get("E3C_REPO", "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3c/repo"))
sys.path[:0] = [str(C), str(C / "backend")]
from domains.production.engine.eval.runner import RunnerConfig, run_tier  # noqa: E402
R = Path("/Users/idantal/ShermanQC/FullProject")
cfg = RunnerConfig(tier_dir=Path(sys.argv[1]), out_dir=Path(sys.argv[2]), repo_root=C,
                   engine_config=Path(sys.argv[3]) if len(sys.argv) > 3 else None,
                   ledger=R / "docs/production/runlog/freeze/LEDGER.json",
                   ledger_anchor="cf01e709079c0fcc348758c5196884bff7d20f7836b88c7616883c0dcec67e67")
card = run_tier(cfg)
print(json.dumps({"verdict": card["verdict"], "regression_verdict": card["regression_verdict"], "qualifying": card["qualifying"]}))

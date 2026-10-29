"""Freeze-style code fingerprints (engine, compile, eval, contracts). usage: fp.py OUT.json"""
import json, sys
from pathlib import Path
from domains.production.engine.eval.runner import code_fingerprint
R = Path("/Users/idantal/ShermanQC/FullProject")
fp = {"engine": code_fingerprint(R, ("domains/production/engine/inspect", "domains/production/engine/__init__.py")),
      "engine_dir": code_fingerprint(R, ("domains/production/engine",)),
      "compile": code_fingerprint(R, ("domains/production/engine/compile",)),
      "eval": code_fingerprint(R, ("domains/production/engine/eval",)),
      "contracts": code_fingerprint(R, ("contracts/production",))}
Path(sys.argv[1]).write_text(json.dumps(fp, indent=1))
print(json.dumps(fp))

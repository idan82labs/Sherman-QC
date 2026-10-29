"""Re-run the delivered slicer on 4 DEV scans and compare with the it7 candidate files (provenance check)."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/idantal/ShermanQC/FullProject"); sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import slicer, custody
inv = custody.dev_inventory()
out = {}
for s in ["sc_03395097cd96", "sc_3c93a0bf6f97", "sc_8935b8a8e8aa", "sc_eaa89c6d1f1f"]:
    p = custody.check_dev_scan(s, inv)
    res = slicer.detect(slicer.load_mesh(str(p)))
    old = json.loads(Path(f"/private/tmp/claude-501/z3work/dev_it7/{s}.json").read_text())["candidates"]
    a = np.array([c["start"] + c["end"] for c in old]) if old else np.zeros((0, 6))
    b = np.array([list(c["start"]) + list(c["end"]) for c in res["candidates"]]) if res["candidates"] else np.zeros((0, 6))
    same = a.shape == b.shape and (np.allclose(np.sort(a, 0), np.sort(b, 0), atol=1e-6) if len(a) else True)
    out[s] = {"old": len(old), "new": len(res["candidates"]), "identical_endpoints": bool(same)}
    print(s, out[s], flush=True)
(REPO / "docs/production/runlog/zeroshot_2026-10-25/audit_z3/determinism.json").write_text(json.dumps(out, indent=1))

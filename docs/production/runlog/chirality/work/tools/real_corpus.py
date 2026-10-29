"""real_corpus.py -> work/real_steps.json {sku: step_path}: the CAD corpus of real parts.
Rule: an inventory SKU (zeroshot_2026-10-25/INVENTORY.json) with a located STEP, at least one DEV_CONSUMED scan,
and no scan in FRESH / DEV-CHECK-only / sealed custody (CONSUMED_FRESH; a SKU whose only scans are CONSUMED_DEVCHECK
is excluded). Only the CAD is used; no scan of these SKUs is read for the synthetic suite."""
import json, os, collections
from pathlib import Path
R = Path("/Users/idantal/ShermanQC/FullProject")
W = R / "docs/production/runlog/chirality/work"
inv = json.load(open(R / "docs/production/runlog/zeroshot_2026-10-25/INVENTORY.json"))
by = collections.defaultdict(list); step = {}
for s in inv["scans"]:
    cad = s.get("cad") or {}
    if cad.get("sku") and cad.get("step_path"):
        by[cad["sku"]].append(s.get("custody")); step[cad["sku"]] = os.path.expanduser(cad["step_path"])
out = {}
for sku, cust in sorted(by.items()):
    if "DEV_CONSUMED" in cust and not any(c in ("CONSUMED_FRESH",) or "SEALED" in str(c) for c in cust):
        out[sku] = step[sku]
(W / "real_steps.json").write_text(json.dumps(out, indent=1))
print(len(out), list(out))

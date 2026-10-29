"""Append one NOTE to the freeze ledger (docs/production/runlog/freeze/LEDGER.json) via
domains.production.engine.eval.freeze.append_entry. usage: ledger_note.py <items.json> <label>"""
import json
import sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from domains.production.engine.eval import freeze  # noqa: E402

L = REPO / "docs/production/runlog/freeze/LEDGER.json"
items = json.loads(Path(sys.argv[1]).read_text())
label = sys.argv[2]
assert freeze.verify_ledger(L)["ok"]
e = freeze.append_entry(L, kind="NOTE", label=label, items=items, recorded_by="zs_v4/lead (Claude)")
print(json.dumps({"seq": e["seq"], "entry_id": e["entry_id"], "entry_sha256": e["entry_sha256"]}))
print(json.dumps(freeze.verify_ledger(L)))

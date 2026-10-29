"""Append one NOTE entry to the freeze ledger. usage: ledger_note.py ITEMS.json LABEL   (prints the entry)"""
import json, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze import append_entry, verify_ledger  # noqa: E402
L = REPO / "docs/production/runlog/freeze/LEDGER.json"
items = json.loads(Path(sys.argv[1]).read_text())
e = append_entry(L, label=sys.argv[2], items=items, kind="NOTE", recorded_by="final_eval_2026-10-19/fixes3")
print(json.dumps({"seq": e["seq"], "entry_id": e["entry_id"], "verify": verify_ledger(L)}, default=str))

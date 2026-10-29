"""Freeze the zero-shot programme dev / TEST-B split (2026-10-25), BEFORE any new engine output on these scans.
Rule (no labels, no engine output used): TEST-B = usable gold scans that no engine or lane has ever run on
(empty engine_history in INVENTORY.json), complete label, full stage, whose 5-digit family group has no scan
anywhere else in the usable inventory that an engine has seen or that lacks gold. Everything else with gold = DEV.
Writes SPLIT_FROZEN.json (ids only) and prints its sha256."""
import hashlib, json
from pathlib import Path
H = Path(__file__).resolve().parents[1]
inv = json.loads((H / "INVENTORY.json").read_text())
U = [r for r in inv["scans"] if r["usable"]]
G = [r for r in U if r["has_gold"]]
seen_groups = {r["family_group5"] for r in U if r["engine_history"] or not r["has_gold"]}
test = sorted(r["scan_id"] for r in G if not r["engine_history"] and r["stage"] == "full"
              and r["family_group5"] not in seen_groups)
dev = sorted(r["scan_id"] for r in G if r["scan_id"] not in set(test))
tg = sorted({r["family_group5"] for r in G if r["scan_id"] in set(test)})
dg = {r["family_group5"] for r in G if r["scan_id"] in set(dev)}
assert not (set(tg) & dg)
doc = {"schema": "sherman.zeroshot.split.v1", "frozen_at": "2026-10-25", "rule": __doc__,
       "inventory_sha256": hashlib.sha256((H / "INVENTORY.json").read_bytes()).hexdigest(),
       "status_note": "TEST-B scans are CONSUMED_FRESH by the 2026-10-19 rule. Using them as a sealed test needs owner sign-off "
                      "(PLAN.md decision D1); until then they are simply not opened for development.",
       "test_b": {"scans": test, "n": len(test), "family_groups": tg, "n_family_groups": len(tg)},
       "dev": {"scans": dev, "n": len(dev), "n_family_groups": len(dg)}}
body = json.dumps(doc, indent=1, sort_keys=True)
(H / "SPLIT_FROZEN.json").write_text(body)
print(len(test), len(tg), len(dev), len(dg), hashlib.sha256(body.encode()).hexdigest())

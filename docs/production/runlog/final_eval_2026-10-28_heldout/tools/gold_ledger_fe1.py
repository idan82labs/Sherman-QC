"""Final held-out evaluation 2026-10-28, Stage 2: REGISTER the FE1 batch in zeroshot_2026-10-25/gold/GOLD_LEDGER.jsonl,
the way tools/gold_ledger_newscans.py registered N1 (same hash-chained entry format; gold_guard.verify_ledger validates).

Registers the 27 scored scans of SET_MANIFEST.json as portion FINAL_EVAL_HELDOUT, batch FE1, status REGISTERED
(pre-gold). gold_guard.require_runnable therefore refuses them (GOLD_NOT_FROZEN) until gold is frozen.
The hidden canary is NOT registered (as in N1): it is a consumed DEV scan in SPLIT_FROZEN/INVENTORY whose gold of record
already exists; a REGISTERED entry would change its guard status, and its identity stays only in PRIVATE_KEY.json.
Each sha256 is checked on the packet copy (byte-identical to the original, checked at packet time); gids are found by
sha256 over packets/scans, so PRIVATE_KEY.json is not read. Refuses ids in SPLIT_FROZEN, INVENTORY or the ledger.
usage (repo root): .venv/bin/python -B docs/production/runlog/final_eval_2026-10-28_heldout/tools/gold_ledger_fe1.py register
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).resolve().parents[1]
REPO = D.parents[3]
sys.path.insert(0, str(REPO))
from research.zeroshot.common import gold_guard as GG  # noqa: E402

PORTION = "FINAL_EVAL_HELDOUT"
BATCH = "FE1"
BY = "final_eval_2026-10-28_heldout/stage2-register"


def _append(e: dict) -> dict:
    v = GG.verify_ledger()
    if not v["ok"]:
        raise SystemExit(f"refusing to append to a broken gold ledger: {v['problems']}")
    entries = GG.read_ledger()
    e = {"schema": GG.SCHEMA, "seq": len(entries) + 1, **e, "previous_sha256": entries[-1]["entry_sha256"]}
    e["entry_sha256"] = GG._jhash(e)
    with GG.LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        import os
        os.fsync(f.fileno())
    return e


def register() -> list[dict]:
    man = json.loads((D / "SET_MANIFEST.json").read_text())
    gid_of = {GG._sha256_file(p): p.stem for p in sorted((D / "packets/scans").glob("gb_*.ply"))}
    split = GG._split()
    frozen_ids = set(split["dev"]["scans"]) | set(split["test_b"]["scans"])
    inv_ids = {r["scan_id"] for r in json.loads(GG.INVENTORY.read_text())["scans"]}
    known = set(GG.latest_status(GG.read_ledger()))
    rows = sorted(man["scored"], key=lambda r: gid_of.get(r["sha256"], ""))
    for r in rows:      # check everything before the first append
        sid = f"sc_{r['sha256'][:12]}"
        if r["sha256"] not in gid_of:
            raise SystemExit(f"{sid}: no packet copy with this sha256")
        if sid in frozen_ids or sid in inv_ids or sid in known:
            raise SystemExit(f"{sid}: already in SPLIT_FROZEN/INVENTORY/gold ledger")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for r in rows:
        sha = r["sha256"]
        out.append(_append({"event": "REGISTER", "status": "REGISTERED", "scan_id": f"sc_{sha[:12]}",
                            "scan_sha256": sha, "scan_sha256_prefix_is_id": True, "path": r["copies"][0]["path"],
                            "format": "ply", "part_id": r["part_id"], "family_group5": r["family_group5"],
                            "portion": PORTION, "batch_id": BATCH, "gid": gid_of[sha], "split_sha256": GG.SPLIT_SHA256,
                            "note": f"{PORTION}: final one-time held-out evaluation 2026-10-28 (freeze seqs 35-39); "
                                    f"pre-gold, blind marking batch FE1; never tune on these scans",
                            "recorded_at": now, "recorded_by": BY}))
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["register"]:
        sys.exit(__doc__)
    for e in register():
        print(json.dumps({"seq": e["seq"], "scan_id": e["scan_id"], "gid": e["gid"], "entry_sha256": e["entry_sha256"]}))
    print(json.dumps(GG.verify_ledger()))

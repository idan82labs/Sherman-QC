"""Gold-ledger appends for NEW_UNSEEN_FAMILY_TEST batches (N1, 2026-10-25). Narrowest explicit mechanism.

Why this file exists instead of an edit to research/zeroshot/common/gold_guard.py:
  * gold_guard.py is pinned inside the code fingerprints of zs_lane_v1 (config_v1.json) and zs_lane_v2
    (config_v2.json). Editing it would break both lanes' pinned hashes (v2 --strict-pins refuses), so the guard
    itself is left byte-identical.
  * gold_guard.require_runnable already admits any scan whose latest gold-ledger status is FROZEN
    (basis GOLD_LEDGER_FROZEN). The only things N1 needs are (1) REGISTER entries for the 4 new scans, which
    gold_setup.py wrote only for DEV/TEST-B scans, and (2) a FROZEN append whose label lives in the new N1 root,
    which gold_guard.append_event refuses because it only knows the zs_v1 / zs_v1_SEALED roots.
  * Every guard refusal stays as it is: TEST-B seal, not-FROZEN, unknown scan, OD-3, split hash.

What it does (same hash-chained entry format and chain as gold_guard; gold_guard.verify_ledger validates it):
  register --batch N1   append one REGISTER/REGISTERED entry per scan of newscans_<batch>/REGISTRY.json, portion
                        NEW_UNSEEN_FAMILY_TEST. Refuses when the file sha256 differs from the registry, when the scan
                        id is in SPLIT_FROZEN (dev or test_b), in INVENTORY, or already in the ledger.
  freeze --scan sc_ --label-path P --label-sha256 H --by WHO [--note T]
                        append a STATUS/FROZEN entry for a NEW_UNSEEN_FAMILY_TEST scan only; the label must live
                        directly under data/bend_corpus/annotations/ai_bend_lines_newscans_<batch lower>_v1/ and
                        its sha256 on disk must equal H. A scan that is already FROZEN is refused (append-only).
usage (repo root): .venv/bin/python -B docs/production/runlog/zeroshot_2026-10-25/tools/gold_ledger_newscans.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from research.zeroshot.common import gold_guard as GG  # noqa: E402

PORTION = "NEW_UNSEEN_FAMILY_TEST"


def _root_for(batch: str) -> str:
    return f"ai_bend_lines_newscans_{batch.lower()}_v1"


def _append(e: dict) -> dict:
    v = GG.verify_ledger()
    if not v["ok"]:
        raise SystemExit(f"refusing to append to a broken gold ledger: {v['problems']}")
    entries = GG.read_ledger()
    e = {"schema": GG.SCHEMA, "seq": len(entries) + 1, **e,
         "previous_sha256": entries[-1]["entry_sha256"] if entries else None}
    e["entry_sha256"] = GG._jhash(e)
    with GG.LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return e


def register(batch: str, by: str) -> list[dict]:
    prog = GG.PROG
    reg = json.loads((prog / f"newscans_{batch}/REGISTRY.json").read_text())
    key = json.loads((prog / f"newscans_{batch}/PRIVATE_KEY.json").read_text())["key"]
    gid_of = {v["sha256"]: g for g, v in key.items()}
    split = GG._split()                                   # checks the frozen split hash
    frozen_ids = set(split["dev"]["scans"]) | set(split["test_b"]["scans"])
    inv_ids = {r["scan_id"] for r in json.loads(GG.INVENTORY.read_text())["scans"]}
    known = set(GG.latest_status(GG.read_ledger()))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for r in reg["scans"]:
        sha = GG._sha256_file(r["original_path"])
        if sha != r["sha256"]:
            raise SystemExit(f"{r['original_path']}: sha256 differs from REGISTRY")
        sid = f"sc_{sha[:12]}"
        if sid in frozen_ids or sid in inv_ids:
            raise SystemExit(f"{sid}: already in SPLIT_FROZEN/INVENTORY; not a new scan")
        if sid in known:
            raise SystemExit(f"{sid}: already in the gold ledger")
        e = _append({"event": "REGISTER", "status": "REGISTERED", "scan_id": sid, "scan_sha256": sha,
                     "scan_sha256_prefix_is_id": True, "path": r["original_path"], "format": "ply",
                     "part_id": r["part_id"], "family_group5": r["part_id"][:5], "portion": PORTION,
                     "batch_id": batch, "gid": gid_of[sha], "split_sha256": GG.SPLIT_SHA256,
                     "note": f"{PORTION}: {reg['rule']}", "recorded_at": now, "recorded_by": by})
        out.append(e)
    return out


def freeze(sid: str, label_path: str, label_sha256: str, by: str, note: str | None) -> dict:
    last = GG.latest_status(GG.read_ledger()).get(sid)
    if last is None:
        raise SystemExit(f"{sid} is not registered")
    if last["portion"] != PORTION:
        raise SystemExit(f"{sid}: portion {last['portion']}; use gold_guard append for DEV/TEST-B scans")
    if last["status"] == "FROZEN":
        raise SystemExit(f"{sid} is already FROZEN (append-only; a re-freeze needs an owner decision)")
    lp = Path(label_path)
    if lp.parent.name != _root_for(last["batch_id"]):
        raise SystemExit(f"a {PORTION} label of batch {last['batch_id']} must live directly under "
                         f"{_root_for(last['batch_id'])}/")
    if GG._sha256_file(lp) != label_sha256:
        raise SystemExit("label sha256 on disk differs from --label-sha256")
    keep = ("scan_id", "scan_sha256", "path", "format", "part_id", "family_group5", "portion", "batch_id", "gid",
            "split_sha256")
    return _append({"event": "STATUS", "status": "FROZEN", **{k: last[k] for k in keep},
                    "label_path": str(lp), "label_sha256": label_sha256, "note": note,
                    "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recorded_by": by})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--batch", required=True)
    r.add_argument("--by", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--scan", required=True)
    f.add_argument("--label-path", required=True)
    f.add_argument("--label-sha256", required=True)
    f.add_argument("--by", required=True)
    f.add_argument("--note")
    a = ap.parse_args(argv)
    if a.cmd == "register":
        for e in register(a.batch, a.by):
            print(json.dumps({"seq": e["seq"], "scan_id": e["scan_id"], "status": e["status"], "gid": e["gid"],
                              "entry_sha256": e["entry_sha256"]}))
    else:
        e = freeze(a.scan, a.label_path, a.label_sha256, a.by, a.note)
        print(json.dumps({"seq": e["seq"], "scan_id": e["scan_id"], "status": e["status"],
                          "entry_sha256": e["entry_sha256"]}))
    print(json.dumps(GG.verify_ledger()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

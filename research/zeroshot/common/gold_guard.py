"""Gold-ledger guard for every engine / lane driver of the zero-shot programme (PLAN.md 2.1, 2.5; owner OD-1..OD-4).

Every driver calls ``require_runnable(scan)`` (or the CLI ``check``) before an engine or lane touches a scan.
A scan is REFUSED when:
  * it is a TEST-B scan (SPLIT_FROZEN.json test_b) and ``final_testb`` is not given (the one end-of-programme run);
  * it has entries in gold/GOLD_LEDGER.jsonl and its latest status is not FROZEN (gold not yet frozen);
  * it is a TEST-B scan with ``final_testb`` but its gold is not FROZEN;
  * it is unknown: neither in the gold ledger nor a SPLIT_FROZEN dev scan (existing gold of record);
  * ``cad_path`` is set and it is one of the 5 OD-3 DEV-CHECK never-run cases.
Allowed: SPLIT_FROZEN dev scans (gold of record already exists) and ledger scans whose latest status is FROZEN.
The guard reads only the split, the inventory's scan paths and the gold ledger; never labels or engine output.
It also owns appends to the gold ledger (hash-chained, append-only).

CLI (repo root):
  .venv/bin/python -m research.zeroshot.common.gold_guard check --scan <path|sc_id> [--final-testb] [--cad-path]
      exit 0 = allowed, 3 = refused (reason on stdout as JSON)
  .venv/bin/python -m research.zeroshot.common.gold_guard verify
  .venv/bin/python -m research.zeroshot.common.gold_guard status [--scan sc_id]
  .venv/bin/python -m research.zeroshot.common.gold_guard append --scan sc_id --status S [--label-path P --label-sha256 H]
      [--note TEXT] [--by WHO]       (S in REGISTERED, MARKING, ADJUDICATING, FROZEN, REJECTED)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROG = REPO / "docs/production/runlog/zeroshot_2026-10-25"
SPLIT = PROG / "SPLIT_FROZEN.json"
SPLIT_SHA256 = "d25ffd4f4d509624bb70387722d9a8814922cf85fa5d3bf821634db708468642"
INVENTORY = PROG / "INVENTORY.json"
LEDGER = PROG / "gold/GOLD_LEDGER.jsonl"
SCHEMA = "sherman.zeroshot.gold_ledger.v1"
STATUSES = ("REGISTERED", "MARKING", "ADJUDICATING", "FROZEN", "REJECTED")
OD3_NEVER_RUN = frozenset({"sc_eaa89c6d1f1f", "sc_152f1b1aacb2", "sc_c623d2e45bff", "sc_9ef2cf4e52cb",
                           "sc_7a1773eca9fa"})
_SC = re.compile(r"^sc_[0-9a-f]{12}$")


class GoldGuardRefusal(PermissionError):
    def __init__(self, reason: str, detail: dict):
        super().__init__(f"{reason}: {json.dumps(detail, sort_keys=True)}")
        self.reason, self.detail = reason, detail


def _jhash(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_ledger(path=None) -> list[dict]:
    p = Path(LEDGER if path is None else path)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def verify_ledger(path=None) -> dict:
    problems, prev = [], None
    entries = read_ledger(path)
    for i, e in enumerate(entries, 1):
        if e.get("schema") != SCHEMA:
            problems.append(f"{i}:SCHEMA")
        if e.get("seq") != i:
            problems.append(f"{i}:SEQUENCE")
        if e.get("previous_sha256") != prev:
            problems.append(f"{i}:CHAIN")
        if e.get("entry_sha256") != _jhash({k: v for k, v in e.items() if k != "entry_sha256"}):
            problems.append(f"{i}:HASH")
        if e.get("status") not in STATUSES:
            problems.append(f"{i}:STATUS")
        prev = e.get("entry_sha256")
    return {"ok": not problems, "entries": len(entries), "problems": problems}


def _split() -> dict:
    b = SPLIT.read_bytes()
    if hashlib.sha256(b).hexdigest() != SPLIT_SHA256:
        raise GoldGuardRefusal("SPLIT_CHANGED", {"path": str(SPLIT)})
    return json.loads(b)


def latest_status(entries: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in entries:
        out[e["scan_id"]] = e
    return out


def identify(scan: str, entries: list[dict] | None = None) -> str:
    """scan id for a path or sc_ id: by known path (ledger, inventory), else by sha256 (a hard link or copy)."""
    if _SC.match(scan):
        return scan
    entries = read_ledger() if entries is None else entries
    p = Path(os.path.expanduser(scan)).resolve()
    for e in entries:
        if Path(e["path"]).resolve() == p:
            return e["scan_id"]
    inv = json.loads(INVENTORY.read_text())["scans"]
    for r in inv:
        if r.get("path") and Path(os.path.expanduser(r["path"])).resolve() == p:
            return r["scan_id"]
    if not p.is_file():
        raise GoldGuardRefusal("UNKNOWN_SCAN", {"scan": scan, "why": "not a file and not a known path"})
    sha = _sha256_file(p)
    for e in entries:
        if e["scan_sha256"] == sha:
            return e["scan_id"]
    ids = {r["scan_id"] for r in inv}
    if f"sc_{sha[:12]}" in ids:
        return f"sc_{sha[:12]}"
    raise GoldGuardRefusal("UNKNOWN_SCAN", {"scan": scan, "sha256_prefix": sha[:12]})


def require_runnable(scan: str, *, final_testb: bool = False, cad_path: bool = False) -> dict:
    """Return a decision dict when an engine may run on ``scan``; raise GoldGuardRefusal otherwise."""
    v = verify_ledger()
    if not v["ok"]:
        raise GoldGuardRefusal("GOLD_LEDGER_BROKEN", v)
    entries = read_ledger()
    sp = _split()
    sid = identify(scan, entries)
    testb, dev = set(sp["test_b"]["scans"]), set(sp["dev"]["scans"])
    last = latest_status(entries).get(sid)
    status = None if last is None else last["status"]
    detail = {"scan": scan, "scan_id": sid, "gold_status": status, "final_testb": final_testb}
    if sid in testb and not final_testb:
        raise GoldGuardRefusal("TEST_B_SEALED", detail)
    if cad_path and sid in OD3_NEVER_RUN:
        raise GoldGuardRefusal("OD3_DEVCHECK_NEVER_RUN", detail)
    if last is not None:
        if status != "FROZEN":
            raise GoldGuardRefusal("GOLD_NOT_FROZEN", detail)
        return {**detail, "allowed": True, "basis": "GOLD_LEDGER_FROZEN", "label_sha256": last.get("label_sha256")}
    if sid in testb:
        raise GoldGuardRefusal("GOLD_NOT_FROZEN", detail)
    if sid in dev:
        return {**detail, "allowed": True, "basis": "SPLIT_FROZEN_DEV_GOLD_OF_RECORD"}
    raise GoldGuardRefusal("NOT_REGISTERED", detail)


def append_event(scan_id: str, status: str, *, by: str, label_path: str | None = None,
                 label_sha256: str | None = None, note: str | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    v = verify_ledger()
    if not v["ok"]:
        raise ValueError(f"refusing to append to a broken gold ledger: {v['problems']}")
    entries = read_ledger()
    last = latest_status(entries).get(scan_id)
    if last is None:
        raise ValueError(f"{scan_id} is not registered")
    if last["status"] == "FROZEN":
        raise ValueError(f"{scan_id} is already FROZEN (append-only; a re-freeze needs an owner decision)")
    if status == "FROZEN":
        if not (label_path and label_sha256):
            raise ValueError("FROZEN needs --label-path and --label-sha256")
        root = "ai_bend_lines_zs_v1_SEALED" if last["portion"] == "TEST_B" else "ai_bend_lines_zs_v1"
        if Path(label_path).parent.name != root:
            raise ValueError(f"a {last['portion']} label must live directly under {root}/")
    keep = ("scan_id", "scan_sha256", "path", "format", "part_id", "family_group5", "portion", "batch_id", "gid",
            "split_sha256")
    e = {"schema": SCHEMA, "seq": len(entries) + 1, "event": "STATUS", "status": status,
         **{k: last[k] for k in keep}, "label_path": label_path, "label_sha256": label_sha256, "note": note,
         "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recorded_by": by,
         "previous_sha256": entries[-1]["entry_sha256"]}
    e["entry_sha256"] = _jhash(e)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research.zeroshot.common.gold_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--scan", required=True)
    c.add_argument("--final-testb", action="store_true", help="ONLY the integrator's single end-of-programme TEST-B run")
    c.add_argument("--cad-path", action="store_true", help="the engine uses CAD (refuses the OD-3 never-run cases)")
    sub.add_parser("verify")
    s = sub.add_parser("status")
    s.add_argument("--scan", default=None)
    a = sub.add_parser("append")
    a.add_argument("--scan", required=True)
    a.add_argument("--status", required=True, choices=STATUSES)
    a.add_argument("--label-path")
    a.add_argument("--label-sha256")
    a.add_argument("--note")
    a.add_argument("--by", default="gold-adjudicator")
    x = ap.parse_args(argv)
    if x.cmd == "check":
        try:
            out = require_runnable(x.scan, final_testb=x.final_testb, cad_path=x.cad_path)
        except GoldGuardRefusal as r:
            print(json.dumps({"allowed": False, "reason": r.reason, **r.detail}, sort_keys=True))
            return 3
        print(json.dumps(out, sort_keys=True))
        return 0
    if x.cmd == "verify":
        v = verify_ledger()
        print(json.dumps(v))
        return 0 if v["ok"] else 2
    if x.cmd == "status":
        st = latest_status(read_ledger())
        rows = {k: v["status"] for k, v in st.items() if x.scan in (None, k)}
        counts = {}
        for v in rows.values():
            counts[v] = counts.get(v, 0) + 1
        print(json.dumps({"counts": counts, "scans": rows} if x.scan else {"counts": counts}, indent=1))
        return 0
    e = append_event(identify(x.scan), x.status, by=x.by, label_path=x.label_path, label_sha256=x.label_sha256,
                     note=x.note)
    print(json.dumps({"seq": e["seq"], "scan_id": e["scan_id"], "status": e["status"], "entry_sha256": e["entry_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

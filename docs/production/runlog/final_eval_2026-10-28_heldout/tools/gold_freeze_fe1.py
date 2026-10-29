"""Final held-out evaluation 2026-10-28, batch FE1: FROZEN appends to zeroshot_2026-10-25/gold/GOLD_LEDGER.jsonl.

Mirrors docs/production/runlog/zeroshot_2026-10-25/tools/gold_ledger_newscans.py `freeze` (the N1 tool), for the
FE1 portion. Same hash-chained entry format and chain as research/zeroshot/common/gold_guard.py, which is left
byte-identical (it is inside both lane pins and its sha is bound in freeze seq 36); gold_guard.verify_ledger validates
the result and gold_guard.require_runnable admits a scan whose latest status is FROZEN.

freeze --scan sc_ --label-path P --label-sha256 H --by WHO [--note T]
    Appends one STATUS/FROZEN entry. Refuses unless:
      * the scan's latest ledger entry is REGISTERED (or MARKING/ADJUDICATING) with portion FINAL_EVAL_HELDOUT and
        batch FE1 (the 27 REGISTER entries of tools/gold_ledger_fe1.py, gold seqs 139-165);
      * the scan is not already FROZEN (append-only; a re-freeze needs an owner decision);
      * the label lives directly under data/bend_corpus/annotations/ai_bend_lines_final_eval_2026_10_28_v1/ and its
        sha256 on disk equals H;
      * the label's provenance sidecar records asset_sha256 == the registered scan sha256.
    The hidden canary is never registered, so it can never be frozen here.
verify
    Prints gold_guard.verify_ledger() and the FE1 status counts.
usage (repo root): .venv/bin/python -B docs/production/runlog/final_eval_2026-10-28_heldout/tools/gold_freeze_fe1.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).resolve().parents[1]
REPO = D.parents[3]
sys.path.insert(0, str(REPO))
from research.zeroshot.common import gold_guard as GG  # noqa: E402

PORTION = "FINAL_EVAL_HELDOUT"
BATCH = "FE1"
ROOT_NAME = "ai_bend_lines_final_eval_2026_10_28_v1"
ROOT = REPO / "data/bend_corpus/annotations" / ROOT_NAME
PRE_FROZEN = {"REGISTERED", "MARKING", "ADJUDICATING"}


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
        os.fsync(f.fileno())
    return e


def freeze(sid: str, label_path: str, label_sha256: str, by: str, note: str | None) -> dict:
    last = GG.latest_status(GG.read_ledger()).get(sid)
    if last is None:
        raise SystemExit(f"{sid} is not registered")
    if last.get("portion") != PORTION or last.get("batch_id") != BATCH:
        raise SystemExit(f"{sid}: portion {last.get('portion')} batch {last.get('batch_id')}; this tool is FE1 only")
    if last["status"] == "FROZEN":
        raise SystemExit(f"{sid} is already FROZEN (append-only; a re-freeze needs an owner decision)")
    if last["status"] not in PRE_FROZEN:
        raise SystemExit(f"{sid}: status {last['status']} cannot move to FROZEN")
    lp = Path(label_path).resolve()
    if lp.parent != ROOT.resolve():
        raise SystemExit(f"an FE1 label must live directly under {ROOT.relative_to(REPO)}/")
    if GG._sha256_file(lp) != label_sha256:
        raise SystemExit("label sha256 on disk differs from --label-sha256")
    prov = json.loads((ROOT / "provenance" / lp.name).read_text())
    if prov.get("asset_sha256") != last["scan_sha256"] or prov.get("coordinate_match") != "sha256":
        raise SystemExit(f"{sid}: provenance asset sha256 differs from the registered scan (or not a sha256 match)")
    keep = ("scan_id", "scan_sha256", "path", "format", "part_id", "family_group5", "portion", "batch_id", "gid",
            "split_sha256")
    return _append({"event": "STATUS", "status": "FROZEN", **{k: last[k] for k in keep},
                    "label_path": str(lp), "label_sha256": label_sha256, "note": note,
                    "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recorded_by": by})


def verify() -> dict:
    v = GG.verify_ledger()
    st = GG.latest_status(GG.read_ledger())
    fe1 = {k: e for k, e in st.items() if e.get("batch_id") == BATCH and e.get("portion") == PORTION}
    return {"ledger": v, "fe1_scans": len(fe1), "fe1_status": dict(Counter(e["status"] for e in fe1.values()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--scan", required=True)
    f.add_argument("--label-path", required=True)
    f.add_argument("--label-sha256", required=True)
    f.add_argument("--by", required=True)
    f.add_argument("--note")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        e = freeze(a.scan, a.label_path, a.label_sha256, a.by, a.note)
        print(json.dumps({"seq": e["seq"], "scan_id": e["scan_id"], "gid": e["gid"], "status": e["status"],
                          "entry_sha256": e["entry_sha256"]}))
    print(json.dumps(verify()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

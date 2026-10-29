"""Run-scoped sha256 allowlist in front of the (pinned, unchanged) custody guards. Final eval FE1, 2026-10-28.

Owner decision 2026-10-28, "Allowlist this run only": admit exactly the 27 scans of the frozen
``SET_MANIFEST.json`` plus their 28 FE1 packet copies (the hidden canary included), by sha256, tied to freeze-ledger
seq 35 (NOTE ``frz_3f205ab798e9``). Everything else stays blocked.

Why a separate module instead of an edit to the guards: ``research/bench_labeling/custody.py``,
``research/zeroshot/z3_slicing/custody.py`` and ``research/zeroshot/common/gold_guard.py`` are all inside the code
fingerprints of both production pins (zs_lane_v1 ``0087fb23...``, zs_lane_v3 rev3 ``711bd079...``), and gold_guard's
sha is bound in freeze seq 36. Editing any of them breaks PIN_MISMATCH and seq 36. So the guards stay byte-identical;
this layer authorizes by sha256 first and then hands the pinned code a byte copy staged at a guard-clean path, which
the unchanged guards admit by name (``stage``). Lanes never import this module.

Rules of :func:`check_path`:
  1. a path the name rules admit (``z3_slicing.custody.check_path``, the stricter of the two) is returned unchanged;
  2. a path they refuse is admitted ONLY IF the authorization loads and the file's sha256 is in its list;
  3. the authorization loads only if the freeze ledger verifies, seq 35 is the expected NOTE, the allowlist JSON is
     internally consistent (28 shas, seq 35 named, manifest sha = seq 35's), and a later NOTE references seq 35 and
     records exactly this allowlist file's sha256;
  4. every admission under rule 2 is appended to the admissions log (JSONL); every refusal raises CustodyError.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.bench_labeling.custody import CustodyError  # noqa: E402
from research.zeroshot.z3_slicing.custody import check_path as _name_check  # noqa: E402

RUN_DIR = REPO / "docs/production/runlog/final_eval_2026-10-28_heldout"
ALLOWLIST = RUN_DIR / "RUN_ALLOWLIST.json"
ADMISSIONS_LOG = RUN_DIR / "RUN_ALLOWLIST_ADMISSIONS.jsonl"
FREEZE_LEDGER = REPO / "docs/production/runlog/freeze/LEDGER.json"
AUTH_SEQ = 35
AUTH_ENTRY_ID = "frz_3f205ab798e9"
N_ITEMS = 28
SCHEMA = "sherman.final_eval.run_allowlist.v1"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _ledger(path: Path) -> list[dict]:
    from domains.production.engine.eval import freeze
    v = freeze.verify_ledger(path)
    if not v["ok"]:
        raise CustodyError(f"allowlist: freeze ledger does not verify: {v['problems']}")
    return freeze.read_ledger(path)


def load_authorization(allowlist: Path | str = ALLOWLIST, ledger: Path | str = FREEZE_LEDGER) -> dict:
    """The allowlisted sha256 set plus the ledger NOTE that binds it; raises CustodyError when anything mismatches."""
    allowlist, ledger = Path(allowlist), Path(ledger)
    entries = _ledger(ledger)
    auth = next((e for e in entries if e.get("seq") == AUTH_SEQ), None)
    if auth is None or auth.get("kind") != "NOTE" or auth.get("entry_id") != AUTH_ENTRY_ID:
        raise CustodyError(f"allowlist: ledger seq {AUTH_SEQ} is not the owner authorization {AUTH_ENTRY_ID}")
    if not allowlist.is_file():
        raise CustodyError(f"allowlist: {allowlist.name} missing")
    al_sha = sha256_file(allowlist)
    binds = [e for e in entries if e.get("kind") == "NOTE" and e["seq"] > AUTH_SEQ
             and (e.get("items") or {}).get("run_allowlist", {}).get("authorization_seq") == AUTH_SEQ]
    if not binds:
        raise CustodyError(f"allowlist: no ledger NOTE references seq {AUTH_SEQ} with a run_allowlist")
    note = binds[-1]
    if note["items"]["run_allowlist"].get("allowlist_sha256") != al_sha:
        raise CustodyError("allowlist: file sha256 differs from the ledger NOTE (tampered or unbound)")
    doc = json.loads(allowlist.read_text(encoding="utf-8"))
    shas = doc.get("sha256") or []
    if (doc.get("schema") != SCHEMA or doc.get("authorization", {}).get("seq") != AUTH_SEQ
            or doc["authorization"].get("entry_id") != AUTH_ENTRY_ID
            or doc.get("set_manifest", {}).get("sha256") != auth["items"]["set_manifest"]["sha256"]
            or len(shas) != N_ITEMS or len(set(shas)) != N_ITEMS):
        raise CustodyError("allowlist: content inconsistent with ledger seq 35 / expected 28 items")
    return {"sha256": frozenset(shas), "allowlist_sha256": al_sha, "note_seq": note["seq"],
            "note_entry_id": note["entry_id"]}


def _log(rec: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_path(path, *, allowlist=ALLOWLIST, ledger=FREEZE_LEDGER, log=ADMISSIONS_LOG, by: str = "") -> Path:
    """Name rules first; a refused path passes only by an authorized sha256 (logged). Otherwise CustodyError."""
    try:
        return _name_check(path)
    except CustodyError as refusal:
        name_reason = str(refusal)
    p = Path(path).expanduser().resolve()
    auth = load_authorization(allowlist, ledger)        # raises on a tampered / unbound allowlist
    if not p.is_file() or p.is_symlink():
        raise CustodyError(f"{name_reason} (allowlist: not a regular file)")
    sha = sha256_file(p)
    if sha not in auth["sha256"]:
        raise CustodyError(f"{name_reason} (allowlist: sha256 {sha[:12]} not authorized)")
    _log({"event": "ADMIT", "path": str(p), "sha256": sha, "name_rule": name_reason,
          "allowlist_sha256": auth["allowlist_sha256"], "ledger_note_seq": auth["note_seq"],
          "authorization_seq": AUTH_SEQ, "by": by,
          "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, Path(log))
    return p


def stage(path, out_dir, *, name: str | None = None, allowlist=ALLOWLIST, ledger=FREEZE_LEDGER,
          log=ADMISSIONS_LOG, by: str = "") -> Path:
    """Authorize ``path`` then place a byte copy (mode 0444) at a path the unchanged name rules admit."""
    src = check_path(path, allowlist=allowlist, ledger=ledger, log=log, by=by)
    want = sha256_file(src)
    dst = Path(out_dir).expanduser().resolve() / (name or src.name)
    _name_check(dst)                                     # the staged path itself must be name-clean
    if dst.exists():
        if sha256_file(dst) != want:
            raise CustodyError(f"staged copy {dst.name} exists with different bytes")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(src, tmp)
    if sha256_file(tmp) != want:
        tmp.unlink()
        raise CustodyError(f"staged copy of {src.name} differs from source")
    os.chmod(tmp, 0o444)
    st = src.stat()
    os.utime(tmp, (st.st_atime, st.st_mtime))
    os.replace(tmp, dst)
    _log({"event": "STAGE", "src": str(src), "dst": str(dst), "sha256": want, "by": by,
          "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, Path(log))
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research.zeroshot.common.run_allowlist")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--scan", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--scan", required=True)
    s.add_argument("--out", required=True)
    for p in (c, s):
        p.add_argument("--by", default="")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            print(json.dumps({"allowed": True, "path": str(check_path(a.scan, by=a.by))}))
        else:
            print(json.dumps({"staged": str(stage(a.scan, a.out, by=a.by))}))
    except CustodyError as e:
        print(json.dumps({"allowed": False, "reason": str(e)}))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

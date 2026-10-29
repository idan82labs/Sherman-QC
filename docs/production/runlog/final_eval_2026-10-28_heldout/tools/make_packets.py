"""Final held-out evaluation 2026-10-28, Stage 1 step 4: marker packets (N1 copy discipline, tools/gold_setup_n1.py).
Byte copies (mode 0444, uniform mtime, no quarantine xattr) under neutral random gids, shuffled, of the 27 scored scans
+ 1 hidden canary. Writes, refusing to overwrite:
  packets/batches.json   marker-facing: gid + copy path only
  packets/scans/gb_xxxxxx.ply
  PRIVATE_KEY.json       gid -> scan_id / original / role (markers never read)
No engine output, no CAD, no labels are read or written. usage: python3 -B tools/make_packets.py
"""
import hashlib, json, os, random, secrets, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).resolve().parents[1]
R = D.parents[3]
P = D / "packets"
ZS = R / "docs/production/runlog/zeroshot_2026-10-25"
UNIFORM_MTIME = 1793145600  # 2026-10-28T00:00:00Z
# Canary (N1 criteria): DEV, GOLD_SET_V2, label complete, full stage, PLY, DEV_CONSUMED, not a previous canary, not in
# the A1 tuning set, family group not used by earlier canaries nor by this set. 43895000_B: aug30 daily scan (same
# source folder as HOLDOUT10, so it blends in), 4 gold lines.
CANARY = "sc_589688be58d2"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    for p in (P / "batches.json", P / "scans", D / "PRIVATE_KEY.json"):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    man = json.loads((D / "SET_MANIFEST.json").read_text())
    inv = {r["scan_id"]: r for r in json.loads((ZS / "INVENTORY.json").read_text())["scans"]}
    can = json.loads((ZS / "gold/canaries.json").read_text())
    dev = set(json.loads((ZS / "SPLIT_FROZEN.json").read_text())["dev"]["scans"])
    a1 = (R / "docs/production/runlog/freeze/cases_A1.json").read_text()
    cr = inv[CANARY]; g = cr["gold"]
    assert CANARY in dev and g["set"].startswith("GOLD_SET_V2") and g["label_complete"] and cr["stage"] == "full"
    assert cr["format"] == "ply" and cr["custody"] == "DEV_CONSUMED" and CANARY not in a1
    assert CANARY not in {c["scan_id"] for c in can["canaries"]}
    assert cr["family_group5"] not in {inv[c["scan_id"]]["family_group5"] for c in can["canaries"]}
    assert cr["family_group5"] not in set(man["counts"]["families"])
    csrc = Path(os.path.expanduser(cr["path"])).resolve()
    assert sha(csrc)[:12] == CANARY[3:]

    items = [(r["scan_id"], Path(r["copies"][0]["path"]), r["sha256"], "SCORED", r["subset"]) for r in man["scored"]]
    items.append((CANARY, csrc, None, "CANARY", "CANARY"))
    used = {it["gid"] for b in json.loads((ZS / "gold/batches.json").read_text())["batches"] for it in b["items"]}
    seed = secrets.randbits(64)
    random.Random(seed).shuffle(items)
    (P / "scans").mkdir(parents=True)
    out, key = [], {}
    for sid, src, want, role, subset in items:
        while True:
            gid = "gb_" + secrets.token_hex(3)
            if gid not in used:
                used.add(gid); break
        dst = P / "scans" / f"{gid}.ply"
        shutil.copyfile(src, dst)
        got = sha(dst)
        assert want is None or got == want, (sid, got, want)
        os.chmod(dst, 0o444)
        os.utime(dst, (UNIFORM_MTIME, UNIFORM_MTIME))
        subprocess.run(["xattr", "-c", str(dst)], capture_output=True)
        out.append({"gid": gid, "path": str(dst)})
        key[gid] = {"scan_id": sid, "role": role, "subset": subset, "original_path": str(src), "sha256": got}
        if role == "CANARY":
            key[gid].update(gold_set=g["set"], gold_path=g["path"], gold_n_lines=g["n_lines"], part_id=cr["part_id"])
        print(gid, role, flush=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (P / "batches.json").write_text(json.dumps({
        "schema": "sherman.final_eval.marker_batches.v1", "created": now, "batch": "FE1",
        "note": "Marker-facing. Each item is a neutral id and a scan copy. Mark every item the same way, in "
                "packets/FE1/<gid>/marker_<A|B>/. Instructions: packets/MARKER_INSTRUCTIONS.md. Read nothing else.",
        "items": sorted(out, key=lambda x: x["gid"])}, indent=1) + "\n")
    (D / "PRIVATE_KEY.json").write_text(json.dumps({
        "schema": "sherman.final_eval.private_key.v1", "created": now,
        "PRIVATE": "markers must never read this file (gid -> scan mapping, canary identity)",
        "rng_seed": seed, "n_items": len(key), "key": key}, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

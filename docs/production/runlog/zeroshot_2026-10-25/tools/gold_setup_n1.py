"""Gold batch N1 setup (2026-10-25): 4 NEW_UNSEEN_FAMILY_TEST scans (Shaked email 2026-10-19) + 1 hidden canary.
Same copy discipline as tools/gold_setup.py (byte copies, mode 0444, uniform mtime, no quarantine xattr, neutral
random gids, shuffled order). Reads INVENTORY.json only for the canary row (no engine output). Writes, refusing to
overwrite:
  newscans_N1/REGISTRY.json     registration of the 4 new scans (not DEV, not TEST-B; never tune on them)
  newscans_N1/PRIVATE_KEY.json  PRIVATE gid -> original mapping for the 4 new scans (markers must never read)
  gold/N1/scans/gb_xxxxxx.ply   5 byte copies
and appends batch N1 to gold/batches.json and the canary to gold/canaries.json (after .bak_N1 backups).
Does not touch GOLD_LEDGER.jsonl or any label. usage: python3 tools/gold_setup_n1.py"""
import hashlib, json, os, random, secrets, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parents[1]
G = H / "gold"
NB = H / "newscans_N1"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
UNIFORM_MTIME = 1792886400  # 2026-10-25T00:00:00Z
SRC = Path.home() / "ShermanQC/source-assets/2026-10-19_shaked_new"
NEW = [("47085000", "47085000_A.ply", "part with short tabs"), ("51101001", "51101001_01.ply", "rounded part"),
       ("45351000", "45351000_B.ply", "rounded part"), ("41018000", "41018000_H.ply", "rounded part")]
# DEV, GOLD_SET_V2 (A audited), complete, full stage, PLY, not previously a canary, family group not used by earlier
# canaries, not HOLDOUT10/N0xx/EHO4/Eyal50/SEALED/~/Downloads/daily-scans dir, not DEV-CHECK/FRESH/TEST-B,
# not A1 tuning set. 49125000_A00: rolled shell + joggle + corner tab, 2 gold lines.
CANARY = "sc_fee685eee4db"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ply_header(p):
    with open(p, "rb") as f:
        raw = f.read(4096)
    hdr = raw[:raw.index(b"end_header")].decode("ascii", "replace")
    out = {}
    for ln in hdr.splitlines():
        t = ln.strip().split()
        if t[:1] == ["format"]:
            out["format"] = " ".join(t[1:])
        if t[:1] == ["element"] and len(t) == 3:
            out[t[1]] = int(t[2])
    return out


def main():
    for p in (NB / "REGISTRY.json", NB / "PRIVATE_KEY.json", G / "N1", G / "batches.json.bak_N1",
              G / "canaries.json.bak_N1"):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    batches = json.loads((G / "batches.json").read_text())
    can = json.loads((G / "canaries.json").read_text())
    if any(b["id"] == "N1" for b in batches["batches"]):
        sys.exit("batch N1 already in batches.json")
    inv = {r["scan_id"]: r for r in json.loads((H / "INVENTORY.json").read_text())["scans"]}
    dev = set(json.loads((H / "SPLIT_FROZEN.json").read_text())["dev"]["scans"])
    cr = inv[CANARY]
    g = cr["gold"]
    assert CANARY in dev and g["set"].startswith("GOLD_SET_V2") and g["label_complete"] and cr["stage"] == "full"
    assert cr["format"] == "ply" and cr["custody"] == "DEV_CONSUMED"
    assert CANARY not in {c["scan_id"] for c in can["canaries"]}
    assert cr["family_group5"] not in {inv[c["scan_id"]]["family_group5"] for c in can["canaries"]}
    csrc = Path(os.path.expanduser(cr["path"])).resolve()
    assert sha256_file(csrc)[:12] == CANARY[3:]
    gold_n = len(json.loads((H.parents[3] / g["path"]).read_text())["bends"])
    assert gold_n == g["n_lines"]

    used = {it["gid"] for b in batches["batches"] for it in b["items"]}
    seed = secrets.randbits(64)
    rng = random.Random(seed)
    items = [(pid, SRC / fn, note, "NEW") for pid, fn, note in NEW] + [(cr["part_id"], csrc, None, "CANARY")]
    rng.shuffle(items)
    bdir = G / "N1" / "scans"
    bdir.mkdir(parents=True)
    NB.mkdir(exist_ok=True)
    out_items, reg, key, canary_entry = [], [], {}, None
    for pid, src, note, role in items:
        while True:
            gid = "gb_" + secrets.token_hex(3)
            if gid not in used:
                used.add(gid); break
        sha = sha256_file(src)
        dst = bdir / f"{gid}.ply"
        shutil.copyfile(src, dst)
        assert sha256_file(dst) == sha
        os.chmod(dst, 0o444)
        os.utime(dst, (UNIFORM_MTIME, UNIFORM_MTIME))
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(dst)], capture_output=True)
        out_items.append({"gid": gid, "path": str(dst)})
        if role == "CANARY":
            canary_entry = {"batch": "N1", "gid": gid, "scan_id": CANARY, "original_path": str(csrc),
                            "gold_set": g["set"], "gold_path": g["path"], "gold_n_lines": gold_n}
            can["key"][gid] = {"batch": "N1", "scan_id": CANARY, "role": "CANARY", "original_path": str(csrc),
                               "format": "ply", "portion": "DEV_CANARY"}
            continue
        hd = ply_header(src)
        reg.append({"part_id": pid, "original_path": str(src), "sha256": sha, "size_bytes": src.stat().st_size,
                    "ply_format": hd.get("format"), "vertex_count": hd.get("vertex"), "face_count": hd.get("face"),
                    "customer_note": note, "source": "Shaked email 2026-10-19 SharePoint link",
                    "status": "NEW_UNSEEN_FAMILY_TEST — not DEV, never tuned on", "gold_batch": "N1"})
        key[gid] = {"part_id": pid, "original_path": str(src), "sha256": sha, "customer_note": note}

    shutil.copy2(G / "batches.json", G / "batches.json.bak_N1")
    shutil.copy2(G / "canaries.json", G / "canaries.json.bak_N1")
    batches["batches"].append({"id": "N1", "items": out_items})
    (G / "batches.json").write_text(json.dumps(batches, indent=1) + "\n")
    can["canaries"].append(canary_entry)
    can.setdefault("rng_seeds_later_batches", {})["N1"] = seed
    (G / "canaries.json").write_text(json.dumps(can, indent=1, sort_keys=True) + "\n")

    reg.sort(key=lambda r: r["part_id"])
    (NB / "REGISTRY.json").write_text(json.dumps({
        "schema": "sherman.zeroshot.newscans_registry.v1", "created": NOW, "batch": "N1",
        "rule": "NEW_UNSEEN_FAMILY_TEST: 4 never-seen part families received 2026-10-19. NOT part of DEV or TEST-B. "
                "Nothing may be tuned, calibrated or selected on these scans. Gold is marked blind in gold/N1/.",
        "family_groups5_absent_from_inventory": sorted(r["part_id"][:5] for r in reg),
        "scans": reg}, indent=1, ensure_ascii=False) + "\n")
    (NB / "PRIVATE_KEY.json").write_text(json.dumps({
        "schema": "sherman.zeroshot.newscans_private_key.v1", "created": NOW,
        "PRIVATE": "markers must never read this file (gid -> original scan mapping for gold batch N1)",
        "rng_seed": seed, "note": "the N1 canary is recorded only in gold/canaries.json", "key": key},
        indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"items": out_items, "canary": canary_entry}, indent=1))


if __name__ == "__main__":
    main()

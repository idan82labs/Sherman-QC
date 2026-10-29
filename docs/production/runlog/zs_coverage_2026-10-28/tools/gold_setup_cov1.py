"""Gold batch COV1 setup (2026-10-28): the permitted real scans that lack frozen bend-line gold (from
zs_coverage_2026-10-28/COV_INVENTORY.json) + 1 hidden DEV canary, packaged the N1 way (byte copies, mode 0444,
uniform mtime, no quarantine xattr, neutral random gids, shuffled order). Self-contained in gold_batch_COV1/:
does NOT touch gold/batches.json, gold/canaries.json, GOLD_LEDGER.jsonl or any label. Reads zeroshot gold/canaries.json
only to avoid re-using an earlier canary (scan or family group). Refuses to overwrite.
  gold_batch_COV1/scans/gb_xxxxxx.<ext>   copies (marker-facing)
  gold_batch_COV1/batch.json              marker-facing item list
  gold_batch_COV1/PRIVATE_KEY.json        gid -> original + canary (markers and the engine side never read)
"""
import hashlib, json, os, random, secrets, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
RL = REPO / "docs/production/runlog"
ZS = RL / "zeroshot_2026-10-25"
B = RL / "gold_batch_COV1"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
UNIFORM_MTIME = 1792886400


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    if (B / "scans").exists() or (B / "PRIVATE_KEY.json").exists():
        sys.exit("refusing to overwrite gold_batch_COV1")
    cov = json.loads((RL / "zs_coverage_2026-10-28/COV_INVENTORY.json").read_text())["rows"]
    need = [r for r in cov if r["usable"] and r["physical"] and r["split"] in ("DEV", "TEST-B", "N1", "OTHER_CONSUMED")
            and not (r.get("gold") or {}).get("frozen")]
    inv = {r["scan_id"]: r for r in json.loads((ZS / "INVENTORY.json").read_text())["scans"]}
    dev = set(json.loads((ZS / "SPLIT_FROZEN.json").read_text())["dev"]["scans"])
    can = json.loads((ZS / "gold/canaries.json").read_text())
    used_ids = {c["scan_id"] for c in can["canaries"]}
    used_g5 = {inv[c["scan_id"]]["family_group5"] for c in can["canaries"] if c["scan_id"] in inv}
    need_g5 = {r.get("family_group5") for r in need}
    pool = sorted(s for s in dev if s not in used_ids and inv[s]["custody"] == "DEV_CONSUMED" and inv[s]["format"] == "ply"
                  and inv[s]["stage"] == "full" and str(inv[s]["gold"].get("set", "")).startswith("GOLD_SET_V2")
                  and inv[s]["gold"].get("label_complete") and (inv[s]["gold"].get("n_lines") or 0) >= 2
                  and inv[s]["family_group5"] not in used_g5 | need_g5
                  and "Downloads" not in inv[s]["path"] and "SEALED" not in inv[s]["path"])
    seed = secrets.randbits(64)
    rng = random.Random(seed)
    canary = rng.choice(pool)
    cr = inv[canary]
    items = [(r, "NEW") for r in need] + [({"scan_id": canary, "part_id": cr["part_id"], "path": cr["path"],
                                            "sha256": None}, "CANARY")]
    rng.shuffle(items)
    (B / "scans").mkdir(parents=True)
    used = set()
    out, key = [], {}
    for r, role in items:
        while True:
            gid = "gb_" + secrets.token_hex(3)
            if gid not in used:
                used.add(gid); break
        src = Path(os.path.expanduser(r["path"])).resolve()
        sha = sha256_file(src)
        if role == "NEW":
            assert sha == r["sha256"]
        else:
            assert sha[:12] == canary[3:]
        dst = B / "scans" / f"{gid}{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        assert sha256_file(dst) == sha
        os.chmod(dst, 0o444)
        os.utime(dst, (UNIFORM_MTIME, UNIFORM_MTIME))
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(dst)], capture_output=True)
        out.append({"gid": gid, "path": str(dst)})
        key[gid] = {"role": role, "scan_id": r["scan_id"], "part_id": r["part_id"], "sha256": sha,
                    "original_path": str(src), "split": r.get("split", "DEV (canary)"), "sub": r.get("sub")}
        if role == "CANARY":
            key[gid].update(gold_set=cr["gold"]["set"], gold_path=cr["gold"]["path"], gold_n_lines=cr["gold"]["n_lines"])
    (B / "batch.json").write_text(json.dumps({"schema": "sherman.zs_coverage.gold_batch.v1", "id": "COV1", "created": NOW,
        "rule": "every item gets the same full blind two-marker marking (MARKER_INSTRUCTIONS.md)", "items": out}, indent=1) + "\n")
    (B / "PRIVATE_KEY.json").write_text(json.dumps({"schema": "sherman.zs_coverage.gold_batch_private_key.v1",
        "created": NOW, "PRIVATE": "markers must never read this file", "rng_seed": seed,
        "canary_pool_size": len(pool), "n_new": len(need), "key": key}, indent=1) + "\n")
    print(json.dumps({"n_new": len(need), "n_items": len(out), "canary_pool": len(pool)}))


if __name__ == "__main__":
    main()

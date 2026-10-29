"""Gold-protocol setup (PLAN.md section 2), run once on 2026-10-25. Reads INVENTORY.json + SPLIT_FROZEN.json only
(no labels, no engine output). Writes, refusing to overwrite:
  gold/GOLD_LEDGER.jsonl   pre-registration (status REGISTERED) of the 4 DEV scans needing gold + the 61 TEST-B scans
  dev_folds.json           5-fold assignment over DEV, grouped by 5-digit family group (label-free, deterministic)
  gold/batches.json        MARKER-FACING: per batch a shuffled list of neutral ids (gb_xxxxxx) + scan paths
  gold/canaries.json       PRIVATE (markers must never read it): the gb_id -> scan key and the canaries
  gold/<batch>/scans/      gb_xxxxxx.<ext>: byte-identical COPIES of the original scan (same sha256), all with mode
                           0444, mtime 2026-10-25T00:00Z and no quarantine xattr, so a canary cannot be told apart by
                           path, mtime, mode or xattrs. (The first run on 2026-10-25 made hard links; they leaked the
                           source file's mtime/mode, so they were replaced in place by copies, same gids, same bytes.)
usage: python3 tools/gold_setup.py"""
import hashlib, json, os, random, secrets, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parents[1]
G = H / "gold"
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
BY = "zeroshot_2026-10-25/gold-setup"
UNIFORM_MTIME = 1792886400  # 2026-10-25T00:00:00Z

DEV_NEED = ["sc_186463746daa", "sc_4d5dc2b40b67", "sc_0178c0155934", "sc_766be04dc456"]
# GOLD_SET_V2 (audited) DEV scans, full stage, complete, 3-7 lines, distinct families; none from the
# 2026-08-30_shaked_daily_scans dir (HOLDOUT10 lives there), ~/Downloads, incoming_email, 38173000 or DEV-CHECK.
CANARIES = {"D0": "sc_c27168a2de83", "T1": "sc_3c93a0bf6f97", "T2": "sc_8b72c3ff233f", "T3": "sc_c1d112196ae5",
            "T4": "sc_d1732de18023", "T5": "sc_2c807204c472", "T6": "sc_891a8e4a2ee1"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jhash(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    for p in (G / "GOLD_LEDGER.jsonl", G / "batches.json", G / "canaries.json", H / "dev_folds.json"):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    inv = {r["scan_id"]: r for r in json.loads((H / "INVENTORY.json").read_text())["scans"]}
    split_bytes = (H / "SPLIT_FROZEN.json").read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    sp = json.loads(split_bytes)
    dev, testb = sp["dev"]["scans"], sp["test_b"]["scans"]
    assert len(dev) == 121 and len(testb) == 61
    for b, s in CANARIES.items():
        g = inv[s]["gold"] or {}
        assert s in dev and g["set"].startswith("GOLD_SET_V2") and g.get("label_complete"), s
    assert len(set(inv[s]["family_group5"] for s in CANARIES.values())) == len(CANARIES)

    seed = secrets.randbits(64)
    rng = random.Random(seed)

    # ---- dev folds (label-free: group sizes and a hash tie-break only) ----
    groups = {}
    for s in dev:
        groups.setdefault(inv[s]["family_group5"], []).append(s)
    order = sorted(groups, key=lambda g: (-len(groups[g]), hashlib.sha256(f"zs-folds-v1:{g}".encode()).hexdigest()))
    fold_of_group, sizes = {}, [0] * 5
    for g in order:
        k = min(range(5), key=lambda i: (sizes[i], i))
        fold_of_group[g] = k
        sizes[k] += len(groups[g])
    pending = {}
    for s in DEV_NEED:
        g = inv[s]["family_group5"]
        if g not in fold_of_group:
            k = min(range(5), key=lambda i: (sizes[i], i))
            fold_of_group[g] = k
            sizes[k] += 1
        pending[s] = {"family_group5": g, "fold": fold_of_group[g]}
    folds = {
        "schema": "sherman.zeroshot.dev_folds.v1", "frozen_at": "2026-10-25", "split_sha256": split_sha,
        "rule": "5-fold grouped CV over SPLIT_FROZEN dev, grouped by family_group5 (INVENTORY.json). Groups sorted by "
                "size desc, tie-break sha256('zs-folds-v1:'+group); each assigned to the fold with fewest scans (lowest "
                "index on ties). No labels or engine output used. pending_gold: the 4 DEV scans without gold get the fold "
                "of their family group (new group -> lightest fold); they enter CV only once their gold is FROZEN.",
        "n_folds": 5,
        "folds": [{"fold": k, "n_scans": sum(len(groups[g]) for g in groups if fold_of_group[g] == k),
                   "family_groups": sorted(g for g in groups if fold_of_group[g] == k),
                   "scans": sorted(s for s in dev if fold_of_group[inv[s]["family_group5"]] == k)} for k in range(5)],
        "scan_fold": {s: fold_of_group[inv[s]["family_group5"]] for s in sorted(dev)},
        "pending_gold": pending,
    }
    (H / "dev_folds.json").write_text(json.dumps(folds, indent=1, sort_keys=True) + "\n")

    # ---- batches ----
    tb = list(testb)
    rng.shuffle(tb)
    cuts, chunks, i = [10, 10, 10, 10, 10, 11], [], 0
    for c in cuts:
        chunks.append(tb[i:i + c]); i += c
    plan = {"D0": list(DEV_NEED)}
    for k, ch in enumerate(chunks, 1):
        plan[f"T{k}"] = ch
    used, batches, key = set(), [], {}
    for b, scans in plan.items():
        items = [(s, "TARGET") for s in scans] + [(CANARIES[b], "CANARY")]
        rng.shuffle(items)
        bdir = G / b / "scans"
        bdir.mkdir(parents=True, exist_ok=True)
        out_items = []
        for s, role in items:
            while True:
                gid = "gb_" + secrets.token_hex(3)
                if gid not in used:
                    used.add(gid); break
            src = Path(os.path.expanduser(inv[s]["path"])).resolve()
            ext = src.suffix.lower()
            link = bdir / f"{gid}{ext}"
            shutil.copyfile(src, link)
            assert sha256_file(link) == sha256_file(src)
            os.chmod(link, 0o444)
            os.utime(link, (UNIFORM_MTIME, UNIFORM_MTIME))
            subprocess.run(["xattr", "-d", "com.apple.quarantine", str(link)], capture_output=True)
            out_items.append({"gid": gid, "path": str(link)})
            key[gid] = {"batch": b, "scan_id": s, "role": role, "original_path": str(src), "format": ext[1:],
                        "portion": "DEV_CANARY" if role == "CANARY" else ("DEV_PENDING_GOLD" if b == "D0" else "TEST_B")}
        batches.append({"id": b, "items": out_items})
    (G / "batches.json").write_text(json.dumps({
        "schema": "sherman.zeroshot.gold_batches.v1", "created": NOW,
        "note": "Marker-facing. Each item is a neutral id and a scan path (a copy of the scan). Mark every item the same way. "
                "Work in gold/<batch>/<gid>/marker_<A|B>/. Instructions: gold/MARKER_HOWTO.md. Do not read "
                "gold/canaries.json, gold/GOLD_LEDGER.jsonl, INVENTORY.json, SPLIT_FROZEN.json or any label/engine output.",
        "batches": batches}, indent=1) + "\n")

    # ---- private key / canaries ----
    (G / "canaries.json").write_text(json.dumps({
        "schema": "sherman.zeroshot.gold_canaries.v1", "created": NOW, "PRIVATE": "markers must never read this file",
        "rng_seed": seed,
        "canaries": [{"batch": b, "gid": g, "scan_id": v["scan_id"], "original_path": v["original_path"],
                      "gold_set": inv[v["scan_id"]]["gold"]["set"], "gold_path": inv[v["scan_id"]]["gold"]["path"]}
                     for g, v in key.items() if v["role"] == "CANARY" for b in [v["batch"]]],
        "qc_rule": "marker-vs-canary line-F1 (frozen matcher) >= 0.95, else the batch is redone (PLAN 2.6)",
        "key": key}, indent=1, sort_keys=True) + "\n")

    # ---- gold ledger (append-only, hash-chained) ----
    prev, seq, lines = None, 0, []
    for g, v in key.items():
        if v["role"] == "CANARY":
            continue
        s = v["scan_id"]
        sha = sha256_file(v["original_path"])
        seq += 1
        e = {"schema": "sherman.zeroshot.gold_ledger.v1", "seq": seq, "event": "REGISTER", "status": "REGISTERED",
             "scan_id": s, "scan_sha256": sha, "scan_sha256_prefix_is_id": sha[:12] == s[3:],
             "path": v["original_path"], "format": v["format"], "part_id": inv[s]["part_id"],
             "family_group5": inv[s]["family_group5"], "portion": v["portion"], "batch_id": v["batch"], "gid": g,
             "split_sha256": split_sha, "recorded_at": NOW, "recorded_by": BY, "previous_sha256": prev}
        e["entry_sha256"] = jhash(e)
        prev = e["entry_sha256"]
        lines.append(json.dumps(e, sort_keys=True, ensure_ascii=False))
    (G / "GOLD_LEDGER.jsonl").write_text("\n".join(lines) + "\n")
    print(json.dumps({"ledger_entries": seq, "fold_sizes": sizes, "batches": {b["id"]: len(b["items"]) for b in batches},
                      "dev_folds_sha256": sha256_file(H / "dev_folds.json"),
                      "batches_sha256": sha256_file(G / "batches.json")}, indent=1))


if __name__ == "__main__":
    main()

"""Coverage inventory (2026-10-28), metadata + existing indexes only. No geometry or label content is opened.

Sources: zeroshot_2026-10-25/INVENTORY.json + SPLIT_FROZEN.json + gold/GOLD_LEDGER.jsonl (status/paths only),
newscans_N1/REGISTRY.json, benchmark_v3/inventory.json (not_queued reasons), the FE1 SET_MANIFEST (sha list only),
freeze-ledger seq 34 (30820000_A01 consumed) and the census (tools/census.py; sha256 of bytes only).
Writes COV_INVENTORY.json (one row per unique sha256) and RUN_MANIFEST.json (permitted real scans to run).
"""
import json, os, sys, collections
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
RL = REPO / "docs/production/runlog"
P = RL / "zeroshot_2026-10-25"
OUT = RL / "zs_coverage_2026-10-28"
census = json.loads(Path(sys.argv[1]).read_text())
inv = json.loads((P / "INVENTORY.json").read_text())
split = json.loads((P / "SPLIT_FROZEN.json").read_text())
dev, tb = set(split["dev"]["scans"]), set(split["test_b"]["scans"])
led = {}
for x in (P / "gold/GOLD_LEDGER.jsonl").read_text().splitlines():
    if x.strip():
        e = json.loads(x); led[e["scan_id"]] = e
n1 = json.loads((P / "newscans_N1/REGISTRY.json").read_text())["scans"]
bv3 = json.loads((RL / "benchmark_v3/inventory.json").read_text())
bv3_reason = {q["scan_id"]: q["reason"] for q in bv3["label_queue"]["not_queued"]}
bv3_part = {f["sha256"]: f for f in bv3["files"]}
man = json.loads((RL / "final_eval_2026-10-28_heldout/SET_MANIFEST.json").read_text())
fe1 = {m["sha256"]: m for m in man["scored"]}
paths = collections.defaultdict(list)
for r in census["rows"]:
    paths[r["sha256"]].append(r["path"])

rows = {}
def put(sha, **kw):
    r = rows.setdefault(sha, {"sha256": sha, "scan_id": "sc_" + sha[:12] if sha else None})
    r.update(kw)

def root_of(p):
    if not p: return None
    s = str(p)
    return s.split("annotations/")[1].split("/")[0] if "annotations/" in s else s.rsplit("/", 1)[0]

inv_rows = {r["scan_id"]: r for r in inv["scans"]}
sha_of = {}
for sha in paths:
    sha_of["sc_" + sha[:12]] = sha
for r in inv["scans"] + inv["outside_benchmark"]:
    sid = r["scan_id"]
    sha = sha_of.get(sid)
    g = r.get("gold") or {}
    cust = r["custody"]
    usable = bool(r.get("usable"))
    if sid in tb:
        sp, sub = "TEST-B", "CONSUMED_FRESH (TEST-B, scored once 2026-10-25)"
    elif sid in dev:
        sp = "DEV"; sub = {"DEV_CONSUMED": "DEV (A1 tuning + other consumed)", "CONSUMED_DEVCHECK": "DEV-CHECK",
                           "CONSUMED_FRESH": "FRESH (in DEV)"}[cust]
    elif sid in led and led[sid]["portion"] == "DEV_PENDING_GOLD":
        sp = "DEV"; sub = {"DEV_CONSUMED": "DEV (A1 tuning + other consumed)", "CONSUMED_FRESH": "FRESH (in DEV)"}.get(cust, cust) + "; gold frozen 2026-10-25 (D0)"
    elif cust.startswith("QUARANTINE_EYAL50") or cust.startswith("FORBIDDEN"):
        sp, sub = "EXCLUDED", cust
    else:
        sp, sub = "EXCLUDED", r.get("why_not_usable") or cust
    wn = r.get("why_not_usable") or ""
    synthetic = wn.startswith("EXCLUDED_SYNTHETIC") or "LIDAR_DEMO" in str(r.get("path"))
    physical = not synthetic and not wn.startswith("POSSIBLE_CAD_REFERENCE")
    gold = None
    if sid in led and led[sid]["status"] == "FROZEN":
        gold = {"frozen": True, "root": root_of(led[sid]["label_path"]), "basis": "GOLD_LEDGER FROZEN (" + led[sid]["batch_id"] + ")",
                "secondary_root": root_of(g.get("path"))}
    elif r.get("has_gold") and g.get("label_complete", True) and g.get("path"):
        gold = {"frozen": True, "root": root_of(g.get("path")), "basis": "INVENTORY gold of record (SPLIT_FROZEN d25ffd4f), quality " + str(g.get("quality", ""))[:2],
                "n_lines": g.get("n_lines"), "no_bends_present": g.get("no_bends_present")}
    rec = dict(scan_id=sid, part_id=r["part_id"], family_group5=r.get("family_group5"), format=r.get("format"),
               path=r["path"], on_disk=sha is not None, split=sp, sub=sub, usable=usable and sp in ("DEV", "TEST-B"),
               physical=physical, synthetic=synthetic, has_step=bool((r.get("cad") or {}).get("has_step")),
               gold=gold, stage=r.get("stage"))
    if sha is None:
        rec["sha256_note"] = "not hashed this census (pruned by path or absent); id from INVENTORY"
        rows["id:" + sid] = rec
    else:
        put(sha, **rec)

for s in n1:
    sha = s["sha256"]; sid = "sc_" + sha[:12]
    e = led.get(sid)
    put(sha, part_id=s["part_id"], family_group5=s["part_id"][:5], format="ply", path=s["original_path"],
        on_disk=sha in paths, split="N1", sub="NEW_UNSEEN_FAMILY_TEST (one-time run 2026-10-25, consumed)",
        usable=True, physical=True, synthetic=False, has_step=False, stage="full",
        gold={"frozen": True, "root": root_of(e["label_path"]), "basis": "GOLD_LEDGER FROZEN (N1)"} if e and e["status"] == "FROZEN" else None)

for m in man["listed_not_scored_consumed"]:
    sha = m["copies"][0]["sha256"]
    dl = [p for p in paths.get(sha, []) if "/Downloads/" in p]
    put(sha, part_id=m["part_id"], family_group5=m["part_id"][:5], format="ply", path=sorted(dl, key=len)[-1] if dl else None,
        on_disk=bool(dl), split="OTHER_CONSUMED",
        sub="FE1-listed, not FE1-scored (" + m["source_name"] + ")" + ("; consumed by owner blind demo, freeze seq 34" if sha.startswith("f81ebf01") else "; consumed by FE1 rule (pair of a consumed scan)"),
        usable=True, physical=True, synthetic=False, has_step=False, stage=None, gold=None,
        note="sealed evaluator manifest exists in its source folder; not opened")

for sha, m in fe1.items():
    put(sha, part_id=m["part_id"], family_group5=m["family_group5"], format="ply", path=None, on_disk=True,
        split="FE1", sub=m["subset"] + " (final held-out; NOT run here)", usable=False, physical=True, synthetic=False,
        has_step=False, gold={"frozen": False, "root": None, "basis": "FE1 gold pending (REGISTERED only)"},
        face_count=m.get("face_count"))

FACE = "Documents/Codex/2026-10-02/open-this/outputs"
for sha, ps in paths.items():
    if sha in rows: continue
    p0 = ps[0]
    if FACE in p0:
        continue  # face meshes (rhino-preview-like biometric): hash discarded, not reported by sha
    f = bv3_part.get(sha)
    if f is not None and f.get("kind") == "scan":
        reason = bv3_reason.get("sc_" + sha[:12], "")
        put(sha, part_id=f.get("part_id"), format=f.get("ext"), path=p0, on_disk=True, split="EXCLUDED",
            sub="FORMAT_ALTERNATE_OF_PLY (same capture as an inventoried PLY; BV3 reason: " + (reason or "n/a") + ")",
            usable=False, physical=True, synthetic=False, has_step=None, gold=None)
    elif "/cache/" in p0 or "derived_zs_v1" in p0:
        put(sha, path=p0, on_disk=True, split="EXCLUDED", sub="DERIVED_CACHE (re-export/aligned copy of a scan)",
            usable=False, physical=True, synthetic=False, gold=None)
    elif "cad_mesh" in p0 or "reference" in p0 or p0.endswith("/cad.ply") or any("reference" in q or "_demo.stl" in q for q in ps):
        put(sha, path=p0, on_disk=True, split="EXCLUDED", sub="CAD_REFERENCE_MESH (not a scan)", usable=False,
            physical=False, synthetic=False, gold=None)
    elif "1790929922" in p0:
        put(sha, path=p0, on_disk=True, split="EXCLUDED", sub="UNVERIFIED_PHONE_CAPTURE (no SKU; not opened)", usable=False,
            physical=None, synthetic=False, gold=None)
    else:
        put(sha, path=p0, on_disk=True, split="EXCLUDED", sub="UNCLASSIFIED", usable=False, physical=None, gold=None)

for r in rows.values():
    if r.get("sha256") and r["sha256"] in paths:
        r["n_copies"] = len(paths[r["sha256"]])

doc = {"schema": "sherman.zs_coverage.inventory.v1", "created": "2026-10-28",
       "builder": "docs/production/runlog/zs_coverage_2026-10-28/tools/build_cov_inventory.py",
       "custody": ["pruned before hashing: *_SEALED*, *eyal*, rhino*, 41880000_D, HOLDOUT10 source dir, FE1 stems/paths, final_eval/fe1_stage dirs, shaked_new_good_bad_pair(s) dirs, f1-sep02 email holdout",
                   "FE1 rows come from SET_MANIFEST sha list only; FE1 files not hashed, opened or run",
                   "incident: 4 Eyal50-quarantine scans (paths without 'eyal') and 6 face meshes in ~/Documents/Codex/.../outputs were sha256-hashed (bytes only, no geometry) before classification; face hashes discarded, quarantine rows reported by INVENTORY id only"],
       "rows": sorted(rows.values(), key=lambda r: (r["split"], r.get("part_id") or "", r.get("scan_id") or ""))}
for r in doc["rows"]:
    if str(r.get("sub", "")).startswith("QUARANTINE_EYAL50"):
        r.pop("sha256", None); r["n_copies"] = None
(OUT / "COV_INVENTORY.json").write_text(json.dumps(doc, indent=1))
run = [r for r in doc["rows"] if r["usable"] and r["split"] in ("DEV", "TEST-B", "N1", "OTHER_CONSUMED") and r["physical"]]
for r in run:
    r["run_path"] = os.path.expanduser(r["path"])
    assert os.path.isfile(r["run_path"]), r
fe1set = set(fe1)
assert not any(r["sha256"] in fe1set for r in run)
rm = {"schema": "sherman.zs_coverage.run_manifest.v1", "created": "2026-10-28", "n": len(run),
      "scans": [{"scan_id": r["scan_id"], "sha256": r["sha256"], "part_id": r["part_id"], "split": r["split"], "path": r["run_path"]} for r in run]}
(OUT / "RUN_MANIFEST.json").write_text(json.dumps(rm, indent=1, sort_keys=True))
c = collections.Counter((r["split"], r.get("sub") if r["split"] in ("DEV", "EXCLUDED") else "") for r in doc["rows"])
for k, v in sorted(c.items()): print(v, k)
print("run manifest", len(run))

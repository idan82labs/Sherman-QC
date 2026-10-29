"""real_jobs.py -> runs_real.tsv (candidate) + runs_real_live.tsv (live) + realscans/: the already-seen real-data
mirror sets of E3_MIRROR_IDENTITY.md (regression only; inputs as listed in its results.json):
G1 natives (50, regenerated tier) + G1 mirrors (74), A1 natives (26) + A1 mirrors (78), held-out DEV cohort natives
(12) + mirrors (36). Mirrors: coordinates and normals negated in one axis, winding reversed (e3_mirror/tools/mirror_scan.py).
Real scan paths come from INVENTORY.json (A1 case pc_X = scan sc_X)."""
import json, os, subprocess
from pathlib import Path
R = Path("/Users/idantal/ShermanQC/FullProject")
W = R / "docs/production/runlog/chirality/work"
M = R / "docs/production/runlog/zeroshot_2026-10-25/station/e3_mirror"
inv = {s["scan_id"]: s for s in json.load(open(R / "docs/production/runlog/zeroshot_2026-10-25/INVENTORY.json"))["scans"]}
corpus = {c["id"]: c for c in json.load(open(W / "corpus.json"))}
res = json.load(open(M / "results.json"))
(W / "realscans").mkdir(exist_ok=True)
jobs = []
def real_scan(sc):
    s = inv[sc]; return os.path.expanduser(s["path"]), s["cad"]["sku"]
def staged(src, name, axis):
    ext = Path(src).suffix.lower()
    dst = W / "realscans" / f"{name}{ext}"
    if not dst.exists():
        if axis is None:
            os.symlink(src, dst)
        else:
            subprocess.run([str(R / ".venv/bin/python"), str(M / "tools/mirror_scan.py"), src, str(dst), axis], check=True,
                           capture_output=True)
    return str(dst)
def add(group, name, scan, cid, axis):
    c = corpus[cid]
    mp = str(W / "maps" / cid / "chirality_map_v0.json")
    jobs.append((group, name, scan, c["cdir"], c["recipe"], mp))
for kind in ("native", "mirror"):
    for r in res["runs"][kind]:
        axis = r.get("axis")
        if r["corpus"] == "G1":
            src = str(W / "g1tier" / r["case"] / "scan.ply") if kind == "mirror" else str(W / "g1tier" / r["id"][2:] / "scan.ply")
            cid = r["case"] if kind == "mirror" else r["id"][2:]
        else:
            pc = r["case"] if kind == "mirror" else r["id"][2:]
            src, cid = real_scan("sc_" + pc[3:])
        add(f"{r['corpus']}_{kind}", r["id"], staged(src, r["id"], axis), cid, axis)
for row in res["heldout_confirmation"]["rows"]:
    src, cid = real_scan(row["case"])
    add("HELDOUT_native", f"HN_{row['case']}", staged(src, f"HN_{row['case']}", None), cid, None)
    for ax in ("x", "y", "z"):
        add("HELDOUT_mirror", f"HM{ax}_{row['case']}", staged(src, f"HM{ax}_{row['case']}", ax), cid, ax)
with open(W / "runs_real.tsv", "w") as f, open(W / "runs_real_live.tsv", "w") as fl, open(W / "real_groups.json", "w") as g:
    groups = {}
    for group, name, scan, cdir, recipe, mp in jobs:
        f.write("\t".join([name, scan, cdir, recipe, mp, str(W / "runs_real" / name)]) + "\n")
        fl.write("\t".join(["L:" + name, scan, cdir, recipe, "-", str(W / "runs_live/real" / name)]) + "\n")
        groups[name] = group
    json.dump(groups, g, indent=1)
print(len(jobs))

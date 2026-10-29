"""Full-Z2 equivalence: Python zs_lane_v2.core.run_z2 (Z2 fallback on; load_scan with the D1 thickness probe,
segmentation, flange graph, E3 metrology, D1 snap) vs zs_rust z2full, lines bitwise.
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/z2full_check.py WORKDIR OUT.json [sid ...]"""
import json, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v2 import core as V2
from research.zeroshot.common.gold_guard import require_runnable
from research.zeroshot.z3_slicing.custody import check_path
BIN = REPO / "research/zeroshot/zs_rust/target/release/zs_rust"
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
W = Path(sys.argv[1]); W.mkdir(parents=True, exist_ok=True); OUT = Path(sys.argv[2])
tsv = dict(l.rstrip("\n").split("\t") for l in open(P / "integration/dev_scans.tsv"))
ids = sys.argv[3:] or sorted(tsv)
res = json.loads(OUT.read_text()) if OUT.is_file() else {}
V2.Z2_FALLBACK = True
skip = {"sc_9b930fe58a1b", "sc_a5e534e6097c", "sc_f362d8de507c"}   # point clouds
for sid in ids:
    if sid in res or sid in skip: continue
    require_runnable(sid, final_testb=False)
    path = str(check_path(tsv[sid]))
    t0 = time.time(); lines, scan, info = V2.run_z2(path); tp = time.time() - t0
    (W / "job.json").write_text(json.dumps({"scan": path, "fallback": True}))
    t0 = time.time()
    r = subprocess.run([str(BIN), "z2full", "--job", str(W / "job.json"), "--out", str(W / "out.json"), "--threads", "8"], capture_output=True, text=True)
    tr = time.time() - t0
    if r.returncode != 0:
        res[sid] = {"error": r.stderr[-500:]}; print(sid, res[sid], flush=True); continue
    rs = json.loads((W / "out.json").read_text())
    pyl = json.loads(json.dumps(lines))
    row = {"lines_eq": pyl == rs["lines"], "t_eq": float(scan.t) == rs["t"], "npts_eq": len(scan.P) == rs["n_points"],
           "n_py": len(pyl), "n_rs": len(rs["lines"]), "fallback_eq": ("fallback" in info) == ("fallback" in rs["info"]),
           "py_s": round(tp, 2), "rs_s": round(tr, 2)}
    if not row["lines_eq"] and len(pyl) == len(rs["lines"]):
        row["max_abs"] = max(abs(a - b) for x, y in zip(pyl, rs["lines"]) for k in ("start", "end") for a, b in zip(x[k], y[k]))
        row["incl_eq"] = [x["incl"] for x in pyl] == [y["incl"] for y in rs["lines"]]
    if not row["t_eq"]: row["t"] = [float(scan.t), rs["t"]]
    row["ok"] = row["lines_eq"] and row["t_eq"] and row["npts_eq"]
    res[sid] = row
    print(sid, json.dumps(row), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
print("OK", sum(1 for r in res.values() if r.get("ok")), "of", len(res))

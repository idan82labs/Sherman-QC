"""Full-Z3 equivalence: Python (z3s.load_mesh + zs_lane_v3 rev2 z3_merge.detect, fallback_frac_only) vs zs_rust z3full
(mesh read, thickness ray casts, families, slicing ... NMS), bitwise after the JSON round trip.
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/z3full_check.py WORKDIR OUT.json [sid ...]"""
import json, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
import numpy as np
from research.zeroshot.zs_lane_v2 import z3s
from research.zeroshot.zs_lane_v3_rs import rev2ref; rev2ref.install()  # frozen revision 2, not the live package
from research.zeroshot.zs_lane_v3 import z3_merge
from research.zeroshot.common.gold_guard import require_runnable
from research.zeroshot.z3_slicing.custody import check_path
BIN = REPO / "research/zeroshot/zs_rust/target/release/zs_rust"
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
W = Path(sys.argv[1]); W.mkdir(parents=True, exist_ok=True); OUT = Path(sys.argv[2])
tsv = dict(l.rstrip("\n").split("\t") for l in open(P / "integration/dev_scans.tsv"))
ids = sys.argv[3:] or sorted(tsv)
res = json.loads(OUT.read_text()) if OUT.is_file() else {}
rt = lambda x: json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
for sid in ids:
    if sid in res: continue
    require_runnable(sid, final_testb=False)
    path = str(check_path(tsv[sid]))
    t0 = time.time()
    m = z3s.load_mesh(path)
    if len(m.F) == 0: continue
    py = z3_merge.detect(m, z3s.Params(fallback_frac_only=True)); tp = time.time() - t0
    (W / "job.json").write_text(json.dumps({"scan": path, "fallback_frac_only": True}))
    t0 = time.time()
    r = subprocess.run([str(BIN), "z3full", "--job", str(W / "job.json"), "--out", str(W / "out.json"), "--threads", "8"], capture_output=True, text=True)
    tr = time.time() - t0
    if r.returncode != 0:
        res[sid] = {"error": r.stderr[-500:]}; print(sid, res[sid], flush=True); continue
    rs = json.loads((W / "out.json").read_text())
    row = {"t_eq": py["t_mm"] == rs["t_mm"], "thick_eq": rt(py["thickness"]) == rt(rs["thickness"]),
           "fam_eq": rt(py["families"]) == rt(rs["families"]), "fallback_eq": py["family_fallback"] == rs["family_fallback"],
           "raw_eq": py["raw_count"] == rs["raw_count"], "cand_eq": rt(py["candidates"]) == rt(rs["candidates"]),
           "n_cand": len(py["candidates"]), "py_s": round(tp, 2), "rs_s": round(tr, 2)}
    if not row["t_eq"]: row["t"] = [py["t_mm"], rs["t_mm"]]
    if not row["fam_eq"]: row["nfam"] = [len(py["families"]), len(rs["families"])]
    row["ok"] = all(v for k, v in row.items() if k.endswith("_eq"))
    res[sid] = row
    print(sid, json.dumps(row), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
print("OK", sum(1 for r in res.values() if r.get("ok")), "of", len(res))

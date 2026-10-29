"""Mesh reading equivalence: open3d.io.read_triangle_mesh vs zs_rust meshio on every DEV scan (V, F bitwise)."""
import json, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
import numpy as np, open3d as o3d
from research.zeroshot.common.gold_guard import require_runnable
from research.zeroshot.z3_slicing.custody import check_path
from research.zeroshot.zs_lane_v3_rs import rev2ref; rev2ref.install()  # frozen revision 2, not the live package
from research.zeroshot.zs_lane_v3 import budget
BIN = REPO / "research/zeroshot/zs_rust/target/release/zs_rust"
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
W = Path(sys.argv[1]); W.mkdir(parents=True, exist_ok=True)
out = Path(sys.argv[2])
tsv = dict(l.rstrip("\n").split("\t") for l in open(P / "integration/dev_scans.tsv"))
res = {}
for sid in sorted(tsv):
    require_runnable(sid, final_testb=False)
    path = str(check_path(tsv[sid]))
    m = o3d.io.read_triangle_mesh(path)
    V = np.asarray(m.vertices, np.float64); F = np.asarray(m.triangles, np.int64)
    (W / "job.json").write_text(json.dumps({"mesh": path}))
    subprocess.run([str(BIN), "selftest", "--job", str(W / "job.json"), "--out", str(W / "out.json")], check=True)
    o = json.loads((W / "out.json").read_text())["mesh"]
    RV = np.load(W / "mesh_V.npy"); RF = np.load(W / "mesh_F.npy")
    ok = RV.shape == V.shape and RF.shape == F.shape and np.array_equal(RV, V) and np.array_equal(RF, F)
    res[sid] = {"ok": bool(ok), "nv": len(V), "nf": len(F), "rs_nv": len(RV), "rs_nf": len(RF),
                "face_count_eq": o["face_count"] == budget.face_count(path), "point_cloud": len(F) == 0}
    print(sid, res[sid], flush=True)
out.write_text(json.dumps(res, indent=1))
print("ALL OK", all(r["ok"] for r in res.values()), "point clouds", [s for s, r in res.items() if r["point_cloud"]])

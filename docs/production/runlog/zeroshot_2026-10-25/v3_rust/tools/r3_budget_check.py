"""Revision-3 budget step, mesh level: Python zs_lane_v3 budget.decimate (frozen revision 3; files checked against
config_v3.json's fingerprint dc23a905...) against zs_rust's budget (selftest decimate, cell_mm from the config, 16
threads): resampled vertices and faces bitwise, budget info equal. Engineering only (no lines, nothing scored).
DEV scans are given as sid (guard + custody); derived dense meshes as paths.
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/r3_budget_check.py OUT.json sid|mesh.ply [...]"""
import hashlib, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v3_rs import rs
cfg = json.loads((REPO / "research/zeroshot/zs_lane_v3/config_v3.json").read_text())
assert cfg["revision"] == 3 and cfg["code_fingerprint"]["sha256"].startswith("dc23a905")
for f_, h_ in cfg["code_fingerprint"]["files"].items():
    assert hashlib.sha256((REPO / f_).read_bytes()).hexdigest() == h_, f_
from research.zeroshot.zs_lane_v3 import budget
import open3d as o3d
MF, CELL = int(cfg["budget"]["max_faces"]), float(cfg["budget"]["cell_mm"])
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
tsv = dict(l.rstrip("\n").split("\t") for l in open(P / "integration/dev_scans.tsv"))
out = Path(sys.argv[1])
res = json.loads(out.read_text()) if out.is_file() else {}
for item in sys.argv[2:]:
    if item in res:
        continue
    if item.startswith("sc_"):
        from research.zeroshot.common.gold_guard import require_runnable
        from research.zeroshot.z3_slicing.custody import check_path
        require_runnable(item, final_testb=False)
        path = str(check_path(tsv[item]))
    else:
        path = item
    m = o3d.io.read_triangle_mesh(path)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.triangles)
    fn = np.asarray(m.vertex_normals, dtype=np.float64) if m.has_vertex_normals() else None
    t0 = time.time()
    X, T, _No, info = budget.decimate(V, F, MF, fn, CELL)
    pyw = time.time() - t0
    del m, V, F, fn
    with tempfile.TemporaryDirectory(prefix="zs_r3b_") as d:
        Path(d, "job.json").write_text(json.dumps({"decimate": {"scan": path, "max_faces": MF, "cell_mm": CELL}}))
        t0 = time.time()
        r = subprocess.run([str(rs.binary()), "selftest", "--job", f"{d}/job.json", "--out", f"{d}/o.json", "--threads", "16"],
                           capture_output=True, text=True)
        rsw = time.time() - t0
        assert r.returncode == 0, r.stderr[-500:]
        o = json.loads(Path(d, "o.json").read_text())
        RX, RT = np.load(Path(d, "dec_V.npy")), np.load(Path(d, "dec_F.npy"))
    ri = o["decimate"]
    row = {"V_equal": bool(RX.shape == X.shape and np.array_equal(RX, X)),
           "F_equal": bool(RT.shape == T.shape and np.array_equal(RT, T.astype(np.int64))),
           "info_equal": json.dumps(ri) == json.dumps(info), "info_py": info, "info_rs": ri,
           "python_decimate_s": round(pyw, 2), "rust_selftest_s": round(rsw, 2)}
    if not row["V_equal"] and RX.shape == X.shape:
        row["V_max_abs"] = float(np.abs(RX - X).max())
    res[item] = row
    print(item, {k: row[k] for k in ("V_equal", "F_equal", "info_equal", "python_decimate_s", "rust_selftest_s")}, flush=True)
    out.write_text(json.dumps(res, indent=1))
print("all equal", sum(v["V_equal"] and v["F_equal"] and v["info_equal"] for v in res.values()), "of", len(res))

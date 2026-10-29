"""Ray-cast equivalence: open3d RaycastingScene (Embree) vs zs_rust raycast on the lane's own rays (Z3 thickness
rays and D1 two-skin probe rays) of DEV scans. Guard + custody per scan; run under the slot."""
import json, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
import numpy as np, open3d as o3d
from research.zeroshot.zs_lane_v2 import z3s
from research.zeroshot.d1_two_skin import two_skin as D1
from research.zeroshot.common.gold_guard import require_runnable
from research.zeroshot.z3_slicing.custody import check_path
BIN = REPO / "research/zeroshot/zs_rust/target/release/zs_rust"
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
W = Path(sys.argv[1]); W.mkdir(parents=True, exist_ok=True)
tsv = dict(l.rstrip("\n").split("\t") for l in open(P / "integration/dev_scans.tsv"))
res = {}
for sid in sys.argv[2:]:
    require_runnable(sid, final_testb=False)
    m = z3s.load_mesh(str(check_path(tsv[sid])))
    rng = np.random.default_rng(0)
    p = m.fa / m.fa.sum()
    idx = rng.choice(len(m.F), size=min(20000, len(m.F)), replace=False, p=p)
    c = m.V[m.F[idx]].mean(1); d = -m.fn[idx]
    r1 = np.concatenate([c + d * 1e-3, d], 1)
    # D1 probe rays (vertices along -vn)
    sel = np.arange(0, len(m.V), max(1, len(m.V) // 30000))
    o = m.V[sel] + 0.02 * (-m.vn[sel]); r2 = np.concatenate([o, -m.vn[sel]], 1)
    R = np.vstack([r1, r2])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(m.V.astype(np.float32)), o3d.core.Tensor(m.F.astype(np.uint32)))
    ans = scene.cast_rays(o3d.core.Tensor(R.astype(np.float32)))
    th = ans["t_hit"].numpy().astype(np.float64); pid = ans["primitive_ids"].numpy().astype(np.int64)
    pid[~np.isfinite(th)] = -1
    np.save(W / "V.npy", m.V); np.save(W / "F.npy", m.F.astype(np.int64)); np.save(W / "R.npy", R.astype(np.float32).astype(np.float64))
    (W / "job.json").write_text(json.dumps({"rays": {"V": "V.npy", "F": "F.npy", "R": "R.npy"}}))
    subprocess.run([str(BIN), "selftest", "--job", str(W / "job.json"), "--out", str(W / "out.json"), "--threads", "8"], check=True)
    rt = np.load(W / "ray_t.npy"); rp = np.load(W / "ray_pid.npy")
    teq = (th == rt) | (~np.isfinite(th) & ~np.isfinite(rt))
    peq = pid == rp
    res[sid] = {"rays": len(R), "t_equal": int(teq.sum()), "pid_equal": int(peq.sum()),
                "t_diff_examples": [[float(a), float(b)] for a, b in zip(th[~teq][:5], rt[~teq][:5])]}
    print(sid, json.dumps(res[sid]), flush=True)

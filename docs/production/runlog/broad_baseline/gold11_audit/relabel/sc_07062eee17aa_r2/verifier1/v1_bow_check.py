"""B2 bowed-fold check: label vs best straight lines through verifier1's measured outer-apex stations (no scan load)."""
import json, numpy as np, os
from scipy.optimize import minimize
H = os.path.dirname(os.path.abspath(__file__))
for bid in ("B1", "B2"):
    D = np.load(os.path.join(H, f"apex3d_{bid}.npz")); A = D["apex"]; BL = D["bl3"]; T = D["t"]
    B = [b for b in json.load(open(os.path.join(H, "../lines.json")))["bends"] if b["bend_id"] == bid][0]
    s = np.array(B["label_line"]["start"]); e = np.array(B["label_line"]["end"]); d = (e - s) / np.linalg.norm(e - s)
    def stats(p0, u):
        foot = p0 + np.outer((A - p0) @ u, u); r = foot - A
        dep = np.einsum("ij,ij->i", r, BL); dist = np.linalg.norm(r, axis=1)
        return dep, dist
    def report(tag, p0, u):
        dep, dist = stats(p0, u)
        print(f"{bid} {tag:8s} depth med {np.median(dep):+.3f} mean {dep.mean():+.3f} | dist med {np.median(dist):.3f} p90 {np.percentile(dist,90):.3f} max {dist.max():.3f} | <=0.5mm {np.mean(dist<=0.5):.2f} <=1.0mm {np.mean(dist<=1.0):.2f} | ends dist t<10: {np.median(dist[T<10]):.2f}  t>L-10: {np.median(dist[T>T.max()-10]):.2f}")
    report("label", s, d)
    c = A.mean(0); _, _, vt = np.linalg.svd(A - c); u = vt[0] * np.sign(vt[0] @ d)
    report("LSQ", c, u)
    # minimax line (minimise max distance), start from LSQ
    e1 = np.cross(u, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    def f(x):
        p0 = c + x[0] * e1 + x[1] * e2; uu = u + x[2] * e1 + x[3] * e2; uu /= np.linalg.norm(uu)
        return np.percentile(stats(p0, uu)[1], 99)
    r = minimize(f, np.zeros(4), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-5, maxiter=4000))
    x = r.x; p0 = c + x[0] * e1 + x[1] * e2; uu = u + x[2] * e1 + x[3] * e2; uu /= np.linalg.norm(uu)
    report("minimax", p0, uu)

import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
pts = np.load(W / "pts.npy"); nrm = np.load(W / "nrm.npy"); L = np.load(W / "local.npy")
c, e1, e2, e3 = np.load(W / "pca.npy")
G = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.load(open(W / "gold_local.json")).items()}
nl = nrm @ np.vstack([e1, e2, e3]).T
# 1) plate points: |n3|>0.97, |h - median| < 3 and in central region away from walls
plate = (np.abs(nl[:, 2]) > 0.97) & (np.abs(L[:, 0]) < 140) & (np.abs(L[:, 1]) < 135) & (L[:, 2] < 0)
P = L[plate]
# fit plane to all plate points (both skins mixed -> mid-surface)
A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
res = P[:, 2] - A @ coef
print("plate plane fit: h = %.5f x + %.5f y + %.3f ; resid std %.3f, p2/p98 %.2f %.2f" % (coef[0], coef[1], coef[2], res.std(), *np.percentile(res, [2, 98])))
# separate the two skins by residual sign
hist, edges = np.histogram(res, bins=60, range=(-2, 2))
print("residual histogram (centre:count)", [(round((edges[i] + edges[i + 1]) / 2, 2), int(hist[i])) for i in range(60) if hist[i] > 0.01 * hist.max()])
n_plate = np.array([-coef[0], -coef[1], 1.0]); n_plate /= np.linalg.norm(n_plate)
print("plate normal in local", n_plate.round(5), "tilt vs e3 deg %.3f" % math.degrees(math.acos(n_plate[2])))
# 2) wall points: |n3| < 0.15, h > 8
wall = (np.abs(nl[:, 2]) < 0.15) & (L[:, 2] > 8) & (L[:, 2] < 30)
Wp = L[wall]; Wn = nl[wall]
for name, sgn in (("M01-side wall (e2>0)", 1), ("M02-side wall (e2<0)", -1)):
    m = (Wp[:, 1] * sgn > 100)
    Q = Wp[m]
    # wall plane: fit e2 = a + b e1 + c h
    A2 = np.c_[np.ones(len(Q)), Q[:, 0], Q[:, 2]]
    cf, *_ = np.linalg.lstsq(A2, Q[:, 1], rcond=None)
    r2 = Q[:, 1] - A2 @ cf
    print(name, "n=%d" % len(Q), "e1 range %.1f..%.1f (p0.5/p99.5 %.1f..%.1f)" % (Q[:, 0].min(), Q[:, 0].max(), *np.percentile(Q[:, 0], [0.5, 99.5])),
          "plane e2 = %.3f + %.5f e1 + %.5f h" % tuple(cf), "resid p2/p98 %.2f %.2f" % tuple(np.percentile(r2, [2, 98])))
    # dir of wall in plate: (1, cf[1], ?) -> angle vs e1
    print("   wall trace angle vs e1 (deg) %.3f" % math.degrees(math.atan(cf[1])))
    # height extent of wall
    print("   wall h range p1/p99 %.1f %.1f" % tuple(np.percentile(Q[:, 2], [1, 99])))
# 3) lip points: |n3| > 0.9, h > 25
lip = (np.abs(nl[:, 2]) > 0.9) & (L[:, 2] > 25)
Lp = L[lip]
for name, sgn in (("M01 lip", 1), ("M02 lip", -1)):
    m = Lp[:, 1] * sgn > 90
    Q = Lp[m]
    print(name, "n=%d" % len(Q), "e1 range p0.5/p99.5 %.1f..%.1f" % tuple(np.percentile(Q[:, 0], [0.5, 99.5])), "e2 range %.1f..%.1f" % tuple(np.percentile(Q[:, 1], [0.5, 99.5])), "h p5/p95 %.1f %.1f" % tuple(np.percentile(Q[:, 2], [5, 95])))
# 4) gold lines in this frame
for k, (a, b) in G.items():
    d = b - a
    print(k, "len %.1f" % np.linalg.norm(d), "angle vs e1 %.2f deg" % math.degrees(math.atan2(d[1], d[0])), "start", a.round(1), "end", b.round(1))
np.save(W / "plate_coef.npy", coef)

"""Part frame for sc_f2475286312b: fit the main plate (both skins), define
(a = along long edges / M01 dir, b = across, h = plate normal), print gold lines in that frame."""
import json
import numpy as np
from scipy.spatial import cKDTree
from a3common import load_mesh, load_gold, unit, OUT

V, N, F = load_mesh()
G = load_gold()
d1 = unit(G["M01"][1] - G["M01"][0])
d2 = unit(G["M07"][1] - G["M07"][0])
d2 = unit(d2 - (d2 @ d1) * d1)
n0 = unit(np.cross(d1, d2))
c = V.mean(0)
# plate: vertices whose normal is ~parallel to n0, then robust plane fit per skin
cosn = N @ n0
cand = np.abs(cosn) > 0.995
h = (V - c) @ n0
print("cand", cand.sum())
hist, edges = np.histogram(h[cand], bins=400)
top = np.argsort(hist)[::-1][:6]
print("dominant h bins (count, h):", [(int(hist[i]), round(float((edges[i] + edges[i + 1]) / 2), 2)) for i in top])
# take the most populated h level +-1.5 mm and fit a plane (both skins separately by normal sign)
hpk = (edges[top[0]] + edges[top[0] + 1]) / 2
res = {}
for sgn in (1, -1):
    m = cand & (np.abs(h - hpk) < 3.0) & (np.sign(cosn) == sgn)
    P = V[m]
    for _ in range(3):
        cc = P.mean(0)
        u, s, vt = np.linalg.svd(P - cc, full_matrices=False)
        nn = vt[2] if vt[2] @ n0 > 0 else -vt[2]
        r = (P - cc) @ nn
        keep = np.abs(r) < 3 * np.std(r) + 0.05
        P = P[keep]
    res[sgn] = (cc, nn, float(np.std((P - cc) @ nn)), len(P))
    print("skin normal-sign", sgn, "pts", len(P), "normal", np.round(nn, 5), "rms %.3f" % res[sgn][2],
          "h(center) %.3f" % ((cc - c) @ n0))
n = unit(res[1][1] + res[-1][1])
# re-orthogonalise frame: h = n, a = d1 projected, b = n x a
a = unit(d1 - (d1 @ n) * n)
b = np.cross(n, a)
hs = sorted([float((res[s][0] - c) @ n) for s in (1, -1)])
print("plate skins h:", np.round(hs, 3), "thickness %.3f" % (hs[1] - hs[0]))
mid_h = 0.5 * (hs[0] + hs[1])
origin = c + mid_h * n
frame = {"origin": origin.tolist(), "a": a.tolist(), "b": b.tolist(), "n": n.tolist(),
         "plate_skin_h": hs, "thickness": hs[1] - hs[0],
         "skin_normal_sign_to_h": {str(s): float((res[s][0] - origin) @ n) for s in (1, -1)}}
(OUT / "frame.json").write_text(json.dumps(frame, indent=1))
X = V - origin
A, B, H = X @ a, X @ b, X @ n
print("vertex extent a [%.1f, %.1f] b [%.1f, %.1f] h [%.1f, %.1f]" % (A.min(), A.max(), B.min(), B.max(), H.min(), H.max()))
for k, (s, e) in G.items():
    ps = (s - origin) @ np.array([a, b, n]).T
    pe = (e - origin) @ np.array([a, b, n]).T
    print(k, "start(a,b,h)", np.round(ps, 2), "end", np.round(pe, 2))

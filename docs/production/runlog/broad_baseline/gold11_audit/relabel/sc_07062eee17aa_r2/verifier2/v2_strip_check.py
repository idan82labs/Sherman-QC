"""Verifier2: full-res sections of the -Y web strip (window B to flange) and the +X/-Y corner step, to rule out missed formed bends."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_sections import load_ply, LINES, SCAN, HERE

PREP = os.path.join(HERE, "..", "..", "..", "prep", "sc_07062eee17aa")
pat = {p["id"]: p for p in json.load(open(os.path.join(PREP, "patches.json")))["patches"]}
cfg = json.load(open(LINES))["bends"]
X, _ = load_ply(SCAN)
nw = np.array(pat[1]["normal"]); nw /= np.linalg.norm(nw)
ex = np.array(cfg[0]["direction"]); ex -= (ex @ nw) * nw; ex /= np.linalg.norm(ex)
ey = np.cross(nw, ex)
x, y, z = X @ ex, X @ ey, X @ nw
m0 = (y > 215) & (y < 272)
x, y, z = x[m0], y[m0], z[m0]
fig, axs = plt.subplots(2, 5, figsize=(26, 9)); axs = axs.ravel()
plan = [(-470, 230), (-420, 230), (-380, 225), (-340, 225), (-300, 225), (-262, 225), (-230, 230), (-205, 235), (-198, 235), (-193, 235)]
res = []
for ax, (x0, ylo) in zip(axs, plan):
    m = (np.abs(x - x0) < 0.4) & (y > ylo)
    ax.scatter(y[m], z[m], s=2, c="k")
    ax.set_aspect(2.0); ax.grid(alpha=0.3)
    # tilt of the web strip: fit lines to the upper skin between y 250..261 and 232..245
    zz = z[m]; yy = y[m]
    def tilt(a, b):
        k = (yy > a) & (yy < b) & (zz > np.median(zz[(yy > a) & (yy < b)]) if ((yy > a) & (yy < b)).sum() > 5 else np.zeros_like(yy, bool))
        if k.sum() < 5: return None
        p = np.polyfit(yy[k], zz[k], 1); return float(np.degrees(np.arctan(p[0])))
    t1 = tilt(251, 261); t0 = tilt(225, 245)
    res.append(dict(x=x0, tilt_strip_deg=t1, tilt_inner_deg=t0))
    ax.set_title(f"x={x0} (part frame), z x2; strip tilt {t1 if t1 is None else round(t1,1)} deg", fontsize=9)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("z (web normal)")
fig.suptitle("verifier2: -Y side web strip / corner sections at full scan resolution (looking for creases, joggles, small formed bends)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "v2_strip_sections.png"), dpi=75); plt.close(fig)
# top view of the +X/-Y corner, coloured by z
m = (x > -235) & (y > 235)
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(x[m], y[m], c=z[m], s=0.5, cmap="viridis"); plt.colorbar(sc, ax=ax)
ax.set_aspect("equal"); ax.set_title("+X/-Y corner, top view coloured by z (web normal)")
fig.savefig(os.path.join(HERE, "v2_corner_top.png"), dpi=80); plt.close(fig)
print(json.dumps(res))

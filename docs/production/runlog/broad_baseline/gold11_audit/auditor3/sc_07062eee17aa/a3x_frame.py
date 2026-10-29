"""Part frame + overview renders (top view coloured by height off the web; side elevations)."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from a3x_common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = load_points()
c = p.mean(0)
rng = np.random.default_rng(0)
s = p[rng.choice(len(p), 60000, replace=False)] - c
_, _, vt = np.linalg.svd(s, full_matrices=False)
X = unit(-vt[0])            # along the long bends (gold direction sign)
Z0 = unit(vt[2])            # plate normal; side flanges point to -Z0 per shared render
# robust web plane: iterate on points near the plane
h = (p - c) @ Z0
sel = np.abs(h - np.median(h)) < 2.5
for _ in range(4):
    q = p[sel]
    cq = q.mean(0)
    _, _, v2 = np.linalg.svd(q - cq, full_matrices=False)
    n = unit(v2[2]); n = n if n @ Z0 > 0 else -n
    h = (p - cq) @ n
    sel = np.abs(h) < 1.6
Z = n
X = unit(X - Z * (X @ Z))
Y = np.cross(Z, X)
O = cq
loc = np.c_[(p - O) @ X, (p - O) @ Y, (p - O) @ Z]
# two skins of the web: histogram of h near 0
hh = loc[np.abs(loc[:, 2]) < 2.5, 2]
hist, e = np.histogram(hh, bins=100, range=(-2.5, 2.5))
pk = e[:-1][np.argsort(hist)[-6:]] + 0.025
print('web skins h-hist top bins', sorted(np.round(pk, 3)))
frame = {"O": O.tolist(), "X": X.tolist(), "Y": Y.tolist(), "Z": Z.tolist()}
json.dump(frame, open(OUT / "a3x_frame.json", "w"), indent=1)
print('frame', frame)
print('extent X', loc[:, 0].min(), loc[:, 0].max(), 'Y', loc[:, 1].min(), loc[:, 1].max(), 'Z', loc[:, 2].min(), loc[:, 2].max())

gold = load_gold()
G = {}
for bid, a, b in gold:
    la = [(a - O) @ X, (a - O) @ Y, (a - O) @ Z]
    lb = [(b - O) @ X, (b - O) @ Y, (b - O) @ Z]
    G[bid] = (np.array(la), np.array(lb))
    print(bid, 'start XYZ', np.round(la, 2), 'end XYZ', np.round(lb, 2), 'len', round(float(np.linalg.norm(b - a)), 1))

sub = rng.choice(len(p), 250000, replace=False)
L = loc[sub]
fig, axs = plt.subplots(2, 1, figsize=(16, 22), gridspec_kw={"height_ratios": [3, 1.2]})
ax = axs[0]
sc_ = ax.scatter(L[:, 0], L[:, 1], c=np.clip(L[:, 2], -16, 16), s=0.3, cmap="coolwarm", vmin=-16, vmax=16)
plt.colorbar(sc_, ax=ax, fraction=0.03, label="height off web mid-surface (mm)")
for bid, (a, b) in G.items():
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#1a9c3a", lw=1.2)
    ax.text(b[0] + 2, b[1], bid, color="#1a9c3a", fontsize=11, weight="bold")
    ax.plot([a[0]], [a[1]], "o", color="#1a9c3a", ms=4)
ax.set_aspect("equal"); ax.grid(alpha=.3)
ax.set_title("sc_07062eee17aa top view in part frame (X along bends, Y across, colour = Z off web). green = human gold")
ax = axs[1]
# end elevation: Y-Z
ax.scatter(L[:, 1], L[:, 2], s=0.2, c="0.4")
for bid, (a, b) in G.items():
    ax.plot([a[1], b[1]], [a[2], b[2]], "o-", color="#1a9c3a", ms=4)
    ax.text(a[1] + 1, a[2] + 1, bid, color="#1a9c3a", fontsize=10)
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title("end elevation Y-Z (all X projected)")
fig.tight_layout()
fig.savefig(OUT / "a3x_overview_topZ.png", dpi=80)

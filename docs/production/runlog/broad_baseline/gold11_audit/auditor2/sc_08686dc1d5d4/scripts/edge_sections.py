import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_08686dc1d5d4")
L = np.load(W / "local.npy")
th = math.radians(-6.225)
R = np.array([[math.cos(th), math.sin(th), 0], [-math.sin(th), math.cos(th), 0], [0, 0, 1]])
P = L @ R.T   # plate frame: x along walls, y toward M01 wall, z plate normal (walls up)
np.save(W / "plateframe.npy", P); np.save(W / "R_plate.npy", R)
G = {k: (np.array(v[0]) @ R.T, np.array(v[1]) @ R.T) for k, v in json.load(open(W / "gold_local.json")).items()}
json.dump({k: [a.tolist(), b.tolist()] for k, (a, b) in G.items()}, open(W / "gold_plateframe.json", "w"), indent=1)
for k, (a, b) in G.items():
    print(k, "start", a.round(2), "end", b.round(2))
print("plate-frame bbox", P.min(0).round(1), P.max(0).round(1))
ys = [-140, -100, -60, -20, 20, 60, 100, 140]
fig, axs = plt.subplots(len(ys), 2, figsize=(26, 3.2 * len(ys)))
for i, y0 in enumerate(ys):
    m = (np.abs(P[:, 1] - y0) < 0.5) & (P[:, 2] < 6)
    for j, (lo, hi) in enumerate(((-200, -60), (110, 200))):
        ax = axs[i, j]; mm = m & (P[:, 0] > lo) & (P[:, 0] < hi)
        ax.scatter(P[mm, 0], P[mm, 2], s=1.0, c="k")
        for k, (a, b) in G.items():
            d = b - a
            if abs(d[1]) > 1e-6:
                t = (y0 - a[1]) / d[1]
                if -0.02 <= t <= 1.02:
                    p = a + t * d
                    if lo <= p[0] <= hi:
                        ax.plot(p[0], p[2], "*", color="#1a9c3a", ms=14, mec="k"); ax.text(p[0], p[2] + 0.8, k, color="#1a9c3a", fontsize=11)
        ax.set_xlim(lo, hi); ax.set_ylim(-9, 1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"plate-frame section y={y0} (x along walls, z plate normal) {'LEFT edge + M03' if j == 0 else 'RIGHT edge + M04'}", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "a2_edge_sections_joggles.png", dpi=70)
print("saved")

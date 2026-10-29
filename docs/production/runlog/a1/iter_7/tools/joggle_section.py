"""2D cross-section of a joggle on an E3 registered scan (diagnostic only). usage: joggle_section.py ENGINE_DIR OUT.png Y0 HALF X0 X1 Z0 Z1"""
import sys, json
import numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from domains.production.engine.inspect.scan_io import read_scan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = sys.argv[1]; out = sys.argv[2]
y0, half, x0, x1, z0, z1 = map(float, sys.argv[3:9])
pts = read_scan(d + "/registered_scan.ply").points
m = (np.abs(pts[:, 1] - y0) < half) & (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 2] > z0) & (pts[:, 2] < z1)
q = pts[m]
geo = json.load(open(d + "/bend_geometry.json"))
fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(q[:, 0], q[:, 2], s=0.5, c="k")
for b in geo["bends"]:
    for key, col in (("measured_polyline_mm", "r"), ("nominal_polyline_mm", "b")):
        pl = b.get(key)
        if not pl:
            continue
        pl = np.asarray(pl)
        # point on the line at y0
        a = pl[-1] - pl[0]
        if abs(a[1]) < 1e-6:
            continue
        tpar = (y0 - pl[0][1]) / a[1]
        p = pl[0] + tpar * a
        if x0 < p[0] < x1 and z0 < p[2] < z1:
            ax.plot(p[0], p[2], col + "o", ms=6)
            ax.annotate(b["label"] + ("m" if col == "r" else "n"), (p[0], p[2]), color=col)
ax.set_aspect("equal"); ax.grid(True, lw=0.3)
ax.set_xlabel("x mm"); ax.set_ylabel("z mm"); ax.set_title(f"section y={y0}+-{half}  n={len(q)}")
fig.savefig(out, dpi=110)
print(len(q))

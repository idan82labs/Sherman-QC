"""Measure the vertical-axis kink at the +a end of each window tab (h=const sections)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3common import load_mesh, slice_mesh, OUT

V, N, F = load_mesh()
fr = json.loads((OUT / "frame.json").read_text())
o, a, b, n = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([a, b, n])
TC = V[F].mean(1)
fig, axes = plt.subplots(2, 4, figsize=(22, 9), dpi=95)
for row, (name, bc) in enumerate((("TAB_+b", 83.0), ("TAB_-b", -91.0))):
    for col, hv in enumerate((-2.0, -5.0, -9.0, -13.0)):
        c3 = o + np.array([10.0, bc, hv]) @ R
        segs = slice_mesh(V, F, c3, n, tri_centroid=TC, radius=30)
        P = ((segs.reshape(-1, 3) - o) @ R.T).reshape(-1, 2, 3)
        pts = P.reshape(-1, 3)
        m = (np.abs(pts[:, 1] - bc) < 8) & (pts[:, 0] > -20) & (pts[:, 0] < 25)
        pts = pts[m]
        ax = axes[row, col]
        for s in P:
            if abs(s[0, 1] - bc) < 8 and -20 < s[0, 0] < 25:
                ax.plot(s[:, 0], s[:, 1], color="k", lw=0.8)
        # fit main face (a in [-15, 5]) and end facet (last 4 mm before the tip)
        if len(pts) > 20:
            main = pts[(pts[:, 0] > -15) & (pts[:, 0] < 5)]
            tip_a = pts[:, 0].max()
            end = pts[(pts[:, 0] > tip_a - 4.5) & (pts[:, 0] < tip_a - 1.0)]
            def dirfit(Q):
                c = Q[:, :2].mean(0)
                u, s_, vt = np.linalg.svd(Q[:, :2] - c)
                d = vt[0] * np.sign(vt[0][0])
                return c, d
            if len(main) > 5 and len(end) > 5:
                c1, d1 = dirfit(main)
                c2, d2 = dirfit(end)
                ang = np.degrees(np.arccos(np.clip(abs(d1 @ d2), 0, 1)))
                ax.set_title(f"{name} h={hv}: kink {ang:.1f} deg, tip at a={tip_a:.1f}", fontsize=8)
                print(name, "h", hv, "kink angle %.1f" % ang, "tip a %.1f" % tip_a, "main dir", d1.round(3), "end dir", d2.round(3))
        ax.set_xlim(-5, 25)
        ax.set_ylim(bc - 6, bc + 6)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "a3_zoom_tab_kink.png")
plt.close(fig)

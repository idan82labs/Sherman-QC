"""Auditor 2: zoomed e2-cuts near the ends of the short-side bottom bends (M01/M02) to establish the fold extent."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}
st = [-97, -92, -87, -82, -77, -72, -60, 75, 82, 87, 92, 97, 102]
fig, axs = plt.subplots(2, len(st), figsize=(3.2*len(st), 8))
for j, (side, x0, keys) in enumerate([('e1-min', -149.0, ['M01']), ('e1-max', 163.9, ['M02'])]):
    for i, s in enumerate(st):
        ax = axs[j, i]
        m = (np.abs(P[:, 1]-s) < 0.6) & (np.abs(P[:, 0]-x0) < 16) & (P[:, 2] < -8) & (P[:, 2] > -25)
        col = np.where(Nn[m, 2] < -0.5, 'k', np.where(Nn[m, 2] > 0.5, 'tab:orange', 'tab:blue'))
        ax.scatter(P[m, 0], P[m, 2], s=1.0, c=col)
        for k in keys:
            A, B = GL[k]; t = (s - A[1])/(B[1]-A[1])
            if -0.02 <= t <= 1.02:
                g = A + t*(B-A); ax.plot(g[0], g[2], '*', color='lime', mec='k', ms=13)
        ax.set_xlim(x0-6 if x0 < 0 else x0-16, x0+16 if x0 < 0 else x0+6); ax.set_ylim(-24, -10); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_title(f'{side} bottom, e2={s}', fontsize=9)
plt.tight_layout(); plt.savefig(OUT + '/a2_short_bottom_ends.png', dpi=60); plt.close()
print('ok')

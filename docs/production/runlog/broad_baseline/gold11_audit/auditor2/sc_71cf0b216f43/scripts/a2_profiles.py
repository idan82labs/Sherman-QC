"""Auditor 2: full cross-section profiles of the long sides (cut planes perpendicular to e1) with gold crossings."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U)
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}
cols = {'M01':'tab:green','M02':'tab:olive','M03':'tab:cyan','M04':'tab:purple','M05':'red','M06':'orange','M07':'magenta','M08':'blue'}

def gold_at(k, axis, val):
    A, B = GL[k]
    if abs(B[axis]-A[axis]) < 1e-6: return None
    t = (val - A[axis]) / (B[axis]-A[axis])
    if t < -0.02 or t > 1.02: return None
    return A + t*(B-A), t

stations = [150, 120, 80, 40, 0, -40, -80, -120, -140]
slab = 0.6
fig, axs = plt.subplots(len(stations), 2, figsize=(14, 4.2*len(stations)))
for i, s in enumerate(stations):
    m = np.abs(P[:, 0]-s) < slab
    for j, (side, lo, hi, keys) in enumerate([('e2-min long side (M05/M06)', -112, -80, ['M05','M06']), ('e2-max long side (M07/M08)', 90, 122, ['M07','M08'])]):
        ax = axs[i, j]
        mm = m & (P[:, 1] > lo) & (P[:, 1] < hi)
        ax.scatter(P[mm, 1], P[mm, 2], s=0.6, c='k')
        for k in keys:
            r = gold_at(k, 0, s)
            if r is None: continue
            g, t = r
            ax.plot(g[1], g[2], marker='*', ms=14, color=cols[k], mec='k')
            ax.annotate(f'{k} t={t:.2f}', (g[1], g[2]), xytext=(6, 4), textcoords='offset points', color=cols[k], fontsize=9)
        ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_xlim(lo, hi); ax.set_ylim(-26, 26)
        ax.set_title(f'{side}: cut e1={s} (+-{slab} mm)', fontsize=10)
        ax.set_xlabel('e2 (mm)'); ax.set_ylabel('e3 (mm)')
plt.tight_layout()
plt.savefig(OUT + '/a2_long_side_profiles.png', dpi=70)
print('ok')

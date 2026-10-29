"""Auditor 2: short-side profiles (cuts perpendicular to e2) and full-width long-side profiles."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U)
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}
cols = {'M01':'green','M02':'olive','M03':'darkcyan','M04':'purple','M05':'red','M06':'orange','M07':'magenta','M08':'blue'}
def gold_at(k, axis, val):
    A, B = GL[k]
    if abs(B[axis]-A[axis]) < 1e-6: return None
    t = (val - A[axis]) / (B[axis]-A[axis])
    if t < -0.02 or t > 1.02: return None
    return A + t*(B-A), t
# short sides: cut perpendicular to e2 at stations
st = [-85, -60, -30, 0, 30, 60, 95]
fig, axs = plt.subplots(len(st), 2, figsize=(14, 4.0*len(st)))
for i, s in enumerate(st):
    m = np.abs(P[:, 1]-s) < 0.6
    for j, (side, lo, hi, keys) in enumerate([('e1-min short side (M01/M03)', -156, -118, ['M01','M03']), ('e1-max short side (M02/M04)', 132, 170, ['M02','M04'])]):
        ax = axs[i, j]
        mm = m & (P[:, 0] > lo) & (P[:, 0] < hi)
        ax.scatter(P[mm, 0], P[mm, 2], s=0.6, c='k')
        for k in keys:
            r = gold_at(k, 1, s)
            if r is None: continue
            g, t = r
            ax.plot(g[0], g[2], marker='*', ms=14, color=cols[k], mec='k')
            ax.annotate(f'{k} t={t:.2f}', (g[0], g[2]), xytext=(6, 4), textcoords='offset points', color=cols[k], fontsize=9)
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlim(lo, hi); ax.set_ylim(-26, 26)
        ax.set_title(f'{side}: cut e2={s} (+-0.6)', fontsize=10); ax.set_xlabel('e1'); ax.set_ylabel('e3')
plt.tight_layout(); plt.savefig(OUT + '/a2_short_side_profiles.png', dpi=70); plt.close()
# full-width long-side profiles (whole e2 range) at 3 stations
fig, axs = plt.subplots(3, 1, figsize=(18, 10))
for i, s in enumerate([120, 0, -120]):
    m = np.abs(P[:, 0]-s) < 0.6
    ax = axs[i]
    ax.scatter(P[m, 1], P[m, 2], s=0.5, c='k')
    for k in ['M05','M06','M07','M08']:
        r = gold_at(k, 0, s)
        if r is None: continue
        g, t = r; ax.plot(g[1], g[2], marker='*', ms=12, color=cols[k], mec='k'); ax.annotate(k, (g[1], g[2]), xytext=(5,3), textcoords='offset points', color=cols[k])
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f'full section cut e1={s}', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + '/a2_full_sections_e1.png', dpi=70); plt.close()
fig, axs = plt.subplots(2, 1, figsize=(18, 7))
for i, s in enumerate([60, -40]):
    m = np.abs(P[:, 1]-s) < 0.6
    ax = axs[i]
    ax.scatter(P[m, 0], P[m, 2], s=0.5, c='k')
    for k in ['M01','M02','M03','M04']:
        r = gold_at(k, 1, s)
        if r is None: continue
        g, t = r; ax.plot(g[0], g[2], marker='*', ms=12, color=cols[k], mec='k'); ax.annotate(k, (g[0], g[2]), xytext=(5,3), textcoords='offset points', color=cols[k])
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f'full section cut e2={s}', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + '/a2_full_sections_e2.png', dpi=70); plt.close()
print('ok')

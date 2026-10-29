"""Auditor 2: map points whose normals are not aligned with any frame axis (bend arcs, joggle webs, tabs)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}
cols = {'M01':'green','M02':'olive','M03':'darkcyan','M04':'purple','M05':'red','M06':'orange','M07':'magenta','M08':'blue'}
amax = np.abs(Nn).max(1)
off = amax < 0.93   # more than ~21 deg from any axis
print('off-axis points', off.sum(), 'of', len(P))
fig, axs = plt.subplots(3, 1, figsize=(18, 22))
views = [(0, 1, 'top view e1-e2 (off-axis-normal points)'), (0, 2, 'front view e1-e3'), (1, 2, 'end view e2-e3')]
for ax, (i, j, t) in zip(axs, views):
    ax.scatter(P[::15, i], P[::15, j], s=0.2, c='0.85')
    ax.scatter(P[off, i], P[off, j], s=0.4, c='k')
    for k, (A, B) in GL.items():
        ax.plot([A[i], B[i]], [A[j], B[j]], color=cols[k], lw=1.6)
        ax.annotate(k, ((A[i]+B[i])/2, (A[j]+B[j])/2), color=cols[k], fontsize=10)
    ax.set_aspect('equal'); ax.set_title(t); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + '/a2_offaxis_map.png', dpi=60); plt.close()
# wall face-on views: e2-min wall (points with e2<-90) in e1-e3 and e2-max wall
fig, axs = plt.subplots(4, 1, figsize=(18, 16))
sel = [('e2-min long wall region (e2<-88)', P[:, 1] < -88, 0, 2, ['M05', 'M06']),
       ('e2-max long wall region (e2>95)', P[:, 1] > 95, 0, 2, ['M07', 'M08']),
       ('e1-min short wall region (e1<-138)', P[:, 0] < -138, 1, 2, ['M01', 'M03']),
       ('e1-max short wall region (e1>152)', P[:, 0] > 152, 1, 2, ['M02', 'M04'])]
for ax, (t, m, i, j, keys) in zip(axs, sel):
    ax.scatter(P[m, i][::4], P[m, j][::4], s=0.2, c='0.75')
    mo = m & off
    ax.scatter(P[mo, i], P[mo, j], s=0.5, c='k')
    for k in keys:
        A, B = GL[k]; ax.plot([A[i], B[i]], [A[j], B[j]], color=cols[k], lw=2); ax.annotate(k, (A[i], A[j]), color=cols[k], fontsize=11)
    ax.set_aspect('equal'); ax.set_title(t + ' face-on; black = off-axis normals'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + '/a2_walls_faceon.png', dpi=60); plt.close()
print('ok')

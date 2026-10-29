import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V, F, N = load_mesh()
c, R = pca_frame(V)
P = (V - c) @ R.T
G = load_gold()
GP = {k: ((s - c) @ R.T, (e - c) @ R.T) for k, (s, e) in G.items()}
stations = [-150, -130, -110, -90, -70, -50, -25, 0, 25, 50, 75, 100, 125, 150]
fig, axes = plt.subplots(len(stations), 2, figsize=(16, 2.2 * len(stations)))
for i, x0 in enumerate(stations):
    m = np.abs(P[:, 0] - x0) < 0.6
    Q = P[m]
    for j, (lo, hi) in enumerate([(-120, -60), (60, 120)]):
        ax = axes[i, j]
        mm = (Q[:, 1] > lo) & (Q[:, 1] < hi)
        ax.scatter(Q[mm, 1], Q[mm, 2], s=0.3, c='k')
        for k, (ps, pe) in GP.items():
            d = pe - ps
            t = (x0 - ps[0]) / d[0]
            g = ps + t * d
            if lo < g[1] < hi:
                inside = 0 <= t <= 1
                ax.plot(g[1], g[2], marker='*' if inside else 'x', ms=9, color='g' if inside else 'orange')
                ax.annotate(k, (g[1], g[2]), fontsize=7, color='g')
        ax.set_xlim(lo, hi); ax.set_ylim(-18, 18); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_title(f'PC1={x0}  PC2 in [{lo},{hi}]', fontsize=8)
plt.tight_layout()
plt.savefig(OUT + '/a1_profiles_along_PC1.png', dpi=80)
print('saved')

import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
V, F, N = load_mesh()
c, R = pca_frame(V)
P = (V - c) @ R.T
Np = N @ R.T
Np /= np.linalg.norm(Np, axis=1, keepdims=True)
G = load_gold()
GP = {k: ((s - c) @ R.T, (e - c) @ R.T) for k, (s, e) in G.items()}
# max |component| -> axis-aligned faces; intermediate normals -> bend arcs / cut-edge rounds
amax = np.abs(Np).max(1)
arc = amax < 0.93   # >~21 deg from every axis
print('arc-normal points', arc.sum(), 'of', len(P))
fig, axes = plt.subplots(3, 1, figsize=(18, 22))
ax = axes[0]
sc = ax.scatter(P[::7, 0], P[::7, 1], c=P[::7, 2], s=0.2, cmap='coolwarm', vmin=-15, vmax=15)
plt.colorbar(sc, ax=ax, label='PC3 (mm)')
for k, (ps, pe) in GP.items():
    ax.plot([ps[0], pe[0]], [ps[1], pe[1]], 'g-', lw=1); ax.annotate(k, (pe[0], pe[1]), color='g')
ax.set_aspect('equal'); ax.set_title('top view coloured by PC3 (height); gold in green'); ax.grid(alpha=.3)
ax = axes[1]
ax.scatter(P[arc, 0], P[arc, 1], s=0.3, c=P[arc, 2], cmap='coolwarm', vmin=-15, vmax=15)
for k, (ps, pe) in GP.items():
    ax.plot([ps[0], pe[0]], [ps[1], pe[1]], 'g-', lw=0.8, alpha=.6); ax.annotate(k, (pe[0], pe[1]), color='g')
ax.set_aspect('equal'); ax.set_title('points with non-axis normals (bend arcs + cut-edge rounds), top view'); ax.grid(alpha=.3)
ax = axes[2]
ax.scatter(P[arc, 0], P[arc, 2], s=0.3, c=P[arc, 1], cmap='viridis')
ax.set_aspect('equal'); ax.set_title('non-axis-normal points, front view PC1-PC3 (colour=PC2)'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + '/a1_bendmap_top.png', dpi=70)
np.save('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_frame.npy', np.vstack([c, R]))
print('saved')

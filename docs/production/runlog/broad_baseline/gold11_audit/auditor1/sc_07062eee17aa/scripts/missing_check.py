"""Missing-bend sweep: (1) plate top-skin height residual + slope-change (crease) map, (2) zoomed sections
at the flange notches, the tab ends and the part ends, (3) leg-face inventory (any face not plate/flange/tab)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
V, F, N = load_mesh()
c, R = pca_frame(V)
P = (V - c) @ R.T
Np = N @ R.T; Np /= np.linalg.norm(Np, axis=1, keepdims=True)
# (1) top skin of the plate: normal ~ +PC3 (or -PC3, whichever skin is up), |PC3| < 3
for sgn, name in [(1, 'top'), (-1, 'bottom')]:
    m = (Np[:, 2] * sgn > 0.95) & (np.abs(P[:, 2] + 0.75) < 3)
    Q = P[m]
    A = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
    coef, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
    res = Q[:, 2] - A @ coef
    print(name, 'skin pts', len(Q), 'plane resid p1/p50/p99', np.percentile(res, [1, 50, 99]).round(3))
    if name == 'top':
        Qt, rt = Q, res
# local slope via grid
gx = np.arange(-166, 162, 3.0); gy = np.arange(-114, 115, 3.0)
H = np.full((len(gy), len(gx)), np.nan)
ix = np.clip(((Qt[:, 0] + 166) / 3).astype(int), 0, len(gx) - 1); iy = np.clip(((Qt[:, 1] + 114) / 3).astype(int), 0, len(gy) - 1)
s = np.zeros_like(H); n = np.zeros_like(H)
np.add.at(s, (iy, ix), rt); np.add.at(n, (iy, ix), 1)
H = np.where(n > 5, s / np.maximum(n, 1), np.nan)
gyy, gxx = np.gradient(H, 3.0)
slope = np.degrees(np.arctan(np.hypot(gxx, gyy)))
fig, axes = plt.subplots(2, 1, figsize=(16, 13))
im = axes[0].imshow(H, origin='lower', extent=[gx[0], gx[-1] + 3, gy[0], gy[-1] + 3], cmap='coolwarm', vmin=-1.5, vmax=1.5)
plt.colorbar(im, ax=axes[0], label='top-skin height residual from plane (mm)'); axes[0].set_title('plate top skin: deviation from best-fit plane (3 mm cells)')
im = axes[1].imshow(slope, origin='lower', extent=[gx[0], gx[-1] + 3, gy[0], gy[-1] + 3], cmap='magma', vmin=0, vmax=6)
plt.colorbar(im, ax=axes[1], label='local slope vs plane (deg)'); axes[1].set_title('local slope of top skin (a crease/bend would show as a line of high slope)')
for ax in axes: ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + '/a1_plate_flatness.png', dpi=70)
print('slope >3deg cells', int(np.nansum(slope > 3)), 'of', int(np.isfinite(slope).sum()))
# (2) zoomed sections
def sec(ax, x0, ylo, yhi, zlo=-17, zhi=17, half=0.5, title=''):
    mm = (np.abs(P[:, 0] - x0) < half) & (P[:, 1] > ylo) & (P[:, 1] < yhi)
    ax.scatter(P[mm, 1], P[mm, 2], s=0.5, c='k'); ax.set_xlim(ylo, yhi); ax.set_ylim(zlo, zhi); ax.set_aspect('equal')
    ax.grid(alpha=.3); ax.set_title(title or f'PC1={x0}', fontsize=8)
fig, axes = plt.subplots(4, 6, figsize=(22, 13))
for j, x0 in enumerate([-110, -104, -95, -80, -66, -60]):
    sec(axes[0, j], x0, -118, -96, title=f'L flange notch PC1={x0}')
    sec(axes[1, j], x0, 96, 118, title=f'R flange notch PC1={x0}')
for j, x0 in enumerate([-165, -160, -153, -151, 157, 159]):
    sec(axes[2, j], x0, -118, -90, title=f'L end PC1={x0}')
    sec(axes[3, j], x0, 90, 118, title=f'R end PC1={x0}')
plt.tight_layout(); plt.savefig(OUT + '/a1_notch_and_end_sections.png', dpi=65)
# (3) inventory of face orientations far from plate/flange/tab planes
odd = (np.abs(Np[:, 2]) < 0.9) & (np.abs(Np[:, 1]) < 0.9) & (np.abs(Np[:, 0]) < 0.9)
print('points with normals >25 deg from all axes:', int(odd.sum()))
# side views of a PC1-PC2 top region at each end for small features
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (xlo, xhi) in zip(axes, [(-168, -140), (140, 163)]):
    mm = (P[:, 0] > xlo) & (P[:, 0] < xhi)
    sc = ax.scatter(P[mm, 0], P[mm, 1], c=P[mm, 2], s=0.3, cmap='coolwarm', vmin=-15, vmax=15)
    ax.set_aspect('equal'); ax.set_title(f'top view PC1 [{xlo},{xhi}] coloured by PC3'); ax.grid(alpha=.3)
plt.colorbar(sc, ax=axes[1]); plt.tight_layout(); plt.savefig(OUT + '/a1_part_ends_top.png', dpi=65)
print('ok')

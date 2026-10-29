"""Auditor-1: where are the tilted (non-axis) surfaces? candidates for unlabeled formed features."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import *
xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c); N = nrm @ R.T
ang = np.degrees(np.arccos(np.clip(np.abs(N), 0, 1)))   # angle to each axis
amin = ang.min(1)
# tilted: 8..82 deg from every axis (i.e. not within 8 deg of any axis)
tilt = amin > 8
print("tilted frac", tilt.mean())
# classify tilted by which plane the normal lies in: rotation about x (normal in y-z plane) etc.
nx, ny, nz = np.abs(N.T)
rot_about_x = tilt & (nx < 0.15)   # normal in y-z plane -> fold line along x
rot_about_y = tilt & (ny < 0.15)   # fold line along y
rot_about_z = tilt & (nz < 0.15)
other = tilt & ~(rot_about_x | rot_about_y | rot_about_z)
for nm, m in [("fold-along-x", rot_about_x), ("fold-along-y", rot_about_y), ("fold-along-z", rot_about_z), ("other", other)]:
    print(nm, m.sum())
fig, axs = plt.subplots(2, 2, figsize=(22, 18))
for ax, (nm, m, i, j) in zip(axs.ravel(), [("normals tilted about x (fold lines along x): top view x-y", rot_about_x, 0, 1),
                                             ("normals tilted about y (fold lines along y): top view x-y", rot_about_y, 0, 1),
                                             ("tilted about x: side view y-z", rot_about_x, 1, 2),
                                             ("other tilted: top view x-y", other, 0, 1)]):
    ax.scatter(P[::7, i], P[::7, j], s=0.05, c="#dddddd", rasterized=True)
    ax.scatter(P[m, i], P[m, j], s=0.3, c=amin[m], cmap="viridis", vmin=8, vmax=45, rasterized=True)
    ax.set_title(nm); ax.set_aspect("equal"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUTD + "a1_tilted_normals.png", dpi=60)
np.save(W + "tilt_masks.npy", np.stack([rot_about_x, rot_about_y, rot_about_z, other]))

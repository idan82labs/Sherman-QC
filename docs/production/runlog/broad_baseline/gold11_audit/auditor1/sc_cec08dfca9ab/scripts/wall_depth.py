"""Auditor-1: end walls and base/top plates face-on, coloured by deviation from a fitted plane (shows creases/steps/joggles)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import *
xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c); N = nrm @ R.T
fig, axs = plt.subplots(3, 2, figsize=(24, 22))
def plane_dev(m, axis):
    Q = P[m]
    # fit plane coordinate = a + b*u + c*v over the other two axes
    oth = [i for i in range(3) if i != axis]
    A = np.c_[np.ones(m.sum()), Q[:, oth[0]], Q[:, oth[1]]]
    coef, *_ = np.linalg.lstsq(A, Q[:, axis], rcond=None)
    return Q, Q[:, axis] - A @ coef, oth
cases = [("left end wall outer skin (x<-156.5, |nx|>0.9)", (P[:, 0] < -156.6) & (np.abs(N[:, 0]) > 0.9), 0),
         ("right end wall outer skin (x>153.6, |nx|>0.9)", (P[:, 0] > 153.6) & (np.abs(N[:, 0]) > 0.9), 0),
         ("channel wall outer skin (y>116.5, |ny|>0.9)", (P[:, 1] > 116.3) & (np.abs(N[:, 1]) > 0.9), 1),
         ("base plate bottom skin (z<-10, nz<-0.9 side) ", (P[:, 2] < -9.5) & (np.abs(N[:, 2]) > 0.9) & (P[:, 1] < 112), 2),
         ("top plate (z>24, |nz|>0.9)", (P[:, 2] > 24) & (np.abs(N[:, 2]) > 0.9) & (P[:, 1] > 78), 2)]
for ax, (nm, m, axis) in zip(axs.ravel(), cases):
    Q, dev, oth = plane_dev(m, axis)
    sc = ax.scatter(Q[:, oth[0]], Q[:, oth[1]], s=0.3, c=dev, cmap="coolwarm", vmin=-2.5, vmax=2.5, rasterized=True)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="deviation from plane fit (mm)")
    ax.set_title(nm + f"  axes {['x','y','z'][oth[0]]}-{['x','y','z'][oth[1]]}"); ax.set_aspect("equal"); ax.grid(alpha=.3)
axs.ravel()[-1].axis("off")
fig.tight_layout(); fig.savefig(OUTD + "a1_plate_deviation_maps.png", dpi=55)
print("ok")

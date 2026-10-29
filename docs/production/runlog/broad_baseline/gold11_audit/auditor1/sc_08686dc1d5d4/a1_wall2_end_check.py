"""Check wall-2 bottom edge outside the lip extent (u 70..102) for small tabs; also wall-1 equivalents."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1_common import load_all, OUT

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
n = nrm @ R.T
fig, axs = plt.subplots(2, 5, figsize=(26, 10))
for r, (side, us) in enumerate(((1, (60, 75, 85, 95, 100)), (-1, (-95, -80, 70, 85, 95)))):
    for c, u0 in enumerate(us):
        m = (np.abs(p[:, 0] - u0) < 0.5) & (side * p[:, 1] > 135) & (p[:, 2] < -25)
        ax = axs[r, c]
        ax.scatter(p[m, 1], p[m, 2], s=2, c=np.abs(n[m, 2]), cmap="coolwarm", vmin=0, vmax=1)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title("wall %d bottom, u=%d (colour |n_w|)" % (2 if side > 0 else 1, u0), fontsize=9)
        ax.set_xlabel("v"); ax.set_ylabel("w")
# plan of horizontal-normal points at the wall-2 bottom, u 60..105
fig.tight_layout()
fig.savefig(OUT / "a1_wall_bottom_ends.png", dpi=65)
m = (p[:, 1] > 130) & (p[:, 2] < -36) & (np.abs(n[:, 2]) > 0.9) & (p[:, 0] > 70)
print("wall2 bottom horiz pts u>70:", m.sum())
for lo in range(70, 104, 4):
    mm = m & (p[:, 0] >= lo) & (p[:, 0] < lo + 4)
    if mm.sum():
        print("  u %d..%d n=%d v %.1f..%.1f w %.1f..%.1f" % (lo, lo + 4, mm.sum(), p[mm, 1].min(), p[mm, 1].max(), p[mm, 2].min(), p[mm, 2].max()))

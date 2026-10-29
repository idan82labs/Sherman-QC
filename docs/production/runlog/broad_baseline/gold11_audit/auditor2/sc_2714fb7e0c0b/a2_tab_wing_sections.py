"""Sections perpendicular to the slanted tab-end wing (and to the slanted window edge) to see whether the wing
is attached to the main plate along the slanted edge (a second fold) or only via the vertical crease."""
import sys, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a2_common import load, OUT

q, R, O, raw = load()
fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=90)
for row, (name, sgn, anchors) in enumerate((("B09", -1, [(-57.0, 29.9), (-56.0, 29.35), (-55.0, 28.8), (-54.0, 28.25)]),
                                             ("B10", +1, [(-57.0, 205.9), (-56.0, 206.45), (-55.0, 207.0), (-54.0, 207.55)]))):
    t = np.array([math.cos(math.radians(28.5)), sgn * math.sin(math.radians(28.5))])  # wing direction in u-v
    n = np.array([-t[1], t[0]])
    for ax, (u0, v0) in zip(axes[row], anchors):
        c = np.array([u0, v0])
        rel = q[:, :2] - c
        along = rel @ t
        m = (np.abs(along) < 0.35) & (np.abs(rel @ n) < 10) & (q[:, 2] > -4) & (q[:, 2] < 17)
        ax.scatter((rel @ n)[m], q[m, 2], s=3, c="k", linewidths=0)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-8, 8); ax.set_ylim(-3, 16)
        ax.set_title(f"{name} wing section at (u={u0}, v={v0}); x = across wing (mm), y = w", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "a2_tab_wing_sections.png"); plt.close(fig)
print("ok")

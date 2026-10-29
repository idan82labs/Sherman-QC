"""Full-res zoom slices at the step-wall end corners (CADF_CAD_B5/B7) and at gold B06/B08 corners."""
import json
from pathlib import Path
import numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/triage/sc_2714fb7e0c0b")
SCAN = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply"
pd = np.load(OUT / "points_ds.npz")["p"]
c = pd.mean(0); _, _, R = np.linalg.svd(pd - c, full_matrices=False)
pc = o3d.io.read_point_cloud(SCAN); P = np.asarray(pc.points); q = (P - c) @ R.T
corners = {"CAD_B5 corner (x-74,y-112)": (-74, -112), "CAD_B7 corner (x-74,y+112)": (-74, 112),
           "gold B06 corner (x-122,y-112)": (-122, -112), "gold B08 corner (x-123,y+111)": (-123, 111)}
zs = [-26, -20, -15, 0, 10, 18]
fig, axs = plt.subplots(len(corners), len(zs), figsize=(4 * len(zs), 4 * len(corners)))
for r, (name, (x0, y0)) in enumerate(corners.items()):
    for k, z in enumerate(zs):
        ax = axs[r, k]
        m = (np.abs(q[:, 2] - z) < 0.4) & (np.abs(q[:, 0] - x0) < 15) & (np.abs(q[:, 1] - y0) < 15)
        ax.scatter(q[m, 0], q[m, 1], s=1, c="k"); ax.plot(x0, y0, "r+", ms=10)
        ax.set_xlim(x0 - 15, x0 + 15); ax.set_ylim(y0 - 15, y0 + 15); ax.set_aspect("equal")
        ax.set_title(f"{name}\nPC3={z} n={m.sum()}", fontsize=8); ax.grid(alpha=.3)
fig.savefig(OUT / "corner_zoom_slices.png", dpi=70, bbox_inches="tight")

"""PC1-constant slices through the lower box near y=+-112: is the end wall bent from the lower plate (horizontal axis)?"""
from pathlib import Path
import numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/triage/sc_2714fb7e0c0b")
SCAN = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/38172000_01_V.ply"
pd = np.load(OUT / "points_ds.npz")["p"]; c = pd.mean(0); _, _, R = np.linalg.svd(pd - c, full_matrices=False)
q = (np.asarray(o3d.io.read_point_cloud(SCAN).points) - c) @ R.T
fig, axs = plt.subplots(2, 4, figsize=(24, 10))
for r, (lo, hi) in enumerate([(-125, -95), (95, 125)]):
    for k, x in enumerate([-115, -100, -85, -72]):
        m = (np.abs(q[:, 0] - x) < 0.4) & (q[:, 1] > lo) & (q[:, 1] < hi)
        ax = axs[r, k]; ax.scatter(q[m, 1], q[m, 2], s=1, c="k"); ax.set_aspect("equal")
        ax.set_title(f"PC1={x} slice, PC2 in [{lo},{hi}]  (x: PC2, y: PC3)"); ax.grid(alpha=.3)
        ax.set_ylim(-40, 30)
fig.savefig(OUT / "endwall_pc1_slices.png", dpi=70, bbox_inches="tight")

"""Close-up of the step-wall end corners (CADF_CAD_B5 / CADF_CAD_B7 FP region) for sc_2714fb7e0c0b."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/triage/sc_2714fb7e0c0b")
p = np.load(OUT / "points_ds.npz")["p"]
c = p.mean(0); _, _, R = np.linalg.svd(p - c, full_matrices=False)
q = (p - c) @ R.T
info = json.load(open(OUT / "lines_pc_frame.json"))
for side, ylo, yhi in [("neg", -130, -85), ("pos", 85, 130)]:
    m = (q[:, 0] > -135) & (q[:, 0] < -30) & (q[:, 1] > ylo) & (q[:, 1] < yhi)
    s = q[m]
    fig = plt.figure(figsize=(24, 8))
    for k, (el, az) in enumerate([(20, -60), (20, 30), (60, -100)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.3, c=s[:, 2], cmap="viridis")
        for key, v in info.items():
            a, b = np.array(v["a"]), np.array(v["b"])
            mid = (a + b) / 2
            if not (-140 < mid[0] < -25 and ylo - 30 < mid[1] < yhi + 30) and not key.startswith("gold:B0"):
                continue
            col = {"gold": "green", "best": "red", "ph": "blue"}[key.split(":")[0]]
            # clip to box
            t = np.linspace(0, 1, 200); L = a[None] + t[:, None] * (b - a)[None]
            mm = (L[:, 0] > -135) & (L[:, 0] < -30) & (L[:, 1] > ylo) & (L[:, 1] < yhi)
            if mm.sum() < 2: continue
            ax.plot(L[mm, 0], L[mm, 1], L[mm, 2], color=col, lw=2.5)
            ax.text(*L[mm][len(L[mm]) // 2], key.split(":")[1].replace("CADF_", "").replace("F1C_", "F")[:8], color=col, fontsize=8)
        ax.view_init(el, az); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_box_aspect((105, yhi - ylo, 65))
    fig.suptitle(f"step-wall end corner, PC2 {side}: green gold, red best_f1, blue runtime placeholders")
    fig.savefig(OUT / f"corner_{side}.png", dpi=75, bbox_inches="tight"); plt.close(fig)
# horizontal slices (PC3 levels) of corner neg to see wall/flange connectivity
fig, axs = plt.subplots(1, 5, figsize=(30, 6))
for ax, z in zip(axs, [-28, -20, -5, 10, 20]):
    m = (np.abs(q[:, 2] - z) < 1.0) & (q[:, 0] > -135) & (q[:, 0] < -30) & (q[:, 1] < -85)
    ax.scatter(q[m, 0], q[m, 1], s=2, c="k"); ax.set_aspect("equal"); ax.set_title(f"PC3={z} slice, PC2<-85")
    for key in ["best:CADF_CAD_B5", "gold:B03", "gold:B05", "gold:B01", "gold:B04"]:
        a = np.array(info[key]["a"]); b = np.array(info[key]["b"])
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="red" if key.startswith("best") else "green", lw=1)
    ax.set_xlim(-135, -30); ax.set_ylim(-125, -85); ax.grid(alpha=.3)
fig.savefig(OUT / "corner_neg_zslices.png", dpi=75, bbox_inches="tight"); plt.close(fig)

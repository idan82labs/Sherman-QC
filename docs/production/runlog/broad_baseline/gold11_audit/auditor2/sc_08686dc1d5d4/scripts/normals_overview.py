import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_08686dc1d5d4")
pts = np.load(W / "pts.npy")
pca = np.load(W / "pca.npy"); c, e1, e2, e3 = pca
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=24))
nrm = np.asarray(pc.normals)
np.save(W / "nrm.npy", nrm)
L = (pts - c) @ np.vstack([e1, e2, e3]).T   # local coords
np.save(W / "local.npy", L)
gold = json.load(open("/Users/idantal/ShermanQC/FullProject/output/bend_label_02_10839000/bend_instance_labels.json"))["manual_bends"]
G = {b["bend_id"]: ((np.array(b["line_start"]) - c) @ np.vstack([e1, e2, e3]).T, (np.array(b["line_end"]) - c) @ np.vstack([e1, e2, e3]).T) for b in gold}
json.dump({k: [v[0].tolist(), v[1].tolist()] for k, v in G.items()}, open(W / "gold_local.json", "w"), indent=1)
for k, (a, b) in G.items():
    print(k, "start", a.round(2), "end", b.round(2))
print("local bbox", L.min(0).round(1), L.max(0).round(1))
# |n.e3| classes
ne3 = np.abs(nrm @ e3)
print("frac |n.e3|>0.95 (plate/lip-like):", (ne3 > 0.95).mean().round(3), " <0.2 (walls):", (ne3 < 0.2).mean().round(3))
rng = np.random.default_rng(0)
sel = rng.choice(len(pts), 400000, replace=False)
fig, axs = plt.subplots(1, 2, figsize=(22, 10))
ax = axs[0]
h = L[sel, 2]
sc = ax.scatter(L[sel, 0], L[sel, 1], c=np.clip(h, -40, 8), s=0.2, cmap="turbo")
plt.colorbar(sc, ax=ax, label="height along plate normal e3 (mm)")
for k, (a, b) in G.items():
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="k", lw=1.2)
    m = (a + b) / 2; ax.text(m[0], m[1], k, fontsize=11, color="k", bbox=dict(fc="w", alpha=0.7))
ax.set_aspect("equal"); ax.set_title("top view (e1 along M01/M02, e2 along M03/M04), colour = height"); ax.grid(alpha=0.3)
ax = axs[1]
near = np.abs(L[sel, 2]) < 6
s2 = sel[near]
sc = ax.scatter(L[s2, 0], L[s2, 1], c=L[s2, 2], s=0.2, cmap="coolwarm", vmin=np.percentile(L[s2, 2], 2), vmax=np.percentile(L[s2, 2], 98))
plt.colorbar(sc, ax=ax, label="height (plate band only, |h|<6)")
for k, (a, b) in G.items():
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="k", lw=1.0)
ax.set_aspect("equal"); ax.set_title("plate band height map (reveals joggles/steps/emboss)"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "a2_top_heightmap.png", dpi=80)
# wide sections across e2 (perp to M01/M02) at several e1 positions, and across e1 at several e2
fig, axs = plt.subplots(2, 4, figsize=(24, 10))
for j, x0 in enumerate([-150, -60, 30, 120]):
    ax = axs[0, j]; m = np.abs(L[:, 0] - x0) < 0.6
    ax.scatter(L[m, 1], L[m, 2], s=0.3, c="k"); ax.set_title(f"section e1={x0} (view e2-e3)")
    for k, (a, b) in G.items():
        d = b - a
        if abs(d[0]) > 1e-6:
            t = (x0 - a[0]) / d[0]
            if 0 <= t <= 1:
                p = a + t * d; ax.plot(p[1], p[2], "g*", ms=12); ax.text(p[1], p[2] + 3, k, color="g")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
for j, y0 in enumerate([-120, -40, 40, 120]):
    ax = axs[1, j]; m = np.abs(L[:, 1] - y0) < 0.6
    ax.scatter(L[m, 0], L[m, 2], s=0.3, c="k"); ax.set_title(f"section e2={y0} (view e1-e3)")
    for k, (a, b) in G.items():
        d = b - a
        if abs(d[1]) > 1e-6:
            t = (y0 - a[1]) / d[1]
            if 0 <= t <= 1:
                p = a + t * d; ax.plot(p[0], p[2], "g*", ms=12); ax.text(p[0], p[2] + 3, k, color="g")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "a2_wide_sections.png", dpi=80)
print("saved")

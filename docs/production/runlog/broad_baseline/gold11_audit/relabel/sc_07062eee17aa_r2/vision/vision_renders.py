"""Gold11 relabel r2 sc_07062eee17aa: independent vision renders (8 global views, per-bend
close-ups and perpendicular sections at 25/50/75 %). Read-only; run under the geometry lock."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from research.bench_labeling.scanio import load_scan
HERE = Path(__file__).resolve().parent
LJ = json.loads((HERE.parent / "lines.json").read_text())
scan = load_scan(LJ["scan_path"]); P = scan.points
ds = np.load(Path(LJ["prep_dir"]) / "points_ds.npz")["points"]
bends = [b for b in LJ["bends"] if b["status"] == "ok"]
COL = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#e377c2"]
c = ds.mean(0); U, S, Vt = np.linalg.svd(ds - c, full_matrices=False); E1, E2, E3 = Vt
def view(ax, elev, azim, title):
    a, e = np.radians(azim), np.radians(elev)
    fwd = np.cos(e) * (np.cos(a) * E1 + np.sin(a) * E2) + np.sin(e) * E3
    right = np.cross(fwd, E3); right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else E1
    up = np.cross(right, fwd)
    q = ds - c; x, y, z = q @ right, q @ up, q @ fwd
    o = np.argsort(z)
    ax.scatter(x[o], y[o], s=0.15, c=z[o], cmap="Greys", alpha=0.6, linewidths=0)
    for k, b in enumerate(bends):
        s, en = np.array(b["label_line"]["start"]) - c, np.array(b["label_line"]["end"]) - c
        ax.plot([s @ right, en @ right], [s @ up, en @ up], color=COL[k], lw=1.6)
        m = (s + en) / 2; ax.text(m @ right, m @ up, b["bend_id"], color=COL[k], fontsize=8, weight="bold")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=8); ax.axis("off")
views = [(30, 30, "iso NE"), (30, 120, "iso NW"), (30, 210, "iso SW"), (30, 300, "iso SE"),
         (89, 0, "top"), (-89, 0, "bottom"), (0, 0, "front (along PC1)"), (0, 90, "side (along PC2)"),
         (-30, 45, "iso under NE"), (-30, 225, "iso under SW")]
fig, axs = plt.subplots(2, 5, figsize=(30, 12))
for ax, (e, a, t) in zip(axs.ravel(), views): view(ax, e, a, t)
fig.tight_layout(); fig.savefig(HERE / "global_views.png", dpi=110); plt.close(fig)
# per-bend sheets
for k, b in enumerate(bends):
    s0, e0 = np.array(b["label_line"]["start"]), np.array(b["label_line"]["end"])
    sh0 = np.array(b["sharp"]["start"]); d = np.array(b["direction"]); bis = np.array(b["bisector"])
    L = np.linalg.norm(e0 - s0); lat = np.cross(d, bis)
    fig, axs = plt.subplots(1, 4, figsize=(28, 7))
    # close-up face-on: points within 25 mm of line, projected on (d, bis+lat plane viewed from outside)
    rel = P - s0; t = rel @ d; w = rel - np.outer(t, d)
    near = (np.linalg.norm(w, axis=1) < 25) & (t > -15) & (t < L + 15)
    vdir = -bis  # look from outside onto the apex
    v2 = lat
    ax = axs[0]; q = P[near] - s0
    dep = q @ vdir; o = np.argsort(-dep)
    ax.scatter((q @ d)[o], (q @ v2)[o], s=0.3, c=dep[o], cmap="viridis", linewidths=0)
    ax.plot([0, L], [0, 0], color=COL[k], lw=2); ax.set_aspect("equal")
    ax.set_title(f"{b['bend_id']} close-up from outside (colour=depth), line length {L:.1f} mm", fontsize=9)
    for j, f in enumerate((0.25, 0.5, 0.75)):
        tt = f * L; slab = near & (np.abs(t - tt) <= 0.5)
        q = w[slab]; xb, xl = q @ bis, q @ lat
        ax = axs[j + 1]
        ax.scatter(xl, xb, s=2, c="0.4")
        shp = (sh0 + tt * d) - (s0 + tt * d)
        ax.plot([shp @ lat], [shp @ bis], "kx", ms=12, mew=2)
        ax.plot([0], [0], "o", color=COL[k], ms=9, mec="k")
        # nearest scanned point distance from the label point
        dd = np.sqrt(xb ** 2 + xl ** 2); dmin = dd.min() if len(dd) else np.nan
        # nearest surface along the bisector ray (|lat|<=0.35)
        ray = np.abs(xl) <= 0.35
        hit = xb[ray].min() if ray.any() else np.nan
        ax.set_xlim(-6, 6); ax.set_ylim(-4, 8); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("lateral (mm)"); ax.set_ylabel("along bisector, + = into the corner (mm)")
        ax.set_title(f"{b['bend_id']} section {int(f*100)}% (t={tt:.1f}); x = sharp, dot = label\n"
                     f"nearest scan pt {dmin:.2f} mm; surface along bisector at {hit:+.2f} mm", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / f"bend_{b['bend_id']}.png", dpi=90); plt.close(fig)
print("ok")

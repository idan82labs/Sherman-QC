"""plot_cad.py COMPILED_DIR OUT.png: 10839000 final stage coloured by its deviation from its own mirror image under E1's
best MIRROR candidate (equivalently the best proper rotation of the mirrored part), three orthographic views."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cdir, outp = sys.argv[1], sys.argv[2]
d = json.load(open(f"{cdir}/compiled_part.json")); st = d["stages"][-1]
P = np.load(f"{cdir}/stages/{st['stage_model_id']}.npz")["points"].astype(float)
cand = min([c for c in st["symmetry"]["candidates"] if c["kind"] == "MIRROR"], key=lambda c: c["rms_mm"])
M = np.asarray(cand["transform"]); Q = P @ M[:3, :3].T + M[:3, 3]
d1, _ = cKDTree(P).query(Q); d2, _ = cKDTree(Q).query(P); dd = np.maximum(d1, d2)
stats = dict(rms=float(np.sqrt((d2 ** 2).mean())), p99=float(np.percentile(dd, 99)), mx=float(dd.max()),
             f1=float((dd > 1).mean()))
print(json.dumps(stats))
fig, axs = plt.subplots(1, 3, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.6, 1, 1]})
o = np.argsort(dd)
lo, hi = P[:, 1] < P[:, 1].min() + 3, P[:, 1] > P[:, 1].max() - 3
views = [(np.ones(len(P), bool), 1, 2, "whole part, y-z"), (lo, 0, 2, f"side wall y = {P[lo, 1].mean():.0f} (x-z)"),
         (hi, 0, 2, f"side wall y = {P[hi, 1].mean():.0f} (x-z)")]
for ax, (m, i, j, name) in zip(axs, views):
    k = o[m[o]]
    sc = ax.scatter(P[k, i], P[k, j], c=np.clip(dd[k], 0, 3), s=0.4, cmap="viridis", vmin=0, vmax=3, rasterized=True)
    ax.set_aspect("equal"); ax.set_title(name); ax.set_xlabel("xyz"[i] + " (mm)"); ax.set_ylabel("xyz"[j] + " (mm)")
cb = fig.colorbar(sc, ax=axs, shrink=0.8); cb.set_label("distance to the mirrored CAD (mm, clipped at 3)")
fig.suptitle(f"10839000 (A1 compile DEV-EA3E4820DF), final stage {st['stage_model_id']}: CAD vs its mirror image under E1's best MIRROR "
             f"candidate\nRMS {cand['rms_mm']:.3f} mm (E1 certifies at 0.300), P99 {stats['p99']:.2f} mm, max {stats['mx']:.1f} mm, "
             f"{100*stats['f1']:.2f} % of points > 1 mm: one ~15 x 28 mm cutout plus 8 hole edges ~2 mm off", fontsize=11)
fig.savefig(outp, dpi=110, bbox_inches="tight")

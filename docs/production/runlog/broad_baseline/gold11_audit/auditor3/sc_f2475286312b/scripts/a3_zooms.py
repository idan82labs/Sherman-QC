"""Zoomed perpendicular mesh sections around every fold region, with gold crossing points."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3common import load_mesh, load_gold, slice_mesh, OUT

V, N, F = load_mesh()
G = load_gold()
fr = json.loads((OUT / "frame.json").read_text())
o, a, b, n = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([a, b, n])
Gf = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in G.items()}
COL = {"M01": "#1a9c3a", "M02": "#1a9c3a", "M03": "#d62728", "M04": "#ff7f0e", "M05": "#d62728",
       "M06": "#ff7f0e", "M07": "#1f77b4", "M08": "#9467bd"}
AX = {"a": 0, "b": 1, "h": 2}
VEC = {"a": a, "b": b, "h": n}


def panel(ax, cut_axis, cut_val, xaxis, yaxis, xlim, ylim, title):
    segs = slice_mesh(V, F, o + cut_val * VEC[cut_axis], VEC[cut_axis])
    if len(segs):
        P = ((segs.reshape(-1, 3) - o) @ R.T).reshape(-1, 2, 3)
        for s in P:
            ax.plot(s[:, AX[xaxis]], s[:, AX[yaxis]], color="k", lw=0.7)
    ci = AX[cut_axis]
    for k, (s, e) in Gf.items():
        if abs(e[ci] - s[ci]) < 1e-6:
            continue
        t = (cut_val - s[ci]) / (e[ci] - s[ci])
        if -0.02 <= t <= 1.02:
            p = s + t * (e - s)
            ax.plot(p[AX[xaxis]], p[AX[yaxis]], "*", color=COL[k], ms=11, mec="k", mew=0.5)
            ax.annotate(k, (p[AX[xaxis]], p[AX[yaxis]]), fontsize=8, color=COL[k], xytext=(4, 4),
                        textcoords="offset points")
            ax.add_patch(plt.Circle((p[AX[xaxis]], p[AX[yaxis]]), 4.0, fill=False, ls=":", color=COL[k], lw=0.8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xaxis, fontsize=7)
    ax.set_ylabel(yaxis, fontsize=7)
    ax.tick_params(labelsize=6)


jobs = {
    "lip1_M01_M03_M04": dict(cut="a", vals=[-145, -100, -50, 0, 50, 68, 80, 110], x="b", y="h", xl=(70, 125), yl=(-20, 17)),
    "lip2_M02_M05_M06": dict(cut="a", vals=[-145, -100, -50, 0, 50, 68, 80, 110], x="b", y="h", xl=(-125, -65), yl=(-20, 17)),
    "wall_M07_M08": dict(cut="b", vals=[-108, -100, -50, 0, 50, 100, 108, 112], x="a", y="h", xl=(95, 130), yl=(-68, 8)),
    "shortlip_a-140": dict(cut="b", vals=[-105, -100, -50, 0, 50, 100, 104, 108], x="a", y="h", xl=(-152, -125), yl=(-6, 16)),
}
for name, j in jobs.items():
    vals = j["vals"]
    nc = 4
    nr = int(np.ceil(len(vals) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 5.0 * nr * (j["yl"][1] - j["yl"][0]) / (j["xl"][1] - j["xl"][0]) * 0.9 + 1.2), dpi=95)
    for ax, v in zip(axes.ravel(), vals):
        panel(ax, j["cut"], v, j["x"], j["y"], j["xl"], j["yl"], f"{name}: {j['cut']} = {v}")
    for ax in axes.ravel()[len(vals):]:
        ax.axis("off")
    fig.suptitle(f"sc_f2475286312b mesh sections ({name}); star = gold crossing, dotted = 4 mm", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"a3_zoom_{name}.png")
    plt.close(fig)
print("ok")

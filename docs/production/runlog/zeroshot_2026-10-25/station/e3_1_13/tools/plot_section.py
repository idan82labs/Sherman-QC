"""B5 section before (e3-1.12.0) / after (e3-1.13.0-rc) on the 23553000 B full scan, 95% subsample seed 9501. usage: plot_section.py OUT.png"""
import json, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3c/probe"
L = pickle.load(open(f"{P}/live_s9501/probe.pkl", "rb"))["bends"]
C = pickle.load(open(f"{P}/cand_s9501/probe.pkl", "rb"))["bends"]
N = pickle.load(open(f"{P}/live_nat/probe.pkl", "rb"))["bends"]
T = lambda d: np.array(json.load(open(f"{P}/{d}/engine/result.json"))["inspection_validity"]["registration"]["raw_to_frame"])
M = T("live_s9501") @ np.linalg.inv(T("live_nat")); Rm, tv = M[:3, :3], M[:3, 3]
def mapped(pl):
    n = Rm @ np.array(pl[0]); return n, float(pl[1] + n @ tv)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=110)
for row, (lab, side, sname) in enumerate((("B5", 0, "B5, short panel as parent"), ("B6", 1, "B6, short panel as child"))):
    rl, rc = L[lab]["reseat"][side], C[lab]["reseat"][side]
    pts = rl["pts"].astype(float); n0, d0 = np.array(rl["plane_in"][0]), rl["plane_in"][1]
    t = rl["t"]
    key = "plane_parent" if side == 0 else "plane_child"
    nl, dl = np.array(L[lab][key][0]), L[lab][key][1]
    nc, dc = np.array(C[lab][key][0]), C[lab][key][1]
    nn, dn = mapped(N[lab][key])
    h = pts @ n0 - d0
    m = (h > -2.6) & (h < 1.6)
    # in-plane coordinate: along the bend line direction-free axis (largest spread of the strip in the plane)
    q = pts[m] - pts[m].mean(0); q = q - np.outer(q @ n0, n0)
    u, s_, vt = np.linalg.svd(q[::max(1, len(q) // 4000)], full_matrices=False); w = vt[1]
    x = (pts[m] - pts[m].mean(0)) @ w
    ax = axes[row, 0]
    ax.scatter(x, h[m], s=1, c="#9a9a9a", lw=0, rasterized=True, label="scan points in the station strip")
    xs = np.linspace(x.min(), x.max(), 50); c0 = pts[m].mean(0)
    for (nv, dv), col, ls, nm in (((nl, dl), "#c0392b", "-", "e3-1.12.0 OML plane (one-skin region)"),
                                  ((nc, dc), "#1f5fa8", "-", "e3-1.13.0-rc OML plane (re-seated +-t)"),
                                  ((nn, dn), "#2e8b57", ":", "native scan, e3-1.12.0 (two-skin region)")):
        p = c0[None, :] + np.outer(xs, w)
        ax.plot(xs, (p @ nv - dv) * 0 + ((c0 + np.outer(xs, w)) @ n0 - d0) - ((c0 + np.outer(xs, w)) @ nv - dv), ls, c=col, lw=1.8, label=nm)
    ax.set_xlabel("across the strip (mm)"); ax.set_ylabel("height over the e3-1.12.0 region plane (mm)")
    ax.set_title(f"{sname}: strip section (seed 9501, t = {t:.2f} mm)", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False); ax.set_ylim(-2.6, 1.6)
    ax = axes[row, 1]
    bins = np.arange(-2.6, 1.6, 0.05)
    ax.hist(pts @ nl - dl, bins=bins, color="#c0392b", alpha=.55, label="height over e3-1.12.0 OML plane")
    ax.hist(pts @ nc - dc, bins=bins, color="#1f5fa8", alpha=.55, label="height over e3-1.13.0-rc OML plane")
    ax.axvline(0, c="k", lw=.8); ax.axvline(-t, c="k", lw=.8, ls="--")
    ax.text(0.02, ax.get_ylim()[1] * .92, " OML (0)", fontsize=8); ax.text(-t + 0.02, ax.get_ylim()[1] * .92, " inner (-t)", fontsize=8)
    ax.set_xlabel("height along the plane normal (mm, outward +)"); ax.set_ylabel("strip points / 0.05 mm")
    ax.set_title("material outside the OML plane = a skin taken for the other", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.85))
fig.suptitle("23553000 B full scan, 95% subsample s9501: the short panel's region holds only its inner skin.\n"
             "e3-1.12.0 takes that skin for the OML (one thickness low); e3-1.13.0-rc re-seats it on the outer skin seen in the strip.",
             fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(sys.argv[1]); print("ok")

"""Close-up cross-sections perpendicular to each gold line (part frame), gold point vs bend geometry."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1_common import load_all, gold_lines, OUT

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
n = nrm @ R.T
g = gold_lines()
gp = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in g.items()}
W = json.loads((OUT / "a1_wall_folds.json").read_text())


def gold_at(key, coord, val):
    s, e = gp[key]
    t = (val - s[coord]) / (e[coord] - s[coord])
    return s + t * (e - s), t


def wall_panel(ax, key, side, u0, half=0.5, win=9.0):
    m = (np.abs(p[:, 0] - u0) < half) & (side * p[:, 1] > 120)
    P = p[m]
    q, t = gold_at(key, 0, u0)
    row = min(W[key], key=lambda r: abs(r["u"] - u0))
    cx, cy = (row["apex"] if (row.get("status") == "fold" and row.get("apex") and abs(row["u"] - u0) < 2.1) else (q[1], q[2]))
    ax.scatter(P[:, 1], P[:, 2], s=1.5, c="0.25")
    ax.plot(q[1], q[2], marker="*", ms=16, color="#1a9c3a", mec="k", zorder=5)
    circ = plt.Circle((q[1], q[2]), 4.0, fill=False, ls=":", color="#1a9c3a"); ax.add_patch(circ)
    title = "%s  u=%.0f  (gold t=%.2f%s)" % (key, u0, t, "" if 0 <= t <= 1 else " OUTSIDE gold span")
    if row.get("status") == "fold" and abs(row["u"] - u0) < 2.1:
        a = row["apex"]; ax.plot(a[0], a[1], "D", ms=8, color="c", mec="k", zorder=6)
        xo = row["Xo"]; ax.plot(xo[0], xo[1], "x", ms=10, mew=2, color="#1f77b4")
        if row.get("Xi"):
            ax.plot(row["Xi"][0], row["Xi"][1], "x", ms=10, mew=2, color="#9467bd")
        title += "\nfold %.1f deg; gold-apex %.2f mm (along-in %.2f, across %.2f); gold-scan %.2f" % (
            row["bend_deg"], row.get("gold_apex_dist", float("nan")), row.get("gold_minus_apex_along_bis_in", float("nan")),
            row.get("gold_minus_apex_across", float("nan")), row.get("gold_nearest_scan_2d", float("nan")))
    else:
        d = np.min(np.linalg.norm(P[:, 1:3] - q[1:3], axis=1)) if len(P) else float("nan")
        title += "\nNO FOLD at this u (wall absent): gold-scan %.2f mm" % d
    ax.set_xlim(cx - win, cx + win); ax.set_ylim(cy - win, cy + win)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(title, fontsize=8)
    ax.set_xlabel("v (mm)"); ax.set_ylabel("w (mm)")


fig, axs = plt.subplots(2, 6, figsize=(30, 11))
for r, (key, side) in enumerate((("M01", -1), ("M02", 1))):
    for c, u0 in enumerate((92, 26, -41, -96, -110, -150)):
        wall_panel(axs[r, c], key, side, u0)
fig.suptitle("sc_08686dc1d5d4 auditor1: sections perpendicular to the wall folds (plane u = const). star = gold, dotted = 4 mm, "
             "cyan diamond = measured outer apex, blue x = outer virtual sharp, purple x = inner virtual sharp. "
             "Wall-1 fold spans u -102.5..97.5, wall-2 fold u -97.5..102.5; gold M01 spans u -175.7..93.7, M02 u -172.6..90.7", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "a1_xsec_M01_M02.png", dpi=72)
plt.close(fig)


def end_panel(ax, key, v0, ulo, uhi, half=0.5, wlo=-6, whi=4):
    m = (np.abs(p[:, 1] - v0) < half) & (p[:, 0] > ulo) & (p[:, 0] < uhi) & (p[:, 2] > -8)
    P = p[m]
    q, t = gold_at(key, 1, v0)
    ax.scatter(P[:, 0], P[:, 2], s=1.5, c="0.25")
    ax.plot(q[0], q[2], marker="*", ms=16, color="#1a9c3a", mec="k", zorder=5)
    ax.add_patch(plt.Circle((q[0], q[2]), 4.0, fill=False, ls=":", color="#1a9c3a"))
    d = np.min(np.linalg.norm(P[:, [0, 2]] - q[[0, 2]], axis=1)) if len(P) else float("nan")
    ax.set_title("%s  v=%.0f  gold (u,w)=(%.1f,%.2f) t=%.2f  gold-scan %.2f mm" % (key, v0, q[0], q[2], t, d), fontsize=8)
    ax.set_xlim(ulo, uhi); ax.set_ylim(wlo, whi); ax.grid(alpha=0.3)
    ax.set_xlabel("u (mm)"); ax.set_ylabel("w (mm)")


fig, axs = plt.subplots(4, 3, figsize=(30, 16))
for c, v0 in enumerate((-100, 0, 100)):
    end_panel(axs[0, c], "M04", v0, -182, -150)                       # M04 close: strip + left joggle
    end_panel(axs[1, c], "M03", v0, 80, 180, wlo=-10, whi=8)          # M03 wide: flat plate to right joggle
    end_panel(axs[2, c], "M03", v0, 80, 105)                          # M03 close
    # right joggle close-up for reference
    m = (np.abs(p[:, 1] - v0) < 0.5) & (p[:, 0] > 150) & (p[:, 0] < 182) & (p[:, 2] > -8)
    axs[3, c].scatter(p[m, 0], p[m, 2], s=1.5, c="0.25"); axs[3, c].set_xlim(150, 182); axs[3, c].set_ylim(-6, 4)
    axs[3, c].grid(alpha=0.3); axs[3, c].set_title("right joggle (no gold line) v=%.0f" % v0, fontsize=8)
    axs[3, c].set_xlabel("u (mm)"); axs[3, c].set_ylabel("w (mm)")
for ax in axs[0]:
    ax.set_aspect("equal")
for ax in axs[2]:
    ax.set_aspect("equal")
for ax in axs[3]:
    ax.set_aspect("equal")
fig.suptitle("sc_08686dc1d5d4 auditor1: sections perpendicular to M04 (row 1, left plate end with joggle), M03 (rows 2-3; wide and close), "
             "and the right joggle (row 4). star = gold, dotted = 4 mm. Note different axis scales in row 2.", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "a1_xsec_M03_M04_joggles.png", dpi=72)
print("ok")

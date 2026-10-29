"""Summary overlay: gold lines coloured by auditor-3 verdict, fitted formed-fold apex lines, ear axes."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from a3x_common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = load_points()
F = json.load(open(OUT / "a3x_frame.json"))
O, X, Y, Z = (np.array(F[k]) for k in ("O", "X", "Y", "Z"))
L = np.c_[(p - O) @ X, (p - O) @ Y, (p - O) @ Z]
A = json.load(open(OUT / "a3x_apexlines.json"))
gold = load_gold()
GL = {bid: (np.array([(a - O) @ X, (a - O) @ Y, (a - O) @ Z]), np.array([(b - O) @ X, (b - O) @ Y, (b - O) @ Z])) for bid, a, b in gold}
V = {"M01": ("REAL_ERROR: 295 mm line, fold only X -2..76.5 (27% of line on fold); locus 1.0 mm", "#ff7f0e"),
     "M02": ("OK: 0.46 mm from outer apex, 96% of fold", "#1a9c3a"),
     "M03": ("NOT_A_BEND: free edge of +Y flange (14.4 mm from M02 apex)", "#d62728"),
     "M04": ("REAL_ERROR: 300 mm line, fold only X -2..76.5 (26% on fold); locus 0.8 mm", "#ff7f0e"),
     "M05": ("OK: 1.29 mm from outer apex, 95% of fold", "#1a9c3a"),
     "M06": ("NOT_A_BEND: free edge of -Y flange (12.9 mm from M05 apex)", "#d62728")}
rng = np.random.default_rng(3)
S = L[rng.choice(len(L), 220000, replace=False)]
fig = plt.figure(figsize=(18, 16))
ax = fig.add_axes([0.04, 0.36, 0.92, 0.6])
ax.scatter(S[:, 0], S[:, 1], c=np.clip(S[:, 2], -16, 16), s=0.25, cmap="coolwarm", vmin=-16, vmax=16, alpha=0.6)
for fold, f in A["folds"].items():
    if "apex_line_part" in f:
        a, b = np.array(f["apex_line_part"])
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="tab:blue", lw=2.5, alpha=0.9)
    else:
        k = f["axis_line_part_mid_surface"][0]
        ax.plot(k[0], k[1], "P", color="magenta", ms=16, mec="k")
        ax.text(k[0] - 30, k[1] + (5 if k[1] > 0 else -9), f"MISSING {fold}: ~30 deg ear, vertical axis, Z 2..14.5", color="magenta", fontsize=10, weight="bold")
for bid, (a, b) in GL.items():
    txt, col = V[bid]
    off = 0 if bid in ("M01", "M04") else 0
    ax.plot([a[0], b[0]], [a[1] + off, b[1] + off], "-", color=col, lw=1.6)
    ax.text(a[0] - 2, a[1] + (2.5 if bid in ("M01", "M02", "M04", "M05") else -5.5) * (1 if a[1] > 0 else -1), f"{bid} {txt}", color=col, fontsize=9, weight="bold")
ax.set_aspect("equal"); ax.grid(alpha=.25)
ax.set_xlim(-170, 170); ax.set_ylim(-125, 122)
ax.set_title("sc_07062eee17aa (38172000 SCAN_1) auditor 3: gold lines coloured by verdict; blue dashed = fitted outer-apex line of each formed 90 deg fold;\nmagenta = formed 30 deg ears with no gold line. Colour = height off web (red up / blue down). Part frame X along bends.", fontsize=11)
# end elevation with gold points
ax2 = fig.add_axes([0.04, 0.04, 0.92, 0.28])
m = (L[:, 0] > 20) & (L[:, 0] < 22)
T = L[m]
ax2.scatter(T[:, 1], T[:, 2], s=1.5, c="0.3")
for bid, (a, b) in GL.items():
    t = (21 - a[0]) / (b[0] - a[0]); g = a + t * (b - a)
    ax2.plot(g[1], g[2], "*", ms=15, mfc=V[bid][1], mec="k")
    ax2.text(g[1] + 1.2, g[2] - 0.5, bid, color=V[bid][1], fontsize=11, weight="bold")
ax2.set_aspect("equal"); ax2.grid(alpha=.3)
ax2.set_title("perpendicular section at X = 21 mm (slab +-1 mm): web, side flanges down, tabs up; stars = gold points", fontsize=11)
fig.savefig(OUT / "a3x_summary_verdicts.png", dpi=70)
print("ok")
